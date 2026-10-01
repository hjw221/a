"""
train_final.py — 周期性重训脚本 (替代原 retrain_weekly.py)
用指定变体 (默认 v3bal: v3特征34个 + 平衡几何 3.0/1.143xATR/6h, 胜率下限标定) 在最近36个月上重训。
流程与走查验证完全同口径:
  训练窗 = 最近36个月, 右端purge(标签前瞻+embargo, 按变体horizon)
  内部80/20切分(带purge): 前80%拟合+早停, 后20%阈值校准 (变体阈值目标: v3族=盈亏比优先, bal族=胜率下限)
  输出: 模型文件 + trading_config.json (阈值/特征/障碍参数) + feature_importance_v3.json
用法:  python train_final.py [--variant v3bal|v3aggr|v3|legacy|...] [--data_path CSV] [--out_dir production_models]
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CFG
from data import load_data
from features import build_features_legacy, build_features_v2
from features_v3 import build_features_v3
from labeling import make_labels
from walkforward import time_decay_weights
from models import fit_lgb, fit_xgb, predict_ens
from backtest import calibrate_threshold
from run_all import VARIANTS, variant_cfg
from variants_ext import ensemble_weights, predict_weighted

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results")


def build_features_for(fspec, m5, atr_window):
    """按变体特征规格构建, 返回 (F, feats, atr)。"""
    if fspec == "v3":
        return build_features_v3(m5, atr_window)
    if fspec == "v2":
        return build_features_v2(m5, atr_window)
    F, feats = build_features_legacy(m5)
    F = F.astype(np.float32)
    _, _, atr = build_features_v2(m5, atr_window)
    return F, feats, atr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v3bal", choices=sorted(VARIANTS.keys()))
    ap.add_argument("--data_path", default=CFG["data_path"])
    ap.add_argument("--out_dir", default=os.path.join(BASE, "production_models"))
    ap.add_argument("--months", type=int, default=CFG["max_train_months"])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    spec = VARIANTS[args.variant]
    fspec, geo, param_src, thr_obj = spec[0], spec[1], spec[2], spec[3]
    extra = spec[4] if len(spec) > 4 else {}
    use_ens = extra.get("mode") == "ens"
    cfg = variant_cfg(args.variant)
    cfg["data_path"] = args.data_path
    print(f"[final] 变体={args.variant} (特征{fspec}, 几何{cfg['tp_atr_mult']}/{cfg['sl_atr_mult']}xATR/"
          f"{cfg['horizon_m1']}M1, 阈值目标={thr_obj}, 集成={'lgb/xgb×' + str(len(extra.get('seeds', []))) + '种子' if use_ens else '单种子'})")

    print("[final] 加载数据...")
    m5, m1_pack, spread_cost, monthly = load_data(cfg)

    # 最近N个月训练窗 (右端purge, 按变体horizon)
    months = m5.index.to_period("M")
    uniq = list(months.unique())
    train_months = uniq[-args.months:]
    mask = np.isin(months, train_months)
    purge = int(np.ceil(cfg["horizon_m1"] / 5)) + cfg["purge_extra_m5"]
    rows_all = np.where(mask)[0]
    train_rows = rows_all[: len(rows_all) - purge]
    print(f"[final] 训练窗: {train_months[0]}~{train_months[-1]} ({len(train_months)}月, "
          f"{len(train_rows):,}行, 右端purge {purge}根M5)")

    F, feats, atr = build_features_for(fspec, m5, cfg["atr_window_m5"])
    lab, valid = make_labels(m5, m1_pack, spread_cost, atr, cfg)

    # 内部80/20 (带purge gap) — 与走查验证同口径
    n_val = int(len(train_rows) * cfg["inner_val_frac"])
    val_rows = train_rows[-n_val:]
    fit_rows = train_rows[: -n_val - purge]

    tuned = json.load(open(os.path.join(RES, "tuned_params.json")))[param_src]
    w = time_decay_weights(m5.index, fit_rows, cfg["sample_halflife_days"])

    Xf = F.iloc[fit_rows].to_numpy(np.float32)
    Xv = F.iloc[val_rows].to_numpy(np.float32)
    valid_val = valid[val_rows]
    models, rounds_info, importance = {}, {}, {}
    members_cfg = {"long": [], "short": []}
    for direction in ["long", "short"]:
        y = (lab["out_long" if direction == "long" else "out_short"].to_numpy() == 1).astype(np.int8)
        yv = y[val_rows]
        if use_ens:
            # ---- ens: lgb/xgb × 多种子, val段AUC加权 (与走查验证同一套variants_ext逻辑) ----
            import xgboost as xgb
            from sklearn.metrics import roc_auc_score
            seeds = extra.get("seeds", [42, 1337, 2024])
            drop_below = float(extra.get("drop_below", 0.51))
            members = []
            for seed in seeds:
                for kind in ["lgb", "xgb"]:
                    params = dict(tuned[direction][kind])
                    if kind == "lgb":
                        params.update(seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
                        m, r = fit_lgb(Xf, y[fit_rows], w, params, Xv, yv, np.ones(n_val),
                                       es_rounds=cfg["early_stopping_rounds"], max_rounds=cfg["max_rounds"])
                        pv = m.predict(Xv, num_iteration=m.best_iteration if m.best_iteration else None)
                    else:
                        params.update(seed=seed)
                        m, r = fit_xgb(Xf, y[fit_rows], w, params, Xv, yv, np.ones(n_val),
                                       es_rounds=cfg["early_stopping_rounds"], max_rounds=cfg["max_rounds"])
                        pv = m.predict(xgb.DMatrix(Xv))
                    ok = valid_val if valid_val.sum() > 50 else np.ones(len(yv), dtype=bool)
                    auc = float(roc_auc_score(yv[ok], pv[ok])) if (yv[ok].sum() >= 10 and (len(yv[ok]) - yv[ok].sum()) >= 10) else float("nan")
                    members.append({"kind": kind, "seed": int(seed), "model": m,
                                    "rounds": int(r), "auc_val": auc})
            wts, fb = ensemble_weights(members, drop_below)
            models[direction] = members
            print(f"[final] {direction}: {len(members)}成员 (AUC加权, 剔除<{drop_below}), "
                  f"fallback_uniform={fb}, 成员AUC=" + ", ".join(
                      f"{m['kind']}s{m['seed']}:{m['auc_val']:.4f}(w{wi:.3f})"
                      for m, wi in zip(members, wts)))
            for m, wi in zip(members, wts):
                fn = f"{direction}_{m['kind']}_s{m['seed']}_latest" + (".txt" if m["kind"] == "lgb" else ".json")
                m["file"] = fn
                m["weight"] = float(wi)
                members_cfg[direction].append({"kind": m["kind"], "seed": m["seed"], "file": fn,
                                               "rounds": m["rounds"], "weight": float(wi),
                                               "auc_val": m["auc_val"]})
            # 诊断重要性取首个LGB成员
            m0 = next(m for m in members if m["kind"] == "lgb")
            imp = m0["model"].feature_importance(importance_type="gain")
            importance[direction] = {feats[j]: float(imp[j] / max(imp.sum(), 1e-12)) for j in np.argsort(-imp)[:15]}
        else:
            m_lgb, r_lgb = fit_lgb(Xf, y[fit_rows], w, tuned[direction]["lgb"], Xv, yv,
                                   np.ones(n_val), es_rounds=cfg["early_stopping_rounds"],
                                   max_rounds=cfg["max_rounds"])
            m_xgb, r_xgb = fit_xgb(Xf, y[fit_rows], w, tuned[direction]["xgb"], Xv, yv,
                                   np.ones(n_val), es_rounds=cfg["early_stopping_rounds"],
                                   max_rounds=cfg["max_rounds"])
            models[direction] = [("lgb", m_lgb), ("xgb", m_xgb)]
            print(f"[final] {direction}: LGB {r_lgb}轮  XGB {r_xgb}轮")
            # LGB gain重要性 (诊断用)
            imp = m_lgb.feature_importance(importance_type="gain")
            importance[direction] = {feats[j]: float(imp[j] / max(imp.sum(), 1e-12)) for j in np.argsort(-imp)[:15]}

    # 阈值: 用与走查一致的分位数校准 (同一批拟合模型, 尾段20%, 变体目标)
    if use_ens:
        pl = predict_weighted(models["long"], [m["weight"] for m in models["long"]], Xv)
        ps = predict_weighted(models["short"], [m["weight"] for m in models["short"]], Xv)
    else:
        pl = predict_ens(models["long"], Xv)
        ps = predict_ens(models["short"], Xv)
    (thr_l, thr_s), thr_info = calibrate_threshold(pl, ps, val_rows, lab, valid, m5, m1_pack, cfg)
    thr_desc = "回退q=0.975" if thr_info is None else "q={}, 内部验证段{}笔".format(thr_info["q"], thr_info["trades"])
    print(f"[final] 阈值: long={thr_l:.4f} short={thr_s:.4f} ({thr_desc})")

    if use_ens:
        for direction in ["long", "short"]:
            for m in models[direction]:
                fp = os.path.join(args.out_dir, m["file"])
                m["model"].save_model(fp)
                rounds_info[f"{direction}_{m['kind']}_s{m['seed']}"] = m["rounds"]
                print(f"[final] 保存 {m['file']} ({m['rounds']}轮, w={m['weight']:.3f})")
    else:
        for direction in ["long", "short"]:
            for kind, m in models[direction]:
                fp = os.path.join(args.out_dir, f"{direction}_{kind}_latest.json" if kind == "xgb"
                                  else f"{direction}_{kind}_latest.txt")
                m.save_model(fp)
                rounds_info[f"{direction}_{kind}"] = int(getattr(m, "best_iteration", None) or 0)
                print(f"[final] 保存 {os.path.basename(fp)} (最优轮数 {rounds_info[f'{direction}_{kind}']})")
    cfg_out = {
        "variant": args.variant, "feature_set": f"{fspec}{len(feats)}",
        "long_threshold": float(thr_l), "short_threshold": float(thr_s),
        "features_list": feats, "rounds": rounds_info,
        "members": members_cfg if use_ens else None,
        "tp_atr_mult": cfg["tp_atr_mult"], "sl_atr_mult": cfg["sl_atr_mult"],
        "atr_window_m5": cfg["atr_window_m5"], "horizon_m1": cfg["horizon_m1"],
        "cooldown_m1": cfg["cooldown_m1"], "point_value": cfg["point_value"],
        "spread_floor_mult": cfg["spread_floor_mult"], "sl_floor_usd": cfg["sl_floor_usd"],
        "min_m5_bars": (2885 if fspec == "v3" else cfg["atr_window_m5"] + 5),  # v3含10天分母窗口
        "train_window": f"{train_months[0]}~{train_months[-1]}",
        "oos_performance_reference": f"results/summary_{args.variant}.json (24个月走查真实结果)",
    }
    with open(os.path.join(args.out_dir, "trading_config.json"), "w") as f:
        json.dump(cfg_out, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, f"feature_importance_{fspec}.json"), "w") as f:
        json.dump(importance, f, indent=2, ensure_ascii=False)
    print(f"[final] 完成 -> {args.out_dir}")
    print("       注意: ATR障碍意味着每笔交易的TP/SL距离随波动率变化, EA需按 predict_live.py 返回的动态距离下单")


if __name__ == "__main__":
    main()
