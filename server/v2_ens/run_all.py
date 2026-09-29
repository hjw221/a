"""
主编排器: prep(数据+特征+标签缓存) -> tune(首折内部调参, 冻结) -> wf(走查回测) -> report
用法:
  python run_all.py --stage all            # 全流程
  python run_all.py --stage tune --features v2
  python run_all.py --stage wf   --features v2      (断点续跑)
每折保存checkpoint, 中断后重跑自动跳过已完成折。
"""
import os, sys, json, time, argparse, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CFG
from data import load_data
from features import build_features_v2, build_features_legacy
from features_v3 import build_features_v3
from labeling import make_labels
from walkforward import build_folds, time_decay_weights
from models import fit_lgb, fit_xgb, predict_ens, tune_on_window
from backtest import (build_signals, run_backtest, metrics, calibrate_threshold)
from variants_ext import train_variant_fold
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results")
CKPT = os.path.join(RES, "checkpoints")
os.makedirs(CKPT, exist_ok=True)

FP = lambda *a: os.path.join(*a)
MAX_FOLDS = None
TIME_BUDGET = None
DIRECTION = "both"

# ---------------------------------------------------------------- 变体规格
# featset -> (特征构建器, 几何, 参数源, 阈值目标)
#   几何: None=CFG默认(2.0/1.143/120) | "fixed"=原固定$3.5/$2.0
#         "plr"=盈亏比主几何(4.0/1.0/360) | "plr_aggr"=激进(4.0/0.8/360)
#         "bal"=平衡几何(3.0/1.143/360, 胜率下限标定)
#   参数源: 复用同一特征集的已冻结超参 (几何不同但特征相同, 树超参可迁移; 已在报告中注明)
VARIANTS = {
    "v2":            ("v2",     None,        "v2",     "mean_pnl_x_sqrtN"),
    "legacy":        ("legacy", None,        "legacy", "mean_pnl_x_sqrtN"),
    "fixed_v2":      ("v2",     "fixed",    "v2",     "mean_pnl_x_sqrtN"),
    "fixed_legacy":  ("legacy", "fixed",    "legacy", "mean_pnl_x_sqrtN"),
    "v3":            ("v3",     "plr",      "v3",     "plr"),
    "v3aggr":        ("v3",     "plr_aggr", "v3",     "plr"),
    "legacy_plrgeo": ("legacy", "plr",      "legacy", "plr"),
    "legacy_aggr":   ("legacy", "plr_aggr", "legacy", "plr"),
    "legacy_plrgeo_mean": ("legacy", "plr", "legacy", "mean_pnl_x_sqrtN"),  # 阈值目标消融
    "v3bal":         ("v3",     "bal",      "v3",     "plr_wr"),   # 平衡: 胜率下限+PLR
    "legacy_bal":    ("legacy", "bal",      "legacy", "plr_wr"),   # 平衡: 旧特征+平衡几何
    # ---- 信号强度三张牌 (2026-09-14): 复用v3bal的featset/几何/参数源/pack, 只动训练过程 ----
    "v3bal_uniq":    ("v3",     "bal",      "v3",     "plr_wr",
                      {"mode": "stride", "pack": "v3bal", "stride": 12}),
    "v3bal_ens":     ("v3",     "bal",      "v3",     "plr_wr",
                      {"mode": "ens", "pack": "v3bal", "seeds": [42, 1337, 2024], "drop_below": 0.51}),
    "v3bal_meta":    ("v3",     "bal",      "v3",     "plr_wr",
                      {"mode": "meta", "pack": "v3bal", "cand_q": 0.85}),
}


def variant_extra(featset):
    """第5元素拓展dict (4元组旧变体返回空)。"""
    v = VARIANTS[featset]
    return v[4] if len(v) > 4 else {}


def variant_cfg(featset):
    """变体生效配置: 几何覆盖(含horizon, 影响purge与标签) + 阈值目标。"""
    spec = VARIANTS[featset]
    geo, thr_obj = spec[1], spec[3]
    cfg = dict(CFG)
    if geo is None:
        pass
    elif geo == "fixed":
        cfg.update(tp_atr_mult=3.5, sl_atr_mult=2.0)
    else:
        cfg.update(CFG[{"plr": "plr_geometry", "plr_aggr": "plr_geometry_aggr",
                        "bal": "balance_geometry"}[geo]])
    cfg["threshold_objective"] = thr_obj
    return cfg


# ---------------------------------------------------------------- prep
def stage_prep(featsets):
    m5, m1_pack, spread_cost, monthly = load_data(CFG)
    folds = build_folds(m5, CFG)
    print(f"[prep] 走查折: {len(folds)} 个, 首折 {folds[0]['oos_month']} (训练窗 {folds[0]['window']}, "
          f"{folds[0]['train_months']}个月, purge {folds[0]['purge_bars']}根M5)")
    print(f"[prep] 末折 {folds[-1]['oos_month']} (训练窗 {folds[-1]['window']}, {folds[-1]['train_months']}个月)")
    for fs in featsets:
        pack_name = variant_extra(fs).get("pack", fs)
        out = FP(BASE, "cache", f"pack_{pack_name}.pkl")
        if os.path.exists(out):
            print(f"[prep] {fs} 复用缓存 pack_{pack_name}.pkl, 跳过")
            continue
        t0 = time.time()
        fspec, geo = VARIANTS[fs][0], VARIANTS[fs][1]
        cfg_v = variant_cfg(fs)
        if fspec == "v3":
            F, feats, atr = build_features_v3(m5, CFG["atr_window_m5"])
        elif fspec == "v2":
            F, feats, atr = build_features_v2(m5, CFG["atr_window_m5"])
        else:
            F, feats = build_features_legacy(m5)
            F = F.astype(np.float32)
            _, _, atr = build_features_v2(m5, CFG["atr_window_m5"])
        if geo == "fixed":
            # 原固定美元障碍 TP=$3.5 / SL=$2.0 (atr≡1, 乘数=美元数; 供消融对比)
            atr_use = pd.Series(1.0, index=m5.index)
            lab, valid = make_labels(m5, m1_pack, spread_cost, atr_use,
                                      dict(cfg_v, tp_atr_mult=3.5, sl_atr_mult=2.0))
        else:
            lab, valid = make_labels(m5, m1_pack, spread_cost, atr, cfg_v)
        print(f"[prep] {fs}: {F.shape[1]}特征 x {len(F):,}行, 标签有效 {valid.mean()*100:.1f}%, "
              f"TP/SL/horizon={cfg_v['tp_atr_mult']}/{cfg_v['sl_atr_mult']}/{cfg_v['horizon_m1']}M1, 耗时 {time.time()-t0:.0f}s "
              f"| long TP率 {(lab['out_long'][valid]==1).mean()*100:.1f}% "
              f"short {(lab['out_short'][valid]==1).mean()*100:.1f}%")
        with open(out, "wb") as f:
            pickle.dump({"F": F, "feats": feats, "lab": lab, "valid": valid,
                         "spread_cost": spread_cost, "monthly_spread": monthly}, f, protocol=4)
    print("[prep] 完成")


def load_pack(fs):
    pack_name = variant_extra(fs).get("pack", fs)
    with open(FP(BASE, "cache", f"pack_{pack_name}.pkl"), "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- tune
def stage_tune(featsets):
    m5, m1_pack, spread_cost, monthly = load_data(CFG)
    for fs in featsets:
        if variant_extra(fs).get("pack"):
            print(f"[tune] {fs} 复用 {variant_extra(fs)['pack']} 的冻结参数, 跳过")
            continue
        cfg_v = variant_cfg(fs)
        folds = build_folds(m5, cfg_v)          # 变体感知horizon的purge
        f0 = folds[0]
        pack = load_pack(fs)
        F, lab, valid = pack["F"], pack["lab"], pack["valid"]
        cfg_t = dict(cfg_v)
        cfg_t["tune"] = dict(CFG["tune"], n_configs=(12 if fs.startswith("v3") else 6))
        tuned_path = FP(RES, "tuned_params.json")
        tuned = json.load(open(tuned_path)) if os.path.exists(tuned_path) else {}
        tr = f0["train_rows"]
        n = len(tr)
        # 段A: 拟合[0,60%) 验证[60%+purge, 85%) | 段B: 拟合[0,85%) 验证[85%+purge, 100%)
        pb = f0["purge_bars"]
        fit_a, val_a = tr[: int(n * 0.60)], tr[int(n * 0.60) + pb: int(n * 0.85)]
        fit_b, val_b = tr[: int(n * 0.85)], tr[int(n * 0.85) + pb:]
        halflife = CFG["sample_halflife_days"]
        W_all = np.ones(len(F), dtype=np.float64)
        W_all[tr] = time_decay_weights(m5.index, tr, halflife)
        params_out = {}
        for direction in ["long", "short"]:
            if DIRECTION != "both" and direction != DIRECTION:
                continue
            if fs in tuned and direction in tuned.get(fs, {}) and tuned[fs][direction]:
                print(f"[tune] {fs}/{direction} 已有参数, 跳过")
                continue
            col = "out_long" if direction == "long" else "out_short"
            y = (lab[col].to_numpy() == 1).astype(np.int8)
            t0 = time.time()
            best, results = tune_on_window(F, y, cfg_t, fit_a, val_a, fit_b, val_b,
                                           W_all, seed=CFG["random_seed"] + (0 if direction == "long" else 1))
            best_lgb_auc = results["lgb"][0]["auc"] if results["lgb"] else np.nan
            best_xgb_auc = results["xgb"][0]["auc"] if results["xgb"] else np.nan
            print(f"[tune] {fs}/{direction}: LGB最优AUC={best_lgb_auc:.4f}  XGB最优AUC={best_xgb_auc:.4f}  "
                  f"({time.time()-t0:.0f}s)")
            params_out[direction] = {"lgb": best["lgb"], "xgb": best["xgb"],
                                     "inner_auc": {"lgb": best_lgb_auc, "xgb": best_xgb_auc}}
            tuned.setdefault(fs, {})[direction] = params_out[direction]
            json.dump(tuned, open(tuned_path, "w"), indent=2)
    print(f"[tune] 参数已冻结 -> {tuned_path}")


# ---------------------------------------------------------------- walk-forward
def stage_wf(featset):
    t_start = time.time()
    m5, m1_pack, spread_cost, monthly = load_data(CFG)
    cfg_v = variant_cfg(featset)                # 阈值目标/几何生效
    folds = build_folds(m5, cfg_v)              # 变体感知horizon的purge
    pack = load_pack(featset)
    F, feats, lab, valid = pack["F"], pack["feats"], pack["lab"], pack["valid"]
    param_src = VARIANTS[featset][2]
    tuned = json.load(open(FP(RES, "tuned_params.json")))[param_src]

    y_long = (lab["out_long"].to_numpy() == 1).astype(np.int8)
    y_short = (lab["out_short"].to_numpy() == 1).astype(np.int8)
    ml_logs, base_logs_all, per_fold_rows = [], [], []
    rng_master = np.random.default_rng(CFG["random_seed"])

    for i, fold in enumerate(folds):
        if MAX_FOLDS is not None and i >= MAX_FOLDS:
            break
        if TIME_BUDGET is not None and time.time() - t_start > TIME_BUDGET:
            print(f"[wf] 达到时间预算{TIME_BUDGET}s, 剩余折待续跑")
            break
        ck = FP(CKPT, f"wf_{featset}_fold{i:02d}.pkl")
        if os.path.exists(ck):
            d = pickle.load(open(ck, "rb"))
            ml_logs.append(d["log"])
            per_fold_rows.append(d["stats"])
            for k, v in d.get("base_logs", {}).items():
                base_logs_all.append(v)
            print(f"[wf {featset}] fold {i} {fold['oos_month']} 已完成(checkpoint), 跳过")
            continue
        t0 = time.time()
        fit_r, val_r, oos_r = fold["fit_rows"], fold["val_rows"], fold["oos_rows"]
        extra = variant_extra(featset)
        ex_stats = {}
        if extra:
            # ---- 拓展变体: 训练过程由variants_ext接管, 阈值校准/回测口径与基线完全一致 ----
            tv = train_variant_fold(extra, F, y_long, y_short, fold, tuned, cfg_v, m5, valid, i)
            pl_val, ps_val = tv["pl_val"], tv["ps_val"]
            pl_oos, ps_oos = tv["pl_oos"], tv["ps_oos"]
            rounds = tv["rounds"]
            ex_stats = tv["ex_stats"]
            auc_val_pair, auc_oos_pair = tv["auc_val_src"], tv["auc_oos_src"]
        else:
            w_fit = time_decay_weights(m5.index, fit_r, CFG["sample_halflife_days"])
            w_val = np.ones(len(val_r))
            Xf = F.iloc[fit_r].to_numpy(np.float32)
            Xv = F.iloc[val_r].to_numpy(np.float32)
            Xo = F.iloc[oos_r].to_numpy(np.float32)
            models = {}
            rounds = {}
            for direction, y in [("long", y_long), ("short", y_short)]:
                yf, yv = y[fit_r], y[val_r]
                m_lgb, r_lgb = fit_lgb(Xf, yf, w_fit, tuned[direction]["lgb"],
                                       Xv, yv, w_val,
                                       es_rounds=CFG["early_stopping_rounds"], max_rounds=CFG["max_rounds"])
                m_xgb, r_xgb = fit_xgb(Xf, yf, w_fit, tuned[direction]["xgb"],
                                       Xv, yv, w_val,
                                       es_rounds=CFG["early_stopping_rounds"], max_rounds=CFG["max_rounds"])
                models[direction] = [("lgb", m_lgb), ("xgb", m_xgb)]
                rounds[direction] = {"lgb": int(r_lgb), "xgb": int(r_xgb)}
            pl_val = predict_ens(models["long"], Xv)
            ps_val = predict_ens(models["short"], Xv)
            pl_oos = predict_ens(models["long"], Xo)
            ps_oos = predict_ens(models["short"], Xo)
            auc_val_pair, auc_oos_pair = (pl_val, ps_val), (pl_oos, ps_oos)

        thr, thr_info = calibrate_threshold(pl_val, ps_val, val_r, lab, valid, m5, m1_pack, cfg_v)
        thr_l, thr_s = thr
        sig = build_signals(pl_oos, ps_oos, thr_l, thr_s)
        log = run_backtest(sig, oos_r, lab, valid, m5, m1_pack, cfg_v,
                           probs=(pl_oos, ps_oos), tag=f"ml_{featset}")
        log["fold"] = fold["oos_month"]
        m_ml = metrics(log, f"ml_{featset}")

        # ---- 基准对照 (同一模拟器, 同一障碍, 无任何学习) ----
        base_logs = {}
        vv = valid[oos_r]
        sig_al = np.where(vv, 1, 0).astype(np.int8)
        base_logs["always_long"] = run_backtest(sig_al, oos_r, lab, valid, m5, m1_pack, cfg_v, tag="always_long")
        sig_as = np.where(vv, -1, 0).astype(np.int8)
        base_logs["always_short"] = run_backtest(sig_as, oos_r, lab, valid, m5, m1_pack, cfg_v, tag="always_short")
        rng = np.random.default_rng(cfg_v["random_seed"] * 1000 + i)
        coin = rng.integers(0, 2, size=len(oos_r))
        sig_rc = np.where(vv, np.where(coin == 0, 1, -1), 0).astype(np.int8)
        base_logs["random_coin"] = run_backtest(sig_rc, oos_r, lab, valid, m5, m1_pack, cfg_v, tag="random_coin")
        # 频率匹配随机: 与ML同数量的信号, 随机时刻与方向
        n_sig = int((sig != 0).sum())
        cand = np.where(vv)[0]
        if n_sig > 0 and len(cand) > 0:
            pick = rng.choice(cand, size=min(n_sig, len(cand)), replace=False)
            sig_rm = np.zeros(len(oos_r), dtype=np.int8)
            dircoin = rng.integers(0, 2, size=len(pick))
            sig_rm[pick] = np.where(dircoin == 0, 1, -1)
            base_logs["random_matched"] = run_backtest(sig_rm, oos_r, lab, valid, m5, m1_pack, cfg_v, tag="random_matched")
        else:
            base_logs["random_matched"] = log.iloc[0:0].copy()
        for k, v in base_logs.items():
            v["fold"] = fold["oos_month"]

        stats = {
            "fold": i, "oos_month": fold["oos_month"], "window": fold["window"],
            "train_months": fold["train_months"], "n_fit": len(fit_r), "n_oos": len(oos_r),
            "threshold": {"long": thr_l, "short": thr_s}, "thr_fallback": thr_info is None,
            "rounds": rounds,
            "auc_val_long": float(roc_auc_score(y_long[val_r][valid[val_r]], auc_val_pair[0][valid[val_r]])) if valid[val_r].sum() > 50 else np.nan,
            "auc_val_short": float(roc_auc_score(y_short[val_r][valid[val_r]], auc_val_pair[1][valid[val_r]])) if valid[val_r].sum() > 50 else np.nan,
            "auc_oos_long": float(roc_auc_score(y_long[oos_r][vv], auc_oos_pair[0][vv])) if vv.sum() > 50 else np.nan,
            "auc_oos_short": float(roc_auc_score(y_short[oos_r][vv], auc_oos_pair[1][vv])) if vv.sum() > 50 else np.nan,
            "base_long": float(y_long[oos_r][vv].mean()), "base_short": float(y_short[oos_r][vv].mean()),
            "ml": m_ml,
            "baselines": {k: metrics(v, k) for k, v in base_logs.items()},
            "n_signals": n_sig,
            "extra": ex_stats,
        }
        for k, v in base_logs.items():
            v["variant"] = k
            base_logs_all.append(v)
        ml_logs.append(log)
        per_fold_rows.append(stats)
        pickle.dump({"stats": stats, "log": log, "base_logs": base_logs},
                    open(ck, "wb"), protocol=4)
        print(f"[wf {featset}] fold {i:02d} {fold['oos_month']} 窗{fold['window']}({fold['train_months']}月) "
              f"thr(L/S)={thr_l:.3f}/{thr_s:.3f}{'(回退)' if thr_info is None else ''} 交易{m_ml['trades']}笔 "
              f"胜率{(m_ml['win_rate']*100 if m_ml['trades'] else float('nan')):.1f}% "
              f"盈亏比{(m_ml['plr'] if m_ml['trades'] else float('nan')):.2f} "
              f"PnL${m_ml['total_pnl']:.1f} AUC(L/S)={stats['auc_oos_long']:.3f}/{stats['auc_oos_short']:.3f} "
              f"({time.time()-t0:.0f}s)")

    # ---- 汇总 ----
    trades = pd.concat(ml_logs, ignore_index=True) if ml_logs else log.iloc[0:0].copy()
    trades.to_csv(FP(RES, f"trades_{featset}.csv"), index=False)
    pd.DataFrame(per_fold_rows).to_json(FP(RES, f"per_fold_{featset}.json"), orient="records", indent=2)
    m_all = metrics(trades, f"ml_{featset}")
    print(f"\n===== {featset} 走查汇总 (真实OOS, {len(folds)}个月) =====")
    for k, v in m_all.items():
        print(f"  {k}: {v}")
    base_all = {}
    for name in ["always_long", "always_short", "random_coin", "random_matched"]:
        bl = [l for l in base_logs_all if len(l) > 0 and str(l["variant"].iloc[0]) == name]
        bl = pd.concat(bl, ignore_index=True) if bl else trades.iloc[0:0].copy()
        base_all[name] = metrics(bl, name)
        bl.to_csv(FP(RES, f"trades_baseline_{name}.csv"), index=False)
    json.dump({"ml": m_all, "baselines": base_all},
              open(FP(RES, f"summary_{featset}.json"), "w"), indent=2)
    for name, mm in base_all.items():
        print(f"  基准[{name}]: 交易{mm['trades']} 胜率{mm['win_rate']} PnL${mm['total_pnl']:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prep", "tune", "wf", "all", "report"], default="all")
    ap.add_argument("--features", default="v2,legacy")
    ap.add_argument("--max-folds", type=int, default=None, help="只跑前N折(冒烟测试)")
    ap.add_argument("--time-budget", type=int, default=None, help="秒: 超时后不再开新折(断点续跑)")
    ap.add_argument("--direction", choices=["long", "short", "both"], default="both", help="tune阶段分方向跑")
    args = ap.parse_args()
    featsets = args.features.split(",")
    MAX_FOLDS = args.max_folds
    TIME_BUDGET = args.time_budget
    DIRECTION = args.direction

    if args.stage in ("prep", "all"):
        stage_prep(featsets)
    if args.stage in ("tune", "all"):
        stage_tune(featsets)
    if args.stage in ("wf", "all"):
        for fs in featsets:
            stage_wf(fs)
    print("\n完成。")
