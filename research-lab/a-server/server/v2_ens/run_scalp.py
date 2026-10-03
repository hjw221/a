"""
M1剥头皮走查编排器 (Task14): prep -> tune -> wf。
用法:
  python run_scalp.py --stage prep
  python run_scalp.py --stage tune
  python run_scalp.py --stage wf --max-folds 3        # 先导3折 (kill检查点)
  python run_scalp.py --stage wf --time-budget 540    # 前台分块续跑
几何(TP=$1.00/SL/horizon)来自 scripts/scalp_geom_result.json (首训练窗标定, 冻结)。
防泄露与既有管线同标准: 训练窗右端purge(horizon+30), fit/val间purge gap,
阈值只在训练窗尾段校准, OOS月零参与, 超参首折内部调参后全程冻结。
"""
import os, sys, json, time, argparse, pickle
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from data import load_raw_m1, impute_spread
from features_s1 import build_features_s1
from scalp import (make_labels_m1, build_folds_m1, run_backtest_m1,
                   calibrate_threshold_m1)
from backtest import build_signals, metrics
from models import tune_on_window
from walkforward import time_decay_weights
from variants_ext import _fold_ens, scale_tree_params
from sklearn.metrics import roc_auc_score

RES = os.path.join(BASE, "results")
CKPT = os.path.join(RES, "checkpoints")
os.makedirs(CKPT, exist_ok=True)
GEOM = json.load(open("/home/z/my-project/scripts/scalp_geom_result.json"))["chosen"]

CFG = {
    "first_oos_month": "2024-08", "last_oos_month": "2026-07",
    "max_train_months": 36,
    "scalp_embargo_m1": 30,          # embargo = 30分钟 (与既有管线30分钟embargo等价)
    "inner_val_frac": 0.20,
    "min_trades_inner": 300,
    "cooldown_m1": 10,
    "sample_halflife_days": 270,
    "early_stopping_rounds": 50,
    "max_rounds": 600,
    "random_seed": 42,
    "tune": {"n_configs": 12, "inner_folds": 2, "subsample_step": 4},
}
PACK = os.path.join(BASE, "cache", "pack_scalp1.pkl")
TUNED_FP = os.path.join(RES, "tuned_params_scalp.json")

# ---------------------------------------------------------------- prep
def stage_prep():
    t0 = time.time()
    print("[prep] 加载M1 ...")
    m1 = load_raw_m1("/home/z/my-project/upload/5_extracted/"
                     "XAUUSDc_M1_202201022305_202606262057.csv")
    print(f"[prep] M1 {len(m1):,} 行 {m1.index[0]} ~ {m1.index[-1]} ({time.time()-t0:.0f}s)")
    spread_cost, monthly = impute_spread(m1, 0.001)

    print(f"[prep] 构建s1特征 (TP=${GEOM['tp_usd']:.2f} SL=${GEOM['sl_usd']:.2f} "
          f"HZ={GEOM['horizon_m1']}min) ...")
    F, feats, atr = build_features_s1(m1)
    print(f"[prep] 特征 {F.shape[1]} 个 x {len(F):,} 行 ({time.time()-t0:.0f}s)")

    m1_t = (m1.index.astype("datetime64[s]").astype("int64") // 60).to_numpy(np.int64)
    m1_pack = (m1_t, m1["OPEN"].to_numpy(np.float64), m1["HIGH"].to_numpy(np.float64),
               m1["LOW"].to_numpy(np.float64), m1["CLOSE"].to_numpy(np.float64))
    lab, valid = make_labels_m1(m1_pack, spread_cost, GEOM)
    folds = build_folds_m1(m1.index, CFG, GEOM["horizon_m1"])
    print(f"[prep] 标签有效 {valid.mean()*100:.2f}% | 折数 {len(folds)} "
          f"(首折 {folds[0]['oos_month']} 训练窗 {folds[0]['window']} "
          f"purge {folds[0]['purge_bars']}根M1)")
    print(f"[prep] 全期标签口径: long TP率 {(lab['out_long'][valid]==1).mean()*100:.1f}% "
          f"short {(lab['out_short'][valid]==1).mean()*100:.1f}% "
          f"({time.time()-t0:.0f}s)")
    with open(PACK, "wb") as f:
        pickle.dump({"F": F, "feats": feats, "lab": lab, "valid": valid,
                     "spread_cost": spread_cost, "monthly_spread": monthly,
                     "m1_pack": m1_pack, "geometry": GEOM}, f, protocol=4)
    print(f"[prep] -> {PACK} ({time.time()-t0:.0f}s 总耗时)")


# ---------------------------------------------------------------- tune
def stage_tune():
    with open(PACK, "rb") as f:
        pack = pickle.load(f)
    F, lab, valid, m1_pack = pack["F"], pack["lab"], pack["valid"], pack["m1_pack"]
    m1_index = pd.to_datetime(m1_pack[0], unit="m")
    folds = build_folds_m1(m1_index, CFG, GEOM["horizon_m1"])
    f0 = folds[0]
    y_long = (lab["out_long"].to_numpy() == 1).astype(np.int8)
    y_short = (lab["out_short"].to_numpy() == 1).astype(np.int8)
    tuned = json.load(open(TUNED_FP)) if os.path.exists(TUNED_FP) else {}
    if "scalp1" in tuned and tuned["scalp1"].get("long"):
        print("[tune] 已有冻结参数, 跳过")
        return
    tr = f0["train_rows"]
    n = len(tr)
    pb = f0["purge_bars"]
    fit_a, val_a = tr[: int(n * 0.60)], tr[int(n * 0.60) + pb: int(n * 0.85)]
    fit_b, val_b = tr[: int(n * 0.85)], tr[int(n * 0.85) + pb:]
    W_all = np.ones(len(F), dtype=np.float64)
    W_all[tr] = time_decay_weights(m1_index, tr, CFG["sample_halflife_days"])
    cfg_t = dict(CFG)
    cfg_t["tune"] = dict(CFG["tune"])
    out = {}
    for direction, y in [("long", y_long), ("short", y_short)]:
        t0 = time.time()
        best, results = tune_on_window(F, y, cfg_t, fit_a, val_a, fit_b, val_b,
                                       W_all, seed=CFG["random_seed"] +
                                       (0 if direction == "long" else 1))
        auc_l = results["lgb"][0]["auc"] if results["lgb"] else np.nan
        auc_x = results["xgb"][0]["auc"] if results["xgb"] else np.nan
        print(f"[tune] scalp1/{direction}: LGB AUC={auc_l:.4f} XGB AUC={auc_x:.4f} "
              f"({time.time()-t0:.0f}s)")
        out[direction] = {"lgb": best["lgb"], "xgb": best["xgb"],
                          "inner_auc": {"lgb": auc_l, "xgb": auc_x}}
    tuned["scalp1"] = out
    json.dump(tuned, open(TUNED_FP, "w"), indent=2)
    print(f"[tune] 参数冻结 -> {TUNED_FP}")


# ---------------------------------------------------------------- walk-forward
def stage_wf(max_folds=None, time_budget=None, train_stride=1):
    t_start = time.time()
    with open(PACK, "rb") as f:
        pack = pickle.load(f)
    F, lab, valid, m1_pack = pack["F"], pack["lab"], pack["valid"], pack["m1_pack"]
    m1_index = pd.to_datetime(m1_pack[0], unit="m")
    folds = build_folds_m1(m1_index, CFG, GEOM["horizon_m1"])
    tuned = json.load(open(TUNED_FP))["scalp1"]
    if train_stride > 1:
        tuned = {d: {"lgb": scale_tree_params(tuned[d]["lgb"], "lgb", 1.0 / train_stride),
                    "xgb": scale_tree_params(tuned[d]["xgb"], "xgb", 1.0 / train_stride)}
                 for d in tuned}
    y_long = (lab["out_long"].to_numpy() == 1).astype(np.int8)
    y_short = (lab["out_short"].to_numpy() == 1).astype(np.int8)
    cfg_v = dict(CFG, cooldown_m1=CFG["cooldown_m1"], random_seed=CFG["random_seed"])
    extra = {"mode": "ens", "seeds": [42, 1337, 2024], "drop_below": 0.51}

    ml_logs, base_logs_all, per_fold = [], [], []
    done = 0
    for i, fold in enumerate(folds):
        if max_folds is not None and i >= max_folds:
            break
        if time_budget is not None and time.time() - t_start > time_budget:
            print(f"[wf] 达到时间预算{time_budget}s, 剩余折待续跑")
            break
        ck = os.path.join(CKPT, f"wf_scalp1_fold{i:02d}.pkl")
        if os.path.exists(ck):
            d = pickle.load(open(ck, "rb"))
            ml_logs.append(d["log"]); per_fold.append(d["stats"])
            for k, v in d.get("base_logs", {}).items():
                base_logs_all.append(v)
            print(f"[wf scalp1] fold {i} {fold['oos_month']} 已完成(checkpoint), 跳过")
            done += 1
            continue
        t0 = time.time()
        f_eff = dict(fold)
        if train_stride > 1:
            phase = (i * 7) % train_stride
            f_eff["fit_rows"] = fold["fit_rows"][phase::train_stride]
        from variants_ext import train_variant_fold
        val_r, oos_r = fold["val_rows"], fold["oos_rows"]
        tv = train_variant_fold(extra, F, y_long, y_short, f_eff, tuned, cfg_v,
                                pd.DataFrame(index=m1_index), valid, i)
        pl_val, ps_val = tv["pl_val"], tv["ps_val"]
        pl_oos, ps_oos = tv["pl_oos"], tv["ps_oos"]
        thr, thr_info = calibrate_threshold_m1(pl_val, ps_val, val_r, lab, valid,
                                                m1_pack, cfg_v)
        thr_l, thr_s = thr
        sig = build_signals(pl_oos, ps_oos, thr_l, thr_s)
        log = run_backtest_m1(sig, oos_r, lab, valid, m1_pack, cfg_v,
                              probs=(pl_oos, ps_oos), tag="ml_scalp1")
        log["fold"] = fold["oos_month"]
        m_ml = metrics(log, "ml_scalp1")

        base_logs = {}
        vv = valid[oos_r]
        base_logs["always_long"] = run_backtest_m1(
            np.where(vv, 1, 0).astype(np.int8), oos_r, lab, valid, m1_pack, cfg_v, tag="always_long")
        base_logs["always_short"] = run_backtest_m1(
            np.where(vv, -1, 0).astype(np.int8), oos_r, lab, valid, m1_pack, cfg_v, tag="always_short")
        rng = np.random.default_rng(cfg_v["random_seed"] * 1000 + i)
        coin = rng.integers(0, 2, size=len(oos_r))
        base_logs["random_coin"] = run_backtest_m1(
            np.where(vv, np.where(coin == 0, 1, -1), 0).astype(np.int8),
            oos_r, lab, valid, m1_pack, cfg_v, tag="random_coin")
        n_sig = int((sig != 0).sum())
        cand = np.where(vv)[0]
        if n_sig > 0 and len(cand) > 0:
            pick = rng.choice(cand, size=min(n_sig, len(cand)), replace=False)
            sig_rm = np.zeros(len(oos_r), dtype=np.int8)
            sig_rm[pick] = np.where(rng.integers(0, 2, size=len(pick)) == 0, 1, -1)
            base_logs["random_matched"] = run_backtest_m1(
                sig_rm, oos_r, lab, valid, m1_pack, cfg_v, tag="random_matched")
        else:
            base_logs["random_matched"] = log.iloc[0:0].copy()
        for k, v in base_logs.items():
            v["fold"] = fold["oos_month"]

        stats = {
            "fold": i, "oos_month": fold["oos_month"], "window": fold["window"],
            "train_months": fold["train_months"], "n_fit": len(f_eff["fit_rows"]),
            "n_oos": len(oos_r),
            "threshold": {"long": thr_l, "short": thr_s},
            "thr_fallback": thr_info is None, "thr_info": thr_info,
            "rounds": tv["rounds"],
            "auc_val_long": tv["ex_stats"]["directions"]["long"]["mean_auc_val"],
            "auc_val_short": tv["ex_stats"]["directions"]["short"]["mean_auc_val"],
            "auc_oos_long": float(roc_auc_score(y_long[oos_r][vv], pl_oos[vv])) if vv.sum() > 50 else np.nan,
            "auc_oos_short": float(roc_auc_score(y_short[oos_r][vv], ps_oos[vv])) if vv.sum() > 50 else np.nan,
            "base_long": float(y_long[oos_r][vv].mean()),
            "base_short": float(y_short[oos_r][vv].mean()),
            "ml": m_ml,
            "baselines": {k: metrics(v, k) for k, v in base_logs.items()},
            "n_signals": n_sig,
            "extra": {"mode": "ens_m1", "train_stride": train_stride,
                      "geometry": GEOM,
                      "members": tv["ex_stats"]["directions"]},
        }
        for k, v in base_logs.items():
            v["variant"] = k
            base_logs_all.append(v)
        ml_logs.append(log)
        per_fold.append(stats)
        pickle.dump({"stats": stats, "log": log, "base_logs": base_logs},
                    open(ck, "wb"), protocol=4)
        done += 1
        print(f"[wf scalp1] fold {i:02d} {fold['oos_month']} 窗{fold['window']}"
              f"({fold['train_months']}月) thr(L/S)={thr_l:.3f}/{thr_s:.3f}"
              f"{'(回退)' if thr_info is None else ''} 交易{m_ml['trades']}笔 "
              f"胜率{(m_ml['win_rate']*100 if m_ml['trades'] else float('nan')):.1f}% "
              f"PnL${m_ml['total_pnl']:.1f} AUC(L/S)={stats['auc_oos_long']:.3f}/"
              f"{stats['auc_oos_short']:.3f} ({time.time()-t0:.0f}s)")

    # ---- 汇总 (只要有新完成的折) ----
    if done:
        trades = pd.concat(ml_logs, ignore_index=True)
        trades.to_csv(os.path.join(RES, "trades_scalp1.csv"), index=False)
        pd.DataFrame(per_fold).to_json(os.path.join(RES, "per_fold_scalp1.json"),
                                       orient="records", indent=2)
        m_all = metrics(trades, "ml_scalp1")
        json.dump({"ml": m_all, "geometry": GEOM, "train_stride": train_stride,
                   "n_folds_done": done},
                  open(os.path.join(RES, "summary_scalp1.json"), "w"), indent=2)
        print(f"\n===== scalp1 已完成{done}折汇总 (真实OOS) =====")
        for k, v in m_all.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prep", "tune", "wf", "all"], default="all")
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--time-budget", type=int, default=None)
    ap.add_argument("--train-stride", type=int, default=1)
    args = ap.parse_args()
    if args.stage in ("prep", "all"):
        stage_prep()
    if args.stage in ("tune", "all"):
        stage_tune()
    if args.stage in ("wf", "all"):
        stage_wf(args.max_folds, args.time_budget, args.train_stride)
    print("\n完成。")
