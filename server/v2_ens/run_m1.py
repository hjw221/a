"""m1sc_ad 复现与续训驱动 (重建自 pack_m1sc_ad.pkl ground truth, 2026-09-29)。

变体口径 (与 2026-09-27 原始 m1sc_ad 完全一致, 从 pack 逆向验证, 误差<1e-15):
  - 特征: features_s1 (34个 M1 原生特征)
  - 标签: 自适应障碍 — TP = max($1.20, 1.5 x ATR288_M1)
                       SL = max(0.4 x TP, 2 x 点差成本)   [无上限]
          horizon = 90 根 M1, 入场=下一根 M1 开盘, entry_tol=10
  - 训练: lgb + xgb (冻结超参, 见 --stage tune), 时间衰减权重, 早停50, 上限600轮
  - 阈值: plr_wr (wr_floor=0.40, min_trades=300, 分位网格) — 无候选回退 q0.975
  - 走查: build_folds_m1 (purge = 90 + 30 embargo), 单持仓+冷却10根M1
超参说明: 原始 m1sc_ad 的随机搜索超参随驱动脚本一起丢失 (pack 只存特征+标签);
  本驱动 --stage tune 按原协议 (tune_on_window, 12配置x2内折, 子采样步长4,
  seed 42/43) 在首训练窗 2022-01~2024-07 重新调参并冻结到 tuned_params_m1.json。
预注册续训臂 (共享同一 pack / 同一冻结超参 / 同一折结构, 严格消融):
  m1sc_ad           A0 基线重建 (单 lgb+xgb)
  m1sc_ad_ens       A1 多种子集成 (v3bal 制胜牌: lgb+xgb x 3种子, AUC加权)
  m1sc_ad_ens6      A2 深集成 (6种子 x lgb+xgb = 12成员/方向)
  m1sc_ad_ens_hl180 A3 近因加权 (半衰期270d->180d, 针对利润集中2026)
用法:
  python run_m1.py --stage prep --csv <raw.csv>
  python run_m1.py --stage tune --csv <raw.csv>
  python run_m1.py --stage wf --variant m1sc_ad
环境: ML_THREADS 覆盖训练线程 (默认2, 与历史口径一致; 各臂须一致以保证可比)
"""
import os, sys, json, time, argparse, pickle
import numpy as np
import pandas as pd
from numba import njit

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from data import load_raw_m1, impute_spread
from features_s1 import build_features_s1
from scalp import build_folds_m1, run_backtest_m1
from backtest import build_signals, metrics
from models import fit_lgb, fit_xgb, predict_ens, tune_on_window
from walkforward import time_decay_weights
from sklearn.metrics import roc_auc_score

RES = os.path.join(BASE, "results")
CKPT = os.path.join(RES, "checkpoints")
os.makedirs(CKPT, exist_ok=True)
PACK_FP = os.path.join(BASE, "cache", "pack_m1sc_ad.pkl")
TUNED_FP = os.path.join(RES, "tuned_params_m1.json")
RAW_CSV_DEFAULT = "/root/rivermind-fs/xauusd/data/XAUUSDc_M1_202201022305_202606262057.csv"

GEOM = {"tp_floor_usd": 1.20, "tp_atr_mult": 1.5, "atr_win_m1": 288,
        "sl_mult": 0.4, "sl_spread_mult": 2.0, "horizon_m1": 90, "entry_tol_min": 10}

CFG = {"first_oos_month": "2024-08", "last_oos_month": "2026-07",
       "max_train_months": 36, "scalp_embargo_m1": 30, "inner_val_frac": 0.20,
       "min_trades_inner": 300, "wr_floor": 0.40,
       "cooldown_m1": 10, "sample_halflife_days": 270,
       "early_stopping_rounds": 50, "max_rounds": 600, "random_seed": 42}

VARIANTS = {
    "m1sc_ad":           {"mode": "single", "pack": "m1sc_ad"},
    "m1sc_ad_ens":       {"mode": "ens", "seeds": [42, 1337, 2024], "pack": "m1sc_ad"},
    "m1sc_ad_ens6":      {"mode": "ens", "seeds": [42, 1337, 2024, 7, 101, 555], "pack": "m1sc_ad"},
    "m1sc_ad_ens_hl180": {"mode": "ens", "seeds": [42, 1337, 2024], "halflife": 180, "pack": "m1sc_ad"},
}


# ---------------------------------------------------------------- 标签引擎 (自适应障碍)
@njit(cache=True)
def label_all_m1_ad(m1_t, m1_o, m1_h, m1_l, m1_c, spread_cost,
                    tp_floor, tp_atr_mult, sl_mult, sl_spread_mult,
                    atr_win, horizon, entry_tol_min):
    """自适应$三重障碍: 逐bar tp/sl。事件=每根M1, 下一根M1开盘入场。"""
    n = len(m1_t)
    INF = 2**62
    atr = np.full(n, np.nan)
    acc = 0.0
    for i in range(n):
        tr = m1_h[i] - m1_l[i]
        if i > 0:
            a = m1_h[i] - m1_c[i-1]
            if a > tr: tr = a
            b = m1_l[i] - m1_c[i-1]
            if -b > tr: tr = -b
        acc += tr
        if i >= atr_win:
            lo_i = i - atr_win
            tr_old = m1_h[lo_i] - m1_l[lo_i]
            if lo_i > 0:
                a = m1_h[lo_i] - m1_c[lo_i-1]
                if a > tr_old: tr_old = a
                b = m1_l[lo_i] - m1_c[lo_i-1]
                if -b > tr_old: tr_old = -b
            acc -= tr_old
        if i >= atr_win - 1:
            atr[i] = acc / atr_win
    entry_idx = np.full(n, -1, dtype=np.int64)
    out_long = np.zeros(n, dtype=np.int8)
    out_short = np.zeros(n, dtype=np.int8)
    pnl_long = np.zeros(n, dtype=np.float64)
    pnl_short = np.zeros(n, dtype=np.float64)
    exit_bar_long = np.full(n, -1, dtype=np.int64)
    exit_bar_short = np.full(n, -1, dtype=np.int64)
    tp_d = np.zeros(n, dtype=np.float64)
    sl_d = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if np.isnan(atr[i]):
            continue
        sig_time = m1_t[i] + 1
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if m1_t[mid] < sig_time:
                lo = mid + 1
            else:
                hi = mid
        e = lo
        if e >= n or m1_t[e] > sig_time + entry_tol_min:
            continue
        tp = tp_floor
        cand = tp_atr_mult * atr[i]
        if cand > tp:
            tp = cand
        sl = sl_mult * tp
        f = sl_spread_mult * spread_cost[i]
        if f > sl:
            sl = f
        entry = m1_o[e]
        sc = spread_cost[i]
        tp_d[i] = tp
        sl_d[i] = sl
        entry_idx[i] = e
        end = e + horizon
        end_cap = end if end < n else n - 1
        l_tp, l_sl = entry + tp, entry - sl
        s_tp, s_sl = entry - tp, entry + sl
        j_l, j_s = INF, INF
        l_hit, s_hit = 0, 0
        timeout_close = m1_c[end_cap]
        j = e + 1
        while j <= end_cap:
            if l_hit == 0:
                if m1_l[j] <= l_sl:
                    l_hit = -1; j_l = j
                elif m1_h[j] >= l_tp:
                    l_hit = 1; j_l = j
            if s_hit == 0:
                if m1_h[j] >= s_sl:
                    s_hit = -1; j_s = j
                elif m1_l[j] <= s_tp:
                    s_hit = 1; j_s = j
            if l_hit != 0 and s_hit != 0:
                break
            j += 1
        if l_hit == 0:
            j_l = end_cap
            pnl_long[i] = timeout_close - entry - sc
        else:
            j_l = j_l if j_l < n else n - 1
            pnl_long[i] = (l_tp if l_hit == 1 else l_sl) - entry - sc
            out_long[i] = l_hit
        if s_hit == 0:
            j_s = end_cap
            pnl_short[i] = entry - timeout_close - sc
        else:
            j_s = j_s if j_s < n else n - 1
            pnl_short[i] = (s_tp if s_hit == 1 else s_sl) - entry - sc
            out_short[i] = s_hit
        exit_bar_long[i] = j_l if j_l < INF else end_cap
        exit_bar_short[i] = j_s if j_s < INF else end_cap
    return (entry_idx, out_long, out_short, pnl_long, pnl_short,
            exit_bar_long, exit_bar_short, tp_d, sl_d)


def make_labels_m1_ad(m1_pack, spread_cost, g):
    res = label_all_m1_ad(m1_pack[0], m1_pack[1], m1_pack[2], m1_pack[3], m1_pack[4],
                          spread_cost, g["tp_floor_usd"], g["tp_atr_mult"],
                          g["sl_mult"], g["sl_spread_mult"], g["atr_win_m1"],
                          g["horizon_m1"], g["entry_tol_min"])
    (entry_idx, out_long, out_short, pnl_long, pnl_short,
     exit_bar_long, exit_bar_short, tp_d, sl_d) = res
    lab = pd.DataFrame({"entry_idx": entry_idx, "out_long": out_long,
                        "out_short": out_short, "pnl_long": pnl_long,
                        "pnl_short": pnl_short, "exit_bar_long": exit_bar_long,
                        "exit_bar_short": exit_bar_short, "tp_d": tp_d, "sl_d": sl_d})
    valid = (entry_idx >= 0) & np.isfinite(pnl_long)
    return lab, valid


def load_m1_pack(csv_path):
    m1 = load_raw_m1(csv_path)
    m1_t = (m1.index.astype("int64") // 10**9 // 60).to_numpy(np.int64)
    return (m1_t, m1["OPEN"].to_numpy(np.float64), m1["HIGH"].to_numpy(np.float64),
            m1["LOW"].to_numpy(np.float64), m1["CLOSE"].to_numpy(np.float64))


# ---------------------------------------------------------------- 阈值校准 (plr_wr)
def calibrate_threshold_m1_ad(pl, ps, rows, lab, valid, m1_pack, cfg):
    qs = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.985]
    cands = []
    for q in qs:
        thr_l = float(np.quantile(pl, q))
        thr_s = float(np.quantile(ps, q))
        sig = build_signals(pl, ps, thr_l, thr_s)
        log = run_backtest_m1(sig, rows, lab, valid, m1_pack, cfg)
        n = len(log)
        if n < cfg["min_trades_inner"] or log["pnl"].sum() <= 0:
            continue
        mm = metrics(log)
        cands.append((q, thr_l, thr_s, n,
                      mm["win_rate"] if np.isfinite(mm["win_rate"]) else 0.0,
                      mm["plr"] if np.isfinite(mm["plr"]) else 0.0,
                      log["pnl"].mean() * np.sqrt(n)))
    if not cands:
        thr_l = float(np.quantile(pl, 0.975))
        thr_s = float(np.quantile(ps, 0.975))
        return (thr_l, thr_s), None
    meet = [c for c in cands if c[4] >= cfg["wr_floor"]]
    if meet:
        best = max(meet, key=lambda c: (c[5], c[6]))
    else:
        best = max(cands, key=lambda c: (c[6],))
    return (best[1], best[2]), {"q": best[0], "trades": best[3], "wr": best[4],
                                "plr": best[5], "score": best[6]}


# ---------------------------------------------------------------- 阶段
def stage_prep(csv_path):
    t0 = time.time()
    print("[prep] 加载M1 ...")
    m1 = load_raw_m1(csv_path)
    print(f"[prep] M1 {len(m1):,} 行 {m1.index[0]} ~ {m1.index[-1]} ({time.time()-t0:.0f}s)")
    spread_cost, monthly = impute_spread(m1, 0.001)
    print("[prep] 构建 s1 特征 (34个) ...")
    F, feats, _ = build_features_s1(m1)
    print(f"[prep] 特征 {F.shape[1]} 个 x {len(F):,} 行 ({time.time()-t0:.0f}s)")
    m1_pack = load_m1_pack(csv_path)
    print("[prep] 自适应障碍标签 (TP=max(1.2,1.5xATR288), SL=max(0.4TP,2x点差), H=90) ...")
    lab, valid = make_labels_m1_ad(m1_pack, spread_cost, GEOM)
    print(f"[prep] 标签有效 {valid.mean()*100:.2f}% | long TP率 {(lab['out_long'][valid]==1).mean()*100:.1f}% "
          f"short {(lab['out_short'][valid]==1).mean()*100:.1f}% ({time.time()-t0:.0f}s)")
    with open(PACK_FP, "wb") as f:
        pickle.dump({"F": F, "feats": feats, "lab": lab, "valid": valid,
                     "spread_cost": spread_cost, "monthly_spread": monthly,
                     "geometry": GEOM}, f, protocol=4)
    print(f"[prep] -> {PACK_FP} ({time.time()-t0:.0f}s 总耗时)")


def stage_tune(csv_path):
    """首训练窗内重新调参 (原协议: 12配置x2内折, 子采样步长4, seed 42/43)。"""
    with open(PACK_FP, "rb") as f:
        pack = pickle.load(f)
    F, lab, valid = pack["F"], pack["lab"], pack["valid"]
    m1_pack = load_m1_pack(csv_path)
    m1_index = pd.to_datetime(m1_pack[0], unit="m")
    folds = build_folds_m1(m1_index, CFG, GEOM["horizon_m1"])
    f0 = folds[0]
    y_long = (lab["out_long"].to_numpy() == 1).astype(np.int8)
    y_short = (lab["out_short"].to_numpy() == 1).astype(np.int8)
    tuned = json.load(open(TUNED_FP)) if os.path.exists(TUNED_FP) else {}
    if "m1sc_ad" in tuned and tuned["m1sc_ad"].get("long"):
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
    cfg_t["tune"] = {"n_configs": 12, "inner_folds": 2, "subsample_step": 4}
    out = {}
    for direction, y in [("long", y_long), ("short", y_short)]:
        t0 = time.time()
        best, results = tune_on_window(F, y, cfg_t, fit_a, val_a, fit_b, val_b,
                                       W_all, seed=CFG["random_seed"] +
                                       (0 if direction == "long" else 1))
        auc_l = results["lgb"][0]["auc"] if results["lgb"] else np.nan
        auc_x = results["xgb"][0]["auc"] if results["xgb"] else np.nan
        print(f"[tune] m1sc_ad/{direction}: LGB AUC={auc_l:.4f} XGB AUC={auc_x:.4f} "
              f"({time.time()-t0:.0f}s)")
        out[direction] = {"lgb": best["lgb"], "xgb": best["xgb"],
                          "inner_auc": {"lgb": auc_l, "xgb": auc_x}}
    tuned["m1sc_ad"] = out
    json.dump(tuned, open(TUNED_FP, "w"), indent=2)
    print(f"[tune] 参数冻结 -> {TUNED_FP}")


def _fit_pair(Xf, yf, w_fit, params_d, Xv, yv, w_val, es, mr):
    m_lgb, r_lgb = fit_lgb(Xf, yf, w_fit, params_d["lgb"], Xv, yv, w_val, es, mr)
    m_xgb, r_xgb = fit_xgb(Xf, yf, w_fit, params_d["xgb"], Xv, yv, w_val, es, mr)
    return [("lgb", m_lgb), ("xgb", m_xgb)], {"lgb": int(r_lgb), "xgb": int(r_xgb)}


def stage_wf(variant, max_folds=None, csv_path=None):
    from variants_ext import (_lgb_predict, _xgb_predict, _auc,
                              ensemble_weights, predict_weighted)
    t_start = time.time()
    spec = VARIANTS[variant]
    pack_fp = os.path.join(BASE, "cache", f"pack_{spec.get('pack', variant)}.pkl")
    with open(pack_fp, "rb") as f:
        pack = pickle.load(f)
    F, lab, valid = pack["F"], pack["lab"], pack["valid"]
    m1_pack = pack.get("m1_pack") or load_m1_pack(csv_path or RAW_CSV_DEFAULT)
    m1_index = pd.to_datetime(m1_pack[0], unit="m")
    folds = build_folds_m1(m1_index, CFG, GEOM["horizon_m1"])
    tuned = json.load(open(TUNED_FP))["m1sc_ad"]
    halflife = spec.get("halflife", CFG["sample_halflife_days"])

    y_long = (lab["out_long"].to_numpy() == 1).astype(np.int8)
    y_short = (lab["out_short"].to_numpy() == 1).astype(np.int8)
    es, mr = CFG["early_stopping_rounds"], CFG["max_rounds"]

    ml_logs, base_logs_all, per_fold = [], [], []
    done = 0
    for i, fold in enumerate(folds):
        if max_folds is not None and i >= max_folds:
            break
        ck = os.path.join(CKPT, f"wf_{variant}_fold{i:02d}.pkl")
        if os.path.exists(ck):
            d = pickle.load(open(ck, "rb"))
            ml_logs.append(d["log"]); per_fold.append(d["stats"])
            for k, v in d.get("base_logs", {}).items():
                base_logs_all.append(v)
            print(f"[wf {variant}] fold {i} {fold['oos_month']} 已完成(checkpoint), 跳过")
            done += 1
            continue
        t0 = time.time()
        fit_r, val_r, oos_r = fold["fit_rows"], fold["val_rows"], fold["oos_rows"]
        w_fit = time_decay_weights(m1_index, fit_r, halflife)
        w_val = np.ones(len(val_r))
        Xf = F.iloc[fit_r].to_numpy(np.float32)
        Xv = F.iloc[val_r].to_numpy(np.float32)
        Xo = F.iloc[oos_r].to_numpy(np.float32)
        ex_stats = {"mode": spec["mode"], "halflife": halflife}
        if spec["mode"] == "ens":
            seeds = spec["seeds"]
            drop_below = 0.51
            valid_val = valid[val_r]
            probs, rounds = {}, {}
            for d, y in [("long", y_long), ("short", y_short)]:
                yv = y[val_r]
                members = []
                for seed in seeds:
                    for kind in ["lgb", "xgb"]:
                        params = dict(tuned[d][kind])
                        if kind == "lgb":
                            params.update(seed=seed, bagging_seed=seed,
                                          feature_fraction_seed=seed)
                        else:
                            params.update(seed=seed)
                        if kind == "lgb":
                            m, r = fit_lgb(Xf, y[fit_r], w_fit, params, Xv, yv, w_val, es, mr)
                            pv = _lgb_predict(m, Xv)
                        else:
                            m, r = fit_xgb(Xf, y[fit_r], w_fit, params, Xv, yv, w_val, es, mr)
                            pv = _xgb_predict(m, Xv)
                        auc = _auc(yv[valid_val], pv[valid_val])
                        members.append({"kind": kind, "seed": int(seed), "model": m,
                                        "rounds": int(r), "auc_val": auc})
                w, fb = ensemble_weights(members, drop_below)
                probs[(d, "val")] = predict_weighted(members, w, Xv)
                probs[(d, "oos")] = predict_weighted(members, w, Xo)
                rounds[d] = {f"{m['kind']}_s{m['seed']}": m["rounds"] for m in members}
                ex_stats.setdefault("directions", {})[d] = {
                    "fallback_uniform": bool(fb),
                    "n_kept": int((w > 0).sum()), "n_members": len(members),
                    "mean_auc_val": float(np.nanmean([m["auc_val"] for m in members])),
                    "members": [{"kind": m["kind"], "seed": m["seed"],
                                 "auc_val": (None if not np.isfinite(m["auc_val"]) else float(m["auc_val"])),
                                 "weight": float(wi)} for m, wi in zip(members, w)]}
            pl_val, ps_val = probs[("long", "val")], probs[("short", "val")]
            pl_oos, ps_oos = probs[("long", "oos")], probs[("short", "oos")]
        else:
            models, rounds = {}, {}
            for d, y in [("long", y_long), ("short", y_short)]:
                models[d], rounds[d] = _fit_pair(Xf, y[fit_r], w_fit, tuned[d],
                                                 Xv, y[val_r], w_val, es, mr)
            pl_val = predict_ens(models["long"], Xv)
            ps_val = predict_ens(models["short"], Xv)
            pl_oos = predict_ens(models["long"], Xo)
            ps_oos = predict_ens(models["short"], Xo)

        thr, thr_info = calibrate_threshold_m1_ad(pl_val, ps_val, val_r, lab, valid,
                                                  m1_pack, CFG)
        thr_l, thr_s = thr
        sig = build_signals(pl_oos, ps_oos, thr_l, thr_s)
        log = run_backtest_m1(sig, oos_r, lab, valid, m1_pack, CFG,
                              probs=(pl_oos, ps_oos), tag=f"ml_{variant}")
        log["fold"] = fold["oos_month"]
        m_ml = metrics(log, f"ml_{variant}")

        base_logs = {}
        vv = valid[oos_r]
        base_logs["always_long"] = run_backtest_m1(
            np.where(vv, 1, 0).astype(np.int8), oos_r, lab, valid, m1_pack, CFG, tag="always_long")
        base_logs["always_short"] = run_backtest_m1(
            np.where(vv, -1, 0).astype(np.int8), oos_r, lab, valid, m1_pack, CFG, tag="always_short")
        rng = np.random.default_rng(CFG["random_seed"] * 1000 + i)
        coin = rng.integers(0, 2, size=len(oos_r))
        base_logs["random_coin"] = run_backtest_m1(
            np.where(vv, np.where(coin == 0, 1, -1), 0).astype(np.int8),
            oos_r, lab, valid, m1_pack, CFG, tag="random_coin")
        n_sig = int((sig != 0).sum())
        cand = np.where(vv)[0]
        if n_sig > 0 and len(cand) > 0:
            pick = rng.choice(cand, size=min(n_sig, len(cand)), replace=False)
            sig_rm = np.zeros(len(oos_r), dtype=np.int8)
            sig_rm[pick] = np.where(rng.integers(0, 2, size=len(pick)) == 0, 1, -1)
            base_logs["random_matched"] = run_backtest_m1(
                sig_rm, oos_r, lab, valid, m1_pack, CFG, tag="random_matched")
        else:
            base_logs["random_matched"] = log.iloc[0:0].copy()
        for k, v in base_logs.items():
            v["fold"] = fold["oos_month"]

        stats = {
            "fold": i, "oos_month": fold["oos_month"], "window": fold["window"],
            "train_months": fold["train_months"], "n_fit": len(fit_r),
            "n_oos": len(oos_r), "threshold": {"long": thr_l, "short": thr_s},
            "thr_fallback": thr_info is None, "thr_info": thr_info,
            "rounds": rounds,
            "auc_val_long": float(roc_auc_score(y_long[val_r][valid[val_r]], pl_val[valid[val_r]])) if valid[val_r].sum() > 50 else np.nan,
            "auc_val_short": float(roc_auc_score(y_short[val_r][valid[val_r]], ps_val[valid[val_r]])) if valid[val_r].sum() > 50 else np.nan,
            "auc_oos_long": float(roc_auc_score(y_long[oos_r][vv], pl_oos[vv])) if vv.sum() > 50 else np.nan,
            "auc_oos_short": float(roc_auc_score(y_short[oos_r][vv], ps_oos[vv])) if vv.sum() > 50 else np.nan,
            "base_long": float(y_long[oos_r][vv].mean()),
            "base_short": float(y_short[oos_r][vv].mean()),
            "ml": m_ml, "baselines": {k: metrics(v, k) for k, v in base_logs.items()},
            "n_signals": n_sig, "extra": ex_stats,
        }
        for k, v in base_logs.items():
            v["variant"] = k
            base_logs_all.append(v)
        ml_logs.append(log)
        per_fold.append(stats)
        pickle.dump({"stats": stats, "log": log, "base_logs": base_logs},
                    open(ck, "wb"), protocol=4)
        done += 1
        print(f"[wf {variant}] fold {i:02d} {fold['oos_month']} 窗{fold['window']}"
              f"({fold['train_months']}月) thr(L/S)={thr_l:.3f}/{thr_s:.3f}"
              f"{'(回退)' if thr_info is None else ''} 交易{m_ml['trades']}笔 "
              f"胜率{(m_ml['win_rate']*100 if m_ml['trades'] else float('nan')):.1f}% "
              f"PnL${m_ml['total_pnl']:.1f} AUC(L/S)={stats['auc_oos_long']:.3f}/"
              f"{stats['auc_oos_short']:.3f} ({time.time()-t0:.0f}s)")

    if done:
        trades = pd.concat(ml_logs, ignore_index=True)
        trades.to_csv(os.path.join(RES, f"trades_{variant}.csv"), index=False)
        pd.DataFrame(per_fold).to_json(os.path.join(RES, f"per_fold_{variant}.json"),
                                       orient="records", indent=2)
        m_all = metrics(trades, f"ml_{variant}")
        json.dump({"ml": m_all, "geometry": GEOM, "mode": spec["mode"],
                   "halflife": halflife, "n_folds_done": done},
                  open(os.path.join(RES, f"summary_{variant}.json"), "w"), indent=2)
        print(f"\n===== {variant} 已完成{done}折汇总 (真实OOS) =====")
        for k, v in m_all.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prep", "tune", "wf"], required=True)
    ap.add_argument("--variant", default="m1sc_ad")
    ap.add_argument("--csv", default=RAW_CSV_DEFAULT)
    ap.add_argument("--max-folds", type=int, default=None)
    args = ap.parse_args()
    if args.stage == "prep":
        stage_prep(args.csv)
    elif args.stage == "tune":
        stage_tune(args.csv)
    else:
        stage_wf(args.variant, args.max_folds, args.csv)
    print("\n完成。")
