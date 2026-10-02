#!/usr/bin/env python3
"""v15_ctrl2.py — 随机时刻表对照: 吊灯利润需要模型时刻表吗?

流: random = 每折随机抽 K 个 (K=模型信号数) hours_filter 时段内位置, 方向随机 ±1
    (与 model 流同引擎同串行结算). 若 random ≈ model -> 连时刻表都不需要, 纯出场几何.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import stage_c_loop as sc
from v15_exit import simulate_exit, EXIT_CHANDELIER, EXIT_SCALEOUT

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results_v15")
CHAMP = json.load(open(os.path.join(BASE, "results_research", "champion.json")))
CFG = CHAMP["cfg"]
GEOMS = {
    "chandelier": dict(mode=EXIT_CHANDELIER, fixed_tp=0.0, fixed_sl=0.0, chan_atr_mult=3.0,
                       n_bracket=0, bracket_tp_mult=0.0, time_hold=0, tight_tp=0.0, tight_sl=0.0, scale_trail=0.75),
    "scale_out":  dict(mode=EXIT_SCALEOUT,  fixed_tp=0.0, fixed_sl=0.0, chan_atr_mult=3.0,
                       n_bracket=0, bracket_tp_mult=0.0, time_hold=0, tight_tp=0.0, tight_sl=0.0, scale_trail=0.75),
}

def run_geom(geom_name, seed=777):
    p = GEOMS[geom_name]
    rng = np.random.default_rng(seed)
    m5, F, cost, atr, (m1_t, m1_o, m1_h, m1_l) = sc._prep()
    X = F[CFG["features"]]
    hours_f = CFG.get("hours_filter")
    hour_arr = m5.index.hour.to_numpy()
    params = dict(CFG["lgb"]); params["num_threads"] = int(os.environ.get("ML_THREADS", "26"))
    sig_t = (m5.index.astype("datetime64[s]").astype("int64") + 300).to_numpy()
    folds = sc.month_folds(m5.index, "2022-08", "2026-07")
    TRAIN_CAP = pd.Timedelta(days=CFG.get("train_cap_days", 450))
    q_gate = CFG.get("q_gate", 0.0)
    streams = {"model": [], "random": []}
    fills = {"model": 0, "random": 0}
    for oos_s, oos_e in folds:
        tr_end = oos_s - pd.Timedelta(minutes=120)
        tr_start = oos_s - TRAIN_CAP
        tr_mask = (m5.index >= tr_start) & (m5.index < tr_end)
        oos_mask = (m5.index >= oos_s) & (m5.index <= oos_e)
        Xtr, ytr = X[tr_mask].to_numpy(), sc._cache["y"][tr_mask]
        ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr)
        Xtr, ytr = Xtr[ok], ytr[ok]
        Xoos = X[oos_mask].to_numpy()
        ok_o = np.isfinite(Xoos).all(axis=1)
        if len(Xtr) < 5000 or ok_o.sum() < 100:
            continue
        bst = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=CFG.get("rounds", 200))
        pv = np.full(len(Xoos), np.nan)
        pv[ok_o] = bst.predict(Xoos[ok_o])
        trade = np.zeros(len(Xoos), dtype=np.int8)
        if q_gate > 0:
            fin = np.isfinite(pv)
            if fin.sum() > 50:
                hi, lo = np.quantile(pv[fin], [1 - q_gate, q_gate])
                trade[pv >= hi] = 1
                trade[pv <= lo] = -1
        if hours_f:
            hm = hour_arr[oos_mask]
            trade[~np.isin(hm, hours_f)] = 0
        # random 流: 同 K, 随机位置 (hours 内), 随机方向
        K = int((trade != 0).sum())
        rnd = np.zeros(len(Xoos), dtype=np.int8)
        if hours_f:
            cand = np.where(np.isin(hour_arr[oos_mask], hours_f) & ok_o)[0]
        else:
            cand = np.where(ok_o)[0]
        if K > 0 and len(cand) > 0:
            pick = rng.choice(cand, size=min(K, len(cand)), replace=False)
            rnd[pick] = rng.choice([1, -1], size=len(pick))
        for name, tm in (("model", trade), ("random", rnd)):
            pnl, wins, filled = simulate_exit(
                m1_t, m1_o, m1_h, m1_l, sig_t[oos_mask], tm,
                atr[oos_mask], cost[oos_mask],
                p["mode"], float(CFG["tp_mult"]), float(CFG["sl_mult"]), 0.48, 90, 10,
                p["fixed_tp"], p["fixed_sl"], p["chan_atr_mult"],
                p["n_bracket"], p["bracket_tp_mult"],
                p["time_hold"], p["tight_tp"], p["tight_sl"], p["scale_trail"], 1)
            streams[name].append({"month": str(oos_s.date())[:7], "pnl": float(pnl[tm != 0].sum())})
            fills[name] += filled
    out = {}
    for name, rows in streams.items():
        s = pd.Series({r["month"]: r["pnl"] for r in rows})
        by = {}
        for m, v in s.items():
            y = int(m[:4]); by[y] = by.get(y, 0.0) + float(v)
        out[name] = dict(total=round(float(s.sum()), 1), trades=fills[name],
                         avg_per_trade=round(float(s.sum()) / max(fills[name], 1), 3),
                         by_year={k: round(v, 1) for k, v in sorted(by.items())})
    return out

if __name__ == "__main__":
    res_path = os.path.join(RES, "controls2.json")
    res = {}
    if os.path.exists(res_path):
        try: res = json.load(open(res_path))
        except Exception: res = {}
    for g in sys.argv[1:] or list(GEOMS.keys()):
        res[g] = run_geom(g)
        json.dump(res, open(res_path, "w"), indent=1)
        m, r = res[g]["model"], res[g]["random"]
        print(f"[ctrl2:{g}] model={m['total']:+.0f} random={r['total']:+.0f}", flush=True)
        print(f"    model  {m['by_year']}", flush=True)
        print(f"    random {r['by_year']}", flush=True)
        sc.say(f"v15ctrl2 {g}: model {m['total']:.0f} vs random-schedule {r['total']:.0f}")
    print(f"DONE -> {res_path}", flush=True)
