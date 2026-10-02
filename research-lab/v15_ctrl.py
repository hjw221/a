#!/usr/bin/env python3
"""v15_ctrl.py — 方向对照实验: chandelier/scale_out 的巨额利润是 alpha 还是 beta?

问题: chandelier 串行 +$20.6K 且五年全正(含2024) — 黄金同期 $1800->$3400,
      追踪止损天然吃趋势 beta. 必须拆分: 方向价值 vs 时刻表价值 vs 纯 drift.

设计 (同一 champ 信号时刻表, 三路方向流, 各自独立串行结算):
  model : 模型方向 (q_gate top->long, bottom->short)      = v15 已报的数字
  long  : 全部强制做多 (时刻表保留, 方向盲)
  short : 全部强制做空 (时刻表保留, 方向盲)
  flip  : 模型方向取反 (若模型方向有信息, flip 应显著更差)
判读:
  model >> long 且 model >> short  -> 方向 alpha 真实
  model ≈ long >> short            -> 主要是做多 beta (方向无信息, 别高兴太早)
  flip ≈ model                     -> 方向无信息
出场几何: chandelier 3.0*ATR / scale_out / time_exit 30min 三种各跑一遍.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import stage_c_loop as sc
from v15_exit import simulate_exit, EXIT_CHANDELIER, EXIT_SCALEOUT, EXIT_TIME

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results_v15")
os.makedirs(RES, exist_ok=True)
CHAMP = json.load(open(os.path.join(BASE, "results_research", "champion.json")))
CFG = CHAMP["cfg"]

GEOMS = {
    "chandelier": dict(mode=EXIT_CHANDELIER, fixed_tp=0.0, fixed_sl=0.0, chan_atr_mult=3.0,
                       n_bracket=0, bracket_tp_mult=0.0, time_hold=0, tight_tp=0.0, tight_sl=0.0, scale_trail=0.75),
    "scale_out":  dict(mode=EXIT_SCALEOUT,  fixed_tp=0.0, fixed_sl=0.0, chan_atr_mult=3.0,
                       n_bracket=0, bracket_tp_mult=0.0, time_hold=0, tight_tp=0.0, tight_sl=0.0, scale_trail=0.75),
    "time_exit":  dict(mode=EXIT_TIME,      fixed_tp=0.0, fixed_sl=0.0, chan_atr_mult=3.0,
                       n_bracket=0, bracket_tp_mult=0.0, time_hold=30, tight_tp=0.0, tight_sl=0.0, scale_trail=0.75),
}

_orig_sim = sc.simulate


def run_geom(geom_name):
    """复刻 walkforward 逻辑 (与 stage_c_loop.walkforward 同协议), 每折三路方向结算."""
    p = GEOMS[geom_name]
    m5, F, cost, atr, (m1_t, m1_o, m1_h, m1_l) = sc._prep()
    feats = CFG["features"]
    X = F[feats]
    hours_f = CFG.get("hours_filter")
    hour_arr = m5.index.hour.to_numpy()
    tau = CFG["tau"]
    tp_mult = CFG["tp_mult"]; sl_mult = CFG["sl_mult"]
    params = dict(CFG["lgb"]); params["num_threads"] = int(os.environ.get("ML_THREADS", "26"))
    sig_t = (m5.index.astype("datetime64[s]").astype("int64") + 300).to_numpy()
    folds = sc.month_folds(m5.index, "2022-08", "2026-07")[::1]
    TRAIN_CAP = pd.Timedelta(days=CFG.get("train_cap_days", 450))
    q_gate = CFG.get("q_gate", 0.0)
    streams = {"model": [], "long": [], "short": [], "flip": []}
    fills = {"model": 0, "long": 0, "short": 0, "flip": 0}
    t0 = time.time()
    for oos_s, oos_e in folds:
        tr_end = oos_s - pd.Timedelta(minutes=90 + 30)
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
        ds = lgb.Dataset(Xtr, ytr)
        bst = lgb.train(params, ds, num_boost_round=CFG.get("rounds", 200))
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
        # ---- 三路方向流 (同一时刻表) ----
        masks = {
            "model": trade,
            "long": np.where(trade != 0, 1, 0).astype(np.int8),
            "short": np.where(trade != 0, -1, 0).astype(np.int8),
            "flip": (trade * -1).astype(np.int8),
        }
        for name, tm in masks.items():
            pnl, wins, filled = simulate_exit(
                m1_t, m1_o, m1_h, m1_l, sig_t[oos_mask], tm,
                atr[oos_mask], cost[oos_mask],
                p["mode"], tp_mult, sl_mult, 0.48, 90, 10,
                p["fixed_tp"], p["fixed_sl"], p["chan_atr_mult"],
                p["n_bracket"], p["bracket_tp_mult"],
                p["time_hold"], p["tight_tp"], p["tight_sl"], p["scale_trail"], 1)
            streams[name].append({"month": str(oos_s.date())[:7],
                                  "pnl": float(pnl[tm != 0].sum()),
                                  "trades": int(filled)})
            fills[name] += filled
    out = {}
    for name, rows in streams.items():
        s = pd.Series({r["month"]: r["pnl"] for r in rows})
        by = {}
        for m, v in s.items():
            y = int(m[:4]); by[y] = by.get(y, 0.0) + float(v)
        tot = float(s.sum())
        out[name] = dict(total=round(tot, 1), trades=fills[name],
                         avg_per_trade=round(tot / max(fills[name], 1), 3),
                         by_year={k: round(v, 1) for k, v in sorted(by.items())})
    out["_secs"] = round(time.time() - t0)
    return out


def main():
    res_path = os.path.join(RES, "controls.json")
    res = {}
    if os.path.exists(res_path):
        try:
            res = json.load(open(res_path))
        except Exception:
            res = {}
    for g in sys.argv[1:] or list(GEOMS.keys()):
        res[g] = run_geom(g)
        json.dump(res, open(res_path, "w"), indent=1)
        m = res[g]["model"]; lo = res[g]["long"]; sh = res[g]["short"]; fl = res[g]["flip"]
        sc.say(f"v15ctrl {g}: model {m['total']:.0f} vs long {lo['total']:.0f} vs short {sh['total']:.0f} vs flip {fl['total']:.0f}")
        print(f"[ctrl:{g}] model={m['total']:+.0f} long={lo['total']:+.0f} short={sh['total']:+.0f} flip={fl['total']:+.0f} ({res[g]['_secs']}s)", flush=True)
        for nm in ("model", "long", "short", "flip"):
            print(f"    {nm:6s} {res[g][nm]['by_year']}", flush=True)
    print(f"DONE -> {res_path}", flush=True)


if __name__ == "__main__":
    main()
