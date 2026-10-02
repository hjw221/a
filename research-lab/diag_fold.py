#!/usr/bin/env python3
"""diag_fold.py — 深挖 2026-04 折 0 笔之谜"""
import os
os.environ["ML_THREADS"] = "24"
os.chdir("/root/rivermind-data/research-lab")
import numpy as np
import stage_c_loop as sc
import lightgbm as lgb

m5, F, cost, atr, (m1_t, m1_o, m1_h, m1_l) = sc._prep()
cfg = None
import json
recs = []
for w in range(4):
    try:
        recs += [json.loads(l) for l in open(f"results_research/worker{w}_results.jsonl")]
    except Exception:
        pass
cfg = [r for r in recs if r["cfg_id"] == "fd22bb26ce"][0]["cfg"]
print("tau:", cfg["tau"], "tp:", cfg["tp_mult"], "hours:", cfg.get("hours_filter"))

X = F[cfg["features"]]
oos_s, oos_e = np.datetime64("2026-04-01"), np.datetime64("2026-04-30 23:59")
idx = m5.index
oos_mask = (idx >= oos_s) & (idx <= oos_e)
tr_end = np.datetime64("2026-04-01") - np.timedelta64(120, "m")
tr_mask = idx < tr_end
Xtr = X[tr_mask].to_numpy(); ytr = sc._cache["y"][tr_mask]
ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr)
print("训练样本:", ok.sum(), "| OOS bar:", oos_mask.sum())
ds = lgb.Dataset(Xtr[ok], ytr[ok])
bst = lgb.train({**cfg["lgb"], "num_threads": 24}, ds, num_boost_round=cfg.get("rounds", 200))
Xoos = X[oos_mask].to_numpy()
ok_o = np.isfinite(Xoos).all(axis=1)
p = np.full(len(Xoos), np.nan)
p[ok_o] = bst.predict(Xoos[ok_o])
print("p 分位:", np.nanpercentile(p, [5, 25, 50, 75, 95]).round(4))
tau = cfg["tau"]
n_long = (p > tau).sum(); n_short = (p < 1 - tau).sum()
print(f"tau={tau} 下: long {(p>tau).sum()} short {(p<1-tau).sum()}")
hours_f = cfg.get("hours_filter")
if hours_f:
    hm = idx[oos_mask].hour.to_numpy()
    in_h = np.isin(hm, hours_f)
    print("时段过滤保留:", in_h.mean().round(2), "| 过滤后 long:", ((p > tau) & in_h).sum())
# simulate 试跑
sig_t = ((idx.astype("int64") // 10**9) + 300).to_numpy()
trade = np.zeros(len(Xoos), dtype=np.int8)
trade[p > tau] = 1
trade[p < 1 - tau] = -1
pnl, wins = sc.simulate(m1_t, m1_o, m1_h, m1_l, sig_t[oos_mask], p, trade,
                        atr[oos_mask], cost[oos_mask],
                        cfg["tp_mult"], cfg["sl_mult"], 0.48, 2.0, 90, 10)
print("simulate 后: 有效笔", (trade != 0).sum(), "| pnl 总", pnl.sum().round(1))
