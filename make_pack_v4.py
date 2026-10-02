"""make_pack_v4.py — 点差杀手几何: TP=2.5xATR floor $2, SL=0.5xTP.

v9 教训: 4.3万笔 sc$0.195×n=$8400 点差绞肉机(毛利+6596被吃78%).
TP=$1.2 时 sc/TP=16%; TP=$2.5+ 后降到 ~8%.
用法: python3 make_pack_v4.py
"""
import os, sys, time, pickle
import numpy as np
import pandas as pd

BASE_V2 = "/root/rivermind-data/xauusd/v2_ens"
CSV = "/root/rivermind-fs/xauusd/data/XAUUSDc_M1_202201022305_202606262057.csv"
sys.path.insert(0, BASE_V2)
from data import load_raw_m1, impute_spread
from run_m1 import make_labels_m1_ad
from features_s1 import build_features_s1

T0 = time.time()
def log(s): print(f"[P4 {time.time()-T0:6.0f}s] {s}", flush=True)

GEOM4 = {"tp_floor_usd": 2.00, "tp_atr_mult": 2.5, "atr_win_m1": 288,
         "sl_mult": 0.5, "sl_spread_mult": 2.0, "horizon_m1": 120, "entry_tol_min": 10}
OUT = os.path.join(BASE_V2, "cache", "pack_m1sc_ad_v4.pkl")

log("加载 M1 ...")
m1 = load_raw_m1(CSV)
m1_t = (m1.index.astype("datetime64[ns]").astype("int64") // 10**9 // 60)\
    .to_numpy(np.int64)
m1_pack = (m1_t, m1["OPEN"].to_numpy(np.float64), m1["HIGH"].to_numpy(np.float64),
           m1["LOW"].to_numpy(np.float64), m1["CLOSE"].to_numpy(np.float64))
spread_cost, monthly = impute_spread(m1, 0.001)
log("构建 s1 特征 ...")
F, feats, _ = build_features_s1(m1)
log("重标标签 (TP=max(2.0, 2.5xATR288), SL=0.5TP, H=120) ...")
lab, valid = make_labels_m1_ad(m1_pack, spread_cost, GEOM4)
pl = lab["pnl_long"].to_numpy()[valid]; ps = lab["pnl_short"].to_numpy()[valid]
tp = lab["tp_d"].to_numpy()[valid]; sl = lab["sl_d"].to_numpy()[valid]
best = np.maximum(pl, ps)
sc_v = spread_cost[valid]
log(f"有效 {valid.mean()*100:.2f}% | TP率 long {(lab['out_long'][valid]==1).mean()*100:.1f}% "
    f"short {(lab['out_short'][valid]==1).mean()*100:.1f}%")
log(f"tp中位={np.median(tp):.2f} sl中位={np.median(sl):.2f} | sc/tp中位="
    f"{np.median(sc_v)/np.median(tp)*100:.1f}% | dur中位="
    f"{np.median(lab['exit_bar_long'].to_numpy()[valid]-lab['entry_idx'].to_numpy()[valid]):.0f}")
log(f"可吃率(strong 0.6tp) {(best>=0.6*tp).mean()*100:.1f}% | 最优方向笔均 {best.mean():.3f}")
with open(OUT, "wb") as f:
    pickle.dump({"F": F, "feats": feats, "lab": lab, "valid": valid,
                 "spread_cost": spread_cost, "monthly_spread": monthly,
                 "geometry": GEOM4, "m1_t_fixed": m1_t}, f, protocol=4)
log(f"-> {OUT}")
msg = f"pack_v4(点差杀手几何 tp2.5x/slfloor2) done {int(time.time()-T0)}s"
os.system(f"curl -s -m 20 -d '{msg}' ntfy.sh/xauusd-qv7m2zk9-res > /dev/null 2>&1 &")
