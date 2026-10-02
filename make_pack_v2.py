"""make_pack_v2.py — 几何重标 pack (TP=1.8xATR, SL=0.6xTP, floor $1.2).

v3/v4 瓶颈: TP floor $1.2 主导 -> 笔均 $0.125; research 冠军几何 tp1.8/sl0.6 笔均更大。
重标: 同一 features_s1 + label_all_m1_ad(GEOM2) -> cache/pack_m1sc_ad_v2.pkl
用法: python3 make_pack_v2.py
"""
import os, sys, time, pickle
import numpy as np
import pandas as pd

BASE_V2 = "/root/rivermind-data/xauusd/v2_ens"
CSV = "/root/rivermind-fs/xauusd/data/XAUUSDc_M1_202201022305_202606262057.csv"
sys.path.insert(0, BASE_V2)
from data import load_raw_m1, impute_spread
from run_m1 import load_m1_pack, make_labels_m1_ad, GEOM
from features_s1 import build_features_s1

T0 = time.time()
def log(s): print(f"[P2 {time.time()-T0:6.0f}s] {s}", flush=True)

GEOM2 = {"tp_floor_usd": 1.20, "tp_atr_mult": 1.8, "atr_win_m1": 288,
         "sl_mult": 0.6, "sl_spread_mult": 2.0, "horizon_m1": 90, "entry_tol_min": 10}
OUT = os.path.join(BASE_V2, "cache", "pack_m1sc_ad_v2.pkl")

log("加载 M1 ...")
m1 = load_raw_m1(CSV)
log(f"M1 {len(m1):,} 行")
spread_cost, monthly = impute_spread(m1, 0.001)
log("构建 s1 特征 (34) ...")
F, feats, _ = build_features_s1(m1)
log(f"特征 {F.shape[1]} x {len(F):,}")
m1_pack = load_m1_pack(CSV)
log("重标标签 (TP=max(1.2, 1.8xATR288), SL=max(0.6TP, 2x点差), H=90) ...")
lab, valid = make_labels_m1_ad(m1_pack, spread_cost, GEOM2)
pl = lab["pnl_long"].to_numpy()[valid]; ps = lab["pnl_short"].to_numpy()[valid]
tp = lab["tp_d"].to_numpy()[valid]
best = np.maximum(pl, ps)
log(f"标签有效 {valid.mean()*100:.2f}% | long TP率 {(lab['out_long'][valid]==1).mean()*100:.1f}% "
    f"short {(lab['out_short'][valid]==1).mean()*100:.1f}% | tp中位={np.median(tp):.2f} "
    f"sl中位={np.median(lab['sl_d'].to_numpy()[valid]):.2f}")
eat = (best >= 0.6 * tp).mean()
log(f"可吃率(strong) {eat*100:.1f}% | 最优方向笔均 {best.mean():.3f}")

with open(OUT, "wb") as f:
    pickle.dump({"F": F, "feats": feats, "lab": lab, "valid": valid,
                 "spread_cost": spread_cost, "monthly_spread": monthly,
                 "geometry": GEOM2}, f, protocol=4)
log(f"-> {OUT}")
msg = f"pack_v2 done {int(time.time()-T0)}s: tp1.8/sl0.6 可吃率{eat*100:.1f}%"
os.system(f"curl -s -m 20 -d '{msg}' ntfy.sh/xauusd-qv7m2zk9-res > /dev/null 2>&1 &")
