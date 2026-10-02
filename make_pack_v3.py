"""make_pack_v3.py — 修复 m1_t 微秒域bug, 重生成干净 pack.

根因: pandas 2.x DatetimeIndex dtype=datetime64[us], run_m1.load_m1_pack 的
astype(int64)//1e9//60 把微秒当纳秒 -> m1_t 每~17分钟才+1 -> entry 二分跳~160bar
-> _simulate 一笔锁仓吞整段信号 (四臂372笔 vs 历史6496笔的17倍缩水根因, 也是
trades CSV 1970-01-20 时间戳来源).
修复: m1_t 强制 ns 域计算. 重生成 pack_m1sc_ad_v3.pkl (同 GEOM 原几何).
"""
import os, sys, time, pickle
import numpy as np
import pandas as pd

BASE_V2 = "/root/rivermind-data/xauusd/v2_ens"
CSV = "/root/rivermind-fs/xauusd/data/XAUUSDc_M1_202201022305_202606262057.csv"
sys.path.insert(0, BASE_V2)
from data import load_raw_m1, impute_spread
from run_m1 import make_labels_m1_ad, GEOM
from features_s1 import build_features_s1
from numba import njit

T0 = time.time()
def log(s): print(f"[P3 {time.time()-T0:6.0f}s] {s}", flush=True)

OUT = os.path.join(BASE_V2, "cache", "pack_m1sc_ad_v3.pkl")

log("加载 M1 ...")
m1 = load_raw_m1(CSV)
log(f"M1 {len(m1):,} 行, index dtype={m1.index.dtype}")

# ★ 修复: 强制 ns 域 -> 正确绝对分钟
m1_t = (m1.index.astype("datetime64[ns]").astype("int64") // 10**9 // 60)\
    .to_numpy(np.int64)
d = np.diff(m1_t)
log(f"m1_t 域验证: diff==1 占比 {(d==1).mean()*100:.2f}% (应≈99%+), "
    f"首值 {m1_t[0]:,} (应≈2.7千万级)")
assert m1_t[0] > 20_000_000, "m1_t 域仍是坏的!"

m1_o = m1["OPEN"].to_numpy(np.float64)
m1_h = m1["HIGH"].to_numpy(np.float64)
m1_l = m1["LOW"].to_numpy(np.float64)
m1_c = m1["CLOSE"].to_numpy(np.float64)
m1_pack = (m1_t, m1_o, m1_h, m1_l, m1_c)

spread_cost, monthly = impute_spread(m1, 0.001)
log("构建 s1 特征 (34) ...")
F, feats, _ = build_features_s1(m1)
log("重标标签 (原 GEOM: TP=max(1.2,1.5xATR288), SL=max(0.4TP,2x点差), H=90) ...")
lab, valid = make_labels_m1_ad(m1_pack, spread_cost, GEOM)

ei = lab["entry_idx"].to_numpy()
seg = ei[valid]
dd = np.diff(seg)
log(f"entry 验证: entry[i+1]-entry[i]==1 占比 {(dd==1).mean()*100:.2f}% "
    f"(修复前≈0.2%); 跳变分位 {np.percentile(dd, [50, 90, 99]).round(1)}")
pl = lab["pnl_long"].to_numpy()[valid]; ps = lab["pnl_short"].to_numpy()[valid]
tp = lab["tp_d"].to_numpy()[valid]
log(f"标签有效 {valid.mean()*100:.2f}% | long TP率 {(lab['out_long'][valid]==1).mean()*100:.1f}% "
    f"short {(lab['out_short'][valid]==1).mean()*100:.1f}% | dur分位 "
    f"{np.percentile(lab['exit_bar_long'].to_numpy()[valid]-seg, [50, 90]).round(1)}")
best = np.maximum(pl, ps)
log(f"可吃率(strong) {(best>=0.6*tp).mean()*100:.1f}% | 最优方向笔均 {best.mean():.3f}")

with open(OUT, "wb") as f:
    pickle.dump({"F": F, "feats": feats, "lab": lab, "valid": valid,
                 "spread_cost": spread_cost, "monthly_spread": monthly,
                 "geometry": GEOM, "m1_t_fixed": m1_t}, f, protocol=4)
log(f"-> {OUT}")
msg = f"pack_v3(修复m1_t us域bug) done {int(time.time()-T0)}s: entry逐bar占比{(dd==1).mean()*100:.1f}%"
log(msg)
os.system(f"curl -s -m 20 -d '{msg}' ntfy.sh/xauusd-qv7m2zk9-res > /dev/null 2>&1 &")
