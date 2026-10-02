"""diag_sim.py — 解剖 _simulate 吞 bar 之谜: entry/exit 数值域 + 手动状态机."""
import os, sys, pickle
import numpy as np
import pandas as pd

BASE_V2 = "/root/rivermind-data/xauusd/v2_ens"
CSV = "/root/rivermind-fs/xauusd/data/XAUUSDc_M1_202201022305_202606262057.csv"
sys.path.insert(0, BASE_V2)
from scalp import build_folds_m1

pack = pickle.load(open(os.path.join(BASE_V2, "cache", "pack_m1sc_ad.pkl"), "rb"))
lab, valid = pack["lab"], pack["valid"]

dt = pd.read_csv(CSV, sep="\t", usecols=["<DATE>", "<TIME>"])
uniq = dt["<DATE>"].unique()
dmap = {d: pd.to_datetime(d, format="%Y.%m.%d") for d in uniq}
dates = dt["<DATE>"].map(dmap)
secs = dt["<TIME>"].str.slice(0, 2).astype(np.int32) * 3600 + \
       dt["<TIME>"].str.slice(3, 5).astype(np.int32) * 60
idx = pd.DatetimeIndex((dates + pd.to_timedelta(secs, unit="s")).sort_values().to_numpy())

CFG = {"first_oos_month": "2024-08", "last_oos_month": "2026-07",
       "max_train_months": 36, "scalp_embargo_m1": 30, "cooldown_m1": 10,
       "inner_val_frac": 0.20}
folds = build_folds_m1(idx, CFG, 90)
oosr = folds[0]["oos_rows"]
R0, R1 = int(oosr[0]), int(oosr[-1])
print(f"oos_rows: {R0}~{R1} n={len(oosr):,}")

ei = lab["entry_idx"].to_numpy()
xl = lab["exit_bar_long"].to_numpy()
xs = lab["exit_bar_short"].to_numpy()
seg_e, seg_x, seg_v = ei[oosr], xl[oosr], valid[oosr]
print("entry_idx 样本:", seg_e[:8])
print("exit_bar_long 样本:", seg_x[:8])
print("dur=exit-entry 分位:", np.percentile(seg_x - seg_e, [10, 50, 90, 99]).round(1))
print(f"entry∈[{seg_e.min()}, {seg_e.max()}] | oos行号∈[{R0}, {R1}]")
print("entry < R0 占比:", float((seg_e < R0).mean()))
print("exit > R1 占比:", float((seg_x > R1).mean()))
print("dur>90 占比:", float(((seg_x - seg_e) > 90).mean()))

n_free = -(2**62)
cnt = {"invalid": 0, "e<0": 0, "e<free": 0, "xb<e": 0, "deal": 0}
last_deal_k = -1
gaps = []
for k in range(len(oosr)):
    if not seg_v[k]:
        cnt["invalid"] += 1; continue
    e = int(seg_e[k])
    if e < 0:
        cnt["e<0"] += 1; continue
    if e < n_free:
        cnt["e<free"] += 1; continue
    xb = int(seg_x[k])
    if xb < e:
        cnt["xb<e"] += 1; continue
    cnt["deal"] += 1
    if last_deal_k >= 0:
        gaps.append(k - last_deal_k)
    last_deal_k = k
    n_free = xb + 1
print("手动状态机(cd=0):", cnt)
if gaps:
    print("成交间隔bar分位:", np.percentile(gaps, [10, 50, 90, 99]).round(0))
    print("=> 每笔平均锁仓bar:", np.mean([seg_x[kk] - seg_e[kk] + 1 for kk in
          [i for i in range(len(oosr)) if seg_v[i] and int(seg_e[i]) >= 0][:50]][:10]))
