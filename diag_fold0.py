"""diag_fold0.py — 现场诊断 v6/v8 fold0 的概率分布与信号量黑洞."""
import os, sys, pickle, json
import numpy as np
import pandas as pd

BASE_V2 = "/root/rivermind-data/xauusd/v2_ens"
CSV = "/root/rivermind-fs/xauusd/data/XAUUSDc_M1_202201022305_202606262057.csv"
sys.path.insert(0, BASE_V2)
sys.path.insert(0, "/root/rivermind-data/xauusd/stage_c")
from scalp import build_folds_m1, run_backtest_m1
from backtest import metrics
from models import fit_lgb

with open(os.path.join(BASE_V2, "cache", "pack_m1sc_ad.pkl"), "rb") as f:
    pack = pickle.load(f)
F, feats = pack["F"], pack["feats"]
lab, valid = pack["lab"], pack["valid"]
tuned = json.load(open(os.path.join(BASE_V2, "results", "tuned_params_m1.json")))["m1sc_ad"]
LGB_P = dict(tuned["long"]["lgb"])

dt = pd.read_csv(CSV, sep="\t", usecols=["<DATE>", "<TIME>"])
uniq = dt["<DATE>"].unique()
dmap = {d: pd.to_datetime(d, format="%Y.%m.%d") for d in uniq}
dates = dt["<DATE>"].map(dmap)
secs = dt["<TIME>"].str.slice(0, 2).astype(np.int32) * 3600 + \
       dt["<TIME>"].str.slice(3, 5).astype(np.int32) * 60
idx = pd.DatetimeIndex((dates + pd.to_timedelta(secs, unit="s")).sort_values().to_numpy())

# 重建 v6 的 64 特征 (白名单+因子) — 直接 import v6 模块太重, 手动最小集: 只用白名单+8个top因子
WL = ["range_ratio", "squeeze", "slope_1440", "session_ny", "hour_sin", "atr_ratio",
      "dist_hi_1440", "vol_ratio_12_1440", "c_dist_d1", "er_1440", "dist_lo_1440",
      "momn_1440", "vol_z_1440", "dow_sin", "dow_cos", "pos_in_range", "tr_over_atr",
      "hour_cos"]
X = F[WL].to_numpy(np.float32)   # 子集足以看概率分布形态

pl = lab["pnl_long"].to_numpy(np.float64)
ps = lab["pnl_short"].to_numpy(np.float64)
tp = lab["tp_d"].to_numpy(np.float64)
y_eat = ((np.maximum(pl, ps) >= 0.6 * tp) & valid).astype(np.int8)
y_dir = ((pl >= ps) & valid).astype(np.int8)

CFG = {"first_oos_month": "2024-08", "last_oos_month": "2026-07",
       "max_train_months": 36, "scalp_embargo_m1": 30, "cooldown_m1": 10,
       "inner_val_frac": 0.20}
folds = build_folds_m1(idx, CFG, 90)
f0 = folds[0]
fitr, valr, oosr = f0["fit_rows"], f0["val_rows"], f0["oos_rows"]
print(f"fold0: fit={len(fitr):,} val={len(valr):,} oos={len(oosr):,}")

def decay_w(rows, hl=180.0):
    t = np.asarray(idx[rows].asi8, dtype=np.float64) / 1e9 / 86400.0
    return np.exp(-np.log(2.0) * np.maximum(t[-1] - t, 0) / hl)

wtr, wva = decay_w(fitr), decay_w(valr)
mA, _ = fit_lgb(X[fitr], y_eat[fitr], wtr, dict(LGB_P, seed=42),
                X[valr], y_eat[valr], wva, es_rounds=50, max_rounds=600)
mB, _ = fit_lgb(X[fitr], y_dir[fitr], wtr, dict(LGB_P, seed=42),
                X[valr], y_dir[valr], wva, es_rounds=50, max_rounds=600)
pa_v, pb_v = mA.predict(X[valr]), mB.predict(X[valr])
pa_o, pb_o = mA.predict(X[oosr]), mB.predict(X[oosr])

for name, pa, pb in [("VAL", pa_v, pb_v), ("OOS", pa_o, pb_o)]:
    print(f"[{name}] p_eat 分位: {np.percentile(pa, [1,25,50,75,99]).round(4)}")
    print(f"[{name}] p_dir 分位: {np.percentile(pb, [1,25,50,75,99]).round(4)} "
          f"| p_dir>0.5 占比 {(pb>0.5).mean()*100:.1f}% | |p_dir-0.5|>0.02 占比 "
          f"{(np.abs(pb-0.5)>0.02).mean()*100:.1f}%")

# K=1.0/td=0 全 bar 信号直接喂 _simulate
sig = np.ones(len(oosr), dtype=np.int8)   # 全 long
raw = pd.read_csv(CSV, sep="\t", usecols=["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>"])
_dt = raw["<DATE>"].map(dmap)
_ss = raw["<TIME>"].str.slice(0, 2).astype(np.int32) * 3600 + \
      raw["<TIME>"].str.slice(3, 5).astype(np.int32) * 60
_ts = _dt + pd.to_timedelta(_ss, unit="s")
_order = np.argsort(_ts.to_numpy(), kind="stable")
m1_pack = (np.asarray(idx.asi8 // 10**9 // 60, dtype=np.int64),
           raw["<OPEN>"].to_numpy(np.float64)[_order],
           raw["<HIGH>"].to_numpy(np.float64)[_order],
           raw["<LOW>"].to_numpy(np.float64)[_order],
           raw["<CLOSE>"].to_numpy(np.float64)[_order])
for cd in [0, 10]:
    cfg_c = dict(CFG); cfg_c["cooldown_m1"] = cd
    lg = run_backtest_m1(sig, oosr, lab, valid, m1_pack, cfg_c)
    mm = metrics(lg)
    print(f"全bar强制long cd={cd}: n={len(lg)} pnl={lg['pnl'].sum():+.1f} "
          f"wr={mm.get('win_rate')*100 if mm.get('win_rate') else 0:.1f}%")
print("结论: 若全bar只有~36笔 -> _simulate吞bar; 若上千 -> v6/v8信号生成有bug")
