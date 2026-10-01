"""v3特征冒烟测试: 1) 无异常 2) 特征数 3) NaN率 4) 因果性(截断vs全量, 相同行必须一致)"""
import sys, time
sys.path.insert(0, "/home/z/my-project/download/xauusd_ml_v2")
import numpy as np
import pandas as pd
from data import load_data
from features_v3 import build_features_v3
from config import CFG

m5, m1_pack, spread_cost, monthly = load_data(CFG)
n = len(m5)
print(f"M5行数: {n:,}  范围 {m5.index[0]} ~ {m5.index[-1]}")

t0 = time.time()
F_full, feats, atr = build_features_v3(m5, CFG["atr_window_m5"])
t_full = time.time() - t0
print(f"全量构建: {F_full.shape[1]}特征 x {n:,}行, {t_full:.1f}s")

nan_frac = F_full.isna().mean().sort_values(ascending=False)
print("\nNaN率>1%的特征:")
print((nan_frac[nan_frac > 0.01] * 100).round(1).to_string())
print(f"\n整体NaN率(全表): {F_full.isna().mean().mean()*100:.2f}%")
inf_cnt = np.isinf(F_full.to_numpy(np.float64)).sum()
print(f"inf个数: {inf_cnt}")

# 因果性: 用前60%数据构建, 与全量构建在相同行上逐特征比对
cut = int(n * 0.6)
F_trunc, _, _ = build_features_v3(m5.iloc[:cut], CFG["atr_window_m5"])
common = 5000                      # 比对截断末尾5000行
a = F_trunc.iloc[-common:]
b = F_full.iloc[cut - common:cut]
diff = (a.to_numpy(np.float32) != b.to_numpy(np.float32)) & ~(np.isnan(a.to_numpy(np.float32)) & np.isnan(b.to_numpy(np.float32)))
print(f"\n因果性校验(截断 vs 全量, {common}行 x {F_full.shape[1]}特征): 不一致单元 = {diff.sum()}")
bad = [feats[j] for j in range(F_full.shape[1]) if diff[:, j].any()]
print(f"不一致特征: {bad if bad else '无 — 全部因果'}")

# 抽查几个特征的量纲合理性 (最后100行)
print("\n末尾100行抽样统计:")
print(F_full[["momn_96", "er_96", "squeeze", "dist_hi_288", "run_len", "vol_z_288", "spread_rel"]].tail(100).describe().round(3).to_string())
