#!/usr/bin/env python3
"""diag_nan.py — 诊断因子在数据末期的 nan 问题"""
import os
os.environ["ML_THREADS"] = "8"
os.chdir("/root/rivermind-data/research-lab")
import numpy as np
import stage_c_loop as sc

m5, F, cost, atr, _ = sc.load_all()
print("M5 范围:", m5.index.min(), "->", m5.index.max())
months = m5.index.to_period("M")
# 每月每因子的 nan 率
for m in ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2025-12"]:
    mask = months == m
    if mask.sum() == 0:
        print(m, "无数据"); continue
    sub = F[mask]
    nanrate = sub.isna().mean()
    bad = nanrate[nanrate > 0.5]
    print(f"{m}: {mask.sum()} 行 | 高nan因子: {dict(bad.round(2)) if len(bad) else '无'} | 平均nan率 {nanrate.mean():.2f}")
# 看 m5 本体
for m in ["2026-04", "2026-05", "2026-06"]:
    mask = m5.index.strftime("%Y-%m") == m
    if mask.sum():
        print(m, "m5行:", mask.sum(), "| close nan:", m5["close"][mask].isna().sum(),
              "| spread_cost nan:", m5["spread_cost"][mask].isna().sum(),
              "| tickvol 均值:", m5["tickvol"][mask].mean())
# 原始 df 末期
df_raw = m5
print("\n末期 20 行:")
print(m5.tail(3))
