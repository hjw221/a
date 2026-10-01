"""
冒烟测试: features_v3ma10 (v3+MA10块) — 防泄露与 sanity 三件套。
1) 因果性: 全量构建 vs 截断60%构建, 重叠区逐单元对比必须一致 (任何未来信息泄漏都会露馅);
2) 手工对拍: 随机行上用独立实现的SMA10/连续计数复算3个新特征;
3) sanity: NaN率/方差 + 与既有最近特征的相关性 (共线度如实报告)。
"""
import os, sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from config import CFG
from data import load_data
from features_v3ma10 import build_features_v3ma10

print("[smoke] 加载M5数据 (走缓存)...")
m5, m1_pack, spread_cost, monthly = load_data(CFG)
print(f"[smoke] {len(m5):,}根M5, {m5.index[0]} ~ {m5.index[-1]}")

print("[smoke] 全量构建37特征...")
F_full, feats, atr = build_features_v3ma10(m5, CFG["atr_window_m5"])
assert F_full.shape[1] == 37, f"特征数应为37, 实际{F_full.shape[1]}"
assert feats[-3:] == ["ma10_dist", "ma10_slope", "ma10_run"], feats[-3:]

# ---- 1) 因果性: 截断60% ----
cut = int(len(m5) * 0.60)
m5_tr = m5.iloc[:cut]
print(f"[smoke] 截断构建 (前{cut:,}行, {m5_tr.index[-1]})...")
F_tr, _, _ = build_features_v3ma10(m5_tr, CFG["atr_window_m5"])
A = F_full.iloc[:cut]
B = F_tr
bad = ~np.isclose(A.to_numpy(np.float64), B.to_numpy(np.float64),
                  equal_nan=True, atol=1e-9, rtol=0)
n_bad = int(bad.sum())
print(f"[smoke] 因果性: 重叠{A.shape[0]:,}行x{A.shape[1]}特征, 不一致单元 = {n_bad}")
if n_bad != 0:
    cols = np.unique(np.where(bad)[1])
    print("  泄漏嫌疑列:", [feats[c] for c in cols]); sys.exit(1)

# ---- 2) 手工对拍 (独立实现, 随机200行) ----
c = m5["CLOSE"].astype(np.float64)
atr_f = atr.astype(np.float64)
rng = np.random.default_rng(7)
idx = rng.choice(np.arange(300, len(m5) - 10), size=200, replace=False)
ma10 = c.rolling(10, min_periods=10).mean()
err_d = err_s = err_r = 0
for i in idx:
    hand_ma = c.iloc[i - 9: i + 1].mean()
    hand_dist = (c.iloc[i] - hand_ma) / atr_f.iloc[i]
    if abs(hand_dist - F_full["ma10_dist"].iloc[i]) > 1e-6: err_d += 1
    hand_slope = (hand_ma - c.iloc[i - 15: i - 5].mean()) / atr_f.iloc[i]
    if abs(hand_slope - F_full["ma10_slope"].iloc[i]) > 1e-6: err_s += 1
    # run: 从i往回数同侧根数
    side = np.sign(c.iloc[i] - hand_ma)
    r = 0
    for k in range(i, i - 30, -1):
        if np.sign(c.iloc[k] - ma10.iloc[k]) == side and side != 0: r += 1
        else: break
    hand_run = np.clip(side * r, -10, 10) / 10.0
    # F为float32存储, k/10非二进制精确值, 舍入差可达~3e-8 => 容差1e-6(纯舍入, 非语义错)
    if abs(hand_run - F_full["ma10_run"].iloc[i]) > 1e-6: err_r += 1
print(f"[smoke] 手工对拍200行: ma10_dist错{err_d} ma10_slope错{err_s} ma10_run错{err_r}")
if err_d or err_s or err_r:
    sys.exit(1)

# ---- 3) sanity + 共线度 ----
print("\n[smoke] 新特征统计 (排除前288根暖机段):")
warm = F_full.iloc[288:]
for col in ["ma10_dist", "ma10_slope", "ma10_run"]:
    s = warm[col]
    print(f"  {col:12s} NaN率{s.isna().mean()*100:.2f}%  std={s.std():.4f}  "
          f"p5={s.quantile(.05):+.3f}  p50={s.quantile(.50):+.3f}  p95={s.quantile(.95):+.3f}")
print("\n[smoke] 与既有特征的Spearman相关 (共线度):")
for new in ["ma10_dist", "ma10_slope", "ma10_run"]:
    rs = []
    for old in ["momn_12", "momn_48", "c_dist_s4h", "run_len", "slope_96"]:
        rs.append((old, warm[new].corr(warm[old], method="spearman")))
    rs.sort(key=lambda t: -abs(t[1]))
    print(f"  {new:12s} <- " + ", ".join(f"{k}:{v:+.2f}" for k, v in rs))

print("\n[smoke] ===== PASS =====")
