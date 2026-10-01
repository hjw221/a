"""v4跨资产特征冒烟校验:
1) DXY与XAU M5时间轴对齐质量 (新鲜率/陈旧度分布/覆盖范围)
2) 特征数与NaN率
3) 因果性: 同一时间戳截断DXY与XAU后构建特征, 截断前各行必须与全量构建逐单元一致
   (任何不一致 = 有特征看到了未来数据, 必须修复)"""
import sys, os, time
sys.path.insert(0, "/home/z/my-project/download/xauusd_ml_v2")
import numpy as np
import pandas as pd
from data import load_data, load_dxy_m5
from features_v4 import build_features_v4
from config import CFG

m5, m1_pack, spread_cost, monthly = load_data(CFG)
m5d = load_dxy_m5(CFG)
assert m5d is not None, "DXY数据未加载"
n = len(m5)

print(f"\n===== 对齐质量 =====")
print(f"XAU M5: {n:,} 根  {m5.index[0]} ~ {m5.index[-1]}")
print(f"DXY M5: {len(m5d):,} 根  {m5d.index[0]} ~ {m5d.index[-1]}")
present = m5.index.isin(m5d.index)
print(f"XAU有bar而DXY同刻有bar: {present.mean()*100:.2f}%  (新鲜率)")
idx = np.arange(n, dtype=np.float64)
last = np.where(present, idx, np.nan)
stale = idx - pd.Series(last).ffill().to_numpy()
st = pd.Series(stale, index=m5.index).dropna()
print(f"陈旧度(根M5)分位: p50={st.quantile(.5):.0f} p95={st.quantile(.95):.0f} p99={st.quantile(.99):.0f} max={st.max():.0f}")
print(f"陈旧>288根(>24h)的XAU bar占比: {(st>288).mean()*100:.3f}%")
# 逐日覆盖对照
dx_days = set(m5d.index.normalize().unique())
xu_days = set(m5.index.normalize().unique())
print(f"交易日: XAU {len(xu_days)} vs DXY {len(dx_days)}; XAU有而DXY无: {len(xu_days-dx_days)}天; DXY有而XAU无: {len(dx_days-xu_days)}天")

print(f"\n===== 特征构建 =====")
t0 = time.time()
F, feats, atr = build_features_v4(m5, m5d, CFG["atr_window_m5"])
print(f"{F.shape[1]}特征 x {len(F):,}行, 耗时 {time.time()-t0:.1f}s")
nan_rate = F.isna().mean().sort_values(ascending=False)
print("NaN率最高8个:", {k: f"{v*100:.1f}%" for k, v in nan_rate.head(8).items()})

print(f"\n===== 因果性校验 (截断60% vs 全量) =====")
cut = m5.index[int(n * 0.60)]
m5_tr = m5.loc[:cut]
m5d_tr = m5d.loc[:cut]
F_tr, _, _ = build_features_v4(m5_tr, m5d_tr, CFG["atr_window_m5"])
common = F_tr.index
diff = (F.loc[common].to_numpy(np.float64) != F_tr.to_numpy(np.float64))
both_nan = F.loc[common].isna().to_numpy() & F_tr.isna().to_numpy()
bad = diff & ~both_nan
print(f"截断前 {len(common):,} 行 x {F.shape[1]} 特征, 不一致单元: {bad.sum()}  (含NaN对齐豁免)")
if bad.sum() > 0:
    r, c = np.unravel_index(np.argmax(bad), bad.shape)
    print(f"  首个不一致: 行{common[r]} 特征{feats[c]} 全量={F.iloc[r, c]} 截断={F_tr.iloc[r, c]}")
    sys.exit(1)
print("PASS: 截断与全量逐单元一致, 全部特征因果")
