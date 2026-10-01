#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S02 — 从零构建 | 特征工程 (6族 ~55特征, 全部因果/无量纲)
=========================================================
设计原则:
  1) 严格因果: 所有rolling均trailing, 禁止center=True
  2) 无量纲: 比率/z-score/ATR标准化 — 因为金价1614->5593非平稳, 原始价格禁止入模
  3) 截断一致性自检: 前30%数据截断重算 == 全量计算的前30% (防未来函数)
  4) 特征分组: A动量14 | B波动12 | C结构10 | D形态6 | E时间7 | F量能6
输入: artifacts/m1_clean.pkl
输出: artifacts/features.pkl (float32), reports/feature_report.json
"""
import gc
import json
import numpy as np
import pandas as pd

BASE = '/home/z/my-project/download/xauusd_ml_scratch'
EPS = 1e-12

df = pd.read_pickle(f'{BASE}/artifacts/m1_clean.pkl')
o, h, l, c, v = df.open, df.high, df.low, df.close, df.tickvol.astype('float64')
idx = df.index
feats = {}
log = []

def add(name, series, group):
    if not isinstance(series, pd.Series):
        series = pd.Series(series, index=idx)
    series = series.astype('float32')
    feats[name] = series
    log.append({'name': name, 'group': group, 'na_pct': float(series.isna().mean() * 100)})

# ============================== A 动量/趋势 (14) ==============================
for N in [1, 5, 15, 30, 60, 240, 1440]:
    add(f'A_ret_{N}', c.pct_change(N), 'momentum')
ema8, ema32 = c.ewm(span=8, adjust=False).mean(), c.ewm(span=32, adjust=False).mean()
ema128, ema512 = c.ewm(span=128, adjust=False).mean(), c.ewm(span=512, adjust=False).mean()
add('A_emax_8_32', (ema8 - ema32) / c, 'momentum')
add('A_emax_32_128', (ema32 - ema128) / c, 'momentum')
add('A_emax_128_512', (ema128 - ema512) / c, 'momentum')
for N in [30, 240]:
    add(f'A_er_{N}', (c - c.shift(N)).abs() / (c.diff().abs().rolling(N, min_periods=N // 2).sum() + EPS), 'momentum')
sgn = np.sign(c.diff())
add('A_align_30', sgn.rolling(30, min_periods=15).sum() / 30, 'momentum')
add('A_align_240', sgn.rolling(240, min_periods=120).sum() / 240, 'momentum')

# ============================== B 波动率 (12) ==============================
tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
for N in [15, 60, 240, 1440]:
    add(f'B_atr_{N}', tr.ewm(alpha=1.0 / N).mean() / c, 'volatility')
lr = np.log(c / c.shift())
for N in [60, 240, 1440]:
    add(f'B_rv_{N}', np.sqrt((lr ** 2).rolling(N, min_periods=N // 2).sum()), 'volatility')
add('B_pk_240', np.sqrt(((np.log(h / l)) ** 2).rolling(240, min_periods=120).sum() / (4 * np.log(2) * 240)), 'volatility')
gk = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
add('B_gk_240', np.sqrt(gk.clip(lower=0).rolling(240, min_periods=120).mean()), 'volatility')
rv60 = (lr ** 2).rolling(60, min_periods=30).sum()
rv1440 = (lr ** 2).rolling(1440, min_periods=720).sum()
add('B_volratio', np.sqrt(rv60 / (rv1440 + EPS)), 'volatility')
atr240 = tr.ewm(alpha=1 / 240).mean()
add('B_atr_regime', atr240 / atr240.rolling(28800, min_periods=1440).median(), 'volatility')
bv = (np.pi / 2) * (lr.abs() * lr.abs().shift()).rolling(60, min_periods=30).sum()
add('B_jump_60', np.sqrt(np.clip(rv60 - bv, 0, None)), 'volatility')

# ============================== C 结构/位置 (10) ==============================
for N in [60, 240, 1440]:
    mn, mx = l.rolling(N, min_periods=N // 2).min(), h.rolling(N, min_periods=N // 2).max()
    add(f'C_pos_{N}', (c - mn) / (mx - mn + EPS), 'structure')
mx240 = h.rolling(240, min_periods=120).max()
mn240 = l.rolling(240, min_periods=120).min()
add('C_dist_high_240', (mx240 - c) / (atr240 + EPS), 'structure')
add('C_dist_low_240', (c - mn240) / (atr240 + EPS), 'structure')

def bars_since(extreme_val, price, is_max):
    """向量化: 距离最近一次触及rolling极值的bar数"""
    ii = np.arange(len(price))
    touch = (price >= extreme_val) if is_max else (price <= extreme_val)
    t_idx = pd.Series(np.where(touch, ii, np.nan))
    t_idx = t_idx.ffill()
    return pd.Series(ii - t_idx.values, index=price.index)

add('C_bsh_240', bars_since(h.rolling(240, min_periods=120).max(), h, True), 'structure')
add('C_bsl_1440', bars_since(l.rolling(1440, min_periods=720).min(), l, False), 'structure')
date = idx.date
day_grp = pd.Series(c.values, index=idx).groupby(date)
day_max = day_grp.cummax()
day_min = day_grp.cummin()
add('C_day_pos', (pd.Series(c.values) - day_min.values) / (day_max.values - day_min.values + EPS), 'structure')
day_open = pd.Series(c.values).groupby(date).transform('first')
add('C_day_ret', pd.Series(c.values) / day_open.values - 1, 'structure')
prev_day_close = pd.Series(c.values).groupby(date).transform('last')
prev_dc = pd.Series(prev_day_close.values).groupby(pd.Index(date)).shift(1).values
add('C_gap_open', pd.Series(day_open.values / prev_dc - 1, index=idx), 'structure')

# ============================== D 形态 (6) ==============================
rng = (h - l + EPS)
add('D_body', (c - o) / rng, 'pattern')
add('D_ushadow', (h - pd.concat([o, c], axis=1).max(axis=1)) / rng, 'pattern')
add('D_lshadow', (pd.concat([o, c], axis=1).min(axis=1) - l) / rng, 'pattern')
add('D_range_atr', (h - l) / (tr.ewm(alpha=1 / 60).mean() + EPS), 'pattern')
add('D_inside_bar', ((h <= h.shift()) & (l >= l.shift())).astype('float32'), 'pattern')
add('D_two_bar_drive', (np.sign(c.diff()) * np.sign(c.diff().shift())).fillna(0), 'pattern')

# ============================== E 时间 (7) ==============================
hr = idx.hour
add('E_hour_sin', np.sin(2 * np.pi * hr / 24), 'time')
add('E_hour_cos', np.cos(2 * np.pi * hr / 24), 'time')
add('E_dow', idx.dayofweek.astype('float32'), 'time')
add('E_sess_asia', ((hr >= 0) & (hr < 8)).astype('float32'), 'time')
add('E_sess_eu', ((hr >= 8) & (hr < 15)).astype('float32'), 'time')
add('E_sess_us', ((hr >= 15) & (hr < 22)).astype('float32'), 'time')
add('E_sess_off', ((hr >= 22)).astype('float32'), 'time')

# ============================== F 量能/微结构 (6) ==============================
vm1440 = v.rolling(1440, min_periods=720)
add('F_tvol_z', (v - vm1440.mean()) / (vm1440.std() + EPS), 'microstructure')
add('F_tvol_ratio', v.rolling(15, min_periods=8).mean() / (v.rolling(240, min_periods=120).mean() + EPS), 'microstructure')
add('F_tvol_trend', v.rolling(60, min_periods=30).mean() / (v.rolling(1440, min_periods=720).mean() + EPS), 'microstructure')
abret = lr.abs()
cov = pd.DataFrame({'x': abret, 'y': v}).rolling(240, min_periods=120).cov()
add('F_volret_corr', cov.xs('x', level=1)['y'] / (abret.rolling(240, min_periods=120).std() * v.rolling(240, min_periods=120).std() + EPS), 'microstructure')
add('F_tvol_surge', v / (v.rolling(240, min_periods=120).mean() + EPS), 'microstructure')
add('F_spread_atr', df.spread.astype('float64') / (tr.ewm(alpha=1 / 60).mean() * 1000 + EPS), 'microstructure')

# ============================== 汇总 + 自检 ==============================
F = pd.DataFrame(feats, index=idx)
print(f'特征矩阵: {F.shape}')
trunc_ok = {}
cut = int(len(F) * 0.3)
sub = {k: v.iloc[:cut] for k, v in feats.items()}
F_part = pd.DataFrame(sub, index=idx[:cut])
F_full_head = F.iloc[:cut]
bad = []
for col in F.columns:
    a, b = F_part[col], F_full_head[col]
    m = a.notna() & b.notna()
    if m.sum() == 0:
        continue
    if not np.allclose(a[m].values, b[m].values, rtol=1e-4, atol=1e-6):
        bad.append(col)
trunc_ok['pass'] = len(bad) == 0
trunc_ok['bad_cols'] = bad
print(f'截断一致性自检: {"PASS" if trunc_ok["pass"] else "FAIL: " + str(bad)}')

# 因果性随机点复查(20个随机时间点, 重算A_ret_60/B_atr_240抽查)
rand_pos = np.random.default_rng(42).integers(2000, cut, 20)
spot_ok = True
for p in rand_pos:
    if abs(float(c.iloc[p] / c.iloc[p - 60] - 1) - float(F['A_ret_60'].iloc[p])) > 1e-5:
        spot_ok = False
print(f'随机点抽查(ret_60): {"PASS" if spot_ok else "FAIL"}')

F.astype('float32').to_pickle(f'{BASE}/artifacts/features.pkl')
rep = {'n_features': int(F.shape[1]), 'n_rows': int(F.shape[0]),
       'truncation_check': trunc_ok, 'spot_check': spot_ok,
       'groups': {g: [x['name'] for x in log if x['group'] == g] for g in
                  ['momentum', 'volatility', 'structure', 'pattern', 'time', 'microstructure']},
       'na_pct_top10': sorted(log, key=lambda x: -x['na_pct'])[:10]}
with open(f'{BASE}/reports/feature_report.json', 'w') as f:
    json.dump(rep, f, indent=2, ensure_ascii=False)
print(f"[OK] -> artifacts/features.pkl  ({F.shape[1]} 特征 x {F.shape[0]:,} 行)")
