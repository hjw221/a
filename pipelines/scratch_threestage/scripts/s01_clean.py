#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S01 — 从零构建 | 原始M1数据加载 + 统计/ML辅助清洗 + 质量体检
================================================================
输入: /home/z/my-project/upload/5_extracted/XAUUSDc_M1_202201022305_202606262057.csv
输出: artifacts/m1_clean.pkl, reports/quality_report.json

清洗方法(全部为统计规则, 无人为修改):
  1) 时间戳解析/排序/去重
  2) OHLC 逻辑约束修复: high<->low 不一致时按 max/min 重排
  3) 孤立尖刺检测(稳健统计): close 偏离邻域中值超过 max(10*MAD, $1.2) 且 下一根回归 -> 判定为
     数据毛刺(非行情), 整根bar替换为邻域中值; 真跳变(下一根不回来)保留
  4) 缺口分析: 结构性缺口(每日维护/周末) vs 异常缺口; <=120min 用静止bar填充(标记)
  5) 输出诊断mask: is_filled / is_spike_fixed / is_ohlc_fixed
"""
import json
import numpy as np
import pandas as pd

BASE = '/home/z/my-project/download/xauusd_ml_scratch'
SRC = '/home/z/my-project/upload/5_extracted/XAUUSDc_M1_202201022305_202606262057.csv'

np.random.seed(42)

# ---------------------------------------------------------------- 1. 加载
print('[1] 加载原始CSV ...')
raw = pd.read_csv(SRC, sep='\t')
raw.columns = [c.strip('<>') for c in raw.columns]
raw['dt'] = pd.to_datetime(raw['DATE'] + ' ' + raw['TIME'], format='%Y.%m.%d %H:%M:%S')
raw = raw.rename(columns={'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low',
                          'CLOSE': 'close', 'TICKVOL': 'tickvol', 'SPREAD': 'spread'})
raw = raw[['dt', 'open', 'high', 'low', 'close', 'tickvol', 'spread']]
n_raw = len(raw)
print(f'    原始行数: {n_raw:,}  范围: {raw.dt.min()} ~ {raw.dt.max()}')

# ---------------------------------------------------------------- 2. 去重排序
dup_mask = raw.duplicated(subset='dt', keep='first')
n_dup = int(dup_mask.sum())
raw = raw[~dup_mask].sort_values('dt').reset_index(drop=True)
print(f'[2] 重复时间戳: {n_dup} (已去重, 保留首条)')

# ---------------------------------------------------------------- 3. OHLC 逻辑修复
hi = raw[['open', 'high', 'low', 'close']].max(axis=1)
lo = raw[['open', 'high', 'low', 'close']].min(axis=1)
bad_ohlc = (raw.high < raw.low) | (raw.high < hi - 1e-9) | (raw.low > lo + 1e-9)
n_ohlc = int(bad_ohlc.sum())
raw.loc[bad_ohlc, 'high'] = hi[bad_ohlc]
raw.loc[bad_ohlc, 'low'] = lo[bad_ohlc]
print(f'[3] OHLC逻辑违例修复: {n_ohlc}')

# ---------------------------------------------------------------- 4. 孤立尖刺检测(稳健统计)
c = raw.close.values
med = pd.Series(c).rolling(7, center=True, min_periods=5).median().values
dev = c - med
mad = pd.Series(np.abs(c - med)).rolling(7, center=True, min_periods=5).median().values
thr = np.maximum(10.0 * mad, 1.2)
is_out = np.abs(dev) > thr
n_out = int(is_out.sum())
# 孤立性: 下一根 close 回归到中值 1/3 阈值内
nxt_back = np.zeros(len(c), dtype=bool)
if n_out > 0:
    idx = np.where(is_out)[0]
    nxt_back[idx] = (idx + 1 < len(c)) & (np.abs(c[np.minimum(idx + 1, len(c) - 1)] - med[idx]) < thr[idx] / 3.0)
spike_mask = is_out & nxt_back
n_spike = int(spike_mask.sum())
# 修复: 整根bar替换为邻域中值
if n_spike > 0:
    fix_idx = np.where(spike_mask)[0]
    for i in fix_idx:
        raw.loc[i, ['open', 'high', 'low', 'close']] = med[i]
print(f'[4] 显著离群bar: {n_out}, 其中孤立尖刺(数据毛刺, 已修复): {n_spike}, '
      f'真跳变(保留): {n_out - n_spike}')

# ---------------------------------------------------------------- 5. 缺口分析
dt = raw.dt.values
gap_min = np.diff(dt) / np.timedelta64(1, 'm')
gaps = gap_min[gap_min > 1]
gap_pos = np.where(gap_min > 1)[0]
n_gaps = len(gaps)
print(f'[5] 缺口: {n_gaps} 个; 长度分布: 2-5min={int(((gaps>=2)&(gaps<=5)).sum())}, '
      f'6-60min={int(((gaps>5)&(gaps<=60)).sum())}, 61-120min={int(((gaps>60)&(gaps<=120)).sum())}, '
      f'>120min={int((gaps>120).sum())}')
# 缺口起始时刻的小时分布(定位每日维护窗口)
gap_start_hour = pd.Series(dt[gap_pos + 1]).dt.hour
gap_hour_top = gap_start_hour.value_counts().head(5).to_dict()
print(f'    缺口后首根bar的小时分布(top5): { {int(k): int(v) for k, v in gap_hour_top.items()} }')

# 静止bar填充 <=120min 的缺口
fill_rows = []
for pos, gl in zip(gap_pos, gaps):
    if 2 <= gl <= 120:
        start = dt[pos] + np.timedelta64(1, 'm')
        for k in range(int(gl) - 1):
            fill_rows.append((start + np.timedelta64(k, 'm'),
                              raw.close.iloc[pos], raw.close.iloc[pos],
                              raw.close.iloc[pos], raw.close.iloc[pos], 0, 0))
if fill_rows:
    fill_df = pd.DataFrame(fill_rows, columns=['dt', 'open', 'high', 'low', 'close', 'tickvol', 'spread'])
    raw = pd.concat([raw, fill_df], ignore_index=True).sort_values('dt').reset_index(drop=True)
    is_filled = np.zeros(len(raw), dtype=bool)
    is_filled[raw.tickvol.eq(0) & (raw.open == raw.close) & (raw.high == raw.low)] = True
else:
    is_filled = np.zeros(len(raw), dtype=bool)
n_filled = int(is_filled.sum())
print(f'    静止bar填充: +{n_filled} 根 (o=h=l=c=prev_close, vol=0)')

# 汇总mask
raw['is_filled'] = is_filled
raw['is_spike_fixed'] = False
if n_spike > 0:
    raw.loc[fix_idx, 'is_spike_fixed'] = True
raw['is_ohlc_fixed'] = False
if n_ohlc > 0:
    raw.loc[bad_ohlc.values, 'is_ohlc_fixed'] = True

# ---------------------------------------------------------------- 6. 统计报告
c = raw.close.values
ret1 = np.diff(c) / c[:-1]
year = pd.Series(raw.dt).dt.year
rep = {
    'rows_raw': n_raw, 'rows_clean': int(len(raw)), 'n_dup': n_dup,
    'n_ohlc_fixed': n_ohlc, 'n_outlier': n_out, 'n_spike_fixed': n_spike,
    'n_gaps': int(n_gaps), 'n_filled': n_filled,
    'gap_hour_top': {str(k): int(v) for k, v in gap_hour_top.items()},
    'date_range': [str(raw.dt.min()), str(raw.dt.max())],
    'price': {'start': float(raw.close.iloc[0]), 'end': float(raw.close.iloc[-1]),
              'min': float(raw.close.min()), 'max': float(raw.close.max())},
    'ret1_abs': {'mean_bp': float(np.nanmean(np.abs(ret1)) * 1e4),
                 'p50_bp': float(np.nanpercentile(np.abs(ret1), 50) * 1e4),
                 'p99_bp': float(np.nanpercentile(np.abs(ret1), 99) * 1e4),
                 'p999_bp': float(np.nanpercentile(np.abs(ret1), 99.9) * 1e4)},
    'tickvol': {'zero_bars': int((raw.tickvol == 0).sum()),
                'mean': float(raw.tickvol.mean()), 'p95': float(raw.tickvol.quantile(0.95))},
    'spread_by_year': {str(y): {'mean': float(v), 'p95': float(raw.spread[year == y].quantile(0.95))}
                        for y, v in raw.groupby(year).spread.mean().items()},
    'bars_per_year': {str(y): int(v) for y, v in year.value_counts().sort_index().items()},
    'weekend_check': {'mon': int((pd.Series(raw.dt).dt.dayofweek == 0).sum()),
                      'sat': int((pd.Series(raw.dt).dt.dayofweek == 5).sum())},
}
os_flag = raw.close.isna().sum()

# ---------------------------------------------------------------- 7. 保存
raw = raw.set_index('dt')
for col in ['open', 'high', 'low', 'close']:
    raw[col] = raw[col].astype('float64')
raw['tickvol'] = raw['tickvol'].astype('int32')
raw['spread'] = raw['spread'].astype('int32')
for col in ['is_filled', 'is_spike_fixed', 'is_ohlc_fixed']:
    raw[col] = raw[col].astype('int8')
raw.to_pickle(f'{BASE}/artifacts/m1_clean.pkl')
with open(f'{BASE}/reports/quality_report.json', 'w') as f:
    json.dump(rep, f, indent=2, ensure_ascii=False)
print('\n=== 质量报告 ===')
print(json.dumps(rep, indent=2, ensure_ascii=False))
print('\n[OK] -> artifacts/m1_clean.pkl')
