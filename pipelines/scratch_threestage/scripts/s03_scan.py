#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S03 — 从零构建 | 可预测性扫描: "哪些数据有用、有波段能吃到"
================================================================
铁律: 只用训练期 2022-01 ~ 2024-07-31 (OOS 2024-08+ 留给最终审判)

三组证据:
  [E1] 方向可预测性: 每特征x horizon 的 spearman IC(fwd_ret)
       -> 分5个半年度段, 检验符号一致率/t值
  [E2] 幅度可预测性: 每特征x horizon 的 spearman IC(|fwd_ret|)
       -> 波段大小能否预知(波动率聚集)
  [E3] 分位桶收益: 每特征 x H=120 的5分位桶 fwd_ret(bp) + 单调性
  [E4] 波段分布: vol regime x session x 事件(量能突增/突破/跳变)
       -> |fwd_ret_120| 的 P50/P84 (bp), 回答"波段藏在哪"

输出: reports/scan_report.json
"""
import json
import numpy as np
import pandas as pd

BASE = '/home/z/my-project/download/xauusd_ml_scratch'
TRAIN_END = '2024-08-01'
HORIZONS = [15, 30, 60, 120, 240, 480]

df = pd.read_pickle(f'{BASE}/artifacts/m1_clean.pkl')
F = pd.read_pickle(f'{BASE}/artifacts/features.pkl')
c = df.close

tr_mask = (F.index < TRAIN_END)
print(f'训练期行数: {tr_mask.sum():,} / 总 {len(F):,}')

# 半年度段标签(训练期内)
seg = pd.Series(index=F.index, dtype='object')
d = F.index
seg[:] = None
seg_tr = pd.Series([f'{y}H{"1" if m <= 6 else "2"}' for y, m in zip(d.year, d.month)], index=d)
seg_tr = seg_tr[tr_mask]
SEG_ORDER = ['2022H1', '2022H2', '2023H1', '2023H2', '2024H1']

E1, E2 = {}, {}
for H in HORIZONS:
    fw = (c.shift(-H) / c - 1)[tr_mask]
    fw_abs = fw.abs()
    fw_ok = fw.notna()
    out1, out2 = {}, {}
    for col in F.columns:
        x = F[col][tr_mask]
        m = x.notna() & fw_ok
        xv, fv = x[m], fw[m]
        if m.sum() < 50000:
            continue
        # E1: IC of fwd_ret (分段+全期)
        xr = xv.rank()
        fr = fv.rank()
        ics = []
        for sg in SEG_ORDER:
            ms = seg_tr[m] == sg
            if ms.sum() > 20000:
                ics.append(float(np.corrcoef(xr[ms], fr[ms])[0, 1]))
        ics = np.array(ics)
        ic_all = float(np.corrcoef(xr, fr)[0, 1])
        t = float(np.mean(ics) / (np.std(ics) + 1e-12) * np.sqrt(len(ics))) if len(ics) > 2 else 0.0
        out1[col] = {'ic': ic_all, 'ic_segs': [round(v, 4) for v in ics],
                     'ic_mean': round(float(np.mean(ics)), 5), 't': round(t, 2),
                     'sign_consist': int(np.sum(np.sign(ics) == np.sign(np.mean(ics))))}
        # E2: IC of |fwd_ret|
        fab = fw_abs[m].rank()
        ic_abs = float(np.corrcoef(xr, fab)[0, 1])
        out2[col] = {'ic_abs': round(ic_abs, 5)}
    E1[H], E2[H] = out1, out2
    print(f'[E1/E2] H={H} done ({len(out1)} 特征)')

# E3: 分位桶 (H=120 代表)
H = 120
fw = (c.shift(-H) / c - 1)[tr_mask]
E3 = {}
for col in F.columns:
    x = F[col][tr_mask]
    m = x.notna() & fw.notna()
    if m.sum() < 50000:
        continue
    xv, fv = x[m], fw[m] * 1e4
    try:
        qb = pd.qcut(xv, 5, labels=False, duplicates='drop')
    except Exception:
        continue
    means = fv.groupby(qb).mean()
    if len(means) != 5:
        continue
    mono = float(np.corrcoef(np.arange(5), means.values)[0, 1])
    E3[col] = {'bucket_bp': [round(v, 3) for v in means.values],
               'spread_bp': round(float(means.iloc[-1] - means.iloc[0]), 3), 'mono': round(mono, 3)}
print(f'[E3] 分位桶 H=120 done ({len(E3)} 特征)')

# E4: 波段分布 |fwd_ret_120| (bp) 分条件
fw12 = (c / c.shift(-120) - 1)[tr_mask].abs() * 1e4
cond = pd.DataFrame(index=F.index)
vr = F['B_volratio']
cond['vol_low'] = vr < vr.quantile(0.33)
cond['vol_mid'] = (vr >= vr.quantile(0.33)) & (vr < vr.quantile(0.66))
cond['vol_high'] = vr >= vr.quantile(0.66)
cond['asia'] = F['E_sess_asia'] > 0
cond['eu'] = F['E_sess_eu'] > 0
cond['us'] = F['E_sess_us'] > 0
cond['tvol_surge'] = F['F_tvol_surge'] > 2.0
cond['breakout'] = F['C_pos_240'] > 0.98
cond['breakdown'] = F['C_pos_240'] < 0.02
cond['jump'] = F['B_jump_60'] > F['B_jump_60'].quantile(0.99)
cond['trend_up'] = F['A_er_240'] > 0.35
cond['trend_dn'] = F['A_er_240'] < -0.35
cond['all'] = pd.Series(True, index=F.index)
E4 = {}
for name, mask in cond.items():
    mm = mask[tr_mask] & fw12.notna()
    v = fw12[mm]
    E4[name] = {'n': int(mm.sum()), 'p50_bp': round(float(v.median()), 2),
                'p84_bp': round(float(v.quantile(0.84)), 2),
                'p97_bp': round(float(v.quantile(0.97)), 2)}
print('[E4] 波段分布 done')

# 汇总排名(E1: 按稳定IC排序)
rank_rows = []
for col in F.columns:
    best = None
    for H in HORIZONS:
        d1 = E1[H].get(col)
        if d1 is None:
            continue
        score = abs(d1['ic_mean']) * d1['sign_consist'] / 5.0
        if best is None or score > best['score']:
            best = {'H': H, 'score': score, **d1}
    if best and best['score'] > 0:
        a120 = E2[120].get(col, {}).get('ic_abs', 0)
        rank_rows.append({'feature': col, 'best_H': best['H'], 'ic': best['ic'],
                          'ic_mean': best['ic_mean'], 't': best['t'],
                          'sign_consist': best['sign_consist'],
                          'ic_abs_120': a120})
rank_rows.sort(key=lambda r: -abs(r['ic_mean']) * r['sign_consist'])

rep = {'train_rows': int(tr_mask.sum()), 'E1_ic': {str(k): v for k, v in E1.items()},
       'E2_ic_abs': {str(k): v for k, v in E2.items()}, 'E3_bucket': E3, 'E4_wave_dist': E4,
       'top_by_stability': rank_rows[:25]}
with open(f'{BASE}/reports/scan_report.json', 'w') as f:
    json.dump(rep, f, indent=1, ensure_ascii=False)

print('\n=== TOP15 稳定方向信号 (|ic_mean| x 符号一致率) ===')
for r in rank_rows[:15]:
    print(f"  {r['feature']:<22} H={r['best_H']:<4} ic={r['ic']:+.4f} ic_mean={r['ic_mean']:+.4f} "
          f"t={r['t']:+.1f} sign={r['sign_consist']}/5 ic_abs={r['ic_abs_120']:+.4f}")
print('\n=== |fwd_ret| 波段分布(bp, H=120) ===')
for k, v in E4.items():
    print(f"  {k:<12} n={v['n']:>9,} p50={v['p50_bp']:>7.1f} p84={v['p84_bp']:>7.1f} p97={v['p97_bp']:>7.1f}")
print('\n[OK] -> reports/scan_report.json')
