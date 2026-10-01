#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S05 — 从零构建 | 障碍标注设计 (点差感知, 训练期内定几何)
==========================================================
[1] 补充因子: F_atr_level(z(B_atr_1440)), F_bsl(z(log1p(bars_since_low_1440)))
    (P1 null-importance gain 榜前二的派生) -> 合入因子池并补检验
[2] numba 三障碍模拟器: entry=下一根开盘, TP/SL=k*ATR_120(带绝对下限), 超时T=240min
    同bar双碰保守记SL
[3] 训练期网格扫描: a in {1.5,2,2.5,3} x b in {0.6,0.8,1,1.25}, long/short 分别
    事件采样: 每15根bar取1 (去相关)
[4] 几何选择规则(先写死, 防事后挑):
    - 过滤: 平均TP率 in [0.22, 0.42], 每边样本>=3万
    - 评分: |avg净EV| 最小 (随机口径最中性, 模型负责推正), tie-break 用更接近PLR 2.4
[5] 全期标签生成(选定几何, 事件采样15min) -> labels.pkl
成本: round-trip $0.06 (2 x $0.03)
输出: reports/geometry_report.json, artifacts/labels.pkl, artifacts/factors.pkl(更新)
"""
import json
import numpy as np
import pandas as pd
from numba import njit

BASE = '/home/z/my-project/download/xauusd_ml_scratch'
TRAIN_END = '2024-08-01'
COST_RT = 0.06
T_TIMEOUT = 240  # 分钟

df = pd.read_pickle(f'{BASE}/artifacts/m1_clean.pkl')
F = pd.read_pickle(f'{BASE}/artifacts/features.pkl')
o = df.open.values.astype(np.float64)
h = df.high.values.astype(np.float64)
l = df.low.values.astype(np.float64)
c = df.close.values.astype(np.float64)
atr120 = (pd.concat([(df.high - df.low), (df.high - df.close.shift()).abs(),
                     (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
          .ewm(alpha=1 / 120).mean()).values
# 障碍锚: 120分钟总波动量级 (不是单bar ATR!) = ATR_120 * sqrt(120)
SIGMA_SCALE = np.sqrt(120.0)
atr_anchor = atr120 * SIGMA_SCALE
n = len(df)

# ================================================================ [1] 补充因子
def zsc(s):
    lo, hi = s.quantile(0.001), s.quantile(0.999)
    return ((s.clip(lo, hi) - s.mean()) / (s.std() + 1e-12))

tr = (F.index < TRAIN_END)
F['F_atr_level'] = zsc(F['B_atr_1440'] * 1e4).astype('float32')
F['F_bsl'] = zsc(np.log1p(F['C_bsl_1440'].clip(lower=0))).astype('float32')
fact = pd.read_pickle(f'{BASE}/artifacts/factors.pkl')
fact['F_atr_level'] = F['F_atr_level']
fact['F_bsl'] = F['F_bsl']

# 补检验
fw120 = (df.close.shift(-120) / df.close - 1) * 1e4
supp_check = {}
for name in ['F_atr_level', 'F_bsl']:
    x = fact[name]
    m = (x.notna() & pd.Series(fw120).notna()).values
    ic = float(np.corrcoef(x[m].rank(), pd.Series(fw120).values[m].rank() if hasattr(pd.Series(fw120).values[m], 'rank') else pd.Series(fw120)[m].rank())[0, 1])
    yearly = []
    year_arr = np.asarray(fact.index.year)
    for yy in [2022, 2023, 2024]:
        my = m & (year_arr == yy) & np.asarray(tr)
        if my.sum() > 50000:
            yearly.append(round(float(np.corrcoef(x[my].rank(), pd.Series(fw120)[my].rank())[0, 1]), 4))
    supp_check[name] = {'ic120': round(ic, 4), 'icY120': yearly}
print('[1] 补充因子检验:', json.dumps(supp_check))

# ================================================================ [2] numba 三障碍
@njit(cache=True)
def triple_barrier(ev, o, h, l, c, atr, a_tp, b_sl, T, tp_floor, sl_floor, is_long):
    N = len(ev)
    out = np.full(N, -99, dtype=np.int8)     # 1=TP, -1=SL, 0=timeout
    pnl = np.full(N, np.nan)
    hold = np.full(N, np.nan)
    entry_arr = np.full(N, np.nan)
    for i in range(N):
        e = ev[i]
        if e + 1 >= len(c):
            continue
        entry = o[e + 1]
        tp_d = a_tp * atr[e]
        sl_d = b_sl * atr[e]
        if tp_d < tp_floor:
            tp_d = tp_floor
        if sl_d < sl_floor:
            sl_d = sl_floor
        if is_long:
            tp_p, sl_p = entry + tp_d, entry - sl_d
        else:
            tp_p, sl_p = entry - tp_d, entry + sl_d
        entry_arr[i] = entry
        res = 0
        p = 0.0
        hb = 0
        for k in range(e + 1, min(e + 1 + T, len(c))):
            hb = k - (e + 1)
            if is_long:
                hit_tp = h[k] >= tp_p
                hit_sl = l[k] <= sl_p
            else:
                hit_tp = l[k] <= tp_p
                hit_sl = h[k] >= sl_p
            if hit_tp and hit_sl:
                res, p = -1, -sl_d  # 保守: 同bar双碰记SL
                break
            if hit_sl:
                res, p = -1, -sl_d
                break
            if hit_tp:
                res, p = 1, tp_d
                break
        else:
            k = min(e + 1 + T, len(c)) - 1
            p = (c[k] - entry) if is_long else (entry - c[k])
            res = 0
        out[i] = res
        pnl[i] = p
        hold[i] = hb if hb > 0 else 1
    return out, pnl, hold, entry_arr

# ================================================================ [3] 网格扫描 (训练期)
ev_all = np.arange(0, n - 250, 15)
ev_train = ev_all[df.index.values[ev_all] < np.datetime64(TRAIN_END)]
print(f'[3] 训练期事件数: {len(ev_train):,} (每15根bar取1)')

grid = []
for a in [0.75, 1.0, 1.5, 2.0]:
    for b in [0.5, 0.75, 1.0, 1.5]:
        row = {'a': a, 'b': b}
        for side, is_long in [('long', True), ('short', False)]:
            out, pnl, hold, _ = triple_barrier(ev_train.astype(np.int64), o, h, l, c, atr_anchor,
                                               a, b, T_TIMEOUT, 1.20, 0.70, is_long)
            m = out != -99
            tp_r = float((out[m] == 1).mean())
            sl_r = float((out[m] == -1).mean())
            to_r = float((out[m] == 0).mean())
            net = float((pnl[m] - COST_RT).mean())
            win = pnl[m][out[m] == 1].mean() if (out[m] == 1).any() else 0
            los = abs(pnl[m][out[m] == -1].mean()) if (out[m] == -1).any() else 1
            row[side] = {'tp_rate': round(tp_r, 4), 'sl_rate': round(sl_r, 4), 'to_rate': round(to_r, 4),
                         'net_ev': round(net, 4), 'plr': round(win / (los + 1e-9), 3),
                         'avg_hold': round(float(hold[m].mean()), 1), 'n': int(m.sum())}
        grid.append(row)
    print(f'    a={a} done')

print('\n[3] 障碍网格 (随机口径, 训练期, TP/SL=k*sigma_120=ATR120*sqrt(120), T=240min, RT成本$0.06):')
print(f"{'a':>4}{'b':>6} | {'L_TP%':>6}{'L_EV':>8}{'L_PLR':>7} | {'S_TP%':>6}{'S_EV':>8}{'S_PLR':>7} | {'avgEV':>7}{'TP%':>6}")
for r in grid:
    L, S = r['long'], r['short']
    print(f"{r['a']:>4}{r['b']:>6} | {L['tp_rate']*100:>6.1f}{L['net_ev']:>+8.3f}{L['plr']:>7.2f} | "
          f"{S['tp_rate']*100:>6.1f}{S['net_ev']:>+8.3f}{S['plr']:>7.2f} | "
          f"{(L['net_ev']+S['net_ev'])/2:>+7.3f}{(L['tp_rate']+S['tp_rate'])/2*100:>6.1f}")

# ================================================================ [4] 几何选择 (规则写死)
cands = []
for r in grid:
    L, S = r['long'], r['short']
    avg_tp = (L['tp_rate'] + S['tp_rate']) / 2
    avg_ev = (L['net_ev'] + S['net_ev']) / 2
    if 0.22 <= avg_tp <= 0.42 and L['n'] >= 30000:
        cands.append((abs(avg_ev), -abs((L['plr'] + S['plr']) / 2 - 2.4), r))
cands.sort(key=lambda t: (t[0], t[1]))
sel = cands[0][2]
print(f"\n[4] 选定几何: a={sel['a']} (TP={sel['a']}x sigma_120), b={sel['b']} (SL={sel['b']}x sigma_120), T={T_TIMEOUT}min")
print(f"    sigma_120训练期中位: ${np.nanmedian(atr_anchor[df.index.values < np.datetime64(TRAIN_END)]):.2f}")
print(f"    long : {sel['long']}")
print(f"    short: {sel['short']}")

# ================================================================ [5] 全期标签
lab = []
for side, is_long in [('long', True), ('short', False)]:
    out, pnl, hold, entry = triple_barrier(ev_all.astype(np.int64), o, h, l, c, atr_anchor,
                                           sel['a'], sel['b'], T_TIMEOUT, 1.20, 0.70, is_long)
    lab.append(pd.DataFrame({'ev_idx': ev_all, 'side': side, 'outcome': out,
                             'pnl_gross': pnl, 'hold_bars': hold, 'entry': entry}))
labels = pd.concat(lab, ignore_index=True)
labels['dt'] = df.index.values[labels.ev_idx.values]
labels['pnl_net'] = labels.pnl_gross - COST_RT
labels['is_train'] = labels.dt < np.datetime64(TRAIN_END)
labels.to_pickle(f'{BASE}/artifacts/labels.pkl')

# 随机口径分段对照(几何来自训练期, OOS段是纯对照)
seg_rep = {}
for seg_name, m in [('train_2022_2024H1', labels.is_train), ('oos_2024H2_2026H2', ~labels.is_train)]:
    s = labels[m & (labels.outcome != -99)]
    seg_rep[seg_name] = {side: {'n': int(g.shape[0]),
                                'tp_rate': round(float((g.outcome == 1).mean()), 4),
                                'net_ev': round(float(g.pnl_net.mean()), 4),
                                'plr': round(float(g.pnl_gross[g.outcome == 1].mean() /
                                                   abs(g.pnl_gross[g.outcome == -1]).mean()), 3)}
                         for side, g in s.groupby('side')}
print('\n[5] 随机口径分段对照(模型未介入, 纯几何表现):')
print(json.dumps(seg_rep, indent=1))

fact.astype('float32').to_pickle(f'{BASE}/artifacts/factors.pkl')
rep = {'grid': grid, 'selected': {'a': sel['a'], 'b': sel['b'], 'T': T_TIMEOUT,
                                  'anchor': 'sigma_120 = ATR120(M1) * sqrt(120)',
                                  'tp_floor': 1.20, 'sl_floor': 0.70, 'cost_rt': COST_RT},
       'supp_factor_check': supp_check, 'segment_check': seg_rep,
       'n_events': int(len(ev_all))}
with open(f'{BASE}/reports/geometry_report.json', 'w') as f:
    json.dump(rep, f, indent=1, ensure_ascii=False)
print('\n[OK] -> artifacts/labels.pkl, artifacts/factors.pkl(13因子), reports/geometry_report.json')
