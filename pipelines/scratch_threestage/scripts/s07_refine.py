#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S07 — 从零构建 | 防泄露改进: 训练期内部验证段(2024H1)比拼配置, 冠军进OOS
==========================================================================
铁律: 本脚本所有配置选择只看 <=2024-07 的数据; 2024-08 之后数据禁止参与任何选择
候选:
  V0: 基线(y=TP标签, 固定q85阈值)            [s06 原样]
  V1: V0 + 时段gate(服务器8-21点, E4证据)      [执行层规则]
  V2: y=pnl_net>0标签 + 几何自适应门槛 p>p0+delta [p0=(SLd+cost)/(TPd+SLd)]
  V3: V1+V2
流程: 内部伪WF(train=2022-01~2023-12, test=2024H1月度6折) -> 冠军配置
输出: reports/refine_report.json
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

BASE = '/home/z/my-project/download/xauusd_ml_scratch'
INNER_TEST = ('2024-01-01', '2024-07-01')   # 内部验证段
TRAIN_END = '2024-08-01'
PURGE_BARS = 241
COST = 0.06

df = pd.read_pickle(f'{BASE}/artifacts/m1_clean.pkl')
F = pd.read_pickle(f'{BASE}/artifacts/factors.pkl')
labels = pd.read_pickle(f'{BASE}/artifacts/labels.pkl')
dt_index = df.index
sig120 = ((pd.concat([(df.high - df.low), (df.high - df.close.shift()).abs(),
                      (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
           .ewm(alpha=1 / 120).mean()) * np.sqrt(120)).values
hour_arr = dt_index.hour.values

lab = labels[labels.outcome != -99].copy()
lab['dt64'] = pd.to_datetime(lab.dt).values
lab['pnl_net'] = lab.pnl_gross - COST
lab['y_tp'] = (lab.outcome == 1).astype(int)
lab['y_pnl'] = (lab.pnl_net > 0).astype(int)
lab['month'] = pd.PeriodIndex(lab.dt64, freq='M')
X_ev = F.loc[lab.dt].values
sig_ev = sig120[lab.ev_idx.values]
hour_ev = hour_arr[lab.ev_idx.values]
TPd = np.maximum(1.0 * sig_ev, 1.20)
SLd = np.maximum(0.5 * sig_ev, 0.70)
p0 = (SLd + COST) / (TPd + SLd)  # 盈亏平衡概率(每事件)

inner_test_months = pd.period_range('2024-01', '2024-06', freq='M')

params = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=15,
              min_data_in_leaf=3000, feature_fraction=0.9, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=10.0, verbosity=-1, num_threads=2, seed=42)

def run_innerWF(ycol, gate_hours, use_p0, delta=None, thr_q=0.85):
    """内部伪WF: train=2022-01~2023-12(24个月), test=2024H1; 返回测试段预测事件表"""
    out = []
    for om in inner_test_months:
        test_m = (lab.month == om).values
        train_mask = ((lab.month >= om - 24) & (lab.month <= om - 1)).values
        purge_cut = dt_index.searchsorted(pd.Timestamp(str(om)) - pd.Timedelta(minutes=PURGE_BARS))
        tr_ok = train_mask & (lab.ev_idx.values + 1 + lab.hold_bars.fillna(240).values < purge_cut)
        preds = np.full(len(lab), np.nan)
        for side in ['long', 'short']:
            sm = (lab.side == side).values
            tr_i = np.where(tr_ok & sm)[0]
            te_i = np.where(test_m & sm)[0]
            if len(tr_i) < 3000:
                continue
            y = lab[ycol].values
            dtr = lgb.Dataset(X_ev[tr_i], label=y[tr_i])
            m = lgb.train(params, dtr, num_boost_round=200)
            preds[te_i] = m.predict(X_ev[te_i])
        out.append(preds)
    P = np.full(len(lab), np.nan)
    for preds in out:
        m = ~np.isnan(preds)
        P[m] = preds[m]
    te_any = np.zeros(len(lab), dtype=bool)
    for om in inner_test_months:
        te_any |= (lab.month == om).values
    return te_any, P

def simulate(te_any, P, gate_hours, use_p0, delta, thr_q):
    """串行执行模拟器(事件级)"""
    sel = lab.copy()
    sel['p'] = P
    sel['p0'] = p0
    sel['hour'] = hour_ev
    sel = sel[te_any & sel.p.notna()].copy()
    # 阈值
    if use_p0:
        sel['thr'] = sel.p0 + delta
    else:
        thr = {'long': 0, 'short': 0}
        for side in ['long', 'short']:
            s = sel[sel.side == side]
            if len(s):
                thr[side] = float(np.quantile(s.p, thr_q))
        sel['thr'] = sel.apply(lambda r: thr[r.side], axis=1)
    if gate_hours:
        sel = sel[(sel.hour >= gate_hours[0]) & (sel.hour < gate_hours[1])]
    sel = sel.sort_values('dt64')
    open_until = -1
    trades = []
    last_ev = -7
    for _, r in sel.iterrows():
        if r.ev_idx <= open_until:
            continue
        if r.ev_idx == last_ev:
            continue
        if r.p > r.thr:
            trades.append({'side': r.side, 'pnl_net': r.pnl_gross - COST,
                           'hold': r.hold_bars, 'dt': r.dt64})
            open_until = int(r.ev_idx + 1 + (r.hold_bars if not np.isnan(r.hold_bars) else 240))
            last_ev = int(r.ev_idx)
    t = pd.DataFrame(trades)
    if len(t) == 0:
        return {'n': 0}
    daily = t.set_index('dt').pnl_net.resample('D').sum()
    sh = daily.mean() / (daily.std() + 1e-9) * np.sqrt(252)
    return {'n': int(len(t)),
            'win_rate': round(float((t.pnl_net > 0).mean()), 4),
            'pnl': round(float(t.pnl_net.sum()), 2),
            'per_trade': round(float(t.pnl_net.mean()), 4),
            'sharpe_d': round(float(sh), 3)}

variants = {}
# V0: 基线(y=TP, q85, 无gate)
te, P = run_innerWF('y_tp', None, False)
variants['V0_base'] = simulate(te, P, None, False, None, 0.85)
# V1: +时段gate
variants['V1_hourgate'] = simulate(te, P, (8, 21), False, None, 0.85)
# V2: y=pnl>0 + 几何自适应门槛(delta三档在内部验证段比较)
te2, P2 = run_innerWF('y_pnl', None, True)
for delta in [0.02, 0.04, 0.06]:
    variants[f'V2_p0delta{delta}'] = simulate(te2, P2, None, True, delta, 0.85)
# V3: V2 + gate
best_v2 = max([k for k in variants if k.startswith('V2_')],
              key=lambda k: variants[k].get('sharpe_d', -9))
delta_star = float(best_v2.split('delta')[1])
variants['V3_combo'] = simulate(te2, P2, (8, 21), True, delta_star, 0.85)

print('=== 内部验证段(2024H1)配置比拼 ===')
for k, v in variants.items():
    print(f'  {k:<18} n={v.get("n",0):>4} wr={v.get("win_rate",0):.3f} pnl={v.get("pnl",0):>+8.2f} '
          f'per={v.get("per_trade",0):>+7.3f} sharpe={v.get("sharpe_d",0):>6.2f}')

winner = max(variants.items(), key=lambda kv: (kv[1].get('sharpe_d', -9) + kv[1].get('per_trade', -9)))
print(f'\n冠军: {winner[0]}')
rep = {'variants': variants, 'winner': winner[0], 'delta_star': delta_star,
       'inner_test_period': '2024-01~2024-06'}
with open(f'{BASE}/reports/refine_report.json', 'w') as f:
    json.dump(rep, f, indent=1, ensure_ascii=False)
print('[OK] -> reports/refine_report.json')
