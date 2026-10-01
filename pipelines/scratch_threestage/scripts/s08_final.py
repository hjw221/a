#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S08 — 从零构建 | 冠军配置(V3)最终OOS审判
=========================================
配置(s07内部验证段选出, 未触碰OOS):
  - 标签: y = pnl_net > 0
  - 门槛: p > p0 + 0.06, p0=(SLd+cost)/(TPd+SLd) 每事件盈亏平衡概率
  - 时段gate: 服务器时间 8-21
  - 双方向LGBM, train=24个月滚动, purge=241bar, 串行执行(一次一仓)
  - 几何: TP=1.0*sigma120, SL=0.5*sigma120, T=240min, RT成本$0.06
OOS: 2024-08 ~ 2026-07 (24折, 与旧基准 v3bal_ens 同期)
输出: reports/final_report.json, artifacts/trades_final.csv
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from numba import njit

BASE = '/home/z/my-project/download/xauusd_ml_scratch'
COST = 0.06
DELTA = 0.06
GATE = (8, 21)
PURGE_BARS = 241
TRAIN_WIN = 24

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
lab['y'] = (lab.pnl_net > 0).astype(int)
lab['month'] = pd.PeriodIndex(lab.dt64, freq='M')
X_ev = F.loc[lab.dt].values
sig_ev = sig120[lab.ev_idx.values]
hour_ev = hour_arr[lab.ev_idx.values]
TPd = np.maximum(1.0 * sig_ev, 1.20)
SLd = np.maximum(0.5 * sig_ev, 0.70)
p0 = (SLd + COST) / (TPd + SLd)
thr_ev = p0 + DELTA

params = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=15,
              min_data_in_leaf=3000, feature_fraction=0.9, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=10.0, verbosity=-1, num_threads=2, seed=42)

oos_months = pd.period_range('2024-08', '2026-07', freq='M')
fold_stats = []
for i, om in enumerate(oos_months):
    test_m = (lab.month == om).values
    train_mask = ((lab.month >= om - TRAIN_WIN) & (lab.month <= om - 1)).values
    purge_cut = dt_index.searchsorted(pd.Timestamp(str(om)) - pd.Timedelta(minutes=PURGE_BARS))
    ev_exit = lab.ev_idx.values + 1 + lab.hold_bars.fillna(240).values
    tr_ok = train_mask & (ev_exit < purge_cut)
    preds = np.full(len(lab), np.nan)
    for side in ['long', 'short']:
        sm = (lab.side == side).values
        tr_i = np.where(tr_ok & sm)[0]
        te_i = np.where(test_m & sm)[0]
        if len(tr_i) < 3000 or len(te_i) == 0:
            continue
        dtr = lgb.Dataset(X_ev[tr_i], label=lab.y.values[tr_i])
        m = lgb.train(params, dtr, num_boost_round=200)
        preds[te_i] = m.predict(X_ev[te_i])
    lab.loc[test_m, 'p'] = preds[test_m]
    fold_stats.append({'month': str(om)})
    if (i + 1) % 6 == 0:
        print(f'  fold {i + 1}/24')

# ============ 串行执行 (numba): gate + 自适应门槛 + 择优方向 ============
@njit(cache=True)
def serial_exec(ev_idx, side_code, p, thr, hour, hold, pnl_gross, gate_lo, gate_hi, cost):
    open_until = -1
    n_ev = len(ev_idx)
    trade_ev = []
    trade_side = []
    trade_pnl = []
    trade_hold = []
    i = 0
    while i < n_ev:
        e = ev_idx[i]
        if e <= open_until or np.isnan(p[i]):
            i += 1
            continue
        best_j, best_score = -1, -1.0
        j = i
        while j < n_ev and ev_idx[j] == e:
            if hour[j] >= gate_lo and hour[j] < gate_hi and p[j] > thr[j]:
                sc = p[j] - thr[j]
                if sc > best_score:
                    best_score, best_j = sc, j
            j += 1
        if best_j >= 0:
            hd = hold[best_j]
            if np.isnan(hd):
                hd = 240.0
            trade_ev.append(int(e))
            trade_side.append(side_code[best_j])
            trade_pnl.append(pnl_gross[best_j] - cost)
            trade_hold.append(hd)
            open_until = int(e + 1 + hd)
            i = j
        else:
            i += 1
    return (np.array(trade_ev), np.array(trade_side),
            np.array(trade_pnl), np.array(trade_hold))

oos_mask = ((lab.dt64 >= np.datetime64('2024-08-01')) &
            (lab.dt64 < np.datetime64('2026-07-18')))
lab_o = lab[oos_mask].copy()
order = np.argsort(lab_o.ev_idx.values)
lab_o = lab_o.iloc[order].reset_index(drop=True)

te_v, ts_v, tp_v, th_v = serial_exec(
    lab_o.ev_idx.values.astype(np.int64),
    (lab_o.side == 'long').values.astype(np.int8),
    lab_o.p.values.astype(np.float64),
    thr_ev[oos_mask][order].astype(np.float64),
    hour_ev[oos_mask][order].astype(np.int64),
    lab_o.hold_bars.values.astype(np.float64),
    lab_o.pnl_gross.values.astype(np.float64),
    GATE[0], GATE[1], COST)

trades = pd.DataFrame({'ev_idx': te_v, 'is_long': ts_v, 'pnl_net': tp_v, 'hold': th_v})
trades['side'] = np.where(trades.is_long == 1, 'long', 'short')
trades['dt'] = dt_index.values[trades.ev_idx.values]
print(f'\nOOS交易数: {len(trades)}')

# ============ 指标 ============
def metrics(t):
    if len(t) == 0:
        return {}
    win, lose = t[t.pnl_net > 0], t[t.pnl_net <= 0]
    daily = t.set_index('dt').pnl_net.resample('D').sum()
    sh = daily.mean() / (daily.std() + 1e-9) * np.sqrt(252)
    eq = daily.cumsum()
    return {'n': int(len(t)), 'win_rate': round(len(win) / len(t), 4),
            'plr': round((win.pnl_net.mean() if len(win) else 0) /
                         abs(lose.pnl_net.mean() if len(lose) else 1), 3),
            'pnl': round(float(t.pnl_net.sum()), 2),
            'per_trade': round(float(t.pnl_net.mean()), 4),
            'sharpe_d': round(float(sh), 3),
            'maxdd': round(float((eq - eq.cummax()).min()), 2),
            'avg_hold_min': round(float(t.hold.mean()), 1),
            'long_n': int((t.side == 'long').sum()),
            'long_pnl': round(float(t[t.side == 'long'].pnl_net.sum()), 2),
            'short_n': int((t.side == 'short').sum()),
            'short_pnl': round(float(t[t.side == 'short'].pnl_net.sum()), 2)}

M = metrics(trades)
M_stress = metrics(trades.assign(pnl_net=trades.pnl_net - 0.04))  # 成本压力$0.10
print(json.dumps(M, indent=1, ensure_ascii=False))
print(f"压力口径($0.10): pnl={M_stress['pnl']} sharpe={M_stress['sharpe_d']}")

# 随机对照(同gate同频率, numba)
@njit(cache=True)
def random_control(ev_idx, pnl_gross, hold, hour, fire, n_trial, cost, gate_lo, gate_hi, seed):
    rng_state = seed
    out = np.zeros(n_trial)
    for t in range(n_trial):
        open_until = -1
        s = 0.0
        cnt = 0
        for i in range(len(ev_idx)):
            if ev_idx[i] <= open_until:
                continue
            if hour[i] < gate_lo or hour[i] >= gate_hi:
                continue
            rng_state = (rng_state * 1664525 + 1013904223) % 4294967296
            r = rng_state / 4294967296.0
            if r < fire:
                hd = hold[i]
                if np.isnan(hd):
                    hd = 240.0
                s += pnl_gross[i] - cost
                cnt += 1
                open_until = int(ev_idx[i] + 1 + hd)
        out[t] = s
    return out

ev_uniq = lab_o.drop_duplicates('ev_idx')
fire = len(trades) / max(len(ev_uniq[(ev_uniq.hour.astype(int) >= 8) &
                                     (ev_uniq.hour.astype(int) < 21)]) if 'hour' in ev_uniq else len(ev_uniq), 1)
rand = random_control(lab_o.ev_idx.values.astype(np.int64),
                      lab_o.pnl_gross.values.astype(np.float64),
                      lab_o.hold_bars.values.astype(np.float64),
                      hour_ev[oos_mask][order].astype(np.int64),
                      min(fire, 0.3), 300, COST, GATE[0], GATE[1], 999)
print(f'随机对照(gate内同频率300次): mean=${rand.mean():.1f} '
      f'CI=[{np.percentile(rand,5):.1f}, {np.percentile(rand,95):.1f}]')

rng = np.random.default_rng(7)
bs = [float(np.sum(rng.choice(trades.pnl_net.values, size=len(trades), replace=True)))
      for _ in range(1000)]
print(f'Bootstrap CI5-95: [{np.percentile(bs,5):.1f}, {np.percentile(bs,95):.1f}]')
mo = trades.set_index('dt').pnl_net.resample('ME').sum()
print(f'盈利月: {int((mo > 0).sum())}/{len(mo)}')

trades.to_csv(f'{BASE}/artifacts/trades_final.csv', index=False)
rep = {'config': {'variant': 'V3', 'label': 'pnl_net>0', 'gate': 'server-hour 8-21',
                  'delta': DELTA, 'geometry': 'TP=1.0*sig120 SL=0.5*sig120 T=240',
                  'cost_rt': COST},
       'main': M, 'stress_cost_0.10': M_stress,
       'random_control': {'mean': round(float(rand.mean()), 2),
                          'ci5': round(float(np.percentile(rand, 5)), 2),
                          'ci95': round(float(np.percentile(rand, 95)), 2)},
       'bootstrap_ci': [round(float(np.percentile(bs, 5)), 2), round(float(np.percentile(bs, 95)), 2)],
       'profit_months': f'{int((mo > 0).sum())}/{len(mo)}',
       'old_baseline_v3bal_ens': {'n': 1611, 'win_rate': 0.316, 'plr': 2.57,
                                   'pnl': 1113.5, 'sharpe': 1.74}}
with open(f'{BASE}/reports/final_report.json', 'w') as f:
    json.dump(rep, f, indent=1, ensure_ascii=False)
print('[OK] -> artifacts/trades_final.csv, reports/final_report.json')
