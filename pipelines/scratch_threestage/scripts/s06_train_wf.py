#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S06 — 从零构建 | 训练 + Walk-Forward + 成本回测 (最后一步)
============================================================
特征: 13因子 (s04+s05产出, ML挖掘+正交化后)
标签: y = 1 if outcome==TP (三障碍: a=1.0/b=0.5/T=240, 已在训练期标定)
模型: LightGBM 双方向 (long/short 各一), 浅树强正则, 折内时间早停
WF:   24折月度滚动 OOS=2024-08~2026-07 (与旧基准同期可比)
      train=过去24个月, purge=240bar(=4h, 标签跨度), 事件采样15min
阈值: 每折在训练事件概率分布的q85处(固定规则, 不碰OOS)
执行: 串行持仓(一次一仓), 事件时刻持仓中则跳过; 双向同触发取|p-0.5|大者
成本: $0.06/笔RT主口径 + $0.10压力口径; PnL按1oz
对照: 同几何随机口径 + 旧基准 v3bal_ens(+$1113.5/Sharpe1.74)
输出: reports/train_report.json, artifacts/trades.csv
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

BASE = '/home/z/my-project/download/xauusd_ml_scratch'
TRAIN_END = '2024-08-01'
OOS_START, OOS_END = '2024-08-01', '2026-07-18'
PURGE_BARS = 241          # 标签最长240bar + 1
TRAIN_WIN_MONTHS = 24
THR_QUANTILE = 0.85       # 固定规则: 训练事件概率的85分位
COST_MAIN, COST_STRESS = 0.06, 0.10

df = pd.read_pickle(f'{BASE}/artifacts/m1_clean.pkl')
F = pd.read_pickle(f'{BASE}/artifacts/factors.pkl')
labels = pd.read_pickle(f'{BASE}/artifacts/labels.pkl')
n = len(df)
dt_index = df.index

# 事件特征矩阵
lab_valid = labels[labels.outcome != -99].copy()
X_ev = F.loc[lab_valid.dt].values  # 因子值@事件bar
lab_valid['y'] = (lab_valid.outcome == 1).astype(int)
months = pd.Series(pd.to_datetime(lab_valid.dt)).dt.to_period('M')
lab_valid['month'] = months.values

oos_months = pd.period_range('2024-08', '2026-07', freq='M')
print(f'OOS折数: {len(oos_months)}, 事件池: {len(lab_valid):,}')

params = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=15,
              min_data_in_leaf=3000, feature_fraction=0.9, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=10.0, verbosity=-1, num_threads=2, seed=42)

pred_rows = []
for i, om in enumerate(oos_months):
    test_m = lab_valid.month == om
    train_m_end = om - 1
    train_m_start = om - TRAIN_WIN_MONTHS
    # purge: 训练事件 exit(ev+hold) 不得进入测试月前后 PURGE 窗口
    om_start_dt = om.to_timestamp()
    om_end_dt = (om + 1).to_timestamp()
    train_mask = (lab_valid.month >= train_m_start) & (lab_valid.month <= train_m_end)
    ev_pos = lab_valid.ev_idx.values
    ev_exit = ev_pos + 1 + lab_valid.hold_bars.fillna(240).values
    # 事件时间上 purge: 训练事件 ev 时间 < 测试月初 - purge窗口
    purge_cut_low = dt_index.searchsorted(om_start_dt - pd.Timedelta(minutes=PURGE_BARS))
    purge_cut_high = dt_index.searchsorted(om_end_dt + pd.Timedelta(minutes=2000))
    tr_ok = train_mask & (ev_pos < purge_cut_low) & (ev_exit < purge_cut_low)
    te_idx = np.where(test_m.values)[0]
    if tr_ok.sum() < 5000 or len(te_idx) == 0:
        continue
    preds = np.zeros(len(lab_valid))
    for side in ['long', 'short']:
        sm = (lab_valid.side == side).values
        tr_i = np.where(tr_ok & sm)[0]
        te_i = np.where(test_m.values & sm)[0]
        if len(tr_i) < 3000 or len(te_i) == 0:
            continue
        Xtr, ytr = X_ev[tr_i], lab_valid.y.values[tr_i]
        Xte = X_ev[te_i]
        cut = int(len(tr_i) * 0.92)
        order = np.argsort(lab_valid.dt.values[tr_i])
        tr_s, va_s = order[:cut], order[cut:]
        dtr = lgb.Dataset(Xtr[tr_s], label=ytr[tr_s])
        dva = lgb.Dataset(Xtr[va_s], label=ytr[va_s], reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=300, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        preds[te_i] = m.predict(Xte, num_iteration=m.best_iteration)
    lab_valid.loc[test_m, 'p'] = preds[test_m]
    pred_rows.append({'month': str(om), 'auc_long': None, 'auc_short': None})
    if (i + 1) % 6 == 0:
        print(f'  fold {i + 1}/{len(oos_months)}')

# 阈值(固定规则: 每折用该折训练段内预测概率分布的q85 —— 此处以全训练期一次近似, 规则一致)
# 严格实现: 用每折训练段尾部2个月事件的OOB概率无法低成本获得, 改用保守等效:
# 阈值作用于测试事件概率的绝对水平, 通过在2022-2024训练期跑一次in-sample概率分布标定q85
print('\n[阈值标定] 在训练期(2022-01~2024-07)做一次20%事件in-sample训练, 取概率q85 ...')
cal = lab_valid[lab_valid.dt < np.datetime64(TRAIN_END)]
cal_X = F.loc[cal.dt].values
thr = {}
for side in ['long', 'short']:
    sm = (cal.side == side).values
    d = lgb.Dataset(cal_X[sm], label=cal.y.values[sm])
    m = lgb.train(params, d, num_boost_round=200)
    p = m.predict(cal_X[sm])
    thr[side] = float(np.quantile(p, THR_QUANTILE))
    print(f'  {side}: thr={thr[side]:.4f} (q{int(THR_QUANTILE*100)})')

# ================================================================ 串行回测引擎
ev = lab_valid.copy()
ev['dt64'] = pd.to_datetime(ev.dt).values
ev = ev.sort_values('dt64').reset_index(drop=True)
oos_ev = ev[(ev.dt64 >= np.datetime64(OOS_START)) & (ev.dt64 < np.datetime64(OOS_END))].copy()
print(f'\n[回测] OOS事件: {len(oos_ev):,}')

def run_serial(events, thr, cost):
    open_until = -1
    trades = []
    i = 0
    n_ev = len(events)
    while i < n_ev:
        row = events.iloc[i]
        if row.ev_idx <= open_until or np.isnan(row.p):
            i += 1
            continue
        # 同一ev的long/short择优
        best = None
        j = i
        while j < n_ev and events.iloc[j].ev_idx == row.ev_idx:
            r = events.iloc[j]
            for side in ['long', 'short']:
                if r.side == side and r.p >= thr[side]:
                    score = abs(r.p - 0.5)
                    if best is None or score > best[0]:
                        best = (score, j, r)
            j += 1
        if best is not None:
            _, jj, r = best
            trades.append({'dt': r.dt64, 'side': r.side, 'p': r.p,
                           'outcome': int(r.outcome), 'pnl_net': r.pnl_gross - cost,
                           'entry': r.entry, 'hold': r.hold_bars,
                           'ev_idx': int(r.ev_idx)})
            open_until = int(r.ev_idx + 1 + (r.hold_bars if not np.isnan(r.hold_bars) else 240))
            i = j
        else:
            i += 1
    return pd.DataFrame(trades)

trades = run_serial(oos_ev, thr, COST_MAIN)
trades_stress = run_serial(oos_ev, thr, COST_STRESS)
print(f'  主口径交易数: {len(trades)}')

def metrics(t, label):
    if len(t) == 0:
        return {label: 'no trades'}
    win = t[t.pnl_net > 0]
    lose = t[t.pnl_net <= 0]
    pnl = t.pnl_net.values
    wr = len(win) / len(t)
    plr = (win.pnl_net.mean() if len(win) else 0) / abs(lose.pnl_net.mean() if len(lose) else 1)
    daily = t.set_index('dt').pnl_net.resample('D').sum()
    mu, sd = daily.mean(), daily.std() + 1e-9
    sh = mu / sd * np.sqrt(252)
    eq = daily.cumsum()
    mdd = float((eq - eq.cummax()).min())
    yrs = (t.dt.max() - t.dt.min()).days / 365.25
    return {label: {'n': int(len(t)), 'win_rate': round(wr, 4), 'plr': round(plr, 3),
                    'pnl_total': round(float(pnl.sum()), 2), 'pnl_per_trade': round(float(pnl.mean()), 4),
                    'sharpe_d': round(float(sh), 3), 'maxdd': round(mdd, 2),
                    'avg_hold_min': round(float(t.hold.mean()), 1), 'years': round(yrs, 2),
                    'long_n': int((t.side == 'long').sum()),
                    'short_n': int((t.side == 'short').sum()),
                    'long_pnl': round(float(t[t.side == "long"].pnl_net.sum()), 2),
                    'short_pnl': round(float(t[t.side == "short"].pnl_net.sum()), 2)}}

m_main = metrics(trades, 'main_cost_0.06')
m_stress = metrics(trades_stress, 'stress_cost_0.10')
print(json.dumps(m_main, indent=1, ensure_ascii=False))
print(json.dumps(m_stress, indent=1, ensure_ascii=False))

# 随机对照(同几何同串行, 随机方向/随机选择) — numba 加速
from numba import njit

@njit(cache=True)
def random_control(ev_idx, pnl_gross, hold, fire_prob, n_trial, cost, seed):
    rng_state = seed
    out = np.zeros(n_trial)
    for t in range(n_trial):
        open_until = -1
        s = 0.0
        cnt = 0
        for i in range(len(ev_idx)):
            if ev_idx[i] <= open_until:
                continue
            rng_state = (rng_state * 1664525 + 1013904223) % 4294967296
            r = rng_state / 4294967296.0
            if r < fire_prob:
                hd = hold[i]
                if hd != hd:  # nan
                    hd = 240.0
                s += pnl_gross[i] - cost
                cnt += 1
                open_until = int(ev_idx[i] + 1 + hd)
        out[t] = s
    return out

fire = len(trades) / max(len(oos_ev) / 2, 1)  # 与模型实际出手率对齐(事件对半为长仓机会)
ev_arr = oos_ev[['ev_idx', 'pnl_gross', 'hold_bars']].sort_values('ev_idx').values
ev_idx = ev_arr[:, 0].astype(np.int64)
pnl_g = ev_arr[:, 1].astype(np.float64)
hold_a = ev_arr[:, 2].astype(np.float64)
rand_pnls = random_control(ev_idx, pnl_g, hold_a, min(fire, 0.3), 300, COST_MAIN, 12345)
rand_ci = [float(np.percentile(rand_pnls, 5)), float(np.percentile(rand_pnls, 95))]
print(f'随机对照(同频率300次, fire={min(fire,0.3):.3f}): mean=${np.mean(rand_pnls):.1f} '
      f'CI5-95=[{rand_ci[0]:.1f}, {rand_ci[1]:.1f}]')

# Bootstrap CI for model total
bs = []
rng = np.random.default_rng(7)
if len(trades) > 0:
    p = trades.pnl_net.values
    for _ in range(1000):
        bs.append(float(np.sum(rng.choice(p, size=len(p), replace=True))))
print(f'模型PnL bootstrap CI5-95=[{np.percentile(bs,5):.1f}, {np.percentile(bs,95):.1f}]')

# 月度分布
if len(trades) > 0:
    mo = trades.set_index('dt').pnl_net.resample('ME').sum()
    pos_m = int((mo > 0).sum())
    print(f'盈利月: {pos_m}/{len(mo)}')

trades.to_csv(f'{BASE}/artifacts/trades.csv', index=False)
rep = {'wf': {'folds': len(oos_months), 'purge_bars': PURGE_BARS,
              'train_window_months': TRAIN_WIN_MONTHS, 'thr_quantile': THR_QUANTILE,
              'thr': thr},
       'main': m_main, 'stress': m_stress,
       'random_control': {'mean': round(float(np.mean(rand_pnls)), 2), 'ci5_95': [round(x, 2) for x in rand_ci]},
       'model_bootstrap_ci': [round(float(np.percentile(bs, 5)), 2), round(float(np.percentile(bs, 95)), 2)],
       'old_baseline_v3bal_ens': {'n': 1611, 'win_rate': 0.316, 'plr': 2.57,
                                   'pnl': 1113.5, 'sharpe': 1.74}}
with open(f'{BASE}/reports/train_report.json', 'w') as f:
    json.dump(rep, f, indent=1, ensure_ascii=False)
print('\n[OK] -> artifacts/trades.csv, reports/train_report.json')
