#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S04 — 从零构建 | 因子挖掘 (ML找因子)
=====================================
[P1] null-importance: LightGBM 真实gain vs 15次shuffle-target null分布 -> z-score
     (H=120 fwd_ret 为目标, 训练期60%采样, 浅树快跑, 只辨真伪)
[P2] 事件研究: jump/量能突增/区间极值触及 后 15/60/120/240 分钟累计路径(bootstrap CI)
     -> 找事件性alpha: 延续 or 回补?
[P3] 因子构造(数据证据驱动): 反转族为主 + 波段门控族
     检验: IC@H60/120, 分年稳定性, 因子自相关, 多空spread(扣2bp成本)
[P4] 相关聚类(|r|>0.7): 簇内选ICIR最高 -> 最终因子表
输出: reports/factor_report.json, artifacts/factors.pkl
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

BASE = '/home/z/my-project/download/xauusd_ml_scratch'
TRAIN_END = '2024-08-01'
EXCLUDE = ['C_gap_open']  # 伪影, 已判定剔除

df = pd.read_pickle(f'{BASE}/artifacts/m1_clean.pkl')
F = pd.read_pickle(f'{BASE}/artifacts/features.pkl')
c = df.close
tr = F.index < TRAIN_END
cols = [x for x in F.columns if x not in EXCLUDE]

# ================================================================ P1: null-importance (方向二分类)
H = 120
y_all = (c.shift(-H) / c - 1)[tr]
X_all = F[cols][tr]
ok = y_all.notna()
X, y = X_all[ok], y_all[ok]
y_bin = (y > 0).astype(int)
rng = np.random.default_rng(42)
sub = rng.choice(len(X), size=int(len(X) * 0.6), replace=False)
Xs, ys = X.iloc[sub], y_bin.iloc[sub]

params = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=2000, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=5.0, verbosity=-1, num_threads=2, seed=42)
print(f'[P1] null-importance(方向分类): {Xs.shape[0]:,}行 x {Xs.shape[1]}特征, 8次shuffle ...')
dtrain = lgb.Dataset(Xs, label=ys)
real = lgb.train(params, dtrain, num_boost_round=250)
real_gain = pd.Series(real.feature_importance('gain'), index=cols)

null_gains = []
for i in range(8):
    y_perm = ys.sample(frac=1.0, random_state=i).values
    d = lgb.Dataset(Xs, label=y_perm)
    m = lgb.train(params, d, num_boost_round=250)
    null_gains.append(m.feature_importance('gain'))
    if (i + 1) % 4 == 0:
        print(f'    shuffle {i + 1}/8')
null_gains = np.array(null_gains)  # (8, n_feat)
z = (real_gain.values - null_gains.mean(0)) / (null_gains.std(0) + 1e-9)
rank = pd.DataFrame({'real_gain': real_gain.values, 'z': z}, index=cols).sort_values('z', ascending=False)
print('\n[P1] null-importance TOP20 (z = 真实gain超越null分布的标准差倍数):')
for name, row in rank.head(20).iterrows():
    print(f'  {name:<22} z={row.z:>8.1f}  gain={row.real_gain:>10.1f}')
rank.to_json(f'{BASE}/reports/null_importance.json', orient='index')

# ================================================================ P2: 事件研究
print('\n[P2] 事件研究 ...')
def event_study(mask, name, k_list=(15, 60, 120, 240), n_boot=500):
    pos = np.where(mask & tr & c.notna())[0]
    res = {}
    for k in k_list:
        fwd = (c.values[np.minimum(pos + k, len(c) - 1)] / c.values[pos] - 1) * 1e4
        fwd = fwd[~np.isnan(fwd)]
        bs = rng.choice(fwd, size=(n_boot, min(len(fwd), 2000)), replace=True).mean(1)
        res[f'{k}m'] = {'n': int(len(fwd)), 'mean_bp': round(float(fwd.mean()), 3),
                        'ci5': round(float(np.percentile(bs, 5)), 3),
                        'ci95': round(float(np.percentile(bs, 95)), 3)}
    return res

events = {}
jump_q = F['B_jump_60'].quantile(0.99)
events['after_jump_up'] = event_study((F['B_jump_60'] > jump_q) & (F['A_ret_15'] > 0).values, 'jump_up')
events['after_jump_dn'] = event_study((F['B_jump_60'] > jump_q) & (F['A_ret_15'] < 0).values, 'jump_dn')
events['after_surge'] = event_study((F['F_tvol_surge'] > 2.5).values, 'surge')
events['at_high_240'] = event_study((F['C_pos_240'] > 0.98).values, 'hi')
events['at_low_240'] = event_study((F['C_pos_240'] < 0.02).values, 'lo')
events['at_high_1440'] = event_study((F['C_pos_1440'] > 0.99).values, 'hi1440')
events['at_low_1440'] = event_study((F['C_pos_1440'] < 0.01).values, 'lo1440')
for k, v in events.items():
    s = ' | '.join([f"{kk}: {vv['mean_bp']:+.2f}bp [{vv['ci5']:+.2f},{vv['ci95']:+.2f}] n={vv['n']}"
                    for kk, vv in v.items()])
    print(f'  {k:<16} {s}')

# ================================================================ P3: 因子构造
print('\n[P3] 因子构造与检验 ...')
def zsc(s):
    lo, hi = s.quantile(0.001), s.quantile(0.999)
    s = s.clip(lo, hi)
    return (s - s.mean()) / (s.std() + 1e-12)

cand = {
    'F_rev_short':  -zsc(F['A_ret_15']),                       # 短反转: 15m冲高->看跌(取负=看涨因子)
    'F_rev_fast':   -zsc(F['A_emax_8_32'] * 1e4),              # 快线反转
    'F_rev_pos240': -zsc(F['C_pos_240']),                      # 4h区间位置反转
    'F_rev_pos1440': -zsc(F['C_pos_1440']),                    # 日区间位置反转
    'F_dist_extreme': zsc(F['C_dist_high_240']) - zsc(F['C_dist_low_240']),  # 距端点对称偏离
    'F_align_rev':  -zsc(F['A_align_240']),                   # 一致性反转
    'F_slow_mom':   -zsc(F['A_emax_128_512'] * 1e4),          # 慢趋势反转(证据弱, 检验)
    'F_vol_regime': zsc(F['B_volratio']),                     # 门控: 波动regime
    'F_eu_gate':    F['E_sess_eu'].astype('float64'),         # 门控: 欧时段
    'F_us_gate':    F['E_sess_us'].astype('float64'),         # 门控: 美时段
    'F_jump_state': zsc(F['B_jump_60']),                       # 门控: 跳变后
    'F_tvol':       zsc(F['F_tvol_surge']),                   # 量能
    'F_atr_regime': zsc(F['B_atr_regime']),                    # 门控: 慢波动regime
    'F_dow':        F['E_dow'].astype('float64'),              # 星期(弱)
}
fact = pd.DataFrame({k: v.astype('float32') for k, v in cand.items()}, index=F.index)

# 检验: IC@H, 分年IC, 自相关, 多空
year_arr = np.asarray(F.index.year)
tr_arr = np.asarray(tr)
check = {}
for name in fact.columns:
    row = {}
    for Hh in [60, 120]:
        fw = ((c.shift(-Hh) / c - 1) * 1e4)[tr]  # bp
        x = fact[name][tr]
        m = (x.notna() & fw.notna()).values
        ic = float(np.corrcoef(x[m].rank(), fw[m].rank())[0, 1])
        yearly = []
        for yy in [2022, 2023, 2024]:
            my = m & (year_arr[:len(m)] == yy) & tr_arr[:len(m)]
            if my.sum() > 50000:
                yearly.append(float(np.corrcoef(x[my].rank(), fw[my].rank())[0, 1]))
        row[f'ic_H{Hh}'] = round(ic, 4)
        row[f'icY_H{Hh}'] = [round(v, 4) for v in yearly]
        row[f'signY_H{Hh}'] = int(np.sum(np.sign(yearly) == np.sign(np.mean(yearly))))
    x = fact[name][tr]
    ac = x.autocorr(1)
    fw2 = ((c.shift(-120) / c - 1) * 1e4)[tr]
    m = (x.notna() & fw2.notna()).values
    qb = pd.qcut(x[m].rank(), 5, labels=False, duplicates='drop')
    ls = fw2[m].groupby(qb).mean()
    row['ls_spread_bp'] = round(float(ls.iloc[-1] - ls.iloc[0]), 3) if len(ls) >= 2 else 0.0
    row['q1_bp'] = round(float(ls.iloc[0]), 3) if len(ls) >= 1 else 0.0
    row['q5_bp'] = round(float(ls.iloc[-1]), 3) if len(ls) >= 1 else 0.0
    row['ac1'] = round(float(ac), 3)
    check[name] = row
print(f"{'factor':<16}{'ic60':>8}{'ic120':>8}{'signY':>7}{'ls_bp':>8}{'ac1':>7}")
for name, row in check.items():
    print(f"{name:<16}{row['ic_H60']:>+8.4f}{row['ic_H120']:>+8.4f}{row['signY_H120']:>5}/3"
          f"{row['ls_spread_bp']:>+8.2f}{row['ac1']:>7.2f}")

# ================================================================ P4: 相关聚类
corr = fact[tr].corr('spearman').abs()
keep, dropped = [], []
claimed = set()
for name in fact.columns:  # 按构造顺序贪心
    if name in claimed:
        continue
    keep.append(name)
    claimed.add(name)
    for other in fact.columns:
        if other != name and other not in claimed and corr.loc[name, other] > 0.7:
            dropped.append({'dropped': other, 'because': name, 'rho': round(float(corr.loc[name, other]), 3)})
            claimed.add(other)
print(f'\n[P4] 相关聚类: 保留 {keep}')
print(f'    剔除 {dropped}')

fact[tr].astype('float32').to_pickle(f'{BASE}/artifacts/factors_train.pkl')
fact.astype('float32').to_pickle(f'{BASE}/artifacts/factors.pkl')
rep = {'null_importance_top': rank.head(25).reset_index().rename(
           columns={'index': 'feature'}).to_dict('records'),
       'events': events, 'factor_check': check, 'kept': keep, 'dropped': dropped}
with open(f'{BASE}/reports/factor_report.json', 'w') as f:
    json.dump(rep, f, indent=1, ensure_ascii=False)
print('\n[OK] -> reports/factor_report.json, artifacts/factors.pkl')
