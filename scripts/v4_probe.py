"""v4 — 条件可学性探针 (廉价证伪优先; 段: 训练2022-01~2023-12 / 验证2024-01~2024-07, 不触OOS)

三个种群并行测同一假设: "bar t 方向能否在(3.0/1.5)xATR1440/H120障碍赛跑中先触TP":
  A) 点火事件 (k=2.5, 量能>=0.8x中位, 间隔>=10根)
  B) 常规抽样 bar (每5根取1)
  C) 亚洲区间突破事件 (服务器日 00-06 高/低点, 06-12 首次收盘穿越, 方向=突破侧)

毕业标准(预注册): 验证段 AUC>=0.515 且 预测top五分位 pTP 提升>=+4pp 且 样本>=300
"""
import sys
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, '/home/z/my-project/scripts')
from v4_common import (load_m1, gap_mask, sigma60_excl, atr_m1, spread_usd_imputed,
                       find_events, sim_directional, TRAIN_END_TS)

pd.set_option('display.width', 220)
VAL_TS = int(pd.Timestamp('2024-01-01').timestamp())
OOS_TS = int(pd.Timestamp('2024-08-01').timestamp())

d = load_m1()
t, o, h, l, c, v, sp = d['t'], d['o'], d['h'], d['l'], d['c'], d['v'], d['sp']
n = len(t)
gap = gap_mask(t)
sp_imp, _ = spread_usd_imputed(t, sp)

import v4_featlib
X, names, sig, A, ig2 = v4_featlib.build_all(d, sp_imp)
print(f'特征矩阵: {X.shape}, NaN比例: {np.isnan(X).mean():.4f}')
np.savez_compressed('/home/z/my-project/scripts/v4_cache/featmat.npz', X=X)
with open('/home/z/my-project/scripts/v4_cache/featnames.json', 'w') as f:
    json.dump(names, f)

lr = np.log(c / np.concatenate(([c[0]], c[:-1])))
lr[0] = 0.0
warm = 1500
TP_M, SL_M, H = 3.0, 1.5, 120


def make_labels(ev_idx, dr):
    """ev_idx: 信号bar; dr: 方向; 返回有效mask与标签数组"""
    e_i = ev_idx + 1
    ok = (e_i < n - 1) & (~gap[e_i]) & ~np.isnan(A[ev_idx])
    e_i, ev_idx, dr = e_i[ok], ev_idx[ok], dr[ok]
    aE = A[ev_idx]
    out, xi, xp = sim_directional(o, h, l, c, gap, e_i, dr, TP_M * aE, SL_M * aE, H)
    return ok, ev_idx, e_i, dr, out, xi, xp


# ---- 种群A: 点火事件 ----
vmed = pd.Series(v).rolling(1440, min_periods=1440).median().to_numpy()
evA, drA = find_events(lr, sig, v, vmed, 2.5, 0.8)
evA = evA[evA > warm]
okA, evA, eA, drA, outA, xiA, xpA = make_labels(evA, drA)

# ---- 种群B: 常规抽样 ----
evB = np.arange(warm, n - 2, 5)
evB = evB[lr[evB] != 0]
drB = np.sign(lr[evB]).astype(np.int8)
okB, evB, eB, drB, outB, xiB, xpB = make_labels(evB, drB)

# ---- 种群C: 亚洲区间突破 ----
day_id = t // 86400
hr = pd.to_datetime(t, unit='s').hour.to_numpy()
evC, drC = [], []
for dd in range(day_id[0], day_id[-1] + 1):
    m_day = day_id == dd
    if m_day.sum() < 100:
        continue
    m_asia = m_day & (hr < 6)
    if m_asia.sum() < 30:
        continue
    ah = h[m_asia].max()
    al = l[m_asia].min()
    m_london = np.where(m_day & (hr >= 6) & (hr < 12))[0]
    done_up = done_dn = False
    for i in m_london:
        if not done_up and c[i] > ah:
            evC.append(i); drC.append(1); done_up = True
        if not done_dn and c[i] < al:
            evC.append(i); drC.append(-1); done_dn = True
        if done_up and done_dn:
            break
evC = np.array(evC, dtype=np.int64)
drC = np.array(drC, dtype=np.int8)
okC, evC, eC, drC, outC, xiC, xpC = make_labels(evC, drC)


def probe(arm, ev, dr, out, xi, xp, ev_raw):
    tv = t[ev]
    tr_m = tv < VAL_TS
    va_m = (tv >= VAL_TS) & (tv < OOS_TS)
    # purge: 训练行标签窗口不得入验证段; 验证行标签窗口不得入OOS
    val_start_bar = np.searchsorted(t, VAL_TS)
    oos_start_bar = np.searchsorted(t, OOS_TS)
    tr_m &= xi < val_start_bar
    va_m &= xi < oos_start_bar
    Xa = X[ev]
    y = (out == 1).astype(int)
    res = dict(arm=arm, n_train=int(tr_m.sum()), n_val=int(va_m.sum()),
               base_tr=round(float(y[tr_m].mean()), 3), base_va=round(float(y[va_m].mean()), 3))
    n_train, n_val = res['n_train'], res['n_val']
    if res['n_train'] < 300 or res['n_val'] < 150:
        res['skip'] = '样本不足'
        return res
    ds_tr = lgb.Dataset(Xa[tr_m], label=y[tr_m])
    ds_va = lgb.Dataset(Xa[va_m], label=y[va_m], reference=ds_tr)
    params = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=31,
                  min_data_in_leaf=40, feature_fraction=0.9, bagging_fraction=0.9,
                  bagging_freq=1, lambda_l1=0.1, max_bin=127, num_threads=2,
                  seed=7, deterministic=True, verbosity=-1)
    m = lgb.train(params, ds_tr, num_boost_round=800, valid_sets=[ds_va],
                 callbacks=[lgb.early_stopping(80, verbose=False)])
    p = m.predict(Xa[va_m], num_iteration=m.best_iteration)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y[va_m], p)
    res['auc'] = round(float(auc), 4)
    res['best_iter'] = int(m.best_iteration)
    # 分位表
    q = pd.Series(p)
    dec = pd.qcut(p, 5, labels=False, duplicates='drop')
    tbl = []
    for k in sorted(set(dec)):
        mm = dec == k
        tbl.append(dict(q5=int(k) + 1, n=int(mm.sum()), pTP=round(float(y[va_m][mm].mean()), 3)))
    res['quintiles'] = tbl
    top = dec == max(set(dec))
    res['top_pTP'] = tbl[-1]['pTP']
    res['lift_pp'] = round((res['top_pTP'] - res['base_va']) * 100, 1)
    res['n_top'] = int(top.sum())
    # top五分位 EV (A单位, 扣双口径成本)
    aE = A[ev][va_m][top]
    pnlA = dr[va_m][top] * (xp[va_m][top] - o[ev[va_m][top] + 1]) / aE
    res['EV_A_top'] = round(float(pnlA.mean()), 3)
    res['grad'] = bool(auc >= 0.515 and res['lift_pp'] >= 4.0 and int(top.sum()) >= 300)
    return res


results = []
for arm, ev, dr, out, xi, xp, evr in [
    ('A_点火k2.5', evA, drA, outA, xiA, xpA, evA),
    ('B_常规抽样1/5', evB, drB, outB, xiB, xpB, evB),
    ('C_亚区间突破', evC, drC, outC, xiC, xpC, evC),
]:
    r = probe(arm, ev, dr, out, xi, xp, evr)
    results.append(r)
    print('\n==', arm, '==')
    print(json.dumps(r, ensure_ascii=False, indent=1))
    if 'quintiles' in r:
        print(pd.DataFrame(r['quintiles']).to_string(index=False))

with open('/home/z/my-project/scripts/v4_cache/probe.json', 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
print('\n毕业判定(AUC>=0.515 & top五分位提升>=+4pp & n>=300):')
for r in results:
    print(' ', r['arm'], '->', r.get('grad', 'skip'))
