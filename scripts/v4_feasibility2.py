"""v4 补充测算 — 事件强度k梯度 + 条件拆分(顺势/逆势/时段), 决定事件定义是否有条件燃料
只用训练窗 2022-01~2024-07. 几何 (3.0,1.5)xATR1440, H=120, 成本双口径."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/z/my-project/scripts')
from v4_common import (load_m1, gap_mask, sigma60_excl, atr_m1, spread_usd_imputed,
                       find_events, sim_directional, FIXED_COST_SIDE, TRAIN_END_TS)

pd.set_option('display.width', 220)
d = load_m1()
t, o, h, l, c, v, sp = d['t'], d['o'], d['h'], d['l'], d['c'], d['v'], d['sp']
n = len(t)
gap = gap_mask(t)
lr = np.log(c / np.concatenate(([c[0]], c[:-1])))
lr[0] = 0.0
sig = sigma60_excl(lr)
A1440, _ = atr_m1(h, l, c, 1440)
vmed = pd.Series(v).rolling(1440, min_periods=1440).median().to_numpy()
sp_imp, _ = spread_usd_imputed(t, sp)

# 4h/1日 动量(用于拆分): sum lr 过去240/1440根(不含当前)
sm240 = pd.Series(lr).rolling(240, min_periods=240).sum().shift(1).to_numpy()
sm1440 = pd.Series(lr).rolling(1440, min_periods=1440).sum().shift(1).to_numpy()
hr = pd.to_datetime(t, unit='s').hour.to_numpy()

rows = []
for k in [1.5, 2.0, 2.5, 3.0]:
    ev, dr = find_events(lr, sig, v, vmed, k, 0.8)
    sel = (t[ev] < TRAIN_END_TS) & (ev + 1 < n - 2) & (~gap[ev + 1])
    ev, dr = ev[sel], dr[sel]
    if len(ev) < 500:
        continue
    e_i = ev + 1
    aE = A1440[ev]
    out, xi, xp = sim_directional(o, h, l, c, gap, e_i, dr, 3.0 * aE, 1.5 * aE, 120)
    pnlA = dr * (xp - o[e_i]) / aE          # 以ATR为单位的原始PnL(未扣成本)
    pTP = (out == 1).mean()
    # 拆分: 顺4h势 / 逆4h势
    aligned = np.sign(sm240[ev]) == dr
    rows.append(dict(
        k=k, n=len(ev), per_month=round(len(ev) / 31, 0),
        pTP=round(float(pTP), 3),
        EV_A=round(float(pnlA.mean()), 3),
        pTP_al=round(float((out[aligned] == 1).mean()), 3) if aligned.sum() > 100 else np.nan,
        pTP_ct=round(float((out[~aligned] == 1).mean()), 3) if (~aligned).sum() > 100 else np.nan,
        pTP_al1d=round(float((out[(np.sign(sm1440[ev]) == dr)] == 1).mean()), 3),
        pTP_ct1d=round(float((out[(np.sign(sm1440[ev]) != dr)] == 1).mean()), 3),
    ))
print('== 事件强度梯度 (几何 3.0/1.5 ATR, H=120, EV单位=ATR) ==')
print(pd.DataFrame(rows).to_string(index=False))

# ---- 时段拆分 (k=2.5) ----
ev, dr = find_events(lr, sig, v, vmed, 2.5, 0.8)
sel = (t[ev] < TRAIN_END_TS) & (ev + 1 < n - 2) & (~gap[ev + 1])
ev, dr = ev[sel], dr[sel]
e_i = ev + 1
aE = A1440[ev]
out, xi, xp = sim_directional(o, h, l, c, gap, e_i, dr, 3.0 * aE, 1.5 * aE, 120)
pnlA = dr * (xp - o[e_i]) / aE
hE = hr[ev]
rows2 = []
for nm, hh in [('亚洲0-6', range(0, 6)), ('晨6-12', range(6, 12)), ('午12-18', range(12, 18)), ('晚18-24', range(18, 24))]:
    m = np.isin(hE, list(hh))
    if m.sum() > 200:
        rows2.append(dict(session=nm, n=int(m.sum()), pTP=round(float((out[m] == 1).mean()), 3),
                          EV_A=round(float(pnlA[m].mean()), 3)))
print('\n== 时段拆分 (k=2.5) ==')
print(pd.DataFrame(rows2).to_string(index=False))

# ---- 组合: 顺4h势 × 时段 ----
al = np.sign(sm240[ev]) == dr
rows3 = []
for nm, hh in [('亚洲0-6', range(0, 6)), ('晨6-12', range(6, 12)), ('午12-18', range(12, 18)), ('晚18-24', range(18, 24))]:
    m = np.isin(hE, list(hh))
    for ln, la in [('顺', True), ('逆', False)]:
        mm = m & (al == la)
        if mm.sum() > 100:
            rows3.append(dict(ses=nm, mom=ln, n=int(mm.sum()),
                              pTP=round(float((out[mm] == 1).mean()), 3),
                              EV_A=round(float(pnlA[mm].mean()), 3)))
print('\n== 顺/逆4h势 × 时段 (k=2.5) ==')
print(pd.DataFrame(rows3).to_string(index=False))
