"""v4 — OOS选择度(q)敏感性诊断 + 长-only事后假设 (纯用已存档oosprobs/标签, 不重训)
用途: 判定"探针的top五分位信号是否在任何选择度下转化为正期望", 以及证伪的稳健性."""
import json
import glob
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/z/my-project/scripts')
from v4_common import (load_m1, gap_mask, sigma60_excl, atr_m1, spread_usd_imputed,
                       find_events, sim_directional, FIXED_COST_SIDE, month_id_of)

CACHE = '/home/z/my-project/scripts/v4_cache'
d = load_m1()
t, o, h, l, c, v, sp = d['t'], d['o'], d['h'], d['l'], d['c'], d['v'], d['sp']
n = len(t)
gap = gap_mask(t)
sp_imp, _ = spread_usd_imputed(t, sp)
lr = np.log(c / np.concatenate(([c[0]], c[:-1])))
lr[0] = 0.0
sig = sigma60_excl(lr)
A1440, _ = atr_m1(h, l, c, 1440)
vmed = pd.Series(v).rolling(1440, min_periods=1440).median().to_numpy()
mid, starts = month_id_of(t)
ev, dr = find_events(lr, sig, v, vmed, 2.5, 0.8)
ev = ev[ev > 1500]
ok = (ev + 1 < n - 1) & (~gap[ev + 1]) & ~np.isnan(A1440[ev])
ev, dr = ev[ok], dr[ok]
aE = A1440[ev]
out_all, xi_all, xp_all = sim_directional(o, h, l, c, gap, ev + 1, dr, 3.0 * aE, 1.5 * aE, 120)
month_ev = mid[ev]
pnlA = dr * (xp_all - o[ev + 1])           # 毛
spE = sp_imp[ev + 1]

folds = [json.load(open(f)) for f in sorted(glob.glob(f'{CACHE}/fold_*.json'))]
rows = []
for q in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    tot_old = tot_real = 0.0
    ns, wins = [], []
    totL = totS = 0.0
    nL = nS = 0
    for f in folds:
        mi = f['month']
        sub = pd.read_csv(f'{CACHE}/oosprobs_{mi}.csv')
        if len(sub) == 0:
            continue
        th = float(np.quantile(sub['p'], q))
        sel = sub[sub['p'] >= th]
        ge = sel['ev'].to_numpy()
        pos = np.searchsorted(ev, ge)   # 全局事件行
        last = -10 ** 9
        keep = []
        for k, g_e in enumerate(ge):
            if g_e + 1 > last + 10:
                keep.append(k)
                last = xi_all[pos[k]]
        kp = pos[np.array(keep, dtype=int)]
        if len(kp) == 0:
            continue
        tot_old += float(pnlA[kp].sum() - 2 * FIXED_COST_SIDE * len(kp))
        tot_real += float(pnlA[kp].sum() - 2 * spE[kp].sum())
        ns.append(len(kp))
        wins.append(float((out_all[kp] == 1).mean()))
        mL = dr[kp] > 0
        totL += float(pnlA[kp][mL].sum() - 2 * FIXED_COST_SIDE * mL.sum())
        totS += float(pnlA[kp][~mL].sum() - 2 * FIXED_COST_SIDE * (~mL).sum())
        nL += int(mL.sum()); nS += int((~mL).sum())
    rows.append(dict(q=q, n=int(sum(ns)), wr=round(float(np.mean(wins)), 3),
                     pnl_old=round(tot_old, 1), pnl_real=round(tot_real, 1),
                     long=round(totL, 1), short=round(totS, 1)))
print(pd.DataFrame(rows).to_string(index=False))
json.dump(rows, open('/home/z/my-project/download/xauusd_ml_v4/qsens.json', 'w'), indent=1)
