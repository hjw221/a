"""v4 — OOS聚合评估 + 双安慰剂 + 净值曲线 + 基线对比"""
import json
import glob
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/z/my-project/scripts')
from v4_common import (load_m1, gap_mask, sigma60_excl, atr_m1, spread_usd_imputed,
                       find_events, sim_directional, FIXED_COST_SIDE, month_id_of)

OUT = '/home/z/my-project/download/xauusd_ml_v4'
import os
os.makedirs(OUT, exist_ok=True)

# ---------- 数据与事件重建(与WF完全同路径, 确定性) ----------
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
e_i = ev + 1
aE = A1440[ev]
out_all, xi_all, xp_all = sim_directional(o, h, l, c, gap, e_i, dr, 3.0 * aE, 1.5 * aE, 120)
month_ev = mid[ev]
pnlA = dr * (xp_all - o[e_i])   # 毛(未扣成本)

# ---------- 载入WF产物 ----------
trades = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('/home/z/my-project/scripts/v4_cache/trades_*.csv'))], ignore_index=True)
probs = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('/home/z/my-project/scripts/v4_cache/oosprobs_*.csv'))], ignore_index=True)
folds = [json.load(open(f)) for f in sorted(glob.glob('/home/z/my-project/scripts/v4_cache/fold_*.json'))]
trades = trades.sort_values('exit_t').reset_index(drop=True)

# pooled OOS AUC
from sklearn.metrics import roc_auc_score
pooled_auc = roc_auc_score(probs['y'], probs['p'])

def metrics(pnl, times):
    pnl = np.asarray(pnl, float)
    m = pd.Series(pnl, index=pd.to_datetime(times, unit='s')).resample('ME').sum()
    sharpe = float(m.mean() / (m.std() + 1e-12) * np.sqrt(12)) if m.std() > 0 else 0.0
    cum = np.cumsum(pnl)
    dd = float(np.max(np.maximum.accumulate(cum) - cum))
    win = pnl[pnl > 0]
    los = pnl[pnl <= 0]
    return dict(n=len(pnl), wr=round(float((pnl > 0).mean()), 4),
                plr=round(float(win.mean() / abs(los.mean())), 3) if len(los) else np.nan,
                total=round(float(pnl.sum()), 1), mean=round(float(pnl.mean()), 4),
                pf=round(float(win.sum() / abs(los.sum())), 3) if abs(los.sum()) > 0 else np.nan,
                sharpe=round(sharpe, 2), maxdd=round(dd, 1),
                tstat=round(float(pnl.mean() / (pnl.std() / np.sqrt(len(pnl)))), 2),
                prof_m=int((m > 0).sum()), n_m=len(m))

mt_old = metrics(trades['pnl_old'], trades['exit_t'])
mt_real = metrics(trades['pnl_real'], trades['exit_t'])

# bootstrap CI (总PnL)
rng = np.random.default_rng(7)
B = 2000
idx = rng.integers(0, len(trades), size=(B, len(trades)))
tot_b = trades['pnl_old'].to_numpy()[idx].sum(axis=1)
ci_old = [round(float(np.percentile(tot_b, q)), 1) for q in (2.5, 50, 97.5)]
tot_br = trades['pnl_real'].to_numpy()[idx].sum(axis=1)
ci_real = [round(float(np.percentile(tot_br, q)), 1) for q in (2.5, 50, 97.5)]

# 多空拆分 / 年度拆分
et = pd.to_datetime(trades['exit_t'], unit='s')
trades['year'] = et.dt.year
split_ls = {dd: metrics(trades.loc[trades['dir'] == dd, 'pnl_old'], trades.loc[trades['dir'] == dd, 'exit_t'])
            for dd in (1, -1)}
by_year = {}
for yy, g in trades.groupby('year'):
    by_year[int(yy)] = dict(n=len(g), old=round(float(g['pnl_old'].sum()), 1),
                            real=round(float(g['pnl_real'].sum()), 1),
                            wr=round(float((g['pnl_old'] > 0).mean()), 3))

# ---------- 安慰剂1: 概率置换(同折同阈值, 随机选事件) ----------
th_by_m = {f['month']: f['theta'] for f in folds}
pl1 = []
for rep in range(20):
    tot = 0.0
    for f in folds:
        mi = f['month']
        sub = probs[probs['month'] == mi]
        if len(sub) == 0:
            continue
        pp = rng.permutation(sub['p'].to_numpy())
        sel = pp >= th_by_m[mi]
        rows = np.where(sel)[0]
        # 无重叠
        last = -10 ** 9
        for r in rows:
            gi = np.where(month_ev == mi)[0][r] if False else None
        # 直接用全局事件索引
        ge = sub['ev'].to_numpy()[rows]
        last = -10 ** 9
        keep = []
        for g_e in ge:
            if g_e + 1 > last + 10:
                keep.append(g_e)
                last = xi_all[np.searchsorted(ev, g_e)]
        keep = np.array(keep)
        pos = np.searchsorted(ev, keep)
        tot += pnlA[pos].sum() - 2 * FIXED_COST_SIDE * len(pos)
    pl1.append(tot)
pl1 = np.array(pl1)

# ---------- 安慰剂2: 随机事件(同月同数, 随机bar, 方向=bar方向) ----------
elig = np.where((np.arange(n) > 1500) & (lr != 0) & (np.arange(n) < n - 2) &
                ~np.isnan(A1440))[0]
gap_ok = ~gap[np.minimum(elig + 1, n - 1)]
elig = elig[gap_ok]
mid_elig = mid[elig]
pl2 = []
for rep in range(20):
    tot = 0.0
    for f in folds:
        mi = f['month']
        k = int(f.get('n_trades', 0))
        if k == 0:
            continue
        cand = elig[mid_elig == mi]
        if len(cand) < k:
            continue
        pick = rng.choice(cand, size=k, replace=False)
        dR = np.sign(lr[pick]).astype(np.int8)
        oo, xxi, xpp = sim_directional(o, h, l, c, gap, pick + 1, dR,
                                       3.0 * A1440[pick], 1.5 * A1440[pick], 120)
        tot += float((dR * (xpp - o[pick + 1])).sum() - 2 * FIXED_COST_SIDE * len(pick))
    pl2.append(tot)
pl2 = np.array(pl2)

# ---------- 汇总 ----------
res = dict(
    pooled_oos_auc=round(float(pooled_auc), 4),
    oos_base_pTP=round(float(probs['y'].mean()), 3),
    old=mt_old, real=mt_real, ci_old=ci_old, ci_real=ci_real,
    long=split_ls[1], short=split_ls[-1], by_year=by_year,
    placebo_perm_p=[round(float(pl1.mean()), 1), round(float(np.percentile(pl1, 5)), 1),
                    round(float(np.percentile(pl1, 95)), 1)],
    placebo_rand_ev=[round(float(pl2.mean()), 1), round(float(np.percentile(pl2, 5)), 1),
                     round(float(np.percentile(pl2, 95)), 1)],
    fold_theta={f['month']: f['theta'] for f in folds},
    fold_auc_oos={f['month']: f['auc_oos'] for f in folds},
)
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
trades.to_csv(f'{OUT}/trades_oos.csv', index=False)
pd.DataFrame(folds).to_csv(f'{OUT}/foldtrace.csv', index=False)
json.dump(res, open(f'{OUT}/metrics.json', 'w'), ensure_ascii=False, indent=1, default=str)

# ---------- 净值曲线 ----------
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/lxgw-wenkai/LXGWWenKai-Regular.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['LXGW WenKai', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), constrained_layout=True,
                         sharex=False)
ax = axes[0]
xs = pd.to_datetime(trades['exit_t'], unit='s')
ax.plot(xs, trades['pnl_old'].cumsum(), lw=1.6, color='#1a6faf', label='固定成本 0.03美元/边 (与基线同口径)')
ax.plot(xs, trades['pnl_real'].cumsum(), lw=1.4, color='#c0392b', ls='--',
        label='数据驱动成本 0.16-0.28美元/边 (真实点差)')
ax.axhline(0, color='gray', lw=0.8)
ax.set_title('v4「M1点火-延续」从零路线 — OOS累计净损益 (2024-08 ~ 2026-07, 24折WF)')
ax.set_ylabel('累计净PnL (美元)')
ax.legend(loc='upper left', fontsize=9)
ax.grid(alpha=0.3)
ax2 = axes[1]
mo = pd.Series(trades['pnl_old'].to_numpy(), index=xs).resample('ME').sum()
mo2 = pd.Series(trades['pnl_real'].to_numpy(), index=xs).resample('ME').sum()
ax2.bar(mo.index, mo.values, width=22, color=np.where(mo.values >= 0, '#2e8b57', '#b0533a'),
        label='月度净PnL (固定成本)')
ax2.plot(mo2.index, mo2.values, color='#c0392b', lw=1.3, marker='o', ms=3,
         label='月度净PnL (数据驱动成本)')
ax2.axhline(0, color='gray', lw=0.8)
ax2.set_ylabel('月度净PnL (美元)')
ax2.set_title(f"月度损益 | 总计: 固定成本 {mt_old['total']:+.0f}美元 / 真实点差 {mt_real['total']:+.0f}美元 | "
              f"Sharpe {mt_old['sharpe']:.2f}/{mt_real['sharpe']:.2f}")
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(alpha=0.3)
fig.savefig(f'{OUT}/equity.png', dpi=150)
print('saved equity.png')
