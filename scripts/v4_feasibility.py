"""v4 — 可行性测算 (只用首训练窗 2022-01~2024-07, 不含任何OOS信息)

产出:
  A. 数据质量 + 点差真相(SPREAD列, 决定成本口径)
  B. 波动率分布: sigma60 / ATR1440 按年
  C. 事件率 vs (k, 量能过滤): 定事件参数
  D. 障碍几何网格: 事件后碰线率/净期望 vs 随机入场对照, 方向拆分
"""
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/z/my-project/scripts')
from v4_common import (load_m1, gap_mask, sigma60_excl, atr_m1, spread_usd_imputed,
                       find_events, sim_directional, FIXED_COST_SIDE, TRAIN_END_TS)

pd.set_option('display.width', 200)

d = load_m1()
t, o, h, l, c, v, sp = d['t'], d['o'], d['h'], d['l'], d['c'], d['v'], d['sp']
n = len(t)
gap = gap_mask(t)
res = {}

# ============ A. 数据质量 + 点差 ============
dt = np.diff(t)
yr = (t // 31536000 + 1970).astype(int)
# 注: 用日历近似年即可(统计用)
yr = pd.to_datetime(t, unit='s').year
print(f'== 数据: {n:,}根M1 | {pd.to_datetime(t[0])} ~ {pd.to_datetime(t[-1])} | 去重后单调')
print(f'   缺口>5min: {gap.sum():,}次 | 最大缺口 {dt.max()/3600:.1f}h | 单bar最大TR ${(np.maximum(h-l, np.abs(h-np.concatenate(([c[0]],c[:-1]))))).max():.2f}')
sp_imp, med_m = spread_usd_imputed(t, sp)
sp_raw = sp * 0.001
tbl_sp = []
for y in sorted(set(yr)):
    m = yr == y
    tbl_sp.append(dict(year=int(y), n_bars=int(m.sum()),
                       raw_zero_pct=round(float((sp_raw[m] <= 0).mean()) * 100, 1),
                       med_imp=round(float(np.median(sp_imp[m])), 4),
                       p90_imp=round(float(np.percentile(sp_imp[m], 90)), 4)))
res['spread_by_year'] = tbl_sp
print('\n== 点差($/单边, 月中位数插补后) ==')
print(pd.DataFrame(tbl_sp).to_string(index=False))

# ============ B. 波动率分布 ============
lr = np.log(c / np.concatenate(([c[0]], c[:-1])))
lr[0] = 0.0
sig = sigma60_excl(lr)                     # 1m收益60根std(不含当前)
A1440, tr = atr_m1(h, l, c, 1440)          # M1原生24h ATR
tbl_v = []
for y in sorted(set(yr)):
    m = (yr == y) & ~np.isnan(sig) & ~np.isnan(A1440)
    tbl_v.append(dict(year=int(y), sig60_med=round(float(np.median(sig[m])), 6),
                      sig60_p90=round(float(np.percentile(sig[m], 90)), 6),
                      atr24h_med=round(float(np.median(A1440[m])), 3),
                      atr24h_p10=round(float(np.percentile(A1440[m], 10)), 3),
                      atr24h_p90=round(float(np.percentile(A1440[m], 90)), 3)))
res['vol_by_year'] = tbl_v
print('\n== 波动率(按年) ==')
print(pd.DataFrame(tbl_v).to_string(index=False))

# ============ C. 事件率 vs (k, 量能过滤) — 仅训练窗 ============
vmed = pd.Series(v).rolling(1440, min_periods=1440).median().to_numpy()
train_m = t < TRAIN_END_TS
n_months_train = 31.0
tbl_e = []
for k in [2.0, 2.5, 3.0]:
    for vm in [0.0, 0.8, 1.2]:
        ev, dr = find_events(lr, sig, v, vmed, k, vm)
        ev_tr = ev[t[ev] < TRAIN_END_TS]
        dr_tr = dr[:len(ev)][t[ev] < TRAIN_END_TS]
        cnt = len(ev_tr)
        tbl_e.append(dict(k=k, v_mult=vm, n_train=cnt,
                          per_month=round(cnt / n_months_train, 1),
                          pct_long=round(float((dr_tr > 0).mean()) * 100, 1) if cnt else np.nan))
res['event_rate'] = tbl_e
print('\n== 事件率(训练窗31个月) ==')
print(pd.DataFrame(tbl_e).to_string(index=False))

# ============ D. 障碍几何网格 (k=2.5, vm=0.8) 事件 vs 随机 ============
ev, dr = find_events(lr, sig, v, vmed, 2.5, 0.8)
sel = (t[ev] < TRAIN_END_TS) & (ev + 1 < n - 2)
# 信号->入场不能跨缺口(跨周末等), 否则信息过期
e_ok = ~gap[ev + 1]
sel = sel & e_ok
ev_tr, dr_tr = ev[sel], dr[sel]
print(f'\n== 几何网格事件数(训练窗, k=2.5,vm=0.8): {len(ev_tr)} ({len(ev_tr)/n_months_train:.0f}/月) ==')

rng = np.random.default_rng(42)
elig = np.where((t < TRAIN_END_TS - 200 * 60) & ~np.isnan(A1440) & (np.arange(n) > 1500))[0]
evR = rng.choice(elig, size=len(ev_tr), replace=False)
evR = np.sort(evR)
drR = rng.choice([-1, 1], size=len(ev_tr)).astype(np.int8)

GRID = [(1.0, 0.5), (1.5, 0.75), (2.0, 1.0), (1.5, 1.0), (3.0, 1.5)]
rows = []
for (tpm, slm) in GRID:
    for H in [120, 240]:
        # ---- 事件入场 ----
        e_i = ev_tr + 1
        aE = A1440[ev_tr]
        out, xi, xp = sim_directional(o, h, l, c, gap, e_i, dr_tr, tpm * aE, slm * aE, H)
        pnl = dr_tr * (xp - o[e_i]) - 2 * FIXED_COST_SIDE
        r = dict(src='event', tp=tpm, sl=slm, H=H,
                 pTP=round(float((out == 1).mean()), 3),
                 pSL=round(float((out == -1).mean()), 3),
                 pTO=round(float((out == 0).mean()), 3),
                 EV=round(float(pnl.mean()), 3))
        # 方向拆分
        for dd, nm in [(1, 'L'), (-1, 'S')]:
            mm = dr_tr == dd
            if mm.sum() > 0:
                r[f'pTP_{nm}'] = round(float((out[mm] == 1).mean()), 3)
                r[f'EV_{nm}'] = round(float(pnl[mm].mean()), 3)
        # 时间符号基准
        fwd = np.minimum(e_i + H, n - 1)
        r['time_sign'] = round(float((dr_tr * (c[fwd] - o[e_i]) > 0).mean()), 3)
        rows.append(r)
        # ---- 随机对照 ----
        e_iR = evR + 1
        aR = A1440[evR]
        outR, xiR, xpR = sim_directional(o, h, l, c, gap, e_iR, drR, tpm * aR, slm * aR, H)
        pnlR = drR * (xpR - o[e_iR]) - 2 * FIXED_COST_SIDE
        rows.append(dict(src='random', tp=tpm, sl=slm, H=H,
                        pTP=round(float((outR == 1).mean()), 3),
                        pSL=round(float((outR == -1).mean()), 3),
                        pTO=round(float((outR == 0).mean()), 3),
                        EV=round(float(pnlR.mean()), 3)))
res['grid'] = rows
print(pd.DataFrame(rows).to_string(index=False))

os_path = '/home/z/my-project/scripts/v4_cache/feasibility.json'
with open(os_path, 'w') as f:
    json.dump(res, f, indent=1, default=str)
print(f'\nsaved -> {os_path}')
