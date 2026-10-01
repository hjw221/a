"""v4 — Walk-Forward 全量验证 (预注册配置, 逐折checkpoint可续跑)

冻结配置 (基于训练窗2022-01~2024-07测算+2024-01~07内部探针, 均未触OOS):
  事件: k=2.5 |lr|/sigma60(不含当前根), 量能>=0.8x日中位, 相邻间隔>=10根
  标签: 方向=事件方向; TP=3.0xATR1440, SL=1.5xATR1440, H=120根; 同根双碰按SL; 缺口前按超时
  模型: 单LGBM(dir为特征), 逐月重训, 早停于滚动val(3个月) AUC
  阈值: val上无重叠模拟净PnL最大化(旧成本$0.03/边), >=40笔约束, 网格0.34-0.60
  执行: 事件收盘信号->下一根开盘入场, 单持仓, 平仓后冷却10根
  折叠: OOS 2024-08..2026-07 月度24折, 扩展训练窗(核心<=m-4月), exit-based purge
  成本: 双口径 pnl_old($0.03/边) + pnl_real(入场bar点差$/边)
"""
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/z/my-project/scripts')
from v4_common import (load_m1, gap_mask, sigma60_excl, atr_m1, spread_usd_imputed,
                      find_events, sim_directional, FIXED_COST_SIDE)

CACHE = '/home/z/my-project/scripts/v4_cache'
FOLD0, N_FOLDS = 31, 24          # 月id: 2022-01=0 -> 2024-08=31 .. 2026-07=54
COOLDOWN = 10

# ---------- 数据 ----------
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
mid, starts = None, None
from v4_common import month_id_of
mid, starts = month_id_of(t)

X = np.load(f'{CACHE}/featmat.npz')['X']
names = json.load(open(f'{CACHE}/featnames.json'))
DIRI = names.index('dir')

# ---------- 事件 + 标签 (全期) ----------
ev, dr = find_events(lr, sig, v, vmed, 2.5, 0.8)
ev = ev[ev > 1500]
ok = (ev + 1 < n - 1) & (~gap[ev + 1]) & ~np.isnan(A1440[ev])
ev, dr = ev[ok], dr[ok]
e_i = ev + 1
aE = A1440[ev]
out, xi, xp = sim_directional(o, h, l, c, gap, e_i, dr, 3.0 * aE, 1.5 * aE, 120)
y = (out == 1).astype(int)
month_ev = mid[ev]
print(f'事件总数: {len(ev)} | 月份跨度: {month_ev.min()}~{month_ev.max()} | base pTP={y.mean():.3f}')

LGB_PARAMS = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=31,
                  min_data_in_leaf=40, feature_fraction=0.9, bagging_fraction=0.9,
                  bagging_freq=1, lambda_l1=0.1, max_bin=127, num_threads=2,
                  seed=7, deterministic=True, verbosity=-1)
# 阈值=验证段概率分位数(跨折校准自适应); 网格=选择度
Q_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def exec_sim(mask_ev, p, theta, cooldown=COOLDOWN):
    """无重叠执行: mask_ev为事件行掩码; p与mask内行按序对齐(局部索引).
    返回采纳的局部行号(可索引p与np.where(mask_ev)[0])"""
    rows = np.where(mask_ev)[0]
    taken = []
    last_exit = -10 ** 9
    for i in range(len(rows)):
        r = rows[i]
        if p[i] >= theta and (ev[r] + 1) > last_exit + cooldown:
            taken.append(i)
            last_exit = xi[r]
    return np.array(taken, dtype=int)


def run_fold(mi):
    f = f'{CACHE}/fold_{mi}.json'
    if os.path.exists(f):
        return json.load(open(f))
    t0 = time.time()
    val_start_bar = np.searchsorted(t, starts[mi - 3])
    oos_start_bar = np.searchsorted(t, starts[mi])
    oos_end_bar = np.searchsorted(t, starts[mi + 1])
    core = (month_ev <= mi - 4) & (xi < val_start_bar)
    val = ((month_ev >= mi - 3) & (month_ev < mi)) & (xi < oos_start_bar)
    oos = (month_ev == mi) & (ev < n - 1)
    n_core, n_val, n_oos = int(core.sum()), int(val.sum()), int(oos.sum())
    rec = dict(month=int(mi))
    if n_core < 1000 or n_val < 100 or n_oos < 10:
        rec['skip'] = f'样本不足 core={n_core} val={n_val} oos={n_oos}'
        json.dump(rec, open(f, 'w'))
        return rec
    ds_tr = lgb.Dataset(X[ev[core]], label=y[core])
    ds_va = lgb.Dataset(X[ev[val]], label=y[val], reference=ds_tr)
    m = lgb.train(LGB_PARAMS, ds_tr, num_boost_round=800, valid_sets=[ds_va],
                 callbacks=[lgb.early_stopping(80, verbose=False)])
    p_val = m.predict(X[ev[val]], num_iteration=m.best_iteration)
    p_oos = m.predict(X[ev[oos]], num_iteration=m.best_iteration)
    try:
        auc_va = float(roc_auc_score(y[val], p_val))
    except Exception:
        auc_va = float('nan')
    try:
        auc_oos = float(roc_auc_score(y[oos], p_oos))
    except Exception:
        auc_oos = float('nan')
    # ---- 阈值: val分位数网格, 无重叠净PnL最大化(旧成本) ----
    pnlA_val = dr[val] * (xp[val] - o[ev[val] + 1])   # 毛
    best = (0.0, -1e18, 0)   # (theta_abs, val_net_pnl, n_val_trades)
    for q in Q_GRID:
        th = float(np.quantile(p_val, q))
        tk = exec_sim(val, p_val, th)
        if len(tk) < 30:
            continue
        pnl = pnlA_val[tk].mean() * len(tk) - FIXED_COST_SIDE * 2 * len(tk)
        if pnl > best[1]:
            best = (th, float(pnl), len(tk))
    theta, _, ntv = best
    # ---- OOS执行 ----
    rows_oos = np.where(oos)[0]
    tk = exec_sim(oos, p_oos, theta)
    trades = []
    for r in tk:
        ii = rows_oos[r]
        ee = ev[ii]
        trades.append(dict(
            month=int(mi), ev=int(ee), dir=int(dr[ii]), p=float(p_oos[r]),
            entry_i=int(ee + 1), exit_i=int(xi[ii]), hold=int(xi[ii] - ee - 1),
            outc=int(out[ii]), pnl_old=float(dr[ii] * (xp[ii] - o[ee + 1]) - 2 * FIXED_COST_SIDE),
            pnl_real=float(dr[ii] * (xp[ii] - o[ee + 1]) - 2 * sp_imp[ee + 1]),
            entry_t=int(t[ee + 1]), exit_t=int(t[xi[ii]]),
        ))
    rec.update(theta=theta, ntv=ntv, n_core=n_core, n_val=n_val, n_oos=n_oos,
               auc_va=round(auc_va, 4), auc_oos=round(auc_oos, 4),
               best_iter=int(m.best_iteration), n_trades=len(trades),
               sec=round(time.time() - t0, 1))
    pd.DataFrame(trades).to_csv(f'{CACHE}/trades_{mi}.csv', index=False)
    # OOS全事件概率存档(诊断/安慰剂用)
    pd.DataFrame(dict(ev=ev[oos].astype(int), dir=dr[oos].astype(int), p=p_oos,
                      y=y[oos], out=out[oos], month=int(mi))).to_csv(
        f'{CACHE}/oosprobs_{mi}.csv', index=False)
    json.dump(rec, open(f, 'w'))
    print(f"fold {mi} ({pd.Timestamp(starts[mi], unit='s').strftime('%Y-%m')}): "
          f"θ={theta:.2f} AUC va={rec['auc_va']} oos={rec['auc_oos']} "
          f"trades={rec['n_trades']} val_trades={ntv} [{rec['sec']}s]")
    return rec


if __name__ == '__main__':
    a = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    b = int(sys.argv[2]) if len(sys.argv) > 2 else N_FOLDS
    for k in range(a, min(b, N_FOLDS)):
        r = run_fold(FOLD0 + k)
        if 'skip' in r:
            print('skip', r)
    print('done')
