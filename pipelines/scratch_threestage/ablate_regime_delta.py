"""② 消融 — 波动Regime门控 x δ收紧 (仅2024H1内部验证段, OOS绝不触碰)。

信号定义(因yL与yS近互补, 方向本质一维):
  z = pL - pS  (方向分数, >0看多, <0看空)
  多头信号: z >= Q_z(1-d) ; 空头信号: z <= Q_z(d)   [Q_z为2023拟合段分数分布, 冻结]
  d = 中部不交易带宽度: d=0.50 全交易(各半), d=0.02 只交易两端各2%极值
Regime: ATR60三分位(切点2023冻结): low/mid/high; 门控变体见 REGIME_VARIANTS。
模拟: bar t收盘决策, t+1开盘入场, 冻结几何(TP/SL/H), 一次一仓, 出场后冷却1根,
  往返成本$0.06, 结局来自全窗oracle(含7月pad补完6月末笔)。
每个配置同时输出"随机方向孪生"(同bar同几何, 方向随机) -> 模型-随机 = 纯方向alpha。
"""
import json
import numpy as np
import pandas as pd
import config as C

DELTAS = [0.50, 0.40, 0.30, 0.20, 0.12, 0.08, 0.05, 0.02, 0.01]
REGIME_VARIANTS = {'all': None, 'low': [0], 'no_low': [1, 2],
                   'mid': [1], 'mid_high': [1, 2], 'high': [2], 'wings': [0, 2]}
RNG_SEED = 7


def simulate(sigL, sigS, elig, exitL, exitS, pnlL, pnlS, random_side=False, seed=RNG_SEED):
    n = len(sigL)
    rng = np.random.default_rng(seed)
    i = 0
    tr = []
    while i < n:
        if elig[i]:
            side = 0
            if sigL[i] and sigS[i]:
                side = 0                       # 不可能(带不相交), 防御
            elif sigL[i]:
                side = 1
            elif sigS[i]:
                side = -1
            if side != 0 and random_side:
                side = 1 if rng.random() < 0.5 else -1
            if side == 1:
                ex, g = exitL[i], pnlL[i]
            elif side == -1:
                ex, g = exitS[i], pnlS[i]
            else:
                ex = -1
            if side != 0 and ex > i:
                tr.append((i, side, g - C.COST_RT))
                i = ex + 2                     # 出场后冷却1根
                continue
        i += 1
    return tr


def metrics(tr, n_days):
    if not tr:
        return dict(n=0, per_day=0.0, hit=np.nan, avg=np.nan, tot=0.0,
                    pf=np.nan, maxdd=np.nan, t=np.nan, be_cost=np.nan,
                    nL=0, nS=0, avgL=np.nan, avgS=np.nan)
    a = np.array([t[2] for t in tr])
    sides = np.array([t[1] for t in tr])
    wins, losses = a[a > 0], a[a <= 0]
    cum = np.cumsum(a)
    dd = float((np.maximum.accumulate(cum) - cum).max()) if len(cum) else 0.0
    t_stat = float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))) if len(a) > 1 else np.nan
    aL, aS = a[sides == 1], a[sides == -1]
    return dict(
        n=len(a), per_day=round(len(a) / n_days, 1),
        hit=round(float((a > 0).mean()), 3),
        avg=round(float(a.mean()), 4),
        tot=round(float(a.sum()), 2),
        pf=round(float(wins.sum() / -losses.sum()), 3) if len(losses) and losses.sum() < 0 else np.nan,
        maxdd=round(dd, 2), t=round(t_stat, 2),
        be_cost=round(float(a.mean() + C.COST_RT), 4),   # 盈亏平衡往返成本
        nL=int((sides == 1).sum()), nS=int((sides == -1).sum()),
        avgL=round(float(aL.mean()), 4) if len(aL) else np.nan,
        avgS=round(float(aS.mean()), 4) if len(aS) else np.nan)


def main():
    L = pd.read_pickle(C.CACHE / 'labels.pkl')
    Q = pd.read_pickle(C.CACHE / 'quality.pkl')
    sv = pd.read_pickle(C.CACHE / 'scores_val.pkl')
    sf = pd.read_pickle(C.CACHE / 'scores_fit.pkl')
    geom = json.load(open(C.CACHE / 'geometry.json'))

    # 验证段窗口(含入场可行判据), 全窗数组按位置对齐
    va_idx = L.loc[C.VAL_START:C.VAL_END].index
    pos_of = pd.Series(np.arange(len(L)), index=L.index)
    p0 = int(pos_of.loc[va_idx[0]])
    p1 = int(pos_of.loc[va_idx[-1]]) + 1
    sl = slice(p0, p1)

    exitL = L['exitL'].values[sl].astype(np.int64) - p0   # 全窗坐标 -> 验证子窗坐标
    exitS = L['exitS'].values[sl].astype(np.int64) - p0
    pnlL = L['pnlL_gross'].values[sl]; pnlS = L['pnlS_gross'].values[sl]
    feasible = (L['codeL'].values[sl] != 2)
    clean = Q['clean'].values[sl]
    universe = L['universe'].values[sl]
    regime = L['regime'].values[sl]
    n_days = (va_idx[-1] - va_idx[0]).days + 1

    # z分数: 验证段 + 拟合段(冻结分位)
    zv = (sv['pL'] - sv['pS']).reindex(va_idx)
    zf = (sf['pL'] - sf['pS']).values
    has_z = zv.notna().values
    zv = zv.fillna(0.0).values

    # 冻结阈值表(2023拟合段z分布)
    thr = {}
    for d in DELTAS:
        thr[d] = (float(np.quantile(zf, 1 - d)), float(np.quantile(zf, d)))

    rows = []
    for rv, regs in REGIME_VARIANTS.items():
        reg_ok = np.ones(len(va_idx), bool) if regs is None else np.isin(regime, regs)
        for d in DELTAS:
            th_hi, th_lo = thr[d]
            sigL = has_z & (zv > th_hi)
            sigS = has_z & (zv < th_lo)
            for mode in ('model', 'random'):
                tr = simulate(sigL, sigS, feasible & clean & universe & reg_ok,
                              exitL, exitS, pnlL, pnlS, random_side=(mode == 'random'))
                m = metrics(tr, n_days)
                rows.append(dict(regime=rv, delta=d, side=mode, **m))
    R = pd.DataFrame(rows)
    R.to_csv(C.RESULTS / 'ablation_regime_delta.csv', index=False)

    # ---- 打印: 模型表 + 随机对照表 + 方向alpha ----
    for mode in ('model', 'random'):
        print(f'\n===== {mode} =====')
        piv = R[R['side'] == mode].pivot_table(
            index='regime', columns='delta',
            values=['n', 'hit', 'avg', 'tot', 't'], aggfunc='first')
        print(piv.to_string())

    M = R[R['side'] == 'model'].set_index(['regime', 'delta'])
    Rd = R[R['side'] == 'random'].set_index(['regime', 'delta'])
    alpha = (M['avg'] - Rd['avg']).rename('dir_alpha').to_frame()
    alpha['n_model'] = M['n']; alpha['t_model'] = M['t']
    print('\n===== 方向alpha (模型avg - 随机avg, $/笔) =====')
    print(alpha.pivot_table(index='regime', columns='delta',
                            values='dir_alpha').to_string())

    # ---- 最佳单元格(按t排序) + 多空拆分 ----
    Mo = R[R['side'] == 'model'].copy()
    Mo = Mo[Mo['n'] >= 100]
    top = Mo.sort_values('t', ascending=False).head(10)
    print('\n===== 模型侧最佳单元格(n>=100, 按t排序) =====')
    print(top[['regime', 'delta', 'n', 'nL', 'nS', 'hit', 'avg', 'avgL', 'avgS',
               'tot', 't', 'be_cost', 'per_day']].to_string(index=False))

    out = {'n_days': n_days, 'thresholds_z': {str(d): thr[d] for d in DELTAS},
           'regime_bar_share': {rv: float((np.isin(regime, regs) if regs is not None
                                           else np.ones(len(va_idx))).mean())
                                for rv, regs in REGIME_VARIANTS.items()},
           'rows': R.to_dict(orient='records')}
    with open(C.RESULTS / 'ablation_regime_delta.json', 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print('\nablation_regime_delta.csv/json 已落盘')


if __name__ == '__main__':
    main()
