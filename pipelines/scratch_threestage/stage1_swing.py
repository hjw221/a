"""阶段1b — 波段画像与几何标定(只用2023, 尾部按H purge, 不窥视2024H1)。

1) 三障碍随机净期望网格: TP=kE*ATR60, SL=TP/PLR, 超时H -> 冻结几何 (kE,PLR,H)
2) 可吃地图: hour x ATR-regime 单元格的随机期望/MFE画像 -> 可交易宇宙
3) ML可吃性画像分类器(小特征集, 2023H1训练/2023H2验证): 哪些数据有用(特征重要性)
4) 冻结几何在全工作窗(2022-12~2024-07-15)生成标签/结局oracle, 供阶段2/3/消融使用

口径: entry=next-bar open, 往返成本$0.06, 同bar双触=保守记SL。
"""
import json
import numpy as np
import pandas as pd
from numba import njit
import lightgbm as lgb
import config as C
import dataio


@njit(cache=True)
def barrier_scan(op, hi, lo, cl, tp, sl, H):
    """对每个t: entry=open[t+1]; 上障碍=entry+tp[t], 下障碍=entry-sl[t]。
    返回 long/short 的 code(1=TP先, -1=SL先/同bar双触, 0=超时, 2=不可行),
    exit_idx(含), gross pnl($, 未扣成本)。"""
    n = len(op)
    codeL = np.full(n, 2, np.int8)
    codeS = np.full(n, 2, np.int8)
    exitL = np.full(n, -1, np.int64)
    exitS = np.full(n, -1, np.int64)
    pnlL = np.full(n, np.nan)
    pnlS = np.full(n, np.nan)
    for t in range(n):
        end = t + 1 + H
        if end > n:
            break
        e = op[t + 1]
        up = e + tp[t]
        dn = e - sl[t]
        # --- long: TP=up, SL=dn ---
        res = 0
        ex = end - 1
        for j in range(t + 1, end):
            hu = hi[j] >= up
            hd = lo[j] <= dn
            if hu or hd:
                ex = j
                if hu and hd:
                    res = -1          # 同bar双触, 保守记SL
                elif hu:
                    res = 1
                else:
                    res = -1
                break
        if res == 0:
            pnlL[t] = cl[end - 1] - e
        elif res == 1:
            pnlL[t] = up - e
        else:
            pnlL[t] = dn - e
        codeL[t] = res
        exitL[t] = ex
        # --- short: TP=dn, SL=up ---
        res = 0
        ex = end - 1
        for j in range(t + 1, end):
            hu = hi[j] >= up
            hd = lo[j] <= dn
            if hu or hd:
                ex = j
                if hu and hd:
                    res = -1
                elif hd:
                    res = 1
                else:
                    res = -1
                break
        if res == 0:
            pnlS[t] = e - cl[end - 1]
        elif res == 1:
            pnlS[t] = e - dn
        else:
            pnlS[t] = e - up
        codeS[t] = res
        exitS[t] = ex
    return codeL, exitL, pnlL, codeS, exitS, pnlS


@njit(cache=True)
def mfe_scan(op, hi, lo, H):
    n = len(op)
    mfeL = np.full(n, np.nan)
    mfeS = np.full(n, np.nan)
    for t in range(n):
        end = t + 1 + H
        if end > n:
            break
        e = op[t + 1]
        mx = hi[t + 1]
        mn = lo[t + 1]
        for j in range(t + 2, end):
            if hi[j] > mx:
                mx = hi[j]
            if lo[j] < mn:
                mn = lo[j]
        mfeL[t] = mx - e
        mfeS[t] = e - mn
    return mfeL, mfeS


def atr_series(w: pd.DataFrame, win: int = C.ATR_WIN) -> pd.Series:
    pc = w['close'].shift(1)
    tr = pd.concat([w['high'] - w['low'],
                    (w['high'] - pc).abs(),
                    (w['low'] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(win, min_periods=win).mean()


def run_grid(w23: pd.DataFrame, atr23: np.ndarray):
    op = w23['open'].values; hi = w23['high'].values
    lo = w23['low'].values; cl = w23['close'].values
    rows = []
    for kE in C.KE_GRID:
        for plr in C.PLR_GRID:
            tp = (kE * atr23)
            sl = tp / plr
            for H in C.H_GRID:
                codeL, exitL, pnlL, codeS, exitS, pnlS = barrier_scan(
                    op, hi, lo, cl, tp, sl, H)
                ok = codeL != 2
                nl = int(ok.sum())
                pos = np.arange(len(ok))
                dur = (exitL[ok] - (pos[ok] + 1)).mean()
                hl = float((codeL[ok] == 1).mean())
                hs = float((codeS[ok] == 1).mean())
                el = float(np.nanmean(pnlL[ok]) - C.COST_RT)
                es = float(np.nanmean(pnlS[ok]) - C.COST_RT)
                to = float((codeL[ok] == 0).mean())
                tp_med = float(np.nanmedian(tp[ok]))
                sl_med = tp_med / plr
                # 模型需要补的命中率缺口(经验): hit* - hit_base
                liftL = (sl_med + C.COST_RT) / (tp_med + sl_med) - hl
                liftS = (tp_med + C.COST_RT) / (tp_med + sl_med) - hs
                rows.append(dict(kE=kE, plr=plr, H=H, n=nl,
                                 hitL=round(hl, 4), hitS=round(hs, 4),
                                 expL=round(el, 4), expS=round(es, 4),
                                 expAvg=round((el + es) / 2, 4),
                                 timeout=round(to, 4),
                                 dur_min=round(float(dur), 1),
                                 tp_usd_med=round(tp_med, 3),
                                 liftL_pp=round(liftL * 100, 2),
                                 liftS_pp=round(liftS * 100, 2),
                                 liftAvg_pp=round((liftL + liftS) / 2 * 100, 2)))
    return pd.DataFrame(rows)


def main():
    df = dataio.load_raw()
    w = df.loc[C.WARMUP_START:C.OUTCOME_PAD_END]
    atr = atr_series(w)
    w23 = w.loc[C.DATA_START:C.TRAIN_END]
    atr23_all = atr.loc[C.DATA_START:C.TRAIN_END]
    print('ATR60($) 2023 分位: ',
          {q: round(float(atr23_all.quantile(q)), 3) for q in (.1, .5, .9)})

    # ---------- 1) 网格标定(仅2023, 尾部H根自动不可行=被purge) ----------
    grid = run_grid(w23, atr23_all.values)
    grid.to_csv(C.RESULTS / 'swing_grid_2023.csv', index=False)
    print(grid.to_string(index=False))

    # 选型逻辑(事前声明, 全部由2023数据计算):
    #   (a) PLR 必须使两侧机械基率都不低于0.30 -> PLR<=2.33
    #   (b) TP中位数 >= $0.50 (成本拖累<=12%)
    #   (c) 超时率 <= 0.1% (H足够覆盖barrier解析)
    #   (d) 在以上约束内最小化 liftAvg_pp (模型需要补的命中率缺口最小)
    c = grid[(grid['plr'] <= 2.33) & (grid['tp_usd_med'] >= 0.50)
             & (grid['timeout'] <= 0.001)]
    if len(c) == 0:
        c = grid[(grid['plr'] <= 2.33) & (grid['tp_usd_med'] >= 0.40)]
    if len(c) == 0:
        c = grid
    best = c.sort_values('liftAvg_pp').iloc[0]
    print('[几何选型] 约束: PLR<=2.33 & TP_med>=$0.50 & timeout<=0.1% -> 最小化模型缺口')
    kE, plr, H = float(best['kE']), float(best['plr']), int(best['H'])
    print('选中几何: kE=%.1f PLR=%.1f H=%d' % (kE, plr, H))
    print('  随机(无模型)口径: hitL=%.3f hitS=%.3f expL=$%.3f expS=$%.3f 超时率=%.3f'
          % (best['hitL'], best['hitS'], best['expL'], best['expS'], best['timeout']))

    # ---------- 2) 可吃地图 + 宇宙(仅2023, 波段厚度门槛) ----------
    # 可吃窗口 HEAT=60min(≈5x平均解析11.3min, 与实际持仓寿命匹配), 而非H=360的timeout上限。
    # 宇宙规则(事前声明): hour x regime 单元格的中位MFE(扣成本) >= 10 x 往返成本($0.60)
    #                    且样本 >= 3000。 选"波段厚度"而非单元格盈亏(避免噪声cherry-pick)。
    HEAT = 60
    tp23 = kE * atr23_all.values
    sl23 = tp23 / plr
    codeL, exitL, pnlL, codeS, exitS, pnlS = barrier_scan(
        w23['open'].values, w23['high'].values, w23['low'].values,
        w23['close'].values, tp23, sl23, H)
    ok23 = codeL != 2
    netL = np.where(ok23, pnlL - C.COST_RT, np.nan)
    netS = np.where(ok23, pnlS - C.COST_RT, np.nan)
    mfeL, mfeS = mfe_scan(w23['open'].values, w23['high'].values,
                          w23['low'].values, HEAT)
    mfe_best = np.fmax(mfeL, mfeS)
    ok_mfe = ~np.isnan(mfe_best)
    cut1, cut2 = atr23_all.quantile(C.REGIME_Q[0]), atr23_all.quantile(C.REGIME_Q[1])
    reg23 = np.digitize(atr23_all.values, [cut1, cut2])   # 0=low 1=mid 2=high
    hour23 = w23.index.hour.values
    cell = pd.DataFrame({'hour': hour23, 'reg': reg23,
                         'exp': (netL + netS) / 2.0,
                         'mfe_net': mfe_best - C.COST_RT,
                         'atr': atr23_all.values})
    grp = cell.dropna(subset=['mfe_net']).groupby(['hour', 'reg']).agg(
        n=('mfe_net', 'size'),
        exp=('exp', 'mean'),
        mfe_med=('mfe_net', 'median'),
        p_eat=('mfe_net', lambda s: float((s >= 0.5).mean())))
    MFE_GATE = 10 * C.COST_RT   # $0.60
    MIN_CELL_N = 3000
    keep = grp[(grp['mfe_med'] >= MFE_GATE) & (grp['n'] >= MIN_CELL_N)]
    keep_idx = set(keep.index)
    universe23 = np.array([(h, r) in keep_idx for h, r in zip(hour23, reg23)])
    print('宇宙: 保留 %d/%d 单元格, 覆盖 %.1f%% 的2023可用bar (门槛: 中位MFE_net(%dmin)>=$%.2f)'
          % (len(keep), len(grp), 100 * universe23[ok23].mean(), HEAT, MFE_GATE))
    grp.to_csv(C.RESULTS / 'eatability_map_2023.csv')

    # ---------- 3) ML可吃性画像分类器(哪些数据有用) ----------
    feat = pd.DataFrame(index=w23.index)
    a60 = atr23_all
    a1440 = atr.rolling(1440, min_periods=1440).mean().loc[w23.index]
    feat['h_sin'] = np.sin(2 * np.pi * hour23 / 24)
    feat['h_cos'] = np.cos(2 * np.pi * hour23 / 24)
    feat['dow'] = w23.index.dayofweek
    feat['atr60'] = a60.values
    feat['atr_ratio'] = (a60 / a1440).values
    feat['rv15'] = (w23['close'].diff() ** 2).rolling(15).mean().pow(0.5).values
    feat['vol_z'] = ((w23['volume'] - w23['volume'].rolling(240).mean())
                     / w23['volume'].rolling(240).std()).values
    feat['ret240'] = (w23['close'] - w23['close'].shift(240)).values
    feat['range30_med'] = (w23['high'] - w23['low']).rolling(30).median().values
    y_eat = (mfe_best - C.COST_RT) >= MFE_GATE      # 与宇宙门槛同口径($0.60, 60min窗口)
    _bmask = ok_mfe & universe23
    base = float(y_eat[_bmask].mean()) if _bmask.any() else 0.5
    if not (0.15 <= base <= 0.85):                   # 基率退化 -> TP尺度: 交易需要的波段在60min内存在
        y_eat = (mfe_best - C.COST_RT) >= (kE * a60.values)
        print(f'[y_eat] 绝对门槛基率{base:.2f}退化, 改用TP尺度(kE*ATR60={kE}x)门槛')
    split = np.asarray(w23.index < pd.Timestamp('2023-07-01'))
    pos = np.arange(len(w23))
    purge = pos < len(w23) - H
    m_tr = split & purge & ok_mfe & universe23 & np.isfinite(feat['rv15'].values)
    m_va = (~split) & ok_mfe & universe23 & np.isfinite(feat['rv15'].values)
    Xc = feat.fillna(0.0)
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                             min_child_samples=300, random_state=C.SEED,
                             n_jobs=C.NJOBS, verbose=-1)
    clf.fit(Xc[m_tr], y_eat[m_tr], eval_set=[(Xc[m_va], y_eat[m_va])],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    from sklearn.metrics import roc_auc_score
    auc_eat = float(roc_auc_score(y_eat[m_va], clf.predict_proba(Xc[m_va])[:, 1]))
    imp = sorted(zip(Xc.columns, clf.feature_importances_.round(1)),
                 key=lambda kv: -kv[1])
    print('可吃性分类器 2023H2 AUC = %.4f | y_eat基率=%.3f' % (auc_eat, y_eat[m_va].mean()))
    print('特征重要性(画像):', imp)

    # ---------- 4) 冻结几何 -> 全工作窗标签/结局oracle ----------
    tp_w = kE * atr.values
    sl_w = tp_w / plr
    codeLw, exitLw, pnlLw, codeSw, exitSw, pnlSw = barrier_scan(
        w['open'].values, w['high'].values, w['low'].values,
        w['close'].values, tp_w, sl_w, H)
    mfeLw, mfeSw = mfe_scan(w['open'].values, w['high'].values, w['low'].values, HEAT)
    labels = pd.DataFrame({
        'atr60': atr.values, 'regime': np.digitize(atr.values, [cut1, cut2]),
        'codeL': codeLw, 'exitL': exitLw, 'pnlL_gross': pnlLw,
        'codeS': codeSw, 'exitS': exitSw, 'pnlS_gross': pnlSw,
        'mfeL60': mfeLw, 'mfeS60': mfeSw,
    }, index=w.index)
    # 全窗宇宙: hour x regime 单元格规则来自2023
    hour_w = w.index.hour.values
    universe_w = np.array([(h, r) in keep_idx
                           for h, r in zip(hour_w, labels['regime'].values)])
    labels['universe'] = universe_w
    labels.to_pickle(C.CACHE / 'labels.pkl')

    geom = dict(kE=kE, plr=plr, H=H, atr_cutpoints=[float(cut1), float(cut2)],
                random=dict(hitL=float(best['hitL']), hitS=float(best['hitS']),
                            expL=float(best['expL']), expS=float(best['expS']),
                            timeout=float(best['timeout'])),
                universe_cells=[f'{h}_{r}' for h, r in sorted(keep_idx)],
                universe_rule=f'median MFE_net(over {HEAT}min) >= ${MFE_GATE:.2f} (=10xRT cost) & n>=3000',
                universe_cells_kept=int(len(keep)), universe_cells_total=int(len(grp)),
                universe_cover_2023_pct=round(100 * float(universe23[ok23].mean()), 1),
                eat_classifier=dict(auc_2023H2=round(auc_eat, 4),
                                    base_rate=round(float(y_eat[m_va].mean()), 3),
                                    importance={k: float(v) for k, v in imp}),
                atr60_2023_quantiles={str(q): round(float(atr23_all.quantile(q)), 3)
                                       for q in (.1, .5, .9)})
    with open(C.CACHE / 'geometry.json', 'w') as f:
        json.dump(geom, f, indent=2)
    print('geometry.json / labels.pkl 已落盘')


if __name__ == '__main__':
    main()
