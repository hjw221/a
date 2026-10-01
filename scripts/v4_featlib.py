"""v4 — M1原生特征库 (从零构建, 与v2管线的v3-34特征零重叠)
所有特征因果: 只用 bar t 及更早数据(bar t 已收盘); sigma60 不含当前根防自含.
方向约定: 交易方向 = sign(lr[t]) (当前1m bar自身的方向), 标签按此方向展开."""
import numpy as np
import pandas as pd


def build_all(d, sp_imp):
    """返回全量特征矩阵 X [n,33] float32 + 特征名列表 + 辅助数组"""
    t, o, h, l, c, v = d['t'], d['o'], d['h'], d['l'], d['c'], d['v']
    n = len(t)
    lr = np.log(c / np.concatenate(([c[0]], c[:-1])))
    lr[0] = 0.0
    pc = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))

    # ---- 基准波动率 ----
    sig = pd.Series(lr).rolling(60, min_periods=60).std().shift(1).to_numpy()
    sig = np.maximum(sig, 1e-7)                       # 1m收益σ60(不含当前根)
    A = pd.Series(tr).rolling(1440, min_periods=1440).mean().to_numpy()   # ATR24h
    A60 = pd.Series(tr).rolling(60, min_periods=60).mean().to_numpy()      # ATR1h
    Asafe = np.where(np.isnan(A) | (A <= 0), np.nan, A)
    sigS = pd.Series(sig)

    def S(x):
        return pd.Series(x)

    feats = {}

    # ==== [1] 事件/bar解剖 ====
    feats['ret1_z'] = lr / sig                                    # 当前1m收益z(即事件强度)
    feats['ret3_z'] = (lr + S(lr).shift(1) + S(lr).shift(2)) / sig
    feats['bar_range_A'] = (h - l) / Asafe
    rng1 = np.where((h - l) > 0, h - l, np.nan)
    feats['bar_body'] = np.abs(c - o) / rng1
    feats['close_pos'] = (c - l) / rng1
    vmed = S(v).rolling(1440, min_periods=1440).median().shift(1).to_numpy()
    feats['v_rel1440'] = v / vmed
    abs_lr = np.abs(lr)
    feats['burst_dom60'] = abs_lr / (abs_lr + S(abs_lr).rolling(60, min_periods=60).sum().shift(1))
    feats['burst_vs4h'] = abs_lr / np.maximum(S(abs_lr).rolling(240, min_periods=240).max().shift(1), 1e-9)

    # ==== [2] 多尺度动量 (z形式, 非ATR归一 — 与v3特征不同) ====
    for k, nm in [(15, 'mom15_z'), (60, 'mom60_z'), (240, 'mom240_z'), (1440, 'mom1440_z')]:
        feats[nm] = S(lr).rolling(k, min_periods=k).sum().to_numpy() / (sig * np.sqrt(k))

    # ==== [3] 订单流代理 (TICKVOL多空结构, M1原生) ====
    up = (lr > 0).astype(float)
    dn = (lr < 0).astype(float)
    for k, nm in [(60, 'tv_updn60'), (240, 'tv_updn240')]:
        su = S(v * up).rolling(k, min_periods=k).sum().to_numpy()
        sd_ = S(v * dn).rolling(k, min_periods=k).sum().to_numpy()
        feats[nm] = np.clip(su / np.maximum(sd_, 1e-9), 0.1, 10.0)
    for k, nm in [(15, 'flow_sign15'), (60, 'flow_sign60')]:
        sv = S(v * np.sign(lr)).rolling(k, min_periods=k).sum().to_numpy()
        tot = S(v).rolling(k, min_periods=k).sum().to_numpy()
        feats[nm] = sv / np.maximum(tot, 1e-9)

    # ==== [4] 突刺密度/复发 (k=2基准点火序列) ====
    z = np.abs(lr) / sig
    ig2 = (z >= 2.0) & (lr != 0)
    ig2s = np.sign(lr) * ig2
    feats['n_burst60'] = S(ig2.astype(float)).rolling(60, min_periods=60).sum().shift(1).to_numpy()
    feats['n_burst1440'] = S(ig2.astype(float)).rolling(1440, min_periods=1440).sum().shift(1).to_numpy()
    feats['ev_cnt240'] = S(ig2.astype(float)).rolling(240, min_periods=240).sum().shift(1).to_numpy()
    ar = np.arange(n, dtype=np.float64)
    for s, nm in [(1, 'since_same'), (-1, 'since_opp')]:
        m = np.where(ig2s == s, ar, np.nan)
        last = pd.Series(m).ffill().to_numpy()
        feats[nm] = np.log1p(ar - last)

    # ==== [5] 日内结构 (服务器日) ====
    day_id = t // 86400
    df1 = pd.DataFrame({'d': day_id, 'o': o, 'h': h, 'l': l, 'c': c, 'v': v,
                        'tp': (h + l + c) / 3.0})
    g = df1.groupby('d')
    day_open = g['o'].transform('first').to_numpy()
    day_hi = g['h'].transform('cummax').to_numpy()
    day_lo = g['l'].transform('cummin').to_numpy()
    cum_pv = g.apply(lambda x: (x['tp'] * x['v']).cumsum() / x['v'].cumsum(),
                     include_groups=False).to_numpy()
    vwap = cum_pv
    dr = (day_hi - day_lo)
    feats['day_ret_A'] = (c - day_open) / Asafe
    feats['day_pos'] = np.where(dr > 0, (c - day_lo) / dr, np.nan)
    feats['vwap_dist_A'] = (c - vwap) / Asafe
    feats['dist_dh_A'] = (day_hi - c) / Asafe
    feats['dist_dl_A'] = (c - day_lo) / Asafe

    # ==== [6] 波动率状态 ====
    feats['atr_ratio_60_1440'] = A60 / Asafe
    hi60 = S(h).rolling(60, min_periods=60).max().to_numpy()
    lo60 = S(l).rolling(60, min_periods=60).min().to_numpy()
    hi240 = S(h).rolling(240, min_periods=240).max().to_numpy()
    lo240 = S(l).rolling(240, min_periods=240).min().to_numpy()
    feats['range_exp_60_240'] = (hi60 - lo60) / np.maximum(hi240 - lo240, 1e-9)

    # ==== [7] 成本/流动性 ====
    feats['spread_A'] = sp_imp / Asafe
    mi = pd.PeriodIndex(pd.to_datetime(t, unit='s'), freq='M')
    mm = pd.Series(sp_imp).groupby(mi).transform('median').to_numpy()
    feats['spread_rel_m'] = sp_imp / mm

    # ==== [8] 时间 ====
    dt = pd.to_datetime(t, unit='s')
    feats['hour'] = dt.hour.to_numpy(float)
    feats['dow'] = dt.dayofweek.to_numpy(float)
    feats['dir'] = np.sign(lr).astype(float)   # 交易方向本身(因果: 当前bar方向)

    names = list(feats.keys())
    X = np.column_stack([np.asarray(feats[k], dtype=np.float64) for k in names]).astype(np.float32)
    return X, names, sig, A, ig2
