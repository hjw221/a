"""阶段1c — 特征工程: 66个候选因子(全部因果, bar t收盘可知, 供t+1开盘入场)。

命名带家族前缀(ret_/mom_/rv_/atr_/rng_/dc_/ma_/ker_/vol_/sess_/day_), 供阶段2家族去重。
全部以 atr60($) 或无量纲方式归一, 跨波动状态可比(2023~2024金价1830->2450)。
因果校验: 截断窗 vs 全窗 在重叠行上必须完全一致。
"""
import json
import numpy as np
import pandas as pd
import config as C
import dataio


def build_features(w: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, v = (w[k] for k in ('open', 'high', 'low', 'close', 'volume'))
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr60 = tr.rolling(60, min_periods=60).mean()
    atr1440 = tr.rolling(1440, min_periods=1440).mean()
    a = atr60.replace(0, np.nan)
    F = pd.DataFrame(index=w.index)

    # ---- 动量族: 多尺度收益(以atr60归一) ----
    for k in (1, 3, 5, 15, 30, 60, 120, 240, 480, 1440):
        F[f'ret_{k}'] = ((c - c.shift(k)) / a).astype('float32')
    r1 = c.diff()
    for k in (5, 15, 60):
        F[f'mom_sign_{k}'] = (np.sign(r1).rolling(k).mean()).astype('float32')

    # ---- 波动族 ----
    for k in (5, 15, 60, 240, 1440):
        F[f'rv_{k}'] = (r1.rolling(k).std() / a).astype('float32')
    F['rv_ratio_15_240'] = (F['rv_15'] / F['rv_240'].replace(0, np.nan)).astype('float32')
    F['rv_ratio_60_1440'] = (F['rv_60'] / F['rv_1440'].replace(0, np.nan)).astype('float32')
    F['atr_lvl'] = atr60.astype('float32')
    F['atr_ratio'] = (atr60 / atr1440).astype('float32')
    F['atr_z'] = ((atr60 - atr60.rolling(1440, min_periods=1440).mean())
                  / atr60.rolling(1440, min_periods=1440).std()).astype('float32')

    # ---- K线微观族 ----
    rng = (h - l).replace(0, np.nan)
    F['rng_body_ratio'] = ((c - o).abs() / rng).astype('float32')
    F['rng_close_pos'] = ((c - l) / rng).astype('float32')
    F['rng_gap_atr'] = ((o - pc) / a).astype('float32')
    F['rng_range_z60'] = (((h - l) - (h - l).rolling(60, min_periods=60).mean())
                          / (h - l).rolling(60, min_periods=60).std()).astype('float32')
    s = np.sign(r1)
    grp = (s != s.shift()).cumsum()
    streak = s.groupby(grp).cumsum()
    F['rng_run'] = streak.clip(-10, 10).astype('float32')

    # ---- Donchian/突破族 ----
    for k in (60, 240, 1440):
        hi_k = h.rolling(k, min_periods=k).max()
        lo_k = l.rolling(k, min_periods=k).min()
        F[f'dc_pos_{k}'] = ((c - lo_k) / (hi_k - lo_k).replace(0, np.nan)).astype('float32')
    hi60 = h.rolling(60, min_periods=60).max()
    lo60 = l.rolling(60, min_periods=60).min()
    F['dc_dist_hi60'] = ((c - hi60) / a).astype('float32')
    F['dc_dist_lo60'] = ((lo60 - c) / a).astype('float32')
    hh240 = h.rolling(240, min_periods=240).max().shift(1)
    ll240 = l.rolling(240, min_periods=240).min().shift(1)
    F['dc_bo_up240'] = ((c - hh240).clip(lower=0) / a).astype('float32')
    F['dc_bo_dn240'] = ((ll240 - c).clip(lower=0) / a).astype('float32')
    F['dc_new_hi240'] = (c >= hh240).astype('int8')
    F['dc_new_lo240'] = (c <= ll240).astype('int8')

    # ---- 均线/统计套利族 ----
    for k in (20, 60, 240, 1440):
        ema = c.ewm(span=k, min_periods=k, adjust=False).mean()
        F[f'ma_ema{k}_dist'] = ((c - ema) / a).astype('float32')
    e20 = c.ewm(span=20, min_periods=20, adjust=False).mean()
    e60 = c.ewm(span=60, min_periods=60, adjust=False).mean()
    e240 = c.ewm(span=240, min_periods=240, adjust=False).mean()
    F['ma_ema20_60'] = ((e20 - e60) / a).astype('float32')
    F['ma_ema60_240'] = ((e60 - e240) / a).astype('float32')
    sma60 = c.rolling(60, min_periods=60).mean()
    std60 = c.rolling(60, min_periods=60).std()
    F['ma_z60'] = ((c - sma60) / std60).astype('float32')
    F['ma_bb_pos60'] = ((c - sma60) / (2 * std60)).astype('float32')
    F['ma_bb_width60'] = ((4 * std60) / a).astype('float32')
    for k, tag in ((30, '30'), (120, '120')):
        num = (c - c.shift(k)).abs()
        den = r1.abs().rolling(k, min_periods=k).sum()
        F[f'ker_{tag}'] = (num / den.replace(0, np.nan)).astype('float32')

    # ---- 量能族 ----
    F['vol_z60'] = ((v - v.rolling(60, min_periods=60).mean())
                    / v.rolling(60, min_periods=60).std()).astype('float32')
    F['vol_z240'] = ((v - v.rolling(240, min_periods=240).mean())
                     / v.rolling(240, min_periods=240).std()).astype('float32')
    F['vol_ratio60'] = (v / v.rolling(60, min_periods=60).mean()).astype('float32')
    F['vol_trend'] = (v.rolling(15, min_periods=15).mean()
                      / v.rolling(240, min_periods=240).mean()).astype('float32')
    F['vol_vp_corr60'] = (r1.abs().rolling(60, min_periods=60)
                          .corr(v)).astype('float32')

    # ---- 时段族(服务器时间EET) ----
    hr = w.index.hour
    F['sess_h_sin'] = np.sin(2 * np.pi * hr / 24).astype('float32')
    F['sess_h_cos'] = np.cos(2 * np.pi * hr / 24).astype('float32')
    F['sess_dow'] = w.index.dayofweek.astype('int8')
    F['sess_asia'] = hr.isin(range(3, 9)).astype('int8')
    F['sess_london'] = hr.isin(range(9, 15)).astype('int8')
    F['sess_ny'] = hr.isin(range(15, 22)).astype('int8')
    F['sess_overlap'] = hr.isin(range(15, 17)).astype('int8')
    d = w.index.normalize()
    mins = w.index.hour * 60 + w.index.minute
    day_open_min = pd.Series(mins, index=w.index).groupby(d).transform('min')
    F['sess_mins_from_open'] = (mins - day_open_min.values).astype('float32')

    # ---- 日内累积族(当日已走部分, 因果) ----
    day_o = o.groupby(d).transform('first')
    day_h = h.groupby(d).cummax()
    day_l = l.groupby(d).cummin()
    day_rng = (day_h - day_l).replace(0, np.nan)
    F['day_pos'] = ((c - day_l) / day_rng).astype('float32')
    F['day_vs_open'] = ((c - day_o) / a).astype('float32')
    F['day_dist_hi'] = ((day_h - c) / a).astype('float32')
    F['day_dist_lo'] = ((c - day_l) / a).astype('float32')
    tp_ = (h + l + c) / 3.0
    cum_pv = (tp_ * v).groupby(d).cumsum()
    cum_v = v.groupby(d).cumsum().replace(0, np.nan)
    F['day_vwap_dist'] = ((c - cum_pv / cum_v) / a).astype('float32')

    return F


def main():
    df = dataio.load_raw()
    w_full = df.loc[C.WARMUP_START:C.OUTCOME_PAD_END]
    w_cut = df.loc[C.WARMUP_START:'2024-04-30 23:59:59']

    print('构建全窗特征...', len(w_full), 'bars')
    F_full = build_features(w_full)
    F_full.to_pickle(C.CACHE / 'feats.pkl')

    print('构建截断窗特征(因果校验)...', len(w_cut), 'bars')
    F_cut = build_features(w_cut)
    common = F_cut.index
    A, B = F_cut, F_full.loc[common]
    bad = []
    for col in A.columns:
        x, y = A[col].values, B[col].values
        eq = (x == y) | (np.isnan(x) & np.isnan(y))
        if not eq.all():
            bad.append((col, int((~eq).sum())))
    report = {'n_features': int(F_full.shape[1]),
              'n_bars': int(len(F_full)),
              'causality_violations': bad,
              'nan_ratio_top': {k: round(float(v), 4) for k, v in
                                F_full.isna().mean().nlargest(5).items()}}
    with open(C.RESULTS / 'stage1_feats.json', 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    assert len(bad) == 0, f'因果性校验失败: {bad[:5]}'


if __name__ == '__main__':
    main()
