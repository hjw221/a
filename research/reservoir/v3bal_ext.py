#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v3bal_ext.py — v3bal_ens 扩展实验: 能量条件化障碍几何
======================================================
用户指令: "回头再看看v3bal_ens能不能再扩展一下，提升一下胜率但不压缩交易次数"

设计约束: 1657 笔入场一笔不删 (入场时刻/方向/信号全部原样), 只改出场障碍几何。
能量来源: R2 冠军预测流 R̂ (champ_s42.npz, 决策时刻可用的最新值 — 与 r2_linkage 同口径)。

臂:
  A0  基线复现     : 原几何 tp=3.0×ATR, sl=1.1429×ATR, horizon=360 (必须复现 wr 0.302 / +$740)
  T系列 TP条件化   : 高能(R̂≥1.3) TP放宽×1.4 / 低能(R̂<0.8) TP收紧×0.75
  S系列 SL条件化   : 高能 SL放宽×1.4 / 低能 SL收紧×0.8
  X系列 组合
评估: n(必须=1657) / 胜率 / PLR / 总PnL / 笔均 / maxDD — 胜率↑ 且 PnL 不降才算成功。
"""
import numpy as np
import pandas as pd

TRADES = '/home/z/my-project/remote-ops-record/r2_20261003/trades_v3bal_ens_FIXED.csv'
NPZ = '/tmp/r2arms/champ_s42.npz'
DATA = '/home/z/my-project/research-lab/data.csv'
OUT = '/home/z/my-project/remote-ops-record/r2_20261003/v3bal_ext_results.json'

TP_MULT, SL_MULT, HZN = 3.0, 1.1429, 360
COST = 0.16  # 从交易流推断的每笔点差成本 (中位)


def load_m1():
    df = pd.read_csv(DATA, sep='\t')
    df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread']
    df['dt'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y.%m.%d %H:%M:%S')
    df = df.set_index('dt').sort_index()
    br = df.high - df.low
    rng_z = (br - br.rolling(288).mean()) / br.rolling(288).std()
    cdf = df[~(((df.tickvol <= 5) & (br < 0.01)) | (rng_z.abs() > 15))]
    m5 = (cdf.resample('5min', label='left', closed='left')
          .agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna(subset=['open']))
    tr = np.maximum(m5.high - m5.low,
                    np.maximum((m5.high - m5.close.shift(1)).abs(), (m5.low - m5.close.shift(1)).abs()))
    m5['atr'] = tr.rolling(288, min_periods=288).mean()
    m1 = cdf[['open', 'high', 'low', 'close']].copy()
    return m1, m5


def resim(trade, m1v, m5, tp_f, sl_f, hzn_ext=0):
    """重演单笔: 返回 (outcome, pnl). trade 行 + tp/sl 乘数函数(接收R̂)."""
    R = trade['R']
    sig = trade['signal_time']
    # ATR at signal M5 bar (open = sig-5min)
    m5i = m5.index.get_indexer([sig - pd.Timedelta(minutes=5)], method='pad')[0]
    a = m5['atr'].iloc[m5i]
    if not np.isfinite(a) or a <= 0:
        return trade['outcome'], trade['pnl']  # 无法重演, 保留原值
    tp = TP_MULT * a * tp_f(R)
    sl = max(SL_MULT * a * sl_f(R), 0.30, 3 * COST)
    et = trade['entry_time'].to_datetime64()
    e = m1v['idx'].searchsorted(et, side='left')
    if e >= len(m1v['idx']) or m1v['idx'][e] != et:
        return trade['outcome'], trade['pnl']
    entry = m1v['o'][e]
    d = 1 if trade['dir'] == 'long' else -1
    end = min(e + HZN + hzn_ext, len(m1v['o']) - 1)
    if d == 1:
        l_tp, l_sl = entry + tp, entry - sl
        for j in range(e + 1, end + 1):
            if m1v['l'][j] <= l_sl:
                return 'SL', -(entry - l_sl) - COST
            if m1v['h'][j] >= l_tp:
                return 'TP', (l_tp - entry) - COST
    else:
        s_tp, s_sl = entry - tp, entry + sl
        for j in range(e + 1, end + 1):
            if m1v['h'][j] >= s_sl:
                return 'SL', -(s_sl - entry) - COST
            if m1v['l'][j] <= s_tp:
                return 'TP', (entry - s_tp) - COST
    return 'TIMEOUT', d * (m1v['c'][end] - entry) - COST


def evaluate(tr, m1v, m5, tp_f, sl_f, hzn_ext=0, label=''):
    outs, pnls = [], []
    for _, t in tr.iterrows():
        o, p = resim(t, m1v, m5, tp_f, sl_f, hzn_ext)
        outs.append(o)
        pnls.append(p)
    pnls = np.array(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    eq = np.cumsum(pnls)
    dd = float((eq - np.maximum.accumulate(eq)).min())
    return dict(
        label=label, n=int(len(pnls)), wr=round(float((pnls > 0).mean()), 3),
        plr=round(float(wins.sum() / max(-losses.sum(), 1e-9)), 2),
        pnl=round(float(pnls.sum()), 1), avg=round(float(pnls.mean()), 3),
        maxdd=round(dd, 1),
        tp_n=int(sum(1 for o in outs if o == 'TP')),
        sl_n=int(sum(1 for o in outs if o == 'SL')),
        to_n=int(sum(1 for o in outs if o == 'TIMEOUT')),
    )


def main():
    print('== load M1/M5 ==', flush=True)
    m1, m5 = load_m1()
    m1v = dict(idx=m1.index.values, o=m1['open'].to_numpy(float), h=m1['high'].to_numpy(float),
               l=m1['low'].to_numpy(float), c=m1['close'].to_numpy(float))
    print(f'M1 {len(m1v["idx"])} bars, M5 {len(m5)} bars', flush=True)

    tr = pd.read_csv(TRADES)
    tr['signal_time'] = pd.to_datetime(tr['signal_time'])
    tr['entry_time'] = pd.to_datetime(tr['entry_time'])

    # R̂ at signal (决策时刻可用的最新预测 — avail<=signal)
    z = np.load(NPZ)
    ts15 = pd.to_datetime(z['ts'], unit='ns')
    avail = (ts15 + pd.Timedelta(minutes=15)).to_numpy()
    R = np.expm1(np.clip(z['yhat'][:, 0], 0, 3))
    sig_ns = tr['signal_time'].to_numpy().astype('datetime64[ns]').astype(np.int64)
    idx = np.searchsorted(avail.astype(np.int64), sig_ns, side='right') - 1
    tr['R'] = np.where(idx >= 0, R[np.maximum(idx, 0)], np.nan)
    print(f"R̂ 覆盖: {np.isfinite(tr['R']).mean()*100:.1f}%  分位: "
          f"{np.nanpercentile(tr['R'], [10,33,50,66,90]).round(2)}", flush=True)
    tr['R'] = tr['R'].fillna(1.0)

    # ---- A0 基线复现 ----
    one = lambda R: 1.0
    a0 = evaluate(tr, m1v, m5, one, one, 0, 'A0 基线复现(原几何)')
    print('A0:', a0, flush=True)
    rec_wr, rec_pnl = 0.302, 740.0
    ok = a0['n'] == 1657 and abs(a0['wr'] - rec_wr) < 0.03 and abs(a0['pnl'] - rec_pnl) < 60
    print(f'基线复现判定: {"OK" if ok else "MISMATCH"} (档案 wr {rec_wr} / +${rec_pnl})', flush=True)

    # ---- 条件化臂 ----
    def band(R):
        return 0 if R < 0.8 else (2 if R >= 1.3 else 1)
    band_n = tr['R'].map(band).value_counts().to_dict()
    print('能量分带笔数:', {f'{"低" if k==0 else "中" if k==1 else "高"}': v for k, v in sorted(band_n.items())}, flush=True)

    arms = []
    arms.append(('T1 高能TP×1.4', lambda R: 1.4 if R >= 1.3 else 1.0, one))
    arms.append(('T2 低能TP×0.75', lambda R: 0.75 if R < 0.8 else 1.0, one))
    arms.append(('T3 双向TP', lambda R: 1.4 if R >= 1.3 else (0.75 if R < 0.8 else 1.0), one))
    arms.append(('S1 高能SL×1.4', one, lambda R: 1.4 if R >= 1.3 else 1.0))
    arms.append(('S2 低能SL×0.8', one, lambda R: 0.8 if R < 0.8 else 1.0))
    arms.append(('X1 高能TP×1.4+低能TP×0.75+高能SL×1.3',
                 lambda R: 1.4 if R >= 1.3 else (0.75 if R < 0.8 else 1.0),
                 lambda R: 1.3 if R >= 1.3 else 1.0))
    arms.append(('X2 全收紧TP×0.85(对照: 无条件)', lambda R: 0.85, one))
    arms.append(('X3 全放宽SL×1.25(对照: 无条件)', one, lambda R: 1.25))
    arms.append(('X4 高能TP×1.6', lambda R: 1.6 if R >= 1.3 else 1.0, one))
    arms.append(('X5 低能TP×0.65', lambda R: 0.65 if R < 0.8 else 1.0, one))

    results = [a0]
    for label, tpf, slf in arms:
        r = evaluate(tr, m1v, m5, tpf, slf, 360, label)  # 宽TP需要扩展视野(360+360)
        results.append(r)
        print(f'{label}: n={r["n"]} wr={r["wr"]} plr={r["plr"]} pnl={r["pnl"]} avg={r["avg"]} dd={r["maxdd"]} tp/sl/to={r["tp_n"]}/{r["sl_n"]}/{r["to_n"]}', flush=True)

    # A0 也用扩展视野重评 (公平对照: 同一个模拟器)
    a0e = evaluate(tr, m1v, m5, one, one, 360, 'A0E 基线(同模拟器扩展视野)')
    results.insert(1, a0e)
    print('A0E:', a0e, flush=True)

    import json
    doc = dict(meta=dict(generated='2026-10-03', question='提升胜率但不压缩交易次数',
                         design='1657笔入场一笔不删, 只改能量条件化障碍几何; R̂=R2冠军预测流(决策时可用)',
                         baseline_repro='OK' if ok else 'MISMATCH',
                         bands={'低(<0.8)': band_n.get(0, 0), '中': band_n.get(1, 0), '高(>=1.3)': band_n.get(2, 0)}),
               results=results)
    json.dump(doc, open(OUT, 'w'), indent=1, ensure_ascii=False)
    print(f'DONE -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
