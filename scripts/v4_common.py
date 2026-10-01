"""v4 — XAUUSD M1 从零构建: 公共模块 (与v2管线零代码复用, 仅共享原始CSV数据)

新路设计: M1原生"点火-延续"事件驱动模型
  - 事件: 1m对数收益 |z|>=k*sigma60(不含当前根) 突刺 + 量能确认; 方向=突刺方向
  - 标签: 事件方向上的 ATR1440(M1, 24h) 缩放三障碍 (TP/SL/H)
          同根双碰按SL(保守); 遇>5min缺口在缺口前一根按超时了结
  - 成本: 点差 $0.03/单边 (与基线口径可比) + 数据驱动点差并列报告
"""
import numpy as np
import pandas as pd
import os

CSV = '/home/z/my-project/upload/5_extracted/XAUUSDc_M1_202201022305_202606262057.csv'
CACHE = '/home/z/my-project/scripts/v4_cache'
OUT = '/home/z/my-project/download/xauusd_ml_v4'
POINT_VALUE = 0.001          # 1 SPREAD点 = $0.001 (XAUUSDc 3位小数报价)
FIXED_COST_SIDE = 0.03       # 固定单边成本(与基线EA假设一致)
TRAIN_END_TS = int(pd.Timestamp('2024-08-01').timestamp())  # 首训练窗右界(秒)
OOS_START_TS = int(pd.Timestamp('2024-08-01').timestamp())
DATA_END_TS = int(pd.Timestamp('2026-07-18').timestamp())
GAP_SEC = 300                # >5分钟视为时间缺口(日结/周末)


def load_m1():
    """加载全量M1 -> dict of arrays (缓存npz)"""
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, 'm1.npz')
    if os.path.exists(f):
        z = np.load(f)
        return {k: z[k] for k in z.files}
    df = pd.read_csv(CSV, sep='\t')
    df.columns = [x.strip('<>').lower() for x in df.columns]
    ts = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y.%m.%d %H:%M:%S')
    df = df.assign(ts=ts).sort_values('ts').reset_index(drop=True)
    df = df.drop_duplicates('ts', keep='first').reset_index(drop=True)
    out = {
        't': df['ts'].values.astype('datetime64[s]').astype(np.int64),
        'o': df['open'].to_numpy(np.float64),
        'h': df['high'].to_numpy(np.float64),
        'l': df['low'].to_numpy(np.float64),
        'c': df['close'].to_numpy(np.float64),
        'v': df['tickvol'].to_numpy(np.float64),
        'sp': df['spread'].to_numpy(np.float64),
    }
    np.savez_compressed(f, **out)
    return out


def gap_mask(t, gap_sec=GAP_SEC):
    g = np.zeros(len(t), bool)
    g[1:] = (t[1:] - t[:-1]) > gap_sec
    return g


def sigma60_excl(lr, win=60):
    """过去win根1m对数收益std, 不含当前根(防突刺抬高自身分母); 带下限"""
    s = pd.Series(lr).rolling(win, min_periods=win).std().shift(1).to_numpy()
    return np.maximum(s, 1e-7)


def atr_m1(h, l, c, win):
    """M1原生ATR: 过去win根TR均值(含当前根, 收盘时已知)"""
    pc = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    a = pd.Series(tr).rolling(win, min_periods=win).mean().to_numpy()
    return a, tr


def spread_usd_imputed(t, sp):
    """SPREAD列(点)->$/单边; 0=缺失->当月非零中位数, 仍无->前后最近月"""
    sp_usd = sp * POINT_VALUE
    mi = pd.PeriodIndex(pd.to_datetime(t, unit='s'), freq='M')
    s = pd.Series(sp_usd, index=mi)
    nz = s[s > 0]
    med = nz.groupby(level=0).median()
    all_m = pd.period_range(mi.min(), mi.max(), freq='M')
    med = med.reindex(all_m).ffill().bfill()
    imp = s.where(s > 0, s.index.map(med)).to_numpy(np.float64)
    return imp, med


def find_events(lr, sig, v, vmed, k, v_mult=0.8, min_sep=10, warmup=1500):
    """点火事件: |lr_t| >= k*sigma60[t-1] 且 lr_t!=0; 可选量能确认 v>=v_mult*vmed1440.
    相邻保留事件间隔>=min_sep根(不分方向); 返回(事件bar索引, 方向±1)"""
    n = len(lr)
    z = np.abs(lr) / sig
    ok = np.zeros(n, bool)
    i0 = max(warmup, 1)
    with np.errstate(invalid='ignore'):
        ok[i0:] = (z[i0:] >= k) & (lr[i0:] != 0)
    if v_mult > 0:
        with np.errstate(invalid='ignore'):
            ok &= v >= v_mult * vmed
    ok &= ~np.isnan(sig)
    cand = np.where(ok)[0]
    keep = []
    last = -10 ** 9
    for i in cand:
        if i - last >= min_sep:
            keep.append(i)
            last = i
    ev = np.array(keep, dtype=np.int64)
    if len(ev) == 0:
        return ev, np.zeros(0, np.int8)
    dr = np.sign(lr[ev]).astype(np.int8)
    return ev, dr


def sim_directional(o, h, l, c, gap, entry_i, direction, tp_off, sl_off, H):
    """方向化障碍模拟: 入场=open[entry_i]; TP价=entry+dir*tp_off; SL价=entry-dir*sl_off.
    同根双碰按SL(保守); step>0遇缺口->缺口前一根收盘按超时; 窗口耗尽按超时.
    返回 out(-1=SL,0=TO,1=TP), exit_i(绝对bar), exit_p"""
    m = len(entry_i)
    out = np.zeros(m, np.int8)
    exit_i = np.zeros(m, np.int64)
    exit_p = np.zeros(m, np.float64)
    if m == 0:
        return out, exit_i, exit_p
    ep = o[entry_i]
    tp = ep + direction * tp_off
    sl = ep - direction * sl_off
    done = np.zeros(m, bool)
    n = len(o)
    end = np.minimum(entry_i + int(H), n - 1)
    for step in range(int(H)):
        pos = entry_i + step
        alive = np.where((~done) & (pos <= end))[0]
        if len(alive) == 0:
            break
        p = pos[alive]
        if step > 0:
            g = gap[p]
            if g.any():
                ga = alive[g]
                exit_i[ga] = p[g] - 1
                exit_p[ga] = c[p[g] - 1]
                out[ga] = 0
                done[ga] = True
                alive = alive[~g]
                p = pos[alive]
                if len(alive) == 0:
                    continue
        tpx, slx, dr = tp[alive], sl[alive], direction[alive]
        tp_hit = np.where(dr > 0, h[p] >= tpx, l[p] <= tpx)
        sl_hit = np.where(dr > 0, l[p] <= slx, h[p] >= slx)
        is_sl = sl_hit
        is_tp = tp_hit & ~sl_hit
        if is_sl.any():
            ii = alive[is_sl]
            exit_i[ii] = p[is_sl]
            exit_p[ii] = sl[ii]
            out[ii] = -1
            done[ii] = True
        if is_tp.any():
            ii = alive[is_tp]
            exit_i[ii] = p[is_tp]
            exit_p[ii] = tp[ii]
            out[ii] = 1
            done[ii] = True
    un = np.where(~done)[0]
    if len(un) > 0:
        exit_i[un] = end[un]
        exit_p[un] = c[end[un]]
        out[un] = 0
    return out, exit_i, exit_p


def month_id_of(t):
    """每个bar的月份序号 (2022-01=0)"""
    starts = (pd.date_range('2022-01-01', '2027-01-01', freq='MS')
              .astype('int64') // 10 ** 9).to_numpy()
    return np.searchsorted(starts, t, side='right') - 1, starts
