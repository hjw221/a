"""
三重障碍标签引擎 (numba, M1精度)。
相对原代码的升级:
  1. ATR自适应障碍: TP/SL 距离随波动率缩放 (原固定$3.5/$2.0在金价翻倍后失效)
  2. 入场价 = 信号M5收盘之后的下一根M1开盘 (原代码用当前收盘 = 假装能即时成交)
  3. 超时(垂直障碍)按真实平仓价结算, 不再按全额止损计 (原代码PnL被低估)
  4. 同一根M1内TP/SL同时触发按SL优先 (悲观, 与原代码一致)
  5. 逐笔点差成本 (原代码2022-23按0成本)
  6. 多空两侧各自独立记录离场K线与持仓时长 (回测状态机需要)
每个M5行的交易结果只依赖该行之后的行情, 与是否真开仓无关 —— 回测模拟器直接
复用逐行结果, 保证训练口径与回测口径完全一致。
"""
import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def label_all(m1_t, m1_o, m1_h, m1_l, m1_c,
              m5_t, atr, spread_cost,
              tp_mult, sl_mult, sl_floor_usd, spread_floor_mult,
              horizon, entry_tol_min):
    n = len(m5_t)
    nm1 = len(m1_t)
    INF = 2**62
    entry_idx = np.full(n, -1, dtype=np.int64)
    out_long = np.zeros(n, dtype=np.int8)     # 1=TP  -1=SL  0=超时
    out_short = np.zeros(n, dtype=np.int8)
    pnl_long = np.zeros(n, dtype=np.float64)
    pnl_short = np.zeros(n, dtype=np.float64)
    exit_bar_long = np.full(n, -1, dtype=np.int64)
    exit_bar_short = np.full(n, -1, dtype=np.int64)
    tp_d = np.zeros(n, dtype=np.float64)
    sl_d = np.zeros(n, dtype=np.float64)

    for i in range(n):
        sig_time = m5_t[i] + 5              # M5收盘时刻(信号已知), 入场在其后第一根M1
        lo, hi = 0, nm1                     # 二分: 第一根 >= sig_time 的M1
        while lo < hi:
            mid = (lo + hi) // 2
            if m1_t[mid] < sig_time:
                lo = mid + 1
            else:
                hi = mid
        e = lo
        if e >= nm1 or m1_t[e] > sig_time + entry_tol_min:
            continue                         # 缺口过大/数据尽头
        entry = m1_o[e]
        a = atr[i]
        if not (a > 0.0) or not np.isfinite(a):
            continue
        sc = spread_cost[i]
        tp = tp_mult * a
        sl = max(sl_mult * a, sl_floor_usd, spread_floor_mult * sc)

        tp_d[i] = tp
        sl_d[i] = sl
        entry_idx[i] = e

        end = e + horizon
        end_cap = end if end < nm1 else nm1 - 1
        l_tp, l_sl = entry + tp, entry - sl
        s_tp, s_sl = entry - tp, entry + sl
        j_l, j_s = INF, INF                  # 各侧首次触发K线
        l_hit, s_hit = 0, 0
        timeout_close = m1_c[end_cap]
        j = e + 1
        while j <= end_cap:
            if l_hit == 0:
                if m1_l[j] <= l_sl:
                    l_hit = -1; j_l = j
                elif m1_h[j] >= l_tp:
                    l_hit = 1; j_l = j
            if s_hit == 0:
                if m1_h[j] >= s_sl:
                    s_hit = -1; j_s = j
                elif m1_l[j] <= s_tp:
                    s_hit = 1; j_s = j
            if l_hit != 0 and s_hit != 0:
                break
            j += 1

        # ---- 多头结算 ----
        if l_hit == 1:
            out_long[i] = 1
            pnl_long[i] = tp - sc
            exit_bar_long[i] = j_l
        elif l_hit == -1:
            out_long[i] = -1
            pnl_long[i] = -sl - sc
            exit_bar_long[i] = j_l
        else:
            out_long[i] = 0
            pnl_long[i] = (timeout_close - entry) - sc
            exit_bar_long[i] = end_cap
        # ---- 空头结算 ----
        if s_hit == 1:
            out_short[i] = 1
            pnl_short[i] = tp - sc
            exit_bar_short[i] = j_s
        elif s_hit == -1:
            out_short[i] = -1
            pnl_short[i] = -sl - sc
            exit_bar_short[i] = j_s
        else:
            out_short[i] = 0
            pnl_short[i] = (entry - timeout_close) - sc
            exit_bar_short[i] = end_cap

    return (entry_idx, out_long, out_short, pnl_long, pnl_short,
            exit_bar_long, exit_bar_short, tp_d, sl_d)


def make_labels(m5, m1_pack, spread_cost, atr, cfg):
    """Python包装: 返回标签DataFrame与有效行mask。"""
    m5_t = (m5.index.astype("datetime64[s]").astype("int64") // 60).to_numpy(np.int64)
    m1_t, m1_o, m1_h, m1_l, m1_c = m1_pack
    atr_arr = atr.to_numpy(np.float64)
    res = label_all(m1_t, m1_o, m1_h, m1_l, m1_c, m5_t, atr_arr, spread_cost,
                    cfg["tp_atr_mult"], cfg["sl_atr_mult"], cfg["sl_floor_usd"],
                    cfg["spread_floor_mult"], cfg["horizon_m1"], 10)
    (entry_idx, out_long, out_short, pnl_long, pnl_short,
     exit_bar_long, exit_bar_short, tp_d, sl_d) = res
    lab = pd.DataFrame({
        "entry_idx": entry_idx, "out_long": out_long, "out_short": out_short,
        "pnl_long": pnl_long, "pnl_short": pnl_short,
        "exit_bar_long": exit_bar_long, "exit_bar_short": exit_bar_short,
        "tp_d": tp_d, "sl_d": sl_d}, index=m5.index)
    valid = (entry_idx >= 0) & np.isfinite(pnl_long)
    return lab, valid
