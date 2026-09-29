"""
M1 剥头皮核心 (Task14): 固定$障碍标签引擎 + M1走查折 + 回测封装 + 阈值校准。

与既有管线的关系:
  - 标签/回测/折的口径完全继承 backtest.py / walkforward.py 的设计 (下一根开盘入场,
    同根TP/SL双碰按SL, 超时按真实平仓价, 单持仓+冷却, 点差逐笔), 只把
    "事件网格" 从 M5 换成 M1, 把 "ATR自适应障碍" 换成 "固定$障碍" (用户指定TP=$1)。
  - 模拟器核心 _simulate / 信号规则 build_signals / 指标 metrics / 集成训练 _fold_ens
    直接复用既有实现, 不做任何改动。

防泄露 (与既有管线同标准):
  - 训练窗右端 purge = 标签horizon + 30根M1 embargo; fit/val 之间同样留 purge gap;
  - 阈值只在训练窗内部尾段校准; OOS月零参与;
  - 几何(SL/超时)只在首训练窗随机入口上标定 (scripts/scalp_geom.py)。
"""
import os
import numpy as np
import pandas as pd
from numba import njit

from backtest import _simulate, build_signals, metrics  # noqa: F401 (复用)


# ---------------------------------------------------------------- 标签引擎
@njit(cache=True)
def label_all_m1(m1_t, m1_o, m1_h, m1_l, m1_c, spread_cost,
                 tp_usd, sl_usd, spread_floor_mult, horizon, entry_tol_min):
    """固定$三重障碍, 事件=每根M1。信号在M1收盘已知 -> 下一根M1开盘入场。
    返回与 label_all 相同结构的9元组 (entry_idx指向M1数组)。"""
    n = len(m1_t)
    INF = 2**62
    entry_idx = np.full(n, -1, dtype=np.int64)
    out_long = np.zeros(n, dtype=np.int8)
    out_short = np.zeros(n, dtype=np.int8)
    pnl_long = np.zeros(n, dtype=np.float64)
    pnl_short = np.zeros(n, dtype=np.float64)
    exit_bar_long = np.full(n, -1, dtype=np.int64)
    exit_bar_short = np.full(n, -1, dtype=np.int64)
    tp_d = np.zeros(n, dtype=np.float64)
    sl_d = np.zeros(n, dtype=np.float64)

    for i in range(n):
        sig_time = m1_t[i] + 1              # M1收盘时刻, 入场其后第一根M1
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if m1_t[mid] < sig_time:
                lo = mid + 1
            else:
                hi = mid
        e = lo
        if e >= n or m1_t[e] > sig_time + entry_tol_min:
            continue
        entry = m1_o[e]
        sc = spread_cost[i]
        tp = tp_usd
        sl = max(sl_usd, spread_floor_mult * sc)

        tp_d[i] = tp
        sl_d[i] = sl
        entry_idx[i] = e

        end = e + horizon
        end_cap = end if end < n else n - 1
        l_tp, l_sl = entry + tp, entry - sl
        s_tp, s_sl = entry - tp, entry + sl
        j_l, j_s = INF, INF
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

        if l_hit == 1:
            out_long[i] = 1; pnl_long[i] = tp - sc; exit_bar_long[i] = j_l
        elif l_hit == -1:
            out_long[i] = -1; pnl_long[i] = -sl - sc; exit_bar_long[i] = j_l
        else:
            out_long[i] = 0; pnl_long[i] = (timeout_close - entry) - sc; exit_bar_long[i] = end_cap
        if s_hit == 1:
            out_short[i] = 1; pnl_short[i] = tp - sc; exit_bar_short[i] = j_s
        elif s_hit == -1:
            out_short[i] = -1; pnl_short[i] = -sl - sc; exit_bar_short[i] = j_s
        else:
            out_short[i] = 0; pnl_short[i] = (entry - timeout_close) - sc; exit_bar_short[i] = end_cap

    return (entry_idx, out_long, out_short, pnl_long, pnl_short,
            exit_bar_long, exit_bar_short, tp_d, sl_d)


def make_labels_m1(m1_pack, spread_cost, geometry):
    """Python包装: 固定$几何标签。返回 (标签DataFrame, 有效行mask)。"""
    m1_t, m1_o, m1_h, m1_l, m1_c = m1_pack
    res = label_all_m1(m1_t, m1_o, m1_h, m1_l, m1_c, spread_cost,
                       geometry["tp_usd"], geometry["sl_usd"],
                       geometry["spread_floor_mult"], geometry["horizon_m1"], 10)
    (entry_idx, out_long, out_short, pnl_long, pnl_short,
     exit_bar_long, exit_bar_short, tp_d, sl_d) = res
    idx = pd.RangeIndex(len(m1_t))
    lab = pd.DataFrame({
        "entry_idx": entry_idx, "out_long": out_long, "out_short": out_short,
        "pnl_long": pnl_long, "pnl_short": pnl_short,
        "exit_bar_long": exit_bar_long, "exit_bar_short": exit_bar_short,
        "tp_d": tp_d, "sl_d": sl_d}, index=idx)
    valid = (entry_idx >= 0) & np.isfinite(pnl_long)
    return lab, valid


# ---------------------------------------------------------------- 走查折 (M1行号)
def build_folds_m1(m1_index, cfg, horizon_m1):
    """与 walkforward.build_folds 同构, 但行号是M1数组的行号。
    purge = horizon + 30根M1 (embargo=30, 与既有管线的30分钟embargo等价)。"""
    purge_bars = horizon_m1 + cfg["scalp_embargo_m1"]
    months = m1_index.to_period("M")
    uniq = months.unique()
    month_arr = np.asarray(months)
    pos = np.arange(len(m1_index))
    row_of_month = {}
    for m in uniq:
        mask = month_arr == m
        ii = pos[mask]
        row_of_month[m] = (ii[0], ii[-1])

    first_oos = pd.Period(cfg["first_oos_month"], "M")
    last_oos = pd.Period(cfg["last_oos_month"], "M")
    oos_list = [m for m in uniq if first_oos <= m <= last_oos]
    if not oos_list:
        raise ValueError("OOS月份区间为空")
    data_start = uniq[0]
    folds = []
    for m in oos_list:
        train_months = [x for x in uniq if data_start <= x < m]
        n_train = len(train_months)
        if n_train > cfg["max_train_months"]:
            train_months = train_months[-cfg["max_train_months"]:]
            n_train = cfg["max_train_months"]
        t_lo = row_of_month[train_months[0]][0]
        t_hi = row_of_month[train_months[-1]][1]
        t_hi_eff = t_hi - purge_bars
        train_rows = np.arange(t_lo, t_hi_eff + 1)
        n_val = int(len(train_rows) * cfg["inner_val_frac"])
        val_rows = train_rows[-n_val:]
        fit_rows = train_rows[:-n_val - purge_bars]
        oos_rows = np.arange(row_of_month[m][0], row_of_month[m][1] + 1)
        folds.append({
            "oos_month": str(m), "train_months": n_train,
            "window": f"{train_months[0]}~{train_months[-1]}",
            "train_rows": train_rows, "fit_rows": fit_rows, "val_rows": val_rows,
            "oos_rows": oos_rows, "purge_bars": purge_bars})
    return folds


# ---------------------------------------------------------------- 回测 (M1行号)
def run_backtest_m1(sig, rows, lab, valid, m1_pack, cfg, probs=None, tag=""):
    """与 backtest.run_backtest 同构, 事件索引=M1行号, 时间直接取自M1数组。"""
    m1_t, m1_o, *_ = m1_pack
    sig_dir = np.zeros(len(rows), dtype=np.int8)
    sig_dir[:] = sig
    t_rows, t_dir = _simulate(sig_dir,
                               lab["entry_idx"].to_numpy()[rows],
                               lab["exit_bar_long"].to_numpy()[rows],
                               lab["exit_bar_short"].to_numpy()[rows],
                               lab["pnl_long"].to_numpy()[rows],
                               lab["pnl_short"].to_numpy()[rows],
                               valid[rows], cfg["cooldown_m1"])
    gl = rows[t_rows]
    d = t_dir
    e_idx = lab["entry_idx"].to_numpy()[gl]
    xb_l = lab["exit_bar_long"].to_numpy()[gl]
    xb_s = lab["exit_bar_short"].to_numpy()[gl]
    pnl = np.where(d == 1, lab["pnl_long"].to_numpy()[gl], lab["pnl_short"].to_numpy()[gl])
    outc = np.where(d == 1, lab["out_long"].to_numpy()[gl], lab["out_short"].to_numpy()[gl])
    tp_d = lab["tp_d"].to_numpy()[gl]
    sl_d = lab["sl_d"].to_numpy()[gl]
    entry_px = m1_o[e_idx]
    exit_px = np.empty(len(gl))
    for j in range(len(gl)):
        if outc[j] == 1:
            exit_px[j] = entry_px[j] + tp_d[j] if d[j] == 1 else entry_px[j] - tp_d[j]
        elif outc[j] == -1:
            exit_px[j] = entry_px[j] - sl_d[j] if d[j] == 1 else entry_px[j] + sl_d[j]
        else:
            xb = xb_l[j] if d[j] == 1 else xb_s[j]
            exit_px[j] = m1_pack[4][xb]
    xbi = np.where(d == 1, xb_l, xb_s)
    log = pd.DataFrame({
        "row": gl, "dir": np.where(d == 1, "long", "short"),
        "signal_time": pd.to_datetime(m1_t[gl], unit="m"),
        "entry_time": pd.to_datetime(m1_t[e_idx], unit="m"),
        "exit_time": pd.to_datetime(m1_t[xbi], unit="m"),
        "entry_px": entry_px, "exit_px": exit_px,
        "outcome": np.where(outc == 1, "TP", np.where(outc == -1, "SL", "TIMEOUT")),
        "pnl": pnl, "dur_m1": xbi - e_idx})
    if probs is not None:
        loc = np.searchsorted(rows, gl)
        log["prob"] = np.where(d == 1, probs[0][loc], probs[1][loc])
    if tag:
        log["variant"] = tag
    return log


# ---------------------------------------------------------------- 阈值校准 (M1)
def calibrate_threshold_m1(pl, ps, rows, lab, valid, m1_pack, cfg):
    """与 backtest.calibrate_threshold 同构 (分位网格 + mean_pnl_x_sqrtN目标)。
    scalp用期望×sqrtN目标: 剥头皮的价值=期望×频率, 不追单笔盈亏比。"""
    qs = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.985]
    cands = []
    for q in qs:
        thr_l = float(np.quantile(pl, q))
        thr_s = float(np.quantile(ps, q))
        sig = build_signals(pl, ps, thr_l, thr_s)
        log = run_backtest_m1(sig, rows, lab, valid, m1_pack, cfg)
        n = len(log)
        if n < cfg["min_trades_inner"] or log["pnl"].sum() <= 0:
            continue
        mm = metrics(log)
        cands.append((q, thr_l, thr_s, n,
                      mm["win_rate"] if np.isfinite(mm["win_rate"]) else 0.0,
                      mm["plr"] if np.isfinite(mm["plr"]) else 0.0,
                      log["pnl"].mean() * np.sqrt(n)))
    if not cands:
        thr_l = float(np.quantile(pl, 0.975))
        thr_s = float(np.quantile(ps, 0.975))
        return (thr_l, thr_s), None
    best = max(cands, key=lambda c: (c[6],))
    return (best[1], best[2]), {"q": best[0], "trades": best[3],
                                "wr": best[4], "plr": best[5], "score": best[6],
                                "thr_long": best[1], "thr_short": best[2]}
