"""
回测模拟器 (numba状态机) + 绩效指标 + 阈值校准 + Bootstrap置信区间。
相对原代码的升级:
  - 单持仓 + 平仓后冷却 (原代码每根K线独立开仓, 重叠持仓虚增交易频率与PnL)
  - 信号在M5收盘产生 -> 下一根M1开盘入场 (原代码假装以当根收盘价即时成交)
  - 交易口径与标签引擎完全一致 (同一套障碍/结算逻辑)
  - 阈值校准只在训练窗内部尾段做, OOS月份绝不参与
"""
import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def _simulate(sig_dir, entry_idx, exit_bar_long, exit_bar_short,
              pnl_long, pnl_short, valid, cooldown_m1):
    """单持仓状态机。返回 (成交行号int64[], 方向int8[])。"""
    n = len(sig_dir)
    out_rows = np.empty(n, dtype=np.int64)
    out_dir = np.empty(n, dtype=np.int8)
    n_trades = 0
    next_free = np.int64(-2**62)
    for k in range(n):
        s = sig_dir[k]
        if s == 0:
            continue
        if not valid[k]:
            continue
        e = entry_idx[k]
        if e < 0:
            continue
        if e < next_free:          # 持仓中/冷却中 -> 信号作废
            continue
        if s == 1:
            xb = exit_bar_long[k]
            p = pnl_long[k]
        else:
            xb = exit_bar_short[k]
            p = pnl_short[k]
        if xb < e:                 # 数据残缺
            continue
        out_rows[n_trades] = k
        out_dir[n_trades] = s
        n_trades += 1
        next_free = xb + 1 + cooldown_m1
        _ = p  # pnl由Python侧按行取, 保持numba签名简单
    return out_rows[:n_trades], out_dir[:n_trades]


def build_signals(p_long, p_short, thr_long, thr_short):
    """信号规则: 概率超阈值; 双向同时触发取超额更大的一侧 (与实盘EA一致)。"""
    long_sig = p_long > thr_long
    short_sig = p_short > thr_short
    sig = np.zeros(len(p_long), dtype=np.int8)
    both = long_sig & short_sig
    sig[long_sig & ~short_sig] = 1
    sig[short_sig & ~long_sig] = -1
    if both.any():
        ml = p_long[both] - thr_long
        ms = p_short[both] - thr_short
        sig[both] = np.where(ml >= ms, 1, -1).astype(np.int8)
    return sig


def run_backtest(sig, rows, lab, valid, m5, m1_pack, cfg, probs=None, tag=""):
    """执行模拟并产出逐笔交易日志。rows为本段的M5全局行号(int64数组)。"""
    m1_t, m1_o, m1_h, m1_l, m1_c = m1_pack
    sig_dir = np.zeros(len(rows), dtype=np.int8)
    sig_dir[:] = sig
    t_rows, t_dir = _simulate(sig_dir,
                              lab["entry_idx"].to_numpy()[rows],
                              lab["exit_bar_long"].to_numpy()[rows],
                              lab["exit_bar_short"].to_numpy()[rows],
                              lab["pnl_long"].to_numpy()[rows],
                              lab["pnl_short"].to_numpy()[rows],
                              valid[rows], cfg["cooldown_m1"])
    gl = rows[t_rows]                       # 映射回全局行号
    loc = np.searchsorted(rows, gl)         # 全局行号 -> 本段局部位置 (probs用)
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
            exit_px[j] = m1_c[xb]
    xbi = np.where(d == 1, xb_l, xb_s)
    log = pd.DataFrame({
        "row": gl, "dir": np.where(d == 1, "long", "short"),
        "signal_time": m5.index[gl], "entry_time": pd.to_datetime(m1_t[e_idx], unit="m"),
        "exit_time": pd.to_datetime(m1_t[xbi], unit="m"),
        "entry_px": entry_px, "exit_px": exit_px,
        "outcome": np.where(outc == 1, "TP", np.where(outc == -1, "SL", "TIMEOUT")),
        "pnl": pnl, "dur_m1": xbi - e_idx})
    if probs is not None:
        log["prob"] = np.where(d == 1, probs[0][loc], probs[1][loc])
    if tag:
        log["variant"] = tag
    return log


def metrics(log, label=""):
    """汇总指标 — 全部由真实交易日志计算。盈亏比 = avg(win)/|avg(loss)|。"""
    if len(log) == 0:
        return {"label": label, "trades": 0, "win_rate": np.nan, "total_pnl": 0.0,
                "mean_pnl": np.nan, "profit_factor": np.nan, "max_dd": 0.0,
                "sharpe": np.nan, "plr": np.nan, "avg_win": np.nan, "avg_loss": np.nan}
    pnl = log["pnl"].to_numpy()
    wins = (pnl > 0).sum()
    eq = np.cumsum(pnl)
    dd = (np.maximum.accumulate(eq) - eq).max()
    daily = pd.Series(pnl, index=log["exit_time"].values).resample("D").sum()
    daily = daily[daily.index.dayofweek < 5]
    sharpe = np.sqrt(252) * daily.mean() / daily.std() if daily.std() > 0 else np.nan
    pos, neg = pnl[pnl > 0], pnl[pnl < 0]
    plr = float(pos.mean() / abs(neg.mean())) if len(pos) and len(neg) else np.nan
    return {"label": label, "trades": int(len(log)),
            "win_rate": float(wins / len(log)),
            "total_pnl": float(pnl.sum()), "mean_pnl": float(pnl.mean()),
            "profit_factor": float(pos.sum() / abs(neg.sum())) if len(neg) and neg.sum() != 0 else np.inf,
            "max_dd": float(dd), "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
            "plr": plr, "avg_win": float(pos.mean()) if len(pos) else np.nan,
            "avg_loss": float(neg.mean()) if len(neg) else np.nan}


def calibrate_threshold(pl, ps, rows, lab, valid, m5, m1_pack, cfg):
    """
    在训练窗内部的尾段(从未接触OOS)上选阈值。
    分位数网格: 对多空两个模型的概率分布各自取同一分位点 -> 自动归一化两个分布的基础率差异。
    目标 (cfg["threshold_objective"]):
      - "mean_pnl_x_sqrtN": 均值PnL×sqrt(交易数), 兼顾质量与样本量 (v2/legacy用);
      - "plr": 盈亏比优先 (v3族用) — 在满足 交易数>=min 且 总PnL>0 的候选中
        最大化盈亏比, 平分时用 mean×sqrtN 决胜;
      - "plr_wr": 胜率下限版 (平衡变体用) — 先在满足 交易数>=min 且 总PnL>0 的候选中
        筛出 实测胜率>=cfg["wr_floor"] 者, 在其中最大化盈亏比;
        若无满足胜率下限者, 退回 mean×sqrtN (美元期望自然兼顾胜率与盈亏比)。
    约束: 交易数 >= min_trades_inner 且 总PnL > 0;
    无满足者回退到极高选择性(q=0.975) —— 宁缺勿滥, 但保留信息流。
    """
    obj = cfg.get("threshold_objective", "mean_pnl_x_sqrtN")
    qs = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.985]
    cands = []                     # (q, thr_l, thr_s, n, wr, plr, mean_x_sqrtN)
    for q in qs:
        thr_l = float(np.quantile(pl, q))
        thr_s = float(np.quantile(ps, q))
        sig = build_signals(pl, ps, thr_l, thr_s)
        log = run_backtest(sig, rows, lab, valid, m5, m1_pack, cfg)
        n = len(log)
        if n < cfg["min_trades_inner"] or log["pnl"].sum() <= 0:
            continue
        mm = metrics(log)
        cands.append((q, thr_l, thr_s, n,
                      mm["win_rate"] if np.isfinite(mm["win_rate"]) else 0.0,
                      mm["plr"] if np.isfinite(mm["plr"]) else 0.0,
                      log["pnl"].mean() * np.sqrt(n)))
    if not cands:
        pass                        # 走下面的q0.975回退
    elif obj == "plr":
        best = max(cands, key=lambda c: (c[5], c[6]))
    elif obj == "plr_wr":
        wr_floor = cfg.get("wr_floor", 0.30)
        meet = [c for c in cands if c[4] >= wr_floor]
        if meet:                    # 胜率下限内最大化盈亏比
            best = max(meet, key=lambda c: (c[5], c[6]))
        else:                       # 无一达到下限 -> 退回美元期望目标(自然兼顾胜率)
            best = max(cands, key=lambda c: (c[6],))
    else:
        best = max(cands, key=lambda c: (c[6],))
    if not cands:
        # 回退: 极高选择性 (只在验证段无任何盈利候选时)
        thr_l = float(np.quantile(pl, 0.975))
        thr_s = float(np.quantile(ps, 0.975))
        return (thr_l, thr_s), None
    return (best[1], best[2]), {"q": best[0], "trades": best[3], "objective": obj,
                                "wr": best[4], "plr": best[5], "score": best[6],
                                "thr_long": best[1], "thr_short": best[2]}


# plr_wr两段式: ①有候选达到胜率下限 -> 在其中最大化盈亏比(决胜: 美元期望×样本量);
# ②无一达到 -> 全体候选中最大化美元期望×sqrtN (自然向高胜率端平衡, 不再追盈亏比)。


def bootstrap_ci(pnl, iters=10000, seed=42):
    """交易级Bootstrap 95% CI (均值与总额)。"""
    pnl = np.asarray(pnl, dtype=np.float64)
    if len(pnl) == 0:
        return (np.nan, np.nan), (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pnl), size=(iters, len(pnl)))
    means = pnl[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))), \
           (float(np.percentile(means * len(pnl), 2.5)), float(np.percentile(means * len(pnl), 97.5)))
