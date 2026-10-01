"""从真实交易日志量化低胜率的实际代价: 连亏长度/回撤/每笔期望。
只用已产出的 trades_*.csv, 无任何模拟。"""
import numpy as np
import pandas as pd

RES = "/home/z/my-project/download/xauusd_ml_v2/results"

def max_lose_streak(pnl):
    w = (pnl <= 0).astype(int)
    best = cur = 0
    for x in w:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best

rows = []
for tag in ["legacy", "legacy_plrgeo", "v3aggr"]:
    fp = f"{RES}/trades_{tag}.csv"
    log = pd.read_csv(fp)
    pnl = log["pnl"].to_numpy()
    n = len(pnl)
    wr = (pnl > 0).mean()
    w_, l_ = pnl[pnl > 0], pnl[pnl < 0]
    plr = w_.mean() / abs(l_.mean())
    # 期望最大连亏(独立同分布近似): ln(N)/ln(1/(1-WR))
    exp_streak = np.log(n) / np.log(1 / (1 - wr))
    obs_streak = max_lose_streak(pnl)
    # 连亏区间内的最大回撤(实际观测)
    eq = np.cumsum(pnl)
    dd = (np.maximum.accumulate(eq) - eq).max()
    # 连续亏损最长段内累计亏损
    w = (pnl <= 0).astype(int)
    segs, cur, s0 = [], 0, 0
    for i, x in enumerate(w):
        if x:
            if cur == 0: s0 = i
            cur += 1
        else:
            if cur > 0: segs.append((cur, pnl[s0:i].sum()))
            cur = 0
    if cur > 0: segs.append((cur, pnl[s0:].sum()))
    segs.sort(reverse=True)
    worst_seg = segs[0] if segs else (0, 0)
    rows.append({
        "variant": tag, "trades": n, "win_rate": round(wr * 100, 1),
        "plr": round(plr, 2),
        "E_per_trade$": round(wr * w_.mean() - (1 - wr) * abs(l_.mean()), 2),
        "理论最大连亏": round(exp_streak, 1), "实际最大连亏": worst_seg[0],
        "实际最长连亏段累计$": round(worst_seg[1], 1), "max_dd$": round(dd, 1),
    })

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("===== 低胜率的真实代价 (来自真实OOS交易日志 2024-08~2026-07) =====")
print(df.to_string(index=False))
