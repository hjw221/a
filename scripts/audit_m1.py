#!/usr/bin/env python3
"""M1剥头皮可行性审计 — 全部真实数据, 无任何模拟编造。
产出:
  1) M1波动率/点差分年统计 (成本地量)
  2) 几何网格(固定$TP/SL + ATR自适应下限)在首训练窗2022-01~2024-07的随机入口口径
  3) 候选几何在2024-08~2026-07的对照 (仅作regime披露, 不参与选择)
选择规则(预先声明, 只用首训练窗): TP率∈[25%,45%] 且 净PLR≥1.35 且 随机EV≥-$0.15;
  优先 TP=$1.2 (用户"吃1u"语义: 净赢≈$1), 平手取更短时限。
"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/z/my-project/download/xauusd_ml_v2")
from config import CFG
from data import load_raw_m1, impute_spread
from labeling import label_all

t0 = time.time()
m1 = load_raw_m1(CFG["data_path"])
print(f"[audit] M1加载 {len(m1):,}行 ({time.time()-t0:.0f}s)")

# ---- 波动率/点差分年统计 ----
c, h, l = m1["CLOSE"], m1["HIGH"], m1["LOW"]
tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
atr288 = tr.rolling(288, min_periods=288).mean()          # 4.8h ATR (M1刻度)
spread_cost, monthly = impute_spread(m1, CFG["point_value"])
yr = m1.index.year
print("\n===== [1] M1分年统计 (真实) =====")
print(f"{'年':<6}{'M1 TR中位$':>11}{'ATR288中位$':>12}{'点差中位$':>10}{'M1行数':>10}")
for y in sorted(set(yr)):
    m_ = np.asarray(yr == y)
    sc_y = spread_cost[m_]
    print(f"{y:<6}{tr[m_].median():>11.3f}{atr288[m_].median():>12.3f}"
          f"{np.median(sc_y):>10.3f}{int(m_.sum()):>10,}")

# ---- 事件网格 (随机入口, stride 15 去相关) ----
m1_t = (m1.index.astype("int64") // 10**9 // 60).to_numpy(np.int64)
m1_o = m1["OPEN"].to_numpy(np.float64)
m1_h = m1["HIGH"].to_numpy(np.float64)
m1_l = m1["LOW"].to_numpy(np.float64)
m1_c = m1["CLOSE"].to_numpy(np.float64)
atr_np = atr288.to_numpy(np.float64)

cal_lo = pd.Timestamp("2022-01-01"); cal_hi = pd.Timestamp("2024-08-01")   # 首训练窗右端(不含)
oos_lo = pd.Timestamp("2024-08-01")
mask_cal = (m1.index >= cal_lo) & (m1.index < cal_hi)
mask_oos = (m1.index >= oos_lo)
ev_cal = np.where(np.asarray(mask_cal))[0][::15]
ev_oos = np.where(np.asarray(mask_oos))[0][::15]
print(f"\n[audit] 随机事件: 标定窗 {len(ev_cal):,} / OOS窗 {len(ev_oos):,} (stride 15)")


def run_geo(tp_mult, sl_mult, atr_ev, horizon, ev):
    """跑一个几何点, 返回随机口径指标 (多头侧; 空头对称)。"""
    res = label_all(m1_t, m1_o, m1_h, m1_l, m1_c,
                    m1_t[ev], atr_ev, spread_cost[ev],
                    tp_mult, sl_mult, 0.30, 2.0, horizon, 10, 1)
    (_, out_l, _, pnl_l, _, _, xbl, _, _) = res
    v = pnl_l != 0.0
    n = int(v.sum())
    if n == 0:
        return None
    p = pnl_l[v]
    tp_rate = float((out_l[v] == 1).mean())
    sl_rate = float((out_l[v] == -1).mean())
    to_rate = 1.0 - tp_rate - sl_rate
    wins, losses = p[p > 0], p[p <= 0]
    plr = float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else np.nan
    dur = float(np.median(xbl[v] - ev[v]))
    return dict(n=n, tp=tp_rate, sl=sl_rate, to=to_rate, plr=plr,
                ev=float(p.mean()), dur=dur)


ones = np.ones(len(ev_cal))
# ---- [2] 固定$几何网格 (标定窗) ----
print("\n===== [2] 固定$几何网格 @ 首训练窗2022-01~2024-07 (随机入口, 净口径) =====")
print(f"{'TP$':>5}{'SL$':>6}{'H(min)':>8}{'TP率':>7}{'SL率':>7}{'超时':>7}{'净PLR':>7}{'净EV$':>8}{'中位持仓min':>10}")
rows = []
for tp in [1.0, 1.2, 1.5]:
    for sl in [0.5, 0.6, 0.75]:
        for hz in [60, 90, 120]:
            r = run_geo(tp, sl, ones, hz, ev_cal)
            if r is None:
                continue
            rows.append((tp, sl, hz, r))
            print(f"{tp:>5.2f}{sl:>6.2f}{hz:>8d}{r['tp']*100:>6.1f}%{r['sl']*100:>6.1f}%"
                  f"{r['to']*100:>6.1f}%{r['plr']:>7.2f}{r['ev']:>8.3f}{r['dur']:>10.0f}")

# ---- [3] ATR自适应下限几何 (TP=max(1.2, k*ATR288)) ----
print("\n===== [3] ATR自适应下限几何 (TP=max($1.2, k×ATR288), SL固定RR) =====")
print(f"{'k':>5}{'SL比':>6}{'H':>6}{'TP率':>7}{'SL率':>7}{'超时':>7}{'净PLR':>7}{'净EV$':>8}{'中位持仓':>9}{'TP中位$':>9}")
ad_rows = []
for k in [1.5, 2.0]:
    for slr in [0.4, 0.5]:
        for hz in [90, 120]:
            atr_ev = np.maximum(1.0, (k / 1.2) * atr_np[ev_cal])
            atr_ev[~np.isfinite(atr_ev)] = 1.0
            r = run_geo(1.2, slr * 1.2, atr_ev, hz, ev_cal)
            if r is None:
                continue
            tpd = np.maximum(1.2, k * atr_np[ev_cal])
            ad_rows.append((k, slr, hz, r))
            print(f"{k:>5.1f}{slr:>6.2f}{hz:>6d}{r['tp']*100:>6.1f}%{r['sl']*100:>6.1f}%"
                  f"{r['to']*100:>6.1f}%{r['plr']:>7.2f}{r['ev']:>8.3f}{r['dur']:>9.0f}"
                  f"{np.median(tpd[np.isfinite(tpd)]):>9.2f}")

# ---- [4] 候选几何 OOS期对照 (regime披露, 不参与选择) ----
print("\n===== [4] 候选几何 @ 2024-08~2026-07 (仅regime披露) =====")
cands = [(1.2, 0.6, 90, "fixed"), (1.2, 0.6, 120, "fixed"), (1.5, 0.6, 90, "fixed")]
for k, slr in [(1.5, 0.5), (2.0, 0.5)]:
    cands.append((k, slr, 90, "ad"))
print(f"{'几何':>24}{'TP率':>7}{'SL率':>7}{'超时':>7}{'净PLR':>7}{'净EV$':>8}{'中位持仓':>9}")
for tp, sl, hz, kind in cands:
    if kind == "fixed":
        r = run_geo(tp, sl, np.ones(len(ev_oos)), hz, ev_oos)
        nm = f"固定 TP{tp}/SL{sl}/H{hz}"
    else:
        atr_ev = np.maximum(1.0, (tp / 1.2) * atr_np[ev_oos])
        atr_ev[~np.isfinite(atr_ev)] = 1.0
        r = run_geo(1.2, sl * 1.2, atr_ev, hz, ev_oos)
        nm = f"自适应 k{tp}/rr{1/(sl):.0f}倍/H{hz}"
    if r:
        print(f"{nm:>24}{r['tp']*100:>6.1f}%{r['sl']*100:>6.1f}%{r['to']*100:>6.1f}%"
              f"{r['plr']:>7.2f}{r['ev']:>8.3f}{r['dur']:>9.0f}")
print(f"\n[audit] 总耗时 {time.time()-t0:.0f}s")
