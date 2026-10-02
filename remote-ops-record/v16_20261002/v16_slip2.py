#!/usr/bin/env python3
"""v16_slip2.py — 障碍成交滑点压力测试: 吊灯 stop 触发价加 delta*ATR 惩罚.

背景: 原引擎假设止损/吊灯触达时按精确 stop 价成交(无滑点). 对高频轮转吊灯(m2.0_h120
每笔均~14分钟)这是重大乐观偏差. 本测试: stop 侧成交价劣化 delta*ATR (delta in {0,0.1,0.2}),
entry 不变(市价单滑点较小), end 市价出场不变. 若几何排序在悲观成交下翻转 -> 窄吊灯优势
是假设红利; 若保持 -> 结论可守.
信号: kelt / rnd1000(满载随机). 几何: (m,h) in {2.0,2.5,3.0}x{90,120}.
"""
import os, json, time
import numpy as np
import pandas as pd
from numba import njit
import stage_c_loop as sc

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results_v16")
CHAMP = json.load(open(os.path.join(BASE, "results_research", "champion.json")))
CFG = CHAMP["cfg"]
EXIT_CHANDELIER = 2


@njit(cache=True)
def simulate_exit_slip(m1_t, m1_o, m1_h, m1_l, sig_t, trade_mask,
                       atr, cost, mode, tp_mult, sl_mult, sl_floor, horizon, entry_tol,
                       chan_atr_mult, delta_mult, serial):
    n = len(sig_t)
    nm1 = len(m1_t)
    pnl = np.zeros(n); wins = np.zeros(n)
    filled = 0
    blocked_until = -1
    for i in range(n):
        if trade_mask[i] == 0:
            continue
        t0 = sig_t[i]
        lo, hi = 0, nm1
        while lo < hi:
            mid = (lo + hi) // 2
            if m1_t[mid] < t0:
                lo = mid + 1
            else:
                hi = mid
        e = lo
        if e >= nm1 or m1_t[e] > t0 + entry_tol:
            continue
        if serial == 1 and e <= blocked_until:
            continue
        a = atr[i]
        if not (a > 0.0):
            continue
        direction = trade_mask[i]
        entry = m1_o[e]
        sc_ = cost[i]
        tp = 10**9
        sl = chan_atr_mult * a
        hold = horizon
        end = min(e + hold, nm1 - 1)
        hit_sl = False
        trail = sl
        hi_since = entry; lo_since = entry
        for k in range(e + 1, end + 1):
            if direction == 1:
                if m1_h[k] > hi_since:
                    hi_since = m1_h[k]
            else:
                if m1_l[k] < lo_since:
                    lo_since = m1_l[k]
            if direction == 1:
                trail = entry - (hi_since - chan_atr_mult * a)
            else:
                trail = (lo_since + chan_atr_mult * a) - entry
            if direction == 1:
                if (not hit_sl) and m1_l[k] <= entry - trail:
                    hit_sl = True
                    break
            else:
                if (not hit_sl) and m1_h[k] >= entry + trail:
                    hit_sl = True
                    break
        if hit_sl:
            pnl_i = -trail - delta_mult * a - sc_
        else:
            pnl_i = (m1_o[end] - entry) * direction - sc_
        pnl[i] = pnl_i
        wins[i] = 1 if pnl_i > 0 else 0
        filled += 1
        k_exit = end
        if serial == 1 and k_exit > blocked_until:
            blocked_until = k_exit
    return pnl, wins, filled


def main():
    t00 = time.time()
    m5, F, cost, atr, (m1_t, m1_o, m1_h, m1_l) = sc._prep()
    sig_t = (m5.index.astype("datetime64[s]").astype("int64") + 300).to_numpy()
    folds = sc.month_folds(m5.index, "2022-08", "2026-07")
    c, h, l = m5["close"], m5["high"], m5["low"]
    ema = c.ewm(span=20, adjust=False).mean()
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    atr14 = tr.rolling(14).mean()
    sig = np.zeros(len(m5), dtype=np.int8)
    sig[(c > (ema + 2.0 * atr14)).to_numpy()] = 1
    sig[(c < (ema - 2.0 * atr14)).to_numpy()] = -1
    rng = np.random.default_rng(2000)
    ok_feat = np.isfinite(F[CFG["features"]].to_numpy()).all(axis=1)
    rnd = np.zeros(len(m5), dtype=np.int8)
    m5_month = m5.index.strftime("%Y-%m")
    for mk in pd.unique(m5_month):
        m = (m5_month == mk) & ok_feat
        pos = np.where(m)[0]
        if len(pos):
            rnd[pos] = rng.choice([1, -1], size=len(pos))

    out = {}
    GEOMS = [(2.0, 120), (2.5, 120), (3.0, 120), (2.0, 90), (2.5, 90), (3.0, 90)]
    DELTAS = [0.0, 0.1, 0.2]
    for sname, tm_full in (("kelt", sig), ("rnd1000", rnd)):
        for mult, hold in GEOMS:
            for dm in DELTAS:
                month_pnl = {}; fills = 0; neg_months = 0
                for oos_s, oos_e in folds:
                    oos_mask = np.asarray((m5.index >= oos_s) & (m5.index <= oos_e))
                    tm = tm_full[oos_mask]
                    if (tm != 0).sum() == 0:
                        continue
                    pnl, wins, filled = simulate_exit_slip(
                        m1_t, m1_o, m1_h, m1_l, sig_t[oos_mask], tm,
                        atr[oos_mask], cost[oos_mask], EXIT_CHANDELIER,
                        float(CFG["tp_mult"]), float(CFG["sl_mult"]), 0.48, hold, 10,
                        mult, dm, 1)
                    fills += filled
                    v = float(pnl[tm != 0].sum())
                    month_pnl[str(oos_s.date())[:7]] = v
                    if v < 0:
                        neg_months += 1
                s = pd.Series(month_pnl)
                tot = float(s.sum())
                cum = s.cumsum(); dd = float((cum - cum.cummax()).min())
                by = {}
                for mk, v in month_pnl.items():
                    y = int(mk[:4]); by[y] = by.get(y, 0.0) + v
                key = f"{sname}_m{mult}_h{hold}_d{dm}"
                out[key] = dict(total=round(tot, 1), trades=fills,
                                avg_per_trade=round(tot / max(fills, 1), 3),
                                neg_months=neg_months, maxDD_m=round(dd, 1),
                                by_year={k: round(v, 1) for k, v in sorted(by.items())})
                print(f"[{key}] total={out[key]['total']} n={fills} avg={out[key]['avg_per_trade']} "
                      f"negM={neg_months} dd={out[key]['maxDD_m']}", flush=True)
    json.dump(out, open(os.path.join(RES, "slip2.json"), "w"), indent=1)
    sc.say(f"v16slip2 DONE {(time.time()-t00)/60:.1f}min")
    print(f"DONE {(time.time()-t00)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
