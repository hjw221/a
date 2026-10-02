#!/usr/bin/env python3
"""v16_slip3.py — 终极修正版引擎: 触发即出场, 真实地图扫描.

两个引擎级 bug 修复后的最终口径:
  bug1 时域错位(3.47天) — 已修(astype datetime64[s])
  bug2 吊灯回望结算(触发后trail继续更新到horizon末才结算=触发后回魂) — 本脚本修正:
      stop 触发 -> 立即出场, pnl 用触发时刻通道价; blocked_until 用触发时刻解锁.

出场族(全部正确结算):
  chan_opt  : 吊灯, 同根先吸收极值再查触发(乐观: 同根先高后低)
  chan_cons : 吊灯保守版, 先用上一根通道查当根触发, 未触发再吸收极值(同根先跌后涨)
  fixed     : TP/SL 固定ATR障碍, 先触先出, 同根双触SL优先(保守)
  time      : 纯持有 horizon 分钟市价出场
信号: kelt / rnd1000(满载随机 seed2000). delta: stop侧滑点 {0.0,0.1} x ATR.
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

MODE_CHAN_OPT, MODE_CHAN_CONS, MODE_FIXED, MODE_TIME = 0, 1, 2, 3


@njit(cache=True)
def sim_true(m1_t, m1_o, m1_h, m1_l, sig_t, trade_mask, atr, cost,
             mode, chan_mult, tp_mult, sl_mult, sl_floor, horizon, entry_tol,
             delta_mult, serial):
    n = len(sig_t); nm1 = len(m1_t)
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
        end = min(e + horizon, nm1 - 1)
        if mode == MODE_TIME:
            pnl[i] = (m1_o[end] - entry) * direction - sc_
            wins[i] = 1 if pnl[i] > 0 else 0
            filled += 1
            if serial == 1 and end > blocked_until:
                blocked_until = end
            continue
        hi_since = entry; lo_since = entry
        exit_pnl = 1e18
        k_exit = end
        for k in range(e + 1, end + 1):
            if mode == MODE_CHAN_OPT:
                if direction == 1:
                    if m1_h[k] > hi_since:
                        hi_since = m1_h[k]
                    stop_px = hi_since - chan_mult * a
                    if m1_l[k] <= stop_px:
                        exit_pnl = stop_px - entry - delta_mult * a - sc_
                        k_exit = k
                        break
                else:
                    if m1_l[k] < lo_since:
                        lo_since = m1_l[k]
                    stop_px = lo_since + chan_mult * a
                    if m1_h[k] >= stop_px:
                        exit_pnl = entry - stop_px - delta_mult * a - sc_
                        k_exit = k
                        break
            elif mode == MODE_CHAN_CONS:
                if direction == 1:
                    stop_px = hi_since - chan_mult * a
                    if m1_l[k] <= stop_px:
                        exit_pnl = stop_px - entry - delta_mult * a - sc_
                        k_exit = k
                        break
                    if m1_h[k] > hi_since:
                        hi_since = m1_h[k]
                else:
                    stop_px = lo_since + chan_mult * a
                    if m1_h[k] >= stop_px:
                        exit_pnl = entry - stop_px - delta_mult * a - sc_
                        k_exit = k
                        break
                    if m1_l[k] < lo_since:
                        lo_since = m1_l[k]
            elif mode == MODE_FIXED:
                tp = tp_mult * a
                sl = sl_floor if sl_floor > sl_mult * a else sl_mult * a
                if direction == 1:
                    hit_sl = m1_l[k] <= entry - sl
                    hit_tp = m1_h[k] >= entry + tp
                else:
                    hit_sl = m1_h[k] >= entry + sl
                    hit_tp = m1_l[k] <= entry - tp
                if hit_sl:
                    exit_pnl = -sl - delta_mult * a - sc_
                    k_exit = k
                    break
                if hit_tp:
                    exit_pnl = tp - sc_
                    k_exit = k
                    break
        if exit_pnl > 1e17:
            exit_pnl = (m1_o[end] - entry) * direction - sc_
        pnl[i] = exit_pnl
        wins[i] = 1 if exit_pnl > 0 else 0
        filled += 1
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
    print(f"[sig] kelt {int((sig != 0).sum())} rnd {int((rnd != 0).sum())}", flush=True)

    ARMS = [
        ("chan_opt2.0_h120", MODE_CHAN_OPT, 2.0, 0.0, 0.0, 120),
        ("chan_opt2.5_h120", MODE_CHAN_OPT, 2.5, 0.0, 0.0, 120),
        ("chan_opt3.0_h120", MODE_CHAN_OPT, 3.0, 0.0, 0.0, 120),
        ("chan_opt3.0_h90",  MODE_CHAN_OPT, 3.0, 0.0, 0.0, 90),
        ("chan_cons3.0_h120", MODE_CHAN_CONS, 3.0, 0.0, 0.0, 120),
        ("chan_cons2.5_h120", MODE_CHAN_CONS, 2.5, 0.0, 0.0, 120),
        ("fixed_tp1.5_sl0.4", MODE_FIXED, 0.0, 1.5, 0.4, 90),
        ("fixed_tp2.5_sl0.4", MODE_FIXED, 0.0, 2.5, 0.4, 90),
        ("fixed_tp1.5_sl0.6", MODE_FIXED, 0.0, 1.5, 0.6, 90),
        ("time_h90",  MODE_TIME, 0.0, 0.0, 0.0, 90),
        ("time_h120", MODE_TIME, 0.0, 0.0, 0.0, 120),
    ]
    out = {}
    for sname, tm_full in (("kelt", sig), ("rnd1000", rnd)):
        for aname, mode, cm, tpm, slm, horizon in ARMS:
            for dm in (0.0, 0.1):
                month_pnl = {}; fills = 0; neg = 0
                for oos_s, oos_e in folds:
                    oos_mask = np.asarray((m5.index >= oos_s) & (m5.index <= oos_e))
                    tm = tm_full[oos_mask]
                    if (tm != 0).sum() == 0:
                        continue
                    pnl, wins, filled = sim_true(
                        m1_t, m1_o, m1_h, m1_l, sig_t[oos_mask], tm,
                        atr[oos_mask], cost[oos_mask], mode, cm, tpm, slm,
                        0.48, horizon, 10, dm, 1)
                    fills += filled
                    v = float(pnl[tm != 0].sum())
                    month_pnl[str(oos_s.date())[:7]] = v
                    if v < 0:
                        neg += 1
                s = pd.Series(month_pnl)
                tot = float(s.sum())
                cum = s.cumsum(); dd = float((cum - cum.cummax()).min())
                by = {}
                for mk, v in month_pnl.items():
                    y = int(mk[:4]); by[y] = by.get(y, 0.0) + v
                key = f"{sname}_{aname}_d{dm}"
                out[key] = dict(total=round(tot, 1), trades=fills,
                                avg_per_trade=round(tot / max(fills, 1), 3),
                                neg_months=neg, maxDD_m=round(dd, 1),
                                by_year={k: round(v, 1) for k, v in sorted(by.items())})
                print(f"[{key}] total={out[key]['total']} n={fills} avg={out[key]['avg_per_trade']} negM={neg}", flush=True)
    json.dump(out, open(os.path.join(RES, "slip3.json"), "w"), indent=1)
    sc.say(f"v16slip3 TRUE-ENGINE DONE {(time.time()-t00)/60:.1f}min")
    print(f"DONE {(time.time()-t00)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
