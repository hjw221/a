#!/usr/bin/env python3
"""v15_exit.py — 出场几何消融 v3: 回应 "ATR 出场不是最优解".

v2->v3: 加 --serial 串行口径 (一次一仓, 上一笔出场后才能开新仓).
  动机: chandelier/scale_out 持仓90分钟, 重叠口径下同一波趋势被几十个信号重复计利
  (+50K 假象); 串行口径才是单账户可部署的真实数字.
  --serial 输出 serial_arms.json; 默认(重叠口径, 与818/732研究线同口径)输出 exit_arms.json

v1->v2 修正:
  1. chandelier 追踪止损更新方向写反 -> 无条件每根bar重算 stop=hi_since-mult*ATR (真吊灯)
  2. scale_out 后半仓亏损从未入账 -> 拆账 0.5*(tp-sc)+0.5*(-trail-sc)
  3. cooldown=0 (与 base 原版同口径)
  4. win = pnl>0 (chandelier 无 TP 概念)
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from numba import njit
import stage_c_loop as sc

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results_v15")
os.makedirs(RES, exist_ok=True)
CHAMP = json.load(open(os.path.join(BASE, "results_research", "champion.json")))
CFG = CHAMP["cfg"]
OUT_JSON = os.path.join(RES, "exit_arms.json")

EXIT_STATIC, EXIT_FIXED, EXIT_CHANDELIER, EXIT_BRACKET, EXIT_SCALEOUT, EXIT_TIME, EXIT_TIGHTATR = 0, 1, 2, 3, 4, 5, 6


@njit(cache=True)
def simulate_exit(m1_t, m1_o, m1_h, m1_l, sig_t, trade_mask,
                  atr, cost, mode, tp_mult, sl_mult, sl_floor, horizon, entry_tol,
                  fixed_tp, fixed_sl, chan_atr_mult, n_bracket, bracket_tp_mult,
                  time_hold, tight_tp, tight_sl, scale_trail, serial):
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
        # ---- 出场几何参数化 ----
        if mode == EXIT_STATIC:
            tp = tp_mult * a
            sl = max(sl_mult * a, sl_floor)
            hold = horizon
        elif mode == EXIT_FIXED:
            tp = fixed_tp
            sl = fixed_sl
            hold = horizon
        elif mode == EXIT_CHANDELIER:
            tp = 10**9
            sl = chan_atr_mult * a          # 初始吊灯位 (距 entry 距离)
            hold = horizon
        elif mode == EXIT_BRACKET:
            if direction == 1:
                hi_px = m1_h[e]
                for k in range(max(0, e - n_bracket), e):
                    if m1_h[k] > hi_px:
                        hi_px = m1_h[k]
                tp = (hi_px - entry) * bracket_tp_mult
            else:
                lo_px = m1_l[e]
                for k in range(max(0, e - n_bracket), e):
                    if m1_l[k] < lo_px:
                        lo_px = m1_l[k]
                tp = (entry - lo_px) * bracket_tp_mult
            if tp <= 0:
                tp = tp_mult * a
            sl = max(sl_mult * a, sl_floor)
            hold = horizon
        elif mode == EXIT_SCALEOUT:
            tp = 1.2 * a                     # 半仓目标
            sl = max(sl_mult * a, sl_floor)
            hold = horizon
        elif mode == EXIT_TIME:
            tp = 10**9
            sl = 10**9
            hold = time_hold
        else:
            tp = tight_tp * a
            sl = max(tight_sl * a, sl_floor)
            hold = horizon
        end = min(e + hold, nm1 - 1)
        pnl_i = 0.0
        hit_tp = False; hit_sl = False; k_tp = -1; k_sl = -1
        trail = sl                           # 止损距 entry 的距离 (可为负=盈利锁)
        hi_since = entry; lo_since = entry
        for k in range(e + 1, end + 1):
            if direction == 1:
                if m1_h[k] > hi_since:
                    hi_since = m1_h[k]
            else:
                if m1_l[k] < lo_since:
                    lo_since = m1_l[k]
            # ---- 追踪更新 ----
            if mode == EXIT_CHANDELIER:
                # 真吊灯: stop_price = hi_since - mult*ATR (多头); trail = entry - stop
                if direction == 1:
                    trail = entry - (hi_since - chan_atr_mult * a)
                else:
                    trail = (lo_since + chan_atr_mult * a) - entry
            elif mode == EXIT_SCALEOUT and hit_tp:
                # 后半仓吊灯 scale_trail*ATR (仅 TP 后激活; 之前保持 sl)
                if direction == 1:
                    trail = entry - (hi_since - scale_trail * a)
                else:
                    trail = (lo_since + scale_trail * a) - entry
            # ---- 障碍判定 ----
            if direction == 1:
                if not hit_tp and m1_h[k] >= entry + tp: hit_tp = True; k_tp = k
                if not hit_sl and m1_l[k] <= entry - trail: hit_sl = True; k_sl = k
            else:
                if not hit_tp and m1_l[k] <= entry - tp: hit_tp = True; k_tp = k
                if not hit_sl and m1_h[k] >= entry + trail: hit_sl = True; k_sl = k
            if hit_tp and hit_sl:
                break
        # ---- 结算 ----
        if mode == EXIT_SCALEOUT and hit_tp:
            first = 0.5 * (tp - sc_)
            if hit_sl:      # k_sl > k_tp (TP 先达, 后半仓被吊灯打掉)
                second = 0.5 * (-trail - sc_)
                k_exit = k_sl
            else:           # 后半仓持有到 end 市价出场
                second = 0.5 * ((m1_o[end] - entry) * direction - sc_)
                k_exit = end
            pnl_i = first + second
        elif hit_tp and (not hit_sl or k_tp < k_sl):
            pnl_i = tp - sc_
            k_exit = k_tp
        elif hit_sl:
            pnl_i = -trail - sc_             # trail<0 时为正 (吊灯锁盈)
            k_exit = k_sl
        else:
            pnl_i = (m1_o[end] - entry) * direction - sc_
            k_exit = end
        pnl[i] = pnl_i
        wins[i] = 1 if pnl_i > 0 else 0
        filled += 1
        if serial == 1 and k_exit > blocked_until:
            blocked_until = k_exit
    return pnl, wins, filled


ARMS = {
    "base":        dict(mode=EXIT_STATIC,     fixed_tp=0.0, fixed_sl=0.0, chan_atr_mult=3.0,
                        n_bracket=0, bracket_tp_mult=0.0, time_hold=0, tight_tp=0.0, tight_sl=0.0, scale_trail=0.75),
    "fixed":       dict(mode=EXIT_FIXED,      fixed_tp=2.16, fixed_sl=0.60, chan_atr_mult=3.0,
                        n_bracket=0, bracket_tp_mult=0.0, time_hold=0, tight_tp=0.0, tight_sl=0.0, scale_trail=0.75),
    "chandelier":  dict(mode=EXIT_CHANDELIER, fixed_tp=0.0, fixed_sl=0.0, chan_atr_mult=3.0,
                        n_bracket=0, bracket_tp_mult=0.0, time_hold=0, tight_tp=0.0, tight_sl=0.0, scale_trail=0.75),
    "chan2.5":     dict(mode=EXIT_CHANDELIER, fixed_tp=0.0, fixed_sl=0.0, chan_atr_mult=2.5,
                        n_bracket=0, bracket_tp_mult=0.0, time_hold=0, tight_tp=0.0, tight_sl=0.0, scale_trail=0.75),
    "chan3.5":     dict(mode=EXIT_CHANDELIER, fixed_tp=0.0, fixed_sl=0.0, chan_atr_mult=3.5,
                        n_bracket=0, bracket_tp_mult=0.0, time_hold=0, tight_tp=0.0, tight_sl=0.0, scale_trail=0.75),
    "bracket_str": dict(mode=EXIT_BRACKET,    fixed_tp=0.0, fixed_sl=0.0, chan_atr_mult=3.0,
                        n_bracket=30, bracket_tp_mult=0.9, time_hold=0, tight_tp=0.0, tight_sl=0.0, scale_trail=0.75),
    "scale_out":   dict(mode=EXIT_SCALEOUT,   fixed_tp=0.0, fixed_sl=0.0, chan_atr_mult=3.0,
                        n_bracket=0, bracket_tp_mult=0.0, time_hold=0, tight_tp=0.0, tight_sl=0.0, scale_trail=0.75),
    "scale06":     dict(mode=EXIT_SCALEOUT,   fixed_tp=0.0, fixed_sl=0.0, chan_atr_mult=3.0,
                        n_bracket=0, bracket_tp_mult=0.0, time_hold=0, tight_tp=0.0, tight_sl=0.0, scale_trail=0.60),
    "scale09":     dict(mode=EXIT_SCALEOUT,   fixed_tp=0.0, fixed_sl=0.0, chan_atr_mult=3.0,
                        n_bracket=0, bracket_tp_mult=0.0, time_hold=0, tight_tp=0.0, tight_sl=0.0, scale_trail=0.90),
    "time_exit":   dict(mode=EXIT_TIME,       fixed_tp=0.0, fixed_sl=0.0, chan_atr_mult=3.0,
                        n_bracket=0, bracket_tp_mult=0.0, time_hold=30, tight_tp=0.0, tight_sl=0.0, scale_trail=0.75),
    "atr_tight":   dict(mode=EXIT_TIGHTATR,   fixed_tp=0.0, fixed_sl=0.0, chan_atr_mult=3.0,
                        n_bracket=0, bracket_tp_mult=0.0, time_hold=0, tight_tp=1.2, tight_sl=0.4, scale_trail=0.75),
}

_orig_sim = sc.simulate
_filled = {"n": 0}


def make_exit_sim(arm, serial=0):
    p = ARMS[arm]
    def sim(m1_t, m1_o, m1_h, m1_l, sig_t, p_long, trade,
            atr, cost, tp_mult, sl_mult, sl_floor, sp_mult, horizon, entry_tol):
        tp_mult = float(np.max(tp_mult) if hasattr(tp_mult, "dtype") else tp_mult)
        sl_mult = float(np.max(sl_mult) if hasattr(sl_mult, "dtype") else sl_mult)
        pnl, wins, filled = simulate_exit(m1_t, m1_o, m1_h, m1_l, sig_t, trade,
                             atr, cost, p["mode"], tp_mult, sl_mult, sl_floor, horizon, entry_tol,
                             p["fixed_tp"], p["fixed_sl"], p["chan_atr_mult"],
                             p["n_bracket"], p["bracket_tp_mult"],
                             p["time_hold"], p["tight_tp"], p["tight_sl"], p["scale_trail"], serial)
        _filled["n"] += filled
        return pnl, wins
    return sim


def main():
    serial = "--serial" in sys.argv
    arms = [a for a in sys.argv[1:] if not a.startswith("--")]
    arms = arms or list(ARMS.keys())
    out_path = os.path.join(RES, "serial_arms.json") if serial else OUT_JSON
    out = {}
    if os.path.exists(out_path):
        try:
            out = json.load(open(out_path))
        except Exception:
            out = {}
    for arm in arms:
        t0 = time.time()
        _filled["n"] = 0
        if arm == "base" and not serial:
            sc.simulate = _orig_sim          # 重叠口径 base: 原版引擎逐位复现 818.05
        else:
            sc.simulate = make_exit_sim(arm, 1 if serial else 0)
        r = sc.final_eval(CFG)
        n_filled = _filled["n"]
        tot_wins = sum(int(f.get("wins", 0)) for f in r.get("folds", []))
        wr = tot_wins / max(n_filled, 1)
        s = pd.Series({f["month"]: float(f["pnl"]) for f in r.get("folds", [])})
        by = {}
        for m, v in s.items():
            y = int(m[:4]); by[y] = by.get(y, 0.0) + float(v)
        tot = float(s.sum())
        avg = tot / max(n_filled, 1)
        out[arm] = dict(
            total=round(tot, 1), trades=int(n_filled), win_rate=round(wr, 3),
            sharpe=round(float(r["sharpe"]), 2), plr=round(float(r["plr"]), 2),
            avg_per_trade=round(avg, 3), serial=bool(serial),
            by_year={k: round(v, 1) for k, v in sorted(by.items())},
            share_2026=round(by.get(2026, 0.0) / tot, 3) if tot else None,
            monthly={m: round(float(v), 2) for m, v in s.items()},
            secs=round(time.time() - t0),
        )
        json.dump(out, open(out_path, "w"), indent=1)
        sc.say(f"v15 {'serial ' if serial else ''}{arm}: total {tot:.0f} filled {n_filled} 笔均 {avg:.3f} | 逐年 {out[arm]['by_year']} ({out[arm]['secs']}s)")
        print(f"[v15{'-serial' if serial else ''}] {arm}: {json.dumps({k: out[arm][k] for k in ('total','trades','plr','avg_per_trade','by_year')})}", flush=True)
    sc.simulate = _orig_sim
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
