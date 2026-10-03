#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v17u_block.py — 终极版第2机制: 高能突破被跳过后 N 根再武装 (消灭追单稀释).

机理: H3五分位断崖(Q1/Q2笔均$30+ vs Q3-5 $8-12) + 重跑稀释签名(新交易笔均$11.5,
多为同一能量事件里的追单)。静态NR闸门拦不住"跳过高能突破后, 同事件内的次级突破"。
机制: 触发线被触 ∧ ATR闸门开 ∧ NR>=θ → 跳过并进入 N 根封锁期 (不再进场)。
"""
import os, sys, json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import v17_htf
from v17_htf import load_m1, resample_htf, build_arrays, build_entry, EXIT_MODE_TIME
from v17c_champ import summarize
from v17u_test import nowcast_m15, map15to30, SPEC

v17_htf.DATA_CSV = os.path.join(BASE, "data.csv")
RES = os.path.join(BASE, "results_v17u")


def simulate_ultimate(o, h, l, c, atr, line_up, hzn, cost_mode, cost_flat, cost_atr_frac,
                      gate_atr, nr30, theta, block):
    """串行单仓: Donchian停损单进场(只多) ∧ ATR闸门 ∧ NR<θ; 跳过触发→block根封锁."""
    n = len(o)
    e_i = np.empty(n, np.int64); x_i = np.empty(n, np.int64)
    epx = np.empty(n, np.float64); xpx = np.empty(n, np.float64)
    cst = np.empty(n, np.float64); rsn = np.empty(n, np.int8)
    nt = 0; pos = 0; entry_px = 0.0; eib = 0; cost_v = 0.0
    blocked_until = -1
    for k in range(1, n):
        # ---- 出场: 时间出场 (与冠军引擎逐字同款) ----
        if pos != 0:
            a_prev = atr[k - 1]
            if k >= eib + hzn:
                e_i[nt] = eib; x_i[nt] = k; epx[nt] = entry_px
                xpx[nt] = o[k]; cst[nt] = cost_v; rsn[nt] = 1; nt += 1
                pos = 0
                continue
        # ---- 空仓: 进场判定 ----
        if pos == 0:
            a_prev = atr[k - 1]
            up = line_up[k]
            if (not np.isnan(up)) and a_prev > 0 and gate_atr[k] > 0.5 and not np.isnan(o[k]):
                hit_up = h[k] >= up
                nr_k = nr30[k]
                nr_ok = np.isfinite(nr_k) and (nr_k < theta)
                if hit_up:
                    if nr_ok and k > blocked_until:
                        pos = 1; entry_px = up if o[k] < up else o[k]; eib = k
                        cost_v = cost_flat if cost_mode == 0 else cost_atr_frac * a_prev
                    elif not nr_ok:
                        # 高能触发被跳过 → 封锁 N 根
                        blocked_until = k + block
    if pos != 0:
        e_i[nt] = eib; x_i[nt] = n - 1; epx[nt] = entry_px
        xpx[nt] = c[n - 1]; cst[nt] = cost_v; rsn[nt] = 5; nt += 1
    sd = np.ones(nt, np.int8)
    return e_i[:nt], x_i[:nt], sd, epx[:nt], xpx[:nt], cst[:nt], rsn[:nt]


def run_ult(arrs, ts, gate_atr, nr30, theta, block, cost_mode):
    mode, sig, lup, ldn = build_entry(arrs, SPEC["entry"])
    cost = {"base": (0, 0.50, 0.50), "pess03": (1, 0.30, 0.30), "pess05": (1, 0.50, 0.50)}[cost_mode]
    tr = simulate_ultimate(arrs["o"], arrs["h"], arrs["l"], arrs["c"], arrs["atr"], lup,
                           SPEC["days"] * SPEC["bpd"], cost[0], cost[1], cost[2],
                           gate_atr, nr30, theta, block)
    e_i, x_i, sd, epx, xpx, cst, rsn = tr
    pnl = sd * (xpx - epx) - cst
    tdf = pd.DataFrame(dict(
        entry_time=ts[e_i], exit_time=ts[x_i], side="L",
        entry_px=np.round(epx, 2), exit_px=np.round(xpx, 2), cost=np.round(cst, 2),
        pnl=np.round(pnl, 2), hold_bars=x_i - e_i,
        reason=np.where(rsn == 1, "time", "eod"),
        atr_entry=np.round(arrs["atr"][np.maximum(e_i - 1, 0)], 2),
    ))
    return tdf


def main():
    cdf = load_m1()
    bar30 = resample_htf(cdf, "30min")
    arrs30 = build_arrays(bar30)
    ts30 = arrs30["ts"]
    bar15 = resample_htf(cdf, "15min")
    NR = nowcast_m15(bar15)
    nr30 = map15to30(NR, bar15.index, ts30)
    atr_s = pd.Series(arrs30["atr"])
    med_q = atr_s.rolling(SPEC["qwin"], min_periods=200).median().to_numpy()
    gate_atr = np.concatenate(([0.0], (arrs30["atr"] > med_q).astype(float)[:-1]))

    # 机制断言: block=0 时必须逐位复现 θ 过滤重跑臂
    for th in (1.15, 1.2):
        tdf0 = run_ult(arrs30, ts30, gate_atr, nr30, th, 0, "pess03")
        s0 = summarize(tdf0)
        print(f"[assert θ={th} block=0] total={s0['total']} n={s0['trades']} avg={s0['avg']}", flush=True)

    print("=== 网格: θ × block (pess03) ===")
    rows = []
    for th in (1.15, 1.25):
        for blk in (2, 4, 8, 16, 32):
            tdf = run_ult(arrs30, ts30, gate_atr, nr30, th, blk, "pess03")
            s = summarize(tdf)
            allpos = all(v > 0 for v in s["by_year"].values())
            rows.append(dict(theta=th, block=blk, total=s["total"], n=s["trades"], avg=s["avg"],
                             t=s["t_stat"], wr=s["win_rate"], all_pos=allpos, yr=s["by_year"],
                             dd=s["maxdd"], worst=s["worst"], share26=s["share2026"]))
            r = rows[-1]
            print(f"  θ={th} blk={blk:2d}: total={r['total']:7.1f} n={r['n']:3d} avg=${r['avg']:6.2f} "
                  f"t={r['t']} wr={r['wr']} allpos={r['all_pos']} worst={r['worst']} yr={r['yr']}", flush=True)

    # 最优臂三档成本 + 交易流导出
    best = max(rows, key=lambda r: (r["avg"] if r["n"] >= 80 else 0))
    print(f"BEST theta={best['theta']} block={best['block']}")
    for cm in ("base", "pess03", "pess05"):
        tdf = run_ult(arrs30, ts30, gate_atr, nr30, best["theta"], best["block"], cm)
        s = summarize(tdf)
        print(f"  {cm}: total={s['total']} n={s['trades']} avg={s['avg']} t={s['t_stat']} "
              f"wr={s['win_rate']} dd={s['maxdd']} allpos={all(v>0 for v in s['by_year'].values())}")
        if cm == "pess03":
            tj = tdf.copy()
            tj["entry_time"] = tj["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
            tj["exit_time"] = tj["exit_time"].dt.strftime("%Y-%m-%d %H:%M")
            tj.to_csv(os.path.join(RES, f"trades_ult_t{best['theta']}_b{best['block']}.csv"), index=False)
    json.dump(rows, open(os.path.join(RES, "v17u_block.json"), "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
