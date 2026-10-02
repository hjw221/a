#!/usr/bin/env python3
"""v17b_refine.py — v17 细化: 持仓期扫描 + turtle N + 方向拆分 + 波动率闸门(规则版Regime Gate预演).

v17 初筛事实:
  - 吊灯(2.5/3/3.5 ATR)在 HTF 正确结算下不出现在 pess03 top10; 赢家=time/turtle 结构出场
  - pess03 全正仅 M30 don20s+time2d; 笔均$5-15可达(H1 don55s+time5d $12.96)但2024转负
  - 2022(区间年)是负年份重灾区(188/210配置为负) -> 低波绞肉 vs 波动闸门假设
  - 赢家持仓中位数 19-240 bar = 多日动量延续

本轮轴:
  exits: time{1,2,3,5,8,13}d, tur10, tur20, time5d+灾难吊灯8ATR, time13d+灾难吊灯8ATR
  direction: both / long / short
  gate: none / ATR14>过去一季度自身中位数 / ATR14>p30分位 (因果: 只用k-1及以前)
  预注册选冠(跑前锁定): pess03下 逐年全正 AND avg>=5 AND trades>=150 AND share26<=70,
    按 total 排序; 稳健性=base/pess05全正 + 邻域平台>=50% + 多空分解披露
"""
import os, json, time
import numpy as np
import pandas as pd
from numba import njit
from v17_htf import load_m1, resample_htf, build_arrays, build_entry, simulate_core, simulate_jit, COSTS
from v17_htf import ENTRY_CLOSE, ENTRY_STOP, EXIT_MODE_CHAND_TIME, EXIT_MODE_TIME, EXIT_MODE_TURTLE

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results_v17")


def simulate_core_g(o, h, l, c, atr, entry_mode, sig, line_up, line_dn,
                    exit_mode, w, hzn, tp_m, sl_m, dxlo, dxhi,
                    cost_mode, cost_flat, cost_atr_frac, gate):
    """v17 simulate_core + gate(1/0数组, k根进场需gate[k]>0.5; gate在k-1收盘时已知)."""
    n = len(o)
    e_i = np.empty(n, np.int64)
    x_i = np.empty(n, np.int64)
    sd = np.empty(n, np.int8)
    epx = np.empty(n, np.float64)
    xpx = np.empty(n, np.float64)
    cst = np.empty(n, np.float64)
    rsn = np.empty(n, np.int8)
    nt = 0
    pos = 0
    entry_px = 0.0
    hi_s = 0.0
    lo_s = 0.0
    eib = 0
    tp = 0.0
    sl = 0.0
    cost_v = 0.0
    pending = False
    for k in range(1, n):
        if pos != 0 and pending:
            e_i[nt] = eib; x_i[nt] = k; sd[nt] = pos; epx[nt] = entry_px
            xpx[nt] = o[k]; cst[nt] = cost_v; rsn[nt] = 6; nt += 1
            pos = 0; pending = False
            continue
        if pos == 0:
            a_prev = atr[k - 1]
            g = gate[k] > 0.5
            if entry_mode == ENTRY_CLOSE:
                s = sig[k - 1]
                if s != 0 and g and a_prev > 0 and not np.isnan(o[k]):
                    pos = s; entry_px = o[k]; eib = k
                    hi_s = entry_px; lo_s = entry_px
                    cost_v = cost_flat if cost_mode == 0 else cost_atr_frac * a_prev
                    if pos == 1:
                        tp = entry_px + tp_m * a_prev
                        sl = entry_px - sl_m * a_prev
                    else:
                        tp = entry_px - tp_m * a_prev
                        sl = entry_px + sl_m * a_prev
            else:
                up = line_up[k]; dn = line_dn[k]
                if g and (not np.isnan(up)) and (not np.isnan(dn)) and a_prev > 0:
                    hit_up = h[k] >= up
                    hit_dn = l[k] <= dn
                    if hit_up and not hit_dn:
                        pos = 1; entry_px = up if o[k] < up else o[k]; eib = k
                        hi_s = entry_px; lo_s = entry_px
                        cost_v = cost_flat if cost_mode == 0 else cost_atr_frac * a_prev
                        tp = entry_px + tp_m * a_prev
                        sl = entry_px - sl_m * a_prev
                    elif hit_dn and not hit_up:
                        pos = -1; entry_px = dn if o[k] > dn else o[k]; eib = k
                        hi_s = entry_px; lo_s = entry_px
                        cost_v = cost_flat if cost_mode == 0 else cost_atr_frac * a_prev
                        tp = entry_px - tp_m * a_prev
                        sl = entry_px + sl_m * a_prev
        if pos != 0:
            a_prev = atr[k - 1]
            done = False
            if exit_mode == EXIT_MODE_CHAND_TIME and a_prev > 0 and w > 0:
                if pos == 1:
                    stop = hi_s - w * a_prev
                    if l[k] <= stop:
                        px = stop if o[k] >= stop else o[k]
                        e_i[nt] = eib; x_i[nt] = k; sd[nt] = pos; epx[nt] = entry_px
                        xpx[nt] = px; cst[nt] = cost_v; rsn[nt] = 0; nt += 1
                        pos = 0; done = True
                else:
                    stop = lo_s + w * a_prev
                    if h[k] >= stop:
                        px = stop if o[k] <= stop else o[k]
                        e_i[nt] = eib; x_i[nt] = k; sd[nt] = pos; epx[nt] = entry_px
                        xpx[nt] = px; cst[nt] = cost_v; rsn[nt] = 0; nt += 1
                        pos = 0; done = True
            if not done and (exit_mode == EXIT_MODE_CHAND_TIME or exit_mode == EXIT_MODE_TIME) and hzn > 0:
                if k >= eib + hzn:
                    e_i[nt] = eib; x_i[nt] = k; sd[nt] = pos; epx[nt] = entry_px
                    xpx[nt] = o[k]; cst[nt] = cost_v; rsn[nt] = 1; nt += 1
                    pos = 0; done = True
            if not done:
                if h[k] > hi_s:
                    hi_s = h[k]
                if l[k] < lo_s:
                    lo_s = l[k]
                if exit_mode == EXIT_MODE_TURTLE and not np.isnan(dxlo[k]):
                    if pos == 1 and c[k] < dxlo[k]:
                        pending = True
                    elif pos == -1 and c[k] > dxhi[k]:
                        pending = True
    if pos != 0:
        e_i[nt] = eib; x_i[nt] = n - 1; sd[nt] = pos; epx[nt] = entry_px
        xpx[nt] = c[n - 1]; cst[nt] = cost_v; rsn[nt] = 5; nt += 1
    return e_i[:nt], x_i[:nt], sd[:nt], epx[:nt], xpx[:nt], cst[:nt], rsn[:nt]


simulate_g_jit = njit(cache=True)(simulate_core_g)
from v17_htf import metrics  # noqa: E402

TF_SPECS = [
    ("M15", "15min", 96, 6048),
    ("M30", "30min", 48, 3024),
    ("H1",  "1h",    24, 1512),
]
ENTRIES = ["don20s", "don40s", "don55s", "don20", "don40", "don55", "kelt20_2.0", "kelt20_2.5"]
DAYS = [1, 2, 3, 5, 8, 13]


def main():
    t00 = time.time()
    cdf = load_m1()
    results = []
    for tf, rule, bpd, qwin in TF_SPECS:
        bar = resample_htf(cdf, rule)
        arrs = build_arrays(bar)
        ts = arrs["ts"]
        atr = arrs["atr"]
        atr_s = pd.Series(atr)
        med_q = atr_s.rolling(qwin, min_periods=200).median().to_numpy()
        p30_q = atr_s.rolling(qwin, min_periods=200).quantile(0.30).to_numpy()
        gates = {
            "none": np.ones(arrs["n"]),
            "atrmed": (atr > med_q).astype(float),      # ATR高于自身过去一季度中位数
            "atrp30": (atr > p30_q).astype(float),
        }
        # 因果校正: k根决策只能用k-1及以前 -> gate右移一根
        for gk in ("atrmed", "atrp30"):
            g = gates[gk].copy()
            gates[gk] = np.concatenate(([0.0], g[:-1]))
        exits = [dict(name=f"t{d}d", mode=EXIT_MODE_TIME, w=0.0, hzn=d * bpd, tp=0.0, sl=0.0) for d in DAYS]
        exits += [
            dict(name="tur10", mode=EXIT_MODE_TURTLE, w=0.0, hzn=0, tp=0.0, sl=0.0),
            dict(name="tur20", mode=EXIT_MODE_TURTLE, w=0.0, hzn=0, tp=0.0, sl=0.0),
            dict(name="t5d_ch8", mode=EXIT_MODE_CHAND_TIME, w=8.0, hzn=5 * bpd, tp=0.0, sl=0.0),
            dict(name="t13d_ch8", mode=EXIT_MODE_CHAND_TIME, w=8.0, hzn=13 * bpd, tp=0.0, sl=0.0),
        ]
        # turkey 出场线: tur20 需要 rolling(20)
        arrs["dxlo20"] = pd.Series(arrs["l"]).rolling(20).min().shift(1).to_numpy()
        arrs["dxhi20"] = pd.Series(arrs["h"]).rolling(20).max().shift(1).to_numpy()
        # 双引擎互验 + 与v17引擎中性对照
        mode, sig, lup, ldn = build_entry(arrs, "don20s")
        ones = np.ones(arrs["n"])
        a_chk = (arrs["o"], arrs["h"], arrs["l"], arrs["c"], arrs["atr"], mode, sig, lup, ldn,
                 EXIT_MODE_CHAND_TIME, 8.0, 5 * bpd, 0.0, 0.0, arrs["dxlo10"], arrs["dxhi10"],
                 1, 0.30, 0.30, ones)
        r_py = simulate_core_g(*a_chk)
        r_jit = simulate_g_jit(*a_chk)
        for nm, A, B in zip(["e", "x", "sd", "ep", "xp", "cs", "rs"], r_py, r_jit):
            assert len(A) == len(B) and np.array_equal(A, B), f"MISMATCH {tf}.{nm}"
        # 中性gate/dir下 与 v17 引擎逐位一致 (t5d_ch8无tp/sl分支差异)
        r_v17 = simulate_jit(arrs["o"], arrs["h"], arrs["l"], arrs["c"], arrs["atr"], mode, sig, lup, ldn,
                              EXIT_MODE_CHAND_TIME, 8.0, 5 * bpd, 0.0, 0.0,
                              arrs["dxlo10"], arrs["dxhi10"], 1, 0.30, 0.30)
        for nm, A, B in zip(["e", "x", "sd", "ep", "xp", "cs", "rs"], r_v17, r_jit):
            assert len(A) == len(B) and np.array_equal(A, B), f"XENGINE MISMATCH {tf}.{nm}"
        print(f"[{tf}] dual + cross-engine check OK ({len(r_jit[0])} trades)", flush=True)
        for kind in ENTRIES:
            mode, sig, lup, ldn = build_entry(arrs, kind)
            for dname in ("both", "long", "short"):
                if dname == "long":
                    sig_m = np.where(sig > 0, sig, 0).astype(np.int8)
                    lup_m, ldn_m = lup, np.full(arrs["n"], -1e18)
                elif dname == "short":
                    sig_m = np.where(sig < 0, sig, 0).astype(np.int8)
                    lup_m, ldn_m = np.full(arrs["n"], 1e18), ldn
                else:
                    sig_m, lup_m, ldn_m = sig, lup, ldn
                for gname, gate in gates.items():
                    for ex in exits:
                        dxlo = arrs["dxlo10"] if ex["name"] in ("tur10",) else arrs["dxlo20"]
                        dxhi = arrs["dxhi10"] if ex["name"] in ("tur10",) else arrs["dxhi20"]
                        if ex["name"] not in ("tur10", "tur20"):
                            dxlo, dxhi = arrs["dxlo10"], arrs["dxhi10"]
                        for cost in COSTS:
                            tr = simulate_g_jit(arrs["o"], arrs["h"], arrs["l"], arrs["c"], arrs["atr"],
                                                 mode, sig_m, lup_m, ldn_m,
                                                 ex["mode"], ex["w"], ex["hzn"], ex["tp"], ex["sl"],
                                                 dxlo, dxhi, cost[1], cost[2], cost[2], gate)
                            m = metrics(dict(zip(["e_i", "x_i", "sd", "epx", "xpx", "cst", "rsn"], tr)),
                                        ts, float(np.nanmedian(arrs["sp"])) * 0.001)
                            m.update(tf=tf, entry=kind, dir=dname, gate=gname, exit=ex["name"], cost=cost[0])
                            results.append(m)
        sub = pd.DataFrame([{k: v for k, v in r.items() if k != "months"} for r in results if r["tf"] == tf])
        d3 = sub[(sub["cost"] == "pess03") & (sub["all_pos"])]
        print(f"[{tf}] {len(sub)} runs | pess03 all_pos={len(d3)} | best_pass_total={d3['total'].max() if len(d3) else None}", flush=True)
    json.dump(results, open(os.path.join(RES, "htf_refine.json"), "w"), indent=1)
    print(f"DONE {(time.time() - t00) / 60:.1f}min -> results_v17/htf_refine.json ({len(results)} runs)", flush=True)


if __name__ == "__main__":
    main()
