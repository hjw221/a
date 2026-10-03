#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v17u_test.py — v17 终极版候选: 冠军 + 因果低能量 nowcast 入场过滤器
====================================================================
用户指令: "把那个均盈利24以上的v17终极版写一个mq5"

来源线索 (v19 第二轮 H3 诊断): 基线 159 笔入场时点的因果 nowcast
NR(t)=当前4根M15实现方差/EMA96基线 — NR<1.25 的入场笔均 $24.57 (83笔, wr 0.614)
vs NR>=1.25 的 $12.38 (76笔)。NR 是纯因果定义(无shift无学习), 阈值 1.25 与
v19 nc 臂同屋 — 可部署。

本脚本做三件事:
  A. 基线复现断言: CHAMP_M30_don55s_long_atrmed_t5d pess03 = $2,980.2 / 159笔 / $18.74
  B. H3 诊断复现: 基线交易按入场时点 NR<1.25 过滤 → 83笔 / $24.57 / wr 0.614
  C. 完整重跑网格(串行单仓, gate=atrmed ∧ NR过滤, 同引擎同成本):
     - 阈值阶梯 θ ∈ {0.8, 1.0, 1.15, 1.25, 1.4, 1.5, 1.75, 2.0} (稳健性/平台)
     - 滚动中位变体(免常数): NR < 自身过去一季度(6048根M15)滚动中位
     - 三档成本 (base / pess03 / pess05)
预注册判据: 笔均>=24 ∧ 逐年全正 ∧ 笔数>=80 ∧ maxDD不低于基线量级。
"""
import os, sys, json, time
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import v17_htf
from v17_htf import load_m1, resample_htf, build_arrays, build_entry, EXIT_MODE_TIME
from v17b_refine import simulate_core_g
from v17c_champ import summarize

v17_htf.DATA_CSV = os.path.join(BASE, "data.csv")

RES = os.path.join(BASE, "results_v17u")
os.makedirs(RES, exist_ok=True)

SPEC = dict(tf="M30", rule="30min", bpd=48, qwin=3024, entry="don55s", dir="long",
            gate="atrmed", exit="t5d", days=5)


def run_gated(arrs, ts, gate, cost_mode):
    mode, sig, lup, ldn = build_entry(arrs, SPEC["entry"])
    sig = np.where(sig > 0, sig, 0).astype(np.int8)   # long only
    ldn = np.full(arrs["n"], -1e18)
    cost = {"base": (0, 0.50, 0.50), "pess03": (1, 0.30, 0.30), "pess05": (1, 0.50, 0.50)}[cost_mode]
    tr = simulate_core_g(arrs["o"], arrs["h"], arrs["l"], arrs["c"], arrs["atr"],
                         mode, sig, lup, ldn, EXIT_MODE_TIME, 0.0, SPEC["days"] * SPEC["bpd"],
                         0.0, 0.0, arrs["dxlo10"], arrs["dxhi10"],
                         cost[0], cost[1], cost[2], gate)
    e_i, x_i, sd, epx, xpx, cst, rsn = tr
    pnl = sd * (xpx - epx) - cst
    tdf = pd.DataFrame(dict(
        entry_time=ts[e_i], exit_time=ts[x_i], side=np.where(sd == 1, "L", "S"),
        entry_px=np.round(epx, 2), exit_px=np.round(xpx, 2), cost=np.round(cst, 2),
        pnl=np.round(pnl, 2), hold_bars=x_i - e_i,
        reason=np.where(rsn == 1, "time", np.where(rsn == 5, "eod", "other")),
        atr_entry=np.round(arrs["atr"][np.maximum(e_i - 1, 0)], 2),
    ))
    return tdf


def nowcast_m15(bar15):
    """因果 nowcast: NR(t) = 当前4bar实现方差 / EMA96基线 (v19 同公式, 无 shift)."""
    c = bar15["close"].to_numpy(float)
    pc = np.concatenate(([c[0]], c[:-1]))
    ret = np.log(c / pc)
    r2 = ret * ret
    v4p = pd.Series(r2).rolling(4).sum().to_numpy()
    base = pd.Series(np.nan_to_num(v4p)).ewm(span=96, adjust=False).mean().to_numpy()
    return np.clip(np.nan_to_num(v4p / np.maximum(base, 1e-14)), 0.0, 6.0)


def map15to30(series15, ts15, ts30):
    """M30 bar k 的决策 NR = 最后先于 ts30[k] 收盘的 M15 bar 的 NR (v19 同约定)."""
    j = np.searchsorted(ts15.asi8, ts30.asi8, side="left") - 1
    ok = j >= 0
    jm = np.where(ok, j, 0)
    vals = series15[jm]
    return np.where(ok, vals, np.nan)


def main():
    t00 = time.time()
    cdf = load_m1()
    print(f"[data] M1 cleaned: {len(cdf)} bars ({time.time()-t00:.0f}s)", flush=True)

    bar30 = resample_htf(cdf, "30min")
    arrs30 = build_arrays(bar30)
    ts30 = arrs30["ts"]
    bar15 = resample_htf(cdf, "15min")
    ts15 = bar15.index
    NR = nowcast_m15(bar15)
    nr30 = map15to30(NR, ts15, ts30)

    # ---- A. 基线复现 ----
    atr_s = pd.Series(arrs30["atr"])
    med_q = atr_s.rolling(SPEC["qwin"], min_periods=200).median().to_numpy()
    gate_atr = np.concatenate(([0.0], (arrs30["atr"] > med_q).astype(float)[:-1]))
    tdf_base = run_gated(arrs30, ts30, gate_atr, "pess03")
    s_base = summarize(tdf_base)
    print(f"[BASE] total={s_base['total']} n={s_base['trades']} avg={s_base['avg']} "
          f"t={s_base['t_stat']} wr={s_base['win_rate']} yr={s_base['by_year']}", flush=True)
    assert abs(s_base["total"] - 2980.2) < 1.0 and s_base["trades"] == 159, "BASE REPRO FAIL"
    print("[BASE] repro assertion OK ($2,980.2 / 159)", flush=True)

    # ---- B. H3 诊断复现 (入场时点 NR<1.25 过滤基线交易清单) ----
    pos_map = {ts: i for i, ts in enumerate(ts30)}
    ei = np.array([pos_map[t] for t in tdf_base["entry_time"]], dtype=int)
    rv = nr30[ei]
    pnl = tdf_base["pnl"].to_numpy(float)
    m = np.isfinite(rv) & (rv > 0)
    lo_mask = m & (rv < 1.25)
    hi_mask = m & (rv >= 1.25)
    print(f"[H3 diag] NR<1.25: n={lo_mask.sum()} avg={pnl[lo_mask].mean():.2f} "
          f"wr={(pnl[lo_mask]>0).mean():.3f} | NR>=1.25: n={hi_mask.sum()} "
          f"avg={pnl[hi_mask].mean():.2f} wr={(pnl[hi_mask]>0).mean():.3f}", flush=True)

    # ---- C. 完整重跑网格 ----
    results = []
    thetas = [0.8, 1.0, 1.15, 1.25, 1.4, 1.5, 1.75, 2.0]
    for th in thetas:
        g = gate_atr * np.where(np.nan_to_num(nr30, nan=9.0) < th, 1.0, 0.0)
        for cm in ("base", "pess03", "pess05"):
            tdf = run_gated(arrs30, ts30, g, cm)
            s = summarize(tdf)
            allpos = all(v > 0 for v in s["by_year"].values())
            results.append(dict(arm=f"nrlo_{th}", theta=th, cost=cm, **{k: v for k, v in s.items() if k != "months"},
                                all_pos=allpos))
            if cm == "pess03":
                print(f"[nrlo θ={th}] total={s['total']} n={s['trades']} avg={s['avg']} "
                      f"t={s['t_stat']} wr={s['win_rate']} allpos={allpos} yr={s['by_year']} "
                      f"dd={s['maxdd']} sh26={s['share2026']}%", flush=True)
                tdf.to_csv(os.path.join(RES, f"trades_nrlo_{th}.csv"), index=False)

    # 滚动中位变体 (免常数)
    nr_med15 = pd.Series(NR).rolling(6048, min_periods=400).median().to_numpy()
    nrmed30 = map15to30(nr_med15, ts15, ts30)
    g = gate_atr * np.where(np.nan_to_num(nr30, nan=9.0) < np.nan_to_num(nrmed30, nan=0.0), 1.0, 0.0)
    for cm in ("base", "pess03", "pess05"):
        tdf = run_gated(arrs30, ts30, g, cm)
        s = summarize(tdf)
        allpos = all(v > 0 for v in s["by_year"].values())
        results.append(dict(arm="nrlo_rollmed", theta="rollmed6048", cost=cm, **{k: v for k, v in s.items() if k != "months"},
                            all_pos=allpos))
        if cm == "pess03":
            print(f"[nrlo rollmed] total={s['total']} n={s['trades']} avg={s['avg']} "
                  f"t={s['t_stat']} wr={s['win_rate']} allpos={allpos} yr={s['by_year']} "
                  f"dd={s['maxdd']} sh26={s['share2026']}%", flush=True)
            tdf.to_csv(os.path.join(RES, "trades_nrlo_rollmed.csv"), index=False)

    json.dump(results, open(os.path.join(RES, "v17u_grid.json"), "w"), indent=1)
    print(f"DONE {time.time()-t00:.0f}s -> results_v17u/v17u_grid.json", flush=True)


if __name__ == "__main__":
    main()
