#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v17u_refine.py — 终极版规格细化: θ峰值区间 + 新交易稀释分析 + NR分位条件统计."""
import os, sys, json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import v17_htf
from v17_htf import load_m1, resample_htf, build_arrays, build_entry, EXIT_MODE_TIME
from v17b_refine import simulate_core_g
from v17c_champ import summarize
from v17u_test import run_gated, nowcast_m15, map15to30, SPEC

v17_htf.DATA_CSV = os.path.join(BASE, "data.csv")
RES = os.path.join(BASE, "results_v17u")


def main():
    cdf = load_m1()
    bar30 = resample_htf(cdf, "30min")
    arrs30 = build_arrays(bar30)
    ts30 = arrs30["ts"]
    bar15 = resample_htf(cdf, "15min")
    ts15 = bar15.index
    NR = nowcast_m15(bar15)
    nr30 = map15to30(NR, ts15, ts30)

    atr_s = pd.Series(arrs30["atr"])
    med_q = atr_s.rolling(SPEC["qwin"], min_periods=200).median().to_numpy()
    gate_atr = np.concatenate(([0.0], (arrs30["atr"] > med_q).astype(float)[:-1]))
    tdf_base = run_gated(arrs30, ts30, gate_atr, "pess03")
    pos_map = {ts: i for i, ts in enumerate(ts30)}
    ei = np.array([pos_map[t] for t in tdf_base["entry_time"]], dtype=int)
    rv = nr30[ei]
    pnlb = tdf_base["pnl"].to_numpy(float)
    yrsb = pd.to_datetime(tdf_base["exit_time"]).dt.year.to_numpy()

    # ---- NR 分位条件统计 (基线159笔入场NR的十分位 → PnL) ----
    m = np.isfinite(rv) & (rv > 0)
    qs = pd.qcut(rv[m], 5, labels=False, duplicates="drop")
    print("=== 基线入场 NR 五分位 → 笔均/胜率 ===")
    for q in range(5):
        mask = m.copy(); mask[m] = qs == q
        print(f"  Q{q+1}: n={mask.sum():3d} avg=${pnlb[mask].mean():6.2f} wr={(pnlb[mask]>0).mean():.3f} "
              f"2022avg=${pnlb[mask & (yrsb==2022)].mean() if (mask & (yrsb==2022)).sum() else float('nan'):6.2f}")

    # ---- θ 细化网格 + 诊断/重跑双口径 ----
    print("=== θ 细化 (pess03) ===")
    rows = []
    for th in [1.05, 1.10, 1.15, 1.20]:
        g = gate_atr * np.where(np.nan_to_num(nr30, nan=9.0) < th, 1.0, 0.0)
        tdf = run_gated(arrs30, ts30, g, "pess03")
        s = summarize(tdf)
        # 诊断口径: 基线交易清单过滤
        dmask = m & (rv < th)
        # 新交易分析: 重跑交易不在基线入场时刻集合里的
        base_entries = set(tdf_base["entry_time"])
        new_mask = ~tdf["entry_time"].isin(base_entries).to_numpy()
        pnlr = tdf["pnl"].to_numpy(float)
        allpos = all(v > 0 for v in s["by_year"].values())
        rows.append(dict(theta=th, total=s["total"], n=s["trades"], avg=s["avg"], t=s["t_stat"],
                         all_pos=allpos, yr=s["by_year"], dd=s["maxdd"], worst=s["worst"],
                         diag_n=int(dmask.sum()), diag_avg=round(float(pnlb[dmask].mean()), 2),
                         new_n=int(new_mask.sum()), new_avg=round(float(pnlr[new_mask].mean()), 2),
                         kept_avg=round(float(pnlr[~new_mask].mean()), 2)))
        r = rows[-1]
        print(f"  θ={th}: RERUN total={r['total']} n={r['n']} avg=${r['avg']} t={r['t']} allpos={r['all_pos']} "
              f"| DIAG n={r['diag_n']} avg=${r['diag_avg']} | new-trades n={r['new_n']} avg=${r['new_avg']} "
              f"kept avg=${r['kept_avg']} worst={r['worst']}")

    # base 成本视角
    print("=== θ 细化 (base 成本) ===")
    for th in [1.10, 1.15, 1.20]:
        g = gate_atr * np.where(np.nan_to_num(nr30, nan=9.0) < th, 1.0, 0.0)
        tdf = run_gated(arrs30, ts30, g, "base")
        s = summarize(tdf)
        print(f"  θ={th}: base total={s['total']} n={s['trades']} avg=${s['avg']} allpos={all(v>0 for v in s['by_year'].values())}")

    json.dump(rows, open(os.path.join(RES, "v17u_refine.json"), "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
