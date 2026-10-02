#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v19b_refine.py — v19 第二轮: 诊断驱动的事后假设检验 (明确标注: 非预注册)
============================================================================

第一轮预注册结果 (v19_r1gate.json):
  路线A(替换闸门)全灭: 6/6 臂 total $1,437~$1,957 < 基线 $2,980, 全部出现负年份(2026)
  路线B(独立事件驱动): −$3,286 / 3,126笔 / 13.5笔每周 / t=−3.45 / 逐年全负
  oracle 质量无问题: IC_var 逐年 0.25~0.42 稳定为真
  失败签名: 闸门重叠仅 49% (both-ON 26% / r1-only 22% / atrmed-only 27%)
            → r1-only 窗口的交易笔均 $11.4 (基线 $18.7); atrmed-only 窗口含 2026 大赢家

第二轮假设 (诊断推导, 非预注册, 如实标注):
  H1 视界失配: 用户规格"未来1–4小时"上限 = 4h = h16(M15根). 重跑同一施密特闸门机器
     于 h16 oracle; 另加长时锁 {48,96} 根 × 低阈 {0.90,1.00} (贴近 regime 尺度的
     平滑能量过滤变体). 若仍 < 基线 → 视界假设被否定.
  H2 先见无用控制 (决定性对照): 因果 nowcast NR(t)=当前4bar实现方差/EMA96基线
     (与 y_var 同公式, 无 shift = 无先见, 无学习). 同一闸门机器. 若 nowcast 闸门 ≈
     R̂ 闸门 → 储层学习在该尺度零增量, 路线A归约为"更快的ATR过滤器", 而冠军边际
     生活在 63 天 regime 尺度 (R1 阶段一: IC_var 在 24h 归零).
  H3 入场条件诊断: 基线 159 笔入场时点的 R̂/NR 与单笔 PnL 的条件统计 —
     能否按预测能量"选交易" (2022 净化假设的直接检验).

成本与因果口径与第一轮/v17 完全同屋.
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import v17_htf  # noqa: E402
from v17_htf import resample_htf, build_arrays  # noqa: E402
from v19_r1gate import (DATA, ORACLE_CFG, COSTS, schmitt_gate, gate_to_m30,  # noqa: E402  (先导入: 建立 reservoir sys.path)
                        run_arm, coverage_by_year, m15_from_cdf)
import reservoir_engine as R1  # noqa: E402
from v17c_champ import summarize  # noqa: E402

RES = os.path.join(BASE, "results_v19")
v17_htf.DATA_CSV = DATA

H1_GRID = [(on, round(on - 0.15, 2), lo)
           for on in (1.15, 1.25, 1.35) for lo in (8, 16)]
H1_LONG = [(on, round(on - 0.15, 2), lo) for on in (0.90, 1.00) for lo in (48, 96)]
H2_GRID = [(on, round(on - 0.15, 2), lo)
           for on in (1.15, 1.25) for lo in (8, 16)]
H2_LONG = [(on, round(on - 0.15, 2), lo) for on in (0.90, 1.00) for lo in (48, 96)]
MIN_OFF = 4


def nowcast_series(D):
    """因果 nowcast: NR(t) = 当前4bar实现方差 / EMA96基线 (y_var 同公式, 无 shift)."""
    r2 = D["ret"] ** 2
    v4p = pd.Series(r2).rolling(4).sum().to_numpy()
    base = pd.Series(np.nan_to_num(v4p)).ewm(span=96, adjust=False).mean().to_numpy()
    return np.clip(np.nan_to_num(v4p / np.maximum(base, 1e-14)), 0.0, 6.0)


def entry_cond_stats(tdf_base, series, m15_ts, ts30, label):
    """基线159笔: 入场时点 series 值 (决策M15 bar = 最后先于 o30[e_i] 收盘者) 与 PnL 条件统计."""
    j = np.searchsorted(m15_ts.asi8, ts30.asi8, side="left") - 1
    ok = j >= 0
    jm = np.where(ok, j, 0)
    vals = series[jm]  # 每个 M30 bar 开盘前的 oracle/nowcast 值
    e_i = tdf_base.index.to_numpy()  # tdf 按序, 行号即交易序 — 需要真实 bar 索引
    # 重算真实 e_i: 用 entry_time 匹配 ts30
    pos_map = {ts: i for i, ts in enumerate(ts30)}
    ei = np.array([pos_map[t] for t in pd.to_datetime(tdf_base["entry_time"])], dtype=int)
    rv = vals[ei]
    pnl = tdf_base["pnl"].to_numpy(float)
    yrs = pd.to_datetime(tdf_base["exit_time"]).dt.year.to_numpy()
    m = np.isfinite(rv) & (rv > 0)
    hi = rv >= 1.25
    lo_ = rv < 1.25
    def _cs(mask):
        if mask.sum() < 5:
            return None
        return dict(n=int(mask.sum()), avg=round(float(pnl[mask].mean()), 2),
                    wr=round(float((pnl[mask] > 0).mean()), 3))
    m22 = m & (yrs == 2022)
    corr = float(np.corrcoef(rv[m], pnl[m])[0, 1]) if m.sum() > 10 else 0.0
    return dict(label=label, corr_pnl=round(corr, 4),
                r_hi=_cs(hi), r_lo=_cs(lo_), r_2022_hi=_cs(m22 & hi), r_2022_lo=_cs(m22 & lo_))


def main():
    t00 = time.time()
    print("== v19b · 第二轮 (事后假设检验) ==", flush=True)
    cdf = v17_htf.load_m1()
    bar30 = resample_htf(cdf, "30min")
    arrs30 = build_arrays(bar30)
    ts30 = arrs30["ts"]
    atr_s = pd.Series(arrs30["atr"])
    med_q = atr_s.rolling(3024, min_periods=200).median().to_numpy()
    gate_atrmed = np.concatenate(([0.0], (arrs30["atr"] > med_q).astype(float)[:-1]))
    bar15 = m15_from_cdf(cdf)
    D4 = R1.build_feats(bar15, horizon=4)
    warm = ORACLE_CFG["warm"]
    m15_ts = D4["ts"]

    # ---- 重放 h4 oracle (与第一轮同种子同配置, 提取 R/S 供诊断) ----
    t_or = time.time()
    orc4 = R1.run_pass(D4, ORACLE_CFG, 42, collect=False, label="h4diag")
    R4 = np.clip(orc4["yhat"][:, 0], 0.0, 6.0)
    print(f"[h4 replay] {time.time()-t_or:.0f}s", flush=True)

    # ---- H1: h16 oracle (4h, 用户规格上限) ----
    t_or = time.time()
    D16 = R1.build_feats(bar15, horizon=16)
    cfg16 = dict(ORACLE_CFG)
    cfg16["horizon"] = 16
    orc16 = R1.run_pass(D16, cfg16, 42, collect=False, label="h16")
    R16 = np.clip(orc16["yhat"][:, 0], 0.0, 6.0)
    print(f"[h16 oracle] {time.time()-t_or:.0f}s", flush=True)

    # ---- H2: 因果 nowcast (无学习对照) ----
    NR = nowcast_series(D4)

    # ---- 基线 (复现第一轮) ----
    tdf_base = run_arm(arrs30, ts30, gate_atrmed, "pess03")
    s_base = summarize(tdf_base)
    base_total = s_base["total"]

    def eval_grid(series, grid, tag):
        recs = []
        best = None
        for (on, off, lo) in grid:
            st = schmitt_gate(series, on, off, lo, MIN_OFF, warm)
            g30 = gate_to_m30(st, m15_ts, ts30)
            tdf = run_arm(arrs30, ts30, g30, "pess03")
            s = summarize(tdf)
            allpos = all(v > 0 for v in s["by_year"].values())
            name = f"{tag}_on{on:.2f}_lo{lo}"
            recs.append(dict(name=name, theta_on=on, theta_off=off, min_on=lo,
                             pess03=s, all_pos=allpos,
                             coverage=coverage_by_year(g30, ts30)))
            print(f"[{name}] total={s['total']} n={s['trades']} "
                  f"yr={s['by_year']} allpos={allpos}", flush=True)
            if allpos and s["trades"] >= 100 and (best is None or s["total"] > best[1]):
                best = (g30, s["total"], tdf, name)
        if best is None:
            bx = max(recs, key=lambda r: r["pess03"]["total"])
            st = schmitt_gate(series, bx["theta_on"], bx["theta_off"], bx["min_on"], MIN_OFF, warm)
            g30 = gate_to_m30(st, m15_ts, ts30)
            tdf = run_arm(arrs30, ts30, g30, "pess03")
            best = (g30, bx["pess03"]["total"], tdf, bx["name"])
        return recs, best

    print("-- H1: h16(4h) oracle 施密特闸门 --", flush=True)
    h1_recs, h1_best = eval_grid(R16, H1_GRID, "h16")
    print("-- H1b: h16 长时锁 regime 变体 --", flush=True)
    h1b_recs, h1b_best = eval_grid(R16, H1_LONG, "h16L")
    print("-- H2: nowcast 对照 (无学习) --", flush=True)
    h2_recs, h2_best = eval_grid(NR, H2_GRID, "nc")
    print("-- H2b: nowcast 长时锁 --", flush=True)
    h2b_recs, h2b_best = eval_grid(NR, H2_LONG, "ncL")

    # ---- intersect 变体 (h16 best / nowcast best) ----
    inter = {}
    for tag, (g30, tot, tdf, name) in (("h16", h1_best), ("nc", h2_best)):
        gi = np.minimum(g30, gate_atrmed)
        tdf_i = run_arm(arrs30, ts30, gi, "pess03")
        s_i = summarize(tdf_i)
        inter[tag] = dict(with_gate=name, pess03=s_i,
                          coverage=coverage_by_year(gi, ts30))
        print(f"[INTERSECT {tag} x atrmed] total={s_i['total']} n={s_i['trades']} "
              f"yr={s_i['by_year']}", flush=True)

    # ---- H3: 入场条件诊断 (基线159笔) ----
    h3 = dict(
        r4=entry_cond_stats(tdf_base, R4, m15_ts, ts30, "R̂(1h前瞻)"),
        r16=entry_cond_stats(tdf_base, R16, m15_ts, ts30, "R̂(4h前瞻)"),
        nr=entry_cond_stats(tdf_base, NR, m15_ts, ts30, "NR(现在cast)"),
    )
    for k, v in h3.items():
        print(f"[H3 {k}] corr={v['corr_pnl']} hi={v['r_hi']} lo={v['r_lo']} "
              f"2022hi={v['r_2022_hi']} 2022lo={v['r_2022_lo']}", flush=True)

    # ---- 汇总判定 ----
    all_round2 = [("h16", h1_best), ("h16L", h1b_best), ("nc", h2_best), ("ncL", h2b_best)]
    best_r2 = max(all_round2, key=lambda kv: kv[1][1])
    any_beat = [(tag, b[1]) for tag, b in all_round2 if b[1] > base_total]
    inter_beat = [(t, v["pess03"]["total"]) for t, v in inter.items()
                  if v["pess03"]["total"] > base_total]
    verdict2 = dict(
        round2_nature="事后假设检验(非预注册), 全部结果如实报告",
        best_round2_arm=best_r2[1][3], best_round2_total=best_r2[1][1],
        base_total=base_total,
        any_replacement_beats=bool(any_beat), beating_arms=any_beat,
        any_intersect_beats=bool(inter_beat), intersect_beats=inter_beat,
        h1_verdict=("h16(4h) 前瞻闸门 "
                    + ("仍有臂超基线" if max(h1_best[1], h1b_best[1]) > base_total
                       else "无任何臂超基线 → 视界假设否定")),
        h2_verdict=(f"无学习 nowcast 最佳 {max(h2_best[1], h2b_best[1])} "
                    + (">" if max(h2_best[1], h2b_best[1]) > 1957.0 else "<")
                    + " 全部学习前瞻闸门(第一轮网格最佳 1,957 / h16 最佳 "
                    + f"{max(h1_best[1], h1b_best[1])}) → 学习先见在该尺度为"
                    + ("正增量" if max(h2_best[1], h2b_best[1]) > 1957.0 else "负增量(先见不如持续性)")
                    + "; 但仍不敌基线 " + f"{base_total}"),
    )
    print(f"[verdict2] {verdict2}", flush=True)

    doc = dict(
        meta=dict(engine="v19b_refine", generated=pd.Timestamp.now().isoformat(),
                  round="2 (post-hoc, hypotheses H1/H2/H3 as stated in docstring)",
                  wall_s=round(time.time() - t00, 1)),
        h1_grid=h1_recs, h1_long=h1b_recs, h2_grid=h2_recs, h2_long=h2b_recs,
        intersect=inter, h3_entry_cond=h3, verdict2=verdict2,
        base=dict(pess03=s_base, coverage=coverage_by_year(gate_atrmed, ts30)),
    )
    out = os.path.join(RES, "v19b_round2.json")
    with open(out, "w") as f:
        json.dump(doc, f, indent=1, default=str)
    print(f"== DONE -> {out} ({os.path.getsize(out)} bytes, wall {time.time()-t00:.0f}s) ==", flush=True)


if __name__ == "__main__":
    main()
