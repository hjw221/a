#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v19_r1gate.py — 路线A定案: R1 能量神谕 × v17 规则冠军 = 自适应宏观闸门 (2026-10-02)
====================================================================================

用户两条路线 (最后指令, 附"根据你的见解来"):
  路线A: R1 储层引擎输出作 v17 M30 Donchian 突破的实时前瞻闸门
         — 替换 63 天滞后的 ATR 季中位后视镜 → 毫秒级「能量压缩/释放前瞻」
  路线B: R1 独立端到端, 施密特触发器 + 加仓锁 + 事件驱动离散化

定案 (基于 R1 阶段一已测数据, 非直觉):
  主干 = 路线A. 理由: ①能量头 IC 0.343 是洗牌对照验证过的真 alpha, 方向头 IC 0.020
  经济上太弱(毛利 $436/4.5y); ②v17 冠军是已证明的边际($2,980/159笔/t=3.13), 闸门型
  融合的最坏情况有界(退回基线), 独立方向模型的最坏情况是负期望; ③波动率可预测性
  远高于方向(波动聚集+均值回复是物理结构, M15 方向近乎有效市场).
  同时: 路线B 的三大机制(施密特触发/迟滞带/最小时锁)整体移植到【闸门状态机】——
  用户的事件驱动纪律作用对象从仓位改为闸门, 精神完整保留;
  路线B 本体作为同框架对照臂诚实运行, 用数据而非观点回答 A vs B.

闸门语义 (与 v17 冠军闸门同一因果约定, 无回望):
  R1 oracle: M15 bar t 收盘 → ŷ_var_rel(t) = 预测「未来4根M15实现方差 / 过去96bar EMA基线」
             (即未来 1 小时能量前瞻, R̂>1=扩张, <1=压缩; 储层洗出 warm=2000)
  施密特状态机 (M15 网格, 路线B机制移植):
    OFF→ON : R̂ >= θ_on   (能量扩张前瞻亮绿灯)
    ON→OFF : R̂ <= θ_off  (迟滞带 θ_off=θ_on−0.15, 防阈值抖动)
    min_on : ON 后至少锁 L_on 根 M15 (2h/4h), 禁止闪烁关闸
    min_off: OFF 后至少 4 根才可再武装
  M30 采样: gate30[k] = 状态@最后一根在 o30[k] 之前收盘的 M15 bar (≡ k−1 收盘已知,
            与 v17 冠军 gate[k]=(ATR[k−1]>med[k−1]) 完全同屋)

预注册选冠规则 (跑前锁定, 沿用 v17 纪律):
  成功   = R1 闸门臂 pess03 total > BASE_atrmed($2,980.2) ∧ 逐年全正 ∧ 笔数 >= 100
  网格择优 = 满足全正者中 total 最大; 若无满足者, 如实报告失败
  union(R1∪atrmed) / intersect(R1∩atrmed) = 保护变体, 报告但非冠军候选
  基线复现断言: BASE_atrmed pess03 必须 = $2,980.2 ± 0.5 (引擎复用正确性)

臂清单 (全部 M30 don55s · 只做多 · 持5日, 即 v17 冠军几何, 仅闸门不同):
  A0 BASE_atrmed  — v17 冠军原样复现          A1 BASE_none — 无闸门参照
  B* R1 网格      — θ_on∈{1.15,1.25,1.35} × L_on∈{8,16} (M15根)
  C1 UNION        — R1(best) ∪ atrmed (地板)   C2 INTERSECT — R1(best) ∩ atrmed (净化的极限)
  D1 ROUTE_B      — R1 独立事件驱动 (用户规格: |ŝ|≥0.40 ∧ R̂≥1.0 触发, 锁16根 M15,
                    能量回落 R̂<1.0 或 64 根时间帽出场, 固定 1oz, 双成本口径)

成本: 与 v17 完全同屋 — base=$0.50/RT 固定 | pess03=0.3×ATR/RT | pess05=0.5×ATR/RT
      ROUTE_B 另报 side 口径 2×(0.3×ATR+点差)/RT (R1 原报告口径) 作敏感度
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results_v19")
os.makedirs(RES, exist_ok=True)
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(os.path.dirname(BASE), "research", "reservoir"))

import v17_htf  # noqa: E402
from v17_htf import resample_htf, build_arrays, build_entry, EXIT_MODE_TIME  # noqa: E402
from v17b_refine import simulate_g_jit  # noqa: E402
from v17c_champ import summarize  # noqa: E402  (逐字同款指标语义)
import reservoir_engine as R1  # noqa: E402

# ---------------- 数据路径 (data.csv 缺失时回退 data.csv.gz, pandas 透明解压) ------
DATA = os.path.join(BASE, "data.csv")
if not os.path.exists(DATA):
    DATA = os.path.join(BASE, "data.csv.gz")
v17_htf.DATA_CSV = DATA  # load_m1 在调用时读模块全局, patch 生效

# ---------------- R1 oracle 主运行配置 (与 reservoir_v1.json 主运行逐字段一致) ----
ORACLE_CFG = dict(n_res=500, density=0.10, spectral=0.95, in_scale=0.30, lam=0.999,
                  delta=1e-6, p0=4.0, p_cap=1e4, k_sig=2.0, k_vol=0.5, chaos_gate=0.6,
                  smooth=0.5, rate=0.20, dead=0.05, horizon=4, warm=2000, k_max=16,
                  spawn_d2=36.0, spawn_cool=96, w_spawn=0.10, lr=0.02, wd=0.01,
                  slip_atr=0.30, oz=1.0)
COSTS = {"base": (0, 0.50, 0.50), "pess03": (1, 0.30, 0.30), "pess05": (1, 0.50, 0.50)}
BASELINE_TOTAL_PESS03 = 2980.2  # v17 冠军档案数, 复现断言用

GRID = [(on, round(on - 0.15, 2), lo) for on in (1.15, 1.25, 1.35) for lo in (8, 16)]
MIN_OFF = 4
ROUTE_B = dict(th_sig=0.40, th_var=1.0, hold_min=16, hold_max=64)


def m15_from_cdf(cdf):
    """与 R1.load_m15 完全同款的 M15 重采样 (清洗已在 load_m1 完成, 两处过滤逻辑逐字一致)."""
    return (cdf.resample("15min", label="left", closed="left")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last",
                  "tickvol": "sum", "spread": "median"})
            .dropna(subset=["open"]))


def schmitt_gate(R, theta_on, theta_off, min_on, min_off, warm):
    """路线B机制作用于闸门: 施密特触发 + 迟滞带 + 最小ON时锁 + 最小OFF再武装."""
    n = len(R)
    out = np.zeros(n, np.int8)
    state = 0
    last_change = warm
    for t in range(warm + 1, n):
        age = t - last_change
        if state == 0:
            if age >= min_off and R[t] >= theta_on:
                state = 1
                last_change = t
        else:
            if age >= min_on and R[t] <= theta_off:
                state = 0
                last_change = t
        out[t] = state
    return out


def gate_to_m30(state15, m15_ts, ts30):
    """gate30[k] = 状态@最后一根 o30[k] 之前收盘的 M15 bar (≡ k−1 收盘已知, v17 同屋)."""
    j = np.searchsorted(m15_ts.asi8, ts30.asi8, side="left") - 1
    g = np.zeros(len(ts30))
    ok = j >= 0
    g[ok] = state15[j[ok]]
    return g


def run_arm(arrs30, ts30, gate30, cost_mode):
    """v17 冠军几何 (M30 don55s · 只做多 · 240根时间出场), 仅替换闸门数组."""
    mode, sig, lup, ldn = build_entry(arrs30, "don55s")
    sig = np.where(sig > 0, sig, 0).astype(np.int8)
    ldn = np.full(arrs30["n"], -1e18)
    cm, cf, ca = COSTS[cost_mode]
    tr = simulate_g_jit(arrs30["o"], arrs30["h"], arrs30["l"], arrs30["c"], arrs30["atr"],
                         mode, sig, lup, ldn, EXIT_MODE_TIME, 0.0, 5 * 48,
                         0.0, 0.0, arrs30["dxlo10"], arrs30["dxhi10"],
                         cm, cf, ca, gate30)
    e_i, x_i, sd, epx, xpx, cst, rsn = tr
    pnl = sd * (xpx - epx) - cst
    tdf = pd.DataFrame(dict(
        entry_time=ts30[e_i], exit_time=ts30[x_i], side=np.where(sd == 1, "L", "S"),
        entry_px=np.round(epx, 2), exit_px=np.round(xpx, 2), cost=np.round(cst, 2),
        pnl=np.round(pnl, 2), hold_bars=x_i - e_i,
        reason=np.where(rsn == 1, "time", np.where(rsn == 5, "eod", "other")),
        atr_entry=np.round(arrs30["atr"][np.maximum(e_i - 1, 0)], 2),
    ))
    return tdf


def arm_record(name, gate_desc, tdf_p03, coverage_by_year=None, extra=None):
    rec = dict(name=name, gate=gate_desc, pess03=summarize(tdf_p03))
    if coverage_by_year is not None:
        rec["coverage_by_year"] = coverage_by_year
    if extra:
        rec.update(extra)
    return rec


def coverage_by_year(gate30, ts30):
    yrs = ts30.year
    return {int(y): round(float(gate30[yrs == y].mean()), 3) for y in sorted(set(yrs.tolist()))}


def run_route_b(D, R, S):
    """路线B本体 (用户规格): 施密特触发 + 16根锁 + 能量回落/64根帽, 固定 1oz, M15."""
    o, c, atr, sp = D["o"], D["c"], D["atr"], D["sp_pts"]
    ts, n = D["ts"], D["n"]
    warm = ORACLE_CFG["warm"]
    th_sig, th_var = ROUTE_B["th_sig"], ROUTE_B["th_var"]
    hmin, hmax = ROUTE_B["hold_min"], ROUTE_B["hold_max"]
    trades = []
    pos = 0
    cur = None
    t = warm
    while t < n - 1:
        if pos == 0:
            if abs(S[t]) >= th_sig and R[t] >= th_var:
                pos = 1 if S[t] > 0 else -1
                eib = t + 1                      # t 收盘决策 → t+1 开盘成交
                cur = dict(eib=eib, side=pos, epx=float(o[t + 1]),
                           cost=float(0.30 * atr[t]))  # RT 口径 (v17 同屋)
                t += 1
                continue
            t += 1
        else:
            held = t + 1 - cur["eib"]            # 若本根收盘出场, 持仓根数
            reason = None
            if held >= hmax:
                reason = "time"
            elif held >= hmin and R[t] < th_var:
                reason = "energy"
            if reason is not None:
                xib = t + 1
                xpx = float(o[t + 1])
                cur.update(xib=xib, xpx=xpx, reason=reason,
                           pnl=float(cur["side"] * (xpx - cur["epx"]) - cur["cost"]))
                trades.append(cur)
                pos = 0
                cur = None
            t += 1
    if pos != 0 and cur is not None:
        xpx = float(c[n - 1])
        cur.update(xib=n - 1, xpx=xpx, reason="eod",
                   pnl=float(cur["side"] * (xpx - cur["epx"]) - cur["cost"]))
        trades.append(cur)
    # side 口径敏感度: pnl_side = 毛价差 − 2×(0.3ATR+点差) = pnl + rt_cost − side_cost
    rows = []
    for r in trades:
        if "xib" not in r:
            continue
        rt_cost = r["cost"]
        dec = r["eib"] - 1
        side_cost = 2.0 * (0.30 * float(atr[dec]) + float(sp[dec]) * 0.001)
        rows.append(dict(entry_time=ts[r["eib"]], exit_time=ts[r["xib"]],
                         side="L" if r["side"] == 1 else "S",
                         entry_px=round(r["epx"], 2), exit_px=round(r["xpx"], 2),
                         cost=round(rt_cost, 2), pnl=round(r["pnl"], 2),
                         hold_bars=r["xib"] - r["eib"], reason=r["reason"],
                         atr_entry=round(float(atr[dec]), 2),
                         pnl_side=round(r["pnl"] + rt_cost - side_cost, 2)))
    return pd.DataFrame(rows)


def main():
    t00 = time.time()
    print("== v19 · R1 能量神谕 × v17 规则冠军 ==", flush=True)
    cdf = v17_htf.load_m1()
    print(f"[data] M1 cleaned bars: {len(cdf)}  ({time.time()-t00:.0f}s)", flush=True)

    # ---- M30 (v17 引擎网格) ----
    bar30 = resample_htf(cdf, "30min")
    arrs30 = build_arrays(bar30)
    ts30 = arrs30["ts"]
    atr_s = pd.Series(arrs30["atr"])
    med_q = atr_s.rolling(3024, min_periods=200).median().to_numpy()
    gate_atrmed = np.concatenate(([0.0], (arrs30["atr"] > med_q).astype(float)[:-1]))
    gate_none = np.ones(arrs30["n"])
    print(f"[M30] bars={arrs30['n']}", flush=True)

    # ---- M15 + R1 oracle (与 reservoir_v1 主运行逐字段同配置) ----
    bar15 = m15_from_cdf(cdf)
    D = R1.build_feats(bar15, horizon=4)
    print(f"[M15] bars={D['n']}  {D['ts'][0]} .. {D['ts'][-1]}", flush=True)
    t_or = time.time()
    oracle = R1.run_pass(D, ORACLE_CFG, 42, collect=False, label="oracle")
    R = np.clip(oracle["yhat"][:, 0], 0.0, 6.0)
    S = oracle["yhat"][:, 1]
    print(f"[oracle] done {time.time()-t_or:.0f}s  "
          f"{oracle['timing']['total_ms']['mean']:.3f} ms/bar (本地单线程)", flush=True)

    # ---- oracle 质量 (对照 R1 阶段一: IC_var 0.343 / IC_skew 0.0197) ----
    n15, warm = D["n"], ORACLE_CFG["warm"]
    Y = D["Y"]
    valid = np.zeros(n15, bool)
    valid[warm:n15 - 5] = True
    m_icv = valid & np.isfinite(Y[:, 0])
    m_ics = valid & np.isfinite(Y[:, 1])
    ic_var = float(np.corrcoef(R[m_icv], Y[m_icv, 0])[0, 1])
    ic_skew = float(np.corrcoef(S[m_ics], Y[m_ics, 1])[0, 1])
    ic_var_y = {}
    yrs15 = np.asarray(D["ts"].year)
    for y in sorted(set(yrs15[warm:].tolist())):
        mm = m_icv & (yrs15 == y)
        if mm.sum() > 500:
            ic_var_y[int(y)] = round(float(np.corrcoef(R[mm], Y[mm, 0])[0, 1]), 4)
    rq = {q: round(float(np.percentile(R[warm:], q)), 3) for q in (10, 25, 50, 75, 90)}
    print(f"[oracle] IC_var={ic_var:.4f} IC_skew={ic_skew:.4f}  R̂分位={rq}", flush=True)

    # ---- A0 基线复现断言 ----
    tdf_base = run_arm(arrs30, ts30, gate_atrmed, "pess03")
    s_base = summarize(tdf_base)
    repro_ok = abs(s_base["total"] - BASELINE_TOTAL_PESS03) < 0.5
    print(f"[A0 BASE_atrmed] total={s_base['total']} n={s_base['trades']} "
          f"复现断言={'OK' if repro_ok else 'FAIL!!'}", flush=True)
    assert repro_ok, "基线复现失败 — 引擎复用有误, 停止"

    # ---- A1 无闸门参照 ----
    tdf_none = run_arm(arrs30, ts30, gate_none, "pess03")
    s_none = summarize(tdf_none)

    # ---- B* R1 闸门网格 ----
    grid_recs = []
    gate_best = None
    best_name = None
    for (on, off, lo) in GRID:
        st = schmitt_gate(R, on, off, lo, MIN_OFF, warm)
        g30 = gate_to_m30(st, D["ts"], ts30)
        tdf = run_arm(arrs30, ts30, g30, "pess03")
        s = summarize(tdf)
        name = f"r1g_on{on:.2f}_lo{lo}"
        allpos = all(v > 0 for v in s["by_year"].values())
        grid_recs.append(dict(name=name, theta_on=on, theta_off=off, min_on=lo,
                              pess03=s, all_pos=allpos,
                              coverage=coverage_by_year(g30, ts30)))
        print(f"[B {name}] total={s['total']} n={s['trades']} avg={s['avg']} "
              f"yr={s['by_year']} allpos={allpos}", flush=True)
        if allpos and s["trades"] >= 100 and (gate_best is None or s["total"] > gate_best[0]):
            gate_best = (s["total"], st, g30, tdf, name)
            best_name = name
    if gate_best is None:  # 无满足全正者 → 按预注册退化为 total 最大
        best = max(grid_recs, key=lambda r: r["pess03"]["total"])
        st = schmitt_gate(R, best["theta_on"], best["theta_off"], best["min_on"], MIN_OFF, warm)
        g30 = gate_to_m30(st, D["ts"], ts30)
        tdf = run_arm(arrs30, ts30, g30, "pess03")
        gate_best = (best["pess03"]["total"], st, g30, tdf, best["name"])
        best_name = best["name"]
        print(f"[grid] 无全正臂, 退化取 total 最大: {best_name}", flush=True)

    _, st_best, g_best, tdf_best, _ = gate_best
    s_best = summarize(tdf_best)

    # ---- C 组合 (保护变体) ----
    g_union = np.maximum(g_best, gate_atrmed)
    g_inter = np.minimum(g_best, gate_atrmed)
    tdf_u = run_arm(arrs30, ts30, g_union, "pess03")
    tdf_i = run_arm(arrs30, ts30, g_inter, "pess03")
    s_u, s_i = summarize(tdf_u), summarize(tdf_i)
    print(f"[C1 UNION] total={s_u['total']} n={s_u['trades']} yr={s_u['by_year']}", flush=True)
    print(f"[C2 INTERSECT] total={s_i['total']} n={s_i['trades']} yr={s_i['by_year']}", flush=True)

    # ---- 闸门重叠诊断 (post-warmup M30) ----
    m30_from = ts30 >= D["ts"][warm]
    ov = dict(
        p_r1_on=float(g_best[m30_from].mean()),
        p_atr_on=float(gate_atrmed[m30_from].mean()),
        p_both=float(((g_best > 0.5) & (gate_atrmed > 0.5))[m30_from].mean()),
        p_r1_only=float(((g_best > 0.5) & (gate_atrmed < 0.5))[m30_from].mean()),
        p_atr_only=float(((g_best < 0.5) & (gate_atrmed > 0.5))[m30_from].mean()),
    )
    ov["p_both_giv_atr"] = round(ov["p_both"] / max(ov["p_atr_on"], 1e-9), 3)

    # ---- D1 路线B (独立事件驱动) ----
    tdf_b = run_route_b(D, R, S)
    if len(tdf_b) > 0:
        s_b = summarize(tdf_b)
        pnl_side = tdf_b["pnl_side"].to_numpy()
        s_b_side = dict(total=round(float(pnl_side.sum()), 1), trades=int(len(pnl_side)),
                        avg=round(float(pnl_side.mean()), 2))
    else:
        s_b = dict(total=0.0, trades=0, avg=0.0, by_year={}, months={}, t_stat=0.0)
        s_b_side = dict(total=0.0, trades=0, avg=0.0)
    weeks = max((D["ts"][-1] - D["ts"][warm]).days / 7.0, 1.0)
    print(f"[D1 ROUTE_B] total={s_b['total']} n={s_b['trades']} "
          f"freq={s_b['trades']/weeks:.2f}/周 side口径={s_b_side['total']}", flush=True)

    # ---- 预注册判定 ----
    best_allpos = all(v > 0 for v in s_best["by_year"].values())
    route_a_ok = (s_best["total"] > BASELINE_TOTAL_PESS03) and best_allpos and s_best["trades"] >= 100
    union_beats = s_u["total"] > BASELINE_TOTAL_PESS03
    verdict = dict(
        route_a_ok=bool(route_a_ok),
        best_arm=best_name,
        best_total=s_best["total"], base_total=s_base["total"],
        delta=round(s_best["total"] - s_base["total"], 1),
        union_beats_base=bool(union_beats),
        union_total=s_u["total"],
        route_b_total=s_b["total"], route_b_side_total=s_b_side["total"],
        route_b_freq_per_week=round(s_b["trades"] / weeks, 2),
    )
    print(f"[verdict] {verdict}", flush=True)

    # ---- 成本阶梯 (关键臂) ----
    ladder = {}
    for nm, g in (("BASE_atrmed", gate_atrmed), ("BEST_R1", g_best),
                  ("UNION", g_union), ("INTERSECT", g_inter)):
        lad = {}
        for cm in ("base", "pess03", "pess05"):
            tdf = run_arm(arrs30, ts30, g, cm)
            lad[cm] = summarize(tdf)
        ladder[nm] = lad

    # ---- 逐笔导出 ----
    for nm, tdf in (("BASE_atrmed", tdf_base), (f"BEST_{best_name}", tdf_best),
                    ("UNION", tdf_u), ("ROUTE_B", tdf_b)):
        tj = tdf.copy()
        if "entry_time" in tj:
            tj["entry_time"] = pd.to_datetime(tj["entry_time"]).dt.strftime("%Y-%m-%d %H:%M")
            tj["exit_time"] = pd.to_datetime(tj["exit_time"]).dt.strftime("%Y-%m-%d %H:%M")
        tj.to_csv(os.path.join(RES, f"trades_v19_{nm}.csv"), index=False)

    doc = dict(
        meta=dict(
            engine="v19_r1gate", generated=pd.Timestamp.now().isoformat(),
            decision="路线A为主干(R1能量神谕→v17闸门); 路线B机制移植到闸门状态机; "
                     "路线B本体作为对照臂诚实运行",
            prereg=("成功=R1闸门臂pess03>$2,980.2∧逐年全正∧笔数>=100; 网格择优按total; "
                    "union/intersect为保护变体; 基线复现断言$2,980.2±0.5"),
            cost="base=$0.50/RT | pess03=0.3×ATR/RT | pess05=0.5×ATR/RT (与v17同屋)",
            oracle_cfg=ORACLE_CFG, grid=[list(g) for g in GRID], min_off=MIN_OFF,
            route_b_spec=ROUTE_B,
            oracle_wall_s=round(time.time() - t_or, 1),
            oracle_ms_per_bar=oracle["timing"]["total_ms"]["mean"],
            total_wall_s=round(time.time() - t00, 1),
        ),
        oracle=dict(ic_var=round(ic_var, 4), ic_skew=round(ic_skew, 4),
                    ic_var_by_year=ic_var_y, r_quantiles=rq),
        baseline_repro=dict(total=s_base["total"], expected=BASELINE_TOTAL_PESS03, ok=repro_ok),
        arms=dict(
            base_atrmed=dict(pess03=s_base, coverage=coverage_by_year(gate_atrmed, ts30)),
            base_none=dict(pess03=s_none),
            best_r1=dict(name=best_name, pess03=s_best,
                         coverage=coverage_by_year(g_best, ts30),
                         theta=[r for r in grid_recs if r["name"] == best_name][0]["theta_on"] if
                         any(r["name"] == best_name for r in grid_recs) else None),
            union=dict(pess03=s_u, coverage=coverage_by_year(g_union, ts30)),
            intersect=dict(pess03=s_i, coverage=coverage_by_year(g_inter, ts30)),
        ),
        grid=grid_recs,
        overlap=ov,
        route_b=dict(pess03=s_b, side_cost=s_b_side,
                     freq_per_week=round(s_b["trades"] / weeks, 2)),
        ladder=ladder,
        verdict=verdict,
    )
    out = os.path.join(RES, "v19_r1gate.json")
    with open(out, "w") as f:
        json.dump(doc, f, indent=1, default=str)
    print(f"== DONE -> {out} ({os.path.getsize(out)} bytes, wall {time.time()-t00:.0f}s) ==", flush=True)


if __name__ == "__main__":
    main()
