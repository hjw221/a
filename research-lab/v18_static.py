#!/usr/bin/env python3
"""v18_static.py — 静态灾难止损 (Static Disaster Stop) 实验 [分支A/P0].

用户指令 (2026-10-02):
  动态吊灯失败根因: 移动止损随价格上移, 任何正常的5日动量回踩都会被精确扫损.
  方案: 入场点锁定的静态灾难止损 —
        A) Entry_Price - k × Entry_ATR   (k 扫 2.0~8.0)
        B) 入场时刻的 Donchian Lower Band (N 扫 10/20/40/55)
        止损线全程不跟随上移; 给5日趋势运行留全部呼吸空间,
        只在系统性暴跌或假突破快速破位时才斩仓.
  目标: 保留 $2,500+ 利润 (v17冠军 $2,980.2) 的前提下把最坏尾部封死.

冻结: M30 don55s(停损单) × long-only × ATR>季度中位闸门 × 持5交易日(240根M30)
  = v17 冠军, base 臂(无止损)必须逐位复现 v17c pess03 total=$2,980.2/159笔.

预注册验收 (跑前锁定):
  R1 引擎等价: base臂 与 v17b simulate_g_jit(EXIT_MODE_TIME) 输出逐位一致
  R2 用户目标: pess03口径 total>=2500 AND 逐年全正 AND worst_trade 显著改善
  R3 双向披露: 被斩笔的"保险赔付"(避免损失) vs "误杀成本"(错失盈利) 分别报告
  R4 纯保险overlay: base 159笔同笔列表叠加止损(零路径耦合) 单独报告
  R5 稳健性: 成本三档(base/pess03/pess05) + 止损宽度邻域检查

结算纪律 (v17同款, 两bug教训后):
  bar t 覆盖 [t, t+Delta) label=left; 止损单进场用当根h/l触发;
  静态止损价 = 进场瞬间由 (entry_px, ATR[k-1] 或 dlo_N[k]) 锁定, 全程不动;
  触发当根即出场(l[k]<=sl), gap穿越按更差open成交, 止损优先于时间出场(同根保守);
  进场当根同样检查止损(假突破快速破位正是要斩的形态);
  出场当根不再进场(保守, 禁同根翻仓).
"""
import os, json, time
import numpy as np
import pandas as pd
from numba import njit
from v17_htf import (load_m1, resample_htf, build_arrays, build_entry, COSTS,
                     ENTRY_CLOSE, ENTRY_STOP, EXIT_MODE_TIME)
from v17b_refine import simulate_g_jit  # 跨引擎对照用

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results_v18")
os.makedirs(RES, exist_ok=True)

EXIT_MODE_TIME_SL = 5          # 时间出场 + 静态灾难止损
SL_NONE, SL_ATR, SL_REF = 0, 1, 2


# ================= 核心引擎 (纯Python源, numba同源编译, 双引擎互验) =================
def simulate_core_ts(o, h, l, c, atr, entry_mode, sig, line_up, line_dn,
                     exit_mode, w, hzn, tp_m, sl_m, sl_kind, sl_ref_dn, sl_ref_up,
                     dxlo, dxhi, cost_mode, cost_flat, cost_atr_frac, gate):
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
    sl = np.nan           # 静态灾难止损价: 进场瞬间锁定, 全程不动
    cost_v = 0.0
    pending = False
    for k in range(1, n):
        # -- 挂起的 turtle 出场(结构保留, TIME_SL 不触发) --
        if pos != 0 and pending:
            e_i[nt] = eib; x_i[nt] = k; sd[nt] = pos; epx[nt] = entry_px
            xpx[nt] = o[k]; cst[nt] = cost_v; rsn[nt] = 6; nt += 1
            pos = 0; pending = False
            continue
        # -- 空仓: 尝试进场 --
        if pos == 0:
            a_prev = atr[k - 1]
            g = gate[k] > 0.5
            if entry_mode == ENTRY_CLOSE:
                s = sig[k - 1]
                if s != 0 and g and a_prev > 0 and not np.isnan(o[k]):
                    pos = s; entry_px = o[k]; eib = k
                    hi_s = entry_px; lo_s = entry_px
                    cost_v = cost_flat if cost_mode == 0 else cost_atr_frac * a_prev
                    # 静态灾难止损: 进场瞬间锁定
                    sl = np.nan
                    if sl_kind == SL_ATR and sl_m > 0.0:
                        sl = entry_px - sl_m * a_prev if pos == 1 else entry_px + sl_m * a_prev
                    elif sl_kind == SL_REF:
                        r_ = sl_ref_dn[k] if pos == 1 else sl_ref_up[k]
                        sl = r_ if not np.isnan(r_) else np.nan
            else:
                up = line_up[k]; dn = line_dn[k]
                if g and (not np.isnan(up)) and (not np.isnan(dn)) and a_prev > 0:
                    hit_up = h[k] >= up
                    hit_dn = l[k] <= dn
                    if hit_up and not hit_dn:
                        pos = 1; entry_px = up if o[k] < up else o[k]; eib = k
                        hi_s = entry_px; lo_s = entry_px
                        cost_v = cost_flat if cost_mode == 0 else cost_atr_frac * a_prev
                        sl = np.nan
                        if sl_kind == SL_ATR and sl_m > 0.0:
                            sl = entry_px - sl_m * a_prev
                        elif sl_kind == SL_REF:
                            r_ = sl_ref_dn[k]
                            sl = r_ if not np.isnan(r_) else np.nan
                    elif hit_dn and not hit_up:
                        pos = -1; entry_px = dn if o[k] > dn else o[k]; eib = k
                        hi_s = entry_px; lo_s = entry_px
                        cost_v = cost_flat if cost_mode == 0 else cost_atr_frac * a_prev
                        sl = np.nan
                        if sl_kind == SL_ATR and sl_m > 0.0:
                            sl = entry_px + sl_m * a_prev
                        elif sl_kind == SL_REF:
                            r_ = sl_ref_up[k]
                            sl = r_ if not np.isnan(r_) else np.nan
                    # 双向同根触发: 路径不明, 保守跳过
        # -- 持仓: 出场检查 --
        if pos != 0:
            done = False
            # 1) 静态灾难止损: 触发当根即出场; gap穿越按更差open; 止损优先于时间出场
            if exit_mode == EXIT_MODE_TIME_SL and not np.isnan(sl):
                if pos == 1 and l[k] <= sl:
                    px = sl if o[k] >= sl else o[k]
                    e_i[nt] = eib; x_i[nt] = k; sd[nt] = pos; epx[nt] = entry_px
                    xpx[nt] = px; cst[nt] = cost_v; rsn[nt] = 4; nt += 1
                    pos = 0; done = True
                elif pos == -1 and h[k] >= sl:
                    px = sl if o[k] <= sl else o[k]
                    e_i[nt] = eib; x_i[nt] = k; sd[nt] = pos; epx[nt] = entry_px
                    xpx[nt] = px; cst[nt] = cost_v; rsn[nt] = 4; nt += 1
                    pos = 0; done = True
            # 2) 时间出场
            if not done and exit_mode == EXIT_MODE_TIME_SL and hzn > 0:
                if k >= eib + hzn:
                    e_i[nt] = eib; x_i[nt] = k; sd[nt] = pos; epx[nt] = entry_px
                    xpx[nt] = o[k]; cst[nt] = cost_v; rsn[nt] = 1; nt += 1
                    pos = 0; done = True
            if not done:
                if h[k] > hi_s:
                    hi_s = h[k]
                if l[k] < lo_s:
                    lo_s = l[k]
    if pos != 0:  # 数据末尾强平
        e_i[nt] = eib; x_i[nt] = n - 1; sd[nt] = pos; epx[nt] = entry_px
        xpx[nt] = c[n - 1]; cst[nt] = cost_v; rsn[nt] = 5; nt += 1
    return e_i[:nt], x_i[:nt], sd[:nt], epx[:nt], xpx[:nt], cst[:nt], rsn[:nt]


simulate_ts_jit = njit(cache=True)(simulate_core_ts)


# ================= 指标 (v17c summarize 同款 + 尾部专项) =================
def summarize(tr, ts, arrs):
    e_i, x_i, sd, epx, xpx, cst, rsn = tr
    pnl = sd * (xpx - epx) - cst
    n_tr = len(pnl)
    if n_tr == 0:
        return dict(total=0.0, trades=0)
    xts = ts[x_i]
    yrs = xts.year.to_numpy()
    by = {int(y): round(float(pnl[yrs == y].sum()), 1) for y in np.unique(yrs)}
    mo = pd.Series(pnl, index=xts).groupby(xts.to_period("M")).sum()
    eq = mo.cumsum()
    dd = float((eq - eq.cummax()).min()) if len(eq) else 0.0
    tstat = float(pnl.mean() / (pnl.std(ddof=1) / np.sqrt(n_tr)))
    # 止损专项
    n_sl = int((rsn == 4).sum())
    a_entry = arrs["atr"][np.maximum(e_i - 1, 0)]
    dist_atr = (epx - xpx)[rsn == 4] / a_entry[rsn == 4] if n_sl else np.array([])
    return dict(
        total=round(float(pnl.sum()), 1), trades=n_tr, avg=round(float(pnl.mean()), 2),
        med=round(float(np.median(pnl)), 2),
        p05=round(float(np.percentile(pnl, 5)), 2), p10=round(float(np.percentile(pnl, 10)), 2),
        worst=round(float(pnl.min()), 2), best=round(float(pnl.max()), 2),
        win_rate=round(float((pnl > 0).mean()), 3),
        by_year=by, all_pos=all(v > 0 for v in by.values()),
        share2026=round(100.0 * by.get(2026, 0.0) / max(float(pnl.sum()), 1e-9), 1),
        t_stat=round(tstat, 2), sharpe=round(float(mo.mean() / (mo.std() + 1e-9) * np.sqrt(12)), 2),
        maxdd=round(dd, 1),
        n_sl_hit=n_sl, sl_hit_rate=round(n_sl / n_tr, 3),
        med_sl_dist_atr=round(float(np.median(dist_atr)), 2) if n_sl else None,
        worst_sl_atr=round(float(np.min(dist_atr)), 2) if n_sl else None,
        months={str(k): round(float(v), 1) for k, v in mo.items()},
    )


def trades_df(tr, ts, arrs):
    e_i, x_i, sd, epx, xpx, cst, rsn = tr
    pnl = sd * (xpx - epx) - cst
    return pd.DataFrame(dict(
        entry_time=ts[e_i], exit_time=ts[x_i], side=np.where(sd == 1, "L", "S"),
        entry_px=np.round(epx, 2), exit_px=np.round(xpx, 2), cost=np.round(cst, 2),
        pnl=np.round(pnl, 2), hold_bars=x_i - e_i,
        reason=np.where(rsn == 1, "time", np.where(rsn == 4, "static_stop",
                    np.where(rsn == 5, "eod", "other"))),
        atr_entry=np.round(arrs["atr"][np.maximum(e_i - 1, 0)], 2),
    ))


# ================= R4: 纯保险 overlay (base 同笔列表, 零路径耦合) =================
def overlay_static(arrs, tr_base, sl_kind, sl_m, ref_dn):
    """base 臂每笔交易上直接叠加静态止损: 扫 e_i..x_i 触碰即斩, 不改变交易集合.
    返回 (pnl_overlay, hit_mask, hit_bar, sl_price)."""
    o, h, l, atr = arrs["o"], arrs["h"], arrs["l"], arrs["atr"]
    e_i, x_i, sd, epx, xpx, cst, rsn = tr_base
    nt = len(e_i)
    out = np.empty(nt)
    hit = np.zeros(nt, dtype=bool)
    hj = np.full(nt, -1, dtype=np.int64)
    slp = np.full(nt, np.nan)
    for t in range(nt):
        e, x = int(e_i[t]), int(x_i[t])
        a_prev = atr[e - 1] if e > 0 else np.nan
        if sl_kind == SL_ATR:
            sl = epx[t] - sl_m * a_prev if sd[t] == 1 else epx[t] + sl_m * a_prev
        elif sl_kind == SL_REF:
            sl = ref_dn[e] if sd[t] == 1 else np.nan
        else:
            sl = np.nan
        slp[t] = sl
        j_hit = -1
        if not np.isnan(sl):
            for j in range(e, x + 1):  # 含进场根与原出场根(止损优先, 同引擎保守排序)
                if sd[t] == 1 and l[j] <= sl:
                    j_hit = j; break
                if sd[t] == -1 and h[j] >= sl:
                    j_hit = j; break
        if j_hit >= 0:
            if sd[t] == 1:
                px = sl if o[j_hit] >= sl else o[j_hit]
            else:
                px = sl if o[j_hit] <= sl else o[j_hit]
            out[t] = sd[t] * (px - epx[t]) - cst[t]
            hit[t] = True; hj[t] = j_hit
        else:
            out[t] = sd[t] * (xpx[t] - epx[t]) - cst[t]
    return out, hit, hj, slp


# ================= 主流程 =================
STOPS = [
    ("none",   SL_NONE, 0.0, None),
    ("atr2.0", SL_ATR,  2.0, None), ("atr2.5", SL_ATR, 2.5, None),
    ("atr3.0", SL_ATR,  3.0, None), ("atr3.5", SL_ATR, 3.5, None),
    ("atr4.0", SL_ATR,  4.0, None), ("atr5.0", SL_ATR, 5.0, None),
    ("atr6.0", SL_ATR,  6.0, None), ("atr8.0", SL_ATR, 8.0, None),
    ("don10",  SL_REF,  0.0, 10),   ("don20", SL_REF, 0.0, 20),
    ("don40",  SL_REF,  0.0, 40),   ("don55", SL_REF, 0.0, 55),
]

def _ref_arrays(arrs, refN, nan_arr):
    """Donchian 轨道数组: dlo10 在 build_arrays 里叫 dxlo10 (同 shift(1) 因果)."""
    if refN is None:
        return nan_arr, nan_arr
    if refN == 10:
        return arrs["dxlo10"], arrs["dxhi10"]
    return arrs[f"dlo{refN}"], arrs[f"dhi{refN}"]


SPEC = dict(tf="M30", rule="30min", bpd=48, qwin=3024, entry="don55s", dir="long",
            gate="atrmed", days=5)


def main():
    t00 = time.time()
    cdf = load_m1()
    bar = resample_htf(cdf, SPEC["rule"])
    arrs = build_arrays(bar)
    ts = arrs["ts"]
    n = arrs["n"]
    # gate: ATR14 > 过去一季度自身中位数 (因果右移一根) — v17c 同款
    atr_s = pd.Series(arrs["atr"])
    med_q = atr_s.rolling(SPEC["qwin"], min_periods=200).median().to_numpy()
    gate = np.concatenate(([0.0], (arrs["atr"] > med_q).astype(float)[:-1]))
    # long-only don55s
    mode, sig, lup, ldn = build_entry(arrs, SPEC["entry"])
    sig = np.where(sig > 0, sig, 0).astype(np.int8)
    ldn = np.full(n, -1e18)
    hzn = SPEC["days"] * SPEC["bpd"]
    nan_arr = np.full(n, np.nan)
    print(f"[setup] {SPEC['tf']} bars={n} hzn={hzn} gate_on={gate.sum()}", flush=True)

    results = {}
    trades_dump = {}

    # ---- R1: base 臂 (none) 双引擎互验 + 与 v17b 引擎跨引擎逐位对照 ----
    base_tr = {}
    for cost in COSTS:
        args = (arrs["o"], arrs["h"], arrs["l"], arrs["c"], arrs["atr"], mode, sig, lup, ldn,
                EXIT_MODE_TIME_SL, 0.0, hzn, 0.0, 0.0, SL_NONE, nan_arr, nan_arr,
                arrs["dxlo10"], arrs["dxhi10"], cost[1], cost[2], cost[2], gate)
        r_py = simulate_core_ts(*args)
        r_jit = simulate_ts_jit(*args)
        for nm, A, B in zip(["e", "x", "sd", "ep", "xp", "cs", "rs"], r_py, r_jit):
            assert len(A) == len(B) and np.array_equal(A, B), f"DUAL MISMATCH {nm}"
        if cost[0] == "pess03":
            # 跨引擎: 与 v17b EXIT_MODE_TIME 引擎逐位一致
            r_v17 = simulate_g_jit(arrs["o"], arrs["h"], arrs["l"], arrs["c"], arrs["atr"], mode, sig, lup, ldn,
                                   EXIT_MODE_TIME, 0.0, hzn, 0.0, 0.0,
                                   arrs["dxlo10"], arrs["dxhi10"], cost[1], cost[2], cost[2], gate)
            for nm, A, B in zip(["e", "x", "sd", "ep", "xp", "cs", "rs"], r_v17, r_jit):
                assert len(A) == len(B) and np.array_equal(A, B), f"XENGINE MISMATCH {nm}"
            m_base = summarize(r_jit, ts, arrs)
            print(f"[R1] base({cost[0]}) == v17b engine bitwise OK: total={m_base['total']} "
                  f"n={m_base['trades']} worst={m_base['worst']} yr={m_base['by_year']}", flush=True)
        base_tr[cost[0]] = r_jit

    # ---- 主矩阵: 13 止损 × 3 成本 ----
    for sname, skind, sm, refN in STOPS:
        ref_dn, ref_up = _ref_arrays(arrs, refN, nan_arr)
        for cost in COSTS:
            args = (arrs["o"], arrs["h"], arrs["l"], arrs["c"], arrs["atr"], mode, sig, lup, ldn,
                    EXIT_MODE_TIME_SL, 0.0, hzn, 0.0, sm, skind, ref_dn, ref_up,
                    arrs["dxlo10"], arrs["dxhi10"], cost[1], cost[2], cost[2], gate)
            r_py = simulate_core_ts(*args)
            r_jit = simulate_ts_jit(*args)
            for nm, A, B in zip(["e", "x", "sd", "ep", "xp", "cs", "rs"], r_py, r_jit):
                assert len(A) == len(B) and np.array_equal(A, B), f"DUAL MISMATCH {sname}.{nm}"
            m = summarize(r_jit, ts, arrs)
            m.update(stop=sname, cost=cost[0])
            results[f"{sname}|{cost[0]}"] = m
            if cost[0] == "pess03":
                trades_dump[sname] = trades_df(r_jit, ts, arrs)
        s = results[f"{sname}|pess03"]
        print(f"[{sname:>6}] pess03 total={s['total']:>7} n={s['trades']:>3} avg={s['avg']:>6} "
              f"worst={s['worst']:>7} p05={s['p05']:>7} hit={s['n_sl_hit']:>3}({s['sl_hit_rate']:.0%}) "
              f"dd={s['maxdd']:>6} allpos={s['all_pos']} yr={s['by_year']}", flush=True)

    # ---- R4: overlay 纯保险 (base 159 笔同笔列表) ----
    tr_b = base_tr["pess03"]
    e_i, x_i, sd, epx, xpx, cst, rsn = tr_b
    pnl_base = sd * (xpx - epx) - cst
    overlays = {}
    print(f"\n[R4] overlay on base trades n={len(e_i)} (pess03 costs)", flush=True)
    print(f"{'stop':>6} {'ov_total':>8} {'n_hit':>5} {'worst':>8} {'avoided':>8} {'killed':>8} {'net':>8}")
    for sname, skind, sm, refN in STOPS:
        if sname == "none":
            continue
        ref_dn, _ = _ref_arrays(arrs, refN, nan_arr)
        ov, hit, hj, slp = overlay_static(arrs, tr_b, skind, sm, ref_dn)
        # 双向披露: hit 笔上 base pnl 的正负分解
        avoided = float(-pnl_base[hit & (pnl_base < 0)].sum())   # 止损避免掉的亏损(保险赔付)
        killed = float(pnl_base[hit & (pnl_base > 0)].sum())     # 止损误杀掉的盈利
        net = float((ov - pnl_base).sum())
        yrs = ts[x_i].year.to_numpy()[hit]
        overlays[sname] = dict(
            total=round(float(ov.sum()), 1), n_hit=int(hit.sum()),
            hit_rate=round(float(hit.mean()), 3),
            worst=round(float(ov.min()), 2), p05=round(float(np.percentile(ov, 5)), 2),
            avoided=round(avoided, 1), killed=round(killed, 1), net=round(net, 1),
            hit_by_year={int(y): int((yrs == y).sum()) for y in np.unique(yrs)},
        )
        print(f"{sname:>6} {ov.sum():>8.1f} {int(hit.sum()):>5} {ov.min():>8.1f} "
              f"{avoided:>8.1f} {killed:>8.1f} {net:>8.1f}")

    # ---- R3: counterfactual 对齐 (引擎臂被斩笔 vs base 同进场时刻笔) ----
    cf = {}
    base_e = e_i.copy()
    print(f"\n[R3] counterfactual: engine stopped trades vs base same-entry-bar trades", flush=True)
    for sname in ("atr4.0", "don55", "atr6.0", "don20"):
        r = None
        for cost in COSTS:
            if cost[0] != "pess03":
                continue
            skind, sm, refN = [(k, m, rN) for nm, k, m, rN in STOPS if nm == sname][0]
            ref_dn, ref_up = _ref_arrays(arrs, refN, nan_arr)
            args = (arrs["o"], arrs["h"], arrs["l"], arrs["c"], arrs["atr"], mode, sig, lup, ldn,
                    EXIT_MODE_TIME_SL, 0.0, hzn, 0.0, sm, skind, ref_dn, ref_up,
                    arrs["dxlo10"], arrs["dxhi10"], cost[1], cost[2], cost[2], gate)
            r = simulate_ts_jit(*args)
        e2, x2, sd2, ep2, xp2, cs2, rs2 = r
        stop_mask = rs2 == 4
        matched, diffs = [], []
        bmap = {int(e): i for i, e in enumerate(base_e)}
        for t in np.where(stop_mask)[0]:
            key = int(e2[t])
            if key in bmap:
                bi = bmap[key]
                pnl_stop = sd2[t] * (xp2[t] - ep2[t]) - cs2[t]
                diffs.append(float(pnl_stop - pnl_base[bi]))
                matched.append(key)
        diffs = np.array(diffs) if diffs else np.array([])
        cf[sname] = dict(
            engine_stopped=int(stop_mask.sum()), matched=len(matched),
            saved_sum=round(float(-diffs[diffs < 0].sum()), 1) if len(diffs) else 0.0,
            cost_sum=round(float(diffs[diffs > 0].sum()), 1) if len(diffs) else 0.0,
            net=round(float(diffs.sum()), 1) if len(diffs) else 0.0,
        )
        print(f"[{sname:>6}] stopped={int(stop_mask.sum()):>3} matched={len(matched):>3} "
              f"saved={cf[sname]['saved_sum']:>7} extra_cost={cf[sname]['cost_sum']:>7} net={cf[sname]['net']:>7}")

    # ---- R2 验收汇总 ----
    print("\n[R2] acceptance (pess03): keep>=2500 & all_pos & worst improved", flush=True)
    base_worst = results["none|pess03"]["worst"]
    for sname, *_ in STOPS:
        s = results[f"{sname}|pess03"]
        keep = s["total"] >= 2500
        ap = s["all_pos"]
        imp = s["worst"] > base_worst
        verdict = "PASS" if (keep and ap and imp) else ("TAIL-ONLY" if imp else "FAIL")
        print(f"{sname:>6}: total={s['total']:>7} keep2500={keep} allpos={ap} "
              f"worst={s['worst']:>7}(base {base_worst}) -> {verdict}")

    out = dict(spec=SPEC, results=results, overlays=overlays, counterfactual=cf,
               base_pess03=dict(total=results["none|pess03"]["total"],
                                worst=base_worst, trades=results["none|pess03"]["trades"],
                                by_year=results["none|pess03"]["by_year"]))
    json.dump(out, open(os.path.join(RES, "static_stop.json"), "w"), indent=1)
    # 逐笔导出: base + 有止损的赢家变体(先导 base 与 atr4.0/don55 全套)
    for sname in ("none", "atr3.0", "atr4.0", "atr5.0", "atr6.0", "don20", "don55"):
        if sname in trades_dump:
            trades_dump[sname].to_csv(os.path.join(RES, f"trades_{sname}.csv"), index=False)
    print(f"\nDONE {(time.time() - t00) / 60:.1f}min -> results_v18/static_stop.json", flush=True)


if __name__ == "__main__":
    main()
