#!/usr/bin/env python3
"""v18c_cooldown.py — 静态止损的最后一个生存假设: 止损触发=假突破确认 -> 冷却禁进场.

v18/v18b 已证: 任何静态止损(宽度2-20ATR/Donchian/时变)在v17冠军上都是负期望,
且宽止损反而恶化worst(atr20 worst-373 vs base-168) — 根因是止损触发释放仓位后,
突破信号在震荡下跌里反复再进场(159->189-307笔)的绞肉循环.

最后假设 H_cool: 被斩=假突破信息, 若止损触发后冷却禁进场(一个完整持仓周期=240根),
消除绞肉循环, 止损能否转正期望?
  预注册判定: 任一 (止损x冷却) pess03 total > base 2980.1 AND worst改善 -> H_cool成立;
               否则 价格底线概念在该结构上彻底证伪, 分支A终结.

引擎: simulate_core_tv + cooldown (last_stop_bar 记录, k > last_stop+cd 才可进场).
双引擎互验同纪律.
"""
import os, json, time
import numpy as np
import pandas as pd
from numba import njit
from v17_htf import load_m1, resample_htf, build_arrays, build_entry, ENTRY_STOP
from v18_static import summarize, SL_ATR, SL_REF, SL_NONE, EXIT_MODE_TIME_SL, _ref_arrays, SPEC

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results_v18")


def simulate_core_cd(o, h, l, c, atr, entry_mode, sig, line_up, line_dn,
                     exit_mode, w, hzn, tp_m, sl_m, sl_kind, sl_ref_dn, sl_ref_up,
                     cooldown, dxlo, dxhi, cost_mode, cost_flat, cost_atr_frac, gate):
    n = len(o)
    e_i = np.empty(n, np.int64); x_i = np.empty(n, np.int64)
    sd = np.empty(n, np.int8); epx = np.empty(n, np.float64)
    xpx = np.empty(n, np.float64); cst = np.empty(n, np.float64)
    rsn = np.empty(n, np.int8)
    nt = 0; pos = 0; entry_px = 0.0; hi_s = 0.0; lo_s = 0.0
    eib = 0; tp = 0.0; sl = np.nan; cost_v = 0.0; pending = False
    last_stop = -10**9
    for k in range(1, n):
        if pos != 0 and pending:
            e_i[nt] = eib; x_i[nt] = k; sd[nt] = pos; epx[nt] = entry_px
            xpx[nt] = o[k]; cst[nt] = cost_v; rsn[nt] = 6; nt += 1
            pos = 0; pending = False
            continue
        if pos == 0:
            a_prev = atr[k - 1]
            g = gate[k] > 0.5
            if entry_mode == ENTRY_STOP:
                if k > last_stop + cooldown:   # 冷却: 止损触发后禁进场
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
        if pos != 0:
            done = False
            if exit_mode == EXIT_MODE_TIME_SL and not np.isnan(sl):
                if pos == 1 and l[k] <= sl:
                    px = sl if o[k] >= sl else o[k]
                    e_i[nt] = eib; x_i[nt] = k; sd[nt] = pos; epx[nt] = entry_px
                    xpx[nt] = px; cst[nt] = cost_v; rsn[nt] = 4; nt += 1
                    pos = 0; done = True; last_stop = k
                elif pos == -1 and h[k] >= sl:
                    px = sl if o[k] <= sl else o[k]
                    e_i[nt] = eib; x_i[nt] = k; sd[nt] = pos; epx[nt] = entry_px
                    xpx[nt] = px; cst[nt] = cost_v; rsn[nt] = 4; nt += 1
                    pos = 0; done = True; last_stop = k
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
    if pos != 0:
        e_i[nt] = eib; x_i[nt] = n - 1; sd[nt] = pos; epx[nt] = entry_px
        xpx[nt] = c[n - 1]; cst[nt] = cost_v; rsn[nt] = 5; nt += 1
    return e_i[:nt], x_i[:nt], sd[:nt], epx[:nt], xpx[:nt], cst[:nt], rsn[:nt]


simulate_cd_jit = njit(cache=True)(simulate_core_cd)


def main():
    t00 = time.time()
    cdf = load_m1()
    bar = resample_htf(cdf, SPEC["rule"])
    arrs = build_arrays(bar)
    ts = arrs["ts"]
    n = arrs["n"]
    atr_s = pd.Series(arrs["atr"])
    med_q = atr_s.rolling(SPEC["qwin"], min_periods=200).median().to_numpy()
    gate = np.concatenate(([0.0], (arrs["atr"] > med_q).astype(float)[:-1]))
    mode, sig, lup, ldn = build_entry(arrs, SPEC["entry"])
    sig = np.where(sig > 0, sig, 0).astype(np.int8)
    ldn = np.full(n, -1e18)
    hzn = SPEC["days"] * SPEC["bpd"]
    nan_arr = np.full(n, np.nan)

    from v18_static import simulate_ts_jit
    args_base = (arrs["o"], arrs["h"], arrs["l"], arrs["c"], arrs["atr"], mode, sig, lup, ldn,
                 EXIT_MODE_TIME_SL, 0.0, hzn, 0.0, 0.0, SL_NONE, nan_arr, nan_arr,
                 arrs["dxlo10"], arrs["dxhi10"], 1, 0.30, 0.30, gate)
    m_base = summarize(simulate_ts_jit(*args_base), ts, arrs)
    print(f"[base] total={m_base['total']} n={m_base['trades']} worst={m_base['worst']}", flush=True)

    out = {}
    print(f"\n[H_cool] stop+cooldown (pess03):")
    for sname, skind, sm, refN in (("atr4.0", SL_ATR, 4.0, None), ("atr6.0", SL_ATR, 6.0, None),
                                   ("atr8.0", SL_ATR, 8.0, None), ("don55", SL_REF, 0.0, 55)):
        ref_dn, ref_up = _ref_arrays(arrs, refN, nan_arr)
        for cd_name, cd in (("cd5d", 240), ("cd10d", 480), ("cd20d", 960)):
            args = (arrs["o"], arrs["h"], arrs["l"], arrs["c"], arrs["atr"], mode, sig, lup, ldn,
                    EXIT_MODE_TIME_SL, 0.0, hzn, 0.0, sm, skind, ref_dn, ref_up,
                    cd, arrs["dxlo10"], arrs["dxhi10"], 1, 0.30, 0.30, gate)
            r_py = simulate_core_cd(*args)
            r_jit = simulate_cd_jit(*args)
            for nm, A, B in zip(["e", "x", "sd", "ep", "xp", "cs", "rs"], r_py, r_jit):
                assert len(A) == len(B) and np.array_equal(A, B), f"DUAL MISMATCH {sname}.{nm}"
            m = summarize(r_jit, ts, arrs)
            out[f"{sname}+{cd_name}"] = dict(total=m["total"], trades=m["trades"], worst=m["worst"],
                                             n_sl_hit=m["n_sl_hit"], all_pos=m["all_pos"],
                                             by_year=m["by_year"])
            print(f"   {sname}+{cd_name}: total={m['total']:>7} n={m['trades']:>3} worst={m['worst']:>8} "
                  f"hit={m['n_sl_hit']:>3} allpos={m['all_pos']} yr={m['by_year']}")

    best = max(out.items(), key=lambda kv: kv[1]["total"])
    verdict = "H_cool 成立" if (best[1]["total"] > m_base["total"] and best[1]["worst"] > m_base["worst"]) else "H_cool 证伪 — 价格底线概念终结"
    print(f"\n[判定] best={best[0]} total={best[1]['total']} worst={best[1]['worst']} "
          f"(base {m_base['total']}/{m_base['worst']}) -> {verdict}")
    json.dump(dict(base=dict(total=m_base["total"], worst=m_base["worst"], trades=m_base["trades"]),
                   cooldown=out, verdict=verdict),
              open(os.path.join(RES, "cooldown.json"), "w"), indent=1)
    print(f"DONE {(time.time() - t00) / 60:.1f}min -> results_v18/cooldown.json", flush=True)


if __name__ == "__main__":
    main()
