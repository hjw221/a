#!/usr/bin/env python3
"""v18b_disaster.py — 静态灾难止损的三个根因检验 (v18 否定性结果的钉死实验).

v18 主结果 (引擎口径 pess03, base=$2,980.1/159笔/worst=-168.3):
  12种静态止损(2~8 ATR, don10/20/40/55)无一保留$2,500+; 最好atr4.0=$2,181(-27%);
  overlay纯保险: avoided~$2,2-2,5K 但 killed更大, net全负(-$826~-$3,117);
  触发率惊人(atr4.0=59%, 8ATR=36%): 5日动量路径的正常回踩就达数倍ATR.

用户假设的字面拆解 (跑前锁定三问):
  Q1 极限宽度: atr10/12/16 是否单调趋近base但永不超越? (全宽度负贡献确认)
  Q2 可分离性: base 159笔的MAE(ATR单位)分布是否bimodal(正常回踩vs灾难尾部有无干净gap)?
     若连续无gap -> "只挡黑天鹅"的阈值物理上不存在.
  Q3 "快速破位": 用户原话是"系统性暴跌或假突破**快速**破位时才斩仓" —
     时变止损: 只在进场后W根内止损生效, 之后撤除纯时间出场 (快跌=斩, 慢回踩=放行).
     W ∈ {1d=48, 2d=96, 3d=144}; 止损 ∈ {atr4.0, atr6.0, don55}.

预注册判定:
  J1 若 atr10/12/16 total 单调升且全部 < base -> 全宽度负贡献 (静态止损在该结构上无生存空间)
  J2 若 MAE 分布 p50~p90 连续覆盖 2~10 ATR 无 bimodal gap -> 阈值不存在, Q2 证伪
  J3 若某时变止损保留 >=$2,800 且 worst 改善 -> "早期灾难止损"成立; 否则同样证伪
  MAE-终局相关性: corr(MAE, final_pnl) 显著为正 -> 深回踩笔恰恰是最终盈利笔(斩=纯误杀)

结算纪律: 同v18 (进场瞬间锁定止损价, 触发当根即出场, gap按更差open, 止损优先于时间).
"""
import os, json, time
import numpy as np
import pandas as pd
from numba import njit
from v17_htf import load_m1, resample_htf, build_arrays, build_entry, COSTS, ENTRY_STOP
from v18_static import (simulate_ts_jit, summarize, trades_df, STOPS as V18_STOPS,
                        SL_ATR, SL_REF, SL_NONE, EXIT_MODE_TIME_SL, _ref_arrays, SPEC)

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results_v18")


# ================= 时变止损引擎: sl_until>0 时只在 k < eib+sl_until 检查止损 =================
def simulate_core_tv(o, h, l, c, atr, entry_mode, sig, line_up, line_dn,
                     exit_mode, w, hzn, tp_m, sl_m, sl_kind, sl_ref_dn, sl_ref_up,
                     sl_until, dxlo, dxhi, cost_mode, cost_flat, cost_atr_frac, gate):
    n = len(o)
    e_i = np.empty(n, np.int64); x_i = np.empty(n, np.int64)
    sd = np.empty(n, np.int8); epx = np.empty(n, np.float64)
    xpx = np.empty(n, np.float64); cst = np.empty(n, np.float64)
    rsn = np.empty(n, np.int8)
    nt = 0; pos = 0; entry_px = 0.0; hi_s = 0.0; lo_s = 0.0
    eib = 0; tp = 0.0; sl = np.nan; cost_v = 0.0; pending = False
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
        if pos != 0:
            done = False
            # 时变静态止损: 仅 k < eib + sl_until 时检查 (sl_until=0 表示全程)
            if exit_mode == EXIT_MODE_TIME_SL and not np.isnan(sl):
                if sl_until <= 0 or k < eib + sl_until:
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


simulate_tv_jit = njit(cache=True)(simulate_core_tv)


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

    # base 臂 (复用 v18 引擎, none)
    args_base = (arrs["o"], arrs["h"], arrs["l"], arrs["c"], arrs["atr"], mode, sig, lup, ldn,
                 EXIT_MODE_TIME_SL, 0.0, hzn, 0.0, 0.0, SL_NONE, nan_arr, nan_arr,
                 arrs["dxlo10"], arrs["dxhi10"], 1, 0.30, 0.30, gate)
    tr_base = simulate_ts_jit(*args_base)
    m_base = summarize(tr_base, ts, arrs)
    e_i, x_i, sd, epx, xpx, cst, rsn = tr_base
    pnl_base = sd * (xpx - epx) - cst
    print(f"[base] total={m_base['total']} n={m_base['trades']} worst={m_base['worst']}", flush=True)

    # ================= Q2: MAE 分布审计 (可分离性检验) =================
    l_, o_, h_, atr_ = arrs["l"], arrs["o"], arrs["h"], arrs["atr"]
    mae_atr = np.empty(len(e_i))
    for t in range(len(e_i)):
        e, x = int(e_i[t]), int(x_i[t])
        a_e = atr_[e - 1]
        if sd[t] == 1:
            mae_atr[t] = (epx[t] - l_[e:x + 1].min()) / a_e
        else:
            mae_atr[t] = (h_[e:x + 1].max() - epx[t]) / a_e
    corr = float(np.corrcoef(mae_atr, pnl_base)[0, 1])
    qs = np.percentile(mae_atr, [50, 75, 90, 95, 99])
    # 直方 (1 ATR 分桶)
    hist, edges = np.histogram(mae_atr, bins=np.arange(0, 22, 1.0))
    deep = mae_atr > 6.0
    print(f"\n[Q2] MAE(ATR units) p50={qs[0]:.1f} p75={qs[1]:.1f} p90={qs[2]:.1f} "
          f"p95={qs[3]:.1f} p99={qs[4]:.1f} max={mae_atr.max():.1f}")
    print(f"[Q2] corr(MAE, final_pnl) = {corr:.3f}  (正=深回踩笔恰是盈利笔)")
    print(f"[Q2] MAE hist (1-ATR buckets, n={len(e_i)}):")
    for cnt, lo in zip(hist, edges[:-1]):
        if cnt:
            bar_ = "#" * int(cnt)
            print(f"   {lo:>4.0f}-{lo+1:>3.0f} ATR | {cnt:>3} {bar_}")
    print(f"[Q2] MAE>6ATR: n={int(deep.sum())} their final pnl sum={pnl_base[deep].sum():.1f} "
          f"(正->深回踩后回血, 斩=误杀)")
    mae_deep_pos = int((pnl_base[deep] > 0).sum())
    print(f"[Q2] MAE>6ATR 笔中最终盈利笔数: {mae_deep_pos}/{int(deep.sum())}")

    # ================= Q1: 极限宽度 atr10/12/16 =================
    print(f"\n[Q1] extreme widths (pess03, full-horizon stop):")
    q1 = {}
    for k in (10.0, 12.0, 16.0, 20.0):
        args = (arrs["o"], arrs["h"], arrs["l"], arrs["c"], arrs["atr"], mode, sig, lup, ldn,
                EXIT_MODE_TIME_SL, 0.0, hzn, 0.0, k, SL_ATR, nan_arr, nan_arr,
                arrs["dxlo10"], arrs["dxhi10"], 1, 0.30, 0.30, gate)
        r = simulate_ts_jit(*args)
        m = summarize(r, ts, arrs)
        q1[f"atr{k}"] = dict(total=m["total"], trades=m["trades"], worst=m["worst"],
                             n_sl_hit=m["n_sl_hit"], all_pos=m["all_pos"])
        print(f"   atr{k:>4}: total={m['total']:>7} n={m['trades']:>3} worst={m['worst']:>8} "
              f"hit={m['n_sl_hit']:>3} allpos={m['all_pos']}")

    # ================= Q3: 时变止损 (快速破位才斩) =================
    print(f"\n[Q3] time-variant stop (active first W bars only, then removed):")
    q3 = {}
    for W_name, W in (("1d", 48), ("2d", 96), ("3d", 144)):
        for sname, skind, sm, refN in (("atr4.0", SL_ATR, 4.0, None), ("atr6.0", SL_ATR, 6.0, None),
                                       ("don55", SL_REF, 0.0, 55)):
            ref_dn, ref_up = _ref_arrays(arrs, refN, nan_arr)
            args = (arrs["o"], arrs["h"], arrs["l"], arrs["c"], arrs["atr"], mode, sig, lup, ldn,
                    EXIT_MODE_TIME_SL, 0.0, hzn, 0.0, sm, skind, ref_dn, ref_up,
                    W, arrs["dxlo10"], arrs["dxhi10"], 1, 0.30, 0.30, gate)
            r_py = simulate_core_tv(*args)
            r_jit = simulate_tv_jit(*args)
            for nm, A, B in zip(["e", "x", "sd", "ep", "xp", "cs", "rs"], r_py, r_jit):
                assert len(A) == len(B) and np.array_equal(A, B), f"DUAL MISMATCH {W_name}.{sname}.{nm}"
            m = summarize(r_jit, ts, arrs)
            q3[f"{sname}@{W_name}"] = dict(total=m["total"], trades=m["trades"], worst=m["worst"],
                                           n_sl_hit=m["n_sl_hit"], all_pos=m["all_pos"],
                                           by_year=m["by_year"])
            print(f"   {sname}@{W_name}: total={m['total']:>7} n={m['trades']:>3} worst={m['worst']:>8} "
                  f"hit={m['n_sl_hit']:>3} allpos={m['all_pos']} yr={m['by_year']}")

    # ---- 判定汇总 ----
    print(f"\n[J1] {'CONFIRMED 全宽度负贡献' if all(q1[k]['total'] < m_base['total'] for k in q1) else 'REFUTED'}"
          f" (base={m_base['total']})")
    gaps = np.diff(np.sort(mae_atr))
    max_gap_atr = float(gaps.max())
    gap_pos = float(np.sort(mae_atr)[int(np.argmax(gaps))]) if len(gaps) else 0.0
    print(f"[J2] MAE max gap = {max_gap_atr:.2f} ATR @ {gap_pos:.1f} ATR "
          f"({'无干净分离带' if max_gap_atr < 3.0 else '存在分离带?!'})")
    best_tv = max(q3.items(), key=lambda kv: kv[1]["total"])
    print(f"[J3] best time-variant = {best_tv[0]} total={best_tv[1]['total']} "
          f"(base {m_base['total']}) -> {'成立' if best_tv[1]['total'] >= 2800 else '证伪'}")

    out = dict(base=dict(total=m_base["total"], worst=m_base["worst"], trades=m_base["trades"]),
               mae=dict(p={str(q): round(float(v), 2) for q, v in zip([50, 75, 90, 95, 99], qs)},
                        max=round(float(mae_atr.max()), 2), corr_pnl=round(corr, 3),
                        hist={str(float(lo)): int(c) for c, lo in zip(hist, edges[:-1]) if c},
                        deep6_n=int(deep.sum()), deep6_pnl=round(float(pnl_base[deep].sum()), 1),
                        deep6_pos_n=mae_deep_pos),
               q1_extreme=q1, q3_time_variant=q3,
               j1=bool(all(q1[k]["total"] < m_base["total"] for k in q1)),
               j2_max_gap_atr=round(max_gap_atr, 2), j2_gap_pos=round(gap_pos, 2))
    json.dump(out, open(os.path.join(RES, "disaster_deep.json"), "w"), indent=1)
    print(f"\nDONE {(time.time() - t00) / 60:.1f}min -> results_v18/disaster_deep.json", flush=True)


if __name__ == "__main__":
    main()
