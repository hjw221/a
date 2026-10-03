#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
r2_linkage.py — R1(冠军配置) × v3bal_ens 联动实验 (2026-10-03)
===============================================================
用户指令: "要不就试试联动v3bal_ens"

数据:
  * v3bal_ens 修复版交易流 (1657笔, 2024-08~2026-07, 服务器重跑, 时间戳已验证)
  * R1 v2 冠军预测流 (champ_s42.npz: M15 ŷ_var(log) / ŷ_skew / d2 Mahalanobis)

联动机制 (对应四个 use case):
  L1 执行择时闸门 (战场2b 滑点卫士): 信号时刻查 R1 能量前瞻 R=expm1(ŷ_var)
      - gate_high: R>=θ 才交易 (能量扩张期执行) / gate_low: R<=θ 才交易 (平静期执行)
  L2 动态仓位呼吸 (use case 2): size = min(1, 1/R) — 预测高波时缩仓
  L3 断路器 (use case 3): d2>thr 时冻结新入场
  L0 对照: shuffle R1 预测时间轴 → 联动增益应消失 (学习真实性)

口径: v3bal_ens pnl 已含点差成本 (1oz); sizing 线性缩放 PnL。
"""
import json
import numpy as np
import pandas as pd

TRADES = "/home/z/my-project/remote-ops-record/r2_20261003/trades_v3bal_ens_FIXED.csv"
NPZ = "/tmp/r2arms/champ_s42.npz"
OUT = "/home/z/my-project/remote-ops-record/r2_20261003/linkage_results.json"


def load():
    tr = pd.read_csv(TRADES)
    tr["signal_time"] = pd.to_datetime(tr["signal_time"])
    tr["entry_time"] = pd.to_datetime(tr["entry_time"])
    z = np.load(NPZ)
    ts = pd.to_datetime(z["ts"], unit="ns")  # M15 bar open times
    yhat = z["yhat"]          # (n,2): [log var_rel, skew]
    d2 = z["d2"]
    # 预测在 bar t 收盘可得 => 可用时刻 = ts + 15min
    avail = ts + pd.Timedelta(minutes=15)
    R = np.expm1(np.clip(yhat[:, 0], 0, 3))  # 相对能量比 (~0-19)
    return tr, avail.to_numpy(), R, d2, yhat[:, 1]


def lookup(signal_times, avail, values):
    """信号时刻 -> 最新可用预测 (avail <= signal). avail 已排序."""
    idx = np.searchsorted(avail, signal_times.astype("datetime64[ns]").astype(np.int64), side="right") - 1
    ok = idx >= 0
    out = np.full(len(signal_times), np.nan)
    out[ok] = values[idx[ok]]
    return out


def summarize(pnl, sizes=None, label=""):
    eq = np.cumsum(pnl)
    dd = float((eq - np.maximum.accumulate(eq)).min()) if len(eq) else 0.0
    wins = pnl[pnl > 0]; losses = pnl[pnl < 0]
    return dict(
        label=label, n=int(len(pnl)),
        pnl=round(float(pnl.sum()), 1),
        wr=round(float((pnl > 0).mean()), 3),
        plr=round(float(wins.sum() / max(-losses.sum(), 1e-9)), 2),
        sharpe=round(float(pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(252 * 22)), 2),
        maxdd=round(dd, 1),
        avg_size=round(float(np.mean(sizes)) if sizes is not None else 1.0, 3),
    )


def by_year(tr, pnl, sizes):
    out = {}
    yr = tr["signal_time"].dt.year.to_numpy()
    for y in sorted(set(yr.tolist())):
        m = yr == y
        out[int(y)] = round(float((pnl[m] * (sizes[m] if sizes is not None else 1.0)).sum()), 1)
    return out


def main():
    tr, avail, R, d2, skew = load()
    sig = tr["signal_time"].to_numpy()
    R_sig = lookup(sig, avail, R)
    d2_sig = lookup(sig, avail, d2)
    sk_sig = lookup(sig, avail, skew)
    base_pnl = tr["pnl"].to_numpy(float)
    med_R = float(np.nanmedian(R_sig))
    print(f"trades={len(tr)}  R@signal: med={med_R:.2f} p10={np.nanpercentile(R_sig,10):.2f} "
          f"p90={np.nanpercentile(R_sig,90):.2f}  d2@signal: med={np.nanmedian(d2_sig):.1f} "
          f"p95={np.nanpercentile(d2_sig,95):.1f}")

    results = {"meta": dict(trades=len(tr), med_R=round(med_R, 3),
                            span=[str(tr['signal_time'].min()), str(tr['signal_time'].max())])}

    # ---- L0 基线 ----
    results["baseline"] = dict(**summarize(base_pnl, label="v3bal_ens flat 1oz"),
                               by_year=by_year(tr, base_pnl, None))

    # ---- L1 执行择时: 能量扩张期才执行 (R>=θ) ----
    l1 = {}
    for th in (0.8, 1.0, 1.2, 1.5):
        m = R_sig >= th
        l1[f"gate_high_R{th}"] = dict(**summarize(base_pnl[m], label=f"R>={th}"),
                                      by_year=by_year(tr[m], base_pnl[m], None))
    for th in (0.7, 0.9, 1.1):
        m = R_sig <= th
        l1[f"gate_low_R{th}"] = dict(**summarize(base_pnl[m], label=f"R<={th}"),
                                     by_year=by_year(tr[m], base_pnl[m], None))
    results["L1_timing"] = l1

    # ---- L2 呼吸阀: size = min(1, med_R / R) ----
    l2 = {}
    for cap_mode, cap in (("cap1", 1.0), ("cap075", 0.75), ("floor05", 0.5)):
        if cap_mode == "floor05":
            size = np.clip(med_R / np.maximum(R_sig, 1e-9), 0.5, 1.0)
        else:
            size = np.clip(med_R / np.maximum(R_sig, 1e-9), 0.0, cap)
        sized = base_pnl * size
        l2[f"breathe_{cap_mode}"] = dict(**summarize(sized, sizes=size, label=cap_mode),
                                         by_year=by_year(tr, base_pnl, size))
    results["L2_breathing"] = l2

    # ---- L3 断路器: d2 > thr 冻结新入场 ----
    l3 = {}
    for thr in (20.0, 30.0, 40.0):
        m = d2_sig <= thr
        l3[f"break_d2{thr:.0f}"] = dict(**summarize(base_pnl[m], label=f"d2<={thr:.0f}"),
                                        by_year=by_year(tr[m], base_pnl[m], None))
    results["L3_breaker"] = l3

    # ---- L4 组合: 断路器 + 呼吸 ----
    l4 = {}
    m = d2_sig <= 30.0
    size = np.clip(med_R / np.maximum(R_sig, 1e-9), 0.5, 1.0)
    sized = base_pnl * size
    l4["breaker30_breathe"] = dict(**summarize(sized, sizes=size, label="combo"),
                                   by_year=by_year(tr, base_pnl, size))
    results["L4_combo"] = l4

    # ---- 对照: shuffle R1 预测 (时间轴乱序) ----
    rng = np.random.default_rng(777)
    perm = rng.permutation(len(R))
    R_shuf = R[perm]
    d2_shuf = d2[perm]
    R_shuf_sig = lookup(sig, avail, R_shuf)
    d2_shuf_sig = lookup(sig, avail, d2_shuf)
    ctl = {}
    m = R_shuf_sig >= 1.0
    ctl["gate_high_R1_shuffle"] = dict(**summarize(base_pnl[m], label="shuffle R>=1"),
                                       by_year=by_year(tr[m], base_pnl[m], None))
    size_s = np.clip(med_R / np.maximum(R_shuf_sig, 1e-9), 0.5, 1.0)
    ctl["breathe_shuffle"] = dict(**summarize(base_pnl * size_s, sizes=size_s, label="shuffle breathe"),
                                  by_year=by_year(tr, base_pnl, size_s))
    m = d2_shuf_sig <= 30.0
    ctl["breaker30_shuffle"] = dict(**summarize(base_pnl[m], label="shuffle d2<=30"),
                                    by_year=by_year(tr[m], base_pnl[m], None))
    results["control_shuffle"] = ctl

    with open(OUT, "w") as f:
        json.dump(results, f, indent=1, default=str)

    # ---- 打印核心表 ----
    print("\n== 基线 ==")
    b = results["baseline"]
    print(f"  {b['label']}: n={b['n']} PnL={b['pnl']} PLR={b['plr']} Sharpe={b['sharpe']} by_year={b['by_year']}")
    print("\n== L1 择时 (能量前瞻闸门) ==")
    for k, v in l1.items():
        print(f"  {k:18s} n={v['n']:4d} PnL={v['pnl']:8.1f} PLR={v['plr']:5.2f} Sharpe={v['sharpe']:5.2f} per/trade={v['pnl']/max(v['n'],1):5.2f}")
    print("\n== L2 呼吸阀 ==")
    for k, v in l2.items():
        print(f"  {k:18s} n={v['n']:4d} PnL={v['pnl']:8.1f} PLR={v['plr']:5.2f} avg_size={v['avg_size']}")
    print("\n== L3 断路器 ==")
    for k, v in l3.items():
        print(f"  {k:18s} n={v['n']:4d} PnL={v['pnl']:8.1f} PLR={v['plr']:5.2f} per/trade={v['pnl']/max(v['n'],1):5.2f}")
    print("\n== L4 组合 ==")
    for k, v in l4.items():
        print(f"  {k:22s} n={v['n']:4d} PnL={v['pnl']:8.1f} PLR={v['plr']:5.2f}")
    print("\n== 对照 (shuffle) ==")
    for k, v in ctl.items():
        print(f"  {k:24s} n={v['n']:4d} PnL={v['pnl']:8.1f} PLR={v['plr']:5.2f}")
    print(f"\n== DONE -> {OUT} ==")


if __name__ == "__main__":
    main()
