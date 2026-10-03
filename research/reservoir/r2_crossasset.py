#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
r2_crossasset.py — Use Case 4 验证: R1 作为跨资产非线性时序编码器 (2026-10-03)
================================================================================
用户规格: 把储层状态(DXY/XAG 的非线性记忆指纹)作为 XAUUSD 模型的底层输入。

实验设计 (三段):
  P1 传输检验: DXY/XAG 各自跑 R1 冠军引擎 → 能量预测 R_dxy/R_xag 是否
      领先 XAUUSD 未来实现方差? (宏观传导: 美元/白银动 → 黄金波动)
  P2 嵌合检验: XAU 引擎输入 8→10/12 维 (加 R_dxy/R_xag/d2 滞后特征),
      IC_var 是否超过单资产冠军 0.491? (use case 4 的直接检验)
  P3 对照: 嵌合特征 shuffle → 增益应消失。

因果纪律: DXY/XAG 特征滞后 ≥30min 进入 XAU bar (跨市场时钟不对齐 + 无前视)。
"""
import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/z/my-project/research/reservoir")
import reservoir_engine2 as E2  # noqa: E402

XAU_CSV = "/home/z/my-project/research-lab/data.csv"
DXY_CSV = "/home/z/data-backup/asset/DXYm_M1.csv"
XAG_CSV = "/home/z/data-backup/asset/XAGUSDc_M1.csv"
OUT = "/home/z/my-project/remote-ops-record/r2_20261003/crossasset_results.json"

CHAMP = dict(n_res=50, density=0.10, spectral=0.95, in_scale=0.30, lam=0.9999,
             delta=1e-6, p0=4.0, p_cap=1e4, k_sig=2.0, k_vol=0.5, chaos_gate=0.6,
             smooth=0.5, rate=0.20, dead=0.05, horizon=4, warm=2000, k_max=16,
             spawn_d2=36.0, spawn_cool=96, w_spawn=0.10, lr=0.02, wd=0.01,
             slip_atr=0.30, oz=1.0, target="log", leaks="1.0", n_feat=8,
             lam_ens="", huber=0.0, exec="continuous", schmitt_on=0.25,
             schmitt_hold=8, schmitt_q=0.5)
CHAMP["lam_ens"] = str(CHAMP["lam"])


def ic(pred, y, mask):
    a, b = pred[mask], y[mask]
    return float(np.corrcoef(a, b)[0, 1]) if mask.sum() > 100 else 0.0


def run_engine(csv, label):
    t0 = time.perf_counter()
    bar = E2.load_m15(csv, use_cache=True)
    D = E2.build_feats(bar, horizon=4, n_feat=8)
    res = E2.run_pass(D, CHAMP, 42, label=label)
    R = np.expm1(np.clip(res["yhat"][:, 0], 0, 3))
    print(f"  [{label}] {D['n']} bars {D['ts'][0]}..{D['ts'][-1]} "
          f"({time.perf_counter()-t0:.0f}s) own_IC_var={E2.metrics(D, res, CHAMP)['overall']['ic_var']}")
    return D, res, R


def asof_lookup(target_ts_ns, avail_ns, values, lag_min):
    """target 时刻 -> 最新 avail-lag <= target 的值."""
    cut = avail_ns - int(lag_min) * 60 * 1e9
    idx = np.searchsorted(cut, target_ts_ns, side="right") - 1
    ok = idx >= 0
    out = np.full(len(target_ts_ns), 0.0)
    out[ok] = np.clip(values[idx[ok]], -4, 4)
    return out


def main():
    print("== P1a: XAU 单资产冠军 (回归锚) ==")
    D_x, res_x, R_x = run_engine(XAU_CSV, "xau")
    n = D_x["n"]
    valid = np.zeros(n, bool); valid[CHAMP["warm"]:n - 40] = True
    yv = D_x["y_var_raw"]
    m = valid & np.isfinite(yv)
    ic_x = ic(R_x, yv, m)
    print(f"  XAU champion IC_var(canonical) = {ic_x:.4f}")

    print("== P1b: DXY / XAG 引擎 ==")
    D_d, res_d, R_d = run_engine(DXY_CSV, "dxy")
    D_s, res_s, R_s = run_engine(XAG_CSV, "xag")

    # ---- 传输检验: R_dxy(t) vs XAU 未来方差 (多视界) ----
    ts_x_close = D_x["ts"].to_numpy()  # bar open; 预测在 close 可用
    avail_d = (D_d["ts"] + pd.Timedelta(minutes=15)).to_numpy()
    avail_s = (D_s["ts"] + pd.Timedelta(minutes=15)).to_numpy()
    ts_x_ns = ts_x_close.astype("datetime64[ns]").astype(np.int64)
    trans = {}
    for lag in (30, 60, 120):
        Rd = asof_lookup(ts_x_ns, avail_d.astype(np.int64), R_d, lag)
        Rs = asof_lookup(ts_x_ns, avail_s.astype(np.int64), R_s, lag)
        for hh in (4, 16):
            y = D_x["vh"][hh]
            mm = valid & np.isfinite(y)
            trans[f"dxy_lag{lag}_h{hh}"] = round(ic(Rd, y, mm), 4)
            trans[f"xag_lag{lag}_h{hh}"] = round(ic(Rs, y, mm), 4)
    # 领先性参照: XAU 自身能量 vs 未来方差
    for hh in (4, 16):
        trans[f"xau_self_h{hh}"] = round(ic(R_x, D_x["vh"][hh], m & np.isfinite(D_x["vh"][hh])), 4)
    print("  transmission:", trans)

    # ---- P2: 嵌合引擎 (XAU 8 + DXY/XAG 滞后能量) ----
    print("== P2: 嵌合 XAU+DXY+XAG 引擎 ==")
    results = {}
    for name, extra in (
        ("aug_R_only", [asof_lookup(ts_x_ns, avail_d.astype(np.int64), R_d, 30),
                        asof_lookup(ts_x_ns, avail_s.astype(np.int64), R_s, 30)]),
        ("aug_R_d2", [asof_lookup(ts_x_ns, avail_d.astype(np.int64), R_d, 30),
                      asof_lookup(ts_x_ns, avail_s.astype(np.int64), R_s, 30),
                      np.clip(asof_lookup(ts_x_ns, avail_d.astype(np.int64), res_d["d2"], 30) / 25.0, 0, 4),
                      np.clip(asof_lookup(ts_x_ns, avail_s.astype(np.int64), res_s["d2"], 30) / 25.0, 0, 4)]),
    ):
        t0 = time.perf_counter()
        D_aug = dict(D_x)  # shallow copy: 共享标签, 换 U
        D_aug["U"] = np.column_stack([D_x["U"]] + [np.asarray(e, float) for e in extra])
        res_aug = E2.run_pass(D_aug, CHAMP, 42, label=name)
        R_aug = np.expm1(np.clip(res_aug["yhat"][:, 0], 0, 3))
        ic_aug = ic(R_aug, yv, m)
        im = {}
        for hh in (4, 8, 16, 32):
            y = D_x["vh"][hh]
            im[f"h{hh}"] = round(ic(R_aug, y, m & np.isfinite(y)), 4)
        results[name] = dict(ic_var=round(ic_aug, 4), ic_multi=im,
                             ms=round(res_aug["timing"]["total_ms"]["mean"], 3))
        print(f"  {name}: IC_var={ic_aug:.4f} multi={im} ({time.perf_counter()-t0:.0f}s)")

    # ---- P3: shuffle 对照 (嵌合特征乱序) ----
    print("== P3: shuffle 对照 ==")
    rng = np.random.default_rng(555)
    Rd_shuf = R_d[rng.permutation(len(R_d))]
    Rs_shuf = R_s[rng.permutation(len(R_s))]
    D_aug = dict(D_x)
    D_aug["U"] = np.column_stack([D_x["U"],
                                  asof_lookup(ts_x_ns, avail_d.astype(np.int64), Rd_shuf, 30),
                                  asof_lookup(ts_x_ns, avail_s.astype(np.int64), Rs_shuf, 30)])
    res_shuf = E2.run_pass(D_aug, CHAMP, 42, label="aug_shuffle")
    R_shuf = np.expm1(np.clip(res_shuf["yhat"][:, 0], 0, 3))
    ic_shuf = ic(R_shuf, yv, m)
    print(f"  aug_shuffle: IC_var={ic_shuf:.4f}")

    doc = dict(
        meta=dict(generated="2026-10-03", champion_cfg="nres50 lam0.9999 log feat8",
                  xau_ic=round(ic_x, 4), transmission=trans),
        augmented=results, shuffle_control=round(ic_shuf, 4))
    json.dump(doc, open(OUT, "w"), indent=1)
    print(f"== DONE -> {OUT} ==")


if __name__ == "__main__":
    main()
