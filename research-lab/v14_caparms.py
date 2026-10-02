#!/usr/bin/env python3
"""v14_caparms.py — Arm 2: champ book 几何美元钳制 Pareto (年度均衡 vs 总量前沿)

原理: simulate() 的障碍由 tp_mult*atr / sl_mult*atr 决定 -> 钳制 atr 到 [lo,hi]
      等价于把单笔美元规模钳到 [tp_mult*lo, tp_mult*hi] (恒定美元风险近似).
      标签/模型/门控全部不动, 只重新仿真交易几何.

臂 (champ tp_mult=1.8):
  base      : 原样复跑 (线程非确定性基准, 应复现 ~$818)
  cap3.5    : TP 封顶 $3.5 (压 2026, 放过 2025)
  cap2.5    : TP 封顶 $2.5 (强压 2026)
  eq2.4_3.0 : 美元风险钳到 [2.4, 3.0] (抬 2022-24 + 压 2026)
  eq1.2_2.0 : 美元风险钳到 [1.2, 2.0] (最激进压平, 接近 v3-pack 尺寸)

输出: results_v14/caparms.json (逐臂 total/by_year/volflat/monthly)
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import stage_c_loop as sc

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results_v14")
os.makedirs(RES, exist_ok=True)

CHAMP = json.load(open(os.path.join(BASE, "results_research", "champion.json")))
CFG = CHAMP["cfg"]
TPM = float(CFG["tp_mult"])

ARMS = {
    "base":     None,
    "cap3.5":   (0.0, 3.5 / TPM),
    "cap2.5":   (0.0, 2.5 / TPM),
    "eq2.4_3.0": (2.4 / TPM, 3.0 / TPM),
    "eq1.2_2.0": (1.2 / TPM, 2.0 / TPM),
}

_orig_sim = sc.simulate

def make_clamped(lo, hi):
    def sim(m1_t, m1_o, m1_h, m1_l, sig_t, p, trade,
            atr, cost, tp_mult, sl_mult, sl_floor, sp_mult, horizon, entry_tol):
        a2 = np.clip(atr, lo, hi)
        return _orig_sim(m1_t, m1_o, m1_h, m1_l, sig_t, p, trade, a2, cost,
                         tp_mult, sl_mult, sl_floor, sp_mult, horizon, entry_tol)
    return sim

def main():
    sc.ALL_FEATS = None
    m5, F, cost, atr, _ = sc._prep()
    # 月度 ATR (vol-flat 口径)
    mo = pd.Series(np.asarray(atr, dtype=float), index=m5.index)
    matr = mo.groupby(m5.index.strftime("%Y-%m")).mean()
    med_atr = float(np.nanmedian(matr.values))

    out_path = os.path.join(RES, "caparms.json")
    out = {}
    for name, clamp in ARMS.items():
        sc.simulate = _orig_sim if clamp is None else make_clamped(*clamp)
        t0 = time.time()
        r = sc.final_eval(CFG)   # 54 折全窗, 与 732 同口径
        s = pd.Series({f["month"]: float(f["pnl"]) for f in r.get("folds", [])})
        by, vf_by = {}, {}
        for m, v in s.items():
            y = int(m[:4]); by[y] = by.get(y, 0.0) + float(v)
            a = matr.get(m, np.nan)
            vfn = float(v) * (med_atr / a) if (a == a and a > 0) else 0.0
            vf_by[y] = vf_by.get(y, 0.0) + vfn
        tot = float(s.sum())
        rec = dict(
            total=tot, trades=int(r["trades"]), win_rate=float(r.get("win_rate", 0)),
            sharpe=float(r["sharpe"]), plr=float(r["plr"]),
            by_year={int(k): round(v, 1) for k, v in sorted(by.items())},
            by_year_volflat={int(k): round(v, 1) for k, v in sorted(vf_by.items())},
            share_2026=(by.get(2026, 0.0) / tot) if tot else None,
            min_year=(min(by.values()) if by else None),
            clamp_atr_dollar=[round(TPM * clamp[0], 2) if clamp else None,
                              round(TPM * clamp[1], 2) if clamp else None],
            monthly={m: round(float(v), 2) for m, v in s.items()},
            secs=round(time.time() - t0),
        )
        out[name] = rec
        json.dump(out, open(out_path, "w"), indent=1)
        sc.say(f"v14 caparm {name}: total {tot:.0f} | 2026占比 {rec['share_2026']:.0%} | "
               f"最差年 {rec['min_year']:.0f} | 逐年 {rec['by_year']} ({rec['secs']}s)")
        print(f"[caparm] {name}: {json.dumps({k: rec[k] for k in ('total','share_2026','min_year','by_year')})}", flush=True)
    sc.simulate = _orig_sim
    json.dump(out, open(out_path, "w"), indent=1)
    print("DONE -> results_v14/caparms.json", flush=True)

if __name__ == "__main__":
    main()
