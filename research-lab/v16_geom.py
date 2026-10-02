#!/usr/bin/env python3
"""v16_geom.py — 正确时域下的吊灯出场几何扫描 (宽度 x 持仓上限), 双信号源.

背景: 时域bug修复后, 出场几何的旧结论(2.5ATR>3.0ATR, 错位域)需重验.
信号源: kelt (Keltner突破, 质量型) + rnd1000 (满载随机 seed2000, 吞吐型).
网格: chan_atr_mult {2.0,2.5,3.0,3.5} x horizon {60,90,120} = 12 几何 x 2 信号源.
口径: 串行一次一仓, 逐月折, 成本同 v15.
"""
import os, json, time
import numpy as np
import pandas as pd
import stage_c_loop as sc
from v15_exit import simulate_exit, EXIT_CHANDELIER

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results_v16")
CHAMP = json.load(open(os.path.join(BASE, "results_research", "champion.json")))
CFG = CHAMP["cfg"]
MULTS = [2.0, 2.5, 3.0, 3.5]
HOLDS = [60, 90, 120]


def engine(m1_ohl, sig_t, tm, atr, cost, mult, hold):
    m1_t, m1_o, m1_h, m1_l = m1_ohl
    return simulate_exit(m1_t, m1_o, m1_h, m1_l, sig_t, tm, atr, cost,
                         EXIT_CHANDELIER, float(CFG["tp_mult"]), float(CFG["sl_mult"]), 0.48,
                         hold, 10, 0.0, 0.0, mult, 0, 0.0, 0, 0.0, 0.0, 0.75, 1)


def main():
    t00 = time.time()
    m5, F, cost, atr, (m1_t, m1_o, m1_h, m1_l) = sc._prep()
    sig_t = (m5.index.astype("datetime64[s]").astype("int64") + 300).to_numpy()
    folds = sc.month_folds(m5.index, "2022-08", "2026-07")
    c, h, l = m5["close"], m5["high"], m5["low"]
    ema = c.ewm(span=20, adjust=False).mean()
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    atr14 = tr.rolling(14).mean()
    sig = np.zeros(len(m5), dtype=np.int8)
    sig[(c > (ema + 2.0 * atr14)).to_numpy()] = 1
    sig[(c < (ema - 2.0 * atr14)).to_numpy()] = -1

    rng = np.random.default_rng(2000)
    ok_feat = np.isfinite(F[CFG["features"]].to_numpy()).all(axis=1)
    rnd = np.zeros(len(m5), dtype=np.int8)
    m5_month = m5.index.strftime("%Y-%m")
    for mk in pd.unique(m5_month):
        m = (m5_month == mk) & ok_feat
        pos = np.where(m)[0]
        if len(pos) == 0:
            continue
        rnd[pos] = rng.choice([1, -1], size=len(pos))
    print(f"[sig] kelt {int((sig!=0).sum())} rnd {int((rnd!=0).sum())}", flush=True)

    m1_ohl = (m1_t, m1_o, m1_h, m1_l)
    out = {}
    for sname, tm_full in (("kelt", sig), ("rnd1000", rnd)):
        for mult in MULTS:
            for hold in HOLDS:
                month_pnl = {}; fills = 0
                for oos_s, oos_e in folds:
                    oos_mask = np.asarray((m5.index >= oos_s) & (m5.index <= oos_e))
                    tm = tm_full[oos_mask]
                    if (tm != 0).sum() == 0:
                        continue
                    pnl, wins, filled = engine(m1_ohl, sig_t[oos_mask], tm,
                                                atr[oos_mask], cost[oos_mask], mult, hold)
                    fills += filled
                    month_pnl[str(oos_s.date())[:7]] = float(pnl[tm != 0].sum())
                s = pd.Series(month_pnl)
                by = {}
                for mk, v in month_pnl.items():
                    y = int(mk[:4]); by[y] = by.get(y, 0.0) + v
                tot = float(s.sum())
                cum = s.cumsum(); dd = float((cum - cum.cummax()).min())
                sharpe = float(s.mean() / s.std() * np.sqrt(12)) if len(s) > 6 and s.std() > 0 else None
                key = f"{sname}_m{mult}_h{hold}"
                out[key] = dict(total=round(tot, 1), trades=fills,
                                avg_per_trade=round(tot / max(fills, 1), 3),
                                by_year={k: round(v, 1) for k, v in sorted(by.items())},
                                share2026_pct=round(100.0 * by.get(2026, 0.0) / max(tot, 1e-9), 1),
                                maxDD_m=round(dd, 1), sharpe_m=round(sharpe, 2) if sharpe else None)
                print(f"[{key}] total={out[key]['total']} n={fills} avg={out[key]['avg_per_trade']} "
                      f"dd={out[key]['maxDD_m']} sh={out[key]['sharpe_m']}", flush=True)
    json.dump(out, open(os.path.join(RES, "geom.json"), "w"), indent=1)
    sc.say(f"v16geom DONE {(time.time()-t00)/60:.1f}min: best kelt " +
           str(max((v["total"], k) for k, v in out.items() if k.startswith("kelt"))))
    print(f"DONE {(time.time()-t00)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
