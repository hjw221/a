#!/usr/bin/env python3
"""v17c_champ.py — 冠军档案导出: 逐笔交易/月度曲线/消融表/t统计 (v17b 三强 + 关键消融)."""
import os, json
import numpy as np
import pandas as pd
from v17_htf import load_m1, resample_htf, build_arrays, build_entry, COSTS, EXIT_MODE_TIME
from v17b_refine import simulate_g_jit

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results_v17")

TOP3 = [
    dict(name="CHAMP_M30_don55s_long_atrmed_t5d", tf="M30", rule="30min", bpd=48, qwin=3024,
         entry="don55s", dir="long", gate="atrmed", exit="t5d", days=5),
    dict(name="R2_H1_don55s_long_none_t5d", tf="H1", rule="1h", bpd=24, qwin=1512,
         entry="don55s", dir="long", gate="none", exit="t5d", days=5),
    dict(name="R3_H1_don20_long_atrp30_t5d", tf="H1", rule="1h", bpd=24, qwin=1512,
         entry="don20", dir="long", gate="atrp30", exit="t5d", days=5),
]


def run_cfg(cdf, spec, cost_mode):
    bar = resample_htf(cdf, spec["rule"])
    arrs = build_arrays(bar)
    ts = arrs["ts"]
    atr_s = pd.Series(arrs["atr"])
    q = atr_s.rolling(spec["qwin"], min_periods=200)
    med_q = q.median().to_numpy()
    p30_q = q.quantile(0.30).to_numpy()
    gates = {"none": np.ones(arrs["n"]),
             "atrmed": np.concatenate(([0.0], (arrs["atr"] > med_q).astype(float)[:-1])),
             "atrp30": np.concatenate(([0.0], (arrs["atr"] > p30_q).astype(float)[:-1]))}
    mode, sig, lup, ldn = build_entry(arrs, spec["entry"])
    if spec["dir"] == "long":
        sig = np.where(sig > 0, sig, 0).astype(np.int8)
        ldn = np.full(arrs["n"], -1e18)
    elif spec["dir"] == "short":
        sig = np.where(sig < 0, sig, 0).astype(np.int8)
        lup = np.full(arrs["n"], 1e18)
    cost = {"base": (0, 0.50, 0.50), "pess03": (1, 0.30, 0.30), "pess05": (1, 0.50, 0.50)}[cost_mode]
    tr = simulate_g_jit(arrs["o"], arrs["h"], arrs["l"], arrs["c"], arrs["atr"],
                         mode, sig, lup, ldn, EXIT_MODE_TIME, 0.0, spec["days"] * spec["bpd"],
                         0.0, 0.0, arrs["dxlo10"], arrs["dxhi10"],
                         cost[0], cost[1], cost[2], gates[spec["gate"]])
    e_i, x_i, sd, epx, xpx, cst, rsn = tr
    pnl = sd * (xpx - epx) - cst
    tdf = pd.DataFrame(dict(
        entry_time=ts[e_i], exit_time=ts[x_i], side=np.where(sd == 1, "L", "S"),
        entry_px=np.round(epx, 2), exit_px=np.round(xpx, 2), cost=np.round(cst, 2),
        pnl=np.round(pnl, 2), hold_bars=x_i - e_i,
        reason=np.where(rsn == 1, "time", np.where(rsn == 5, "eod", "other")),
        atr_entry=np.round(arrs["atr"][np.maximum(e_i - 1, 0)], 2),
    ))
    return tdf, ts


def summarize(tdf):
    pnl = tdf["pnl"].to_numpy()
    yrs = tdf["exit_time"].dt.year.to_numpy()
    by = {int(y): round(float(pnl[yrs == y].sum()), 1) for y in np.unique(yrs)}
    nby = {int(y): int((yrs == y).sum()) for y in np.unique(yrs)}
    mo = tdf.set_index("exit_time")["pnl"].resample("ME").sum()
    eq = mo.cumsum()
    tstat = float(pnl.mean() / (pnl.std(ddof=1) / np.sqrt(len(pnl))))
    return dict(
        total=round(float(pnl.sum()), 1), trades=int(len(pnl)), avg=round(float(pnl.mean()), 2),
        std=round(float(pnl.std(ddof=1)), 2), t_stat=round(tstat, 2),
        win_rate=round(float((pnl > 0).mean()), 3),
        p10=round(float(np.percentile(pnl, 10)), 2), p90=round(float(np.percentile(pnl, 90)), 2),
        worst=round(float(pnl.min()), 2), best=round(float(pnl.max()), 2),
        by_year=by, n_by_year=nby, share2026=round(100.0 * by.get(2026, 0.0) / float(pnl.sum()), 1),
        sharpe=round(float(mo.mean() / (mo.std() + 1e-9) * np.sqrt(12)), 2),
        maxdd=round(float((eq - eq.cummax()).min()), 1),
        med_cost=round(float(tdf["cost"].median()), 2),
        med_atr=round(float(tdf["atr_entry"].median()), 2),
        months={str(k.date())[:7]: round(float(v), 1) for k, v in mo.items()},
        total54=round(float(pnl[tdf["exit_time"] >= "2022-08-01"].sum()), 1),
    )


def main():
    cdf = load_m1()
    out = {}
    for spec in TOP3:
        rec = {}
        for cm in ("base", "pess03", "pess05"):
            tdf, _ = run_cfg(cdf, spec, cm)
            rec[cm] = summarize(tdf)
            if cm == "pess03":
                tj = tdf.copy()
                tj["entry_time"] = tj["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
                tj["exit_time"] = tj["exit_time"].dt.strftime("%Y-%m-%d %H:%M")
                rec["trades_pess03"] = tj.to_dict("records")
        out[spec["name"]] = dict(spec={k: v for k, v in spec.items() if k != "rule"}, **rec)
        s = rec["pess03"]
        print(f"[{spec['name']}] total={s['total']} n={s['trades']} avg={s['avg']} t={s['t_stat']} "
              f"yr={s['by_year']} n/yr={s['n_by_year']} sh26={s['share2026']}% dd={s['maxdd']} "
              f"wr={s['win_rate']} med_cost={s['med_cost']} med_atr={s['med_atr']}", flush=True)
        tdf = pd.DataFrame(rec["trades_pess03"])
        tdf.to_csv(os.path.join(RES, f"trades_{spec['name']}.csv"), index=False)
    # 消融: M30 don55s x {long,both} x {none,atrmed,atrp30} @t5d pess03
    abl = {}
    for di in ("long", "both"):
        for g in ("none", "atrmed", "atrp30"):
            spec = dict(name=f"abl_{di}_{g}", tf="M30", rule="30min", bpd=48, qwin=3024,
                        entry="don55s", dir=di, gate=g, exit="t5d", days=5)
            tdf, _ = run_cfg(cdf, spec, "pess03")
            s = summarize(tdf)
            abl[f"{di}/{g}"] = s
            print(f"[abl {di}/{g}] total={s['total']} n={s['trades']} avg={s['avg']} yr={s['by_year']}", flush=True)
    out["ablation_M30_don55s_t5d_pess03"] = abl
    # 基准对照
    out["benchmarks"] = dict(
        hist_m1_732_7=dict(total=732.7, note="M1历史冠军(54月窗,引擎存疑同款回望写法,2026占88.9%)"),
        v16_m1_kelt_fixed=dict(total=2599.0, trades=27265, avg=0.095,
                               note="M1正确结算真实冠军 kelt+fixed TP2.5/SL0.4/90min, 2024年-110, delta0.1滑点蒸发至236"),
        v17_htf=dict(note="本页: HTF正确结算, 0.3xATR悲观成本"),
    )
    json.dump(out, open(os.path.join(RES, "champion_v17.json"), "w"), indent=1)
    print("DONE -> results_v17/champion_v17.json", flush=True)


if __name__ == "__main__":
    main()
