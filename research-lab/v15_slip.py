#!/usr/bin/env python3
"""v15_slip.py — 吊灯冠军滑点敏感性: cost x2 / x3 (入场+出场各加一档滑点)."""
import os, sys, json
import numpy as np
import pandas as pd
import stage_c_loop as sc
from v15_exit import simulate_exit, EXIT_CHANDELIER

BASE = os.path.dirname(os.path.abspath(__file__))
CHAMP = json.load(open(os.path.join(BASE, "results_research", "champion.json")))
CFG = CHAMP["cfg"]
P = dict(mode=EXIT_CHANDELIER, fixed_tp=0.0, fixed_sl=0.0, chan_atr_mult=3.0,
         n_bracket=0, bracket_tp_mult=0.0, time_hold=0, tight_tp=0.0, tight_sl=0.0, scale_trail=0.75)

def run(mult):
    orig = sc.simulate
    def sim(m1_t, m1_o, m1_h, m1_l, sig_t, p_long, trade, atr, cost, tp_mult, sl_mult,
            sl_floor, sp_mult, horizon, entry_tol):
        pnl, wins, filled = simulate_exit(m1_t, m1_o, m1_h, m1_l, sig_t, trade, atr, cost * mult,
            P["mode"], float(tp_mult), float(sl_mult), sl_floor, horizon, entry_tol,
            P["fixed_tp"], P["fixed_sl"], P["chan_atr_mult"], P["n_bracket"], P["bracket_tp_mult"],
            P["time_hold"], P["tight_tp"], P["tight_sl"], P["scale_trail"], 1)
        return pnl, wins
    sc.simulate = sim
    r = sc.final_eval(CFG)
    sc.simulate = orig
    s = pd.Series({f["month"]: float(f["pnl"]) for f in r.get("folds", [])})
    by = {}
    for m, v in s.items():
        y = int(m[:4]); by[y] = by.get(y, 0.0) + float(v)
    return dict(total=round(float(s.sum()), 1), by_year={k: round(v, 1) for k, v in sorted(by.items())})

if __name__ == "__main__":
    out = {}
    for mult in (1.0, 2.0, 3.0):
        out[f"cost_x{mult:.0f}"] = run(mult)
        print(f"[slip] cost x{mult:.0f}: {out[f'cost_x{mult:.0f}']}", flush=True)
    json.dump(out, open(os.path.join(BASE, "results_v15", "slip.json"), "w"), indent=1)
    print("DONE", flush=True)
