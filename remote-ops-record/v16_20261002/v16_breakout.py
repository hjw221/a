#!/usr/bin/env python3
"""v16_breakout.py — 路线C: 三方擂台 (纯规则突破 vs ML时刻 vs 随机时刻, 全部同一 Chandelier 3.0ATR 出场).

用户路线C原文: Donchian Breakout(海龟通道)/Keltner Channel 突破触发进场, 搭载 3xATR Chandelier 出场,
作为纯 Rule-based 基准与 "随机+Chandelier" "ML+Chandelier" 打三方擂台.

臂:
  don20 / don20_h : Donchian-20 突破 (海龟 System1), _h = champ hours_filter 版
  don55 / don55_h : Donchian-55 突破 (海龟 System2)
  kelt  / kelt_h  : Keltner(EMA20 +- 2*ATR14m5) 突破
口径: m5 信号, 串行一次一仓, chandelier 3.0ATR 追踪出场, 成本/ATR 与 v15 完全同引擎同参数.
对照(来自 controls2.json): model $20078/4862 | random $30051/8517
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import stage_c_loop as sc
from v15_exit import simulate_exit, EXIT_CHANDELIER

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results_v16")
os.makedirs(RES, exist_ok=True)
CHAMP = json.load(open(os.path.join(BASE, "results_research", "champion.json")))
CFG = CHAMP["cfg"]
HOURS = CFG.get("hours_filter")
GEOM = dict(mode=EXIT_CHANDELIER, fixed_tp=0.0, fixed_sl=0.0, chan_atr_mult=3.0,
            n_bracket=0, bracket_tp_mult=0.0, time_hold=0, tight_tp=0.0, tight_sl=0.0, scale_trail=0.75)
TPM, SLM, SLF, HZN, ETOL = float(CFG["tp_mult"]), float(CFG["sl_mult"]), 0.48, 90, 10


def engine(m1_ohl, sig_t, tm, atr, cost):
    m1_t, m1_o, m1_h, m1_l = m1_ohl
    p = GEOM
    return simulate_exit(m1_t, m1_o, m1_h, m1_l, sig_t, tm, atr, cost,
                         p["mode"], TPM, SLM, SLF, HZN, ETOL,
                         p["fixed_tp"], p["fixed_sl"], p["chan_atr_mult"],
                         p["n_bracket"], p["bracket_tp_mult"],
                         p["time_hold"], p["tight_tp"], p["tight_sl"], p["scale_trail"], 1)


def build_signals(m5, kind):
    """返回全期 int8 信号数组 (与 m5 等长): +1 上破 / -1 下破, 未触发 0."""
    c, h, l = m5["close"], m5["high"], m5["low"]
    if kind.startswith("don"):
        N = int(kind[3:])
        dc_hi = h.rolling(N).max().shift(1)
        dc_lo = l.rolling(N).min().shift(1)
        up = c > dc_hi
        dn = c < dc_lo
    elif kind == "kelt":
        ema = c.ewm(span=20, adjust=False).mean()
        tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
        atr14 = tr.rolling(14).mean()
        up = c > (ema + 2.0 * atr14)
        dn = c < (ema - 2.0 * atr14)
    else:
        raise ValueError(kind)
    sig = np.zeros(len(m5), dtype=np.int8)
    sig[up.to_numpy()] = 1
    sig[dn.to_numpy()] = -1
    return sig


def run_arm(m5, m1_ohl, sig_t, hour_arr, atr, cost, folds, kind, use_hours):
    sig = build_signals(m5, kind)
    if use_hours and HOURS:
        sig = np.where(np.isin(hour_arr, HOURS), sig, 0).astype(np.int8)
    fills = 0
    month_pnl = {}
    trades = []
    for oos_s, oos_e in folds:
        oos_mask = (m5.index >= oos_s) & (m5.index <= oos_e)
        tm = sig[oos_mask]
        if (tm != 0).sum() == 0:
            continue
        pnl, wins, filled = engine(m1_ohl, sig_t[oos_mask], tm, atr[oos_mask], cost[oos_mask])
        fills += filled
        mk = str(oos_s.date())[:7]
        month_pnl[mk] = float(pnl[tm != 0].sum())
        oos_pos = np.where(oos_mask)[0]
        for pos in np.where((tm != 0) & (pnl != 0.0))[0]:
            trades.append(dict(month=mk, dir=int(tm[pos]), pnl=float(pnl[pos])))
    s = pd.Series(month_pnl)
    by = {}
    for mk, v in month_pnl.items():
        y = int(mk[:4]); by[y] = by.get(y, 0.0) + v
    t_ = pd.DataFrame(trades)
    long_pnl = float(t_[t_["dir"] == 1]["pnl"].sum()) if len(t_) else 0.0
    short_pnl = float(t_[t_["dir"] == -1]["pnl"].sum()) if len(t_) else 0.0
    share26 = round(100.0 * by.get(2026, 0.0) / max(float(s.sum()), 1e-9), 1)
    return dict(total=round(float(s.sum()), 1), trades=fills,
                avg_per_trade=round(float(s.sum()) / max(fills, 1), 3),
                by_year={k: round(v, 1) for k, v in sorted(by.items())},
                n_months=int(len(month_pnl)), share2026_pct=share26,
                long_pnl=round(long_pnl, 1), short_pnl=round(short_pnl, 1),
                long_share_pct=round(100 * long_pnl / max(long_pnl + short_pnl, 1e-9), 1))


def main():
    t00 = time.time()
    m5, F, cost, atr, m1_ohl = sc._prep()
    sig_t = (m5.index.astype("datetime64[s]").astype("int64") + 300).to_numpy()
    hour_arr = m5.index.hour.to_numpy()
    folds = sc.month_folds(m5.index, "2022-08", "2026-07")
    arms = {}
    plan = [("don20", False), ("don20", True), ("don55", False), ("don55", True),
            ("kelt", False), ("kelt", True)]
    for kind, uh in plan:
        name = kind + ("_h" if uh else "")
        arms[name] = run_arm(m5, m1_ohl, sig_t, hour_arr, atr, cost, folds, kind, uh)
        a = arms[name]
        print(f"[{name}] total={a['total']} trades={a['trades']} avg={a['avg_per_trade']} "
              f"by_year={a['by_year']} share26={a['share2026_pct']}% L/S={a['long_share_pct']}%", flush=True)
    # 擂台总表
    try:
        c2 = json.load(open(os.path.join(BASE, "results_v15", "controls2.json")))
        ref_model = c2["chandelier"]["model"]; ref_rnd = c2["chandelier"]["random"]
    except Exception:
        ref_model = ref_rnd = None
    arena = dict(arms=arms,
                 refs=dict(model=ref_model, random=ref_rnd),
                 meta=dict(engine="chandelier 3.0ATR serial", folds=len(folds),
                           generated=time.strftime("%Y-%m-%d %H:%M:%S")))
    json.dump(arena, open(os.path.join(RES, "breakout.json"), "w"), indent=1)
    if ref_model:
        print(f"[REF] model={ref_model['total']}/{ref_model['trades']} "
              f"random={ref_rnd['total']}/{ref_rnd['trades']}", flush=True)
    sc.say(f"v16breakout DONE {(time.time()-t00)/60:.1f}min: " +
           " | ".join(f"{k}:{v['total']}" for k, v in arms.items()))
    print(f"DONE {(time.time()-t00)/60:.1f}min -> results_v16/breakout.json", flush=True)


if __name__ == "__main__":
    main()
