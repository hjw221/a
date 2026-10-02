#!/usr/bin/env python3
"""v16_diag2.py — P5 补充: 采样率扫描闭合 N 效应 (v16_diag 的 P3 配平因长持仓撞车未真正配平).

设计: 每折随机抽样数 = n_sig * r, r in {0.1,0.2,0.35,0.5,0.7,1.0}, 方向随机;
      5 seeds/r -> (fills, PnL) 曲线; 同 r 跑 model 方向版对照方向alpha随N的变化.
判读: 在 fills=4862 (model 实际成交数) 处插值随机流 PnL, 与 model $20078 对比 -> N 效应终判.
"""
import os, json, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import stage_c_loop as sc
from v15_exit import simulate_exit, EXIT_CHANDELIER

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results_v16")
CHAMP = json.load(open(os.path.join(BASE, "results_research", "champion.json")))
CFG = CHAMP["cfg"]
GEOM = dict(mode=EXIT_CHANDELIER, fixed_tp=0.0, fixed_sl=0.0, chan_atr_mult=3.0,
            n_bracket=0, bracket_tp_mult=0.0, time_hold=0, tight_tp=0.0, tight_sl=0.0, scale_trail=0.75)
TPM, SLM, SLF, HZN, ETOL = float(CFG["tp_mult"]), float(CFG["sl_mult"]), 0.48, 90, 10
RATES = [0.1, 0.2, 0.35, 0.5, 0.7, 1.0]
SEEDS = [2000 + i for i in range(5)]


def engine(m1_ohl, sig_t, tm, atr, cost):
    m1_t, m1_o, m1_h, m1_l = m1_ohl
    p = GEOM
    return simulate_exit(m1_t, m1_o, m1_h, m1_l, sig_t, tm, atr, cost,
                         p["mode"], TPM, SLM, SLF, HZN, ETOL,
                         p["fixed_tp"], p["fixed_sl"], p["chan_atr_mult"],
                         p["n_bracket"], p["bracket_tp_mult"],
                         p["time_hold"], p["tight_tp"], p["tight_sl"], p["scale_trail"], 1)


def main():
    t00 = time.time()
    m5, F, cost, atr, (m1_t, m1_o, m1_h, m1_l) = sc._prep()
    X = F[CFG["features"]]
    hours_f = CFG.get("hours_filter")
    hour_arr = m5.index.hour.to_numpy()
    params = dict(CFG["lgb"]); params["num_threads"] = int(os.environ.get("ML_THREADS", "26"))
    sig_t = (m5.index.astype("datetime64[s]").astype("int64") + 300).to_numpy()
    folds = sc.month_folds(m5.index, "2022-08", "2026-07")
    TRAIN_CAP = pd.Timedelta(days=CFG.get("train_cap_days", 450))
    q_gate = CFG.get("q_gate", 0.0)

    fold_ctx = []
    for oos_s, oos_e in folds:
        tr_end = oos_s - pd.Timedelta(minutes=120)
        tr_start = oos_s - TRAIN_CAP
        tr_mask = (m5.index >= tr_start) & (m5.index < tr_end)
        oos_mask = (m5.index >= oos_s) & (m5.index <= oos_e)
        Xtr, ytr = X[tr_mask].to_numpy(), sc._cache["y"][tr_mask]
        ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr)
        Xtr, ytr = Xtr[ok], ytr[ok]
        Xoos = X[oos_mask].to_numpy()
        ok_o = np.isfinite(Xoos).all(axis=1)
        if len(Xtr) < 5000 or ok_o.sum() < 100:
            continue
        bst = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=CFG.get("rounds", 200))
        pv = np.full(len(Xoos), np.nan)
        pv[ok_o] = bst.predict(Xoos[ok_o])
        if hours_f:
            cand = np.where(np.isin(hour_arr[oos_mask], hours_f) & ok_o)[0]
        else:
            cand = np.where(ok_o)[0]
        n_sig_cap = int(len(cand))
        fold_ctx.append(dict(oos_pos=np.where(oos_mask)[0], cand=cand, pv=pv, n_avail=n_sig_cap))
    print(f"[ctx] {len(fold_ctx)} folds prepared ({(time.time()-t00)/60:.1f}min)", flush=True)

    def run_stream(rng, rate, use_model_dir):
        tot = 0.0; fl = 0
        by_year = {}
        for fc in fold_ctx:
            K = int(round(fc["n_avail"] * rate))
            cand = fc["cand"]
            if K <= 0 or len(cand) == 0:
                continue
            pick = rng.choice(cand, size=min(K, len(cand)), replace=False)
            tm = np.zeros(len(fc["pv"]), dtype=np.int8)
            if use_model_dir:
                d = np.where(fc["pv"][pick] > 0.5, 1, -1)
                nanp = ~np.isfinite(fc["pv"][pick])
                if nanp.any():
                    d[nanp] = rng.choice([1, -1], size=int(nanp.sum()))
                tm[pick] = d
            else:
                tm[pick] = rng.choice([1, -1], size=len(pick))
            pnl, wins, filled = engine((m1_t, m1_o, m1_h, m1_l),
                                       sig_t[fc["oos_pos"]], tm, atr[fc["oos_pos"]], cost[fc["oos_pos"]])
            m_pnl = float(pnl[tm != 0].sum())
            tot += m_pnl; fl += filled
            mk = str(pd.Timestamp(sig_t[fc["oos_pos"][0]] * 10**9).date())[:7]
            y = int(mk[:4]); by_year[y] = by_year.get(y, 0.0) + m_pnl
        return tot, fl, by_year

    scan = {}
    for rate in RATES:
        for label, umd in (("rnd", False), ("rnd_dir", True)):
            rows = []
            for sd in SEEDS:
                tot, fl, by = run_stream(np.random.default_rng(sd), rate, umd)
                rows.append(dict(seed=sd, total=round(tot, 1), fills=fl))
            t_ = np.array([r["total"] for r in rows]); f_ = np.array([r["fills"] for r in rows])
            scan[f"{label}@{rate}"] = dict(rate=rate, mean=round(float(t_.mean()), 1),
                                           std=round(float(t_.std()), 1),
                                           fills_mean=round(float(f_.mean()), 0),
                                           per_trade=round(float(t_.mean() / max(f_.mean(), 1)), 3),
                                           seeds=rows)
            print(f"[scan] {label}@{rate}: PnL={scan[f'{label}@{rate}']['mean']} fills~{int(f_.mean())} "
                  f"per={scan[f'{label}@{rate}']['per_trade']}", flush=True)
        sc.say(f"v16diag2 scan r={rate}: rnd {scan[f'rnd@{rate}']['mean']:.0f} / rnd_dir {scan[f'rnd_dir@{rate}']['mean']:.0f}")

    # 在 fills=4862 处插值 (对 rnd 流的 fills-PnL 曲线)
    fx = np.array([scan[f"rnd@{r}"]["fills_mean"] for r in RATES], dtype=float)
    fy = np.array([scan[f"rnd@{r}"]["mean"] for r in RATES], dtype=float)
    fx2 = np.array([scan[f"rnd_dir@{r}"]["fills_mean"] for r in RATES], dtype=float)
    fy2 = np.array([scan[f"rnd_dir@{r}"]["mean"] for r in RATES], dtype=float)
    interp = float(np.interp(4862, fx, fy))
    interp_dir = float(np.interp(4862, fx2, fy2))
    n_verdict = dict(
        model_ref=20078.5, model_fills=4862,
        rnd_at_4862_interp=round(interp, 1),
        rnd_dir_at_4862_interp=round(interp_dir, 1),
        conclusion=("随机流在同笔数4862处插值仍高于model" if interp > 20078.5
                    else "随机流优势主要是笔数N(同笔数处model更高)"),
        note="线性插值; rnd_dir=model方向随机时刻表")
    print(f"[N-verdict] rnd@4862 = {interp:.0f} vs model 20078 -> {n_verdict['conclusion']}", flush=True)

    out = dict(scan=scan, n_verdict=n_verdict,
               meta=dict(rates=RATES, seeds=SEEDS, engine="chandelier3.0ATR serial",
                         generated=time.strftime("%Y-%m-%d %H:%M:%S")))
    json.dump(out, open(os.path.join(RES, "diag2_scan.json"), "w"), indent=1)
    sc.say(f"v16diag2 DONE {(time.time()-t00)/60:.1f}min: rnd@4862={interp:.0f} vs model 20078")
    print(f"DONE {(time.time()-t00)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
