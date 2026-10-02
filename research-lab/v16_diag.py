#!/usr/bin/env python3
"""v16_diag.py — 路线B: 解剖 "随机时刻表优于模型时刻表 (+$10K)" 的物理本质.

用户三问 -> 实验设计:
  B1 信号聚集度: model 流 fills/n_sig (串行skip率), 相邻信号间隔分布 vs random
  B2 N效应配平:  random 每折抽样数 = model 该折实际fills -> 总笔数对齐~4862, PnL谁高? (20 seeds)
  B3 追高陷阱:   四流逐笔 MFE90/MAE90/方向命中/过去90min位置分位 -> 检验ML是否在趋势尾部进场

四流 (时刻 x 方向 单变量剥离):
  model    = model 时刻 + model 方向                      [基准 $20078/4862]
  rnd      = random 时刻 + 随机方向                        [ctrl2 复现流 $30051/8517]
  rnd_dir  = random 时刻 + model 方向 (pv-0.5 符号)        [剥离"方向输出"的价值]
  mdl_rdir = model 时刻 + 随机方向                          [剥离"时刻选择"的价值]

口径: chandelier 3.0ATR 出场, 串行一次一仓, 与 v15 冠军完全同引擎同参数.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from numba import njit
import stage_c_loop as sc
from v15_exit import simulate_exit, EXIT_CHANDELIER

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results_v16")
os.makedirs(RES, exist_ok=True)
CHAMP = json.load(open(os.path.join(BASE, "results_research", "champion.json")))
CFG = CHAMP["cfg"]
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


@njit(cache=True)
def entry_stats(m1_t, m1_o, m1_h, m1_l, sig_t, dirs, atr_arr):
    """每笔: [pos_past, mfe90/atr, mae90/atr, dir_hit90, chase_edge]"""
    n = len(sig_t); nm1 = len(m1_t)
    out = np.full((n, 5), np.nan)
    for i in range(n):
        t0 = sig_t[i]
        lo, hi = 0, nm1
        while lo < hi:
            mid = (lo + hi) // 2
            if m1_t[mid] < t0:
                lo = mid + 1
            else:
                hi = mid
        e = lo
        if e >= nm1 or m1_t[e] > t0 + 10:
            continue
        entry = m1_o[e]
        # future 90min window [e, e+90 bars) approx by time
        j_end = e
        while j_end < nm1 and m1_t[j_end] <= t0 + 5400:
            j_end += 1
        if j_end <= e + 2:
            continue
        fhi = -1e18; flo = 1e18
        for k in range(e, j_end):
            if m1_h[k] > fhi: fhi = m1_h[k]
            if m1_l[k] < flo: flo = m1_l[k]
        d = dirs[i]
        if d == 1:
            mfe = fhi - entry; mae = entry - flo
        else:
            mfe = entry - flo; mae = fhi - entry
        last = j_end - 1
        end_px = m1_o[last]
        hit = 1.0 if (end_px - entry) * d > 0 else 0.0
        # past 90min range position of entry price
        j0 = e
        while j0 > 0 and m1_t[j0 - 1] >= t0 - 5400:
            j0 -= 1
        phi = -1e18; plo = 1e18
        for k in range(j0, e):
            if m1_h[k] > phi: phi = m1_h[k]
            if m1_l[k] < plo: plo = m1_l[k]
        rng_ = phi - plo
        pos_past = (entry - plo) / rng_ if rng_ > 1e-12 else 0.5
        chase = pos_past if d == 1 else (1.0 - pos_past)
        a = atr_arr[i]
        out[i, 0] = pos_past
        out[i, 1] = mfe / a if a > 0 else np.nan
        out[i, 2] = mae / a if a > 0 else np.nan
        out[i, 3] = hit
        out[i, 4] = chase
    return out


def agg_stream(trades):
    """trades: list of dict(month,gidx,dir,pnl,atr,sig_t)"""
    if not trades:
        return dict(n=0)
    s = pd.Series([t["pnl"] for t in trades])
    by = {}
    for t in trades:
        y = int(str(t["month"])[:4]); by[y] = by.get(y, 0.0) + t["pnl"]
    return dict(n=len(trades), total=round(float(s.sum()), 1),
                avg=round(float(s.mean()), 3), by_year={k: round(v, 1) for k, v in sorted(by.items())})


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

    # ---------- Phase 1: model 流 (唯一需要 LGB 训练的流) ----------
    sc.say("v16diag phase1 start: model stream 54 folds LGB")
    model_trades = []
    fold_ctx = []   # 每折: oos位置数组, cand, pv, model_tm, fills_m, n_sig
    n_sig_tot = fills_model = 0
    gaps_model = []
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
        trade = np.zeros(len(Xoos), dtype=np.int8)
        if q_gate > 0:
            fin = np.isfinite(pv)
            if fin.sum() > 50:
                hi, lo = np.quantile(pv[fin], [1 - q_gate, q_gate])
                trade[pv >= hi] = 1
                trade[pv <= lo] = -1
        if hours_f:
            hm = hour_arr[oos_mask]
            trade[~np.isin(hm, hours_f)] = 0
        pnl, wins, filled = engine((m1_t, m1_o, m1_h, m1_l), sig_t[oos_mask], trade,
                                   atr[oos_mask], cost[oos_mask])
        n_sig = int((trade != 0).sum()); n_sig_tot += n_sig; fills_model += filled
        sig_pos = np.where(trade != 0)[0]
        if len(sig_pos) > 1:
            gaps_model.extend(np.diff(sig_pos).tolist())
        oos_pos = np.where(oos_mask)[0]
        for pos in np.where((trade != 0) & (pnl != 0.0))[0]:
            model_trades.append(dict(month=str(oos_s.date())[:7], gidx=int(oos_pos[pos]),
                                     dir=int(trade[pos]), pnl=float(pnl[pos])))
        # candidates for random streams (hours 内, 特征可用)
        if hours_f:
            cand = np.where(np.isin(hour_arr[oos_mask], hours_f) & ok_o)[0]
        else:
            cand = np.where(ok_o)[0]
        fold_ctx.append(dict(month=str(oos_s.date())[:7], oos_pos=oos_pos, cand=cand,
                             pv=pv, model_tm=trade, fills_m=int(filled), n_sig=n_sig))
    p1 = agg_stream(model_trades)
    p1.update(n_sig=n_sig_tot, fills=fills_model,
              fill_rate=round(fills_model / max(n_sig_tot, 1), 3),
              gap_med_m5=int(np.median(gaps_model)) if gaps_model else None,
              gap_p90_m5=int(np.percentile(gaps_model, 90)) if gaps_model else None)
    print(f"[P1] model total={p1['total']} fills={fills_model} n_sig={n_sig_tot} "
          f"fill_rate={p1['fill_rate']} gap_med={p1['gap_med_m5']}bars", flush=True)
    sc.say(f"v16diag P1 model {p1['total']:.0f} fills {fills_model} sig {n_sig_tot} fillrate {p1['fill_rate']}")

    def make_rnd_stream(rng, fills_as_k, use_model_dir, model_positions):
        """构造一条随机流并跑引擎. fills_as_k=True 用 fills_m 作抽样数(配平), 否则用 n_sig(复现ctrl2).
        use_model_dir=True 方向取 pv-0.5 符号; model_positions=True 用 model 时刻(方向随机)."""
        trades = []; tot = 0; fl = 0; gaps = []
        for fc in fold_ctx:
            if model_positions:
                tm = fc["model_tm"].copy()
                K = int((tm != 0).sum())
                sig_pos = np.where(tm != 0)[0]
                dirs = rng.choice([1, -1], size=K)
                tm[sig_pos] = dirs
                k_used = K
            else:
                K = fc["fills_m"] if fills_as_k else fc["n_sig"]
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
                k_used = len(pick)
                if len(pick) > 1:
                    gaps.extend(np.diff(np.sort(pick)).tolist())
            pnl, wins, filled = engine((m1_t, m1_o, m1_h, m1_l),
                                       sig_t[fc["oos_pos"]], tm,
                                       atr[fc["oos_pos"]], cost[fc["oos_pos"]])
            tot += float(pnl[tm != 0].sum()); fl += filled
            for pos in np.where((tm != 0) & (pnl != 0.0))[0]:
                trades.append(dict(month=fc["month"], gidx=int(fc["oos_pos"][pos]),
                                   dir=int(tm[pos]), pnl=float(pnl[pos])))
        return trades, tot, fl, gaps

    # ---------- Phase 2: 非配平 (seed 777, 复现 ctrl2 的 rnd 流做 sanity) ----------
    sc.say("v16diag phase2: unmatched streams seed777")
    rnd_tr, rnd_tot, rnd_fl, gaps_rnd = make_rnd_stream(np.random.default_rng(777), False, False, False)
    rd_tr, rd_tot, rd_fl, _ = make_rnd_stream(np.random.default_rng(777), False, True, False)
    mr_tr, mr_tot, mr_fl, _ = make_rnd_stream(np.random.default_rng(777), False, False, True)
    p2 = dict(
        rnd=dict(total=round(rnd_tot, 1), fills=rnd_fl, **{k: v for k, v in agg_stream(rnd_tr).items() if k in ("n", "avg", "by_year")}),
        rnd_dir=dict(total=round(rd_tot, 1), fills=rd_fl, **{k: v for k, v in agg_stream(rd_tr).items() if k in ("n", "avg", "by_year")}),
        mdl_rdir=dict(total=round(mr_tot, 1), fills=mr_fl, **{k: v for k, v in agg_stream(mr_tr).items() if k in ("n", "avg", "by_year")}),
        sanity=dict(ctrl2_random_ref=30051.5, ctrl2_fills_ref=8517,
                    rnd_dev_pct=round(100 * (rnd_tot - 30051.5) / 30051.5, 3),
                    rnd_fills_dev= rnd_fl - 8517))
    print(f"[P2] rnd={rnd_tot:.0f}/{rnd_fl} (ctrl2 ref 30051/8517 dev {p2['sanity']['rnd_dev_pct']}%) "
          f"rnd_dir={rd_tot:.0f}/{rd_fl} mdl_rdir={mr_tot:.0f}/{mr_fl}", flush=True)
    sc.say(f"v16diag P2 rnd {rnd_tot:.0f} rnd_dir {rd_tot:.0f} mdl_rdir {mr_tot:.0f} (unmatched)")

    # ---------- Phase 3: 配平 20 seeds ----------
    sc.say("v16diag phase3: fill-matched 20 seeds x3 streams")
    SEEDS = list(range(1000, 1020))
    matched = dict(rnd=[], rnd_dir=[], mdl_rdir=[])
    matched_trades_sample = {}
    for s_i, sd in enumerate(SEEDS):
        for name, umd, mp in (("rnd", False, False), ("rnd_dir", True, False), ("mdl_rdir", False, True)):
            tr, tot, fl, _ = make_rnd_stream(np.random.default_rng(sd), True, umd, mp)
            matched[name].append(dict(seed=sd, total=round(tot, 1), fills=fl))
            if s_i == 0:
                matched_trades_sample[name] = tr
    p3 = {}
    for name, rows in matched.items():
        t_ = np.array([r["total"] for r in rows]); f_ = np.array([r["fills"] for r in rows])
        p3[name] = dict(mean=round(float(t_.mean()), 1), std=round(float(t_.std()), 1),
                        p5=round(float(np.percentile(t_, 5)), 1), p95=round(float(np.percentile(t_, 95)), 1),
                        fills_mean=round(float(f_.mean()), 0),
                        beats_model=round(float((t_ > p1["total"]).mean()), 3))
        print(f"[P3] {name}: mean={p3[name]['mean']} std={p3[name]['std']} fills~{p3[name]['fills_mean']} "
              f"beat_model={p3[name]['beats_model']}", flush=True)
    sc.say(f"v16diag P3 matched: rnd {p3['rnd']['mean']:.0f} rnd_dir {p3['rnd_dir']['mean']:.0f} "
           f"mdl_rdir {p3['mdl_rdir']['mean']:.0f} vs model {p1['total']:.0f}")

    # ---------- Phase 4: 进场质量 (追高陷阱) ----------
    def q_stats(trades, label):
        if not trades:
            return dict(label=label, n=0)
        st = np.array([[t["gidx"], t["dir"], t["pnl"]] for t in trades], dtype=np.float64)
        gidx = st[:, 0].astype(np.int64); dirs = st[:, 1].astype(np.int8)
        sig = sig_t[gidx]; a = atr[gidx]
        M = entry_stats(m1_t, m1_o, m1_h, m1_l, sig, dirs, a)
        okm = np.isfinite(M[:, 0])
        chase = M[okm, 4]
        out = dict(label=label, n=int(okm.sum()),
                   mfe_med=round(float(np.nanmedian(M[okm, 1])), 2),
                   mae_med=round(float(np.nanmedian(M[okm, 2])), 2),
                   mfe_over_mae=round(float(np.nanmedian(M[okm, 1]) / np.nanmedian(M[okm, 2])), 2),
                   dir_hit90=round(float(np.nanmean(M[okm, 3])), 3),
                   chase_edge_med=round(float(np.median(chase)), 3),
                   chase_pct_gt80=round(float((chase > 0.8).mean()), 3))
        for d, nm in ((1, "long"), (-1, "short")):
            m_ = okm & (dirs == d)
            if m_.sum() > 30:
                out[nm] = dict(n=int(m_.sum()),
                               pnl=round(float(st[m_, 2].sum()), 1),
                               dir_hit=round(float(np.nanmean(M[m_, 3])), 3),
                               chase_med=round(float(np.median(M[m_, 4])), 3))
        return out

    p4 = dict(
        model=q_stats(model_trades, "model"),
        rnd=q_stats(rnd_tr, "rnd@777"),
        rnd_dir=q_stats(rd_tr, "rnd_dir@777"),
        mdl_rdir=q_stats(mr_tr, "mdl_rdir@777"))
    for k, v in p4.items():
        print(f"[P4] {k}: mfe={v.get('mfe_med')} mae={v.get('mae_med')} hit={v.get('dir_hit90')} "
              f"chase>0.8={v.get('chase_pct_gt80')} long={v.get('long', {}).get('dir_hit')} short={v.get('short', {}).get('dir_hit')}", flush=True)

    # ---------- 判读 ----------
    verdict = dict(
        b2_n_effect=dict(
            model=p1["total"], rnd_matched_mean=p3["rnd"]["mean"],
            rnd_beats_model_rate=p3["rnd"]["beats_model"],
            conclusion=("N配平后随机仍胜" if p3["rnd"]["mean"] > p1["total"] else "N配平后模型反超(笔数是主因)")),
        dir_alpha=dict(
            rnd=p3["rnd"]["mean"], rnd_dir=p3["rnd_dir"]["mean"],
            model_dir_delta=round(p3["rnd_dir"]["mean"] - p3["rnd"]["mean"], 1),
            conclusion=("model方向为正贡献" if p3["rnd_dir"]["mean"] > p3["rnd"]["mean"] else "model方向为负贡献")),
        timing_alpha=dict(
            rnd=p3["rnd"]["mean"], mdl_rdir=p3["mdl_rdir"]["mean"],
            model_timing_delta=round(p3["mdl_rdir"]["mean"] - p3["rnd"]["mean"], 1),
            conclusion=("model时刻为正贡献" if p3["mdl_rdir"]["mean"] > p3["rnd"]["mean"] else "model时刻为负贡献")),
        b3_chase=dict(
            model_chase_pct=p4["model"].get("chase_pct_gt80"),
            rnd_chase_pct=p4["rnd"].get("chase_pct_gt80"),
            model_dir_hit=p4["model"].get("dir_hit90"),
            rnd_dir_hit=p4["rnd"].get("dir_hit90")))
    out = dict(phase1_model=p1, phase2_unmatched=p2, phase3_matched=p3,
               phase4_entry_quality=p4, verdict=verdict,
               meta=dict(folds=len(fold_ctx), seeds=20, engine="chandelier3.0ATR serial",
                         generated=time.strftime("%Y-%m-%d %H:%M:%S")))
    json.dump(out, open(os.path.join(RES, "diag.json"), "w"), indent=1)
    # 逐笔留档 (model + 777 三流)
    rows = []
    for tr, tag in ((model_trades, "model"), (rnd_tr, "rnd"), (rd_tr, "rnd_dir"), (mr_tr, "mdl_rdir")):
        for t in tr:
            rows.append(dict(stream=tag, **t))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "diag_trades.csv"), index=False)
    sc.say(f"v16diag DONE in {(time.time()-t00)/60:.1f}min -> results_v16/diag.json")
    print(f"DONE in {(time.time()-t00)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
