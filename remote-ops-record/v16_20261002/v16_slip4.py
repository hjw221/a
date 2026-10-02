#!/usr/bin/env python3
"""v16_slip4.py — 真实引擎补齐: model 流 + 随机多seed方差 + 动量延续衰减.

slip3 结论: 正确结算下 kelt_fixed_tp2.5_sl0.4 +$2599(27K笔) 是最好组合但薄;
其余全灭或2026漂移. 本脚本补:
  1. model 流(champ q_gate 信号) x {chan_cons2.5_h120, fixed_tp2.5_sl0.4, time_h120, time_h90}
  2. rnd 5 seeds x {fixed_tp2.5_sl0.4, time_h120} -> 单seed +$6604 是噪声还是稳定
  3. kelt x time_h{60,120,240} -> 突破动量延续半衰期
全部 sim_true 正确结算(触发即出场, 触发时刻解锁).
"""
import os, json, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from numba import njit
import stage_c_loop as sc
from v16_slip3 import sim_true, MODE_CHAN_CONS, MODE_FIXED, MODE_TIME

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results_v16")
CHAMP = json.load(open(os.path.join(BASE, "results_research", "champion.json")))
CFG = CHAMP["cfg"]


def main():
    t00 = time.time()
    m5, F, cost, atr, (m1_t, m1_o, m1_h, m1_l) = sc._prep()
    sig_t = (m5.index.astype("datetime64[s]").astype("int64") + 300).to_numpy()
    folds = sc.month_folds(m5.index, "2022-08", "2026-07")
    TRAIN_CAP = pd.Timedelta(days=CFG.get("train_cap_days", 450))
    hours_f = CFG.get("hours_filter")
    hour_arr = m5.index.hour.to_numpy()
    params = dict(CFG["lgb"]); params["num_threads"] = int(os.environ.get("ML_THREADS", "26"))
    q_gate = CFG.get("q_gate", 0.0)
    X = F[CFG["features"]]

    # ---- model 流: 复用 ctrl2 训练循环, 收集每折 champ 信号 ----
    sc.say("v16slip4 phase1: model stream LGB")
    model_sig = np.zeros(len(m5), dtype=np.int8)
    for oos_s, oos_e in folds:
        tr_end = oos_s - pd.Timedelta(minutes=120)
        tr_start = oos_s - TRAIN_CAP
        tr_mask = np.asarray((m5.index >= tr_start) & (m5.index < tr_end))
        oos_mask = np.asarray((m5.index >= oos_s) & (m5.index <= oos_e))
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
            trade[~np.isin(hour_arr[oos_mask], hours_f)] = 0
        oos_pos = np.where(oos_mask)[0]
        model_sig[oos_pos] = trade
    print(f"[model] signals {int((model_sig != 0).sum())}", flush=True)

    # ---- kelt 信号 ----
    c, h, l = m5["close"], m5["high"], m5["low"]
    ema = c.ewm(span=20, adjust=False).mean()
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    atr14 = tr.rolling(14).mean()
    kelt = np.zeros(len(m5), dtype=np.int8)
    kelt[(c > (ema + 2.0 * atr14)).to_numpy()] = 1
    kelt[(c < (ema - 2.0 * atr14)).to_numpy()] = -1

    def run(tm_full, mode, cm, tpm, slm, horizon, dm=0.0):
        month_pnl = {}; fills = 0; neg = 0
        for oos_s, oos_e in folds:
            oos_mask = np.asarray((m5.index >= oos_s) & (m5.index <= oos_e))
            tm = tm_full[oos_mask]
            if (tm != 0).sum() == 0:
                continue
            pnl, wins, filled = sim_true(m1_t, m1_o, m1_h, m1_l, sig_t[oos_mask], tm,
                                          atr[oos_mask], cost[oos_mask], mode, cm, tpm, slm,
                                          0.48, horizon, 10, dm, 1)
            fills += filled
            v = float(pnl[tm != 0].sum())
            month_pnl[str(oos_s.date())[:7]] = v
            if v < 0:
                neg += 1
        s = pd.Series(month_pnl)
        by = {}
        for mk, v in month_pnl.items():
            y = int(mk[:4]); by[y] = by.get(y, 0.0) + v
        return dict(total=round(float(s.sum()), 1), trades=fills,
                    avg_per_trade=round(float(s.sum()) / max(fills, 1), 3),
                    neg_months=neg, by_year={k: round(v, 1) for k, v in sorted(by.items())})

    out = {}
    # 1) model 流
    for nm, (mode, cm, tpm, slm, hz) in {
        "model_chancons2.5_h120": (MODE_CHAN_CONS, 2.5, 0, 0, 120),
        "model_fixed2.5_0.4": (MODE_FIXED, 0, 2.5, 0.4, 90),
        "model_time_h120": (MODE_TIME, 0, 0, 0, 120),
        "model_time_h90": (MODE_TIME, 0, 0, 0, 90),
    }.items():
        out[nm] = run(model_sig, mode, cm, tpm, slm, hz)
        print(f"[{nm}] {out[nm]}", flush=True)
    # 2) rnd 5 seeds
    ok_feat = np.isfinite(X.to_numpy()).all(axis=1)
    m5_month = m5.index.strftime("%Y-%m")
    for tag, (mode, cm, tpm, slm, hz) in {
        "fixed2.5_0.4": (MODE_FIXED, 0, 2.5, 0.4, 90),
        "time_h120": (MODE_TIME, 0, 0, 0, 120),
    }.items():
        tots = []
        for sd in (2000, 2001, 2002, 2003, 2004):
            rng = np.random.default_rng(sd)
            rnd = np.zeros(len(m5), dtype=np.int8)
            for mk in pd.unique(m5_month):
                m = (m5_month == mk) & ok_feat
                pos = np.where(m)[0]
                if len(pos):
                    rnd[pos] = rng.choice([1, -1], size=len(pos))
            r = run(rnd, mode, cm, tpm, slm, hz)
            tots.append(r["total"])
            out[f"rnd{sd}_{tag}"] = r
        arr = np.array(tots)
        out[f"rnd_multiseed_{tag}"] = dict(mean=round(float(arr.mean()), 1),
                                           std=round(float(arr.std()), 1),
                                           totals=[round(x, 1) for x in tots])
        print(f"[rnd_multiseed_{tag}] mean={arr.mean():.0f} std={arr.std():.0f} {tots}", flush=True)
    # 3) kelt 动量延续衰减
    for hz in (60, 120, 240):
        out[f"kelt_time_h{hz}"] = run(kelt, MODE_TIME, 0, 0, 0, hz)
        print(f"[kelt_time_h{hz}] {out[f'kelt_time_h{hz}']}", flush=True)

    json.dump(out, open(os.path.join(RES, "slip4.json"), "w"), indent=1)
    sc.say(f"v16slip4 DONE {(time.time()-t00)/60:.1f}min")
    print(f"DONE {(time.time()-t00)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
