#!/usr/bin/env python3
"""v16_gate.py — 路线A: 波动率与动能门控 (Regime Gate) 叠加在路线C赢家 (Keltner突破) 上.

用户路线A原文: 放弃方向预测, 预测未来90分钟 RV_90m > Threshold 或 ER > 0.4;
              低波/震荡期闭嘴, 只在挤压突破/流动性扩张前夕开门, 交给 Chandelier 跑满90分钟.
路线C已证明: Keltner突破+吊灯 = $72287/16230笔/笔均$4.45 (吞吐满载).
本实验: gate 能否砍掉低质量突破, 提笔均/总量/Sharpe?

臂:
  base     : kelt 无 gate (基线)
  slip2/3  : 成本 x2/x3 敏感性
  sqz_rule : TTM squeeze 规则 gate (BB20,2 内含于 KC20,1.5 = 挤压; 挤压期间或解除后12根m5内的信号才交易) 零训练
  atr_rule : ATR 水平 gate (当前ATR288 >= 其1440根m5均值 = 波动扩张状态才交易) 零训练
  ml_mfe   : LGB P(方向性MFE90/ATR > 1.0) > 0.5   [突破延续性分类]
  ml_mfe60 : 同上阈值 0.6 (更严)
  ml_er    : LGB P(ER90 > 0.4) > 0.5              [用户标签: 动量效率比]
  ml_rv    : LGB P(RV90/ATR > 训练窗70分位) > 0.5  [用户标签: 实现波动率扩张]
训练协议: 仅 kelt 信号bar 为样本, walkforward 54折, train_cap 450d, champ LGB 超参 (与历线一致).
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


def engine(m1_ohl, sig_t, tm, atr, cost):
    m1_t, m1_o, m1_h, m1_l = m1_ohl
    p = GEOM
    return simulate_exit(m1_t, m1_o, m1_h, m1_l, sig_t, tm, atr, cost,
                         p["mode"], TPM, SLM, SLF, HZN, ETOL,
                         p["fixed_tp"], p["fixed_sl"], p["chan_atr_mult"],
                         p["n_bracket"], p["bracket_tp_mult"],
                         p["time_hold"], p["tight_tp"], p["tight_sl"], p["scale_trail"], 1)


def stats(month_pnl, fills, trades_df):
    s = pd.Series(month_pnl)
    by = {}
    for mk, v in month_pnl.items():
        y = int(mk[:4]); by[y] = by.get(y, 0.0) + v
    tot = float(s.sum())
    cum = s.cumsum()
    dd = (cum - cum.cummax()).min()
    sharpe = float(s.mean() / s.std() * np.sqrt(12)) if len(s) > 6 and s.std() > 0 else None
    long_pnl = short_pnl = 0.0
    if trades_df is not None and len(trades_df):
        long_pnl = float(trades_df[trades_df["dir"] == 1]["pnl"].sum())
        short_pnl = float(trades_df[trades_df["dir"] == -1]["pnl"].sum())
    return dict(total=round(tot, 1), trades=fills,
                avg_per_trade=round(tot / max(fills, 1), 3),
                by_year={k: round(v, 1) for k, v in sorted(by.items())},
                share2026_pct=round(100.0 * by.get(2026, 0.0) / max(tot, 1e-9), 1),
                maxDD=round(float(dd), 1), sharpe_m=round(sharpe, 2) if sharpe else None,
                long_pnl=round(long_pnl, 1), short_pnl=round(short_pnl, 1))


def main():
    t00 = time.time()
    m5, F, cost, atr, (m1_t, m1_o, m1_h, m1_l) = sc._prep()
    sig_t = (m5.index.astype("datetime64[s]").astype("int64") + 300).to_numpy()
    folds = sc.month_folds(m5.index, "2022-08", "2026-07")
    TRAIN_CAP = pd.Timedelta(days=CFG.get("train_cap_days", 450))
    c, h, l = m5["close"], m5["high"], m5["low"]
    cl = c.to_numpy(); hi = h.to_numpy(); lo = l.to_numpy()
    atr_arr = atr.to_numpy() if hasattr(atr, "to_numpy") else np.asarray(atr)
    cost_arr = cost.to_numpy() if hasattr(cost, "to_numpy") else np.asarray(cost)

    # ---- kelt 信号 (与 v16_breakout 完全一致) ----
    ema = c.ewm(span=20, adjust=False).mean()
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    atr14 = tr.rolling(14).mean()
    up = (c > (ema + 2.0 * atr14)).to_numpy()
    dn = (c < (ema - 2.0 * atr14)).to_numpy()
    sig = np.zeros(len(m5), dtype=np.int8)
    sig[up] = 1; sig[dn] = -1
    n_sig = int((sig != 0).sum())
    print(f"[sig] kelt signals {n_sig} / {len(m5)} bars ({100*n_sig/len(m5):.1f}%)", flush=True)

    # ---- 标签 (未来 18 根 m5 = 90min, 不含当根) ----
    W = 18
    fut_hi = pd.Series(hi).rolling(W).max().shift(-W).to_numpy()
    fut_lo = pd.Series(lo).rolling(W).min().shift(-W).to_numpy()
    fut_cl = pd.Series(cl).shift(-W).to_numpy()
    dd = np.abs(np.diff(cl, prepend=cl[0]))
    fut_path = pd.Series(dd).rolling(W).sum().shift(-W).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        mfe_sig = np.where(sig == 1, (fut_hi - cl), (cl - fut_lo)) / atr_arr
        er90 = np.abs(fut_cl - cl) / np.maximum(fut_path, 1e-9)
        rv90 = (fut_hi - fut_lo) / atr_arr
    y_mfe = (mfe_sig > 1.0).astype(float)
    y_er = (er90 > 0.4).astype(float)
    ok_lab = np.isfinite(mfe_sig) & np.isfinite(er90) & np.isfinite(rv90)
    y_mfe[~ok_lab] = np.nan; y_er[~ok_lab] = np.nan
    print(f"[labels] mfe>1 rate {np.nanmean(y_mfe[sig!=0]):.3f} er>0.4 rate {np.nanmean(y_er[sig!=0]):.3f}", flush=True)

    # ---- 规则 gates ----
    mid = c.rolling(20).mean(); sd = c.rolling(20).std()
    bb_hi, bb_lo = mid + 2 * sd, mid - 2 * sd
    kc_hi, kc_lo = ema + 1.5 * atr14, ema - 1.5 * atr14
    sqz_on = ((bb_hi < kc_hi) & (bb_lo > kc_lo)).to_numpy()
    recent_on = pd.Series(sqz_on.astype(int)).rolling(12).max().fillna(0).astype(bool).to_numpy()
    gate_sqz = recent_on
    atr_long = pd.Series(atr_arr).rolling(1440).mean().to_numpy()
    gate_atr = atr_arr >= atr_long
    print(f"[gates] sqz coverage {gate_sqz[sig!=0].mean():.2f} atr coverage {gate_atr[sig!=0].mean():.2f}", flush=True)

    # ---- ML walkforward: 3 标签, 样本 = 训练窗内 kelt 信号 bar ----
    X_all = F[CFG["features"]]
    params = dict(CFG["lgb"]); params["num_threads"] = int(os.environ.get("ML_THREADS", "26"))
    p_mfe = np.full(len(m5), np.nan); p_er = np.full(len(m5), np.nan); p_rv = np.full(len(m5), np.nan)
    for oos_s, oos_e in folds:
        tr_end = oos_s - pd.Timedelta(minutes=120)
        tr_start = oos_s - TRAIN_CAP
        tr_mask = (m5.index >= tr_start) & (m5.index < tr_end)
        oos_mask = (m5.index >= oos_s) & (m5.index <= oos_e)
        tr_sig = np.asarray(tr_mask) & (sig != 0)
        oos_sig = np.asarray(oos_mask) & (sig != 0)
        Xtr = X_all[tr_sig].to_numpy()
        if len(Xtr) < 800:
            continue
        Xoos = X_all[oos_sig].to_numpy()
        oos_pos = np.where(oos_sig)[0]
        if len(oos_pos) == 0:
            continue
        for ylab, parr in ((y_mfe, p_mfe), (y_er, p_er), (rv90, None)):
            if parr is None:
                # rv 标签: 训练窗70分位阈值 (因果)
                rv_tr = rv90[tr_sig]
                rv_tr = rv_tr[np.isfinite(rv_tr)]
                th = np.percentile(rv_tr, 70) if len(rv_tr) > 100 else 2.0
                yv = (rv90 > th).astype(float)
                parr = p_rv
            else:
                yv = ylab
            ytr = yv[tr_sig]
            ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr)
            if ok.sum() < 500:
                continue
            bst = lgb.train(params, lgb.Dataset(Xtr[ok], ytr[ok]), num_boost_round=CFG.get("rounds", 200))
            ok_o = np.isfinite(Xoos).all(axis=1)
            if ok_o.sum() > 0:
                parr[oos_pos[ok_o]] = bst.predict(Xoos[ok_o])
    fin_mfe = np.isfinite(p_mfe); fin_er = np.isfinite(p_er); fin_rv = np.isfinite(p_rv)
    print(f"[ml] coverage on signals: mfe {fin_mfe[sig!=0].mean():.2f} er {fin_er[sig!=0].mean():.2f} rv {fin_rv[sig!=0].mean():.2f}", flush=True)
    # 标签 OOS AUC (在信号 bar 上, 二月聚合粗算)
    from sklearn.metrics import roc_auc_score
    m = (sig != 0) & fin_mfe & np.isfinite(y_mfe)
    auc_mfe = roc_auc_score(y_mfe[m], p_mfe[m]) if m.sum() > 500 else None
    m = (sig != 0) & fin_er & np.isfinite(y_er)
    auc_er = roc_auc_score(y_er[m], p_er[m]) if m.sum() > 500 else None
    print(f"[ml] OOS AUC (insample-ish, signal bars): mfe {auc_mfe} er {auc_er}", flush=True)
    sc.say(f"v16gate ML done: auc_mfe {auc_mfe} auc_er {auc_er} ({(time.time()-t00)/60:.1f}min)")

    # ---- 各臂结算 ----
    gates = {
        "base":     np.ones(len(m5), dtype=bool),
        "sqz_rule": gate_sqz,
        "atr_rule": gate_atr,
        "ml_mfe":   np.where(fin_mfe, p_mfe, 0.0) > 0.5,
        "ml_mfe60": np.where(fin_mfe, p_mfe, 0.0) > 0.6,
        "ml_er":    np.where(fin_er, p_er, 0.0) > 0.5,
        "ml_rv":    np.where(fin_rv, p_rv, 0.0) > 0.5,
    }
    m1_ohl = (m1_t, m1_o, m1_h, m1_l)
    out = {}
    for name, g in gates.items():
        tm_full = np.where(g, sig, 0).astype(np.int8)
        month_pnl = {}; fills = 0; rows = []
        for oos_s, oos_e in folds:
            oos_mask = (m5.index >= oos_s) & (m5.index <= oos_e)
            tm = tm_full[np.asarray(oos_mask)]
            if (tm != 0).sum() == 0:
                continue
            pnl, wins, filled = engine(m1_ohl, sig_t[oos_mask], tm, atr_arr[np.asarray(oos_mask)],
                                       cost_arr[np.asarray(oos_mask)])
            fills += filled
            mk = str(oos_s.date())[:7]
            month_pnl[mk] = float(pnl[tm != 0].sum())
            oos_pos = np.where(oos_mask)[0]
            for pos in np.where((tm != 0) & (pnl != 0.0))[0]:
                rows.append(dict(dir=int(tm[pos]), pnl=float(pnl[pos])))
        out[name] = stats(month_pnl, fills, pd.DataFrame(rows))
        a = out[name]
        print(f"[{name}] total={a['total']} n={a['trades']} avg={a['avg_per_trade']} share26={a['share2026_pct']}% "
              f"maxDD={a['maxDD']} sharpe={a['sharpe_m']}", flush=True)
    # 成本敏感性 (base 信号)
    for cx, nm in ((2, "slip2"), (3, "slip3")):
        month_pnl = {}; fills = 0
        for oos_s, oos_e in folds:
            oos_mask = (m5.index >= oos_s) & (m5.index <= oos_e)
            tm = sig[np.asarray(oos_mask)]
            if (tm != 0).sum() == 0:
                continue
            pnl, wins, filled = engine(m1_ohl, sig_t[oos_mask], tm, atr_arr[np.asarray(oos_mask)],
                                       cost_arr[np.asarray(oos_mask)] * cx)
            fills += filled
            month_pnl[str(oos_s.date())[:7]] = float(pnl[tm != 0].sum())
        out[nm] = stats(month_pnl, fills, None)
        print(f"[{nm}] total={out[nm]['total']} n={out[nm]['trades']}", flush=True)

    res = dict(arms=out,
               labels=dict(sig_bars=n_sig,
                           mfe_gt1_rate=round(float(np.nanmean(y_mfe[sig != 0])), 3),
                           er_gt04_rate=round(float(np.nanmean(y_er[sig != 0])), 3),
                           auc_mfe=round(float(auc_mfe), 3) if auc_mfe else None,
                           auc_er=round(float(auc_er), 3) if auc_er else None),
               meta=dict(signal="keltner(EMA20±2ATR14) breakout", engine="chandelier3.0ATR serial hold90",
                         folds=len(folds), generated=time.strftime("%Y-%m-%d %H:%M:%S")))
    json.dump(res, open(os.path.join(RES, "gate.json"), "w"), indent=1)
    sc.say(f"v16gate DONE {(time.time()-t00)/60:.1f}min: " +
           " | ".join(f"{k}:{v['total']}" for k, v in out.items()))
    print(f"DONE {(time.time()-t00)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
