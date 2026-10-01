#!/usr/bin/env python3
"""M1剥头皮实验分析 — m1sc(固定$1.2) vs m1sc_ad(自适应), 只报真实数字。"""
import json
import numpy as np
import pandas as pd

BASE = "/home/z/my-project/download/xauusd_ml_v2/results"
OUT = f"{BASE}/analysis_m1.json"


def load(fs):
    per = json.load(open(f"{BASE}/per_fold_{fs}.json"))
    tr = pd.read_csv(f"{BASE}/trades_{fs}.csv")
    tr["exit_time"] = pd.to_datetime(tr["exit_time"])
    tr["signal_time"] = pd.to_datetime(tr["signal_time"])
    return per, tr


def stats(tr, rng_seed=42):
    pnl = tr["pnl"].to_numpy()
    n = len(tr)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    # bootstrap CI (trade-level, 10k)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n, size=(10000, n))
    means = pnl[idx].mean(axis=1)
    ci = (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))
    # 最长连亏
    mcl = cur = 0
    for p in pnl:
        cur = cur + 1 if p <= 0 else 0
        mcl = max(mcl, cur)
    # 月度
    mo = tr.groupby(tr["exit_time"].dt.to_period("M"))["pnl"].agg(["sum", "count"])
    eq = np.concatenate([[0.0], np.cumsum(pnl)])
    mdd = float((np.maximum.accumulate(eq) - eq).max())
    daily = pd.Series(pnl, index=tr["exit_time"].values).resample("D").sum()
    daily = daily[daily.index.dayofweek < 5]
    sharpe = float(np.sqrt(252) * daily.mean() / daily.std()) if daily.std() > 0 else np.nan
    lo, shrt = tr[tr.dir == "long"], tr[tr.dir == "short"]
    return dict(
        n=n, wr=float((pnl > 0).mean()), pnl=float(pnl.sum()), mean=float(pnl.mean()),
        plr=float(wins.mean() / abs(losses.mean())),
        avg_win=float(wins.mean()), avg_loss=float(losses.mean()),
        pf=float(wins.sum() / abs(losses.sum())), mdd=mdd, sharpe=sharpe, max_lose_streak=int(mcl),
        ci95_mean=ci, ci_excl_0=bool(ci[0] > 0 or ci[1] < 0),
        dur_med_min=float(tr.dur_m1.median()), dur_p90_min=float(tr.dur_m1.quantile(0.9)),
        n_long=len(lo), n_short=len(shrt), pnl_long=float(lo.pnl.sum()), pnl_short=float(shrt.pnl.sum()),
        pos_months=int((mo["sum"] > 0).sum()), total_months=len(mo),
        monthly={str(k): round(float(v), 1) for k, v in mo["sum"].items()},
    )


out = {"generated": "2026-09-17", "oos": "2024-08~2026-07 (24个月)", "design": {
    "m1sc": "固定 TP$1.20/SL$0.50/H90, M1事件网格, v3-34特征, plr_wr阈值(wr_floor=0.40), 单种子lgb+xgb",
    "m1sc_ad": "自适应 TP=max($1.2, 1.5×ATR288M1)/SL=0.4×TP/H90, 其余同m1sc",
}}
cmp_rows = {}
for fs, name in [("m1sc", "固定$1.2(用户字面口径)"), ("m1sc_ad", "自适应下限$1.2")]:
    per, tr = load(fs)
    s = stats(tr)
    fb = sum(1 for f in per if f["thr_fallback"])
    auc_l = np.nanmean([f["auc_oos_long"] for f in per])
    auc_s = np.nanmean([f["auc_oos_short"] for f in per])
    s.update(thr_fallback_folds=fb, auc_oos_long=float(auc_l), auc_oos_short=float(auc_s))
    cmp_rows[name] = s
    # 同频随机对照(变体自己的random_matched基准)
    rm = pd.read_csv(f"{BASE}/trades_baseline_random_matched.csv")
    rm = rm[rm.variant == "random_matched"] if "variant" in rm.columns else rm
    s["random_matched_pnl"] = None  # per-variant基准已覆盖写, 用per_fold取
    rms = []
    for f in per:
        b = f["baselines"].get("random_matched", {})
        if b and b.get("trades", 0) > 0:
            rms.append(b["total_pnl"])
    s["random_matched_pnl"] = float(np.sum(rms))

out["summary"] = cmp_rows
print(f"{'指标':<22}{'m1sc 固定$1.2':>18}{'m1sc_ad 自适应':>18}")
keys = ["n", "wr", "plr", "avg_win", "avg_loss", "pnl", "mean", "pf", "sharpe", "mdd",
        "max_lose_streak", "pos_months", "dur_med_min", "n_long", "n_short",
        "pnl_long", "pnl_short", "ci95_mean", "thr_fallback_folds", "auc_oos_long", "auc_oos_short",
        "random_matched_pnl"]
fmt = {"n": "{:.0f}", "pos_months": "{}", "n_long": "{:.0f}", "n_short": "{:.0f}",
       "max_lose_streak": "{:.0f}", "thr_fallback_folds": "{:.0f}", "dur_med_min": "{:.0f}",
       "ci95_mean": "[{:.3f},{:.3f}]"}
for k in keys:
    a = cmp_rows["固定$1.2(用户字面口径)"].get(k)
    b = cmp_rows["自适应下限$1.2"].get(k)
    def _f(v):
        if v is None:
            return "-"
        if isinstance(v, (tuple, list)):
            return "[{:.3f},{:.3f}]".format(*v)
        return fmt.get(k, "{:.3f}").format(v)
    print(f"{k:<22}{_f(a):>18}{_f(b):>18}")

json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False, default=str)
print(f"\nsaved -> {OUT}")
