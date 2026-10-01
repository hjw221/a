"""v6频率拓展 — 真实结果汇总分析 (全部数字来自真实运行, 无任何虚构)。
对比: v3bal(2.6笔/天) / v3bal_ens(3.2笔/天) vs v3bal_hi(单种子+hi_freq) / v3bal_ens_hi(集成+hi_freq)。
同一套: v3特征34 + 平衡几何(3.0/1.1429/360M1) + 复用pack_v3bal与冻结超参 + 24个真实OOS月。
hi变体唯一差异: 阈值校准目标 plr_wr -> hi_freq(频率优先, 内部验证段≥10笔/天)。"""
import sys, os, json
sys.path.insert(0, "/home/z/my-project/download/xauusd_ml_v2")
import numpy as np
import pandas as pd

BASE = "/home/z/my-project/download/xauusd_ml_v2"
RES = f"{BASE}/results"

VARIANTS_INFO = {
    "v3bal":        ("v3bal基线",  "plr_wr阈值, 2.6笔/天"),
    "v3bal_ens":    ("牌2集成",    "plr_wr阈值+6成员加权, 3.2笔/天"),
    "v3bal_hi":     ("频率·单种子", "hi_freq阈值(≥10笔/天目标), 训练与v3bal完全一致"),
    "v3bal_ens_hi": ("频率·集成",  "hi_freq阈值 + 与v3bal_ens完全相同的6成员加权"),
}


def boot_ci(pnl, iters=10000, seed=42):
    pnl = np.asarray(pnl, np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pnl), size=(iters, len(pnl)))
    means = pnl[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def max_lose_streak(pnl):
    best = cur = 0
    for x in (pnl <= 0):
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


def full_metrics(log):
    pnl = log["pnl"].to_numpy()
    w, l = pnl[pnl > 0], pnl[pnl < 0]
    eq = np.cumsum(pnl)
    daily = pd.Series(pnl, index=pd.to_datetime(log["exit_time"])).resample("D").sum()
    daily = daily[daily.index.dayofweek < 5]
    return {
        "trades": int(len(log)),
        "trades_per_day": float(len(log) / 24 / 21),
        "win_rate": float(len(w) / len(pnl)),
        "plr": float(w.mean() / abs(l.mean())),
        "avg_win": float(w.mean()), "avg_loss": float(l.mean()),
        "total_pnl": float(pnl.sum()), "mean_pnl": float(pnl.mean()),
        "pf": float(w.sum() / abs(l.sum())),
        "sharpe": float(np.sqrt(252) * daily.mean() / daily.std()) if daily.std() > 0 else np.nan,
        "max_dd": float((np.maximum.accumulate(eq) - eq).max()),
        "max_lose_streak": int(max_lose_streak(pnl)),
    }


def per_fold_extra(tag):
    fp = f"{RES}/per_fold_{tag}.json"
    if not os.path.exists(fp):
        return None
    rows = json.load(open(fp))
    out = {"n_folds": len(rows),
           "thr_fallback_folds": int(sum(1 for r in rows if r.get("thr_fallback")))}
    # hi变体: 阈值选择细节 (频率目标/回退/q分位)
    ti = [r.get("threshold") for r in rows]
    info = [r for r in rows if r.get("extra") or True]
    fb_freq = 0
    qs = []
    for r in rows:
        # thr_info没有直接存json, 但fallback与q可以从stats推断不了 -> 用统计字段
        pass
    out["auc_oos_long"] = float(np.nanmean([r["auc_oos_long"] for r in rows]))
    out["auc_oos_short"] = float(np.nanmean([r["auc_oos_short"] for r in rows]))
    return out


def main():
    rows, ci_rows = [], []
    logs = {}
    for tag, (name, note) in VARIANTS_INFO.items():
        fp = f"{RES}/trades_{tag}.csv"
        if not os.path.exists(fp):
            print(f"[skip] {tag}: 无交易日志")
            continue
        log = pd.read_csv(fp)
        logs[tag] = log
        m = full_metrics(log)
        lo, hi = boot_ci(log["pnl"].to_numpy())
        m.update({"variant": tag, "name": name, "note": note,
                  "ci95_mean": [round(lo, 3), round(hi, 3)], "ci_excl_0": bool(lo > 0)})
        pf = per_fold_extra(tag)
        if pf:
            m.update(pf)
        s = json.load(open(f"{RES}/summary_{tag}.json"))
        m["random_matched_pnl"] = float(s["baselines"]["random_matched"]["total_pnl"])
        m["edge_vs_random"] = m["total_pnl"] - m["random_matched_pnl"]
        rows.append(m)
        ci_rows.append({"variant": tag, "ci_lo": lo, "ci_hi": hi, "ci_excl_0": bool(lo > 0)})

    df = pd.DataFrame(rows).set_index("variant")
    cols = ["name", "trades", "trades_per_day", "win_rate", "plr", "mean_pnl",
            "total_pnl", "pf", "sharpe", "max_dd", "max_lose_streak",
            "auc_oos_long", "auc_oos_short", "edge_vs_random"]
    cols = [c for c in cols if c in df.columns]
    pd.set_option("display.width", 300)
    print("===== 频率拓展: 真实OOS对比 (2024-08~2026-07, 24个月, 单持仓+冷却+点差) =====")
    print(df[cols].round(3).to_string())
    print("\n===== 交易级Bootstrap 95%CI (均值PnL $/笔) =====")
    for r in ci_rows:
        print(f"  {r['variant']:14s} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  不含0: {r['ci_excl_0']}")
    print("\n===== 操作点前沿: 频率 vs 单笔期望 (信号稀释曲线) =====")
    for _, r in df.sort_values("trades_per_day").iterrows():
        print(f"  {r['name']:10s} {r['trades_per_day']:5.1f}笔/天  {r['trades']:5d}笔  "
              f"单笔期望${r['mean_pnl']:+.3f}  WR {r['win_rate']*100:.1f}%  PLR {r['plr']:.2f}  "
              f"总PnL${r['total_pnl']:+.0f}")

    # ---- 月度对照: 基线/集成 在 plr_wr vs hi_freq 下的逐月差异 ----
    detail = {}
    for tag in VARIANTS_INFO:
        if tag not in logs:
            continue
        log = logs[tag].copy()
        log["month"] = pd.to_datetime(log["exit_time"]).dt.to_period("M")
        g = log.groupby("month")["pnl"].sum()
        dirg = {}
        for dd in log["dir"].unique():
            x = log[log["dir"] == dd]["pnl"].to_numpy()
            w_, l_ = x[x > 0], x[x < 0]
            dirg[dd] = {"n": int(len(x)), "pnl": float(x.sum()),
                        "plr": float(w_.mean() / abs(l_.mean())) if len(w_) and len(l_) else np.nan,
                        "wr": float(len(w_) / len(x))}
        detail[tag] = {
            "monthly": {str(k): round(v, 1) for k, v in g.items()},
            "profit_months": int((g > 0).sum()), "total_months": int(len(g)),
            "monthly_top2_share_pct": float(g.sort_values(ascending=False).iloc[:2].sum()
                                            / max(log["pnl"].sum(), 1e-9) * 100) if g.sum() > 0 else None,
            "dirs": dirg}
    print("\n===== 多空拆分 =====")
    for tag, d in detail.items():
        for dd, v in d["dirs"].items():
            print(f"  [{tag:14s}] {dd:5s}: n={v['n']:5d} WR={v['wr']*100:5.1f}% PLR={v['plr']:5.2f} PnL${v['pnl']:+8.1f}")

    if "v3bal_hi" in detail and "v3bal_ens_hi" in detail:
        m1, m2 = detail["v3bal_hi"]["monthly"], detail["v3bal_ens_hi"]["monthly"]
        months = sorted(set(m1) | set(m2))
        print("\n===== 月度PnL对照: v3bal_hi -> v3bal_ens_hi ($) =====")
        for mo in months:
            a, b = m1.get(mo, 0.0), m2.get(mo, 0.0)
            print(f"  {mo}: {a:+8.1f} -> {b:+8.1f}  ({'+' if b >= a else ''}{b - a:.1f})")

    # ---- hi阈值校准诊断: 逐折实际笔/天与回退 ----
    print("\n===== hi_freq阈值校准诊断 (内部验证段) =====")
    diag = {}
    for tag in ["v3bal_hi", "v3bal_ens_hi"]:
        fp = f"{RES}/per_fold_{tag}.json"
        if not os.path.exists(fp):
            continue
        rows_f = json.load(open(fp))
        tpd = [r["ml"]["trades"] / 21 for r in rows_f]
        diag[tag] = {"oos_trades_per_day_min": float(min(tpd)),
                     "oos_trades_per_day_median": float(np.median(tpd)),
                     "oos_trades_per_day_max": float(max(tpd)),
                     "thr_fallback_folds": int(sum(1 for r in rows_f if r.get("thr_fallback")))}
        print(f"  [{tag}] {json.dumps(diag[tag])}")

    out = {"summary": df.reset_index().to_dict(orient="records"),
           "bootstrap": ci_rows, "detail": detail, "diag": diag,
           "generated": "2026-09-15", "oos": "2024-08~2026-07",
           "design": "hi变体与对应底座唯一差异=阈值校准目标(plr_wr->hi_freq, 内部验证段≥10笔/天), "
                     "特征/几何/超参/训练全一致, 复用pack_v3bal"}
    json.dump(out, open(f"{RES}/analysis_v6.json", "w"), indent=2, ensure_ascii=False, default=str)
    print(f"\n已写入 {RES}/analysis_v6.json")


if __name__ == "__main__":
    main()
