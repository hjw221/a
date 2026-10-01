"""v5信号强度三张牌 — 真实结果汇总分析 (全部数字来自真实运行, 无任何虚构)。
对比: v3bal基线 vs v3bal_uniq(去重叠) / v3bal_ens(多种子AUC加权) / v3bal_meta(元标签)。
同一套: v3特征34 + 平衡几何(3.0/1.1429/360M1) + plr_wr阈值 + 24个真实OOS月。"""
import sys, os, json
sys.path.insert(0, "/home/z/my-project/download/xauusd_ml_v2")
import numpy as np
import pandas as pd

BASE = "/home/z/my-project/download/xauusd_ml_v2"
RES = f"{BASE}/results"

VARIANTS_INFO = {
    "v3bal":      ("v3bal基线",  "逐根训练+单种子lgb/xgb均匀平均"),
    "v3bal_uniq": ("牌1去重叠",  "M5事件采样(每12根取1)+叶子参数等效缩放"),
    "v3bal_ens":  ("牌2集成",    "lgb/xgb×3种子=6成员, val-AUC加权, <0.51剔除"),
    "v3bal_meta": ("牌3元标签",  "主模型同基线+top15%候选+第二层LGB过滤"),
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


def per_fold_auc(tag):
    fp = f"{RES}/per_fold_{tag}.json"
    if not os.path.exists(fp):
        return None
    rows = json.load(open(fp))
    aucs = [(r.get("auc_oos_long"), r.get("auc_oos_short")) for r in rows]
    aucs = [(a, b) for a, b in aucs if a is not None and b is not None]
    if not aucs:
        return None
    return {"auc_oos_long": float(np.nanmean([a for a, _ in aucs])),
            "auc_oos_short": float(np.nanmean([b for _, b in aucs])),
            "n_folds": len(aucs),
            "thr_fallback_folds": int(sum(1 for r in rows if r.get("thr_fallback")))}


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
        pf = per_fold_auc(tag)
        if pf:
            m.update(pf)
        # 同频随机对照 (每变体自己的随机基线)
        s = json.load(open(f"{RES}/summary_{tag}.json"))
        m["random_matched_pnl"] = float(s["baselines"]["random_matched"]["total_pnl"])
        m["edge_vs_random"] = m["total_pnl"] - m["random_matched_pnl"]
        rows.append(m)
        ci_rows.append({"variant": tag, "ci_lo": lo, "ci_hi": hi, "ci_excl_0": bool(lo > 0)})

    df = pd.DataFrame(rows).set_index("variant")
    cols = ["name", "trades", "trades_per_day", "win_rate", "plr", "E_per_trade" if "E_per_trade" in df else "mean_pnl",
            "total_pnl", "pf", "sharpe", "max_dd", "max_lose_streak",
            "auc_oos_long", "auc_oos_short", "thr_fallback_folds"]
    cols = [c for c in cols if c in df.columns]
    pd.set_option("display.width", 300)
    print("===== 信号强度三张牌: 真实OOS对比 (2024-08~2026-07, 24个月, 单持仓+冷却+点差) =====")
    print(df[cols].round(3).to_string())
    print("\n===== 交易级Bootstrap 95%CI (均值PnL $/笔) =====")
    for r in ci_rows:
        print(f"  {r['variant']:12s} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  不含0: {r['ci_excl_0']}")
    print("\n===== 模型真实边际 (总PnL - 同频随机对照, $/24月) =====")
    for _, r in df.iterrows():
        print(f"  {r['name']:8s} 总PnL {r['total_pnl']:+8.1f}  同频随机 {r['random_matched_pnl']:+8.1f}  "
              f"模型边际 {r['edge_vs_random']:+8.1f}")

    # ---- 冠军细节: v3bal_ens vs v3bal 月度对照 + 多空拆分 ----
    detail = {}
    for tag in ["v3bal", "v3bal_ens", "v3bal_uniq", "v3bal_meta"]:
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
                                            / max(log["pnl"].sum(), 1e-9) * 100),
            "dirs": dirg}
    print("\n===== 多空拆分 =====")
    for tag, d in detail.items():
        for dd, v in d["dirs"].items():
            print(f"  [{tag:10s}] {dd:5s}: n={v['n']:4d} WR={v['wr']*100:5.1f}% PLR={v['plr']:5.2f} PnL${v['pnl']:+8.1f}")

    if "v3bal" in detail and "v3bal_ens" in detail:
        m1, m2 = detail["v3bal"]["monthly"], detail["v3bal_ens"]["monthly"]
        months = sorted(set(m1) | set(m2))
        print("\n===== 月度PnL对照: v3bal基线 -> v3bal_ens ($) =====")
        for mo in months:
            a, b = m1.get(mo, 0.0), m2.get(mo, 0.0)
            print(f"  {mo}: {a:+8.1f} -> {b:+8.1f}  ({'+' if b >= a else ''}{b - a:.1f})")
        print(f"  盈利月: {detail['v3bal']['profit_months']}/24 -> {detail['v3bal_ens']['profit_months']}/24")
        print(f"  前2月利润占比: {detail['v3bal']['monthly_top2_share_pct']:.0f}% -> "
              f"{detail['v3bal_ens']['monthly_top2_share_pct']:.0f}%")

    # ---- 牌内诊断 (per-fold extra) ----
    print("\n===== 牌内诊断 =====")
    diag = {}
    for tag in ["v3bal_uniq", "v3bal_ens", "v3bal_meta"]:
        fp = f"{RES}/per_fold_{tag}.json"
        if not os.path.exists(fp):
            continue
        rows_f = json.load(open(fp))
        d = {}
        if tag == "v3bal_uniq":
            d["n_fit_mean"] = float(np.mean([r["extra"]["n_fit"] for r in rows_f]))
        if tag == "v3bal_ens":
            fb = [r["extra"]["directions"][dd]["fallback_uniform"]
                  for r in rows_f for dd in ["long", "short"]]
            kept = [r["extra"]["directions"][dd]["n_kept"] for r in rows_f for dd in ["long", "short"]]
            aucm = [r["extra"]["directions"][dd]["mean_auc_val"] for r in rows_f for dd in ["long", "short"]]
            d["uniform_fallback_rate"] = float(np.mean(fb))
            d["n_kept_mean"] = float(np.mean(kept))
            d["member_auc_val_mean"] = float(np.mean(aucm))
        if tag == "v3bal_meta":
            prim_v = [r["extra"]["directions"][dd]["auc_primary_val"] for r in rows_f for dd in ["long", "short"]]
            meta_v = [r["extra"]["directions"][dd]["auc_meta_val"] for r in rows_f
                      for dd in ["long", "short"] if r["extra"]["directions"][dd].get("auc_meta_val") is not None]
            fbk = [r["extra"]["directions"][dd]["fallback"] for r in rows_f for dd in ["long", "short"]]
            d["primary_auc_val_mean"] = float(np.nanmean(prim_v))
            d["meta_auc_val_mean"] = float(np.nanmean(meta_v)) if meta_v else None
            d["meta_fallback_rate"] = float(np.mean(fbk))
        diag[tag] = d
        print(f"  [{tag}] {json.dumps(d, default=str)}")

    out = {"summary": df.reset_index().to_dict(orient="records"),
           "bootstrap": ci_rows, "detail": detail, "diag": diag,
           "generated": "2026-09-14", "oos": "2024-08~2026-07",
           "design": "三张牌只动训练过程(特征/几何/阈值口径与v3bal完全一致), 复用pack_v3bal与冻结超参"}
    json.dump(out, open(f"{RES}/analysis_v5.json", "w"), indent=2, ensure_ascii=False, default=str)
    print(f"\n已写入 {RES}/analysis_v5.json")


if __name__ == "__main__":
    main()
