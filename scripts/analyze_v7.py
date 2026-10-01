"""v7路线A(DXY跨资产) — 真实结果汇总分析 (全部数字来自真实运行, 无任何虚构)。
对比: v3bal/v3bal_ens (纯XAUUSD底座) vs v4dxy/v4dxy_ens (34+8个DXY因果特征)。
同一套: 平衡几何(3.0/1.1429/360M1) + plr_wr阈值 + 24个真实OOS月。
v4dxy系: 42特征(DXY与XAU同MT5服务器导出, 新鲜率98.8%, 截断vs全量因果校验0不一致) + 新调参冻结。"""
import sys, os, json
sys.path.insert(0, "/home/z/my-project/download/xauusd_ml_v2")
import numpy as np
import pandas as pd

BASE = "/home/z/my-project/download/xauusd_ml_v2"
RES = f"{BASE}/results"

VARIANTS_INFO = {
    "v3bal":     ("v3·单种子", "34特征"),
    "v3bal_ens": ("v3·集成",   "34特征+6成员加权"),
    "v4dxy":     ("v4·单种子", "42特征(+8个DXY)"),
    "v4dxy_ens": ("v4·集成",   "42特征(+8个DXY)+6成员加权"),
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


def per_fold_diag(tag):
    fp = f"{RES}/per_fold_{tag}.json"
    if not os.path.exists(fp):
        return None
    rows = json.load(open(fp))
    return {"auc_oos_long": float(np.nanmean([r["auc_oos_long"] for r in rows])),
            "auc_oos_short": float(np.nanmean([r["auc_oos_short"] for r in rows])),
            "thr_fallback_folds": int(sum(1 for r in rows if r.get("thr_fallback"))),
            "n_folds": len(rows)}


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
        pf = per_fold_diag(tag)
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
    print("===== 路线A: DXY跨资产 真实OOS对比 (2024-08~2026-07, 24个月, 单持仓+冷却+点差) =====")
    print(df[cols].round(3).to_string())
    print("\n===== 交易级Bootstrap 95%CI (均值PnL $/笔) =====")
    for r in ci_rows:
        print(f"  {r['variant']:12s} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  不含0: {r['ci_excl_0']}")

    # DXY特征的实际贡献归因: 同结构两两对比
    print("\n===== DXY特征贡献归因 (同结构对比) =====")
    for a, b in [("v3bal", "v4dxy"), ("v3bal_ens", "v4dxy_ens")]:
        if a in df.index and b in df.index:
            print(f"  {df.loc[a,'name']:8s} +8个DXY特征 -> {df.loc[b,'name']:8s}: "
                  f"PnL ${df.loc[a,'total_pnl']:+.0f} -> ${df.loc[b,'total_pnl']:+.0f} "
                  f"({df.loc[b,'total_pnl']-df.loc[a,'total_pnl']:+.0f})  "
                  f"Sharpe {df.loc[a,'sharpe']:.2f} -> {df.loc[b,'sharpe']:.2f}  "
                  f"AUC(L) {df.loc[a,'auc_oos_long']:.4f} -> {df.loc[b,'auc_oos_long']:.4f}")
    # 集成贡献归因
    for a, b in [("v3bal", "v3bal_ens"), ("v4dxy", "v4dxy_ens")]:
        if a in df.index and b in df.index:
            print(f"  {df.loc[a,'name']:8s} +6成员集成  -> {df.loc[b,'name']:8s}: "
                  f"PnL ${df.loc[a,'total_pnl']:+.0f} -> ${df.loc[b,'total_pnl']:+.0f} "
                  f"({df.loc[b,'total_pnl']-df.loc[a,'total_pnl']:+.0f})  "
                  f"Sharpe {df.loc[a,'sharpe']:.2f} -> {df.loc[b,'sharpe']:.2f}")

    # 月度与多空
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
        detail[tag] = {"monthly": {str(k): round(v, 1) for k, v in g.items()},
                       "profit_months": int((g > 0).sum()), "total_months": int(len(g)),
                       "dirs": dirg}
    print("\n===== 多空拆分 =====")
    for tag, d in detail.items():
        for dd, v in d["dirs"].items():
            print(f"  [{tag:12s}] {dd:5s}: n={v['n']:5d} WR={v['wr']*100:5.1f}% PLR={v['plr']:5.2f} PnL${v['pnl']:+8.1f}")

    out = {"summary": df.reset_index().to_dict(orient="records"),
           "bootstrap": ci_rows, "detail": detail,
           "generated": "2026-09-15", "oos": "2024-08~2026-07",
           "design": "v4dxy系= v3全部34特征+8个DXY因果特征(同MT5导出, 新鲜率98.8%, 因果校验通过)+新调参; "
                     "标签/几何/阈值口径/回测与v3bal系完全一致"}
    json.dump(out, open(f"{RES}/analysis_v7.json", "w"), indent=2, ensure_ascii=False, default=str)
    print(f"\n已写入 {RES}/analysis_v7.json")


if __name__ == "__main__":
    main()
