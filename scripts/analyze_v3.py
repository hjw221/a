"""v3盈亏比重写 — 真实结果汇总分析 (全部数字来自真实运行, 无任何虚构)"""
import sys, os, json, pickle
sys.path.insert(0, "/home/z/my-project/download/xauusd_ml_v2")
import numpy as np
import pandas as pd

BASE = "/home/z/my-project/download/xauusd_ml_v2"
RES = f"{BASE}/results"

VARIANTS_INFO = {
    # 变体: (特征集, 几何, 阈值目标, 定位)
    "legacy":             ("legacy 23特征", "旧 2.0/1.14/2h", "mean", "原基线"),
    "v2":                 ("v2 70特征", "旧 2.0/1.14/2h", "mean", "上代大特征集(失败)"),
    "v3":                 ("v3 34特征", "4.0/1.0/6h", "plr", "重写特征+PLR几何"),
    "legacy_plrgeo":      ("legacy 23特征", "4.0/1.0/6h", "plr", "归因:同几何旧特征"),
    "v3aggr":             ("v3 34特征", "4.0/0.8/6h", "plr", "重写特征+激进几何"),
    "legacy_aggr":        ("legacy 23特征", "4.0/0.8/6h", "plr", "归因:激进几何旧特征"),
    "legacy_plrgeo_mean": ("legacy 23特征", "4.0/1.0/6h", "mean", "阈值目标消融"),
}


def boot_ci(pnl, iters=10000, seed=42):
    pnl = np.asarray(pnl, np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pnl), size=(iters, len(pnl)))
    means = pnl[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def full_metrics(log):
    pnl = log["pnl"].to_numpy()
    w, l = pnl[pnl > 0], pnl[pnl < 0]
    eq = np.cumsum(pnl)
    daily = pd.Series(pnl, index=pd.to_datetime(log["exit_time"])).resample("D").sum()
    daily = daily[daily.index.dayofweek < 5]
    return {
        "trades": int(len(log)),
        "win_rate": float(len(w) / len(pnl)),
        "plr": float(w.mean() / abs(l.mean())),
        "avg_win": float(w.mean()), "avg_loss": float(l.mean()),
        "total_pnl": float(pnl.sum()), "mean_pnl": float(pnl.mean()),
        "pf": float(w.sum() / abs(l.sum())),
        "sharpe": float(np.sqrt(252) * daily.mean() / daily.std()) if daily.std() > 0 else np.nan,
        "max_dd": float((np.maximum.accumulate(eq) - eq).max()),
    }


def main():
    rows, ci_rows = [], []
    logs = {}
    for tag, (feat, geo, obj, note) in VARIANTS_INFO.items():
        fp = f"{RES}/trades_{tag}.csv"
        if not os.path.exists(fp):
            print(f"[skip] {tag}: 无交易日志")
            continue
        log = pd.read_csv(fp)
        logs[tag] = log
        m = full_metrics(log)
        lo, hi = boot_ci(log["pnl"].to_numpy())
        m.update({"variant": tag, "features": feat, "geometry": geo, "thr_obj": obj, "note": note,
                  "ci95_mean": [round(lo, 3), round(hi, 3)]})
        rows.append(m)
        ci_rows.append({"variant": tag, "ci_lo": lo, "ci_hi": hi, "ci_excl_0": bool(lo > 0)})

    df = pd.DataFrame(rows).set_index("variant")
    cols = ["features", "geometry", "thr_obj", "trades", "win_rate", "plr", "avg_win", "avg_loss",
            "total_pnl", "mean_pnl", "pf", "sharpe", "max_dd"]
    pd.set_option("display.width", 250)
    print("===== 7变体真实OOS对比 (2024-08 ~ 2026-07, 24个月, 单持仓+冷却+点差成本) =====")
    print(df[cols].round(3).to_string())
    print("\n===== 交易级Bootstrap 95%CI (均值PnL $/笔) =====")
    for r in ci_rows:
        print(f"  {r['variant']:20s} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  不含0: {r['ci_excl_0']}")

    # 月度集中度 + 多空拆分 (关键变体)
    print("\n===== 关键变体: 月度集中度 / 多空拆分 =====")
    detail = {}
    for tag in ["legacy", "v3", "legacy_plrgeo", "v3aggr", "legacy_aggr"]:
        if tag not in logs:
            continue
        log = logs[tag].copy()
        log["month"] = pd.to_datetime(log["exit_time"]).dt.to_period("M")
        g = log.groupby("month")["pnl"].sum().sort_values(ascending=False)
        top2 = g.iloc[:2].sum()
        dirg = log.groupby("dir")["pnl"].agg(["count", "sum"])
        d = {}
        for dd in dirg.index:
            x = log[log["dir"] == dd]["pnl"].to_numpy()
            w_, l_ = x[x > 0], x[x < 0]
            d[dd] = {"n": int(len(x)), "pnl": float(x.sum()),
                     "plr": float(w_.mean() / abs(l_.mean())) if len(w_) and len(l_) else np.nan,
                     "wr": float(len(w_) / len(x))}
        detail[tag] = {
            "monthly_top2_share_pct": float(top2 / max(log["pnl"].sum(), 1e-9) * 100),
            "best_months": {str(k): round(v, 1) for k, v in g.iloc[:3].items()},
            "worst_month": {str(g.index[-1]): round(g.iloc[-1], 1)},
            "dirs": d,
        }
        print(f"  [{tag}] 前2月利润占比 {detail[tag]['monthly_top2_share_pct']:.0f}%  "
              f"最赚月 {list(detail[tag]['best_months'].items())[0]}  最亏月 {list(detail[tag]['worst_month'].items())[0]}")
        for dd, v in d.items():
            print(f"      {dd:5s}: n={v['n']:4d} WR={v['wr']*100:.1f}% PLR={v['plr']:.2f} PnL=${v['pnl']:.1f}")

    out = {"summary": df.reset_index().to_dict(orient="records"),
           "bootstrap": ci_rows, "detail": detail,
           "generated": "2026-09-08", "oos": "2024-08~2026-07"}
    json.dump(out, open(f"{RES}/analysis_v3.json", "w"), indent=2, ensure_ascii=False, default=str)
    print(f"\n已写入 {RES}/analysis_v3.json")


if __name__ == "__main__":
    main()
