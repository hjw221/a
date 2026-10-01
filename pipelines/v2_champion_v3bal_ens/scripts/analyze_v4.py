"""v4平衡几何 — 真实结果汇总分析 (全部数字来自真实运行, 无任何虚构)。
新增: 连亏统计(用户关心胜率的直接痛点) + 胜率/盈亏比操作点前沿。"""
import sys, os, json
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG_ROOT)
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = f"{BASE}/results"

VARIANTS_INFO = {
    # 变体: (特征集, 几何, 阈值目标, 定位)
    "legacy":             ("legacy 23特征", "2.0/1.14/2h",  "mean",   "原基线(胜率最高)"),
    "legacy_bal":         ("legacy 23特征", "3.0/1.14/6h",  "plr_wr", "平衡几何+旧特征"),
    "v3bal":              ("v3 34特征",     "3.0/1.14/6h",  "plr_wr", "平衡几何+v3特征"),
    "v3":                 ("v3 34特征",     "4.0/1.0/6h",   "plr",    "PLR主几何"),
    "legacy_plrgeo":      ("legacy 23特征", "4.0/1.0/6h",   "plr",    "归因:同几何旧特征"),
    "v3aggr":             ("v3 34特征",     "4.0/0.8/6h",   "plr",    "激进几何(盈亏比最高)"),
    "legacy_aggr":        ("legacy 23特征", "4.0/0.8/6h",   "plr",    "归因:激进几何旧特征"),
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


def streak_pnl(pnl):
    """最长连亏段的累计亏损$。"""
    w = (pnl <= 0).astype(int)
    segs, cur, s0 = [], 0, 0
    for i, x in enumerate(w):
        if x:
            if cur == 0: s0 = i
            cur += 1
        else:
            if cur > 0: segs.append((cur, pnl[s0:i].sum()))
            cur = 0
    if cur > 0: segs.append((cur, pnl[s0:].sum()))
    return max(segs, key=lambda t: t[0]) if segs else (0, 0.0)


def full_metrics(log):
    pnl = log["pnl"].to_numpy()
    w, l = pnl[pnl > 0], pnl[pnl < 0]
    eq = np.cumsum(pnl)
    daily = pd.Series(pnl, index=pd.to_datetime(log["exit_time"])).resample("D").sum()
    daily = daily[daily.index.dayofweek < 5]
    st_n, st_usd = streak_pnl(pnl)
    return {
        "trades": int(len(log)),
        "win_rate": float(len(w) / len(pnl)),
        "plr": float(w.mean() / abs(l.mean())),
        "avg_win": float(w.mean()), "avg_loss": float(l.mean()),
        "total_pnl": float(pnl.sum()), "mean_pnl": float(pnl.mean()),
        "pf": float(w.sum() / abs(l.sum())),
        "sharpe": float(np.sqrt(252) * daily.mean() / daily.std()) if daily.std() > 0 else np.nan,
        "max_dd": float((np.maximum.accumulate(eq) - eq).max()),
        "E_per_trade": float(len(w) / len(pnl) * w.mean() - len(l) / len(pnl) * abs(l.mean())),
        "max_lose_streak": int(st_n),
        "streak_loss_usd": float(st_usd),
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
            "E_per_trade", "max_lose_streak", "streak_loss_usd", "total_pnl", "pf", "sharpe", "max_dd"]
    pd.set_option("display.width", 260)
    print("===== 真实OOS对比 (2024-08 ~ 2026-07, 24个月, 单持仓+冷却+点差成本) =====")
    print(df[cols].round(3).to_string())
    print("\n===== 交易级Bootstrap 95%CI (均值PnL $/笔) =====")
    for r in ci_rows:
        print(f"  {r['variant']:20s} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  不含0: {r['ci_excl_0']}")

    # 胜率-盈亏比操作点前沿 (几何决定, 模型在同一几何族内可平移)
    print("\n===== 胜率/盈亏比操作点前沿 (真实OOS实现值) =====")
    frontier = df[["geometry", "win_rate", "plr", "total_pnl", "sharpe", "max_lose_streak"]].copy()
    print(frontier.round(3).to_string())

    # v3bal 细节: 多空拆分 + 月度集中度
    print("\n===== v3bal 细节 =====")
    detail = {}
    for tag in ["v3bal", "legacy_bal"]:
        if tag not in logs:
            continue
        log = logs[tag].copy()
        log["month"] = pd.to_datetime(log["exit_time"]).dt.to_period("M")
        g = log.groupby("month")["pnl"].sum().sort_values(ascending=False)
        dirg = log.groupby("dir")["pnl"].agg(["count", "sum"])
        d = {}
        for dd in dirg.index:
            x = log[log["dir"] == dd]["pnl"].to_numpy()
            w_, l_ = x[x > 0], x[x < 0]
            d[dd] = {"n": int(len(x)), "pnl": float(x.sum()),
                     "plr": float(w_.mean() / abs(l_.mean())) if len(w_) and len(l_) else np.nan,
                     "wr": float(len(w_) / len(x))}
        top2 = g.iloc[:2].sum()
        detail[tag] = {
            "monthly_top2_share_pct": float(top2 / max(log["pnl"].sum(), 1e-9) * 100),
            "best_months": {str(k): round(v, 1) for k, v in g.iloc[:3].items()},
            "worst_month": {str(g.index[-1]): round(g.iloc[-1], 1)},
            "dirs": d, "profit_months": int((g > 0).sum()), "total_months": int(len(g)),
        }
        print(f"  [{tag}] 盈利月 {detail[tag]['profit_months']}/{detail[tag]['total_months']}  "
              f"前2月利润占比 {detail[tag]['monthly_top2_share_pct']:.0f}%  "
              f"最赚月 {list(detail[tag]['best_months'].items())[0]}")
        for dd, v in d.items():
            print(f"      {dd:5s}: n={v['n']:4d} WR={v['wr']*100:.1f}% PLR={v['plr']:.2f} PnL=${v['pnl']:.1f}")

    out = {"summary": df.reset_index().to_dict(orient="records"),
           "bootstrap": ci_rows, "detail": detail,
           "generated": "2026-09-08", "oos": "2024-08~2026-07",
           "balanced_geometry": "TP=3.0xATR SL=1.1429xATR horizon=360M1 (v4标定, 首训练窗)"}
    json.dump(out, open(f"{RES}/analysis_v4.json", "w"), indent=2, ensure_ascii=False, default=str)
    print(f"\n已写入 {RES}/analysis_v4.json")


if __name__ == "__main__":
    main()
