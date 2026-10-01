"""
分析 v8 — MA10特征实验: v3bal/v3bal_ens/v3ma10/v3ma10_ens 四变体24月真实OOS对比。
归因设计: v3ma10与v3bal家族共享bal几何/param_src=v3冻结超参/plr_wr阈值目标/回测口径,
唯一变量 = 特征集(34 -> 37, +ma10_dist/ma10_slope/ma10_run)。输出 results/analysis_v8.json。
"""
import os, sys, json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from backtest import metrics, bootstrap_ci

RES = os.path.join(BASE, "results")
VARIANTS = ["v3bal", "v3bal_ens", "v3ma10", "v3ma10_ens"]
LABELS = {"v3bal": "v3bal基线(单种子,34特征)", "v3bal_ens": "v3bal_ens(生产,6成员,34特征)",
          "v3ma10": "v3ma10(单种子,37特征=34+MA10x3)",
          "v3ma10_ens": "v3ma10_ens(6成员,37特征=34+MA10x3)"}

out = {"variants": {}, "attribution": {}, "note":
       "OOS 2024-08~2026-07 (24个月, expanding window, purge+embargo); "
       "四变体同bal几何(TP=3xATR/SL=1.143xATR/6h)同param_src=v3同plr_wr; 唯一变量=MA10特征块"}

trades = {}
for v in VARIANTS:
    df = pd.read_csv(os.path.join(RES, f"trades_{v}.csv"))
    for col in ["signal_time", "entry_time", "exit_time"]:
        df[col] = pd.to_datetime(df[col])
    trades[v] = df
    m = metrics(df, v)
    (mlo, mhi), (tlo, thi) = bootstrap_ci(df["pnl"].to_numpy(), iters=10000, seed=42)
    oos_days = (pd.to_datetime(df["exit_time"]).max() - pd.to_datetime(df["exit_time"]).min()).days
    m.update({
        "ci95_mean": [round(mlo, 4), round(mhi, 4)], "ci95_total": [round(tlo, 1), round(thi, 1)],
        "ci_excludes_zero": bool(mlo > 0 or mhi < 0),
        "trades_per_day": round(len(df) / (oos_days * 5 / 7), 1),   # 交易日近似=自然日×5/7
        "mean_pnl": round(m["mean_pnl"], 4), "total_pnl": round(m["total_pnl"], 1),
        "win_rate": round(m["win_rate"], 4), "plr": round(m["plr"], 3),
        "sharpe": round(m["sharpe"], 3) if np.isfinite(m["sharpe"]) else None,
        "max_dd": round(m["max_dd"], 1), "trades": int(m["trades"]),
        "avg_win": round(m["avg_win"], 2), "avg_loss": round(m["avg_loss"], 2),
    })
    # 多空拆分
    for d in ["long", "short"]:
        sub = df[df["dir"] == d]
        m[f"{d}_pnl"] = round(float(sub["pnl"].sum()), 1)
        m[f"{d}_trades"] = int(len(sub))
    # 月度集中度
    mo = df.copy()
    mo["month"] = pd.to_datetime(mo["exit_time"]).dt.to_period("M").astype(str)
    mp = mo.groupby("month")["pnl"].sum().sort_values(ascending=False)
    m["top2_month_pnl_share"] = round(float(mp.head(2).sum() / max(1e-9, mp[mp > 0].sum())), 3)
    m["profit_months"] = int((mp > 0).sum())
    out["variants"][v] = m

# ---- 归因: 特征增量(单种子与集成两个口径) ----
for a, b, tag in [("v3bal", "v3ma10", "单种子: +MA10特征"), ("v3bal_ens", "v3ma10_ens", "集成: +MA10特征")]:
    ma_, mb_ = out["variants"][a], out["variants"][b]
    out["attribution"][tag] = {
        "pnl_delta": round(mb_["total_pnl"] - ma_["total_pnl"], 1),
        "sharpe_delta": round((mb_["sharpe"] or 0) - (ma_["sharpe"] or 0), 2),
        "wr_delta_pp": round((mb_["win_rate"] - ma_["win_rate"]) * 100, 1),
        "plr_delta": round(mb_["plr"] - ma_["plr"], 3),
        "trades_delta": mb_["trades"] - ma_["trades"],
    }
# 集成增益在同特征集下的稳定性(对照)
for base, ens, tag in [("v3bal", "v3bal_ens", "34特征下集成增益"), ("v3ma10", "v3ma10_ens", "37特征下集成增益")]:
    ma_, mb_ = out["variants"][base], out["variants"][ens]
    out["attribution"][tag] = {
        "pnl_delta": round(mb_["total_pnl"] - ma_["total_pnl"], 1),
        "sharpe_delta": round((mb_["sharpe"] or 0) - (ma_["sharpe"] or 0), 2),
    }

# ---- OOS AUC 均值对比 (特征是否提升了排序能力) ----
auc = {}
for v in VARIANTS:
    pf = pd.read_json(os.path.join(RES, f"per_fold_{v}.json"))
    auc[v] = {"long": round(float(pf["auc_oos_long"].mean()), 4),
              "short": round(float(pf["auc_oos_short"].mean()), 4)}
out["oos_auc_mean"] = auc

# ---- 控制面: 同频随机对照边际 (模型真实边际) ----
edge = {}
for v in VARIANTS:
    s = json.load(open(os.path.join(RES, f"summary_{v}.json")))
    rm = s["baselines"]["random_matched"]["total_pnl"]
    edge[v] = {"ml_pnl": round(s["ml"]["total_pnl"], 1),
               "random_matched_pnl": round(rm, 1),
               "model_edge": round(s["ml"]["total_pnl"] - rm, 1)}
out["model_edge_vs_samefreq_random"] = edge

json.dump(out, open(os.path.join(RES, "analysis_v8.json"), "w"), indent=2, ensure_ascii=False)

# ---- 控制台报告 ----
print("=" * 100)
print("MA10特征实验 — 24月真实OOS对比 (2024-08 ~ 2026-07, 单持仓+冷却+点差)")
print("=" * 100)
hdr = f"{'变体':<28}{'笔数':>6}{'笔/天':>6}{'胜率':>7}{'PLR':>6}{'单笔$':>8}{'总PnL$':>9}{'Sharpe':>8}{'maxDD':>7}{'CI95(单笔)':>20}"
print(hdr)
for v in VARIANTS:
    m = out["variants"][v]
    ci = f"[{m['ci95_mean'][0]:+.3f},{m['ci95_mean'][1]:+.3f}]{'*不含0' if m['ci_excludes_zero'] else ''}"
    print(f"{LABELS[v]:<28}{m['trades']:>6}{m['trades_per_day']:>6}{m['win_rate']*100:>6.1f}%"
          f"{m['plr']:>6.2f}{m['mean_pnl']:>8.3f}{m['total_pnl']:>9.1f}"
          f"{m['sharpe']:>8.2f}{m['max_dd']:>7.0f}{ci:>22}")
print("\n---- 归因 (唯一变量=MA10特征块) ----")
for k, v in out["attribution"].items():
    extra = ""
    if "wr_delta_pp" in v:
        extra = f"  胜率{v['wr_delta_pp']:+.1f}pp  PLR{v['plr_delta']:+.3f}  笔数{v['trades_delta']:+d}"
    print(f"  {k}: PnL{v['pnl_delta']:+.1f}  Sharpe{v['sharpe_delta']:+.2f}{extra}")
print("\n---- OOS AUC均值 (排序能力) ----")
for v, a in auc.items():
    print(f"  {v:<14} long {a['long']:.4f}  short {a['short']:.4f}")
print("\n---- 模型边际 vs 同频随机 ----")
for v, e in edge.items():
    print(f"  {v:<14} ML {e['ml_pnl']:+.1f} vs 同频随机 {e['random_matched_pnl']:+.1f} => 边际 {e['model_edge']:+.1f}")
print(f"\n输出: results/analysis_v8.json")
