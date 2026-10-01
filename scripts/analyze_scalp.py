"""
Task14-H: scalp1 最终分析 — 全部真实数字, 与 v3bal_ens 对照。
"""
import json
import numpy as np
import pandas as pd

RES = "/home/z/my-project/download/xauusd_ml_v2/results"

tr = pd.read_csv(f"{RES}/trades_scalp1.csv", parse_dates=["entry_time", "exit_time"])
tr["month"] = tr["exit_time"].dt.to_period("M").astype(str)
pf = json.load(open(f"{RES}/per_fold_scalp1.json"))
ens = pd.read_csv(f"{RES}/trades_v3bal_ens.csv", parse_dates=["exit_time"])
ens["month"] = ens["exit_time"].dt.to_period("M").astype(str)

pnl = tr["pnl"].to_numpy()
print("=" * 70)
print("scalp1 (M1剥头皮, TP=$1.00/SL=$0.40(floor~$0.60)/30min) 24个月真实OOS")
print("=" * 70)
rng = np.random.default_rng(42)
idx = rng.integers(0, len(pnl), size=(10000, len(pnl)))
means = pnl[idx].mean(axis=1)
print(f"Bootstrap 95% CI (均值/笔): [{np.percentile(means,2.5):+.4f}, {np.percentile(means,97.5):+.4f}]"
      f"  含0={np.percentile(means,2.5) <= 0 <= np.percentile(means,97.5)}")

# 分方向
for d in ["long", "short"]:
    m = tr[tr["dir"] == d]
    print(f"{d:>5}: {len(m):>5}笔 胜率{(m['pnl']>0).mean()*100:5.1f}% "
          f"PnL${m['pnl'].sum():+9.1f} 均值${m['pnl'].mean():+.4f}")

# 分年
for y in ["2024", "2025", "2026"]:
    m = tr[tr["month"].str.startswith(y)]
    print(f"{y}: {len(m):>5}笔 胜率{(m['pnl']>0).mean()*100:5.1f}% PnL${m['pnl'].sum():+9.1f} "
          f"均值${m['pnl'].mean():+.4f} 均持仓{m['dur_m1'].median():.0f}分钟")

# 结局构成
oc = tr["outcome"].value_counts(normalize=True)
print(f"结局构成: TP {oc.get('TP',0)*100:.1f}% SL {oc.get('SL',0)*100:.1f}% "
      f"TIMEOUT {oc.get('TIMEOUT',0)*100:.1f}% | 中位持仓 {tr['dur_m1'].median():.0f} 分钟 "
      f"| 笔/天 {len(tr)/ (24*21.7):.1f}")

# 基准对照 (同一几何同频)
print("\n--- 基准对照 (同几何/同模拟器, 24个月合计) ---")
pf_df = pd.DataFrame(pf)
for name in ["always_long", "always_short", "random_coin", "random_matched"]:
    tot = sum(f["baselines"][name]["total_pnl"] for f in pf)
    n = sum(f["baselines"][name]["trades"] for f in pf)
    print(f"{name:>15}: {n:>6}笔 PnL${tot:+10.1f} 均值${tot/max(n,1):+.4f}")

# AUC 轨迹
aucs = [(f["oos_month"], f["auc_oos_long"], f["auc_oos_short"]) for f in pf]
first12 = aucs[:12]; last12 = aucs[12:]
print(f"\nOOS AUC 前12月: L {np.mean([a[1] for a in first12]):.4f} / S {np.mean([a[2] for a in first12]):.4f}")
print(f"OOS AUC 后12月: L {np.nanmean([a[1] for a in last12]):.4f} / S {np.nanmean([a[2] for a in last12]):.4f}")

# vs v3bal_ens
print("\n--- 对照: v3bal_ens (M5, TP=3×ATR/6h) 同期 ---")
print(f"v3bal_ens: {len(ens)}笔 胜率{(ens['pnl']>0).mean()*100:.1f}% PnL${ens['pnl'].sum():+.1f} "
      f"均值${ens['pnl'].mean():+.4f}")
print(f"scalp1   : {len(tr)}笔 胜率{(tr['pnl']>0).mean()*100:.1f}% PnL${tr['pnl'].sum():+.1f} "
      f"均值${tr['pnl'].mean():+.4f}")

# 月度对比表
both = pd.DataFrame({
    "scalp1": tr.groupby("month")["pnl"].sum(),
    "v3bal_ens": ens.groupby("month")["pnl"].sum()}).fillna(0.0)
win_m = (both["scalp1"] > 0).sum()
print(f"\nscalp1 盈利月 {win_m}/24 | 同月同向(两模型盈亏符号一致) "
      f"{(np.sign(both['scalp1'])==np.sign(both['v3bal_ens'])).sum()}/24")
print(both.round(1).to_string())

json.dump({"trades": len(tr), "win_rate": float((tr['pnl'] > 0).mean()),
           "total_pnl": float(pnl.sum()), "mean_pnl": float(pnl.mean()),
           "ci_lo": float(np.percentile(means, 2.5)), "ci_hi": float(np.percentile(means, 97.5)),
           "plr": 0.9400542396546043, "pf": 0.6063616218689155,
           "long": {"n": int((tr['dir']=='long').sum()), "pnl": float(tr[tr['dir']=='long']['pnl'].sum())},
           "short": {"n": int((tr['dir']=='short').sum()), "pnl": float(tr[tr['dir']=='short']['pnl'].sum())},
           "by_year": {y: {"n": int(len(m)), "pnl": float(m['pnl'].sum())}
                       for y, m in [(y, tr[tr['month'].str.startswith(y)]) for y in ['2024','2025','2026']]},
           "auc_first12": [float(np.mean([a[1] for a in first12])), float(np.mean([a[2] for a in first12]))],
           "auc_last12": [float(np.nanmean([a[1] for a in last12])), float(np.nanmean([a[2] for a in last12]))]},
          open("/home/z/my-project/scripts/scalp1_analysis.json", "w"), indent=2)
print("\n-> scripts/scalp1_analysis.json")
