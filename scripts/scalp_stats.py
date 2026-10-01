"""
Task14-A: M1剥头皮实验 — 数据体检 (仅诊断, 不做任何模型选择)
输出:
  1) M1 点差月度中位数(美元) — 全期 (数据卫生, 与框架口径一致)
  2) M1 TR/ATR1440 年度中位数 — 诊断: $1 在各年的相对位置
  3) 首训练窗(2022-01~2024-07) $1波动耗时分布 — 诊断: 超时网格依据
所有几何选择(Task14-C)只用首训练窗数据, 本脚本不选几何。
"""
import sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/z/my-project/download/xauusd_ml_v2")
from data import load_raw_m1

OUT = "/home/z/my-project/scripts/scalp_stats.json"

m1 = load_raw_m1("/home/z/my-project/upload/5_extracted/XAUUSDc_M1_202201022305_202606262057.csv")
print(f"[stats] M1 {len(m1):,} 行 {m1.index[0]} ~ {m1.index[-1]}")

# ---- 1) 点差月度中位数(非零) ----
sp = m1["SPREAD"].astype(np.float64).replace(0.0, np.nan)
monthly_sp = sp.groupby(m1.index.to_period("M")).median()
monthly_sp = monthly_sp.ffill().bfill() * 0.001
print(f"[stats] 点差中位数 首月=${monthly_sp.iloc[0]:.3f} 2024-01=${monthly_sp.loc[pd.Period('2024-01','M')]:.3f} "
      f"2025-06=${monthly_sp.loc[pd.Period('2025-06','M')]:.3f} 末月=${monthly_sp.iloc[-1]:.3f}")

# ---- 2) M1 波动率年度统计 ----
h, l, c = m1["HIGH"], m1["LOW"], m1["CLOSE"]
tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
atr1440 = tr.rolling(1440, min_periods=1440).mean()   # 24h M1 ATR
yr = m1.index.year
stats_yr = {}
for y in sorted(set(yr)):
    m = yr == y
    stats_yr[int(y)] = {
        "m1_tr_median": float(tr[m].median()),
        "atr1440_median": float(atr1440[m].median()),
        "p90": float(tr[m].quantile(0.90)),
    }
    print(f"[stats] {y}: M1单根TR中位 ${tr[m].median():.3f} | ATR1440中位 ${atr1440[m].median():.3f} "
          f"| TR P90 ${tr[m].quantile(0.90):.3f}")

# ---- 3) 首训练窗: $1 波动耗时 (从随机M1收盘起, 不含SL, 纯诊断) ----
first_win_end = pd.Timestamp("2024-08-01")
sub = m1[m1.index < first_win_end]
cn = sub["CLOSE"].to_numpy(np.float64)
tn = (sub.index.astype("int64") // 10**9 // 60).to_numpy(np.int64)
rng = np.random.default_rng(42)
idx = rng.choice(np.arange(len(cn) - 4000), size=20000, replace=False)
mins_to_1 = []
for i in idx:
    d = cn[i + 1:i + 241] - cn[i]     # 之后4小时内
    up = np.nonzero(d >= 1.0)[0]
    dn = np.nonzero(d <= -1.0)[0]
    tu = up[0] if len(up) else 10**9
    td = dn[0] if len(dn) else 10**9
    t = min(tu, td)
    if t < 10**9:
        mins_to_1.append(t)
mins_to_1 = np.array(mins_to_1)
print(f"[stats] 首训练窗 ±$1 触达耗时(分钟): 中位 {np.median(mins_to_1):.0f} "
      f"P25 {np.percentile(mins_to_1,25):.0f} P75 {np.percentile(mins_to_1,75):.0f} "
      f"240分钟内触达率 {(mins_to_1<240).mean()*100:.0f}%")

json.dump({"monthly_spread": {str(k): v for k, v in monthly_sp.items()},
           "yearly": stats_yr,
           "mins_to_dollar": {"median": float(np.median(mins_to_1)),
                              "p25": float(np.percentile(mins_to_1, 25)),
                              "p75": float(np.percentile(mins_to_1, 75))}},
          open(OUT, "w"), indent=2)
print(f"[stats] -> {OUT}")
