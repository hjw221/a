"""
Task14-C: M1剥头皮几何标定 — 只用首训练窗(2022-01~2024-07)随机入口, 无未来信息。

预注册规则 (写在看任何结果之前):
  - TP = $1.00 固定 (用户指定"一单吃1u波动"); spread_floor_mult = 3.0 (框架惯例)
  - 网格: SL ∈ {0.40,0.50,0.65,0.80,1.00} × horizon ∈ {30,45,60,90} 分钟
  - 约束: 随机入口 TP率(多/空两侧) >= 0.30 且 超时率 <= 0.45
  - 目标: 最大化 min(EV_long, EV_short)  [给模型最好的起点]
  - 事件截止到 (2024-08-01 - horizon - 2min), 标签绝不越过 2024-08-01
输出: scripts/scalp_geom_result.json (全部网格真实数字 + 选定几何)
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/z/my-project/download/xauusd_ml_v2")
from data import load_raw_m1, impute_spread
from scalp import label_all_m1

CSV = "/home/z/my-project/upload/5_extracted/XAUUSDc_M1_202201022305_202606262057.csv"
WIN_END = pd.Timestamp("2024-08-01")
TP_USD, FLOOR = 1.00, 3.0
SL_GRID = [0.40, 0.50, 0.65, 0.80, 1.00]
HZ_GRID = [30, 45, 60, 90]
EMBARGO = 2   # 分钟, 保证标签不越过窗口端

print("[geom] 加载M1...")
m1 = load_raw_m1(CSV)
spread_cost, _ = impute_spread(m1, 0.001)

t = (m1.index.astype("int64") // 10**9 // 60).to_numpy(np.int64)
o = m1["OPEN"].to_numpy(np.float64)
h = m1["HIGH"].to_numpy(np.float64)
l = m1["LOW"].to_numpy(np.float64)
c = m1["CLOSE"].to_numpy(np.float64)

end_min = int(WIN_END.value // 10**9 // 60)
max_hz = max(HZ_GRID)
cut_end = np.searchsorted(t, end_min)                 # 数组切到 2024-08-01
ts, to, th, tl, tc = t[:cut_end], o[:cut_end], h[:cut_end], l[:cut_end], c[:cut_end]
sc = spread_cost[:cut_end]
cut_ev = np.searchsorted(ts, end_min - max_hz - EMBARGO)   # 事件截止(标签不出窗)
print(f"[geom] 窗口事件 {cut_ev:,} 根, 数组 {len(ts):,} 根")

rows = []
for sl_usd in SL_GRID:
    for hz in HZ_GRID:
        res = label_all_m1(ts, to, th, tl, tc, sc, TP_USD, sl_usd, FLOOR, hz, 10)
        entry_idx, out_l, out_s, pnl_l, pnl_s = res[0], res[1], res[2], res[3], res[4]
        v = (entry_idx[:cut_ev] >= 0)
        ol, os_ = out_l[:cut_ev][v], out_s[:cut_ev][v]
        pl, ps = pnl_l[:cut_ev][v], pnl_s[:cut_ev][v]

        def stat(outc, pnl):
            tp = (outc == 1); sl_ = (outc == -1); to_ = (outc == 0)
            W = pnl[tp].mean() if tp.any() else np.nan
            L = pnl[sl_].mean() if sl_.any() else np.nan
            T = pnl[to_].mean() if to_.any() else np.nan
            ev = pnl.mean()
            # 模型需要把多少比例的SL结局翻转成TP才能打平: Δ = -EV/(W-L)
            delta = -ev / (W - L) if (W - L) > 0 else np.nan
            return {"tp_rate": float(tp.mean()), "sl_rate": float(sl_.mean()),
                    "to_rate": float(to_.mean()), "ev": float(ev),
                    "avg_win": float(W), "avg_loss": float(L), "avg_to": float(T),
                    "lift_needed": float(delta)}
        sl_i, ss_i = stat(ol, pl), stat(os_, ps)
        rows.append({"sl_usd": sl_usd, "horizon": hz, "long": sl_i, "short": ss_i,
                     "min_ev": min(sl_i["ev"], ss_i["ev"]),
                     "max_lift": max(sl_i["lift_needed"], ss_i["lift_needed"])})
        print(f"[geom] SL={sl_usd:.2f} HZ={hz:3d} | L: TP{sl_i['tp_rate']*100:5.1f}% "
              f"TO{sl_i['to_rate']*100:5.1f}% EV${sl_i['ev']:+.3f} lift{sl_i['lift_needed']*100:5.1f}pp | "
              f"S: TP{ss_i['tp_rate']*100:5.1f}% TO{ss_i['to_rate']*100:5.1f}% "
              f"EV${ss_i['ev']:+.3f} lift{ss_i['lift_needed']*100:5.1f}pp")

ok = [r for r in rows
      if r["long"]["tp_rate"] >= 0.30 and r["short"]["tp_rate"] >= 0.30
      and r["long"]["to_rate"] <= 0.45 and r["short"]["to_rate"] <= 0.45]
pool = ok if ok else rows
best = max(pool, key=lambda r: r["min_ev"])
print(f"\n[geom] 可行格子 {len(ok)}/{len(rows)}; 选定: SL=${best['sl_usd']:.2f} "
      f"horizon={best['horizon']}min (min EV ${best['min_ev']:+.3f}, 需抬升 "
      f"{best['max_lift']*100:.1f}pp)")
if not ok:
    print("[geom] 警告: 无格子满足约束, 已取全体中最优 (如实记录)")

json.dump({"rule": "max min(EV_L,EV_S) s.t. tp_rate>=0.30, to_rate<=0.45; TP=1.00 fixed; "
                  "events capped so labels stay before 2024-08-01",
           "window": "2022-01-02~2024-07-31 (首训练窗, 无未来信息)",
           "grid": rows, "feasible": len(ok),
           "chosen": {"tp_usd": TP_USD, "sl_usd": best["sl_usd"],
                      "horizon_m1": best["horizon"], "spread_floor_mult": FLOOR}},
          open("/home/z/my-project/scripts/scalp_geom_result.json", "w"), indent=2)
print("[geom] -> scripts/scalp_geom_result.json")
