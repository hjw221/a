# 路线C: 出场几何快测 (不重训模型)
#   模式A[隔离]: 恢复的概率流 + 原折阈值, 只换障碍几何 -> 纯出场几何效应
#   模式B[忠实]: 新几何下用训练窗val段重校准阈值(plr_wr, 与真实变体口径一致)
#   附: 基线交易的ATR分位/时段/方向 边际诊断 (只诊断, 不做OOS择门)
# 断点续跑: 每几何结果存 screen_cache.pkl, 重跑自动跳过
import os, sys, json, time, pickle
import numpy as np
import pandas as pd

BASE = "/home/z/my-project/download/xauusd_ml_v2"
sys.path.insert(0, BASE)
from config import CFG
from data import load_data
from run_all import variant_cfg
from walkforward import build_folds
from labeling import make_labels
from backtest import (build_signals, run_backtest, metrics,
                      calibrate_threshold, bootstrap_ci)

CACHE = "/home/z/my-project/scripts/screen_cache.pkl"
COLS = ["trades", "win_rate", "plr", "total_pnl", "mean_pnl", "sharpe", "max_dd",
        "avg_win", "avg_loss", "tp_pct", "to_pct", "profit_months", "total_months",
        "top2_share", "long_pnl", "short_pnl", "ci_total"]

t0 = time.time()
m5, m1_pack, spread_cost, monthly = load_data(CFG)
cfg_bal = variant_cfg("v3bal_ens")
folds = build_folds(m5, cfg_bal)

with open("/home/z/my-project/scripts/probs_recovered.pkl", "rb") as f:
    rec = pickle.load(f)
thr_rows = json.load(open(os.path.join(BASE, "results", "per_fold_v3bal_ens.json")))

# ATR序列 (与features_v3逐位一致)
h, l, c = m5["HIGH"], m5["LOW"], m5["CLOSE"]
tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
atr = tr.rolling(288, min_periods=288).mean()
atr_const = pd.Series(1.0, index=m5.index)

GEOS = [
    ("bal_3.0_1.143_h360",  "atr", 3.0, 1.1429, 360),
    ("atr_2.0_1.143_h360", "atr", 2.0, 1.1429, 360),
    ("atr_4.0_1.143_h360", "atr", 4.0, 1.1429, 360),
    ("atr_3.0_0.8_h360",    "atr", 3.0, 0.8,   360),
    ("atr_3.0_1.5_h360",    "atr", 3.0, 1.5,   360),
    ("atr_2.5_1.0_h360",    "atr", 2.5, 1.0,   360),
    ("atr_3.0_1.143_h120",  "atr", 3.0, 1.1429, 120),
    ("atr_3.0_1.143_h720",  "atr", 3.0, 1.1429, 720),
    ("atr_3.0_1.143_h1440", "atr", 3.0, 1.1429, 1440),
    ("fx_1.0_1.0_h360",  "fixed", 1.0, 1.0, 360),
    ("fx_2.0_1.0_h360",  "fixed", 2.0, 1.0, 360),
    ("fx_2.6_1.0_h360",  "fixed", 2.6, 1.0, 360),
    ("fx_3.0_1.0_h360",  "fixed", 3.0, 1.0, 360),
    ("fx_1.0_0.5_h360",  "fixed", 1.0, 0.5, 360),
    ("fx_2.0_0.5_h360",  "fixed", 2.0, 0.5, 360),
    ("fx_1.5_0.75_h360", "fixed", 1.5, 0.75, 360),
    ("fx_3.5_2.0_h360",  "fixed", 3.5, 2.0, 360),
    ("fx_1.0_0.5_h120",  "fixed", 1.0, 0.5, 120),
    ("fx_2.6_1.0_h120",  "fixed", 2.6, 1.0, 120),
]


def rich_metrics(log, name):
    mm = metrics(log, name)
    if len(log) == 0:
        return mm
    pnl = log["pnl"].to_numpy()
    mon = log.groupby("fold")["pnl"].sum()
    top2 = mon.nlargest(2).clip(lower=0).sum()
    (ci_m, ci_t) = bootstrap_ci(pnl, iters=2000)
    mm.update({
        "tp_pct": float((log["outcome"] == "TP").mean()),
        "to_pct": float((log["outcome"] == "TIMEOUT").mean()),
        "profit_months": int((mon > 0).sum()), "total_months": int(len(mon)),
        "top2_share": float(top2 / pnl.sum()) if pnl.sum() > 0 else np.nan,
        "long_pnl": float(log.loc[log["dir"] == "long", "pnl"].sum()),
        "short_pnl": float(log.loc[log["dir"] == "short", "pnl"].sum()),
        "ci_total": f"[{ci_t[0]:.0f},{ci_t[1]:.0f}]"})
    return mm


def print_row(tag, name, m):
    if m.get("trades", 0) == 0:
        print(f"[{tag}] {name:22s} 0笔"); return
    print(f"[{tag}] {name:22s} {m['trades']:5.0f}笔 WR{m['win_rate']*100:5.1f}% PLR{m['plr']:5.2f} "
          f"PnL${m['total_pnl']:9.1f} Shr{m['sharpe']:6.2f} TP%{m['tp_pct']*100:4.1f} "
          f"TO%{m['to_pct']*100:4.1f} 月+{m['profit_months']}/{m['total_months']}")


results = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}

for name, kind, tp, sl, hz in GEOS:
    if name in results:
        continue
    t1 = time.time()
    atr_use = atr if kind == "atr" else atr_const
    cfg_g = dict(cfg_bal)
    cfg_g.update(tp_atr_mult=tp, sl_atr_mult=sl, horizon_m1=hz)
    lab_g, valid_g = make_labels(m5, m1_pack, spread_cost, atr_use, cfg_g)
    t_lab = time.time() - t1

    # 模式A: 原折阈值
    logs_a = []
    for i, fold in enumerate(folds):
        d = rec[f"fold{i:02d}"]
        oos_r = np.arange(d["oos_lo"], d["oos_hi"] + 1)
        thr_l = float(thr_rows[i]["threshold"]["long"])
        thr_s = float(thr_rows[i]["threshold"]["short"])
        sig = build_signals(d["pl_oos"], d["ps_oos"], thr_l, thr_s)
        lg = run_backtest(sig, oos_r, lab_g, valid_g, m5, m1_pack, cfg_g)
        lg["fold"] = fold["oos_month"]
        logs_a.append(lg)
    log_a = pd.concat(logs_a, ignore_index=True)
    m_a = rich_metrics(log_a, name)
    print_row("A", name, m_a)

    # 模式B: val重校准阈值
    logs_b = []
    for i, fold in enumerate(folds):
        d = rec[f"fold{i:02d}"]
        oos_r = np.arange(d["oos_lo"], d["oos_hi"] + 1)
        val_r = np.arange(d["val_lo"], d["val_hi"] + 1)
        (thr_l, thr_s), _ = calibrate_threshold(
            d["pl_val"], d["ps_val"], val_r, lab_g, valid_g, m5, m1_pack, cfg_g)
        sig = build_signals(d["pl_oos"], d["ps_oos"], thr_l, thr_s)
        lg = run_backtest(sig, oos_r, lab_g, valid_g, m5, m1_pack, cfg_g)
        lg["fold"] = fold["oos_month"]
        logs_b.append(lg)
    log_b = pd.concat(logs_b, ignore_index=True)
    m_b = rich_metrics(log_b, name)
    print_row("B", name, m_b)

    results[name] = {"A": m_a, "B": m_b, "cfg": dict(kind=kind, tp=tp, sl=sl, hz=hz)}
    if name.startswith("bal_"):
        log_a.to_csv("/home/z/my-project/scripts/bal_trades_recovered.csv", index=False)
    pickle.dump(results, open(CACHE, "wb"), protocol=4)
    print(f"    ({t_lab:.0f}s标注 + {time.time()-t1-t_lab:.0f}s模拟, 累计{time.time()-t0:.0f}s)")

# ---- 汇总 ----
if len(results) < len(GEOS):
    print(f"\n[geo] 进度 {len(results)}/{len(GEOS)}, 续跑本脚本")
    sys.exit(0)

rows = []
for name, r in results.items():
    for mode in ["A", "B"]:
        rows.append({"geo": name, "mode": mode, **{c: r[mode].get(c) for c in COLS}})
df = pd.DataFrame(rows)
df.to_csv("/home/z/my-project/scripts/geo_screen_result.csv", index=False)
for mode in ["A", "B"]:
    print(f"\n===== 几何快测汇总 mode={mode} (按总PnL排序) =====")
    cols = ["geo", "trades", "win_rate", "plr", "total_pnl", "sharpe", "max_dd",
            "tp_pct", "to_pct", "profit_months", "top2_share", "long_pnl", "short_pnl"]
    print(df[df["mode"] == mode][cols].sort_values("total_pnl", ascending=False).to_string(
        index=False, float_format=lambda x: f"{x:9.3f}"))

# ---- 基线交易边际诊断 ----
bl = pd.read_csv("/home/z/my-project/scripts/bal_trades_recovered.csv",
                 parse_dates=["signal_time", "entry_time", "exit_time"])
bl["atr"] = atr.to_numpy()[bl["row"].to_numpy()]
bl["hour"] = bl["signal_time"].dt.hour
print("\n===== 基线(bal)交易边际诊断 =====")
print("ATR分布(信号时):", bl["atr"].describe().round(2).to_dict())
q = pd.qcut(bl["atr"], 5, duplicates="drop")
g = bl.groupby(q, observed=True).agg(n=("pnl", "size"), wr=("pnl", lambda s: (s > 0).mean()),
                                     pnl=("pnl", "sum"), mean=("pnl", "mean"))
print("\n-- by ATR五分位 --"); print(g.to_string(float_format=lambda x: f"{x:8.2f}"))
g = bl.groupby("hour").agg(n=("pnl", "size"), wr=("pnl", lambda s: (s > 0).mean()),
                           pnl=("pnl", "sum"))
print("\n-- by 小时 --"); print(g.to_string(float_format=lambda x: f"{x:8.2f}"))
g = bl.groupby("dir").agg(n=("pnl", "size"), wr=("pnl", lambda s: (s > 0).mean()),
                          pnl=("pnl", "sum"), mean=("pnl", "mean"))
print("\n-- by 方向 --"); print(g.to_string(float_format=lambda x: f"{x:8.2f}"))
g = bl.groupby(pd.Grouper(key="exit_time", freq="QE")).agg(
    n=("pnl", "size"), pnl=("pnl", "sum"))
print("\n-- by 季度 --"); print(g.to_string(float_format=lambda x: f"{x:8.2f}"))
print(f"\n[geo] 全部完成, 耗时{time.time()-t0:.0f}s")
