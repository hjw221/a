# 路线D: ATR regime 门控 (纯信号过滤, 不重训模型)
#   门控规则: 信号行 ATR >= gate_level 才放行; gate_level = 该折val段ATR分布的q分位 (训练窗信息)
#   每折独立在val段选q (候选 0/0.25/0.5/0.65/0.75), 目标: val总PnL最大, 约束val交易数>=100
#   安慰剂对照: 用逐行随机数属性走完全相同的门控流程 -> 若ATR门OOS增益≈安慰剂, 即为噪声
#   OOS零参与: q的选择只用val段; OOS只用已冻结的gate_level
import os, sys, json, time, pickle
import numpy as np
import pandas as pd

BASE = "/home/z/my-project/download/xauusd_ml_v2"
sys.path.insert(0, BASE)
from config import CFG
from data import load_data
from run_all import variant_cfg, load_pack
from walkforward import build_folds
from backtest import build_signals, run_backtest, metrics, bootstrap_ci

t0 = time.time()
m5, m1_pack, spread_cost, monthly = load_data(CFG)
cfg_bal = variant_cfg("v3bal_ens")
folds = build_folds(m5, cfg_bal)
pack = load_pack("v3bal_ens")
lab, valid = pack["lab"], pack["valid"]

with open("/home/z/my-project/scripts/probs_recovered.pkl", "rb") as f:
    rec = pickle.load(f)
thr_rows = json.load(open(os.path.join(BASE, "results", "per_fold_v3bal_ens.json")))

h, l, c = m5["HIGH"], m5["LOW"], m5["CLOSE"]
tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
atr = tr.rolling(288, min_periods=288).mean()
atr_arr = atr.to_numpy()

# 安慰剂属性: 逐行固定随机数 (与ATR无关, 与行绑定)
rng_pl = np.random.default_rng(777)
placibo = rng_pl.standard_normal(len(m5))

QS = [0.0, 0.25, 0.50, 0.65, 0.75]
MIN_VAL_TRADES = 100

def gated_oos(attribute, label):
    logs, chosen_q = [], []
    for i, fold in enumerate(folds):
        d = rec[f"fold{i:02d}"]
        oos_r = np.arange(d["oos_lo"], d["oos_hi"] + 1)
        val_r = np.arange(d["val_lo"], d["val_hi"] + 1)
        thr_l = float(thr_rows[i]["threshold"]["long"])
        thr_s = float(thr_rows[i]["threshold"]["short"])
        sig_val = build_signals(d["pl_val"], d["ps_val"], thr_l, thr_s)
        sig_oos = build_signals(d["pl_oos"], d["ps_oos"], thr_l, thr_s)
        a_val = attribute[val_r]
        a_oos = attribute[oos_r]
        # -- val段选q --
        best_q, best_pnl = 0.0, -np.inf
        for q in QS:
            gate = float(np.quantile(a_val[~np.isnan(a_val)], q)) if q > 0 else -np.inf
            s = np.where(a_val >= gate, sig_val, 0).astype(np.int8)
            lg = run_backtest(s, val_r, lab, valid, m5, m1_pack, cfg_bal)
            if len(lg) < MIN_VAL_TRADES:
                continue
            p = lg["pnl"].sum()
            if p > best_pnl:
                best_pnl, best_q = p, q
        chosen_q.append(best_q)
        # -- OOS段: 冻结的q -> 用val段分布的绝对gate_level (训练窗信息, 实时可复现) --
        gate = float(np.quantile(a_val[~np.isnan(a_val)], best_q)) if best_q > 0 else -np.inf
        s = np.where(a_oos >= gate, sig_oos, 0).astype(np.int8)
        lg = run_backtest(s, oos_r, lab, valid, m5, m1_pack, cfg_bal)
        lg["fold"] = fold["oos_month"]
        lg["gate_q"] = best_q
        logs.append(lg)
    log = pd.concat(logs, ignore_index=True)
    return log, chosen_q

def report(name, log, chosen_q=None):
    m = metrics(log, name)
    pnl = log["pnl"].to_numpy()
    mon = log.groupby("fold")["pnl"].sum()
    ci = bootstrap_ci(pnl, iters=3000)
    print(f"\n== {name} ==")
    print(f"  {m['trades']}笔 WR{m['win_rate']*100:.1f}% PLR{m['plr']:.2f} "
          f"PnL${m['total_pnl']:.1f} 单笔${m['mean_pnl']:.3f} Sharpe{m['sharpe']:.2f} "
          f"maxDD${m['max_dd']:.0f} 盈利月{(mon>0).sum()}/{len(mon)} "
          f"均值CI[{ci[0][0]:.3f},{ci[0][1]:.3f}]")
    print(f"  多头${log.loc[log['dir']=='long','pnl'].sum():.0f} "
          f"空头${log.loc[log['dir']=='short','pnl'].sum():.0f} "
          f"top2占比{mon.nlargest(2).clip(lower=0).sum()/max(pnl.sum(),1e-9):.2f}")
    if chosen_q is not None:
        print(f"  逐折gate_q: {chosen_q}")
    return m

log_atr, q_atr = gated_oos(atr_arr, "ATR")
log_pl, q_pl = gated_oos(placibo, "PLACIBO")

# 未门控对照(=恢复的bal基线)
logs0 = []
for i, fold in enumerate(folds):
    d = rec[f"fold{i:02d}"]
    oos_r = np.arange(d["oos_lo"], d["oos_hi"] + 1)
    thr_l = float(thr_rows[i]["threshold"]["long"])
    thr_s = float(thr_rows[i]["threshold"]["short"])
    sig = build_signals(d["pl_oos"], d["ps_oos"], thr_l, thr_s)
    lg = run_backtest(sig, oos_r, lab, valid, m5, m1_pack, cfg_bal)
    lg["fold"] = fold["oos_month"]
    logs0.append(lg)
log0 = pd.concat(logs0, ignore_index=True)

print("===== 路线D: ATR门控 vs 未门控 vs 安慰剂 (OOS 2024-08~2026-07, 24个月) =====")
m0 = report("未门控基线(恢复)", log0)
ma = report("ATR regime门控", log_atr, q_atr)
mp = report("安慰剂(随机属性同流程)", log_pl, q_pl)

log_atr.to_csv("/home/z/my-project/scripts/gate_atr_trades.csv", index=False)
print(f"\n[gate] 完成, 耗时{time.time()-t0:.0f}s")
