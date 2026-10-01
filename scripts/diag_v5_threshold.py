"""诊断: v5的wr_freq阈值校准在内部验证段上的完整候选前沿。
重放3个代表折(首/中/末), 重拟合模型后在val段扫全部q, 打印:
q, 交易数, 频率(笔/天), 胜率, 盈亏比, 总PnL, mean×sqrtN —— 看清约束可达性。"""
import sys, json, time
sys.path.insert(0, "/home/z/my-project/download/xauusd_ml_v2")
import numpy as np
import pandas as pd
from config import CFG
from data import load_data
from walkforward import build_folds, time_decay_weights
from models import fit_lgb, fit_xgb, predict_ens
from backtest import build_signals, run_backtest, metrics

BASE = "/home/z/my-project/download/xauusd_ml_v2"
import pickle
pack = pickle.load(open(f"{BASE}/cache/pack_v5.pkl", "rb"))
F, lab, valid = pack["F"], pack["lab"], pack["valid"]
tuned = json.load(open(f"{BASE}/results/tuned_params.json"))["v3"]

cfg = dict(CFG)
cfg.update(CFG["wr40_geometry"])           # wr40几何(含cooldown5/wr_floor/freq_target)
cfg["threshold_objective"] = "wr_freq"

m5, m1_pack, spread_cost, monthly = load_data(cfg)
folds = build_folds(m5, cfg)

y_long = (lab["out_long"].to_numpy() == 1).astype(np.int8)
y_short = (lab["out_short"].to_numpy() == 1).astype(np.int8)

QS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
for i in [0, 12, 23]:
    fold = folds[i]
    fit_r, val_r = fold["fit_rows"], fold["val_rows"]
    w_fit = time_decay_weights(m5.index, fit_r, cfg["sample_halflife_days"])
    Xf, Xv = F.iloc[fit_r].to_numpy(np.float32), F.iloc[val_r].to_numpy(np.float32)
    models = {}
    for direction, y in [("long", y_long), ("short", y_short)]:
        m_lgb, _ = fit_lgb(Xf, y[fit_r], w_fit, tuned[direction]["lgb"], Xv, y[val_r],
                           np.ones(len(val_r)), es_rounds=50, max_rounds=600)
        m_xgb, _ = fit_xgb(Xf, y[fit_r], w_fit, tuned[direction]["xgb"], Xv, y[val_r],
                           np.ones(len(val_r)), es_rounds=50, max_rounds=600)
        models[direction] = [("lgb", m_lgb), ("xgb", m_xgb)]
    pl = predict_ens(models["long"], Xv)
    ps = predict_ens(models["short"], Xv)
    print(f"\n===== fold {i} {fold['oos_month']} 内部验证段候选前沿 (真实回放) =====")
    print(f"{'q':>5} {'笔数':>6} {'笔/天':>6} {'胜率':>6} {'盈亏比':>7} {'总PnL$':>9} {'mean×√N':>8}")
    for q in QS:
        thr_l, thr_s = float(np.quantile(pl, q)), float(np.quantile(ps, q))
        sig = build_signals(pl, ps, thr_l, thr_s)
        log = run_backtest(sig, val_r, lab, valid, m5, m1_pack, cfg)
        n = len(log)
        if n == 0:
            print(f"{q:5.2f} {'0':>6}  (无交易)")
            continue
        mm = metrics(log)
        days = max(1, int(log["signal_time"].dt.normalize().nunique()))
        print(f"{q:5.2f} {n:6d} {n/days:6.1f} {mm['win_rate']*100:5.1f}% "
              f"{mm['plr']:7.2f} {log['pnl'].sum():9.1f} {log['pnl'].mean()*np.sqrt(n):8.2f}")
    # 信号可用率: val段上 p>thr 的bar占比 (频率瓶颈判断)
    for q in [0.50, 0.70]:
        thr_l, thr_s = float(np.quantile(pl, q)), float(np.quantile(ps, q))
        rate = ((pl > thr_l) | (ps > thr_s)).mean()
        print(f"  [信号可用率 q={q:.2f}] {rate*100:.1f}% 的M5 bar有信号"
              f" (~{rate*271:.0f}信号/天, 单持仓执行受占用率限制)")
