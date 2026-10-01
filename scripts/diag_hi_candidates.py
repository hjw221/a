"""诊断: v3bal_hi 校准候选表 — 内部验证段(训练窗尾20%)上,
不同分位q × 冷却(5/10)下的 真实交易数/胜率/PLR/总PnL。
只读训练窗数据, 不触碰OOS。用法: python scripts/diag_hi_candidates.py [fold_idx]
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd

BASE = "/home/z/my-project/download/xauusd_ml_v2"
sys.path.insert(0, BASE)
os.chdir(BASE)
from config import CFG
from data import load_data
from features_v3 import build_features_v3
from walkforward import build_folds, time_decay_weights
from models import fit_lgb, fit_xgb, predict_ens
from backtest import build_signals, run_backtest, metrics
from run_all import variant_cfg, load_pack

fold_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

m5, m1_pack, spread_cost, monthly = load_data(CFG)
cfg_v = variant_cfg("v3bal")            # 几何与v3bal完全一致
folds = build_folds(m5, cfg_v)
fold = folds[fold_idx]
print(f"折{fold_idx} OOS月={fold['oos_month']} 训练窗={fold['window']} 内部val段={len(fold['val_rows'])}根M5")

pack = load_pack("v3bal")
F, lab, valid = pack["F"], pack["lab"], pack["valid"]
tuned = json.load(open(f"{BASE}/results/tuned_params.json"))["v3"]
y_long = (lab["out_long"].to_numpy() == 1).astype(np.int8)
y_short = (lab["out_short"].to_numpy() == 1).astype(np.int8)

fit_r, val_r = fold["fit_rows"], fold["val_rows"]
w_fit = time_decay_weights(m5.index, fit_r, CFG["sample_halflife_days"])
w_val = np.ones(len(val_r))
Xf = F.iloc[fit_r].to_numpy(np.float32)
Xv = F.iloc[val_r].to_numpy(np.float32)
models = {}
for direction, y in [("long", y_long), ("short", y_short)]:
    yf, yv = y[fit_r], y[val_r]
    m_lgb, _ = fit_lgb(Xf, yf, w_fit, tuned[direction]["lgb"], Xv, yv, w_val,
                       es_rounds=CFG["early_stopping_rounds"], max_rounds=CFG["max_rounds"])
    m_xgb, _ = fit_xgb(Xf, yf, w_fit, tuned[direction]["xgb"], Xv, yv, w_val,
                       es_rounds=CFG["early_stopping_rounds"], max_rounds=CFG["max_rounds"])
    models[direction] = [("lgb", m_lgb), ("xgb", m_xgb)]
pl = predict_ens(models["long"], Xv)
ps = predict_ens(models["short"], Xv)
print(f"val段概率: long 均值{pl.mean():.3f} 分位[0.5,0.8,0.95,0.985]="
      f"{np.quantile(pl,[0.5,0.8,0.95,0.985]).round(3)} | short 均值{ps.mean():.3f}")

QS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.985]
for cd in [5, 10, 0]:
    cfg_c = dict(cfg_v); cfg_c["cooldown_m1"] = cd
    rows = []
    for q in QS:
        thr_l, thr_s = float(np.quantile(pl, q)), float(np.quantile(ps, q))
        sig = build_signals(pl, ps, thr_l, thr_s)
        log = run_backtest(sig, val_r, lab, valid, m5, m1_pack, cfg_c)
        if len(log) == 0:
            rows.append((q, 0, np.nan, np.nan, 0.0)); continue
        mm = metrics(log)
        rows.append((q, len(log), mm["win_rate"], mm["plr"], mm["total_pnl"]))
    print(f"\n===== 冷却{cd}分钟 (val段 {pd.Period(m5.index[val_r[0]],'M')}~{pd.Period(m5.index[val_r[-1]],'M')}) =====")
    print(f"{'q':>5} {'笔数':>6} {'笔/天':>6} {'胜率':>7} {'PLR':>6} {'总PnL$':>9}")
    ndays = np.busday_count(m5.index[val_r[0]].date(), m5.index[val_r[-1]].date()) + 1
    for q, n, wr, plr_, pnl in rows:
        wr_s = f"{wr*100:6.1f}%" if np.isfinite(wr) else "   n/a "
        pl_s = f"{plr_:6.2f}" if np.isfinite(plr_) else "   n/a"
        print(f"{q:5.2f} {n:6d} {n/max(ndays,1):6.1f} {wr_s} {pl_s} {pnl:9.1f}")
