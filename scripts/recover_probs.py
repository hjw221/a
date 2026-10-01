# 路线C前置: 恢复v3bal_ens 24折 OOS/val 概率流(同种子同参数重训, 确定性)
# 目的: 概率流与出场几何无关 -> 一次恢复, 之后任意障碍几何的快测只需重标注+重模拟
# 验证: 用恢复的概率流+原折阈值+原bal标签重跑模拟器, 与 trades_v3bal_ens.csv 逐笔对拍
import os, sys, json, time, pickle
import numpy as np
import pandas as pd

BASE = "/home/z/my-project/download/xauusd_ml_v2"
sys.path.insert(0, BASE)
from config import CFG
from data import load_data
from run_all import variant_cfg, variant_extra, load_pack
from walkforward import build_folds
from variants_ext import train_variant_fold
from backtest import build_signals, run_backtest, metrics

OUT = "/home/z/my-project/scripts/probs_recovered.pkl"

t0 = time.time()
m5, m1_pack, spread_cost, monthly = load_data(CFG)
cfg_v = variant_cfg("v3bal_ens")
folds = build_folds(m5, cfg_v)
pack = load_pack("v3bal_ens")
F, lab, valid = pack["F"], pack["lab"], pack["valid"]
y_long = (lab["out_long"].to_numpy() == 1).astype(np.int8)
y_short = (lab["out_short"].to_numpy() == 1).astype(np.int8)
tuned = json.load(open(os.path.join(BASE, "results", "tuned_params.json")))["v3"]
extra = variant_extra("v3bal_ens")
thr_rows = json.load(open(os.path.join(BASE, "results", "per_fold_v3bal_ens.json")))
print(f"[rec] {len(folds)}折, 首折{folds[0]['oos_month']} 末折{folds[-1]['oos_month']}, "
      f"fit行{len(folds[0]['fit_rows']):,}")

# 断点续跑
done = {}
if os.path.exists(OUT):
    done = pickle.load(open(OUT, "rb"))
    print(f"[rec] 已有 {len(done)} 折缓存")

for i, fold in enumerate(folds):
    key = f"fold{i:02d}"
    if key in done:
        continue
    t1 = time.time()
    tv = train_variant_fold(extra, F, y_long, y_short, fold, tuned, cfg_v, m5, valid, i)
    done[key] = {
        "oos_month": fold["oos_month"],
        "oos_lo": int(fold["oos_rows"][0]), "oos_hi": int(fold["oos_rows"][-1]),
        "val_lo": int(fold["val_rows"][0]), "val_hi": int(fold["val_rows"][-1]),
        "pl_oos": tv["pl_oos"], "ps_oos": tv["ps_oos"],
        "pl_val": tv["pl_val"], "ps_val": tv["ps_val"],
    }
    pickle.dump(done, open(OUT, "wb"), protocol=4)
    print(f"[rec] fold{i:02d} {fold['oos_month']} 完成 ({time.time()-t1:.0f}s, 累计{time.time()-t0:.0f}s)")

# ---- 验证: 原阈值 + 原bal标签 重跑模拟器, 与基线逐笔对拍 ----
if len(done) < len(folds):
    print(f"[rec] 尚未完成: {len(done)}/{len(folds)}折, 续跑本脚本即可")
    sys.exit(0)
logs = []
for i, fold in enumerate(folds):
    d = done[f"fold{i:02d}"]
    thr_l = float(thr_rows[i]["threshold"]["long"])
    thr_s = float(thr_rows[i]["threshold"]["short"])
    oos_r = np.arange(d["oos_lo"], d["oos_hi"] + 1)
    sig = build_signals(d["pl_oos"], d["ps_oos"], thr_l, thr_s)
    log = run_backtest(sig, oos_r, lab, valid, m5, m1_pack, cfg_v,
                       probs=(d["pl_oos"], d["ps_oos"]), tag="recovered")
    log["fold"] = fold["oos_month"]
    logs.append(log)
rec = pd.concat(logs, ignore_index=True)

base = pd.read_csv(os.path.join(BASE, "results", "trades_v3bal_ens.csv"),
                    parse_dates=["signal_time", "entry_time", "exit_time"])
mm_rec, mm_base = metrics(rec, "recovered"), metrics(base, "ml_v3bal_ens")
print("\n===== 恢复验证 =====")
print(f"基线:   {mm_base['trades']}笔 胜率{mm_base['win_rate']*100:.1f}% PLR{mm_base['plr']:.3f} "
      f"PnL${mm_base['total_pnl']:.2f} Sharpe{mm_base['sharpe']:.3f}")
print(f"恢复:   {mm_rec['trades']}笔 胜率{mm_rec['win_rate']*100:.1f}% PLR{mm_rec['plr']:.3f} "
      f"PnL${mm_rec['total_pnl']:.2f} Sharpe{mm_rec['sharpe']:.3f}")
same_rows = (len(rec) == len(base) and (rec["row"].to_numpy() == base["row"].to_numpy()).all())
same_pnl = (len(rec) == len(base) and np.allclose(rec["pnl"].to_numpy(), base["pnl"].to_numpy(), atol=1e-9))
print(f"逐笔行号一致: {same_rows} | PnL逐笔一致: {same_pnl}")
if not same_rows:
    a, b = set(rec["row"]), set(base["row"])
    print(f"  仅恢复: {len(a-b)}笔, 仅基线: {len(b-a)}笔")
print(f"\n[rec] 概率流已保存 -> {OUT} (共{len(done)}折)")
