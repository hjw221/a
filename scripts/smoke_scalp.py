"""
Task14-D 冒烟: ① 特征因果性 ② 标签引擎正确性抽查 ③ M1规模LGB/XGB耗时基准
"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/z/my-project/download/xauusd_ml_v2")
from data import load_raw_m1, impute_spread
from features_s1 import build_features_s1
from scalp import label_all_m1

CSV = "/home/z/my-project/upload/5_extracted/XAUUSDc_M1_202201022305_202606262057.csv"

# ---- ① 特征因果性: 截断60% vs 全量, 重叠区必须完全一致 ----
print("[smoke] 加载M1 ...")
m1 = load_raw_m1(CSV)
cut = int(len(m1) * 0.6)
F_full, feats, _ = build_features_s1(m1)
F_trunc, _, _ = build_features_s1(m1.iloc[:cut])
arr_f = F_full.to_numpy()
arr_t = F_trunc.to_numpy()
diff = 0
for j in range(arr_f.shape[1]):
    a, b = arr_f[:cut, j], arr_t[:, j]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        continue
    if not np.allclose(a[m], b[m], equal_nan=True):
        diff += 1
        print(f"  不一致列: {feats[j]}")
n_nan_full = np.isnan(arr_f[cut - 50000:cut]).mean()
print(f"[smoke] ① 因果性: {F_full.shape[1]}特征中不一致列={diff} (必须0), "
      f"截断区NaN率{n_nan_full*100:.1f}% (滚动窗热身正常)")

# ---- ② 标签引擎抽查: 随机抽30个事件手工复核 ----
spread_cost, _ = impute_spread(m1, 0.001)
t = (m1.index.astype("int64") // 10**9 // 60).to_numpy(np.int64)
o = m1["OPEN"].to_numpy(np.float64)
h = m1["HIGH"].to_numpy(np.float64)
l = m1["LOW"].to_numpy(np.float64)
c = m1["CLOSE"].to_numpy(np.float64)
TP, SL, HZ, FLOOR = 1.00, 0.40, 30, 3.0
res = label_all_m1(t, o, h, l, c, spread_cost, TP, SL, FLOOR, HZ, 10)
entry_idx, out_l, out_s, pnl_l, pnl_s, xb_l, xb_s, tp_d, sl_d = res
rng = np.random.default_rng(7)
bad = 0
for i in rng.choice(np.arange(10000, len(t) - 500), 30, replace=False):
    e = entry_idx[i]
    if e < 0:
        continue
    assert e == i + 1 or t[e] - t[i] <= 11, f"入场不是下一根: i={i} e={e}"
    sc = spread_cost[i]
    sl_eff = max(SL, FLOOR * sc)
    assert abs(tp_d[i] - TP) < 1e-12 and abs(sl_d[i] - sl_eff) < 1e-12
    entry = o[e]
    # 手工扫描
    l_tp, l_sl = entry + TP, entry - sl_eff
    hit, jb = 0, -1
    for j in range(e + 1, min(e + HZ + 1, len(t))):
        if l[j] <= l_sl: hit, jb = -1, j; break     # SL优先(悲观)
        if h[j] >= l_tp: hit, jb = 1, j; break
    if hit == 0:
        j = min(e + HZ, len(t) - 1)
        pl_manual = (c[j] - entry) - sc
    elif hit == 1:
        pl_manual = TP - sc
    else:
        pl_manual = -sl_eff - sc
    if out_l[i] != hit or abs(pnl_l[i] - pl_manual) > 1e-9 or xb_l[i] != jb:
        bad += 1
        print(f"  MISMATCH i={i}: 引擎({out_l[i]},{pnl_l[i]:.4f},{xb_l[i]}) 手工({hit},{pl_manual:.4f},{jb})")
print(f"[smoke] ② 标签引擎抽查30事件: 不一致={bad} (必须0)")

# ---- ③ 耗时基准: 730k行×37特征 LGB(200轮)/XGB(100轮) ----
import lightgbm as lgb, xgboost as xgb
n = 730_000
X = F_full.to_numpy(np.float32)[:n]
y = (np.random.default_rng(3).random(n) < 0.35).astype(np.int8)
w = np.ones(n)
t0 = time.time()
lgb.train({"objective": "binary", "num_leaves": 63, "min_data_in_leaf": 200,
           "learning_rate": 0.05, "num_threads": 2, "verbosity": -1},
          lgb.Dataset(X, label=y, weight=w), num_boost_round=200)
t_lgb = time.time() - t0
t0 = time.time()
xgb.train({"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
           "max_depth": 6, "eta": 0.05, "nthread": 2, "verbosity": 0},
          xgb.DMatrix(X, label=y, weight=w), num_boost_round=100)
t_xgb = time.time() - t0
print(f"[smoke] ③ 耗时: LGB 730k×37 200轮={t_lgb:.0f}s | XGB 100轮={t_xgb:.0f}s "
      f"=> 12成员/折 约 {(t_lgb*3+t_xgb*3):.0f}s (估)")
