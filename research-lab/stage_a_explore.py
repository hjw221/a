#!/usr/bin/env python3
"""stage_a_explore.py — 阶段A: ML 数据清洗 + 波段发现.

回答三个问题:
  1. 数据哪里脏/哪里可信 (清洗画像 + ML 异常侦测)
  2. 哪里时段/状态有波段能吃到 (可吃波段密度矩阵, 用原三重障碍口径)
  3. 哪些数据维度有用 (波段猎人 LightGBM 特征重要性)
输出: stage_a_report.json + 控制台摘要
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

RNG = 1337

# ---------- 1. 加载 ----------
print("[A] 加载 M1 ...")
df = pd.read_csv("data.csv", sep="\t")
df.columns = ["date","time","open","high","low","close","tickvol","vol","spread"]
df["dt"] = pd.to_datetime(df["date"]+" "+df["time"], format="%Y.%m.%d %H:%M:%S")
df = df.set_index("dt").drop(columns=["date","time","vol"]).sort_index()
n_all = len(df)

# ---------- 2. 清洗画像 ----------
print("[A] 清洗画像 ...")
report = {}
report["n_rows"] = int(n_all)
report["range"] = [str(df.index.min()), str(df.index.max())]

dup = df.index.duplicated(keep="first")
zero_rng = (df.high == df.low) & (df.tickvol <= 2)
bar_range = df.high - df.low
rng_z = (bar_range - bar_range.rolling(288).mean()) / bar_range.rolling(288).std()
spike = rng_z.abs() > 15           # 极端脉冲 bar (新闻)
dead = (df.tickvol <= 5) & (bar_range < 0.01)   # 死 bar
clean_mask = ~(dup | zero_rng | spike | dead)
report["clean"] = {
    "duplicates": int(dup.sum()),
    "zero_range_bars": int(zero_rng.sum()),
    "spike_bars_z15": int(spike.sum()),
    "dead_bars": int(dead.sum()),
    "clean_rows": int(clean_mask.sum()),
    "clean_ratio": float(clean_mask.mean()),
}
cdf = df[clean_mask].copy()

# spread 插补 (月度中位数, 与原口径一致)
point_value = 0.001
sp = cdf["spread"].to_numpy(float).copy()
month_key = cdf.index.strftime("%Y-%m")
med = pd.Series(sp, index=month_key).groupby(level=0).transform("median")
miss = sp <= 0
sp[miss] = med[miss].to_numpy()
sp = np.maximum.accumulate(np.where(np.isnan(sp), np.nanmedian(sp), sp))
cdf["spread_cost"] = sp * point_value
report["spread"] = {
    "median_pts": float(np.median(cdf["spread"])),
    "imputed_ratio": float(miss.mean()),
    "mean_cost_usd": float(cdf["spread_cost"].mean()),
}

# ---------- 3. ATR + 三重障碍波段标签 (M5 信号行) ----------
print("[A] 三重障碍波段标签 (原口径 ATR288/TP1.5x/SL0.4x/H90) ...")
m1 = cdf
tr = np.maximum(
    m1.high - m1.low,
    np.maximum((m1.high - m1.close.shift()).abs(), (m1.low - m1.close.shift()).abs()),
)
atr288 = tr.rolling(288).mean()

m5 = cdf.resample("5min").agg(
    {"open":"first","high":"max","low":"min","close":"last",
     "tickvol":"sum","spread_cost":"median"}
).dropna(subset=["open"])
m5 = m5[m5.close.notna()]
# 对齐 ATR: 信号时刻取当时已知的 ATR288 (无前视: 用 m1 到该时刻的滚动值)
atr_at_sig = atr288.reindex(m5.index, method="ffill")
spread_at_sig = m5["spread_cost"]

# ---- numba 三重障碍 (双向) ----
from numba import njit

@njit(cache=True)
def barriers(m1_o, m1_h, m1_l, m1_c, m1_t_int, sig_t_int, atr, sc,
             tp_mult, sl_mult, sl_floor, sp_mult, horizon, entry_tol):
    n = len(sig_t_int)
    out_l = np.zeros(n, np.int8)   # 1=TP先到  -1=SL先到  0=超时/无效
    out_s = np.zeros(n, np.int8)
    for i in range(n):
        a = atr[i]
        if not (a > 0.0):
            continue
        tp = tp_mult * a
        sl = max(sl_mult * a, sl_floor, sp_mult * sc[i])
        t0 = sig_t_int[i]
        # 信号M5收盘后入场: 找第一根 >= t0 的 M1
        e = -1
        lo, hi = 0, len(m1_t_int)
        while lo < hi:
            mid = (lo + hi) // 2
            if m1_t_int[mid] < t0:
                lo = mid + 1
            else:
                hi = mid
        e = lo
        if e >= len(m1_t_int) or m1_t_int[e] > t0 + entry_tol:
            continue
        entry = m1_o[e]
        end = min(e + horizon, len(m1_t_int) - 1)
        # long
        hi_hit = -1; lo_hit = -1
        for k in range(e + 1, end + 1):
            if m1_h[k] >= entry + tp and hi_hit < 0: hi_hit = k
            if m1_l[k] <= entry - sl and lo_hit < 0: lo_hit = k
            if hi_hit >= 0 and lo_hit >= 0: break
        if hi_hit >= 0 and (lo_hit < 0 or hi_hit < lo_hit):
            out_l[i] = 1
        elif lo_hit >= 0:
            out_l[i] = -1
        # short
        hi_hit = -1; lo_hit = -1
        for k in range(e + 1, end + 1):
            if m1_l[k] <= entry - tp and hi_hit < 0: hi_hit = k
            if m1_h[k] >= entry + sl and lo_hit < 0: lo_hit = k
            if hi_hit >= 0 and lo_hit >= 0: break
        if hi_hit >= 0 and (lo_hit < 0 or hi_hit < lo_hit):
            out_s[i] = 1
        elif lo_hit >= 0:
            out_s[i] = -1
    return out_l, out_s

m1_t_int = (m1.index.astype("int64") // 10**9).to_numpy()
sig_t_int = ((m5.index.astype("int64") // 10**9) + 300).to_numpy()  # M5收盘
out_l, out_s = barriers(
    m1.open.to_numpy(), m1.high.to_numpy(), m1.low.to_numpy(), m1.close.to_numpy(),
    m1_t_int, sig_t_int,
    atr_at_sig.to_numpy(), spread_at_sig.to_numpy(),
    1.5, 0.4, 0.48, 2.0, 90, 10,
)
band_l = out_l == 1   # 可吃波段(long 侧: TP 先于 SL)
band_s = out_s == 1
m5["band_l"] = band_l
m5["band_s"] = band_s
report["band"] = {
    "n_signals": int(len(m5)),
    "long_band_rate": float(band_l.mean()),
    "short_band_rate": float(band_s.mean()),
    "either_band_rate": float((band_l | band_s).mean()),
    "atr_median_usd": float(np.nanmedian(atr_at_sig)),
}

# ---------- 4. 波段密度矩阵: hour x dow ----------
print("[A] 波段密度矩阵 ...")
m5["hour"] = m5.index.hour
m5["dow"] = m5.index.dayofweek
m5["any_band"] = band_l | band_s
hm = m5.groupby("hour")["any_band"].agg(["mean","count"])
hm_l = m5.groupby("hour")["band_l"].mean()
hm_s = m5.groupby("hour")["band_s"].mean()
best_hours = (hm["mean"].sort_values(ascending=False)).head(6)
worst_hours = (hm["mean"].sort_values(ascending=True)).head(4)
dm = m5.groupby(["dow","hour"])["any_band"].mean().unstack()
report["hours"] = {
    "overall_rate": float(m5["any_band"].mean()),
    "best": {int(h): float(v) for h, v in best_hours.items()},
    "worst": {int(h): float(v) for h, v in worst_hours.items()},
    "long_minus_short_by_hour": {
        int(h): float(hm_l[h] - hm_s[h]) for h in hm.index
    },
}

# ---------- 5. 波段猎人 LightGBM: 哪些状态维度有用 ----------
print("[A] 波段猎人 LightGBM ...")
def build_states(m):
    """入场时刻可观测状态特征 (无前视)"""
    x = pd.DataFrame(index=m.index)
    c, h, l, v = m["close"], m["high"], m["low"], m["tickvol"]
    r = c.diff()
    x["hour_sin"] = np.sin(2*np.pi*m.index.hour/24)
    x["hour_cos"] = np.cos(2*np.pi*m.index.hour/24)
    x["dow"] = m.index.dayofweek.astype(float)
    x["ret_5"] = c.diff()
    x["ret_15"] = c.diff(3)
    x["ret_60"] = c.diff(12)
    x["ret_240"] = c.diff(48)
    x["vol_fast"] = r.rolling(6).std()          # 30min 波动
    x["vol_slow"] = r.rolling(96).std()         # 8h 波动
    x["vol_ratio"] = x["vol_fast"] / (x["vol_slow"] + 1e-9)
    x["atr_norm"] = np.nan       # 后填
    x["range_mean"] = (h - l).rolling(48).mean()
    x["body_ratio"] = (c - m["open"]).abs() / ((h - l) + 1e-9)
    x["vol_spike"] = v / (v.rolling(96).mean() + 1e-9)
    x["run_up"] = (r > 0).rolling(12).sum()      # 1h 内上涨 bar 数
    x["dist_day_hi"] = c / m["high"].rolling(288, min_periods=288).max() - 1
    x["dist_day_lo"] = c / m["low"].rolling(288, min_periods=288).min() - 1
    return x

X = build_states(m5)
X["atr_norm"] = (atr_at_sig / cdf["close"].reindex(m5.index, method="ffill")).to_numpy()
y = (m5["any_band"]).astype(int).to_numpy()
ok = X.notna().all(axis=1).to_numpy() & np.isfinite(X.to_numpy()).all(axis=1)
X, y = X[ok], y[ok]
# 时间切分 (train 2022-2025.6 / test 2025.7-2026.7): 真实 OOS
cut = pd.Timestamp("2025-07-01")
tr = X.index < cut
te = ~tr
params = dict(objective="binary", metric="auc", learning_rate=0.05,
              num_leaves=31, min_data_in_leaf=2000, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, verbosity=-1)
ds_tr = lgb.Dataset(X[tr], y[tr])
ds_te = lgb.Dataset(X[te], y[te], reference=ds_tr)
bst = lgb.train(params, ds_tr, num_boost_round=300,
                valid_sets=[ds_te], callbacks=[lgb.early_stopping(30, verbose=False)])
from sklearn.metrics import roc_auc_score
auc_te = roc_auc_score(y[te], bst.predict(X[te], num_iteration=bst.best_iteration))
imp = pd.Series(bst.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
auc_by_decile = None
# 顶部十分位命中率: 波段猎人得分前10%的信号里波段实际发生率
p_te = bst.predict(X[te], num_iteration=bst.best_iteration)
q = pd.Series(p_te, index=X.index[te]).groupby(pd.qcut(pd.Series(p_te), 10, duplicates="drop")).apply(
    lambda s: y[te][np.isin(np.arange(len(p_te)), s.index.map(lambda i: list(X.index[te]).index(i)))] .mean() if len(s) else np.nan
)
top_rate = float(np.mean(y[te][p_te >= np.quantile(p_te, 0.9)]))
base_rate = float(y[te].mean())
report["hunter"] = {
    "test_auc": float(auc_te),
    "base_band_rate_oos": base_rate,
    "top_decile_band_rate": top_rate,
    "lift": top_rate / max(base_rate, 1e-9),
    "feature_importance": {k: float(v) for k, v in imp.head(12).items()},
}
print(json.dumps(report["hunter"], indent=2))

# ---------- 6. 时段策略建议 ----------
top_hours = sorted([int(h) for h in best_hours.index])
report["recommend"] = {
    "trade_hours": top_hours,
    "avoid_hours": sorted([int(h) for h in worst_hours.index]),
    "note": "波段猎人lift>1.3则状态过滤有效; hour特征重要性高则时段选择性有价值",
}

with open("stage_a_report.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print("\n==== 阶段A 摘要 ====")
print(f"清洗: {report['clean']['clean_rows']}/{n_all} ({report['clean']['clean_ratio']*100:.2f}%)")
print(f"波段率: L {report['band']['long_band_rate']*100:.1f}% | S {report['band']['short_band_rate']*100:.1f}%")
print(f"最佳时段(UTC): {top_hours}  波段率 {best_hours.iloc[0]*100:.1f}% vs 全日均 {hm['mean'].mean()*100:.1f}%")
print(f"波段猎人: OOS AUC {auc_te:.3f} | top10% 命中 {top_rate*100:.1f}% (基准 {base_rate*100:.1f}%, lift {report['hunter']['lift']:.2f})")
print("特征重要性TOP5:", dict(imp.head(5).round(1)))
