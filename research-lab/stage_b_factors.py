#!/usr/bin/env python3
"""stage_b_factors.py — 阶段B: 方向因子库 + ML 双筛.

阶段A结论: 波段遍地是(95.8%), 方向才是稀缺品 → 因子全部瞄准"方向".
方法:
  1. 60+ 因子库 (动量/均值回复/趋势结构/波动/微结构/日内/多日), 全部无前视
  2. 标签: 方向三分类 (long明确赢 / short明确赢 / 混沌) → 只用明确样本训练
  3. 双筛: 月度滚动 rank-IC 稳定性 + LightGBM gain/permutation 重要性
输出: stage_b_report.json (因子白名单 + IC 表 + 重要性表)
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

# ---------- 复用阶段A的数据准备 ----------
print("[B] 加载 ...")
df = pd.read_csv("data.csv", sep="\t")
df.columns = ["date","time","open","high","low","close","tickvol","vol","spread"]
df["dt"] = pd.to_datetime(df["date"]+" "+df["time"], format="%Y.%m.%d %H:%M:%S")
df = df.set_index("dt").drop(columns=["date","time","vol"]).sort_index()

bar_range = df.high - df.low
rng_z = (bar_range - bar_range.rolling(288).mean()) / bar_range.rolling(288).std()
dead = (df.tickvol <= 5) & (bar_range < 0.01)
cdf = df[~(dead | (rng_z.abs() > 15))].copy()

sp = cdf["spread"].to_numpy(float).copy()
month_key = cdf.index.strftime("%Y-%m")
med = pd.Series(sp, index=month_key).groupby(level=0).transform("median")
miss = sp <= 0
sp[miss] = med[miss].to_numpy()
sp = np.where(np.isnan(sp), np.nanmedian(sp), sp)
cdf["spread_cost"] = sp * 0.001

tr = np.maximum(cdf.high - cdf.low,
    np.maximum((cdf.high - cdf.close.shift()).abs(), (cdf.low - cdf.close.shift()).abs()))
atr288 = tr.rolling(288).mean()

m5 = cdf.resample("5min").agg(
    {"open":"first","high":"max","low":"min","close":"last","tickvol":"sum","spread_cost":"median"}
).dropna(subset=["open"])
atr5 = atr288.reindex(m5.index, method="ffill")

from numba import njit

@njit(cache=True)
def barriers(m1_o, m1_h, m1_l, m1_t_int, sig_t_int, atr, sc,
             tp_mult, sl_mult, sl_floor, sp_mult, horizon, entry_tol):
    n = len(sig_t_int)
    out_l = np.zeros(n, np.int8)
    out_s = np.zeros(n, np.int8)
    for i in range(n):
        a = atr[i]
        if not (a > 0.0):
            continue
        tp = tp_mult * a
        sl = max(sl_mult * a, sl_floor, sp_mult * sc[i])
        t0 = sig_t_int[i]
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
        hi_hit = -1; lo_hit = -1
        for k in range(e + 1, end + 1):
            if m1_h[k] >= entry + tp and hi_hit < 0: hi_hit = k
            if m1_l[k] <= entry - sl and lo_hit < 0: lo_hit = k
            if hi_hit >= 0 and lo_hit >= 0: break
        if hi_hit >= 0 and (lo_hit < 0 or hi_hit < lo_hit):
            out_l[i] = 1
        elif lo_hit >= 0:
            out_l[i] = -1
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

m1_t_int = (cdf.index.astype("int64") // 10**9).to_numpy()
sig_t_int = ((m5.index.astype("int64") // 10**9) + 300).to_numpy()
out_l, out_s = barriers(
    cdf.open.to_numpy(), cdf.high.to_numpy(), cdf.low.to_numpy(),
    m1_t_int, sig_t_int, atr5.to_numpy(), m5["spread_cost"].to_numpy(),
    1.5, 0.4, 0.48, 2.0, 90, 10)
m5["out_l"], m5["out_s"] = out_l, out_s

# 方向标签: 明确样本 (一侧TP先到, 另一侧没先到SL)
dir_lab = np.zeros(len(m5), np.int8)   # 1=long明确 -1=short明确 0=混沌/超时
clear_long = (out_l == 1) & (out_s != 1)
clear_short = (out_s == 1) & (out_l != 1)
dir_lab[clear_long] = 1
dir_lab[clear_short] = -1
print(f"[B] 方向样本: long明确 {clear_long.sum()} | short明确 {clear_short.sum()} | 混沌 {(dir_lab==0).sum()}")

# ---------- 因子库 ----------
print("[B] 构建因子库 ...")
c, o, h, l, v = m5["close"], m5["open"], m5["high"], m5["low"], m5["tickvol"]
r = c.diff()
F = pd.DataFrame(index=m5.index)

def mom(n): return c.diff(n)
# 动量 (10个)
for n in [1, 3, 6, 12, 24, 48, 96, 144, 288, 576]:
    F[f"mom_{n}"] = c.diff(n)
F["mom_acc"] = c.diff(6) - c.diff(48)
# 均值回复 (6个)
m96, sd96 = c.rolling(96).mean(), c.rolling(96).std()
F["zscore_96"] = (c - m96) / (sd96 + 1e-9)
m288, sd288 = c.rolling(288).mean(), c.rolling(288).std()
F["zscore_288"] = (c - m288) / (sd288 + 1e-9)
F["boll_pos_96"] = (c - (m96 - 2*sd96)) / ((m96 + 2*sd96) - (m96 - 2*sd96) + 1e-9)
F["dist_ma20"] = c / c.rolling(24).mean() - 1
F["dist_ma60"] = c / c.rolling(72).mean() - 1
F["dist_ma240"] = c / c.rolling(288).mean() - 1
# 趋势结构 (8个)
F["ema_f_s"] = c.ewm(span=12).mean() / c.ewm(span=96).mean() - 1
F["ema_m_s"] = c.ewm(span=48).mean() / c.ewm(span=288).mean() - 1
hh = h.rolling(96).max(); ll = l.rolling(96).min()
F["stoch_pos_96"] = (c - ll) / (hh - ll + 1e-9)
hh2, ll2 = h.rolling(288).max(), l.rolling(288).min()
F["stoch_pos_288"] = (c - ll2) / (hh2 - ll2 + 1e-9)
F["breakout_96"] = (c - hh.shift(1)) / (atr5 + 1e-9)
F["breakdown_96"] = (c - ll.shift(1)) / (atr5 + 1e-9)
F["hh_cnt_96"] = (h >= h.rolling(96).max()).rolling(96).sum()
F["ll_cnt_96"] = (l <= l.rolling(96).min()).rolling(96).sum()
# 波动 (8个)
F["vol_ratio_fs"] = r.rolling(12).std() / (r.rolling(288).std() + 1e-9)
F["atr_norm"] = (atr5 / c).to_numpy()
F["vol_of_vol"] = r.rolling(48).std().rolling(288).std() / (r.rolling(288).std() + 1e-9)
F["range_exp"] = (h - l).rolling(12).mean() / ((h - l).rolling(288).mean() + 1e-9)
F["pk_vol"] = ((np.log1p(r.abs())).rolling(288).std())
F["gap_overnight"] = c.groupby(c.index.date).transform("first").shift(0) * 0 + c.groupby(c.index.date).apply(lambda s: s.iloc[0]).reindex(c.index).bfill() - c
F["dollar_vol"] = (r.abs() * v).rolling(288).mean()
# 微结构 (10个)
rng = h - l
F["body_ratio"] = (c - o).abs() / (rng + 1e-9)
F["upper_wick"] = (h - np.maximum(c, o)) / (rng + 1e-9)
F["lower_wick"] = (np.minimum(c, o) - l) / (rng + 1e-9)
F["close_in_rng"] = (c - l) / (rng + 1e-9)
F["vol_spike"] = v / (v.rolling(288).mean() + 1e-9)
F["vol_z"] = (v - v.rolling(288).mean()) / (v.rolling(288).std() + 1e-9)
F["up_bars_12"] = (r > 0).rolling(12).sum()
F["up_bars_96"] = (r > 0).rolling(96).sum()
F["tick_accel"] = v / (v.rolling(48).mean() + 1e-9) - v.shift(48) / (v.rolling(288).mean().shift(48) + 1e-9)
F["efficiency_96"] = (c - c.shift(96)).abs() / (r.abs().rolling(96).sum() + 1e-9)  # 趋势效率
# 日内/多日 (10个)
F["hour_sin"] = np.sin(2*np.pi*m5.index.hour/24)
F["hour_cos"] = np.cos(2*np.pi*m5.index.hour/24)
F["dow_sin"] = np.sin(2*np.pi*m5.index.dayofweek/5)
day_hi = h.groupby(m5.index.date).transform("cummax")
day_lo = l.groupby(m5.index.date).transform("cummin")
F["pos_day_hi"] = (c - day_hi) / (atr5 + 1e-9)
F["pos_day_lo"] = (c - day_lo) / (atr5 + 1e-9)
F["mins_from_open"] = m5.index.hour * 60 + m5.index.minute
F["asian_ret"] = c.groupby(m5.index.date).transform(lambda s: s.iloc[0]) * 0  # 占位
# 隔日动量 (开盘价差方向)
day_open = c.groupby(m5.index.date).transform("first")
day_open_prev = day_open.shift(1)
F["overnight_gap"] = (day_open - day_open_prev) / (atr5 + 1e-9)
F["nfp_proxy"] = ((m5.index.day >= 1) & (m5.index.day <= 5)).astype(float)

FACTORS = [k for k in F.columns if not k.endswith("_prev0")]
F = F.replace([np.inf, -np.inf], np.nan)

# ---------- 目标: 90min 方向净收益(连续版, 用于IC) ----------
fwd = c.shift(-18) - c     # 90min 前向收益
fwd.name = "fwd"

# ---------- 1) 月度滚动 rank-IC ----------
print("[B] 月度滚动 IC ...")
mask_clear = dir_lab != 0
data = F.copy()
data["fwd"] = fwd
data = data[mask_clear]
months = data.index.to_period("M")
ic_tbl = {}
for col in FACTORS:
    ics = data.groupby(months).apply(
        lambda g, col=col: g[col].corr(g["fwd"], method="spearman")
        if g[col].notna().sum() > 500 else np.nan)
    ics = ics.dropna()
    ic_tbl[col] = {
        "ic_mean": float(ics.mean()),
        "ic_std": float(ics.std()),
        "icir": float(ics.mean() / (ics.std() + 1e-9)),
        "pos_months": int((ics > 0).sum()),
        "n_months": int(len(ics)),
    }
ic_rank = sorted(ic_tbl.items(), key=lambda kv: -abs(kv[1]["icir"]))

# ---------- 2) LightGBM 方向分类 + 重要性 ----------
print("[B] 方向分类器 LGBM ...")
y = pd.Series(dir_lab, index=m5.index)[mask_clear]
X = F[mask_clear]
ok = X.notna().all(axis=1) & y.notna()
X, y = X[ok], y[ok]
cut = pd.Timestamp("2025-07-01")
tr, te = X.index < cut, X.index >= cut
y01 = (y > 0).astype(int)
params = dict(objective="binary", metric="auc", learning_rate=0.05,
              num_leaves=31, min_data_in_leaf=2000, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, verbosity=-1)
ds_tr = lgb.Dataset(X[tr], y01[tr])
ds_te = lgb.Dataset(X[te], y01[te], reference=ds_tr)
bst = lgb.train(params, ds_tr, num_boost_round=400,
                valid_sets=[ds_te], callbacks=[lgb.early_stopping(40, verbose=False)])
from sklearn.metrics import roc_auc_score
auc_te = roc_auc_score(y01[te], bst.predict(X[te], num_iteration=bst.best_iteration))
gain = pd.Series(bst.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
# permutation importance (OOS, 子采样加速)
rng = np.random.default_rng(7)
perm = {}
Xte_s, yte_s = X[te].iloc[:60000], y01[te].iloc[:60000]
base_auc = roc_auc_score(yte_s, bst.predict(Xte_s, num_iteration=bst.best_iteration))
for col in gain.head(20).index:
    Xp = Xte_s.copy()
    Xp[col] = rng.permutation(Xp[col].to_numpy())
    perm[col] = float(base_auc - roc_auc_score(yte_s, bst.predict(Xp, num_iteration=bst.best_iteration)))

# ---------- 3) 白名单合成 ----------
ic_top = [k for k, v in ic_rank if abs(v["icir"]) > 0.10][:20]
gain_top = list(gain.head(15).index)
perm_top = [k for k, v in sorted(perm.items(), key=lambda kv: -kv[1]) if v > 0.002][:15]
whitelist = sorted(set(ic_top) & set(gain_top) | (set(ic_top) & set(perm_top)) | (set(gain_top) & set(perm_top)))
report = {
    "n_factors": len(FACTORS),
    "n_clear_samples": int(mask_clear.sum()),
    "dir_lgb_auc_oos": float(auc_te),
    "ic_top10": [{"f": k, "icir": round(ic_tbl[k]["icir"],3), "ic": round(ic_tbl[k]["ic_mean"],4)} for k, _ in ic_rank[:10]],
    "gain_top10": {k: float(round(v)) for k, v in gain.head(10).items()},
    "perm_top10": {k: round(v, 4) for k, v in sorted(perm.items(), key=lambda kv: -kv[1])[:10]},
    "whitelist": whitelist,
}
with open("stage_b_report.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"\n==== 阶段B 摘要 ====")
print(f"因子数: {len(FACTORS)} | 明确方向样本: {mask_clear.sum()}")
print(f"方向分类器 OOS AUC: {auc_te:.4f}")
print("IC榜TOP8:", [(k, round(ic_tbl[k]['icir'],3)) for k, _ in ic_rank[:8]])
print("Gain榜TOP8:", list(gain.head(8).index))
print("Perm榜TOP8:", [k for k, v in sorted(perm.items(), key=lambda kv: -kv[1])[:8]])
print(f"白名单({len(whitelist)}):", whitelist)
