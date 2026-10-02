"""Stage A — ML 数据清洗 + 特征工程 + 波段可吃性分析 (2026-10-01).

用户指令: "重新用机器学习来清洗数据构建特征工程先找出哪些数据有用有波段能吃到"
三模块:
  M1 数据剖析: 34特征分布/缺失/漂移/冗余簇 + 数据完整性(月bar数)
  M2 波段可吃性: 逐bar最优方向PnL -> 可吃标注 -> 11维regime切片 -> Oracle天花板
  M3 特征信息量: LGB gain(4折时间CV, 双目标) + 互信息 + 月度Spearman IC -> 综合排名
输出: /root/rivermind-data/xauusd/stage_a/ (csv + json + ntfy汇报)
复用: pack_m1sc_ad.pkl (特征+标签缓存, 与历史管线完全同口径)
"""
import os, sys, json, time, pickle
import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import spearmanr

T0 = time.time()
def log(s): print(f"[A {time.time()-T0:6.0f}s] {s}", flush=True)

BASE_V2 = "/root/rivermind-data/xauusd/v2_ens"
OUT = "/root/rivermind-data/xauusd/stage_a"
os.makedirs(OUT, exist_ok=True)
CSV = "/root/rivermind-fs/xauusd/data/XAUUSDc_M1_202201022305_202606262057.csv"
NTFY = "xauusd-qv7m2zk9-res"
HIST_TARGET = 732.7

# ---------------------------------------------------------------- 加载
log("加载 pack_m1sc_ad.pkl ...")
with open(os.path.join(BASE_V2, "cache", "pack_m1sc_ad.pkl"), "rb") as f:
    pack = pickle.load(f)
F, feats = pack["F"], pack["feats"]
lab, valid = pack["lab"], pack["valid"]
spread_cost = pack["spread_cost"]
log(f"pack: F{F.shape} valid={valid.sum():,}/{len(valid):,}")

log("轻量重建时间索引 (只读DATE/TIME两列) ...")
dt = pd.read_csv(CSV, sep="\t", usecols=["<DATE>", "<TIME>"])
uniq = dt["<DATE>"].unique()
dmap = {d: pd.to_datetime(d, format="%Y.%m.%d") for d in uniq}
dates = dt["<DATE>"].map(dmap)
secs = dt["<TIME>"].str.slice(0, 2).astype(np.int32) * 3600 + \
       dt["<TIME>"].str.slice(3, 5).astype(np.int32) * 60
idx_s = (dates + pd.to_timedelta(secs, unit="s")).sort_values()
idx = pd.DatetimeIndex(idx_s.to_numpy())
assert len(idx) == len(F), f"index len {len(idx)} != F {len(F)}"
del dt, dates, secs
log(f"时间索引 {idx[0]} ~ {idx[-1]}")

# ---------------------------------------------------------------- M1 数据剖析
log("M1: 特征剖析 (分布/缺失/半年段漂移) ...")
seg_ids, seg_uniq = pd.factorize(pd.PeriodIndex(idx, freq="2Q"))
prof_rows = []
for c in feats:
    x = F[c].to_numpy(np.float64)
    v = x[valid]
    seg_means, seg_nans = [], []
    for s in range(len(seg_uniq)):
        m = valid & (seg_ids == s)
        if m.sum() < 1000:
            continue
        xs = x[m]
        seg_means.append(np.nanmean(xs) if np.isfinite(xs).any() else 0.0)
        seg_nans.append(float(np.mean(~np.isfinite(xs))))
    seg_means = np.array(seg_means)
    gstd = np.nanstd(v)
    drift = float(np.std(seg_means) / (gstd + 1e-12))
    prof_rows.append({
        "feature": c, "nan_rate": float(np.mean(~np.isfinite(x))),
        "n_unique": int(len(np.unique(v[np.isfinite(v)]))),
        "std": float(gstd),
        "q01": float(np.nanquantile(v, 0.01)), "q25": float(np.nanquantile(v, 0.25)),
        "q50": float(np.nanquantile(v, 0.50)), "q75": float(np.nanquantile(v, 0.75)),
        "q99": float(np.nanquantile(v, 0.99)),
        "drift_score": drift, "seg_nan_max": float(max(seg_nans)) if seg_nans else 0.0,
    })
prof = pd.DataFrame(prof_rows).sort_values("drift_score", ascending=False)
prof.to_csv(os.path.join(OUT, "feature_profile.csv"), index=False)

log("M1: 冗余簇 (Spearman |rho|>0.95, 40万行子采样) ...")
sub_rows = np.where(valid)[0][::4]
Xs = F.iloc[sub_rows][feats].to_numpy(np.float64)
keep = np.all(np.isfinite(Xs), axis=1)
corr = pd.DataFrame(Xs[keep], columns=feats).corr(method="spearman").abs()
visited, clusters = set(), []
for c in feats:
    if c in visited:
        continue
    cl, visited = [c], {c}
    for c2 in feats:
        if c2 in visited:
            continue
        if corr.loc[c, c2] > 0.95:
            cl.append(c2); visited.add(c2)
    if len(cl) > 1:
        clusters.append(cl)

log("M1: 数据完整性 (月bar数) ...")
mcount = pd.Series(1, index=idx).groupby(idx.to_period("M")).sum()
bad_months = mcount[mcount < 20000]
log(f"bar数偏低月: {list(bad_months.index.astype(str))[:12]}")

# ---------------------------------------------------------------- M2 波段可吃性
log("M2: 逐bar最优方向 + 可吃标注 ...")
pl = lab["pnl_long"].to_numpy(np.float64)
ps = lab["pnl_short"].to_numpy(np.float64)
tp = lab["tp_d"].to_numpy(np.float64)
eidx = lab["entry_idx"].to_numpy(np.int64)
exl = lab["exit_bar_long"].to_numpy(np.int64)
exs = lab["exit_bar_short"].to_numpy(np.int64)
best = np.where(valid, np.maximum(pl, ps), np.nan)
best_dir_long = pl >= ps
eat_strong = valid & (best >= 0.6 * tp)
eat_mid = valid & (best >= 0.3 * tp)
eat_weak = valid & (best > 0)
chop = valid & (np.abs(pl - ps) < 0.2 * tp)
exit_best = np.where(best_dir_long, exl, exs)
dur = np.where(valid, np.maximum(exit_best - eidx, 1), np.nan)
log(f"可吃率: strong={eat_strong[valid].mean()*100:.1f}% mid={eat_mid[valid].mean()*100:.1f}% "
    f"weak={eat_weak[valid].mean()*100:.1f}% chop={chop[valid].mean()*100:.1f}%")

@njit(cache=True)
def _oracle(pnl_best, eat, exit_b, cooldown):
    total, n, free = 0.0, 0, 0
    for i in range(len(pnl_best)):
        if not eat[i]:
            continue
        if i < free:
            continue
        total += pnl_best[i]; n += 1
        e = exit_b[i] + cooldown
        if e > free:
            free = e
    return total, n

log("M2: Oracle 天花板 (完美预知最优方向) ...")
to_l = (lab["out_long"].to_numpy() == 0) & valid
to_s = (lab["out_short"].to_numpy() == 0) & valid
dq = np.nanquantile(dur[valid], [0.1, 0.25, 0.5, 0.75, 0.9])
log(f"诊断: timeout占比 long={to_l[valid].mean()*100:.0f}% short={to_s[valid].mean()*100:.0f}% "
    f"| dur分位 {np.round(dq,1)} | tp中位={np.median(tp[valid]):.2f} "
    f"sl中位={np.median(lab['sl_d'].to_numpy()[valid]):.2f} sc均值={spread_cost[valid].mean():.3f}")
orc_tot, orc_tot_n = _oracle(best, valid.astype(np.bool_), exit_best, 10)
orc_cd, orc_cd_n = _oracle(best, eat_strong.astype(np.bool_), exit_best, 10)
eat_m_i = eat_mid.astype(np.bool_)
orc_mid, orc_mid_n = _oracle(best, eat_m_i, exit_best, 10)
orc_nc, orc_nc_n = _oracle(best, eat_strong.astype(np.bool_),
                           np.arange(len(best), dtype=np.int64), 0)
log(f"Oracle(全bar,cd10)={orc_tot:,.0f}$ n={orc_tot_n:,} | strong={orc_cd:,.0f}$ "
    f"n={orc_cd_n:,} | mid={orc_mid:,.0f}$ n={orc_mid_n:,} | strong无冷却={orc_nc:,.0f}$ n={orc_nc_n:,}")

log("M2: 11维regime切片 ...")
df = pd.DataFrame({
    "eat_s": eat_strong, "eat_m": eat_mid, "chop": chop,
    "best": best, "tp": tp, "dur": dur, "dlong": best_dir_long,
    "hour": idx.hour, "dow": idx.dayofweek,
    "yq": idx.to_period("Q").astype(str),
})
sess = np.where((idx.hour >= 7) & (idx.hour < 13), 1,
                np.where((idx.hour >= 13) & (idx.hour < 21), 2,
                         np.where((idx.hour >= 21) | (idx.hour < 1), 3, 0)))
df["session"] = sess
for dim in ["tr_over_atr", "momn_240", "squeeze", "spread_rel", "atr_ratio", "run_len"]:
    df[dim] = pd.qcut(F[dim].to_numpy(np.float64), 5, labels=False, duplicates="drop")
df["abs_mom_240"] = pd.qcut(np.abs(F["momn_240"].to_numpy(np.float64)), 5,
                            labels=False, duplicates="drop")

def agg_dim(dim):
    g = df[valid].groupby(dim, observed=True).agg(
        n=("eat_s", "size"), eat_rate=("eat_s", "mean"), eat_mid_rate=("eat_m", "mean"),
        chop_rate=("chop", "mean"), mean_best=("best", "mean"),
        mean_tp=("tp", "mean"), med_dur=("dur", "median"), long_share=("dlong", "mean"))
    g["oracle_density"] = g["mean_best"] * g["eat_rate"]
    g = g.reset_index(); g.insert(0, "dim", dim)
    return g.rename(columns={dim: "bucket"})

slice_frames = [agg_dim(d) for d in
                ["hour", "dow", "yq", "session", "tr_over_atr", "momn_240",
                 "abs_mom_240", "squeeze", "spread_rel", "atr_ratio", "run_len"]]
slices = pd.concat(slice_frames, ignore_index=True)
slices.to_csv(os.path.join(OUT, "eatability_slices.csv"), index=False)
glob_rate = float(eat_strong[valid].mean())
hi = slices[(slices["n"] > 5000) & (slices["eat_rate"] > glob_rate * 1.25)]
log(f"高可吃桶(>1.25x全局): {len(hi)}个; 全局可吃率={glob_rate*100:.1f}%")

# ---------------------------------------------------------------- M3 特征信息量
log("M3: LGB gain importance (4折时间CV, 双目标) ...")
import lightgbm as lgb
from sklearn.feature_selection import mutual_info_classif

X = F[feats].to_numpy(np.float32)
y_dir = best_dir_long.astype(np.int8)
y_eat = eat_strong.astype(np.int8)
tr_mask = valid & (idx < pd.Timestamp("2024-08-01"))
tr_pos = np.where(tr_mask)[0]
n_tr = len(tr_pos)
cuts = [int(n_tr * k / 4) for k in range(5)]
gain_dir = np.zeros(len(feats)); gain_eat = np.zeros(len(feats))
n_folds = 0
params = dict(n_estimators=300, learning_rate=0.05, num_leaves=63,
              min_child_samples=200, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, n_jobs=30, verbose=-1, random_state=42)
for k in range(1, 4):  # 滚动: train=前k段, val=第k+1段
    tr_i, va_i = tr_pos[:cuts[k]], tr_pos[cuts[k]:cuts[k + 1]]
    if len(va_i) < 5000:
        continue
    n_folds += 1
    for yv, acc in [(y_dir, gain_dir), (y_eat, gain_eat)]:
        m = lgb.LGBMClassifier(**params)
        m.fit(X[tr_i], yv[tr_i], eval_set=[(X[va_i], yv[va_i])],
              callbacks=[lgb.early_stopping(30, verbose=False)])
        imp = m.booster_.feature_importance("gain")
        acc += imp / (imp.sum() + 1e-12)
if n_folds:
    gain_dir /= n_folds; gain_eat /= n_folds

log("M3: 互信息 (40万子采样) ...")
rng = np.random.default_rng(42)
mi_pos = rng.choice(tr_pos, size=min(400000, n_tr), replace=False)
mi_ok = np.all(np.isfinite(X[mi_pos]), axis=1)
mi = mutual_info_classif(X[mi_pos][mi_ok], y_dir[mi_pos][mi_ok],
                         n_neighbors=3, random_state=42)
mi = np.asarray(mi, dtype=np.float64)
mi = mi / (mi.max() + 1e-12)

log("M3: 月度Spearman IC (feat vs 标准化最优PnL比) ...")
y_ic = best / tp
mo_codes, mo_uniq = pd.factorize(idx.to_period("M").astype(str))
ic_sum = np.zeros(len(feats)); ic_sq = np.zeros(len(feats)); n_mo = 0
for mi_ in range(len(mo_uniq)):
    mrows = np.where(valid & (mo_codes == mi_))[0]
    if len(mrows) < 2000:
        continue
    xs, yv = X[mrows], y_ic[mrows]
    ok = np.all(np.isfinite(xs), axis=1) & np.isfinite(yv)
    if ok.sum() < 1500:
        continue
    n_mo += 1
    for j in range(len(feats)):
        r = spearmanr(xs[ok, j], yv[ok]).correlation
        if np.isfinite(r):
            ic_sum[j] += r; ic_sq[j] += r * r
ic_mean = ic_sum / max(n_mo, 1)
ic_std = np.sqrt(np.maximum(ic_sq / max(n_mo, 1) - ic_mean ** 2, 0))
ic_abs = np.abs(ic_mean); ic_abs = ic_abs / (ic_abs.max() + 1e-12)
log(f"IC 覆盖 {n_mo} 个月; top IC特征: {feats[int(np.argmax(ic_abs))]}")

score = 0.45 * gain_dir + 0.20 * gain_eat + 0.20 * mi + 0.15 * ic_abs
imp_df = pd.DataFrame({
    "feature": feats, "score": score, "lgb_dir_gain": gain_dir,
    "lgb_eat_gain": gain_eat, "mi_norm": mi, "ic_mean": ic_mean, "ic_abs_norm": ic_abs,
}).sort_values("score", ascending=False)
imp_df.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)

white = list(imp_df["feature"][:18])
score_map = dict(zip(feats, score))
mi_map = dict(zip(feats, mi)); icn_map = dict(zip(feats, ic_abs))
black = [c for c in feats if score_map[c] < 0.10 and mi_map[c] < 0.20 and icn_map[c] < 0.15]
gray = [c for c in feats if c not in white and c not in black]
redundant_drop = []
for cl in clusters:
    redundant_drop.extend(sorted(cl, key=lambda c: score_map[c])[:-1])
log(f"白名单{len(white)} 灰{len(gray)} 黑{len(black)} 冗余剔除{len(redundant_drop)}")

# ---------------------------------------------------------------- 汇总
cap_needed = HIST_TARGET / orc_cd if orc_cd > 0 else float("inf")
summary = {
    "generated": str(pd.Timestamp.now()),
    "data": {"rows": int(len(F)), "valid": int(valid.sum()),
             "span": f"{idx[0]} ~ {idx[-1]}",
             "bad_months_low_bars": [str(m) for m in bad_months.index]},
    "eatability": {
        "global_rate": glob_rate,
        "strong_mid_weak": [float(eat_strong[valid].mean()), float(eat_mid[valid].mean()),
                            float(eat_weak[valid].mean())],
        "chop_rate": float(chop[valid].mean()),
        "oracle_all_cd10": float(orc_tot), "oracle_all_n": int(orc_tot_n),
        "oracle_strong_cd10": float(orc_cd), "oracle_strong_n": int(orc_cd_n),
        "oracle_mid_cd10": float(orc_mid), "oracle_mid_n": int(orc_mid_n),
        "oracle_strong_nocd": float(orc_nc), "oracle_strong_nocd_n": int(orc_nc_n),
        "hist_target": HIST_TARGET,
        "capture_needed_of_oracle_strong": float(cap_needed),
    },
    "features": {
        "whitelist": white, "graylist": gray, "blacklist": black,
        "redundancy_clusters": clusters, "redundant_drop": redundant_drop,
        "top10": list(imp_df["feature"][:10]),
        "high_drift": list(prof[prof["drift_score"] > 0.5]["feature"]),
    },
    "high_eat_buckets": {f"{r['dim']}={r['bucket']}": float(r["eat_rate"])
                         for _, r in hi.sort_values("eat_rate", ascending=False).head(20).iterrows()},
}
with open(os.path.join(OUT, "stage_a_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

msg = (f"StageA done {int(time.time()-T0)}s: 可吃率{glob_rate*100:.1f}% "
       f"oracle(cd10)={orc_tot:,.0f} strong={orc_cd:,.0f} "
       f"| 复现732需捕获{cap_needed*100:.1f}%of strong-oracle "
       f"| 特征白{len(white)}/灰{len(gray)}/黑{len(black)} 冗余剔{len(redundant_drop)} "
       f"| top3: {list(imp_df['feature'][:3])}")
log(msg)
os.system(f"curl -s -m 20 -d '{msg}' ntfy.sh/{NTFY} > /dev/null 2>&1 &")
log(f"全部输出 -> {OUT}")
