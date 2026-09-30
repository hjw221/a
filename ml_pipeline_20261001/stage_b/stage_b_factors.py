"""Stage B — ML 因子挖掘 (2026-10-01).

用户指令: "然后再用机器学习来找因子"
输入: Stage A 白名单结论 (长期结构特征有效, 微观特征废弃)
流程:
  B1 因子库生成: 8核心特征 x {z1440, d60, d1440} + 精选交互/结构因子 (~60个)
  B2 评估: 月度rank-IC vs 肉量(best/tp) 与 方向超额((pl-ps)/tp) -> IC/ICIR/单调性
  B3 贪心正交选因子 (|IC|排序, 相关<0.5)
  B4 alpha合成: IC加权 z-score 因子 -> alpha_eat(肉量) + alpha_dir(方向)
     权重只用首训练窗 2022-01~2024-07 (与历史调参窗一致, OOS零泄露)
  B5 规则预演回测: alpha_eat分位触发 x alpha_dir方向, OOS 2024-08~2026-07 全期
     cd=10 (历史口径) 与 cd=2 (StageA发现: 冷却杀吞吐) 对照
输出: /root/rivermind-data/xauusd/stage_b/
"""
import os, sys, json, time, pickle, copy
import numpy as np
import pandas as pd

T0 = time.time()
def log(s): print(f"[B {time.time()-T0:6.0f}s] {s}", flush=True)

BASE_V2 = "/root/rivermind-data/xauusd/v2_ens"
OUT = "/root/rivermind-data/xauusd/stage_b"
os.makedirs(OUT, exist_ok=True)
CSV = "/root/rivermind-fs/xauusd/data/XAUUSDc_M1_202201022305_202606262057.csv"
NTFY = "xauusd-qv7m2zk9-res"
sys.path.insert(0, BASE_V2)
from scalp import build_folds_m1, run_backtest_m1
from backtest import metrics

CFG = {"first_oos_month": "2024-08", "last_oos_month": "2026-07",
       "max_train_months": 36, "scalp_embargo_m1": 30, "cooldown_m1": 10,
       "inner_val_frac": 0.20}

# ---------------------------------------------------------------- 加载
log("加载 pack ...")
with open(os.path.join(BASE_V2, "cache", "pack_m1sc_ad.pkl"), "rb") as f:
    pack = pickle.load(f)
F, feats = pack["F"], pack["feats"]
lab, valid = pack["lab"], pack["valid"]

dt = pd.read_csv(CSV, sep="\t", usecols=["<DATE>", "<TIME>"])
uniq = dt["<DATE>"].unique()
dmap = {d: pd.to_datetime(d, format="%Y.%m.%d") for d in uniq}
dates = dt["<DATE>"].map(dmap)
secs = dt["<TIME>"].str.slice(0, 2).astype(np.int32) * 3600 + \
       dt["<TIME>"].str.slice(3, 5).astype(np.int32) * 60
idx = pd.DatetimeIndex((dates + pd.to_timedelta(secs, unit="s")).sort_values().to_numpy())
del dt, dates, secs

pl = lab["pnl_long"].to_numpy(np.float64)
ps = lab["pnl_short"].to_numpy(np.float64)
tp = lab["tp_d"].to_numpy(np.float64)
y_eat = np.where(valid, np.maximum(pl, ps) / tp, np.nan)      # 肉量
y_dir = np.where(valid, (pl - ps) / tp, np.nan)               # 方向超额 (正=多)
log(f"pack: {len(F):,} rows; 目标就绪")

TR_END = pd.Timestamp("2024-08-01")
tr_mask = valid & (idx < TR_END)
mo = idx.to_period("M")
mo_codes, mo_uniq = pd.factorize(mo)
log(f"训练窗月数: {(mo_codes[tr_mask]).max()+1 if tr_mask.any() else 0}, 全期 {len(mo_uniq)} 月")

# ---------------------------------------------------------------- B1 因子库
log("B1: 因子库生成 ...")
c = F
factors = {}

B8 = ["range_ratio", "squeeze", "slope_1440", "atr_ratio", "tr_over_atr",
      "vol_ratio_12_1440", "dist_hi_1440", "momn_1440"]
def z1440(s):
    m = s.rolling(1440, min_periods=720).mean()
    sd = s.rolling(1440, min_periods=720).std()
    return ((s - m) / sd.replace(0.0, np.nan)).clip(-5, 5)

for b in B8:
    s = c[b].astype(np.float64)
    factors[f"{b}__z"] = z1440(s)
    factors[f"{b}__d60"] = (s - s.shift(60)).clip(-10, 10)
    factors[f"{b}__d1440"] = (s - s.shift(1440)).clip(-10, 10)

# 交互/结构因子
sqz, atrr, tr_a = c["squeeze"].astype(np.float64), c["atr_ratio"].astype(np.float64), \
                  c["tr_over_atr"].astype(np.float64)
sl1440 = c["slope_1440"].astype(np.float64)
mom1440 = c["momn_1440"].astype(np.float64)
mom60 = c["momn_60"].astype(np.float64)
sl240 = c["slope_240"].astype(np.float64)
ny = c["session_ny"].astype(np.float64)
lon = c["session_london"].astype(np.float64)
hi, lo = c["dist_hi_1440"].astype(np.float64), c["dist_lo_1440"].astype(np.float64)
vr = c["vol_ratio_12_1440"].astype(np.float64)
dow = pd.Series(idx.dayofweek, index=idx)
hour = pd.Series(idx.hour, index=idx)

factors["sgnsl_atr"] = np.sign(sl1440) * atrr
factors["sgnsl_sqz"] = np.sign(sl1440) * sqz
factors["sgnmom_vr"] = np.sign(mom1440) * vr
factors["vote3"] = (np.sign(sl1440) + np.sign(sl240) + np.sign(mom60)) / 3.0
factors["vote3_atr"] = factors["vote3"] * atrr
factors["ny_sqz"] = ny * sqz
factors["ny_atr"] = ny * atrr
factors["ny_tr"] = ny * tr_a
factors["lon_sqz"] = lon * sqz
factors["fri_mom"] = (dow == 4).astype(float) * mom1440
factors["fri_atr"] = (dow == 4).astype(float) * atrr
factors["pos_asym"] = (hi + lo) / 2.0          # 价格相对24h区间中线偏离(带方向)
factors["pos_asym_z"] = z1440(factors["pos_asym"])
factors["hi_lo_gap"] = hi - lo                  # 突破不对称
factors["bs_break"] = c["bars_since_hi240"].astype(np.float64) - \
                      c["bars_since_lo240"].astype(np.float64)
factors["atr_acc"] = atrr - atrr.shift(240)
factors["sqz_acc"] = sqz - sqz.shift(240)
factors["hr14_sqz"] = ((hour >= 13) & (hour < 18)).astype(float) * sqz
factors["hr14_atr"] = ((hour >= 13) & (hour < 18)).astype(float) * atrr
factors["mom_pers"] = np.sign(sl1440) * np.sign(mom60) * atrr
factors["mom_pers_sqz"] = np.sign(sl1440) * np.sign(mom60) * sqz
factors["sqz_d1_volz"] = sqz * c["vol_z_1440"].astype(np.float64)

fnames = list(factors.keys())
log(f"因子库 {len(fnames)} 个 -> 矩阵化")
FM = pd.DataFrame(factors, index=idx).astype(np.float32)
log(f"因子矩阵 {FM.shape}")

# ---------------------------------------------------------------- B2 评估: 月度 rank-IC
log("B2: 月度rank-IC (肉量 & 方向) ...")
Rk = FM.rank()                                   # 全期rank -> 月内pearson=月内spearman
ye_r = pd.Series(y_eat).rank()
yd_r = pd.Series(y_dir).rank()
ic_eat = pd.DataFrame(0.0, index=mo_uniq, columns=fnames)
ic_dir = pd.DataFrame(0.0, index=mo_uniq, columns=fnames)
tr_months = mo_codes[tr_mask]
tr_month_set = set(np.unique(tr_months))
for mi_ in sorted(tr_month_set):
    rows = np.where(valid & (mo_codes == mi_))[0]
    if len(rows) < 2000:
        continue
    X = Rk.iloc[rows].to_numpy(np.float64)
    ok = np.all(np.isfinite(X), axis=1)
    if ok.sum() < 1500:
        continue
    Xf = X[ok]
    for j, fn in enumerate(fnames):
        ic_eat.iloc[mi_, j] = np.corrcoef(Xf[:, j], ye_r.to_numpy()[rows][ok])[0, 1]
        ic_dir.iloc[mi_, j] = np.corrcoef(Xf[:, j], yd_r.to_numpy()[rows][ok])[0, 1]
# 全期(训练窗)IC = 月IC均值 (tr_month_set 是 factorize codes, 用 iloc 位置索引)
tr_codes = sorted([int(x) for x in tr_month_set])
ic_e_mean = ic_eat.iloc[tr_codes].mean()
ic_d_mean = ic_dir.iloc[tr_codes].mean()
ic_e_ir = ic_e_mean / (ic_eat.iloc[tr_codes].std() + 1e-12)
ic_d_ir = ic_d_mean / (ic_dir.iloc[tr_codes].std() + 1e-12)

# 五分位单调性 (训练窗, 因子分位 vs 肉量均值)
log("B2: 五分位单调性 ...")
mono = {}
tr_pos = np.where(tr_mask)[0]
Xtr = FM.iloc[tr_pos]
ytr = y_eat[tr_pos]
for fn in fnames:
    x = Xtr[fn].to_numpy(np.float64)
    ok = np.isfinite(x) & np.isfinite(ytr)
    if ok.sum() < 10000:
        mono[fn] = 0.0; continue
    qb = pd.qcut(x[ok], 5, labels=False, duplicates="drop")
    dfb = pd.DataFrame({"q": qb, "y": ytr[ok]}).groupby("q")["y"].mean()
    mono[fn] = float(np.corrcoef(np.arange(len(dfb)), dfb.to_numpy())[0, 1]) \
        if len(dfb) >= 3 else 0.0

fac_tbl = pd.DataFrame({
    "ic_eat": ic_e_mean, "ic_dir": ic_d_mean,
    "icir_eat": ic_e_ir, "icir_dir": ic_d_ir,
    "mono_eat": pd.Series(mono),
}).sort_values("icir_eat", key=np.abs, ascending=False)
fac_tbl.to_csv(os.path.join(OUT, "factor_ic.csv"))
log(f"top肉量因子: {list(fac_tbl.index[:8])}")

# ---------------------------------------------------------------- B3 贪心正交
log("B3: 贪心正交选因子 (|ICIR|排序, 相关<0.5) ...")
Xtr_rank = FM.iloc[tr_pos].rank()
eat_order = list(fac_tbl.sort_values("icir_eat", key=np.abs, ascending=False).index)
dir_order = list(fac_tbl.sort_values("icir_dir", key=np.abs, ascending=False).index)
_sel_e, _names_e = [], []
for fn in eat_order:
    if len(_names_e) >= 12: break
    x = Xtr_rank[fn].to_numpy(np.float64); ok = np.isfinite(x)
    xr = np.where(ok, x, np.nanmean(x)); xr = (xr - xr.mean()) / (xr.std() + 1e-12)
    if all(abs(np.corrcoef(xr, s)[0, 1]) < 0.5 for s in _sel_e):
        _sel_e.append(xr); _names_e.append(fn)
_sel_d, _names_d = [], []
for fn in dir_order:
    if len(_names_d) >= 12: break
    x = Xtr_rank[fn].to_numpy(np.float64); ok = np.isfinite(x)
    xr = np.where(ok, x, np.nanmean(x)); xr = (xr - xr.mean()) / (xr.std() + 1e-12)
    if all(abs(np.corrcoef(xr, s)[0, 1]) < 0.5 for s in _sel_d):
        _sel_d.append(xr); _names_d.append(fn)
log(f"正交肉量因子: {_names_e}")
log(f"正交方向因子: {_names_d}")

# ---------------------------------------------------------------- B4 alpha 合成 (训练窗权重)
log("B4: alpha 合成 ...")
w_e = np.array([ic_e_mean[fn] for fn in _names_e]); w_e = w_e / (np.abs(w_e).sum() + 1e-12)
w_d = np.array([ic_d_mean[fn] for fn in _names_d]); w_d = w_d / (np.abs(w_d).sum() + 1e-12)

# 全期因子 z 标准化 (因果 rolling 1440)
def causal_z(fm_col):
    s = fm_col.astype(np.float64)
    m = s.rolling(1440, min_periods=720).mean()
    sd = s.rolling(1440, min_periods=720).std()
    return ((s - m) / sd.replace(0.0, np.nan)).clip(-5, 5)

alpha_eat = np.zeros(len(FM)); alpha_dir = np.zeros(len(FM))
for k, fn in enumerate(_names_e):
    alpha_eat += w_e[k] * causal_z(FM[fn]).to_numpy(np.float64)
for k, fn in enumerate(_names_d):
    alpha_dir += w_d[k] * causal_z(FM[fn]).to_numpy(np.float64)

# 训练窗分位阈值 (零泄露)
ae_tr = alpha_eat[tr_mask]; ad_tr = np.abs(alpha_dir[tr_mask])
q_eat = {q: float(np.nanquantile(ae_tr, q)) for q in [0.80, 0.85, 0.90, 0.95]}
q_dir = {q: float(np.nanquantile(ad_tr, q)) for q in [0.50, 0.60, 0.70]}

# ---------------------------------------------------------------- B5 预演回测
log("B5: OOS 规则预演回测 (cd=10 与 cd=2) ...")
folds = build_folds_m1(idx, CFG, 90)

# 读取 OHLC 供回测 (DATE/TIME/OHLC 一起读, 按时间排序, 与 idx 对齐断言)
log("读取 OHLC 供回测 ...")
raw = pd.read_csv(CSV, sep="\t",
                  usecols=["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>"])
uniq_d = raw["<DATE>"].unique()
dm = {d: pd.to_datetime(d, format="%Y.%m.%d") for d in uniq_d}
_dt = raw["<DATE>"].map(dm)
_ss = raw["<TIME>"].str.slice(0, 2).astype(np.int32) * 3600 + \
      raw["<TIME>"].str.slice(3, 5).astype(np.int32) * 60
_ts = _dt + pd.to_timedelta(_ss, unit="s")
_order = np.argsort(_ts.to_numpy(), kind="stable")
assert (_ts.iloc[_order].to_numpy() == idx.to_numpy()).all(), "OHLC时间轴与pack不对齐"
m1_t = np.asarray(idx.asi8 // 10**9 // 60, dtype=np.int64)
m1_o = raw["<OPEN>"].to_numpy(np.float64)[_order]
m1_h = raw["<HIGH>"].to_numpy(np.float64)[_order]
m1_l = raw["<LOW>"].to_numpy(np.float64)[_order]
m1_c = raw["<CLOSE>"].to_numpy(np.float64)[_order]
m1_pack = (m1_t, m1_o, m1_h, m1_l, m1_c)
del raw, _dt, _ss, _ts

def run_rule2(qe, qd, cd):
    cfg2 = dict(CFG); cfg2["cooldown_m1"] = cd
    sig = np.zeros(len(FM), dtype=np.int8)
    ok = (alpha_eat > q_eat[qe]) & (np.abs(alpha_dir) > q_dir[qd]) & valid
    sig[ok & (alpha_dir > 0)] = 1
    sig[ok & (alpha_dir < 0)] = -1
    logs = []
    for f in folds:
        s = sig[f["oos_rows"]]
        if not (s != 0).any():
            continue
        logs.append(run_backtest_m1(s, f["oos_rows"], lab, valid, m1_pack, cfg2,
                                    tag=f"q{qe}|d{qd}|cd{cd}"))
    return pd.concat(logs, ignore_index=True) if logs else None

results = []
best_log = None
for qe in [0.85, 0.90, 0.95]:
    for qd in [0.60]:
        for cd in [10, 2]:
            llog = run_rule2(qe, qd, cd)
            if llog is None or len(llog) == 0:
                continue
            mm = metrics(llog)
            pnl = float(llog["pnl"].sum())
            plr = mm.get("plr"); plr = float(plr) if plr is not None and np.isfinite(plr) else 0.0
            results.append({"q_eat": qe, "q_dir": qd, "cd": cd,
                            "trades": int(len(llog)), "pnl": pnl,
                            "wr": mm.get("win_rate"), "plr": plr})
            llog.to_csv(os.path.join(OUT, f"trades_rule_q{qe}_d{qd}_cd{cd}.csv"), index=False)
            log(f"  q{qe}/d{qd}/cd{cd}: n={len(llog)} pnl={pnl:+.1f} plr={plr:.2f}")
res_df = pd.DataFrame(results).sort_values("pnl", ascending=False)
res_df.to_csv(os.path.join(OUT, "rule_backtest.csv"), index=False)

# ---------------------------------------------------------------- 汇总
top_factors = {k: float(v) for k, v in fac_tbl["icir_eat"].head(15).items()}
summary = {
    "generated": str(pd.Timestamp.now()),
    "n_factors": len(fnames),
    "orthogonal_eat": _names_e, "orthogonal_dir": _names_d,
    "weights_eat": dict(zip(_names_e, [float(x) for x in w_e])),
    "weights_dir": dict(zip(_names_d, [float(x) for x in w_d])),
    "top_factor_icir": top_factors,
    "rule_backtest": results,
    "best": res_df.iloc[0].to_dict() if len(res_df) else None,
}
with open(os.path.join(OUT, "stage_b_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

msg = (f"StageB done {int(time.time()-T0)}s: 因子{len(fnames)}个 "
       f"| 预演最优 " +
       (f"pnl={res_df.iloc[0]['pnl']:+.1f}$ n={int(res_df.iloc[0]['trades'])} "
        f"plr={float(res_df.iloc[0]['plr']):.2f} (q{res_df.iloc[0]['q_eat']}/cd{res_df.iloc[0]['cd']})"
        if len(res_df) else "无交易") +
       f" | 方向因子top: {_names_d[:3]}")
log(msg)
os.system(f"curl -s -m 20 -d '{msg}' ntfy.sh/{NTFY} > /dev/null 2>&1 &")
log(f"输出 -> {OUT}")
