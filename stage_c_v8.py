"""Stage C v8 — 薄利多销极限放量: K{70,85,100}% + cd{1,0} + 月均>=300 (2026-10-01).

v7 诊断: LGB概率分布极度压缩(99% bar在0.33-0.36), 绝对tau死路;
分位数门控免疫压缩(v3/v6靠它到790笔)。v6只扫到K60%。
v8 = v6配置 + K{70,85,100}% + td{0,0.02} + cd{1,0} + 月均>=300硬约束。
research冠军参考: 24390笔/54月=450笔月均$0.034/笔 PLR2.10。
用法: python stage_c_v8.py
"""
import os, sys, json, time, pickle, argparse
import numpy as np
import pandas as pd

T0 = time.time()
def log(s): print(f"[C2 {time.time()-T0:6.0f}s] {s}", flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["dual", "classic"], default="dual")
ap.add_argument("--seeds", default="42")
ap.add_argument("--hl", type=float, default=180.0)
ap.add_argument("--feats", choices=["full", "wl"], default="full")
args = ap.parse_args()

BASE_V2 = "/root/rivermind-data/xauusd/v2_ens"
CSV = "/root/rivermind-fs/xauusd/data/XAUUSDc_M1_202201022305_202606262057.csv"
TAG = f"v8_{args.mode}_{args.feats}_s{args.seeds.replace(',', '-')}"
OUT = f"/root/rivermind-data/xauusd/stage_c/{TAG}"
os.makedirs(OUT, exist_ok=True)
NTFY = "xauusd-qv7m2zk9-res"
HIST_TARGET = 732.7
sys.path.insert(0, BASE_V2)
from scalp import build_folds_m1, run_backtest_m1
from backtest import metrics
from models import fit_lgb

CFG = {"first_oos_month": "2024-08", "last_oos_month": "2026-07",
       "max_train_months": 36, "scalp_embargo_m1": 30, "cooldown_m1": 10,
       "inner_val_frac": 0.20, "wr_floor": 0.36, "min_trades_cal": 60}
SEEDS = [int(s) for s in args.seeds.split(",")]
KS = [0.70, 0.85, 1.00]         # 精确top-N比例 (极限放量)
TDS = [0.0, 0.02]                # 方向置信门槛
CDS = [1, 0]                     # 高频冷却
MIN_N_CAL = 1200                 # 校准段最低交易数 (月均>=300)
MIN_PER_MONTH = 300              # 月均硬约束 (薄利多销)

# ---------------------------------------------------------------- 加载 (同 v1)
log(f"加载 ... (mode={args.mode} seeds={SEEDS})")
with open(os.path.join(BASE_V2, "cache", "pack_m1sc_ad.pkl"), "rb") as f:
    pack = pickle.load(f)
F, feats = pack["F"], pack["feats"]
lab, valid = pack["lab"], pack["valid"]
tuned = json.load(open(os.path.join(BASE_V2, "results", "tuned_params_m1.json")))["m1sc_ad"]
LGB_P = dict(tuned["long"]["lgb"])

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
y_eat = (np.maximum(pl, ps) >= 0.6 * tp) & valid
y_dir = (pl >= ps) & valid
y_l = (lab["out_long"].to_numpy() == 1) & valid
y_s = (lab["out_short"].to_numpy() == 1) & valid

raw = pd.read_csv(CSV, sep="\t",
                  usecols=["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>"])
_dt = raw["<DATE>"].map(dmap)
_ss = raw["<TIME>"].str.slice(0, 2).astype(np.int32) * 3600 + \
      raw["<TIME>"].str.slice(3, 5).astype(np.int32) * 60
_ts = _dt + pd.to_timedelta(_ss, unit="s")
_order = np.argsort(_ts.to_numpy(), kind="stable")
assert (_ts.iloc[_order].to_numpy() == idx.to_numpy()).all()
m1_pack = (np.asarray(idx.asi8 // 10**9 // 60, dtype=np.int64),
           raw["<OPEN>"].to_numpy(np.float64)[_order],
           raw["<HIGH>"].to_numpy(np.float64)[_order],
           raw["<LOW>"].to_numpy(np.float64)[_order],
           raw["<CLOSE>"].to_numpy(np.float64)[_order])
del raw, _dt, _ss, _ts

WL = ["range_ratio", "squeeze", "slope_1440", "session_ny", "hour_sin", "atr_ratio",
      "dist_hi_1440", "vol_ratio_12_1440", "c_dist_d1", "er_1440", "dist_lo_1440",
      "momn_1440", "vol_z_1440", "dow_sin", "dow_cos", "pos_in_range", "tr_over_atr",
      "hour_cos"]
log("生成因子46 ...")
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
sqz, atrr, tr_a = c["squeeze"].astype(np.float64), c["atr_ratio"].astype(np.float64), \
                  c["tr_over_atr"].astype(np.float64)
sl1440 = c["slope_1440"].astype(np.float64); sl240 = c["slope_240"].astype(np.float64)
mom1440 = c["momn_1440"].astype(np.float64); mom60 = c["momn_60"].astype(np.float64)
ny = c["session_ny"].astype(np.float64); lon = c["session_london"].astype(np.float64)
hi, lo = c["dist_hi_1440"].astype(np.float64), c["dist_lo_1440"].astype(np.float64)
vr = c["vol_ratio_12_1440"].astype(np.float64)
dow = pd.Series(idx.dayofweek, index=idx); hour = pd.Series(idx.hour, index=idx)
factors["sgnsl_atr"] = np.sign(sl1440) * atrr
factors["sgnsl_sqz"] = np.sign(sl1440) * sqz
factors["sgnmom_vr"] = np.sign(mom1440) * vr
factors["vote3"] = (np.sign(sl1440) + np.sign(sl240) + np.sign(mom60)) / 3.0
factors["vote3_atr"] = factors["vote3"] * atrr
factors["ny_sqz"] = ny * sqz; factors["ny_atr"] = ny * atrr; factors["ny_tr"] = ny * tr_a
factors["lon_sqz"] = lon * sqz
factors["fri_mom"] = (dow == 4).astype(float) * mom1440
factors["fri_atr"] = (dow == 4).astype(float) * atrr
factors["pos_asym"] = (hi + lo) / 2.0
factors["pos_asym_z"] = z1440(factors["pos_asym"])
factors["hi_lo_gap"] = hi - lo
factors["bs_break"] = c["bars_since_hi240"].astype(np.float64) - \
                      c["bars_since_lo240"].astype(np.float64)
factors["atr_acc"] = atrr - atrr.shift(240)
factors["sqz_acc"] = sqz - sqz.shift(240)
factors["hr14_sqz"] = ((hour >= 13) & (hour < 18)).astype(float) * sqz
factors["hr14_atr"] = ((hour >= 13) & (hour < 18)).astype(float) * atrr
factors["mom_pers"] = np.sign(sl1440) * np.sign(mom60) * atrr
factors["mom_pers_sqz"] = np.sign(sl1440) * np.sign(mom60) * sqz
factors["sqz_d1_volz"] = sqz * c["vol_z_1440"].astype(np.float64)
FM = pd.DataFrame(factors, index=idx)
if args.feats == "full":
    X = pd.concat([F[WL].astype(np.float32), FM.astype(np.float32)], axis=1)
else:
    X = F[WL].astype(np.float32)
X = X.to_numpy(np.float32)
log(f"特征矩阵 {X.shape}")

def decay_w(rows):
    t = np.asarray(idx[rows].asi8, dtype=np.float64) / 1e9 / 86400.0
    return np.exp(-np.log(2.0) * np.maximum(t[-1] - t, 0) / args.hl)

def fit_seed(Xtr, ytr, wtr, Xva, yva, wva, seed):
    p = dict(LGB_P); p.update(seed=seed)
    m, _ = fit_lgb(Xtr, ytr, wtr, p, Xva, yva, wva, es_rounds=50, max_rounds=600)
    return m

# ---------------------------------------------------------------- q_gate 信号
def qgate_sig(pA_seg, pB_seg, months_seg, K, td, mode):
    """月内 argsort 精确 top-N of pA + td 方向门槛 (dual); classic 双侧topN。"""
    sig = np.zeros(len(pA_seg), dtype=np.int8)
    for m in np.unique(months_seg):
        mi = months_seg == m
        n_m = int(K * mi.sum())
        if n_m < 1:
            continue
        pa = np.where(np.isfinite(pA_seg), pA_seg, -1.0)
        seg_idx = np.where(mi)[0][np.argsort(-pa[mi])[:n_m]]   # 精确 top-N
        pb = pB_seg[seg_idx]
        s = np.zeros(len(seg_idx), dtype=np.int8)
        if mode == "dual":
            ok = np.abs(pb - 0.5) > td
            s[ok & (pb > 0.5)] = 1
            s[ok & (pb < 0.5)] = -1
        else:
            # classic: pA/pB 为双侧概率, top-N 各自取, 冲突取超额大者
            ra = np.quantile(pa[mi], 1 - K)
            s2 = np.zeros(len(seg_idx), dtype=np.int8)
            s2[pa[seg_idx] > ra] = 1
            rb_ok = pb > np.quantile(np.where(np.isfinite(pB_seg), pB_seg, -1.0)[mi], 1 - K)
            s2[rb_ok] = -1
            both = (pa[seg_idx] > ra) & rb_ok
            if both.any():
                s2[both] = np.where(pa[seg_idx][both] - ra >= pb[both] - 0.5,
                                    1, -1).astype(np.int8)
            s = s2
        sig[seg_idx[s != 0]] = s[s != 0]
    return sig

def calibrate(pA_v, pB_v, val_rows):
    months_v = idx[val_rows].to_period("M")
    mv = np.asarray(pd.factorize(months_v)[0])
    n_months_v = len(np.unique(mv))
    best = None
    for K in KS:
        for td in TDS:
            sig = qgate_sig(pA_v, pB_v, mv, K, td, args.mode)
            if not (sig != 0).any():
                continue
            for cd in CDS:
                cfg_c = dict(CFG); cfg_c["cooldown_m1"] = cd
                lg = run_backtest_m1(sig, val_rows, lab, valid, m1_pack, cfg_c)
                n = len(lg)
                if n < MIN_N_CAL:
                    continue
                pnl = float(lg["pnl"].sum())
                mm = metrics(lg)
                wr = float(mm.get("win_rate") or 0); plr = float(mm.get("plr") or 0)
                if not np.isfinite(plr):
                    plr = 0.0
                # 硬约束: 月均>=20笔 (强制放量) + 盈利 + 胜率地板 -> 按 plr 选
                if pnl > 0 and wr >= CFG["wr_floor"] and n / n_months_v >= MIN_PER_MONTH \
                   and (best is None or plr > best[0]):
                    best = (plr, K, td, cd, n)
    if best is None:  # 无盈利候选: K最大+cd最小保交易量
        return KS[-1], TDS[0], CDS[-1], None
    return best[1], best[2], best[3], {"plr": best[0], "n": best[4]}

# ---------------------------------------------------------------- walk-forward
folds = build_folds_m1(idx, CFG, 90)
log(f"folds={len(folds)}")
all_logs, cal_hist = [], []
for fi, f in enumerate(folds):
    fitr, valr, oosr = f["fit_rows"], f["val_rows"], f["oos_rows"]
    wtr, wva = decay_w(fitr), decay_w(valr)
    if args.mode == "dual":
        mA = [fit_seed(X[fitr], y_eat[fitr].astype(np.int8), wtr,
                       X[valr], y_eat[valr].astype(np.int8), wva, s) for s in SEEDS]
        mB = [fit_seed(X[fitr], y_dir[fitr].astype(np.int8), wtr,
                       X[valr], y_dir[valr].astype(np.int8), wva, s) for s in SEEDS]
        pa_v = np.mean([m.predict(X[valr]) for m in mA], axis=0)
        pb_v = np.mean([m.predict(X[valr]) for m in mB], axis=0)
        pa_o = np.mean([m.predict(X[oosr]) for m in mA], axis=0)
        pb_o = np.mean([m.predict(X[oosr]) for m in mB], axis=0)
    else:
        mA = [fit_seed(X[fitr], y_l[fitr].astype(np.int8), wtr,
                       X[valr], y_l[valr].astype(np.int8), wva, s) for s in SEEDS]
        mB = [fit_seed(X[fitr], y_s[fitr].astype(np.int8), wtr,
                       X[valr], y_s[valr].astype(np.int8), wva, s) for s in SEEDS]
        pa_v = np.mean([m.predict(X[valr]) for m in mA], axis=0)
        pb_v = np.mean([m.predict(X[valr]) for m in mB], axis=0)
        pa_o = np.mean([m.predict(X[oosr]) for m in mA], axis=0)
        pb_o = np.mean([m.predict(X[oosr]) for m in mB], axis=0)
    K, td, cd_sel, cal = calibrate(pa_v, pb_v, valr)
    mo_o = np.zeros(len(oosr), dtype=np.int64)   # OOS 一折一月
    sig = qgate_sig(pa_o, pb_o, mo_o, K, td, args.mode)
    cfg_o = dict(CFG); cfg_o["cooldown_m1"] = cd_sel
    cal_hist.append({"fold": f["oos_month"], "K": float(K), "td": float(td),
                     "cd": int(cd_sel), "cal": cal})
    if (sig != 0).any():
        lg = run_backtest_m1(sig, oosr, lab, valid, m1_pack, cfg_o, tag=TAG)
        lg["fold"] = f["oos_month"]
        all_logs.append(lg)
    if fi % 4 == 0 or fi == len(folds) - 1:
        cum = sum(x["pnl"].sum() for x in all_logs)
        n = sum(len(x) for x in all_logs)
        log(f"  fold {f['oos_month']} K={K:.2f} td={td:.2f} cd={cd_sel} cum={cum:+.1f} n={n}")

if not all_logs:
    log("无交易!"); sys.exit(1)
all_logs = pd.concat(all_logs, ignore_index=True)
all_logs.to_csv(os.path.join(OUT, "trades_oos.csv"), index=False)
mm = metrics(all_logs)
pnl = float(all_logs["pnl"].sum())
yr = all_logs.groupby(all_logs["fold"].str[:4])["pnl"].agg(["sum", "count"])
mo_pnl = all_logs.groupby(all_logs["fold"])["pnl"].sum()
sharpe_m = float(mo_pnl.mean() / (mo_pnl.std() + 1e-12) * np.sqrt(12)) if len(mo_pnl) > 1 else float("nan")
summary = {
    "tag": TAG, "mode": args.mode, "feats": args.feats, "seeds": SEEDS, "hl": args.hl,
    "gate": "q_gate_monthly+cd_scan", "n_features": X.shape[1],
    "trades": int(len(all_logs)), "pnl": pnl,
    "wr": float(mm.get("win_rate") or 0), "plr": float(mm.get("plr") or 0),
    "sharpe_monthly_annualized": sharpe_m,
    "by_year": {str(k): [float(v["sum"]), int(v["count"])] for k, v in yr.iterrows()},
    "calibration": cal_hist,
    "hist_target": HIST_TARGET, "gap_to_target": pnl - HIST_TARGET,
}
with open(os.path.join(OUT, "summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
log(f"RESULT {TAG}: n={len(all_logs)} pnl={pnl:+.1f} plr={summary['plr']:.2f} "
    f"wr={summary['wr']*100:.1f}% sharpe={sharpe_m:.2f} | 差距{pnl-HIST_TARGET:+.1f}")
msg = (f"StageC-v2[{TAG}] n={len(all_logs)} pnl={pnl:+.1f}$ plr={summary['plr']:.2f} "
       f"sharpe={sharpe_m:.2f} 差732目标{pnl-HIST_TARGET:+.1f}$")
os.system(f"curl -s -m 20 -d '{msg}' ntfy.sh/{NTFY} > /dev/null 2>&1 &")
