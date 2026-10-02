"""Stage C v12 — 全天候年度均衡版·趋势共识方向 (2026-10-01).

v11 四臂诚实负结果: 12因子IC方向在全窗口无OOS alpha (双向交易=付点差).
证据链: 方向 AUC 0.579≈噪声, 时机 eat AUC 0.75 才是真信号.

v12 分工 (零拟合参数):
  ML 管时机 (M_eat top-N) + 趋势共识管方向:
    慢制度: sign(dist_ma20) (20日MA上下) × 快确认: |z(momn_1440)|
    共识→该方向 ( conviction 随动量强度分级), 分歧→中性不交易
  牛→多 / 熊→空 / 震荡→空仓: 什么行情都能处理, 无行情不䈐血

协议修复 (反强制交易):
  校准无盈利候选 -> K=0 空仓 (v11 是 K=0.01 硬交易 -> 弱regime流血)
  MIN_N_CAL 120->60, MIN_PER_MONTH 30->15, TDS 加 0.06

  1) OOS 全窗 2022-08 ~ 2026-07 (48折, 与历史732同口径)
  2) hl 消融 180/0, pack 消融 v3/v4
  3) 考核: 每年 PnL 为正 + min_year 最大化 + 总量>732.7
用法: ML_THREADS=7 python3 stage_c_v12.py --hl 180 --pack v3
"""
import os, sys, json, time, pickle, argparse
import numpy as np
import pandas as pd

T0 = time.time()
def log(s): print(f"[V12 {time.time()-T0:6.0f}s] {s}", flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--hl", type=float, default=180.0, help="半衰期天数; <=0 等权")
ap.add_argument("--pack", choices=["v3", "v4"], default="v3")
ap.add_argument("--seeds", default="42,1337,2024")
ap.add_argument("--limit", type=int, default=0, help="冒烟: 只跑前N折")
args = ap.parse_args()

BASE_V2 = "/root/rivermind-data/xauusd/v2_ens"
CSV = "/root/rivermind-fs/xauusd/data/XAUUSDc_M1_202201022305_202606262057.csv"
NTFY = "xauusd-qv7m2zk9-res"
HIST_TARGET = 732.7
ML_THREADS = int(os.environ.get("ML_THREADS", "26"))
PACK_FILE = {"v3": "pack_m1sc_ad_v3.pkl", "v4": "pack_m1sc_ad_v4.pkl"}[args.pack]
PURGE_H = {"v3": 90, "v4": 120}[args.pack]   # v4 H=120 -> purge 更长
TAG = f"v12_hl{int(args.hl)}_{args.pack}_s{args.seeds.replace(',', '-')}"
OUT = f"/root/rivermind-data/xauusd/stage_c/{TAG}"
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, BASE_V2)
from scalp import build_folds_m1, run_backtest_m1
from backtest import metrics
from models import fit_lgb

CFG = {"first_oos_month": "2022-08", "last_oos_month": "2026-07",
       "max_train_months": 36, "scalp_embargo_m1": 30, "cooldown_m1": 10,
       "inner_val_frac": 0.20, "wr_floor": 0.36, "min_trades_cal": 60}
SEEDS = [int(s) for s in args.seeds.split(",")]
KS = [0.01, 0.02, 0.05, 0.10, 0.15]   # 精确top-N比例
TDS = [0.0, 0.02, 0.04, 0.06]         # 趋势共识置信门槛 (pb 0.5±0.1·|tanh|)
CDS = [10, 4]                          # 冷却
MIN_N_CAL = 60
MIN_PER_MONTH = 15

# ---------------------------------------------------------------- 加载
log(f"加载 pack={args.pack} ...")
with open(os.path.join(BASE_V2, "cache", PACK_FILE), "rb") as f:
    pack = pickle.load(f)
F, feats = pack["F"], pack["feats"]
lab, valid = pack["lab"], pack["valid"]
tuned = json.load(open(os.path.join(BASE_V2, "results", "tuned_params_m1.json")))["m1sc_ad"]
LGB_P = dict(tuned["long"]["lgb"]); LGB_P["num_threads"] = ML_THREADS

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
y_dir_target = np.where(valid, (pl - ps) / tp, np.nan)

raw = pd.read_csv(CSV, sep="\t",
                  usecols=["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>"])
_dt = raw["<DATE>"].map(dmap)
_ss = raw["<TIME>"].str.slice(0, 2).astype(np.int32) * 3600 + \
      raw["<TIME>"].str.slice(3, 5).astype(np.int32) * 60
_ts = _dt + pd.to_timedelta(_ss, unit="s")
_order = np.argsort(_ts.to_numpy(), kind="stable")
assert (_ts.iloc[_order].to_numpy() == idx.to_numpy()).all()
m1_t = np.asarray(idx.astype("datetime64[ns]").astype("int64") // 10**9 // 60,
                  dtype=np.int64)
m1_pack = (m1_t,
           raw["<OPEN>"].to_numpy(np.float64)[_order],
           raw["<HIGH>"].to_numpy(np.float64)[_order],
           raw["<LOW>"].to_numpy(np.float64)[_order],
           raw["<CLOSE>"].to_numpy(np.float64)[_order])
del raw, _dt, _ss, _ts

# ---------------------------------------------------------------- 特征 (同 v10)
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
X = pd.concat([F[WL].astype(np.float32), FM.astype(np.float32)], axis=1).to_numpy(np.float32)
log(f"特征矩阵 {X.shape}")

# ---------------------------------------------------------------- 趋势共识方向 (零拟合参数)
log("趋势共识方向: sign(close-MA20d) x mom1d/10d_std ...")
mom1d = c["momn_1440"].astype(np.float64)          # 1日动量(价格单位)
m_sd = mom1d.rolling(14400, min_periods=7200).std()   # 10日动量分布
momz = np.nan_to_num((mom1d / m_sd.replace(0.0, np.nan)).clip(-5, 5).to_numpy(np.float64), nan=0.0)
close_s = pd.Series(m1_pack[4], index=idx)         # 已排序 close
ma20 = close_s.rolling(27600, min_periods=13800).mean()   # 20交易日均线
ma_sign = np.sign(np.nan_to_num((close_s - ma20).to_numpy(np.float64), nan=0.0))
agree = np.sign(momz) == ma_sign                       # 动量确认制度方向
pb_arr = 0.5 + ma_sign * 0.1 * np.abs(np.tanh(momz)) * agree
# 共识: pb 偏向制度方向, 强度随动量分级; 分歧/无动量: pb=0.5 中性(td>0 滤掉)
log(f"方向分布: 多共识 {(pb_arr > 0.5).sum()} / 空共识 {(pb_arr < 0.5).sum()} / 中性 {(pb_arr == 0.5).sum()}")

def decay_w(rows):
    if args.hl <= 0:
        return np.ones(len(rows), dtype=np.float64)
    t = idx[rows].to_numpy().astype("datetime64[ns]").astype(np.float64) / 1e9 / 86400.0
    return np.exp(-np.log(2.0) * np.maximum(t[-1] - t, 0) / args.hl)

def fit_seed(Xtr, ytr, wtr, Xva, yva, wva, seed):
    p = dict(LGB_P); p.update(seed=seed)
    m, _ = fit_lgb(Xtr, ytr, wtr, p, Xva, yva, wva, es_rounds=50, max_rounds=600)
    return m

# ---------------------------------------------------------------- q_gate (dual)
def qgate_sig(pA_seg, pB_seg, months_seg, K, td):
    sig = np.zeros(len(pA_seg), dtype=np.int8)
    for m in np.unique(months_seg):
        mi = months_seg == m
        n_m = int(K * mi.sum())
        if n_m < 1:
            continue
        pa = np.where(np.isfinite(pA_seg), pA_seg, -1.0)
        seg_idx = np.where(mi)[0]
        seg_idx = seg_idx[np.argsort(-pa[seg_idx])[:n_m]]
        pb = pB_seg[seg_idx]
        s = np.zeros(len(seg_idx), dtype=np.int8)
        ok = np.abs(pb - 0.5) > td
        s[ok & (pb > 0.5)] = 1
        s[ok & (pb < 0.5)] = -1
        sig[seg_idx[s != 0]] = s[s != 0]
    return sig

def calibrate(pA_v, pB_v, val_rows):
    months_v = idx[val_rows].to_period("M")
    mv = np.asarray(pd.factorize(months_v)[0])
    n_months_v = len(np.unique(mv))
    best = None
    for K in KS:
        for td in TDS:
            sig = qgate_sig(pA_v, pB_v, mv, K, td)
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
                if os.environ.get("DEBUG_CAL"):
                    log(f"    cal? K={K:.2f} td={td:.2f} cd={cd} n={n} pnl={pnl:+.1f} wr={wr:.2f}")
                if pnl > 0 and wr >= 0.33 and n / n_months_v >= MIN_PER_MONTH \
                   and (best is None or pnl > best[5]):
                    best = (plr, K, td, cd, n, pnl)
    if best is None:
        return 0.0, TDS[0], CDS[0], None   # 空仓: 无盈利候选不硬交易 (v11教训)
    return best[1], best[2], best[3], {"plr": best[0], "n": best[4], "pnl": best[5]}

# ---------------------------------------------------------------- walk-forward
folds = build_folds_m1(idx, CFG, PURGE_H)
if args.limit:
    folds = folds[:args.limit]
log(f"folds={len(folds)} ({folds[0]['oos_month']}~{folds[-1]['oos_month']}) "
    f"hl={args.hl} pack={args.pack} threads={ML_THREADS}")
all_logs, cal_hist = [], []
for fi, f in enumerate(folds):
    fitr, valr, oosr = f["fit_rows"], f["val_rows"], f["oos_rows"]
    wtr, wva = decay_w(fitr), decay_w(valr)
    mA = [fit_seed(X[fitr], y_eat[fitr].astype(np.int8), wtr,
                   X[valr], y_eat[valr].astype(np.int8), wva, s) for s in SEEDS]
    pa_v = np.mean([m.predict(X[valr]) for m in mA], axis=0)
    pa_o = np.mean([m.predict(X[oosr]) for m in mA], axis=0)
    pb_v = pb_arr[valr]
    pb_o = pb_arr[oosr]
    K, td, cd_sel, cal = calibrate(pa_v, pb_v, valr)
    mo_o = np.zeros(len(oosr), dtype=np.int64)
    sig = qgate_sig(pa_o, pb_o, mo_o, K, td)
    cfg_o = dict(CFG); cfg_o["cooldown_m1"] = cd_sel
    cal_hist.append({"fold": f["oos_month"], "K": float(K), "td": float(td),
                     "cd": int(cd_sel), "cal": cal})
    if (sig != 0).any():
        lg = run_backtest_m1(sig, oosr, lab, valid, m1_pack, cfg_o, tag=TAG)
        lg["fold"] = f["oos_month"]
        all_logs.append(lg)
    if fi % 6 == 0 or fi == len(folds) - 1:
        cum = sum(x["pnl"].sum() for x in all_logs)
        n = sum(len(x) for x in all_logs)
        nlong = int(sum((x["dir"] == "long").sum() for x in all_logs))
        log(f"  fold {f['oos_month']} K={K:.2f} td={td:.2f} cd={cd_sel} "
            f"cum={cum:+.1f} n={n} (L{nlong}/S{n-nlong})")

if not all_logs:
    log("无交易!"); sys.exit(1)
all_logs = pd.concat(all_logs, ignore_index=True)
all_logs.to_csv(os.path.join(OUT, "trades_oos.csv"), index=False)
mm = metrics(all_logs)
pnl = float(all_logs["pnl"].sum())
yr = all_logs.groupby(all_logs["fold"].str[:4])["pnl"].agg(["sum", "count"])
mo_pnl = all_logs.groupby(all_logs["fold"])["pnl"].sum()
sharpe_m = float(mo_pnl.mean() / (mo_pnl.std() + 1e-12) * np.sqrt(12)) if len(mo_pnl) > 1 else float("nan")
nlong = int((all_logs["dir"] == "long").sum()); nshort = int((all_logs["dir"] == "short").sum())
by_year = {str(k): [float(v["sum"]), int(v["count"])] for k, v in yr.iterrows()}
ys = all_logs.groupby([all_logs["fold"].str[:4], all_logs["dir"]])["pnl"].sum()
by_year_side = {f"{k[0]}_{k[1]}": round(float(v), 2) for k, v in ys.items()}
min_year = min(v[0] for v in by_year.values())
mo_pnl.to_csv(os.path.join(OUT, "monthly_pnl.csv"))
months_traded = int(len(mo_pnl)); months_flat = len(folds) - months_traded
summary = {
    "tag": TAG, "mode": "trend_agree_dir", "pack": args.pack, "seeds": SEEDS,
    "hl": args.hl, "n_features": X.shape[1], "purge_h": PURGE_H,
    "oos_window": f"{folds[0]['oos_month']}~{folds[-1]['oos_month']}",
    "trades": int(len(all_logs)), "pnl": pnl,
    "wr": float(mm.get("win_rate") or 0), "plr": float(mm.get("plr") or 0),
    "sharpe_monthly_annualized": sharpe_m,
    "side": {"long": nlong, "short": nshort},
    "months_traded": months_traded, "months_flat": months_flat,
    "by_year": by_year, "by_year_side": by_year_side, "min_year": float(min_year),
    "calibration": cal_hist,
    "hist_target": HIST_TARGET, "gap_to_target": pnl - HIST_TARGET,
    "champion_criteria": {"all_years_positive": all(v[0] > 0 for v in by_year.values()),
                           "total_gt_732": pnl > HIST_TARGET},
}
with open(os.path.join(OUT, "summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
yr_str = " | ".join(f"{k}:{v[0]:+.0f}({v[1]})" for k, v in by_year.items())
log(f"RESULT {TAG}: n={len(all_logs)}(L{nlong}/S{nshort}) pnl={pnl:+.1f} "
    f"plr={summary['plr']:.2f} sharpe={sharpe_m:.2f} min_year={min_year:+.1f} "
    f"月交易{months_traded}/空仓{months_flat}")
log(f"BY_YEAR {yr_str}")
msg = (f"V12[{TAG}] pnl={pnl:+.1f}$ sharpe={sharpe_m:.2f} L/S={nlong}/{nshort} "
       f"minYr={min_year:+.0f} flat={months_flat}mo | {yr_str}")
os.system(f"curl -s -m 20 -d '{msg}' ntfy.sh/{NTFY} > /dev/null 2>&1 &")
