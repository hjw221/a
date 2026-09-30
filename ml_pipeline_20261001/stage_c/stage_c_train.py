"""Stage C — 最终训练 walk-forward (2026-10-01).

用户指令: "最后一步再训练, pnl没有超过732不要停下来"
设计 (承接 Stage A/B 结论):
  特征: StageA白名单18 + StageB因子46 (波动率变化x时段交互为主)
  目标: dual = M_eat(可吃二分类 best>0.6tp) x M_dir(方向超额 pl>ps)
        classic = M_l(out_long==1) x M_s(out_short==1) [四臂口径对照]
  训练: LGB (tuned_params_m1 冻结超参) + 时间衰减权重 hl=180 (A3制胜牌)
  校准: 折内val段 plr_wr 协议 (dual: 网格te x td; classic: 分位网格)
  回测: OOS 2024-08~2026-07, purge90+30, 单持仓+cd10, 与历史完全同口径
用法: python stage_c_train.py --mode dual|classic --feats full|wl --seeds 42,1337
输出: /root/rivermind-data/xauusd/stage_c/{mode}_{feats}_s{n}/
"""
import os, sys, json, time, pickle, argparse
import numpy as np
import pandas as pd

T0 = time.time()
def log(s): print(f"[C {time.time()-T0:6.0f}s] {s}", flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["dual", "classic"], default="dual")
ap.add_argument("--feats", choices=["full", "wl"], default="full")
ap.add_argument("--seeds", default="42")
ap.add_argument("--hl", type=float, default=180.0)
args = ap.parse_args()

BASE_V2 = "/root/rivermind-data/xauusd/v2_ens"
CSV = "/root/rivermind-fs/xauusd/data/XAUUSDc_M1_202201022305_202606262057.csv"
TAG = f"{args.mode}_{args.feats}_s{args.seeds.replace(',', '-')}"
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
       "inner_val_frac": 0.20, "wr_floor": 0.38, "min_trades_cal": 80}
SEEDS = [int(s) for s in args.seeds.split(",")]

# ---------------------------------------------------------------- 加载
log(f"加载 pack ... (mode={args.mode} feats={args.feats} seeds={SEEDS})")
with open(os.path.join(BASE_V2, "cache", "pack_m1sc_ad.pkl"), "rb") as f:
    pack = pickle.load(f)
F, feats = pack["F"], pack["feats"]
lab, valid = pack["lab"], pack["valid"]
tuned = json.load(open(os.path.join(BASE_V2, "results", "tuned_params_m1.json")))["m1sc_ad"]
LGB_P = dict(tuned["long"]["lgb"])   # 双目标复用long侧冻结超参

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

# OHLC (对齐断言)
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

# ---------------------------------------------------------------- 特征矩阵 (wl18 + 因子46)
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
XN = list(X.columns)
X = X.to_numpy(np.float32)
log(f"特征矩阵 {X.shape}")

# ---------------------------------------------------------------- 工具
def decay_w(rows):
    t = (idx[rows].asi8 // 10**9 / 86400.0) if hasattr(idx[rows], "asi8") else \
        np.asarray(idx[rows].astype("int64")) / 1e9 / 86400.0
    t = np.asarray(t, dtype=np.float64)
    return np.exp(-np.log(2.0) * np.maximum(t[-1] - t, 0) / args.hl)

def fit_seed(Xtr, ytr, wtr, Xva, yva, wva, seed):
    p = dict(LGB_P); p.update(seed=seed)
    m, rounds = fit_lgb(Xtr, ytr, wtr, p, Xva, yva, wva, es_rounds=50, max_rounds=600)
    return m

folds = build_folds_m1(idx, CFG, 90)
log(f"folds={len(folds)} purge={folds[0]['purge_bars']}")

def calibrate_dual(p_eat_val, p_dir_val, val_rows):
    best, grid_info = None, []
    for te in [0.55, 0.65, 0.75]:
        for td in [0.0, 0.05, 0.10, 0.15]:
            sig = np.zeros(len(val_rows), dtype=np.int8)
            ok = (p_eat_val > te) & (np.abs(p_dir_val - 0.5) > td)
            sig[ok & (p_dir_val > 0.5)] = 1
            sig[ok & (p_dir_val < 0.5)] = -1
            if not (sig != 0).any():
                continue
            lg = run_backtest_m1(sig, val_rows, lab, valid, m1_pack, CFG)
            n = len(lg)
            if n < CFG["min_trades_cal"]:
                grid_info.append((te, td, n, 0.0, -999.0)); continue
            mm = metrics(lg); pnl = float(lg["pnl"].sum())
            wr = mm.get("win_rate") or 0.0; plr = mm.get("plr") or 0.0
            plr = float(plr) if np.isfinite(plr) else 0.0
            grid_info.append((te, td, n, wr, plr))
            if pnl > 0 and wr >= CFG["wr_floor"] and (best is None or plr > best[0]):
                best = (plr, te, td, n)
    if best is None:
        te, td = 0.75, 0.10
        for gi in grid_info:
            if gi[2] >= CFG["min_trades_cal"] and gi[4] > 0 and \
               (best is None or gi[4] > best[0]):
                best = (gi[4], gi[0], gi[1], gi[2])
    if best is None:
        return te, td, None
    return best[1], best[2], {"plr": best[0], "n": best[3]}

def calibrate_classic(pl_val, ps_val, val_rows):
    qs = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.985]
    best = None
    for q in qs:
        tl, tsv = float(np.quantile(pl_val, q)), float(np.quantile(ps_val, q))
        sig = np.zeros(len(val_rows), dtype=np.int8)
        sig[(pl_val > tl) & (ps_val <= tsv)] = 1
        sig[(ps_val > tsv) & (pl_val <= tl)] = -1
        both = (pl_val > tl) & (ps_val > tsv)
        if both.any():
            sig[both] = np.where(pl_val[both] - tl >= ps_val[both] - tsv, 1, -1).astype(np.int8)
        if not (sig != 0).any():
            continue
        lg = run_backtest_m1(sig, val_rows, lab, valid, m1_pack, CFG)
        n = len(lg)
        if n < CFG["min_trades_cal"] or lg["pnl"].sum() <= 0:
            continue
        mm = metrics(lg); wr = mm.get("win_rate") or 0.0
        plr = mm.get("plr") or 0.0; plr = float(plr) if np.isfinite(plr) else 0.0
        if wr < CFG["wr_floor"]:
            continue
        if best is None or plr > best[0]:
            best = (plr, tl, tsv, n, q)
    if best is None:
        tl, tsv = float(np.quantile(pl_val, 0.975)), float(np.quantile(ps_val, 0.975))
        return tl, tsv, None
    return best[1], best[2], {"plr": best[0], "n": best[3], "q": best[4]}

# ---------------------------------------------------------------- walk-forward
pA = np.full(len(X), np.nan)   # 主概率 (eat 或 long)
pB = np.full(len(X), np.nan)   # 副概率 (dir 或 short)
cal_hist = []
for fi, f in enumerate(folds):
    fitr, valr, oosr = f["fit_rows"], f["val_rows"], f["oos_rows"]
    wtr, wva = decay_w(fitr), decay_w(valr)
    if args.mode == "dual":
        models_A = [fit_seed(X[fitr], y_eat[fitr].astype(np.int8), wtr,
                             X[valr], y_eat[valr].astype(np.int8), wva, s) for s in SEEDS]
        models_B = [fit_seed(X[fitr], y_dir[fitr].astype(np.int8), wtr,
                             X[valr], y_dir[valr].astype(np.int8), wva, s) for s in SEEDS]
        pa_v = np.mean([m.predict(X[valr]) for m in models_A], axis=0)
        pb_v = np.mean([m.predict(X[valr]) for m in models_B], axis=0)
        te, td, cal = calibrate_dual(pa_v, pb_v, valr)
        pA[oosr] = np.mean([m.predict(X[oosr]) for m in models_A], axis=0)
        pB[oosr] = np.mean([m.predict(X[oosr]) for m in models_B], axis=0)
        sig = np.zeros(len(oosr), dtype=np.int8)
        ok = (pA[oosr] > te) & (np.abs(pB[oosr] - 0.5) > td)
        sig[ok & (pB[oosr] > 0.5)] = 1
        sig[ok & (pB[oosr] < 0.5)] = -1
    else:
        models_A = [fit_seed(X[fitr], y_l[fitr].astype(np.int8), wtr,
                             X[valr], y_l[valr].astype(np.int8), wva, s) for s in SEEDS]
        models_B = [fit_seed(X[fitr], y_s[fitr].astype(np.int8), wtr,
                             X[valr], y_s[valr].astype(np.int8), wva, s) for s in SEEDS]
        pa_v = np.mean([m.predict(X[valr]) for m in models_A], axis=0)
        pb_v = np.mean([m.predict(X[valr]) for m in models_B], axis=0)
        tl, tsv, cal = calibrate_classic(pa_v, pb_v, valr)
        pA[oosr] = np.mean([m.predict(X[oosr]) for m in models_A], axis=0)
        pB[oosr] = np.mean([m.predict(X[oosr]) for m in models_B], axis=0)
        sig = np.zeros(len(oosr), dtype=np.int8)
        sig[pA[oosr] > tl] = 1
        sig[pB[oosr] > tsv] = -1
        both = (pA[oosr] > tl) & (pB[oosr] > tsv)
        if both.any():
            sig[both] = np.where(pA[oosr][both] - tl >= pB[oosr][both] - tsv,
                                 1, -1).astype(np.int8)
        te, td = tl, tsv
    cal_hist.append({"fold": f["oos_month"], "window": f["window"],
                     "te": float(te), "td": float(td), "cal": cal})
    if (sig != 0).any():
        lg = run_backtest_m1(sig, oosr, lab, valid, m1_pack, CFG, tag=TAG)
        lg["fold"] = f["oos_month"]
        if fi == 0:
            all_logs = lg
        else:
            all_logs = pd.concat([all_logs, lg], ignore_index=True)
    else:
        if fi == 0:
            all_logs = pd.DataFrame()
    if fi % 4 == 0 or fi == len(folds) - 1:
        log(f"  fold {f['oos_month']} ({f['window']}) te={te:.3f} td={td:.3f} "
            f"cumPnL={all_logs['pnl'].sum() if len(all_logs) else 0:+.1f}")

# ---------------------------------------------------------------- 结果
if len(all_logs) == 0:
    log("无交易!"); sys.exit(1)
all_logs.to_csv(os.path.join(OUT, "trades_oos.csv"), index=False)
mm = metrics(all_logs)
pnl = float(all_logs["pnl"].sum())
yr = all_logs.groupby(all_logs["signal_time"].dt.year)["pnl"].agg(["sum", "count"])
mo_pnl = all_logs.groupby(all_logs["signal_time"].dt.to_period("M"))["pnl"].sum()
sharpe_m = float(mo_pnl.mean() / (mo_pnl.std() + 1e-12) * np.sqrt(12))
summary = {
    "tag": TAG, "mode": args.mode, "feats": args.feats, "seeds": SEEDS, "hl": args.hl,
    "n_features": X.shape[1], "folds": len(folds),
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
    f"wr={summary['wr']*100:.1f}% sharpe={sharpe_m:.2f} | 目标{HIST_TARGET} 差距{pnl-HIST_TARGET:+.1f}")
msg = (f"StageC[{TAG}] n={len(all_logs)} pnl={pnl:+.1f}$ plr={summary['plr']:.2f} "
       f"sharpe={sharpe_m:.2f} 差732目标{pnl-HIST_TARGET:+.1f}$")
os.system(f"curl -s -m 20 -d '{msg}' ntfy.sh/{NTFY} > /dev/null 2>&1 &")
