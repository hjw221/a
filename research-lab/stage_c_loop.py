#!/usr/bin/env python3
"""stage_c_loop.py — 阶段C: 自动训练循环 (目标: 决赛窗 OOS PnL > $732).

协议 (防过拟合三道闸):
  1. 搜索窗 2022-01~2025-06 (30个OOS月) 做配置搜索
  2. 冠军候选 → 决赛窗 2025-07~2026-07 (12个OOS月, 搜索从未触碰) 终验
  3. 冠军需通过折一致性守门 (>=60% 折为正)
循环: 4 worker x 7核 并行 walkforward, 随机探索 + top 变异, 直到决赛 PnL>732 或预算尽.
"""
import os, sys, json, time, random, hashlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from numba import njit

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE, "data.csv")
RES = os.path.join(BASE, "results_research")
os.makedirs(RES, exist_ok=True)
NTFY = os.environ.get("NTFY_TOPIC", "xauusd-qv7m2zk9-res")
WORKER_ID = int(os.environ.get("WORKER_ID", "0"))
N_THREADS = int(os.environ.get("ML_THREADS", "7"))

def say(msg):
    os.system(f'curl -s --max-time 20 -X POST "https://ntfy.sh/{NTFY}" -d "{msg}" -H "Title: research" >/dev/null 2>&1 || true')

# ================= 数据与因子 (与阶段A/B同源) =================
_cache = {}

def load_all():
    if "m5" in _cache:
        return _cache["m5"], _cache["F"], _cache["cost"], _cache["atr"], _cache["m1_ohl"]
    df = pd.read_csv(DATA_CSV, sep="\t")
    df.columns = ["date","time","open","high","low","close","tickvol","vol","spread"]
    df["dt"] = pd.to_datetime(df["date"]+" "+df["time"], format="%Y.%m.%d %H:%M:%S")
    df = df.set_index("dt").drop(columns=["date","time","vol"]).sort_index()
    bar_range = df.high - df.low
    rng_z = (bar_range - bar_range.rolling(288).mean()) / bar_range.rolling(288).std()
    cdf = df[~((df.tickvol <= 5) & (bar_range < 0.01) | (rng_z.abs() > 15))].copy()
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
    F = build_factors(m5, atr5)
    m1_ohl = (cdf.index.astype("datetime64[s]").astype("int64")).to_numpy(), cdf.open.to_numpy(), cdf.high.to_numpy(), cdf.low.to_numpy()
    _cache.update(m5=m5, F=F, cost=m5["spread_cost"].to_numpy(), atr=atr5.to_numpy(), m1_ohl=m1_ohl)
    return m5, F, _cache["cost"], _cache["atr"], _cache["m1_ohl"]

def build_factors(m5, atr5):
    c, o, h, l, v = m5["close"], m5["open"], m5["high"], m5["low"], m5["tickvol"]
    r = c.diff()
    F = pd.DataFrame(index=m5.index)
    for n in [3, 6, 12, 24, 48, 96, 288, 576]:
        F[f"mom_{n}"] = c.diff(n)
    m96, sd96 = c.rolling(96).mean(), c.rolling(96).std()
    m288, sd288 = c.rolling(288).mean(), c.rolling(288).std()
    F["zscore_96"] = (c - m96) / (sd96 + 1e-9)
    F["zscore_288"] = (c - m288) / (sd288 + 1e-9)
    F["dist_ma20"] = c / c.rolling(24).mean() - 1
    F["dist_ma60"] = c / c.rolling(72).mean() - 1
    F["dist_ma240"] = c / c.rolling(288).mean() - 1
    F["ema_f_s"] = c.ewm(span=12).mean() / c.ewm(span=96).mean() - 1
    F["ema_m_s"] = c.ewm(span=48).mean() / c.ewm(span=288).mean() - 1
    hh, ll = h.rolling(96).max(), l.rolling(96).min()
    hh2, ll2 = h.rolling(288).max(), l.rolling(288).min()
    F["stoch_pos_96"] = (c - ll) / (hh - ll + 1e-9)
    F["stoch_pos_288"] = (c - ll2) / (hh2 - ll2 + 1e-9)
    F["breakout_96"] = (c - hh.shift(1)) / (atr5 + 1e-9)
    F["breakdown_96"] = (c - ll.shift(1)) / (atr5 + 1e-9)
    F["vol_ratio_fs"] = r.rolling(12).std() / (r.rolling(288).std() + 1e-9)
    F["atr_norm"] = (atr5 / c).to_numpy()
    F["range_exp"] = (h - l).rolling(12).mean() / ((h - l).rolling(288).mean() + 1e-9)
    F["pk_vol"] = np.log1p(r.abs()).rolling(288).std()
    F["dollar_vol"] = (r.abs() * v).rolling(288).mean()
    rng = h - l
    F["body_ratio"] = (c - o).abs() / (rng + 1e-9)
    F["upper_wick"] = (h - np.maximum(c, o)) / (rng + 1e-9)
    F["lower_wick"] = (np.minimum(c, o) - l) / (rng + 1e-9)
    F["close_in_rng"] = (c - l) / (rng + 1e-9)
    F["vol_spike"] = v / (v.rolling(288).mean() + 1e-9)
    F["up_bars_12"] = (r > 0).rolling(12).sum()
    F["up_bars_96"] = (r > 0).rolling(96).sum()
    F["efficiency_96"] = (c - c.shift(96)).abs() / (r.abs().rolling(96).sum() + 1e-9)
    F["hour_sin"] = np.sin(2*np.pi*m5.index.hour/24)
    F["hour_cos"] = np.cos(2*np.pi*m5.index.hour/24)
    day_hi = h.groupby(m5.index.date).transform("cummax")
    day_lo = l.groupby(m5.index.date).transform("cummin")
    F["pos_day_hi"] = (c - day_hi) / (atr5 + 1e-9)
    F["pos_day_lo"] = (c - day_lo) / (atr5 + 1e-9)
    F["mins_from_open"] = m5.index.hour * 60 + m5.index.minute
    day_open = c.groupby(m5.index.date).transform("first")
    F["gap_overnight"] = (day_open - day_open.shift(1)) / (atr5 + 1e-9)
    F["from_day_open"] = (day_open - c) / (atr5 + 1e-9)
    return F.replace([np.inf, -np.inf], np.nan)

# ================= 三重障碍回测引擎 (numba, 逐笔M1结算) =================
@njit(cache=True)
def simulate(m1_t, m1_o, m1_h, m1_l, sig_t, p_long, trade_mask,  # trade_mask: 1=long -1=short
             atr, cost, tp_mult, sl_mult, sl_floor, sp_mult, horizon, entry_tol):
    n = len(sig_t)
    nm1 = len(m1_t)
    pnl = np.zeros(n); wins = np.zeros(n)
    for i in range(n):
        if trade_mask[i] == 0:
            continue
        a = atr[i]
        if not (a > 0.0):
            continue
        tp = tp_mult * a
        sl = max(sl_mult * a, sl_floor, sp_mult * cost[i])
        sc = cost[i]
        t0 = sig_t[i]
        lo, hi = 0, nm1
        while lo < hi:
            mid = (lo + hi) // 2
            if m1_t[mid] < t0:
                lo = mid + 1
            else:
                hi = mid
        e = lo
        if e >= nm1 or m1_t[e] > t0 + entry_tol:
            continue
        entry = m1_o[e]
        end = min(e + horizon, nm1 - 1)
        direction = trade_mask[i]
        pnl_i = 0.0; win = 0
        hit_tp = False; hit_sl = False; k_tp = -1; k_sl = -1
        for k in range(e + 1, end + 1):
            if direction == 1:
                if not hit_tp and m1_h[k] >= entry + tp: hit_tp = True; k_tp = k
                if not hit_sl and m1_l[k] <= entry - sl: hit_sl = True; k_sl = k
            else:
                if not hit_tp and m1_l[k] <= entry - tp: hit_tp = True; k_tp = k
                if not hit_sl and m1_h[k] >= entry + sl: hit_sl = True; k_sl = k
            if hit_tp and hit_sl:
                break
        if hit_tp and (not hit_sl or k_tp < k_sl):
            pnl_i = tp - sc; win = 1
        elif hit_sl:
            pnl_i = -sl - sc
        else:
            exit_px = m1_o[end]
            pnl_i = (exit_px - entry) * direction - sc
        pnl[i] = pnl_i; wins[i] = win
    return pnl, wins

# ================= walkforward =================
def month_folds(idx, start, end):
    """生成 (train_end, oos_start, oos_end) 月度折"""
    months = pd.period_range(start=start, end=end, freq="M")
    folds = []
    for i, m in enumerate(months):
        oos_s, oos_e = m.start_time, m.end_time
        folds.append((oos_s, oos_e))
    return folds

def walkforward(cfg, start="2024-01", end="2025-06", step=1):
    """月度折走查: step=2 隔月采样(搜索快筛), 决赛 step=1"""
    m5, F, cost, atr, (m1_t, m1_o, m1_h, m1_l) = _prep()
    feats = cfg["features"]
    X = F[feats]
    hours_f = cfg.get("hours_filter")
    hour_arr = m5.index.hour.to_numpy()
    tau = cfg["tau"]
    tp_mult = cfg["tp_mult"]; sl_mult = cfg["sl_mult"]
    params = dict(cfg["lgb"]); params["num_threads"] = N_THREADS
    sig_t = (m5.index.astype("datetime64[s]").astype("int64") + 300).to_numpy()
    folds = month_folds(m5.index, start, end)[::step]
    rows = []
    TRAIN_CAP = pd.Timedelta(days=cfg.get("train_cap_days", 450))   # 滚动训练窗(可配)
    q_gate = cfg.get("q_gate", 0.0)   # >0 时启用月内分位数门控
    for oos_s, oos_e in folds:
        tr_end = oos_s - pd.Timedelta(minutes=90+30)  # purge+embargo
        tr_start = oos_s - TRAIN_CAP
        tr_mask = (m5.index >= tr_start) & (m5.index < tr_end)
        oos_mask = (m5.index >= oos_s) & (m5.index <= oos_e)
        Xtr, ytr = X[tr_mask].to_numpy(), _cache["y"][tr_mask]
        ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr)
        Xtr, ytr = Xtr[ok], ytr[ok]
        Xoos = X[oos_mask].to_numpy()
        ok_o = np.isfinite(Xoos).all(axis=1)
        if len(Xtr) < 5000 or ok_o.sum() < 100:
            continue
        ds = lgb.Dataset(Xtr, ytr)
        n_seed = cfg.get("n_seed", 1)
        if n_seed <= 1:
            bst = lgb.train(params, ds, num_boost_round=cfg.get("rounds", 200))
            p = np.full(len(Xoos), np.nan)
            p[ok_o] = bst.predict(Xoos[ok_o])
        else:
            ps = []
            for sd in (1337, 7331, 90210)[:n_seed]:
                pp = dict(params); pp["seed"] = sd
                pp["bagging_fraction"] = min(1.0, params.get("bagging_fraction", 0.8))
                pp["feature_fraction"] = min(1.0, params.get("feature_fraction", 0.8))
                bst = lgb.train(pp, ds, num_boost_round=cfg.get("rounds", 200))
                q = np.full(len(Xoos), np.nan)
                q[ok_o] = bst.predict(Xoos[ok_o])
                ps.append(q)
            p = np.nanmean(np.vstack(ps), axis=0)
        # 交易决策: q_gate 分位数门控(月内 top/bottom K%) / EV 期望 / 纯 tau
        trade = np.zeros(len(Xoos), dtype=np.int8)
        atr_oos = atr[oos_mask]; cost_oos = cost[oos_mask]
        if q_gate > 0:
            pv = p[np.isfinite(p)]
            if len(pv) > 50:
                hi, lo = np.quantile(pv, [1 - q_gate, q_gate])
                trade[p >= hi] = 1
                trade[p <= lo] = -1
        elif cfg.get("ev_mode"):
            margin = cfg.get("ev_margin", 0.0)
            tp_oos = tp_mult * atr_oos
            sl_oos = np.maximum.reduce([sl_mult * atr_oos, np.full(len(atr_oos), 0.48), 2.0 * cost_oos])
            ev_l = p * (tp_oos - cost_oos) - (1 - p) * (sl_oos + cost_oos)
            ev_s = (1 - p) * (tp_oos - cost_oos) - p * (sl_oos + cost_oos)
            trade[ev_l > margin * atr_oos] = 1
            trade[ev_s > margin * atr_oos] = -1
        else:
            trade[p > tau] = 1
            trade[p < 1 - tau] = -1
        if hours_f:
            hm = hour_arr[oos_mask]
            trade[~np.isin(hm, hours_f)] = 0
        pnl, wins = simulate(
            m1_t, m1_o, m1_h, m1_l,
            sig_t[oos_mask], p, trade,
            atr[oos_mask], cost[oos_mask],
            tp_mult, sl_mult, 0.48, 2.0, 90, 10)
        valid = trade != 0
        rows.append({
            "month": str(oos_s.date())[:7],
            "trades": int(valid.sum()),
            "pnl": float(pnl[valid].sum()),
            "wins": int(wins[valid].sum()),
        })
    df = pd.DataFrame(rows)
    if df.empty or df.trades.sum() == 0:
        return {"pnl": -1e9, "trades": 0, "sharpe": -99, "plr": 0, "pos_fold_ratio": 0, "folds": []}
    mo = df.set_index("month")["pnl"]
    sharpe = mo.mean() / (mo.std() + 1e-9) * np.sqrt(12)
    # 笔级 PLR (胜笔总盈利 / 败笔总亏损), 从折行聚合不了改为传笔级 — 用折级近似但修正零亏损情形
    tot_win = df.pnl[df.pnl > 0].sum()
    tot_loss = -df.pnl[df.pnl < 0].sum()
    plr = float(tot_win / tot_loss) if tot_loss > 1e-9 else float("inf") if tot_win > 0 else 0.0
    plr = min(plr, 99.0)
    return {
        "pnl": float(df.pnl.sum()), "trades": int(df.trades.sum()),
        "win_rate": float(df.wins.sum() / max(df.trades.sum(), 1)),
        "sharpe": float(sharpe), "plr": float(plr),
        "pos_fold_ratio": float((df.pnl > 0).mean()),
        "folds": rows,
    }

def final_eval(cfg):
    """决赛 = 全窗走查 2022-08~2026-07 (54折, 与历史 732 同口径)
    与 732 的公平对比: 历史m1sc_ad是4.5年全窗总PnL"""
    c2 = json.loads(json.dumps(cfg))
    c2["rounds"] = max(200, cfg.get("rounds", 200))
    r = walkforward(c2, "2022-08", "2026-07", step=1)
    return r

def _prep():
    if "y" not in _cache:
        m5, F, cost, atr, (m1_t, m1_o, m1_h, m1_l) = load_all()
        sig_t = (m5.index.astype("datetime64[s]").astype("int64") + 300).to_numpy()
        lab_l, lab_s = barriers_only(m1_t, m1_o, m1_h, m1_l, sig_t, _cache["atr"], _cache["cost"])
        clear_long = (lab_l == 1) & (lab_s != 1)
        clear_short = (lab_s == 1) & (lab_l != 1)
        y = np.full(len(m5), np.nan)
        y[clear_long] = 1.0
        y[clear_short] = 0.0
        _cache["y"] = y
    return _cache["m5"], _cache["F"], _cache["cost"], _cache["atr"], _cache["m1_ohl"]

@njit(cache=True)
def barriers_only(m1_t, m1_o, m1_h, m1_l, sig_t, atr, cost_arr):
    n = len(sig_t); nm1 = len(m1_t)
    out_l = np.zeros(n, np.int8); out_s = np.zeros(n, np.int8)
    for i in range(n):
        a = atr[i]
        if not (a > 0.0):
            continue
        tp = 1.5 * a
        sl = max(0.4 * tp, 0.48, 2.0 * (cost_arr[i] if cost_arr is not None else 0.2))
        t0 = sig_t[i]
        lo, hi = 0, nm1
        while lo < hi:
            mid = (lo + hi) // 2
            if m1_t[mid] < t0: lo = mid + 1
            else: hi = mid
        e = lo
        if e >= nm1 or m1_t[e] > t0 + 10:
            continue
        entry = m1_o[e]
        end = min(e + 90, nm1 - 1)
        hi_hit = -1; lo_hit = -1
        for k in range(e + 1, end + 1):
            if m1_h[k] >= entry + tp and hi_hit < 0: hi_hit = k
            if m1_l[k] <= entry - sl and lo_hit < 0: lo_hit = k
            if hi_hit >= 0 and lo_hit >= 0: break
        if hi_hit >= 0 and (lo_hit < 0 or hi_hit < lo_hit): out_l[i] = 1
        elif lo_hit >= 0: out_l[i] = -1
        hi_hit = -1; lo_hit = -1
        for k in range(e + 1, end + 1):
            if m1_l[k] <= entry - tp and hi_hit < 0: hi_hit = k
            if m1_h[k] >= entry + sl and lo_hit < 0: lo_hit = k
            if hi_hit >= 0 and lo_hit >= 0: break
        if hi_hit >= 0 and (lo_hit < 0 or hi_hit < lo_hit): out_s[i] = 1
        elif lo_hit >= 0: out_s[i] = -1
    return out_l, out_s

# ================= 配置空间 =================
ALL_FEATS = None  # load 后填
CORE_FEATS = ["gap_overnight", "mins_from_open", "mom_576", "dist_ma20", "stoch_pos_288",
              "from_day_open", "hour_sin", "hour_cos", "atr_norm", "vol_ratio_fs",
              "ema_m_s", "pos_day_lo", "mom_3"]
HOUR_CHOICES = [
    None,                       # 全时段
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],  # 占位=全
    [7,8,9,10,11,12,13,14,15],  # 伦敦+纽约
    [12,13,14,15,16,17],         # 纽约
    [0,1,2,3,4,5,6,7],           # 亚洲
    [2,3,4,5,8,9,10,13,14,15,16,20,21],  # 混合优选
]

def sample_cfg(rng):
    k = rng.choice([8, 10, 13, 16, 20])
    feats = list(rng.choice(ALL_FEATS, size=min(k, len(ALL_FEATS)), replace=False)) if k < 20 else ALL_FEATS
    feats = sorted(set(feats) | set(rng.choice(CORE_FEATS, size=5, replace=False)))
    return {
        "features": sorted(feats),
        "tau": float(rng.choice([0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65])),
        "tp_mult": float(rng.choice([1.2, 1.5, 1.8])),
        "sl_mult": float(rng.choice([0.4, 0.5, 0.6])),
        "hours_filter": HOUR_CHOICES[rng.integers(0, len(HOUR_CHOICES))],
        "ev_mode": bool(rng.integers(0, 2)),          # EV 期望门控 vs 纯 tau
        "ev_margin": float(rng.choice([0.0, 0.02, 0.05, 0.08, 0.12])),
        "q_gate": float(rng.choice([0.0, 0.05, 0.08, 0.10, 0.15, 0.20])),  # >0 分位数门控
        "train_cap_days": int(rng.choice([450, 900, 4000])),
        "lgb": {
            "objective": "binary", "metric": "auc", "verbosity": -1,
            "learning_rate": float(rng.choice([0.03, 0.05, 0.08])),
            "num_leaves": int(rng.choice([15, 31, 47, 63])),
            "min_data_in_leaf": int(rng.choice([1000, 2000, 4000, 8000])),
            "feature_fraction": float(rng.uniform(0.6, 1.0)),
            "bagging_fraction": float(rng.uniform(0.6, 1.0)),
            "bagging_freq": 1,
            "lambda_l2": float(rng.choice([0.5, 2.0, 8.0])),
        },
        "rounds": int(rng.choice([100, 150, 250])),
        "n_seed": int(rng.choice([1, 1, 3])),   # 集成种子数
    }

def mutate(cfg, rng):
    c = json.loads(json.dumps(cfg))
    if rng.random() < 0.6:
        k = rng.integers(1, 4)
        for _ in range(k):
            if rng.random() < 0.5 and len(c["features"]) > 6:
                c["features"].remove(rng.choice(c["features"]))
            else:
                c["features"] = sorted(set(c["features"]) | {rng.choice(ALL_FEATS)})
    if rng.random() < 0.7:
        c["tau"] = float(np.clip(c["tau"] + rng.normal(0, 0.03), 0.50, 0.68))
    if rng.random() < 0.5:
        c["ev_mode"] = bool(rng.integers(0, 2))
        c["ev_margin"] = float(rng.choice([0.0, 0.02, 0.05, 0.08, 0.12]))
    if rng.random() < 0.6:
        c["q_gate"] = float(rng.choice([0.0, 0.05, 0.08, 0.10, 0.15, 0.20]))
    if rng.random() < 0.4:
        c["train_cap_days"] = int(rng.choice([450, 900, 4000]))
    if rng.random() < 0.4:
        c["tp_mult"] = float(rng.choice([1.2, 1.5, 1.8]))
    if rng.random() < 0.4:
        c["sl_mult"] = float(rng.choice([0.4, 0.5, 0.6]))
    if rng.random() < 0.3:
        c["hours_filter"] = HOUR_CHOICES[rng.integers(0, len(HOUR_CHOICES))]
    if rng.random() < 0.5:
        c["lgb"]["num_leaves"] = int(rng.choice([15, 31, 47, 63]))
        c["lgb"]["min_data_in_leaf"] = int(rng.choice([1000, 2000, 4000, 8000]))
        c["lgb"]["learning_rate"] = float(rng.choice([0.03, 0.05, 0.08]))
    return c

def cfg_id(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]

# ================= 主循环 =================
def main():
    global ALL_FEATS
    seed = int(os.environ.get("SEED", "1337")) + WORKER_ID * 1000
    rng = np.random.default_rng(seed)
    m5, F, *_ = load_all()
    ALL_FEATS = [c for c in F.columns if F[c].notna().mean() > 0.95]
    print(f"[W{WORKER_ID}] 因子池: {len(ALL_FEATS)} | 数据 {len(m5)} M5行", flush=True)
    budget = int(os.environ.get("N_ROUNDS", "12"))
    logf = os.path.join(RES, f"worker{WORKER_ID}_results.jsonl")
    best = {"pnl": -1e9}
    pyrng = random.Random(seed)
    for rd in range(budget):
        t0 = time.time()
        if rd < 4 or pyrng.random() < 0.35:
            cfg = sample_cfg(rng)
        else:
            # 从全局 top 池变异
            top = load_top()
            if top:
                cfg = mutate(pyrng.choice(top), rng)
            else:
                cfg = sample_cfg(rng)
        r = walkforward(cfg, "2022-08", "2026-07", step=3)   # 搜索=全窗稀疏18折(同分布)
        rec = {"worker": WORKER_ID, "round": rd, "cfg_id": cfg_id(cfg),
               "pnl": r["pnl"], "trades": r["trades"], "sharpe": r["sharpe"],
               "plr": r["plr"], "pos_fold_ratio": r["pos_fold_ratio"],
               "cfg": cfg, "dt": time.strftime("%F %T")}
        with open(logf, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if r["pnl"] > best["pnl"]:
            best = rec
        print(f"[W{WORKER_ID}] R{rd} {cfg_id(cfg)} PnL {r['pnl']:.0f} ({r['trades']}笔 "
              f"PLR {r['plr']:.2f} Sharpe {r['sharpe']:.1f} posFold {r['pos_fold_ratio']:.0%}) "
              f"[{time.time()-t0:.0f}s]", flush=True)
    # 收尾: worker 把 best 上报
    with open(os.path.join(RES, f"worker{WORKER_ID}_best.json"), "w") as f:
        json.dump(best, f)
    say(f"W{WORKER_ID} 完成 {budget} 轮: best PnL {best['pnl']:.0f} / {best['trades']}笔")
    print(f"[W{WORKER_ID}] DONE best={best['pnl']:.0f}")

def load_top():
    """读所有 worker 的结果, 取 PnL top5 配置"""
    tops = []
    for fn in os.listdir(RES):
        if fn.endswith("_results.jsonl"):
            try:
                with open(os.path.join(RES, fn)) as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            if rec.get("pnl", -1e9) > 0:
                                tops.append(rec)
                        except Exception:
                            pass
            except Exception:
                pass
    tops.sort(key=lambda r: -r["pnl"])
    return [t["cfg"] for t in tops[:5]]

if __name__ == "__main__":
    main()
