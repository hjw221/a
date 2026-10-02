#!/usr/bin/env python3
"""v17_htf.py — 放弃M1微观择时: M1 -> M15/M30/H1 降采样 + 纯规则突破基准 (正确结算).

用户指令 (2026-10-02):
  1. 数据降采样至 M15/M30/H1 — H1 单根 Bar ATR $8~$20, 点差+滑点(按$0.50计)只占2%~5%
  2. 纯规则突破 (Keltner/Donchian) 为第一基准: 突破入场 + 结构支撑/ATR移动出场(吊灯)
     重点检验: 无回望正确撮合下, 扣除 0.3xATR 悲观滑点后能否保留 $5~$15 笔均利润 + 2022-2026 全正
  3. ML 暂不重训, 留作宏观 Regime 门控 (后续)

结算纪律 (吸取 bug1 时域错位 / bug2 吊灯回望结算 两灾教训):
  - bar t 覆盖 [t, t+Delta), label=left; 信号用收盘确认, 进场在下一根 bar 的 open
  - 吊灯止损 k-1 收盘后已知: stop_k = extreme(e..k-1) - w*ATR[k-1]; 触发当根即出场, 严禁用 horizon 末的回魂位结算
  - 同根多事件保守排序: 止损优先于止盈; gap 穿越止损按 open 成交(更差)
  - 成本: base=$0.50/RT固定 | pess03=0.3*ATR(bar) | pess05=0.5*ATR(bar), ATR 取决策时点已知值
  - 双引擎互验: 同一份核心函数, 纯Python与numba各跑一遍, 输出必须逐位一致才开网格
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from numba import njit

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE, "data.csv")
RES = os.path.join(BASE, "results_v17")
os.makedirs(RES, exist_ok=True)

TF_SPECS = [
    ("M15", "15min", 192, 480),
    ("M30", "30min", 96, 240),
    ("H1",  "1h",    48, 120),
]
COSTS = [("base", 0, 0.50), ("pess03", 1, 0.30), ("pess05", 1, 0.50)]
EXIT_MODE_CHAND, EXIT_MODE_CHAND_TIME, EXIT_MODE_TIME, EXIT_MODE_TURTLE, EXIT_MODE_BRACKET = 0, 1, 2, 3, 4
ENTRY_CLOSE, ENTRY_STOP = 0, 1


# ================= 数据 =================
def load_m1():
    df = pd.read_csv(DATA_CSV, sep="\t")
    df.columns = ["date", "time", "open", "high", "low", "close", "tickvol", "vol", "spread"]
    df["dt"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S")
    df = df.set_index("dt").drop(columns=["date", "time", "vol"]).sort_index()
    br = df.high - df.low
    rng_z = (br - br.rolling(288).mean()) / br.rolling(288).std()
    cdf = df[~(((df.tickvol <= 5) & (br < 0.01)) | (rng_z.abs() > 15))].copy()
    return cdf


def resample_htf(cdf, rule):
    return (cdf.resample(rule, label="left", closed="left")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last",
                  "tickvol": "sum", "spread": "median"})
            .dropna(subset=["open"]))


def build_arrays(bar):
    o = bar["open"].to_numpy(float)
    h = bar["high"].to_numpy(float)
    l = bar["low"].to_numpy(float)
    c = bar["close"].to_numpy(float)
    sp = bar["spread"].to_numpy(float)  # 点数; RT$ = sp*0.001 (1点=$0.001)
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.concatenate(([c[0]], c[:-1]))),
                                      np.abs(l - np.concatenate(([c[0]], c[:-1])))))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().to_numpy()
    out = dict(o=o, h=h, l=l, c=c, sp=sp, atr=atr, ema20=ema20, n=len(o), ts=bar.index)
    for N in (20, 40, 55):
        out[f"dhi{N}"] = pd.Series(h).rolling(N).max().shift(1).to_numpy()
        out[f"dlo{N}"] = pd.Series(l).rolling(N).min().shift(1).to_numpy()
    out["dxlo10"] = pd.Series(l).rolling(10).min().shift(1).to_numpy()
    out["dxhi10"] = pd.Series(h).rolling(10).max().shift(1).to_numpy()
    return out


def build_entry(arrs, kind):
    """返回 (mode, sig, line_up, line_dn). sig: +1/-1/0 (bar收盘确认型); line: 止损单型."""
    c, h, l, atr, ema = arrs["c"], arrs["h"], arrs["l"], arrs["atr"], arrs["ema20"]
    n = arrs["n"]
    z = np.zeros(n, dtype=np.int8)
    nan = np.full(n, np.nan)
    if kind.startswith("don") and not kind.endswith("s"):
        N = int(kind[3:])
        dhi, dlo = arrs[f"dhi{N}"], arrs[f"dlo{N}"]
        up = (c > dhi) & (atr > 0)
        dn = (c < dlo) & (atr > 0)
        sig = z.copy()
        sig[up] = 1
        sig[dn] = -1
        return ENTRY_CLOSE, sig, nan, nan
    if kind.startswith("don") and kind.endswith("s"):
        N = int(kind[3:-1])
        return ENTRY_STOP, z, arrs[f"dhi{N}"], arrs[f"dlo{N}"]
    if kind.startswith("kelt"):
        mult = float(kind.split("_")[1])
        up = (c > ema + mult * atr) & (atr > 0)
        dn = (c < ema - mult * atr) & (atr > 0)
        sig = z.copy()
        sig[up] = 1
        sig[dn] = -1
        # EMA 预热 60 根
        sig[:60] = 0
        return ENTRY_CLOSE, sig, nan, nan
    raise ValueError(kind)


# ================= 核心 (纯Python源, numba同源编译, 双引擎互验) =================
def simulate_core(o, h, l, c, atr, entry_mode, sig, line_up, line_dn,
                  exit_mode, w, hzn, tp_m, sl_m, dxlo, dxhi,
                  cost_mode, cost_flat, cost_atr_frac):
    n = len(o)
    e_i = np.empty(n, np.int64)
    x_i = np.empty(n, np.int64)
    sd = np.empty(n, np.int8)
    epx = np.empty(n, np.float64)
    xpx = np.empty(n, np.float64)
    cst = np.empty(n, np.float64)
    rsn = np.empty(n, np.int8)
    nt = 0
    pos = 0
    entry_px = 0.0
    hi_s = 0.0
    lo_s = 0.0
    eib = 0
    tp = 0.0
    sl = 0.0
    cost_v = 0.0
    pending = False  # turtle 收盘信号 -> 次根 open 出场
    for k in range(1, n):
        # -- 挂起的 turtle 出场: 本根 open 成交, 本根不再进场(保守, 禁同根翻仓) --
        if pos != 0 and pending:
            e_i[nt] = eib; x_i[nt] = k; sd[nt] = pos; epx[nt] = entry_px
            xpx[nt] = o[k]; cst[nt] = cost_v; rsn[nt] = 6; nt += 1
            pos = 0; pending = False
            continue
        # -- 空仓: 尝试进场 --
        if pos == 0:
            a_prev = atr[k - 1]
            if entry_mode == ENTRY_CLOSE:
                s = sig[k - 1]
                if s != 0 and a_prev > 0 and not np.isnan(o[k]):
                    pos = s; entry_px = o[k]; eib = k
                    hi_s = entry_px; lo_s = entry_px
                    cost_v = cost_flat if cost_mode == 0 else cost_atr_frac * a_prev
                    if pos == 1:
                        tp = entry_px + tp_m * a_prev
                        sl = entry_px - sl_m * a_prev
                    else:
                        tp = entry_px - tp_m * a_prev
                        sl = entry_px + sl_m * a_prev
            else:
                up = line_up[k]; dn = line_dn[k]
                if (not np.isnan(up)) and (not np.isnan(dn)) and a_prev > 0:
                    hit_up = h[k] >= up
                    hit_dn = l[k] <= dn
                    if hit_up and not hit_dn:
                        pos = 1; entry_px = up if o[k] < up else o[k]; eib = k
                        hi_s = entry_px; lo_s = entry_px
                        cost_v = cost_flat if cost_mode == 0 else cost_atr_frac * a_prev
                        tp = entry_px + tp_m * a_prev
                        sl = entry_px - sl_m * a_prev
                    elif hit_dn and not hit_up:
                        pos = -1; entry_px = dn if o[k] > dn else o[k]; eib = k
                        hi_s = entry_px; lo_s = entry_px
                        cost_v = cost_flat if cost_mode == 0 else cost_atr_frac * a_prev
                        tp = entry_px - tp_m * a_prev
                        sl = entry_px + sl_m * a_prev
                    # 双向同根触发: 路径不明, 保守跳过
        # -- 持仓: 当根出场检查 (触发即出场, 严禁回望) --
        if pos != 0:
            a_prev = atr[k - 1]
            done = False
            if (exit_mode == EXIT_MODE_CHAND or exit_mode == EXIT_MODE_CHAND_TIME) and a_prev > 0:
                if pos == 1:
                    stop = hi_s - w * a_prev
                    if l[k] <= stop:
                        px = stop if o[k] >= stop else o[k]  # gap 穿越按更差的 open 成交
                        e_i[nt] = eib; x_i[nt] = k; sd[nt] = pos; epx[nt] = entry_px
                        xpx[nt] = px; cst[nt] = cost_v; rsn[nt] = 0; nt += 1
                        pos = 0; done = True
                else:
                    stop = lo_s + w * a_prev
                    if h[k] >= stop:
                        px = stop if o[k] <= stop else o[k]
                        e_i[nt] = eib; x_i[nt] = k; sd[nt] = pos; epx[nt] = entry_px
                        xpx[nt] = px; cst[nt] = cost_v; rsn[nt] = 0; nt += 1
                        pos = 0; done = True
            if not done and (exit_mode == EXIT_MODE_CHAND_TIME or exit_mode == EXIT_MODE_TIME) and hzn > 0:
                if k >= eib + hzn:
                    e_i[nt] = eib; x_i[nt] = k; sd[nt] = pos; epx[nt] = entry_px
                    xpx[nt] = o[k]; cst[nt] = cost_v; rsn[nt] = 1; nt += 1
                    pos = 0; done = True
            if not done and exit_mode == EXIT_MODE_BRACKET:
                if pos == 1:
                    if o[k] >= tp:
                        px = o[k]; r = 3
                    elif o[k] <= sl:
                        px = o[k]; r = 4
                    elif h[k] >= tp and l[k] <= sl:
                        px = sl; r = 4  # 同根双触发保守取止损
                    elif h[k] >= tp:
                        px = tp; r = 3
                    elif l[k] <= sl:
                        px = sl; r = 4
                    else:
                        px = 0.0; r = -1
                    if r >= 0:
                        e_i[nt] = eib; x_i[nt] = k; sd[nt] = pos; epx[nt] = entry_px
                        xpx[nt] = px; cst[nt] = cost_v; rsn[nt] = r; nt += 1
                        pos = 0; done = True
                else:
                    if o[k] <= tp:
                        px = o[k]; r = 3
                    elif o[k] >= sl:
                        px = o[k]; r = 4
                    elif l[k] <= tp and h[k] >= sl:
                        px = sl; r = 4
                    elif l[k] <= tp:
                        px = tp; r = 3
                    elif h[k] >= sl:
                        px = sl; r = 4
                    else:
                        px = 0.0; r = -1
                    if r >= 0:
                        e_i[nt] = eib; x_i[nt] = k; sd[nt] = pos; epx[nt] = entry_px
                        xpx[nt] = px; cst[nt] = cost_v; rsn[nt] = r; nt += 1
                        pos = 0; done = True
            if not done:
                # 收盘后更新极值锚 (含当根, 供下一根止损用)
                if h[k] > hi_s:
                    hi_s = h[k]
                if l[k] < lo_s:
                    lo_s = l[k]
                # turtle 结构出场: 收盘确认, 次根 open 成交
                if exit_mode == EXIT_MODE_TURTLE and not np.isnan(dxlo[k]):
                    if pos == 1 and c[k] < dxlo[k]:
                        pending = True
                    elif pos == -1 and c[k] > dxhi[k]:
                        pending = True
    if pos != 0:  # 数据末尾强平
        e_i[nt] = eib; x_i[nt] = n - 1; sd[nt] = pos; epx[nt] = entry_px
        xpx[nt] = c[n - 1]; cst[nt] = cost_v; rsn[nt] = 5; nt += 1
    return e_i[:nt], x_i[:nt], sd[:nt], epx[:nt], xpx[:nt], cst[:nt], rsn[:nt]


simulate_jit = njit(cache=True)(simulate_core)


# ================= 指标 =================
def metrics(tr, ts, spread_med):
    if len(tr["e_i"]) == 0:
        return dict(total=0.0, trades=0, avg=0.0)
    e_i, x_i, sd, epx, xpx, cst = tr["e_i"], tr["x_i"], tr["sd"], tr["epx"], tr["xpx"], tr["cst"]
    pnl = sd * (xpx - epx) - cst
    xts = ts[x_i]
    yrs = xts.year.to_numpy()
    by = {}
    for y in np.unique(yrs):
        by[int(y)] = round(float(pnl[yrs == y].sum()), 1)
    mo = pd.Series(pnl, index=xts).groupby(xts.to_period("M")).sum()
    eq = mo.cumsum()
    dd = float((eq - eq.cummax()).min()) if len(eq) else 0.0
    sharpe = float(mo.mean() / (mo.std() + 1e-9) * np.sqrt(12)) if len(mo) > 2 else 0.0
    w = pnl > 0
    plr = float(pnl[w].sum() / max(-pnl[~w].sum(), 1e-9)) if (~w).sum() else 99.0
    m54 = xts >= pd.Timestamp("2022-08-01")
    total54 = float(pnl[m54].sum())
    y54 = {}
    for y in np.unique(yrs[m54]):
        y54[int(y)] = round(float(pnl[m54 & (yrs == y)].sum()), 1)
    hold = x_i - e_i
    reasons = {}
    for r, nm in [(0, "chand_stop"), (1, "time"), (2, "unused"), (3, "tp"), (4, "sl"), (5, "eod"), (6, "turtle")]:
        cnt = int((tr["rsn"] == r).sum())
        if cnt:
            reasons[nm] = cnt
    return dict(
        total=round(float(pnl.sum()), 1), trades=int(len(pnl)),
        avg=round(float(pnl.mean()), 2), med=round(float(np.median(pnl)), 2),
        p10=round(float(np.percentile(pnl, 10)), 2), p90=round(float(np.percentile(pnl, 90)), 2),
        win_rate=round(float(w.mean()), 3), plr=round(min(plr, 99.0), 2),
        by_year=by, all_pos=all(v > 0 for v in by.values()), n_years=len(by),
        share2026=round(100.0 * by.get(2026, 0.0) / max(float(pnl.sum()), 1e-9), 1),
        sharpe=round(sharpe, 2), maxdd=round(dd, 1), n_months=int(len(mo)),
        long_pnl=round(float(pnl[sd == 1].sum()), 1), short_pnl=round(float(pnl[sd == -1].sum()), 1),
        med_hold=round(float(np.median(hold)), 1),
        reasons=reasons, spread_rt_med=round(spread_med, 3),
        total54=round(total54, 1), all_pos54=all(v > 0 for v in y54.values()),
        by_year54=y54,
        months={str(k): round(float(v), 1) for k, v in mo.items()},
    )


def run_one(arrs, kind, ex, cost, ts):
    mode, sig, lup, ldn = build_entry(arrs, kind)
    args = (arrs["o"], arrs["h"], arrs["l"], arrs["c"], arrs["atr"], mode, sig, lup, ldn,
            ex["mode"], ex["w"], ex["hzn"], ex["tp"], ex["sl"], arrs["dxlo10"], arrs["dxhi10"],
            cost[1], cost[2], cost[2])
    tr = simulate_jit(*args)
    return tr, metrics(dict(zip(["e_i", "x_i", "sd", "epx", "xpx", "cst", "rsn"], tr)), ts, float(np.nanmedian(arrs["sp"])) * 0.001)


def main():
    t00 = time.time()
    cdf = load_m1()
    print(f"[data] M1 cleaned bars: {len(cdf)}", flush=True)
    entries = ["don20", "don20s", "don40", "don55", "don55s", "kelt20_2.0", "kelt20_2.5"]
    results = []
    for tf, rule, h2, h5 in TF_SPECS:
        bar = resample_htf(cdf, rule)
        arrs = build_arrays(bar)
        ts = arrs["ts"]
        atrv = arrs["atr"]
        print(f"[{tf}] bars={arrs['n']} ATR14 med=${np.nanmedian(atrv):.2f} "
              f"p10=${np.nanpercentile(atrv, 10):.2f} p90=${np.nanpercentile(atrv, 90):.2f} "
              f"spreadRT=${np.nanmedian(arrs['sp']) * 0.001:.3f}", flush=True)
        exits = [
            dict(name="ch25", mode=EXIT_MODE_CHAND, w=2.5, hzn=0, tp=0.0, sl=0.0),
            dict(name="ch30", mode=EXIT_MODE_CHAND, w=3.0, hzn=0, tp=0.0, sl=0.0),
            dict(name="ch35", mode=EXIT_MODE_CHAND, w=3.5, hzn=0, tp=0.0, sl=0.0),
            dict(name="ch30_h2d", mode=EXIT_MODE_CHAND_TIME, w=3.0, hzn=h2, tp=0.0, sl=0.0),
            dict(name="ch30_h5d", mode=EXIT_MODE_CHAND_TIME, w=3.0, hzn=h5, tp=0.0, sl=0.0),
            dict(name="time2d", mode=EXIT_MODE_TIME, w=0.0, hzn=h2, tp=0.0, sl=0.0),
            dict(name="time5d", mode=EXIT_MODE_TIME, w=0.0, hzn=h5, tp=0.0, sl=0.0),
            dict(name="tur10", mode=EXIT_MODE_TURTLE, w=0.0, hzn=0, tp=0.0, sl=0.0),
            dict(name="br25_10", mode=EXIT_MODE_BRACKET, w=0.0, hzn=0, tp=2.5, sl=1.0),
            dict(name="br40_20", mode=EXIT_MODE_BRACKET, w=0.0, hzn=0, tp=4.0, sl=2.0),
        ]
        # ---- 双引擎互验 (每个TF抽检一配置) ----
        mode, sig, lup, ldn = build_entry(arrs, "don20")
        a_chk = (arrs["o"], arrs["h"], arrs["l"], arrs["c"], arrs["atr"], mode, sig, lup, ldn,
                 exits[1]["mode"], 3.0, 0, 0.0, 0.0, arrs["dxlo10"], arrs["dxhi10"], 1, 0.30, 0.30)
        r_py = simulate_core(*a_chk)
        r_jit = simulate_jit(*a_chk)
        for nm, A, B in zip(["e", "x", "sd", "ep", "xp", "cs", "rs"], r_py, r_jit):
            assert len(A) == len(B) and np.array_equal(A, B), f"DUAL-ENGINE MISMATCH {tf}.{nm}"
        print(f"[{tf}] dual-engine check OK (don20/ch30/pess03, {len(r_py[0])} trades identical)", flush=True)
        for kind in entries:
            for ex in exits:
                for cost in COSTS:
                    tr, m = run_one(arrs, kind, ex, cost, ts)
                    m.update(tf=tf, entry=kind, exit=ex["name"], cost=cost[0])
                    results.append(m)
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "months"} for r in results if r["tf"] == tf])
        print(f"[{tf}] grid done: {len(df)} runs | all_pos(pess03)="
              f"{int(((df['cost'] == 'pess03') & df['all_pos']).sum())} | best total="
              f"{df[df['cost'] == 'pess03']['total'].max()}", flush=True)
    json.dump(results, open(os.path.join(RES, "htf_grid.json"), "w"), indent=1)
    print(f"DONE {(time.time() - t00) / 60:.1f}min -> results_v17/htf_grid.json ({len(results)} runs)", flush=True)


if __name__ == "__main__":
    main()
