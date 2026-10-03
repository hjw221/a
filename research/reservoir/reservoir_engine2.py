#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streaming RLS-Reservoir Engine v2 — R1 自我进化实验 (2026-10-03)
=================================================================
v1 基线 (IC_var 0.343 / IC_skew 0.020 / 0.478ms/bar) 之上的改进臂矩阵:

  A1 --target log      : 方差头改 log1p(vf/base) — RLS 是线性-高斯学习器,
                         log 空间有界对称, 分布形状对 RLS 友好 (v1 的 clip(0,6) 扔信息)
  A2 --leaks 0.1,0.3,0.9 : 多尺度漏积分储层 — 慢块记忆数日、快块记忆分钟,
                         同参数量零成本扩记忆结构
  A3 --n-feat 8        : 输入 4→8 (加 atr_ratio_slow/ret1h/sin_h/cos_h)
                         — Stage B 时代已证 session 结构是顶级因子族
  A4 --lam-ens         : λ∈{0.998,0.999,0.9995} 三重 RLS 集成 (不同有效记忆)
  A5 --huber 3.0       : RLS 新息软限幅 (方差标签重尾, 防 outlier 绞协方差)
  A6 --exec schmitt    : 施密特触发 + 量化仓位 {0,±0.5,±1} + 最小时锁
                         (v1 报告 P0: 连续调仓成本=毛利33倍 → 事件驱动)
  附加仪表 (use-case 证据链):
    * Mahalanobis d2min 曲线全程记录 (use case 3 断路器)
    * 方差头 decile 校准表 (use case 2 呼吸阀)
    * 多视界 IC {4,8,16,32} (use case 1/2/4 的 1h-4h 视界覆盖)

默认参数 = v1 主运行逐字段一致 (回归测试锚点: IC_var 必须 ≈0.343)。
因果纪律与 v1 完全一致: t 收盘 GMM→储层→RLS.update(φ_{t-4}, y_{t-4})→预测→pos_{t+1}。

用法:
  python3 reservoir_engine2.py --csv data.csv --out r2_base.json
  python3 reservoir_engine2.py --csv data.csv --out r2_log.json --target log
  ...
"""
import argparse
import json
import math
import os
import sys
import time
from collections import deque

import numpy as np
import pandas as pd

BARS_PER_YEAR = 96.0 * 252.0

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "m15_cache.pkl")


# ============================================================ 数据层 ============
def load_m15(csv_path, use_cache=True):
    """M1 CSV → (v17同源清洗) → M15 重采样. 与 v1 逐字一致; 加 pickle 缓存."""
    key = os.path.abspath(csv_path)
    if use_cache and os.path.exists(CACHE):
        try:
            cache = pd.read_pickle(CACHE)
            if cache.get("src") == key:
                return cache["bar"]
        except Exception:
            pass
    df = pd.read_csv(csv_path, sep="\t")
    df.columns = ["date", "time", "open", "high", "low", "close", "tickvol", "vol", "spread"]
    df["dt"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S")
    df = df.set_index("dt").drop(columns=["date", "time", "vol"]).sort_index()
    br = df.high - df.low
    rng_z = (br - br.rolling(288).mean()) / br.rolling(288).std()
    cdf = df[~(((df.tickvol <= 5) & (br < 0.01)) | (rng_z.abs() > 15))].copy()
    bar = (cdf.resample("15min", label="left", closed="left")
           .agg({"open": "first", "high": "max", "low": "min", "close": "last",
                 "tickvol": "sum", "spread": "median"})
           .dropna(subset=["open"]))
    if use_cache:
        try:
            pd.to_pickle({"src": key, "bar": bar}, CACHE)
        except Exception:
            pass
    return bar


def build_feats(bar, horizon=4, head=400, n_feat=4):
    """因果特征 + 延迟标签 + 多视界评估标签. v1 的 4 维是其严格子集."""
    o = bar["open"].to_numpy(float); h = bar["high"].to_numpy(float)
    l = bar["low"].to_numpy(float);  c = bar["close"].to_numpy(float)
    v = bar["tickvol"].to_numpy(float); sp = bar["spread"].to_numpy(float)
    ts = bar.index
    n0 = len(c)
    pc = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    s = slice(head, n0)
    o, h, l, c, v, sp, atr, pc = o[s], h[s], l[s], c[s], v[s], sp[s], atr[s], pc[s]
    ts = ts[s]
    n = len(c)
    ret = np.log(c / pc)
    # ---- v1 原始 4 维 (严格保留) ----
    atr_med = pd.Series(atr).rolling(288, min_periods=96).median().to_numpy()
    f_atr_ratio = atr / np.maximum(atr_med, 1e-9)
    f_ret = ret / np.maximum(atr / c, 1e-12)
    f_hl = (h - l) / np.maximum(atr, 1e-9)
    vm = pd.Series(v).rolling(96, min_periods=32).mean().to_numpy()
    vs = pd.Series(v).rolling(96, min_periods=32).std().to_numpy()
    f_vz = (v - vm) / np.maximum(vs, 1e-9)
    feats = [f_atr_ratio, f_ret, f_hl, f_vz]
    if n_feat >= 8:
        # ---- A3 扩展 4 维 (全部因果 rolling) ----
        atr_med_slow = pd.Series(atr).rolling(1440, min_periods=480).median().to_numpy()
        f_atr_slow = atr / np.maximum(atr_med_slow, 1e-9)
        r4 = pd.Series(ret).rolling(4).sum().to_numpy()
        f_ret1h = r4 / np.maximum(atr / c * 2.0, 1e-12)
        hrs = np.asarray(ts.hour + ts.minute / 60.0, float)
        f_sin = np.sin(2 * np.pi * hrs / 24.0)
        f_cos = np.cos(2 * np.pi * hrs / 24.0)
        feats += [f_atr_slow, f_ret1h, f_sin, f_cos]
    U = np.column_stack(feats)
    U = np.clip(np.nan_to_num(U, nan=0.0), -4.0, 4.0)
    # ---- 标签 (延迟 horizon bar 使用; v1 语义) ----
    r2 = ret * ret
    v4p = pd.Series(r2).rolling(horizon).sum().to_numpy()
    base = pd.Series(np.nan_to_num(v4p)).ewm(span=96, adjust=False).mean().to_numpy()
    vf = pd.Series(r2).rolling(horizon).sum().shift(-horizon).to_numpy()
    rf = pd.Series(ret).rolling(horizon).sum().shift(-horizon).to_numpy()
    base_s = np.maximum(base, 1e-14)
    y_var_raw = np.clip(np.nan_to_num(vf / base_s, nan=0.0), 0.0, 6.0)
    y_var_log = np.clip(np.nan_to_num(np.log1p(vf / base_s), nan=0.0), 0.0, 3.0)
    y_skew = np.clip(np.nan_to_num(rf / np.sqrt(np.nan_to_num(vf) + 0.15 * base_s), nan=0.0), -2.5, 2.5)
    # ---- 多视界评估标签 (仅评估, 不进训练; 未来方差比 at h) ----
    vh = {}
    for hh in (4, 8, 16, 32):
        vfh = pd.Series(r2).rolling(hh).sum().shift(-hh).to_numpy()
        vh[hh] = np.clip(np.nan_to_num(vfh / base_s, nan=0.0), 0.0, 6.0)
    return dict(o=o, h=h, l=l, c=c, v=v, sp_pts=sp, atr=atr, U=U, ts=ts, n=n, ret=ret,
                y_var_raw=y_var_raw, y_var_log=y_var_log, y_skew=y_skew, vh=vh)


# ============================================================ L2: GMM ==========
class StreamingGMM:
    """v1 逐字一致 (含剪枝同步修复); 新增 d2min 返回已存在."""

    def __init__(self, dim, k_max=16, spawn_d2=36.0, spawn_cool=96,
                 w_spawn=0.10, lr=0.02, wd=0.01, sigma0=0.6):
        self.dim = dim; self.k_max = k_max
        self.spawn_d2 = spawn_d2; self.spawn_cool = spawn_cool
        self.w_spawn = w_spawn; self.lr = lr; self.wd = wd; self.sigma0 = sigma0
        self.mu = []; self.sd = []; self.w = []
        self.t = 0; self.last_spawn = -10 ** 9

    def _spawn(self, z, w0):
        self.mu.append(z.astype(float).copy())
        self.sd.append(np.full(self.dim, self.sigma0))
        self.w.append(w0)
        s = sum(self.w)
        self.w = [x / s for x in self.w]

    def step(self, z):
        self.t += 1
        z = np.clip(z, -3.5, 3.5)
        if not self.w:
            self._spawn(z, 1.0)
            return np.array([0.0, 0.5, 0.0, 1.0 / 16.0]), 3, 0.0, 0.0
        mu = np.asarray(self.mu); sd = np.asarray(self.sd); w = np.asarray(self.w)
        d2 = (((z - mu) / sd) ** 2).sum(1)
        ll = np.log(np.maximum(w, 1e-12)) - 0.5 * d2 - np.log(sd).sum(1)
        m = ll.max(); r = np.exp(ll - m); r /= r.sum()
        kb = int(d2.argmin())
        d2min_raw = float(d2[kb])  # 剪枝前捕获 (v1 语义)
        if (d2[kb] > self.spawn_d2 and self.t - self.last_spawn > self.spawn_cool
                and len(self.w) < self.k_max + 4):
            self._spawn(z, self.w_spawn)
            self.last_spawn = self.t
        else:
            mu_old = mu.copy()
            mu += self.lr * r[:, None] * (z - mu)
            var = sd * sd
            var += self.lr * r[:, None] * ((z - mu_old) ** 2 - var)
            sd = np.sqrt(np.maximum(var, 0.01))
            w = (1.0 - self.wd) * w + self.wd * r
            w /= w.sum()
            keep = w > 1e-3
            if keep.sum() >= 2 and (~keep).any():
                kept_idx = np.where(keep)[0]
                if keep[kb]:
                    kb = int(np.searchsorted(kept_idx, kb))  # 组件身份保持 (修越界)
                else:
                    kb = int(kept_idx[np.argmin(d2[kept_idx])])
                mu, sd, w = mu[keep], sd[keep], w[keep] / w[keep].sum()
            self.mu = [row.copy() for row in mu]
            self.sd = [row.copy() for row in sd]
            self.w = list(w)
        K = len(self.w)
        d2min = min(d2min_raw, 100.0)
        vol_lev = np.asarray(self.mu)[:, 0]
        pct = float(np.searchsorted(np.sort(vol_lev), vol_lev[kb])) / max(K - 1, 1)
        ent = float(-(r * np.log(np.maximum(r, 1e-12))).sum()) / max(math.log(K), 1e-9)
        tag = self._tag(np.asarray(self.mu)[kb])
        feats = np.array([d2min / 50.0, pct, ent, min(K, 16) / 16.0])
        return feats, tag, pct, d2min

    @staticmethod
    def _tag(mu_k):
        vol, di = mu_k[0], abs(mu_k[1])
        if vol < 0.85 and di < 0.8:
            return 0
        if vol > 1.35 and di > 1.0:
            return 1
        if vol > 1.35:
            return 2
        return 3


# ============================================================ 主引擎 ===========
def run_pass(D, cfg, seed, y_perm=None, label=""):
    t_wall0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    n_res, n_in = cfg["n_res"], D["U"].shape[1]
    # ---------- L1 储层 (锁死; A2 多尺度漏积分) ----------
    leaks = [float(x) for x in str(cfg["leaks"]).split(",") if x.strip()]
    n_blocks = len(leaks)
    blocks = np.array_split(np.arange(n_res), n_blocks)
    mask = rng.random((n_res, n_res)) < cfg["density"]
    Wr = rng.standard_normal((n_res, n_res)) * mask
    ev = np.linalg.eigvals(Wr)
    Wr *= cfg["spectral"] / (np.abs(ev).max() + 1e-12)
    Win = (rng.random((n_res, n_in)) * 2.0 - 1.0) * cfg["in_scale"]
    del mask, ev
    x = np.zeros(n_res)
    x_new = np.zeros(n_res)
    # ---------- L2 GMM ----------
    gmm = StreamingGMM(n_in, k_max=cfg["k_max"], spawn_d2=cfg["spawn_d2"],
                       spawn_cool=cfg["spawn_cool"], w_spawn=cfg["w_spawn"],
                       lr=cfg["lr"], wd=cfg["wd"])
    # ---------- L3 RLS (A4 λ-ensemble; A5 huber 新息限幅) ----------
    lam_list = [float(v) for v in str(cfg["lam_ens"]).split(",") if v.strip()]
    n_ens = len(lam_list)
    Dd = n_res + 4 + 1
    Ps = [np.eye(Dd) * cfg["p0"] for _ in range(n_ens)]
    Ws = [np.zeros((2, Dd)) for _ in range(n_ens)]
    deltas = [cfg["delta"]] * n_ens
    huber = float(cfg["huber"])
    U = D["U"]
    ts, n = D["ts"], D["n"]
    warm, hzn = cfg["warm"], cfg["horizon"]
    Y = np.column_stack([D["y_var_log"] if cfg["target"] == "log" else D["y_var_raw"],
                         D["y_skew"]])
    if y_perm is not None:
        Y = y_perm
    # ---------- 状态容器 ----------
    pos = np.zeros(n); cost = np.zeros(n)
    yhat = np.zeros((n, 2)); tags = np.full(n, -1, np.int8)
    d2_curve = np.zeros(n)
    W_curve, K_curve = [], []
    phis = deque(maxlen=hzn)
    sm = 0.0
    t_arr = np.empty(n)
    step_i = 0
    exec_mode = cfg["exec"]
    # A6 施密特执行状态
    lvl = 0.0; age = 0
    on_th = float(cfg["schmitt_on"]); off_th = on_th * 0.5
    min_hold = int(cfg["schmitt_hold"]); lvl_q = float(cfg["schmitt_q"])

    def _quant(sm):
        a = abs(sm)
        return 0.0 if a < off_th else (lvl_q if a < on_th + (1 - lvl_q) * 0.5 else 1.0)

    for t in range(n - 1):
        u = U[t]
        _t0 = time.perf_counter()
        # ---- L2 (含 d2min 记录) ----
        g, tag, pct, d2min = gmm.step(u)
        d2_curve[t] = d2min
        # ---- L1 (多尺度漏积分: 分块更新) ----
        pre = Wr @ x + Win @ u
        for bi, blk in enumerate(blocks):
            a = leaks[bi]
            x_new[blk] = (1.0 - a) * x[blk] + a * np.tanh(pre[blk])
        x, x_new = x_new, x
        phi = np.empty(Dd)
        phi[:n_res] = x; phi[n_res:n_res + 4] = g; phi[-1] = 1.0
        # ---- L3 延迟标签闭环 ----
        if t >= hzn and len(phis) == hzn:
            phi_old = phis[0]
            y_old = Y[t - hzn]
            for ei in range(n_ens):
                lam = lam_list[ei]
                P, W = Ps[ei], Ws[ei]
                Px = P @ phi_old
                gden = lam + phi_old @ Px
                k = Px / gden
                e = y_old - W @ phi_old
                if huber > 0:
                    # 新息软限幅: 大 outlier 衰减 (伪 Huber, 保号)
                    scale = np.maximum(np.abs(e) / huber, 1.0)
                    e = e / scale
                W += np.outer(e, k)
                P -= np.outer(k, Px)
                P *= 1.0 / lam
                P[np.diag_indices(Dd)] += deltas[ei]
                Ps[ei], Ws[ei] = P, W
            step_i += 1
            if step_i % 64 == 0:
                for ei in range(n_ens):
                    P = (Ps[ei] + Ps[ei].T) * 0.5
                    tr = float(np.trace(P)) / Dd
                    if tr > cfg["p_cap"]:
                        P *= cfg["p_cap"] / tr
                    Ps[ei] = P
        phis.append(phi)
        # ---- 预测 (集成均值) & 暴露 ----
        yh = np.mean([W @ phi for W in Ws], axis=0)
        yhat[t] = yh
        t_arr[t] = time.perf_counter() - _t0
        if t >= warm:
            tags[t] = tag
            skew_h, var_h = float(yh[1]), float(yh[0])
            tgt = np.clip(cfg["k_sig"] * skew_h, -1.0, 1.0) / (1.0 + cfg["k_vol"] * max(var_h, 0.0))
            if not np.isfinite(tgt):
                tgt = 0.0
            if pct > 0.9:
                tgt *= cfg["chaos_gate"]
            if exec_mode == "schmitt":
                # ---- A6: 施密特触发 + 量化仓位 + 最小时锁 (sm 先更新, 与 v1 顺序一致) ----
                sm = cfg["smooth"] * sm + (1.0 - cfg["smooth"]) * tgt
                if lvl == 0.0:
                    if abs(sm) >= on_th:
                        lvl = np.sign(sm) * _quant(sm); age = 0
                else:
                    age += 1
                    flip = (lvl > 0 and sm <= -on_th) or (lvl < 0 and sm >= on_th)
                    if flip:
                        lvl = np.sign(sm) * _quant(sm); age = 0
                    elif age >= min_hold:
                        tgt_q = np.sign(sm) * _quant(sm) if abs(sm) >= off_th else 0.0
                        if tgt_q != lvl:
                            lvl = tgt_q; age = 0
                d = lvl - pos[t]
            else:
                sm = cfg["smooth"] * sm + (1.0 - cfg["smooth"]) * tgt
                d = sm - pos[t]
                if abs(d) < cfg["dead"]:
                    d = 0.0
                d = float(np.clip(d, -cfg["rate"], cfg["rate"]))
            cost[t] = abs(d) * (cfg["slip_atr"] * D["atr"][t] + D["sp_pts"][t] * 0.001)
            pos[t + 1] = pos[t] + d
        else:
            pos[t + 1] = 0.0
        if t >= warm and t % 500 == 0:
            W_curve.append(round(float(np.linalg.norm(Ws[0])), 5))
            K_curve.append(len(gmm.w))
        if t % 40000 == 0 and t:
            print(f"  [{label}] t={t}/{n} K={len(gmm.w)} ||W||={np.linalg.norm(Ws[0]):.2f}", flush=True)
    # ---- PnL ----
    oz = cfg["oz"]
    pnl = np.zeros(n)
    dprice = np.diff(D["c"])
    pnl[warm + 1:] = oz * pos[warm:n - 1] * dprice[warm:] - cost[warm:n - 1]
    wall = time.perf_counter() - t_wall0
    sl = slice(warm, n - 1)
    timing = dict(wall_s=round(wall, 1), bars=int(n - warm - 1),
                  total_ms=dict(mean=round(float(t_arr[sl].mean()) * 1e3, 4),
                                p99=round(float(np.percentile(t_arr[sl], 99)) * 1e3, 4)))
    engine_mb = (Wr.nbytes + Win.nbytes
                 + sum(P.nbytes + W.nbytes for P, W in zip(Ps, Ws))) / 1e6
    return dict(pos=pos, pnl=pnl, cost=cost, yhat=yhat, tags=tags, d2=d2_curve,
                W_curve=W_curve, K_curve=K_curve, timing=timing,
                engine_mb=round(engine_mb, 2), K_final=len(gmm.w),
                W_final=Ws[0].copy())


# ============================================================ 指标 =============
def metrics(D, res, cfg):
    ts, n, warm = D["ts"], D["n"], cfg["warm"]
    pnl, pos, cost, yhat = res["pnl"], res["pos"], res["cost"], res["yhat"]
    yrs = np.asarray(ts.year)
    valid = np.zeros(n, bool); valid[warm:n - 40] = True
    # ---- IC: 训练口径 (与目标同空间) ----
    tgt_log = cfg["target"] == "log"
    yv_train = D["y_var_log"] if tgt_log else D["y_var_raw"]
    m_icv = valid & np.isfinite(yv_train)
    ic_var_train = float(np.corrcoef(yhat[m_icv, 0], yv_train[m_icv])[0, 1]) if m_icv.sum() > 100 else 0.0
    # ---- IC: 规范口径 (一律 vs raw 相对方差, 与 v1 0.343 可比) ----
    pred_canon = np.expm1(np.clip(yhat[:, 0], 0, 3)) if tgt_log else yhat[:, 0]
    yv_raw = D["y_var_raw"]
    m_can = valid & np.isfinite(yv_raw)
    ic_var = float(np.corrcoef(pred_canon[m_can], yv_raw[m_can])[0, 1]) if m_can.sum() > 100 else 0.0
    m_ic = valid & np.isfinite(D["y_skew"])
    ic_skew = float(np.corrcoef(yhat[m_ic, 1], D["y_skew"][m_ic])[0, 1]) if m_ic.sum() > 100 else 0.0
    # ---- 多视界 IC (能量头; use-case 视界覆盖证据) ----
    ic_multi = {}
    for hh, yvh in D["vh"].items():
        m = valid & np.isfinite(yvh)
        ic_multi[f"h{hh}"] = round(float(np.corrcoef(pred_canon[m], yvh[m])[0, 1]), 4) if m.sum() > 100 else 0.0
    # ---- decile 校准表 (use case 2 呼吸阀: 预测分位 → 实现方差比) ----
    dec = {}
    if m_can.sum() > 1000:
        p = pred_canon[m_can]; y = yv_raw[m_can]
        qs = np.quantile(p, np.linspace(0, 1, 11))
        for i in range(10):
            msk = (p >= qs[i]) & (p <= qs[i + 1] if i == 9 else p < qs[i + 1])
            if msk.sum() > 50:
                dec[f"d{i+1}"] = round(float(np.nanmean(y[msk])), 3)
    # ---- 惊异度事件研究 (use case 3 断路器: d2 spike → 未来1h方差) ----
    d2 = res["d2"]
    sent = {}
    if valid.sum() > 1000:
        yh4 = D["vh"][4]
        m = valid & np.isfinite(yh4)
        base_med = float(np.nanmedian(yh4[m]))
        for thr in (25.0, 40.0, 60.0):
            spike = d2 > thr
            mm = m & spike
            sent[f"d2>{thr:.0f}"] = dict(
                n=int(mm.sum()),
                fwd_var=round(float(np.nanmean(yh4[mm])) / max(base_med, 1e-9), 3) if mm.sum() > 30 else None)
    rows = []
    for y in sorted(set(yrs[warm:].tolist())):
        m = (yrs == y) & (np.arange(n) >= warm)
        p = pnl[m]
        eq = np.cumsum(p)
        mm = m & m_can
        ic_y = float(np.corrcoef(pred_canon[mm], yv_raw[mm])[0, 1]) if mm.sum() > 100 else 0.0
        rows.append(dict(year=int(y), pnl=round(float(p.sum()), 1),
                         gross=round(float((pnl[m] + cost[m]).sum()), 1),
                         cost=round(float(cost[m].sum()), 1),
                         sharpe=round(float(p.mean() / (p.std() + 1e-12) * math.sqrt(BARS_PER_YEAR)), 2),
                         avg_pos=round(float(np.abs(pos[m]).mean()), 3),
                         turnover=round(float(np.abs(np.diff(pos))[m[:-1]].sum()), 1),
                         ic_var=round(ic_y, 4), bars=int(m.sum())))
    eq = np.cumsum(pnl[warm:])
    dp = np.diff(pos)
    overall = dict(
        pnl=round(float(pnl[warm:].sum()), 1),
        gross=round(float((pnl[warm:] + cost[warm:]).sum()), 1),
        cost=round(float(cost[warm:].sum()), 1),
        sharpe=round(float(pnl[warm:].mean() / (pnl[warm:].std() + 1e-12) * math.sqrt(BARS_PER_YEAR)), 2),
        avg_pos=round(float(np.abs(pos[warm:]).mean()), 3),
        turnover=round(float(np.abs(dp)[warm:].sum()), 1),
        ic_skew=round(ic_skew, 4), ic_var=round(ic_var, 4),
        ic_var_train=round(ic_var_train, 4),
    )
    return dict(rows=rows, overall=overall, ic_multi=ic_multi,
                decile=dec, sentinel=sent)


# ============================================================ main =============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/home/z/my-project/research-lab/data.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="arm")
    ap.add_argument("--seed", type=int, default=42)
    # ---- v1 兼容默认 ----
    ap.add_argument("--n-res", type=int, default=500)
    ap.add_argument("--density", type=float, default=0.10)
    ap.add_argument("--spectral", type=float, default=0.95)
    ap.add_argument("--in-scale", type=float, default=0.30)
    ap.add_argument("--lam", type=float, default=0.999)  # 单 λ 模式 (lam_ens 未给时)
    ap.add_argument("--delta", type=float, default=1e-6)
    ap.add_argument("--p0", type=float, default=4.0)
    ap.add_argument("--p-cap", type=float, default=1e4)
    ap.add_argument("--k-sig", type=float, default=2.0)
    ap.add_argument("--k-vol", type=float, default=0.5)
    ap.add_argument("--chaos-gate", type=float, default=0.6)
    ap.add_argument("--smooth", type=float, default=0.5)
    ap.add_argument("--rate", type=float, default=0.20)
    ap.add_argument("--dead", type=float, default=0.05)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--warm", type=int, default=2000)
    ap.add_argument("--k-max", type=int, default=16)
    ap.add_argument("--spawn-d2", type=float, default=36.0)
    ap.add_argument("--spawn-cool", type=int, default=96)
    ap.add_argument("--w-spawn", type=float, default=0.10)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--slip-atr", type=float, default=0.30)
    ap.add_argument("--oz", type=float, default=1.0)
    # ---- v2 改进臂 ----
    ap.add_argument("--target", choices=["raw", "log"], default="raw")
    ap.add_argument("--leaks", default="1.0")          # 1.0 = v1 等价 (无泄漏混合)
    ap.add_argument("--n-feat", type=int, default=4, choices=[4, 8])
    ap.add_argument("--lam-ens", default="")           # 空 = 单 λ (cfg.lam)
    ap.add_argument("--huber", type=float, default=0.0)
    ap.add_argument("--exec", choices=["continuous", "schmitt"], default="continuous")
    ap.add_argument("--schmitt-on", type=float, default=0.25)
    ap.add_argument("--schmitt-hold", type=int, default=8)
    ap.add_argument("--schmitt-q", type=float, default=0.5)
    ap.add_argument("--save-npz", default="")
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()
    cfg = vars(args)
    if not cfg["lam_ens"]:
        cfg["lam_ens"] = str(cfg["lam"])

    print(f"== R2 arm [{args.label}] target={args.target} leaks={args.leaks} "
          f"nfeat={args.n_feat} lamens={cfg['lam_ens']} huber={args.huber} exec={args.exec} ==", flush=True)
    t0 = time.perf_counter()
    bar = load_m15(args.csv)
    D = build_feats(bar, horizon=args.horizon, n_feat=args.n_feat)
    print(f"data: {D['n']} M15 bars {D['ts'][0]} .. {D['ts'][-1]} ({time.perf_counter()-t0:.1f}s)", flush=True)

    y_perm = None
    if args.shuffle:
        rng = np.random.default_rng(999)
        perm = rng.permutation(len(D["y_skew"]))
        Y = np.column_stack([D["y_var_log"] if args.target == "log" else D["y_var_raw"],
                             D["y_skew"]])
        y_perm = Y[perm]

    res = run_pass(D, cfg, args.seed, y_perm=y_perm, label=args.label)
    m = metrics(D, res, cfg)
    doc = dict(
        meta=dict(arm=args.label, seed=args.seed, bars=int(D["n"]),
                  span=[D["ts"][0].isoformat(), D["ts"][-1].isoformat()],
                  cfg={k: cfg[k] for k in ("target", "leaks", "n_feat", "lam_ens", "huber",
                                           "exec", "schmitt_on", "schmitt_hold", "schmitt_q",
                                           "n_res", "spectral", "in_scale", "horizon", "warm")},
                  timing=res["timing"], engine_mb=res["engine_mb"], K_final=res["K_final"]),
        metrics=m,
    )
    with open(args.out, "w") as f:
        json.dump(doc, f, separators=(",", ":"), default=str)
    if args.save_npz:
        np.savez_compressed(args.save_npz,
                            ts=D["ts"].asi8, yhat=res["yhat"], d2=res["d2"],
                            tags=res["tags"], pos=res["pos"], cost=res["cost"],
                            pnl=res["pnl"], atr=D["atr"], vh4=D["vh"][4],
                            y_var_raw=D["y_var_raw"], y_skew=D["y_skew"])
    ov = m["overall"]
    print(f"   PnL={ov['pnl']}$ gross={ov['gross']}$ cost={ov['cost']}$ "
          f"IC_var={ov['ic_var']} (train {ov['ic_var_train']}) IC_skew={ov['ic_skew']} "
          f"turn={ov['turnover']} ms/bar={res['timing']['total_ms']['mean']}", flush=True)
    print(f"   IC_multi={m['ic_multi']} decile={m['decile']}", flush=True)
    print(f"   sentinel={m['sentinel']}", flush=True)
    print(f"== DONE -> {args.out} ==", flush=True)


if __name__ == "__main__":
    main()
