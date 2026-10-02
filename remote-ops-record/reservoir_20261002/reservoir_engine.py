#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streaming RLS-Reservoir Engine v1 — XAUUSD M15 (2022–2026)
============================================================
三层无梯度自学习闭环 (无反向传播, 纯 CPU NumPy, 单线程实测):

  L1 动态记忆层   : 固定稀疏储层网络 ESN (n_res 神经元, 谱半径 ρ<1, tanh)
                   — W_res/W_in 初始化后永久锁死, 纯矩阵乘法提取非线性时间拓扑
  L2 状态自组织层 : Streaming GMM (对角协方差, 惊异度 Mahalanobis² 驱动组件自生长,
                   无需预设状态数; 权重指数衰减 = 状态层自带遗忘)
  L3 突触自演化层 : 指数遗忘因子 RLS, 每走完一根 M15 Bar 对 (φ_{t-4}, y_{t-4})
                   闭环递推更新读出权重 W_out ∈ R^{2×D}, 永不重训

主动推理目标 (无固定止盈止损标签):
  y_var_rel(t) : 未来4根M15相对实现方差 (能量, 相对过去96bar EMA基线)
  y_skew(t)    : 未来4根方向动量偏斜 Σr/√(Σr²+floor), 有界 ±2.5

仓位 = 非对称信息熵释放:
  pos_target = clip(K·ŝkew, ±1) × 1/(1+κ·v̂ar_rel) × chaos_gate
  — 压缩后爆发方向性动能 → ŝkew 大 → 暴露释放; 噪音态 → ŝkew≈0 → 流形塌缩离场
  — 预测能量高 → 波动率倒数缩放 (v18 启示: 尾部管理=波动率倒数仓位, 在线学习版)
  — GMM 混沌态 (状态波动分位>0.9) → 减仓门

执行口径 (与 v17/v18 历史基准同屋):
  1.0 暴露 = 1 oz; PnL $ 线性可缩放; 收盘调仓 bar t 收盘定价, t+1 持仓
  成本 = |Δpos| × (0.3×ATR14$ + spread$) 每交易单位 (悲观滑点+点差)
  速率限制 0.20/bar + 死区 0.05 + EMA 平滑 (防连续调仓绞肉)

因果纪律 (无回望):
  t 收盘: GMM(z_t) → 储层 x_t → RLS.update(φ_{t-4}, y_{t-4}) → 预测 ŷ_t → 设 pos_{t+1}
  所有 rolling 统计含当前 bar (bar 已收盘), 标签延迟 4 bar 才进学习器

工程稳定性 (诚实披露):
  D=505 维 RLS 在 λ=0.995 (有效记忆 200<维度) 会协方差缠绕 →
  三重防护: δ=1e-6 对角正则 + trace 上限 P_cap + 周期对称化; λ 扫描对照见 JSON

用法:
  OMP_NUM_THREADS=1 python3 reservoir_engine.py --csv <M1.csv> --out reservoir_v1.json
"""
import argparse
import json
import math
import os
import sys
import time
import resource
from collections import deque

import numpy as np
import pandas as pd

BARS_PER_YEAR = 96.0 * 252.0  # M15 年化因子


# ============================================================ 数据层 ============
def load_m15(csv_path):
    """M1 CSV → (v17同源清洗) → M15 重采样. label/closed=left, v17 口径."""
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
    return bar


def build_feats(bar, horizon=4, head=400):
    """全部因果特征 + 延迟标签. head 截掉 rolling 预热 NaN 段."""
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
    # ---- 4 维微观几何 (用户规格: ATR_norm, Return_15m, HighLow_Ratio, Volume_Z) ----
    atr_med = pd.Series(atr).rolling(288, min_periods=96).median().to_numpy()
    f_atr_ratio = atr / np.maximum(atr_med, 1e-9)          # ATR 相对其 3 日中位
    f_ret = ret / np.maximum(atr / c, 1e-12)               # ATR 单位化收益
    f_hl = (h - l) / np.maximum(atr, 1e-9)                 # Bar 全距 / ATR
    vm = pd.Series(v).rolling(96, min_periods=32).mean().to_numpy()
    vs = pd.Series(v).rolling(96, min_periods=32).std().to_numpy()
    f_vz = (v - vm) / np.maximum(vs, 1e-9)
    U = np.column_stack([f_atr_ratio, f_ret, f_hl, f_vz])
    U = np.clip(np.nan_to_num(U, nan=0.0), -4.0, 4.0)
    # ---- 主动推理标签 (延迟 4 bar 使用; baseline 全因果) ----
    r2 = ret * ret
    v4p = pd.Series(r2).rolling(horizon).sum().to_numpy()          # 过去4bar方差(至t)
    base = pd.Series(np.nan_to_num(v4p)).ewm(span=96, adjust=False).mean().to_numpy()
    vf = pd.Series(r2).rolling(horizon).sum().shift(-horizon).to_numpy()  # 未来4bar方差
    rf = pd.Series(ret).rolling(horizon).sum().shift(-horizon).to_numpy()  # 未来4bar动量
    base_s = np.maximum(base, 1e-14)
    y_var = np.clip(np.nan_to_num(vf / base_s, nan=0.0), 0.0, 6.0)
    y_skew = np.clip(np.nan_to_num(rf / np.sqrt(np.nan_to_num(vf) + 0.15 * base_s), nan=0.0), -2.5, 2.5)
    Y = np.column_stack([y_var, y_skew])
    return dict(o=o, h=h, l=l, c=c, v=v, sp_pts=sp, atr=atr, U=U, Y=Y, ts=ts, n=n, ret=ret)


# ============================================================ L2: GMM ==========
class StreamingGMM:
    """在线高斯混合: 对角协方差; 惊异度(Mahalanobis²)超阈+冷却期 → 组件自生长;
    权重指数衰减; 弱组件剪枝. 状态标签由组件均值(波动水平,方向强度)在线导出."""
    TAGS = ("COMPRESS", "TREND", "CHAOS", "NORMAL")

    def __init__(self, dim, k_max=16, spawn_d2=36.0, spawn_cool=96,
                 w_spawn=0.10, lr=0.02, wd=0.01, sigma0=0.6):
        self.dim = dim; self.k_max = k_max
        self.spawn_d2 = spawn_d2; self.spawn_cool = spawn_cool
        self.w_spawn = w_spawn; self.lr = lr; self.wd = wd; self.sigma0 = sigma0
        self.mu = []; self.sd = []; self.w = []
        self.t = 0; self.last_spawn = -10**9

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
            return np.array([0.0, 0.5, 0.0, 1.0 / 16.0]), 3, 0.0
        mu = np.asarray(self.mu); sd = np.asarray(self.sd); w = np.asarray(self.w)
        d2 = (((z - mu) / sd) ** 2).sum(1)
        ll = np.log(np.maximum(w, 1e-12)) - 0.5 * d2 - np.log(sd).sum(1)
        m = ll.max(); r = np.exp(ll - m); r /= r.sum()
        kb = int(d2.argmin())
        # ---- 自生长 (惊异度) ----
        if (d2[kb] > self.spawn_d2 and self.t - self.last_spawn > self.spawn_cool
                and len(self.w) < self.k_max + 4):
            self._spawn(z, self.w_spawn)
            self.last_spawn = self.t
        else:
            # ---- 软更新 (responsibility 加权) ----
            mu_old = mu.copy()
            mu += self.lr * r[:, None] * (z - mu)
            var = sd * sd
            var += self.lr * r[:, None] * ((z - mu_old) ** 2 - var)
            sd = np.sqrt(np.maximum(var, 0.01))
            w = (1.0 - self.wd) * w + self.wd * r
            w /= w.sum()
            # ---- 剪枝 (mu/sd/w 同步过滤, kb 若被剪则重指最近组件) ----
            keep = w > 1e-3
            if keep.sum() >= 2 and (~keep).any():
                mu, sd, w = mu[keep], sd[keep], w[keep] / w[keep].sum()
                if not keep[kb]:
                    kb = int(np.where(keep)[0][np.argmin(d2[keep])])
            self.mu = [row.copy() for row in mu]
            self.sd = [row.copy() for row in sd]
            self.w = list(w)
        K = len(self.w)
        d2min = min(float(d2[kb]), 100.0)
        # 状态分位: 按组件波动水平排序
        vol_lev = np.asarray(self.mu)[:, 0]
        pct = float(np.searchsorted(np.sort(vol_lev), vol_lev[kb])) / max(K - 1, 1)
        ent = float(-(r * np.log(np.maximum(r, 1e-12))).sum()) / max(math.log(K), 1e-9)
        tag = self._tag(np.asarray(self.mu)[kb])
        feats = np.array([d2min / 50.0, pct, ent, min(K, 16) / 16.0])
        return feats, tag, pct

    @staticmethod
    def _tag(mu_k):
        vol, di = mu_k[0], abs(mu_k[1])
        if vol < 0.85 and di < 0.8:
            return 0  # COMPRESS
        if vol > 1.35 and di > 1.0:
            return 1  # TREND 单边
        if vol > 1.35:
            return 2  # CHAOS 高波混沌
        return 3      # NORMAL


# ============================================================ 主引擎 ===========
def run_pass(D, cfg, seed, y_perm=None, collect=False, label=""):
    t_wall0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    n_res, n_in = cfg["n_res"], 4
    # ---------- L1 储层 (锁死) ----------
    mask = rng.random((n_res, n_res)) < cfg["density"]
    Wr = rng.standard_normal((n_res, n_res)) * mask
    ev = np.linalg.eigvals(Wr)
    Wr *= cfg["spectral"] / (np.abs(ev).max() + 1e-12)
    Win = (rng.random((n_res, n_in)) * 2.0 - 1.0) * cfg["in_scale"]
    del mask, ev
    x = np.zeros(n_res)
    # ---------- L2 GMM ----------
    gmm = StreamingGMM(4, k_max=cfg["k_max"], spawn_d2=cfg["spawn_d2"],
                       spawn_cool=cfg["spawn_cool"], w_spawn=cfg["w_spawn"],
                       lr=cfg["lr"], wd=cfg["wd"])
    # ---------- L3 RLS ----------
    Dd = n_res + 4 + 1
    P = np.eye(Dd) * cfg["p0"]
    W = np.zeros((2, Dd))
    lam, inv_lam, delta = cfg["lam"], 1.0 / cfg["lam"], cfg["delta"]
    U, Y, c, atr, sp_pts = D["U"], D["Y"], D["c"], D["atr"], D["sp_pts"]
    ts, n = D["ts"], D["n"]
    warm, hzn = cfg["warm"], cfg["horizon"]
    if y_perm is not None:
        Y = y_perm
    # ---------- 状态容器 ----------
    pos = np.zeros(n); cost = np.zeros(n)
    yhat = np.zeros((n, 2)); tags = np.full(n, -1, np.int8)
    xs_norm = 0.0
    phis = deque(maxlen=hzn)
    sm = 0.0
    t_esn_arr = np.empty(n); t_gmm_arr = np.empty(n); t_rls_arr = np.empty(n)
    W_curve, K_curve = [], []
    step_i = 0
    for t in range(n - 1):
        u = U[t]
        # ---- L2 ----
        _t0 = time.perf_counter()
        g, tag, pct = gmm.step(u)
        _t1 = time.perf_counter(); t_gmm_arr[t] = _t1 - _t0
        # ---- L1 ----
        x = np.tanh(Wr @ x + Win @ u)
        _t2 = time.perf_counter(); t_esn_arr[t] = _t2 - _t1
        phi = np.empty(Dd)
        phi[:n_res] = x; phi[n_res:n_res + 4] = g; phi[-1] = 1.0
        # ---- L3 延迟标签闭环 ----
        if t >= hzn and len(phis) == hzn:
            _t0 = time.perf_counter()
            phi_old = phis[0]
            y_old = Y[t - hzn]
            Px = P @ phi_old
            gden = lam + phi_old @ Px
            k = Px / gden
            e = y_old - W @ phi_old
            W += np.outer(e, k)
            P -= np.outer(k, Px)
            P *= inv_lam
            P[np.diag_indices(Dd)] += delta
            t_rls_arr[t] = time.perf_counter() - _t0
            step_i += 1
            if step_i % 64 == 0:
                P = (P + P.T) * 0.5
                tr = float(np.trace(P)) / Dd
                if tr > cfg["p_cap"]:
                    P *= cfg["p_cap"] / tr
        phis.append(phi)
        # ---- 预测 & 暴露 ----
        yh = W @ phi
        yhat[t] = yh
        if t >= warm:
            tags[t] = tag
            skew_h, var_h = float(yh[1]), float(yh[0])
            tgt = np.clip(cfg["k_sig"] * skew_h, -1.0, 1.0) / (1.0 + cfg["k_vol"] * max(var_h, 0.0))
            if not np.isfinite(tgt):
                tgt = 0.0
            if pct > 0.9:
                tgt *= cfg["chaos_gate"]
            sm = cfg["smooth"] * sm + (1.0 - cfg["smooth"]) * tgt
            d = sm - pos[t]
            if abs(d) < cfg["dead"]:
                d = 0.0
            d = float(np.clip(d, -cfg["rate"], cfg["rate"]))
            cost[t] = abs(d) * (cfg["slip_atr"] * atr[t] + sp_pts[t] * 0.001)
            pos[t + 1] = pos[t] + d
        else:
            pos[t + 1] = 0.0
        if collect and t >= warm and t % 500 == 0:
            W_curve.append(round(float(np.linalg.norm(W)), 5))
            K_curve.append(len(gmm.w))
        if t % 20000 == 0 and t:
            xs_norm = float(np.abs(x).mean())
            print(f"  [{label}] t={t}/{n} K={len(gmm.w)} |x|={xs_norm:.3f} "
                  f"||W||={np.linalg.norm(W):.2f}", flush=True)
    # ---- PnL (向量化): pos[t] 持有于 bar t+1 ----
    oz = cfg["oz"]
    pnl = np.zeros(n)
    dprice = np.diff(c)
    pnl[warm + 1:] = oz * pos[warm:n - 1] * dprice[warm:] - cost[warm:n - 1]
    wall = time.perf_counter() - t_wall0
    nb = max(n - warm - 1, 1)
    sl = slice(warm, n - 1)
    tot_arr = t_esn_arr[sl] + t_gmm_arr[sl] + t_rls_arr[sl]

    def _ms(a):
        return dict(mean=round(float(a.mean()) * 1e3, 4),
                    p90=round(float(np.percentile(a, 90)) * 1e3, 4),
                    p99=round(float(np.percentile(a, 99)) * 1e3, 4),
                    max=round(float(a.max()) * 1e3, 4))

    timing = dict(wall_s=round(wall, 1), bars=int(nb),
                  esn_ms=_ms(t_esn_arr[sl]), gmm_ms=_ms(t_gmm_arr[sl]),
                  rls_ms=_ms(t_rls_arr[sl]), total_ms=_ms(tot_arr))
    engine_mb = (Wr.nbytes + Win.nbytes + P.nbytes + W.nbytes) / 1e6
    out = dict(pos=pos, pnl=pnl, cost=cost, yhat=yhat, tags=tags,
               W_curve=W_curve, K_curve=K_curve, timing=timing,
               engine_mb=round(engine_mb, 2), W_final=W.copy(), K_final=len(gmm.w))
    return out


# ============================================================ 指标 =============
def metrics(D, res, cfg):
    ts, n, warm = D["ts"], D["n"], cfg["warm"]
    pnl, pos, cost, yhat = res["pnl"], res["pos"], res["cost"], res["yhat"]
    Y = D["Y"]
    yrs = np.asarray(ts.year)
    valid = np.zeros(n, bool); valid[warm:n - 5] = True
    yv = np.isfinite(Y[:, 1]); m_ic = valid & yv
    ic_skew = float(np.corrcoef(yhat[m_ic, 1], Y[m_ic, 1])[0, 1]) if m_ic.sum() > 100 else 0.0
    m_icv = valid & np.isfinite(Y[:, 0])
    ic_var = float(np.corrcoef(yhat[m_icv, 0], Y[m_icv, 0])[0, 1]) if m_icv.sum() > 100 else 0.0
    rows = []
    for y in sorted(set(yrs[warm:].tolist())):
        m = (yrs == y) & (np.arange(n) >= warm)
        p = pnl[m]
        eq = np.cumsum(p)
        dd = float((eq - np.maximum.accumulate(eq)).min()) if len(eq) else 0.0
        mm = m & m_ic
        ic_y = float(np.corrcoef(yhat[mm, 1], Y[mm, 1])[0, 1]) if mm.sum() > 100 else 0.0
        rows.append(dict(
            year=int(y), pnl=round(float(p.sum()), 1),
            gross=round(float((pnl[m] + cost[m]).sum()), 1),
            cost=round(float(cost[m].sum()), 1),
            sharpe=round(float(p.mean() / (p.std() + 1e-12) * math.sqrt(BARS_PER_YEAR)), 2),
            maxdd=round(dd, 1),
            avg_pos=round(float(np.abs(pos[m]).mean()), 3),
            turnover=round(float(np.abs(np.diff(pos))[m[:-1]].sum()), 1),
            ic_skew=round(ic_y, 4),
            bars=int(m.sum()),
        ))
    eq = np.cumsum(pnl[warm:])
    mdd = float((eq - np.maximum.accumulate(eq)).min())
    dp = np.diff(pos)
    overall = dict(
        pnl=round(float(pnl[warm:].sum()), 1),
        gross=round(float((pnl[warm:] + cost[warm:]).sum()), 1),
        cost=round(float(cost[warm:].sum()), 1),
        sharpe=round(float(pnl[warm:].mean() / (pnl[warm:].std() + 1e-12) * math.sqrt(BARS_PER_YEAR)), 2),
        maxdd=round(mdd, 1),
        avg_pos=round(float(np.abs(pos[warm:]).mean()), 3),
        turnover=round(float(np.abs(dp)[warm:].sum()), 1),
        ic_skew=round(ic_skew, 4), ic_var=round(ic_var, 4),
        plr=round(float(pnl[pnl > 0].sum() / max(-pnl[pnl < 0].sum(), 1e-9)), 2),
    )
    # 权益曲线 (降采样)
    idx = np.linspace(warm, n - 1, min(1200, n - warm)).astype(int)
    eq_pts = [[ts[i].isoformat(), round(float(eq[i - warm]), 1)] for i in idx]
    # 状态占比 (逐年)
    tg = res["tags"]
    state_frac = {}
    for y in sorted(set(yrs[warm:].tolist())):
        m = (yrs == y) & (tg >= 0)
        if m.sum():
            state_frac[int(y)] = [round(float((tg[m] == k).mean()), 3) for k in range(4)]
    return dict(rows=rows, overall=overall, equity=eq_pts, state_frac=state_frac)


# ============================================================ 基线 =============
def momentum_baseline(D, cfg):
    c, atr, sp_pts, ts, n = D["c"], D["atr"], D["sp_pts"], D["ts"], D["n"]
    warm = cfg["warm"]
    pos = np.zeros(n); cost = np.zeros(n); sm = 0.0
    for t in range(warm, n - 1):
        mom = (c[t] - c[t - 4]) / max(2.0 * atr[t], 1e-9)
        tgt = float(np.clip(mom, -1, 1)) * 0.6
        sm = cfg["smooth"] * sm + (1.0 - cfg["smooth"]) * tgt
        d = sm - pos[t]
        if abs(d) < cfg["dead"]:
            d = 0.0
        d = float(np.clip(d, -cfg["rate"], cfg["rate"]))
        cost[t] = abs(d) * (cfg["slip_atr"] * atr[t] + sp_pts[t] * 0.001)
        pos[t + 1] = pos[t] + d
    pnl = np.zeros(n)
    dp_ = np.diff(c)
    pnl[warm + 1:] = cfg["oz"] * pos[warm:n - 1] * dp_[warm:] - cost[warm:n - 1]
    yrs = np.asarray(ts.year)
    by = {}
    for y in sorted(set(yrs[warm:].tolist())):
        m = (yrs == y) & (np.arange(n) >= warm)
        by[int(y)] = round(float(pnl[m].sum()), 1)
    eq = np.cumsum(pnl[warm:])
    return dict(by_year=by, total=round(float(pnl[warm:].sum()), 1),
                cost=round(float(cost[warm:].sum()), 1),
                sharpe=round(float(pnl[warm:].mean() / (pnl[warm:].std() + 1e-12) * math.sqrt(BARS_PER_YEAR)), 2))


# ============================================================ main =============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/root/rivermind-fs/xauusd/data/XAUUSDc_M1_202201022305_202606262057.csv")
    ap.add_argument("--out", default="/root/rivermind-data/research-lab/reservoir_v1.json")
    ap.add_argument("--n-res", type=int, default=500)
    ap.add_argument("--density", type=float, default=0.10)
    ap.add_argument("--spectral", type=float, default=0.95)
    ap.add_argument("--in-scale", type=float, default=0.30)
    ap.add_argument("--lam", type=float, default=0.999)
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
    ap.add_argument("--seeds", default="42,1337,2026")
    ap.add_argument("--lam-sweep", default="0.995,0.998")
    ap.add_argument("--n-res-sweep", default="1000")
    args = ap.parse_args()
    cfg = vars(args)

    print("== Streaming RLS-Reservoir Engine v1 ==", flush=True)
    t0 = time.perf_counter()
    bar = load_m15(args.csv)
    D = build_feats(bar, horizon=args.horizon)
    print(f"data: {D['n']} M15 bars  {D['ts'][0]} .. {D['ts'][-1]}  "
          f"(load {time.perf_counter()-t0:.1f}s)", flush=True)
    atr_yr = {}
    yrs = np.asarray(D["ts"].year)
    for y in sorted(set(yrs.tolist())):
        atr_yr[int(y)] = round(float(D["atr"][yrs == y].mean()), 3)

    runs = {}
    primary_seed = 42
    # ---- 主运行 (3 种子) ----
    for sd in [int(s) for s in args.seeds.split(",")]:
        print(f"-- run seed={sd} lam={args.lam} n_res={args.n_res}", flush=True)
        r = run_pass(D, cfg, sd, collect=(sd == primary_seed), label=f"s{sd}")
        m = metrics(D, r, cfg)
        runs[f"seed{sd}"] = dict(metrics=m, timing=r["timing"], engine_mb=r["engine_mb"],
                                 K_final=r["K_final"],
                                 W_final_l2=round(float(np.linalg.norm(r["W_final"])), 3))
        if sd == primary_seed:
            runs["seed42_curves"] = dict(W_l2=r["W_curve"], K=r["K_curve"])
        print(f"   PnL={m['overall']['pnl']}$ IC_skew={m['overall']['ic_skew']} "
              f"IC_var={m['overall']['ic_var']} Sharpe={m['overall']['sharpe']}", flush=True)
    # ---- λ 扫描 (协方差缠绕对照) ----
    for lam in [float(x) for x in args.lam_sweep.split(",") if x.strip()]:
        c2 = dict(cfg); c2["lam"] = lam
        print(f"-- run seed=42 lam={lam} (windup probe)", flush=True)
        r = run_pass(D, c2, 42, label=f"lam{lam}")
        m = metrics(D, r, c2)
        runs[f"lam{lam}"] = dict(metrics=m, timing=r["timing"], engine_mb=r["engine_mb"])
        print(f"   PnL={m['overall']['pnl']}$ IC_skew={m['overall']['ic_skew']}", flush=True)
    # ---- 规模扫描 ----
    for nr in [int(x) for x in args.n_res_sweep.split(",") if x.strip()]:
        c3 = dict(cfg); c3["n_res"] = nr
        print(f"-- run seed=42 n_res={nr} (scale probe)", flush=True)
        r = run_pass(D, c3, 42, label=f"n{nr}")
        m = metrics(D, r, c3)
        runs[f"nres{nr}"] = dict(metrics=m, timing=r["timing"], engine_mb=r["engine_mb"])
        print(f"   PnL={m['overall']['pnl']}$ IC_skew={m['overall']['ic_skew']} "
              f"total_ms={r['timing']['total_ms']['mean']:.3f}", flush=True)
    # ---- 洗牌对照 (学习真实性) ----
    rng = np.random.default_rng(999)
    perm = rng.permutation(len(D["Y"]))
    print("-- control: shuffled labels", flush=True)
    r = run_pass(D, cfg, 42, y_perm=D["Y"][perm], label="shuf")
    m = metrics(D, r, cfg)
    runs["control_shuffle"] = dict(metrics=m, timing=r["timing"], engine_mb=r["engine_mb"])
    print(f"   PnL={m['overall']['pnl']}$ IC_skew={m['overall']['ic_skew']}", flush=True)
    # ---- 动量基线 (无学习) ----
    print("-- baseline: momentum", flush=True)
    runs["baseline_mom"] = momentum_baseline(D, cfg)

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3
    doc = dict(
        meta=dict(engine="StreamingRLSReservoir v1", bars=int(D["n"]),
                  span=[D["ts"][0].isoformat(), D["ts"][-1].isoformat()],
                  warm=int(args.warm), horizon=int(args.horizon),
                  convention="1.0 exposure = 1 oz; PnL $; cost=|dpos|*(0.3*ATR+spread)",
                  lam=float(args.lam), n_res=int(args.n_res),
                  density=float(args.density), spectral=float(args.spectral),
                  process_peak_mb=round(peak_mb, 1), atr_by_year=atr_yr,
                  bars_per_year=BARS_PER_YEAR),
        runs=runs,
    )
    with open(args.out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    print(f"== DONE -> {args.out} ({os.path.getsize(args.out)} bytes) ==", flush=True)


if __name__ == "__main__":
    main()
