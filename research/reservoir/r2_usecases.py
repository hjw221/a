#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
r2_usecases.py — Use Case 1/2/3 定量检验 (2026-10-03)
=====================================================
UC1 网格防毒流装甲: 被动挂单的逆向选择成本, 条件于 R1 能量预测分位.
    假设: 预测低能期挂单 → 4bar 后持仓损益分布更安全; 高能期挂单 → 被碾.
    检验: 买限价 @close-δ / 卖限价 @close+δ, δ=0.25×ATR, 触及即成交,
          持有至 t+4 收盘, 统计 E[PnL|成交] 按 R 十分位分层.
UC2 呼吸阀: 已有 decile 表 (champion), 此处重导出 + R→实现方差回归斜率.
UC3 断路器: d2 尖峰事件研究 — 极端单根行情(前1% |return|)前后的 d2 轨迹
    (断路器是否有【前瞻】预警时间, 而不只是事后同步).
"""
import json

import numpy as np
import pandas as pd

NPZ = "/tmp/r2arms/champ_s42.npz"
OUT = "/home/z/my-project/remote-ops-record/r2_20261003/usecase_results.json"


def main():
    z = np.load(NPZ)
    ts = pd.to_datetime(z["ts"], unit="ns")
    yhat = z["yhat"]; d2 = z["d2"]; atr = z["atr"]; vh4 = z["vh4"]
    y_var_raw = z["y_var_raw"]
    # 需要 close 序列: 从 npz 没存 —— 用 vh4 + atr 反推不行; 直接重载 M15
    import sys
    sys.path.insert(0, "/home/z/my-project/research/reservoir")
    import reservoir_engine2 as E2
    bar = E2.load_m15("/home/z/my-project/research-lab/data.csv", use_cache=True)
    D = E2.build_feats(bar, horizon=4, n_feat=8)
    c = D["c"]; h = D["h"]; l = D["l"]
    R = np.expm1(np.clip(yhat[:, 0], 0, 3))
    n = len(c); warm = 2000
    valid = np.zeros(n, bool); valid[warm:n - 40] = True

    # ================= UC1: 网格装甲 =================
    print("== UC1 网格防毒流装甲 (被动挂单逆向选择 vs R1 能量预测) ==")
    delta = 0.25  # ×ATR
    fills_buy = np.zeros(n); pnl_buy = np.full(n, np.nan)
    fills_sell = np.zeros(n); pnl_sell = np.full(n, np.nan)
    tgt_b = c - delta * atr   # 买挂单价
    tgt_s = c + delta * atr   # 卖挂单价
    H = 4
    for t in range(warm, n - H - 1):
        # 未来 H 根是否触及挂单价 (low <= tgt_b for buy; high >= tgt_s for sell)
        lb = l[t + 1:t + 1 + H].min()
        hs = h[t + 1:t + 1 + H].max()
        px_end = c[t + H]
        if lb <= tgt_b[t]:
            fills_buy[t] = 1
            pnl_buy[t] = px_end - tgt_b[t]   # 成交后持有到 t+4 的浮盈
        if hs >= tgt_s[t]:
            fills_sell[t] = 1
            pnl_sell[t] = tgt_s[t] - px_end
    mv = valid & np.isfinite(pnl_buy)
    # R 十分位
    qs = np.quantile(R[mv], np.linspace(0, 1, 11))
    rows = []
    for i in range(10):
        msk = mv & (R >= qs[i]) & (R < qs[i + 1] if i < 9 else R <= qs[i + 1])
        if msk.sum() < 100:
            continue
        fb = fills_buy[msk]; pb = pnl_buy[msk]
        fs = fills_sell[msk]; ps = pnl_sell[msk]
        rows.append(dict(
            decile=i + 1,
            n=int(msk.sum()),
            buy_fill=round(float(fb.mean()), 3),
            buy_adverse=round(float(np.nanmean(np.where(fb == 1, pb, np.nan))), 3),
            buy_tail=round(float(np.nanpercentile(np.where(fb == 1, pb, np.nan), 5)), 3) if fb.sum() > 30 else None,
            sell_fill=round(float(fs.mean()), 3),
            sell_adverse=round(float(np.nanmean(np.where(fs == 1, ps, np.nan))), 3),
        ))
    for r in rows:
        print(f"  d{r['decile']:2d} R∈[{qs[r['decile']-1]:.2f},{qs[r['decile']]:.2f}] "
              f"n={r['n']:5d} | 买挂: fill={r['buy_fill']:.2f} adverse=${r['buy_adverse']:+.3f} "
              f"p5=${(r['buy_tail'] or 0):+.3f} | 卖挂: fill={r['sell_fill']:.2f} adverse=${r['sell_adverse']:+.3f}")
    d1 = rows[0]; d10 = rows[-1]
    uc1 = dict(rows=rows,
               verdict=dict(
                   buy_adverse_d1=d1["buy_adverse"], buy_adverse_d10=d10["buy_adverse"],
                   delta_d10_minus_d1=round(d10["buy_adverse"] - d1["buy_adverse"], 3),
                   note="adverse = 挂单成交后持有4bar的浮盈; 负=逆向选择损失"))

    # ================= UC2: 呼吸阀 =================
    print("\n== UC2 呼吸阀 (R 分位 → 未来1h实现方差比) ==")
    m = valid & np.isfinite(vh4)
    qs2 = np.quantile(R[m], np.linspace(0, 1, 11))
    dec = []
    for i in range(10):
        msk = m & (R >= qs2[i]) & (R < qs2[i + 1] if i < 9 else R <= qs2[i + 1])
        dec.append(round(float(np.nanmean(vh4[msk])), 3))
    # 回归斜率: E[realized | R] 线性
    slope = float(np.polyfit(R[m], vh4[m], 1)[0])
    uc2 = dict(decile_mean_realized=dec, slope=round(slope, 3),
               spread=round(dec[-1] / max(dec[0], 1e-9), 2))
    print(f"  deciles={dec}")
    print(f"  slope={slope:.3f} (每+1.0 R → +{slope:.2f}× 实现方差)  spread d10/d1={uc2['spread']}×")

    # ================= UC3: 断路器事件研究 =================
    print("\n== UC3 断路器 (d2 尖峰对极端行情的前瞻性) ==")
    ret = np.diff(np.log(c))
    aret = np.abs(ret)
    thr = np.percentile(aret[warm:], 99.5)  # 极端单根行情
    ev = np.where(aret > thr)[0] + 1  # 发生在 bar t (ret[t] = c[t]/c[t-1]-1)
    ev = ev[(ev > warm + 8) & (ev < n - 8)]
    # 事件窗口 d2 轨迹: [-8..0..+8]
    prof = []
    for e in ev:
        prof.append(d2[e - 8:e + 8])
    prof = np.asarray(prof)
    mean_prof = prof.mean(0)
    # d2 预警率: 事件前 1-4 根内 d2>=25 的比例
    pre = (prof[:, 4:8] >= 25).any(1).mean()
    # 基线率: 全样本任意 4 根窗口 d2>=25 的比例
    base = (d2[warm:n - 40] >= 25).mean()
    # 前瞻方差: 事件前4根的 d2 高 vs 低的未来方差
    m_pre = valid.copy()
    fwd = vh4
    hi_pre = m_pre & np.roll(np.concatenate(([False] * 4, (d2[:-4] >= 25))), 4)
    uc3 = dict(n_events=int(len(ev)),
               mean_d2_profile=[round(x, 1) for x in mean_prof.tolist()],
               pre_alert_rate=round(float(pre), 3),
               base_rate=round(float(base), 4),
               lift=round(float(pre / max(base, 1e-9)), 1),
               fwd_var_after_d2spike=round(float(np.nanmean(fwd[hi_pre])) / max(float(np.nanmedian(fwd[m_pre & np.isfinite(fwd)])), 1e-9), 2))
    print(f"  极端行情事件 n={len(ev)} (|ret|>p99.5={thr:.4f})")
    print(f"  d2 轨迹[-8..+8]: {[round(x,1) for x in mean_prof.tolist()]}")
    print(f"  事件前4根 d2>=25 比例: {pre:.2f} vs 基线 {base:.4f} → lift {uc3['lift']}×")
    print(f"  d2尖峰后未来1h方差: {uc3['fwd_var_after_d2spike']}× 中位")

    # 事件日期样本 (供人工核验: 对应已知新闻/央行日)
    sample = [str(ts[i]) for i in ev[:12]]
    uc3["event_dates_sample"] = sample
    print(f"  事件样本(前12): {sample[:6]} ...")

    doc = dict(meta=dict(generated="2026-10-03", npz="champ_s42", n=int(n)),
               uc1_grid_shield=uc1, uc2_breathing=uc2, uc3_sentinel=uc3)
    json.dump(doc, open(OUT, "w"), indent=1, default=str)
    print(f"\n== DONE -> {OUT} ==")


if __name__ == "__main__":
    main()
