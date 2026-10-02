#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
年度集中度审计 — 回答 "2026年贡献82%是否太多"
解剖四件事:
  A. 市场结构: 各年份的波幅预算/波段数/肉量份额 (市场把多少钱放在哪年)
  B. 历史 $732.7 冠军自己的年度分布 (公平对照尺 — 它的OOS窗口是什么)
  C. v13 组合/两book 的年度分布 + 美元平减(vol-flat)重估 (恒定美元风险下会怎样)
  D. 单笔规模通胀: v11 逐笔 by year (TP/SL 随 ATR 放大) + champ 逐月
  E. Pareto 张力数学: 2026占比上限 vs 非2026所需alpha
"""
import json
import numpy as np
import pandas as pd

CSV   = "/home/z/my-project/research-lab/data.csv"
V13   = "/home/z/my-project/remote-ops-record/v13_champion_20261001"
HIST  = "/home/z/my-project/research-lab/a-server/server/v2_ens/results/historical_reference/per_fold_m1sc_ad.json"
CHAMP = "/home/z/my-project/remote-ops-record/research_20260930/results_research/champion.json"
OUT   = "/home/z/my-project/year_audit_20261001.json"
TARGET = 732.7

# ============ A. 市场结构 ============
print("=" * 100)
print("A. 市场结构 (data.csv 2022-01 ~ 2026-06, M1)")
print("=" * 100)
raw = pd.read_csv(CSV, sep="\t", usecols=["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>"])
dt = pd.to_datetime(raw["<DATE>"] + " " + raw["<TIME>"], format="%Y.%m.%d %H:%M:%S")
h = raw["<HIGH>"].to_numpy(np.float64)
l = raw["<LOW>"].to_numpy(np.float64)
c = raw["<CLOSE>"].to_numpy(np.float64)
pc = np.concatenate([[c[0]], c[:-1]])
tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
atr = pd.Series(tr).rolling(288, min_periods=288).mean().to_numpy()

# 前向90根最大有利波动 (Stage A 的"波段/可吃性"口径)
fw_hi = pd.Series(h).iloc[::-1].rolling(90, min_periods=1).max().iloc[::-1].to_numpy()
fw_lo = pd.Series(l).iloc[::-1].rolling(90, min_periods=1).min().iloc[::-1].to_numpy()
mfe = np.maximum(fw_hi - c, c - fw_lo)
eat = (mfe >= 1.5 * atr).astype(float)   # NaN atr -> 0

year = dt.dt.year.to_numpy()
month = dt.dt.strftime("%Y-%m").to_numpy()
mk = pd.DataFrame({"year": year, "month": month, "tr": tr, "atr": atr,
                   "mfe": mfe, "eat": eat, "c": c})

mkt = []
for y, sub in mk.groupby("year"):
    cc = sub["c"].to_numpy()
    mkt.append(dict(
        year=int(y), n_bars=len(sub),
        px0=float(cc[0]), px1=float(cc[-1]),
        chg_pct=float(cc[-1] / cc[0] - 1) * 100,
        atr_mean=float(np.nanmean(sub["atr"])),
        vol_budget=float(np.nansum(sub["atr"])),      # 风险美元 x 时间
        eat_n=int(sub["eat"].sum()),
        eat_rate=float(sub["eat"].mean()),
        mfe_meat=float(sub["mfe"].sum()),
    ))
for r in mkt:
    r["vol_share"] = r["vol_budget"] / sum(x["vol_budget"] for x in mkt)
    r["eat_share"] = r["eat_n"] / sum(x["eat_n"] for x in mkt)
    r["meat_share"] = r["mfe_meat"] / sum(x["mfe_meat"] for x in mkt)
a23 = [r for r in mkt if r["year"] == 2023][0]["atr_mean"]
for r in mkt:
    r["atr_x_2023"] = r["atr_mean"] / a23

print(f"{'年':>6}{'bars':>9}{'价格':>10}{'→':>1}{'':>9}{'涨跌%':>8}{'ATR288$':>9}{'x2023':>7}"
      f"{'波幅预算%':>10}{'波段数%':>9}{'肉量%':>8}{'可吃率%':>9}")
for r in mkt:
    print(f"{r['year']:>6}{r['n_bars']:>9}{r['px0']:>10.1f}→{r['px1']:>9.1f}{r['chg_pct']:>8.1f}"
          f"{r['atr_mean']:>9.2f}{r['atr_x_2023']:>7.2f}{r['vol_share']*100:>10.1f}"
          f"{r['eat_share']*100:>9.1f}{r['meat_share']*100:>8.1f}{r['eat_rate']*100:>9.1f}")

# 历史冠军窗口内的结构份额 (2024-08 起, 与其OOS同窗)
hw = mk[mk["month"] >= "2024-08"]
h_vol = hw.groupby("year")["atr"].sum()
h_eat = hw.groupby("year")["eat"].sum()
h_vol_share = (h_vol / h_vol.sum()).to_dict()
h_eat_share = (h_eat / h_eat.sum()).to_dict()

# 月度 ATR (vol-flat 用)
matr = mk.groupby("month")["atr"].mean()
med_atr = float(np.nanmedian(matr.values))

# ============ 工具 ============
def by_year(s: pd.Series):
    y = pd.Series(index=[int(str(i)[:4]) for i in s.index], data=s.values.astype(float))
    return y.groupby(level=0).sum().to_dict()

def volflat(s: pd.Series):
    out = {}
    for m, v in s.items():
        a = matr.get(m, np.nan)
        out[m] = float(v) * (med_atr / a) if (a == a and a > 0) else 0.0
    return pd.Series(out)

def share(d, y=2026):
    tot = sum(d.values())
    return (d.get(y, 0.0) / tot) if tot else 0.0

def show_strategy(name, s: pd.Series, structural=None):
    d_raw = {int(k): float(v) for k, v in by_year(s).items()}
    d_flat = {int(k): float(v) for k, v in by_year(volflat(s)).items()}
    tot = float(s.sum())
    print(f"\n--- {name} | 总PnL ${tot:,.1f} | 月数 {len(s)} ---")
    print(f"{'年':>6}{'PnL$':>10}{'占比%':>8}{'volflat$':>10}{'vf占比%':>9}"
          + (f"{'结构份额%':>10}{'捕获比':>8}" if structural else ""))
    for y in sorted(d_raw):
        sr = d_raw[y] / tot * 100 if tot else 0
        tf = sum(d_flat.values())
        sf = d_flat[y] / tf * 100 if tf else 0
        line = f"{y:>6}{d_raw[y]:>10.1f}{sr:>8.1f}{d_flat[y]:>10.1f}{sf:>9.1f}"
        if structural:
            st = structural.get(y, 0.0) * 100
            cap = (d_raw[y] / tot) / structural.get(y, 1e-9) if structural.get(y, 0) > 0 else np.nan
            line += f"{st:>10.1f}{cap:>8.2f}"
        print(line)
    print(f"  2026占比: 原始 {share(d_raw)*100:.1f}% -> vol-flat {share(d_flat)*100:.1f}%")
    return dict(name=name, total=tot, by_year=d_raw, by_year_volflat=d_flat,
                share_2026=share(d_raw), share_2026_volflat=share(d_flat))

# ============ B/C. 各策略年度分布 ============
print()
print("=" * 100)
print("B/C. 策略年度分布: 原始 vs vol-flat(恒定美元风险重估) vs 结构份额/捕获比")
print("=" * 100)

v13 = pd.read_csv(f"{V13}/v13_meta/portfolio_monthly.csv", index_col=0)["pnl"]
v13_3b = pd.read_csv(f"{V13}/v13_meta_3b/portfolio_monthly.csv", index_col=0)["pnl"]
champ_folds = json.load(open(CHAMP))["final"]["folds"]
champ = pd.Series({f["month"]: f["pnl"] for f in champ_folds})
hist_folds = json.load(open(HIST))
hist = pd.Series({f["oos_month"]: f["ml"]["total_pnl"] for f in hist_folds})
v11 = pd.read_csv(f"{V13}/v11_hl0_v3_s42-1337-2024/trades_oos.csv")
v11m = v11.groupby("fold")["pnl"].sum()

vol_share_all = {r["year"]: r["vol_share"] for r in mkt}
res_v13 = show_strategy("v13 2-book 组合 (portfolio)", v13, vol_share_all)
res_3b = show_strategy("v13 3-book 变体", v13_3b, vol_share_all)
res_champ = show_strategy("Book A: 研究线冠军 $818 (48折)", champ, vol_share_all)
res_v11 = show_strategy("Book B: v11_hl0_v3 (全月序列)", v11m, vol_share_all)
res_hist = show_strategy(f"历史 $732.7 冠军 (OOS {hist.index.min()}~{hist.index.max()}, {len(hist)}折)",
                         hist, h_vol_share)

# ============ D. 单笔规模通胀 ============
print()
print("=" * 100)
print("D. 单笔规模通胀 (ATR 驱动 TP/SL 放大)")
print("=" * 100)
v11["yr"] = pd.to_datetime(v11["entry_time"]).dt.year
print("\nv11_hl0_v3 逐笔分解:")
print(f"{'年':>6}{'笔数':>7}{'胜率%':>7}{'avg赢$':>9}{'avg亏$':>9}{'PnL$':>10}{'多头%':>8}{'空头%':>8}")
v11_rows = []
for y, sub in v11.groupby("yr"):
    wins = sub[sub["pnl"] > 0]["pnl"]
    loss = sub[sub["pnl"] <= 0]["pnl"]
    nl = (sub["dir"] == "long").sum()
    v11_rows.append(dict(year=int(y), n=len(sub), wr=float(len(wins) / len(sub)),
                         avg_win=float(wins.mean()), avg_loss=float(loss.mean()),
                         pnl=float(sub["pnl"].sum()), long_pct=float(nl / len(sub))))
    print(f"{y:>6}{len(sub):>7}{len(wins)/len(sub)*100:>7.1f}{wins.mean():>9.2f}{loss.mean():>9.2f}"
          f"{sub['pnl'].sum():>10.1f}{nl/len(sub)*100:>8.1f}{(1-nl/len(sub))*100:>8.1f}")

print("\nBook A (champ) 逐月折 → 年:")
print(f"{'年':>6}{'笔数':>7}{'胜率%':>7}{'PnL$':>10}{'$/笔':>8}")
ch_rows = []
cf = pd.DataFrame(champ_folds)
cf["yr"] = cf["month"].str.slice(0, 4).astype(int)
for y, sub in cf.groupby("yr"):
    ch_rows.append(dict(year=int(y), trades=int(sub["trades"].sum()),
                        wr=float(sub["wins"].sum() / sub["trades"].sum()),
                        pnl=float(sub["pnl"].sum()),
                        per_trade=float(sub["pnl"].sum() / sub["trades"].sum())))
    print(f"{y:>6}{sub['trades'].sum():>7}{sub['wins'].sum()/sub['trades'].sum()*100:>7.1f}"
          f"{sub['pnl'].sum():>10.1f}{sub['pnl'].sum()/sub['trades'].sum():>8.3f}")

# v13 2026 月度内部
print("\nv13 组合 2026 月度 (利润的内部集中):")
for m, v in v13.items():
    if str(m).startswith("2026"):
        print(f"  {m}: {v:+10.1f}")

# ============ E. Pareto 张力数学 ============
print()
print("=" * 100)
print("E. Pareto 张力: 2026占比上限 x 非当年所需alpha (单book口径, 目标$732.7)")
print("=" * 100)
non26_book = 817.8 - 672.8   # v13 单book等价 2022-25 合计
print(f"当前单book等价: 总 $817.8 | 2022-25合计 ${non26_book:.0f} | 2026 ${672.8:.0f} (82.3%)")
print(f"{'2026占比上限':>12}{'需要的非2026$':>14}{'现状倍数':>10}{'可行性':>10}")
for cap in [0.82, 0.70, 0.60, 0.50, 0.40]:
    need = (1 - cap) / cap * TARGET
    print(f"{cap*100:>11.0f}%{need:>14.0f}{need/non26_book:>10.1f}x"
          f"{'当前已是' if abs(cap-0.82)<1e-6 else ('需新alpha' if need > non26_book*1.3 else '边际可达')}")

report = dict(
    market=[{k: v for k, v in r.items()} for r in mkt],
    hist_window=dict(first=str(hist.index.min()), last=str(hist.index.max()), folds=len(hist)),
    hist_window_vol_share={int(k): float(v) for k, v in h_vol_share.items()},
    strategies=[res_v13, res_3b, res_champ, res_v11, res_hist],
    v11_trades_by_year=v11_rows,
    champ_by_year=ch_rows,
    v13_2026_monthly={str(m): float(v) for m, v in v13.items() if str(m).startswith("2026")},
    tension=dict(non26_book_per_equiv=non26_book, target=TARGET),
    med_month_atr=med_atr,
)
json.dump(report, open(OUT, "w"), indent=1, ensure_ascii=False)
print(f"\nsaved -> {OUT}")
