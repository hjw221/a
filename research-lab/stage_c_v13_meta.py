"""Stage C v13 — 策略动量元分配器 (2026-10-01).

证据链:
  v10 $962 = 长仓偏置蹭牛市 (100% long), 2022-23 未考核
  v11 四臂 = 因子方向无OOS alpha, 2022-24 全流血
  研究线冠军 $818 = 2022/23 正但 2024 -178
  => 没有任何单一静态族全天候; 失败模式互补 => 组合 + 因果开关

元分配规则 (纯因果, 零拟合):
  每月初, book 开启 iff 该 book 滞后 W=2 个月实盘 PnL 之和 > 0
  (策略动量: 自己最近的realized表现决定下月是否上岗, 不用任何未来数据)

输入: 各 book 的月度 PnL 序列 (OOS 2022-08~2026-07, 48 月)
  champ   = research-lab/results_research/champion.json final.folds
  v11_*   = stage_c/v11_*/monthly_pnl.csv
  v12_*   = stage_c/v12_*/monthly_pnl.csv
用法: python3 stage_c_v13_meta.py --books champ,v11_hl0 [--W 2]
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd

T0 = time.time()
def log(s): print(f"[V13 {time.time()-T0:6.0f}s] {s}", flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--books", default="champ,v11_hl0")
ap.add_argument("--W", type=int, default=2)
ap.add_argument("--outdir", default="/root/rivermind-data/xauusd/stage_c/v13_meta")
args = ap.parse_args()
NTFY = "xauusd-qv7m2zk9-res"
HIST_TARGET = 732.7
RL = "/root/rivermind-data/research-lab/results_research"
SC = "/root/rivermind-data/xauusd/stage_c"
os.makedirs(args.outdir, exist_ok=True)

def load_book(name):
    if name == "champ":
        c = json.load(open(f"{RL}/champion.json"))
        return pd.Series({r["month"]: float(r["pnl"]) for r in c["final"]["folds"]}), c["cfg"]
    if name.startswith("v11_"):
        d = f"{SC}/{name}_s42-1337-2024/monthly_pnl.csv"
    else:
        d = f"{SC}/{name}_s42-1337-2024/monthly_pnl.csv"
    s = pd.read_csv(d, index_col=0)["pnl"]
    return s, None

books = args.books.split(",")
series, cfgs = {}, {}
for b in books:
    s, cfg = load_book(b)
    series[b] = s
    if cfg: cfgs[b] = cfg
    log(f"book {b}: {len(s)}月 总PnL={s.sum():+.1f} 负月={(s<0).sum()}")
allm = sorted(set().union(*[set(s.index) for s in series.values()]))
df = pd.DataFrame({k: s.reindex(allm).fillna(0.0) for k, s in series.items()})

# ---------------------------------------------------------------- 元分配
W = args.W
active = pd.DataFrame(False, index=df.index, columns=df.columns)
for i, m in enumerate(df.index):
    if i >= W:
        hist = df.iloc[i-W:i]           # 滞后 W 个月 (不含当月) — 严格因果
        active.loc[m] = (hist.sum() > 0).values
alloc = df.where(active, 0.0)
port = alloc.sum(axis=1)
tot = float(port.sum())
nb = len(books)
yr = {y: float(port[port.index.str[:4] == y].sum()) for y in ["2022", "2023", "2024", "2025", "2026"]}
sharpe = float(port.mean() / (port.std() + 1e-12) * np.sqrt(12))
cum = port.cumsum()
maxdd = float((cum - cum.cummax()).min())
allpos = all(v > 0 for v in yr.values())

port.rename("pnl").to_csv(f"{args.outdir}/portfolio_monthly.csv")
active.astype(int).to_csv(f"{args.outdir}/book_active.csv")
alloc.to_csv(f"{args.outdir}/book_alloc_monthly.csv")

# 敏感性附录 (W x 子集 x 规则)
def meta(d, w, rule="sum", subset=None):
    dd = d[subset] if subset else d
    act = pd.DataFrame(False, index=dd.index, columns=dd.columns)
    for i, m in enumerate(dd.index):
        if i >= w:
            h = dd.iloc[i-w:i]
            act.loc[m] = (h.sum() > 0).values if rule == "sum" else (h > 0).all().values
    al = dd.where(act, 0.0)
    p = al.sum(axis=1)
    y = {yy: float(p[p.index.str[:4] == yy].sum()) for yy in ["2022", "2023", "2024", "2025", "2026"]}
    return {"total": round(float(p.sum()), 1), "by_year": {k: round(v, 1) for k, v in y.items()},
            "all_years_positive": all(v > 0 for v in y.values()),
            "sharpe": round(float(p.mean() / (p.std() + 1e-12) * np.sqrt(12)), 2)}

sens = {"W": {w: meta(df, w) for w in [1, 2, 3, 4, 5]},
        "subsets": {",".join(ss): meta(df, W, subset=list(ss))
                    for r in range(1, len(books) + 1)
                    for ss in __import__("itertools").combinations(books, r)},
        "rule": {r: meta(df, W, rule=r) for r in ["sum", "min"]}}

summary = {
    "books": books, "W": W,
    "portfolio_total": round(tot, 1),
    "per_book_equivalent": round(tot / nb, 1),
    "hist_target": HIST_TARGET,
    "beats_target_portfolio": tot > HIST_TARGET,
    "beats_target_per_book": tot / nb > HIST_TARGET,
    "by_year": {k: round(v, 1) for k, v in yr.items()},
    "by_year_per_book": {k: round(v / nb, 1) for k, v in yr.items()},
    "all_years_positive": allpos,
    "sharpe_monthly_annualized": round(sharpe, 2),
    "max_monthly_drawdown": round(maxdd, 1),
    "months_traded": int((port != 0).sum()), "months_total": len(port),
    "book_contribution": {b: round(float(alloc[b].sum()), 1) for b in books},
    "book_active_months": {b: int(active[b].sum()) for b in books},
    "sensitivity": sens,
}
with open(f"{args.outdir}/summary.json", "w") as fh:
    json.dump(summary, fh, indent=2, ensure_ascii=False)

log(f"PORTFOLIO books={books} W={W}")
log(f"  total={tot:+.1f} (per-book {tot/nb:+.1f}) | sharpe={sharpe:.2f} maxDD={maxdd:.1f}")
log(f"  BY_YEAR: " + " | ".join(f"{y}:{v:+.1f}" for y, v in yr.items())
    + f" | 全年正={allpos}")
msg = (f"V13组合[{'+'.join(books)}] W={W} total={tot:+.0f}$ "
       f"per-book={tot/nb:+.0f}$ sharpe={sharpe:.2f} 全年正={allpos} | "
       + " ".join(f"{y}:{v:+.0f}" for y, v in yr.items()))
os.system(f"curl -s -m 20 -d '{msg}' ntfy.sh/{NTFY} > /dev/null 2>&1 &")

# ---------------------------------------------------------------- 权益曲线
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = pd.to_datetime([m + "-15" for m in df.index])
    for b in books:
        ax.plot(x, df[b].cumsum(), lw=1.0, alpha=0.55, label=f"{b} (raw)")
    ax.plot(x, port.cumsum(), lw=2.2, color="black", label=f"portfolio (W={W} gate)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axhline(HIST_TARGET, color="red", ls="--", lw=0.8, label="hist 732.7")
    ax.set_title(f"v13 meta-allocation: {'+'.join(books)}  total={tot:+.0f} "
                 f"(per-book {tot/nb:+.0f})  all-years-positive={allpos}")
    ax.set_ylabel("cum PnL $"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{args.outdir}/equity.png", dpi=110)
    log("equity.png saved")
except Exception as e:
    log(f"画图跳过: {e}")
