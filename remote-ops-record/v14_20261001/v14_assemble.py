#!/usr/bin/env python3
"""v14_assemble.py — v14 年度均衡组合装配 (2026-10-01, 沙箱本地).

背景: 用户质疑 v13 "2026 贡献 82% 太多". v14 双臂(服务器)已跑完:
  Arm2 caparms: champ 的 5 种美元钳制重仿真 (eq1.2_2.0 最好: 2023 +203, 2026 share 7.8%)
  Arm1 weaksearch: 弱年份目标搜索 -> a7f3cb3ee3 决赛 54折 total +$1211
     {2022:+28.7, 2023:+109.0, 2024:-47.8, 2025:+570.0, 2026:+551.3} 2026 share 45.5%

装配协议 (Task 25 预定, 不因结果回改):
  book 池 = champ + weak + v11_hl0 + eq1.2_2.0 (最优钳制臂), 元分配 W=2
  元分配规则 = v13 同款纯因果策略动量: 每月 book 开启 iff 自身滞后 W 月 PnL 和 > 0

验收标准 (Task 25 预定):
  A1 总量 > 732.7   A2 逐年全正   A3 2026 占比 <= 70%   A4 2022-24 捕获比 >= 0.3
附加透明度: 全子集 x W 网格敏感性表 (选择偏差自曝), vol-flat 平减视角
"""
import os, sys, json, time, itertools
import numpy as np
import pandas as pd

T0 = time.time()
def log(s): print(f"[V14 {time.time()-T0:5.0f}s] {s}", flush=True)

BASE = os.path.dirname(os.path.abspath(__file__))
REC = "/home/z/my-project/remote-ops-record"
OUT = f"{REC}/v14_20261001/assembly"
os.makedirs(OUT, exist_ok=True)

HIST_TARGET = 732.7
YEARS = ["2022", "2023", "2024", "2025", "2026"]
ATR_MEAN = {2022: 0.534, 2023: 0.475, 2024: 0.732, 2025: 1.390, 2026: 3.078}
VOLFLAT_SCALE = {y: 0.732 / a for y, a in ATR_MEAN.items()}  # 以 2024 中位波幅为基准

# ---------------------------------------------------------------- 载入 book 池
def load_books():
    books = {}
    # 1) champ — 研究线 $818 冠军 (41e66400ee)
    c = json.load(open(f"{REC}/research_20260930/results_research/champion.json"))
    books["champ"] = pd.Series({r["month"]: float(r["pnl"]) for r in c["final"]["folds"]})
    # 2) weak — weaksearch 决赛 a7f3cb3ee3 (+$1211)
    recs = [json.loads(l) for l in open(f"{REC}/v14_20261001/finals_weak.jsonl") if l.strip()]
    w = [r for r in recs if r["cfg_id"] == "a7f3cb3ee3"][0]
    books["weak"] = pd.Series({r["month"]: float(r["pnl"]) for r in w["final_folds"]})
    # 3) v11_hl0 — v13 组合成员 (+$755)
    s = pd.read_csv(f"{REC}/v13_champion_20261001/v11_hl0_v3_s42-1337-2024/monthly_pnl.csv", index_col=0)["pnl"]
    books["v11_hl0"] = pd.Series({str(m): float(v) for m, v in s.items()})
    # 4/5) eq 钳制臂 — champ 同信号常数美元风险重仿真
    cap = json.load(open(f"{REC}/v14_20261001/caparms.json"))
    for arm in ["eq1.2_2.0", "eq2.4_3.0"]:
        books[arm] = pd.Series({m: float(v) for m, v in cap[arm]["monthly"].items()})
    for b, s in books.items():
        log(f"book {b:10s}: {len(s)}月 总PnL={s.sum():+8.1f} 负月={(s<0).sum():2d} "
            f"2026占比={s[[m for m in s.index if m[:4]=='2026']].sum()/s.sum():.2f}")
    return books

# ---------------------------------------------------------------- 元分配
def meta(df, w):
    active = pd.DataFrame(False, index=df.index, columns=df.columns)
    for i, m in enumerate(df.index):
        if i >= w:
            hist = df.iloc[i-w:i]           # 严格因果: 不含当月
            active.loc[m] = (hist.sum() > 0).values
    return df.where(active, 0.0), active

def metrics(port, nb_books):
    yr = {y: float(port[port.index.str[:4] == y].sum()) for y in YEARS}
    tot = float(port.sum())
    sharpe = float(port.mean() / (port.std() + 1e-12) * np.sqrt(12))
    cum = port.cumsum()
    maxdd = float((cum - cum.cummax()).min())
    vf = {y: round(yr[y] * VOLFLAT_SCALE[int(y)], 1) for y in YEARS}
    return {
        "total": round(tot, 1), "per_book_equivalent": round(tot / nb_books, 1),
        "by_year": {k: round(v, 1) for k, v in yr.items()},
        "by_year_volflat": vf,
        "volflat_total": round(sum(vf.values()), 1),
        "all_years_positive": all(v > 0 for v in yr.values()),
        "share_2026": round(yr["2026"] / tot, 3) if tot > 0 else None,
        "capture_2224": round((yr["2022"] + yr["2023"] + yr["2024"]) / tot, 3) if tot > 0 else None,
        "sharpe": round(sharpe, 2), "max_monthly_drawdown": round(maxdd, 1),
        "months_traded": int((port != 0).sum()), "months_total": len(port),
        "beats_732_total": tot > HIST_TARGET,
        "beats_732_per_book": tot / nb_books > HIST_TARGET,
    }

def accepts(m):
    return (m["beats_732_total"] and m["all_years_positive"]
            and m["share_2026"] is not None and m["share_2026"] <= 0.70
            and m["capture_2224"] is not None and m["capture_2224"] >= 0.30)

books = load_books()
allm = sorted(set().union(*[set(s.index) for s in books.values()]))
df_all = pd.DataFrame({k: s.reindex(allm).fillna(0.0) for k, s in books.items()})

# ---------------------------------------------------------------- 主装配 (预定协议)
PRIMARY = ["champ", "weak", "v11_hl0", "eq1.2_2.0"]
W = 2
dfp = df_all[PRIMARY]
alloc, active = meta(dfp, W)
port = alloc.sum(axis=1)
pm = metrics(port, len(PRIMARY))
pm["books"] = PRIMARY; pm["W"] = W
pm["book_contribution"] = {b: round(float(alloc[b].sum()), 1) for b in PRIMARY}
pm["book_active_months"] = {b: int(active[b].sum()) for b in PRIMARY}
pm["acceptance"] = {"A1_total>732": pm["beats_732_total"], "A2_all_years_pos": pm["all_years_positive"],
                    "A3_share26<=70%": pm["share_2026"] <= 0.70 if pm["share_2026"] else False,
                    "A4_capture2224>=0.3": pm["capture_2224"] >= 0.30 if pm["capture_2224"] else False}
log(f"PRIMARY {PRIMARY} W={W}: total={pm['total']:+.1f} per_book={pm['per_book_equivalent']:+.1f}")
log(f"  by_year={pm['by_year']} share26={pm['share_2026']} cap2224={pm['capture_2224']}")
log(f"  acceptance={pm['acceptance']}")

port.rename("pnl").to_csv(f"{OUT}/portfolio_monthly.csv")
active.astype(int).to_csv(f"{OUT}/book_active.csv")
alloc.to_csv(f"{OUT}/book_alloc_monthly.csv")

# ---------------------------------------------------------------- 全子集 x W 敏感性 (自曝选择偏差)
grid = []
names = list(df_all.columns)
for r in range(1, len(names) + 1):
    for ss in itertools.combinations(names, r):
        d = df_all[list(ss)]
        for w in range(1, 6):
            al, ac = meta(d, w)
            m = metrics(al.sum(axis=1), len(ss))
            m["books"] = list(ss); m["W"] = w
            m["PASS"] = accepts(m)
            grid.append(m)
gdf = pd.DataFrame(grid).sort_values(["PASS", "total"], ascending=[False, False])
gdf.to_json(f"{OUT}/subset_W_grid.jsonl", orient="records", lines=True)
n_pass = int(gdf["PASS"].sum())
log(f"网格: {len(gdf)} 组合xW 配置, 其中 {n_pass} 个过验收")

# 权益曲线
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = pd.to_datetime([m + "-15" for m in dfp.index])
    for b in PRIMARY:
        ax.plot(x, dfp[b].cumsum(), lw=1.0, alpha=0.55, label=f"{b} raw")
    ax.plot(x, port.cumsum(), lw=2.2, color="black", label=f"v14 portfolio W={W}")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axhline(HIST_TARGET, color="red", ls="--", lw=0.8, label="hist 732.7")
    ax.set_title(f"v14 {len(PRIMARY)}-book meta-allocation: total={pm['total']:+.0f} "
                 f"share26={pm['share_2026']:.0%} all-years-pos={pm['all_years_positive']}")
    ax.set_ylabel("cum PnL $"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/equity.png", dpi=110)
    log("equity.png saved")
except Exception as e:
    log(f"画图跳过: {e}")

summary = {
    "primary": pm,
    "grid_pass_count": n_pass, "grid_total": len(gdf),
    "top5_passing": gdf[gdf["PASS"]].head(5)[["books", "W", "total", "by_year", "share_2026", "capture_2224", "sharpe"]].to_dict("records") if n_pass else [],
    "notes": [
        "验收标准为 Task 25 在双臂启动前预定: 总量>732 + 逐年全正 + 2026占比<=70% + 2022-24捕获>=0.3",
        "weak book a7f3cb3ee3 从 144 配置中按 weak2324 目标选出; champ 从 ~558 配置选出 — 多重检验偏差存在",
        "eq1.2_2.0 是 champ 同信号钳制重仿真(与 champ 高相关, 差异仅仓位几何)",
        "2026 占比 45.5% 的 weak book 是本次年度均衡的核心改善来源",
    ],
}
with open(f"{OUT}/summary.json", "w") as fh:
    json.dump(summary, fh, indent=2, ensure_ascii=False)
log("summary.json saved")

# ---------------------------------------------------------------- 控制台总表
print("\n===== PRIMARY 4-book W=2 =====")
print(json.dumps({k: pm[k] for k in ["total", "per_book_equivalent", "by_year", "by_year_volflat",
      "share_2026", "capture_2224", "all_years_positive", "sharpe", "max_monthly_drawdown",
      "months_traded", "book_contribution", "book_active_months", "acceptance"]}, indent=1))
print("\n===== 网格 TOP10 (按 total) =====")
cols = ["books", "W", "total", "share_2026", "capture_2224", "all_years_positive", "PASS"]
print(gdf[cols].head(10).to_string(index=False))
print(f"\n过验收组合数: {n_pass}/{len(gdf)}")
