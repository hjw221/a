"""
PDF报告专用图表: 严格遵循 charts.md (删顶/右脊线、虚线网格20%、线宽2.5pt、
图例无边框置顶、外部caption模式下不加内部标题、刻度稀疏化)。
数据全部来自 results/ 的真实走查输出。
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/noto-serif-sc/NotoSerifSC-Regular.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Serif SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE = "/home/z/my-project/download/xauusd_ml_v2"
RES = os.path.join(BASE, "results")
CH = os.path.join(RES, "charts_pdf")
os.makedirs(CH, exist_ok=True)

# 配色 (palette.cascade 派生, 语义色为低饱和变体)
C_ACCENT = "#87702a"      # XS 主强调
C_ACCENT2 = "#3a95b4"     # XS 次强调
C_HEADER = "#504933"      # M
C_ICON = "#8c7e52"        # S
C_POS = "#46875c"         # 语义-正
C_NEG = "#92453e"         # 语义-负
C_MUT = "#78766f"         # 灰
VARIANTS = ["legacy", "v2", "fixed_legacy", "fixed_v2"]
VCOL = {"legacy": C_NEG, "v2": C_ACCENT2, "fixed_legacy": "#a18347", "fixed_v2": "#6f5b8e"}
VLABEL = {"legacy": "legacy23特征 + ATR障碍 (最优)", "v2": "v2特征(45个) + ATR障碍",
          "fixed_legacy": "legacy23特征 + 固定$3.5/$2.0", "fixed_v2": "v2特征 + 固定$3.5/$2.0"}


def style_ax(ax, grid_axis="both"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cfcab8")
    ax.spines["bottom"].set_color("#cfcab8")
    ax.tick_params(colors="#78766f", labelsize=8.5, length=3)
    ax.grid(axis=grid_axis, linestyle="--", linewidth=0.5, alpha=0.35, color="#78766f")
    ax.set_axisbelow(True)


def load_trades(fs):
    return pd.read_csv(os.path.join(RES, f"trades_{fs}.csv"), parse_dates=["exit_time"])


def main():
    # ---------- 图1: 权益曲线 ----------
    fig, ax = plt.subplots(figsize=(9.6, 5.2), constrained_layout=True)
    for fs in VARIANTS:
        t = load_trades(fs).sort_values("exit_time")
        ax.plot(t["exit_time"], t["pnl"].cumsum(), label=VLABEL[fs], color=VCOL[fs],
                lw=2.2 if fs == "legacy" else 1.6,
                solid_capstyle="round", zorder=3 if fs == "legacy" else 2)
    ax.axhline(0, color="#78766f", lw=0.7, alpha=0.6)
    ax.set_ylabel("累计PnL (美元 / 1盎司名义)", fontsize=10)
    style_ax(ax)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9, handlelength=1.6,
                    labelspacing=0.5, borderaxespad=0.2)
    fig.savefig(os.path.join(CH, "equity_curves.png"), dpi=200)
    plt.close(fig)

    # ---------- 图2: 最优组合逐月PnL + 累计 ----------
    t_leg = load_trades("legacy").sort_values("exit_time")
    mp = t_leg.set_index("exit_time")["pnl"].resample("ME").sum()
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 7.2), constrained_layout=True,
                             height_ratios=[1.15, 1])
    colors = [C_POS if v > 0 else C_NEG for v in mp.values]
    axes[0].bar(range(len(mp)), mp.values, color=colors, alpha=0.9, width=0.72)
    axes[0].set_xticks(range(0, len(mp), 2))
    axes[0].set_xticklabels([d.strftime("%y-%m") for d in mp.index][::2], fontsize=8.5)
    axes[0].set_ylabel("月净PnL (美元)", fontsize=10)
    style_ax(axes[0], grid_axis="y")
    axes[0].axhline(0, color="#78766f", lw=0.7)
    cum = mp.cumsum()
    axes[1].plot(cum.index, cum.values, color=C_NEG, lw=2.5, solid_capstyle="round")
    axes[1].fill_between(cum.index, cum.values, 0, color=C_NEG, alpha=0.12)
    axes[1].set_ylabel("累计PnL (美元)", fontsize=10)
    style_ax(axes[1])
    fig.savefig(os.path.join(CH, "monthly_pnl_winner.png"), dpi=200)
    plt.close(fig)

    # ---------- 图3: 逐月AUC + 胜率 ----------
    pf = pd.DataFrame(json.load(open(os.path.join(RES, "per_fold_legacy.json"))))
    fig, ax = plt.subplots(figsize=(9.6, 4.8), constrained_layout=True)
    x = np.arange(len(pf))
    ax.plot(x, pf["auc_oos_long"], color=C_ACCENT2, lw=2.2, marker="o", ms=3.5,
            label="OOS AUC 多头", zorder=3)
    ax.plot(x, pf["auc_oos_short"], color=C_NEG, lw=2.2, marker="o", ms=3.5,
            label="OOS AUC 空头", zorder=3)
    ax.axhline(0.5, color="#78766f", lw=0.8, ls=":", alpha=0.8)
    ax.set_ylabel("AUC", fontsize=10)
    ax.set_ylim(0.45, 0.62)
    style_ax(ax)
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    wr = pf["ml"].apply(lambda m: 100 * m["win_rate"] if m["trades"] else np.nan)
    ax2.bar(x, wr, alpha=0.22, color="#a18347", width=0.8, zorder=1)
    ax2.set_ylabel("月胜率 (%)", fontsize=10)
    ax2.set_ylim(0, 90)
    ax2.tick_params(colors="#78766f", labelsize=8.5)
    ax.set_xticks(x[::2])
    ax.set_xticklabels(pf["oos_month"][::2], rotation=0, fontsize=8.5)
    ax.legend(loc="upper left", frameon=False, fontsize=9, borderaxespad=0.2)
    fig.savefig(os.path.join(CH, "auc_winrate_trajectory.png"), dpi=200)
    plt.close(fig)

    # ---------- 图4: 金价与波动率背景 ----------
    with open(os.path.join(BASE, "cache", "prep_XAUUSDc_M1_202201022305_202606_5min.pkl"), "rb") as f:
        m5, m1_pack, spread_cost, monthly = pd.read_pickle(f)
    fig, ax = plt.subplots(figsize=(9.6, 4.4), constrained_layout=True)
    ax.plot(m5.index, m5["CLOSE"], lw=1.0, color="#a18347", label="XAUUSD 收盘价")
    ax.set_ylabel("价格 (美元)", fontsize=10)
    style_ax(ax)
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    roll_rng = (m5["HIGH"].rolling(288).max() - m5["LOW"].rolling(288).min())
    ax2.plot(m5.index, roll_rng, lw=1.0, color="#466a8e", alpha=0.9, label="24小时滚动波动区间")
    ax2.set_ylabel("24小时波动区间 (美元)", fontsize=10)
    ax2.tick_params(colors="#78766f", labelsize=8.5)
    ax.axvspan(pd.Timestamp("2024-08-01"), pd.Timestamp("2026-07-31"), color=C_NEG, alpha=0.06)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=9)
    fig.savefig(os.path.join(CH, "price_atr_context.png"), dpi=200)
    plt.close(fig)

    # ---------- 图5: 特征重要性 (末折模型, 横向条形图) ----------
    imp = json.load(open(os.path.join(RES, "feature_importance.json")))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.6), constrained_layout=True)
    for ax, d, cc in [(axes[0], "long", C_ACCENT2), (axes[1], "short", C_NEG)]:
        s = pd.Series(imp[d]).sort_values(ascending=True).tail(12)
        s = s / s.max() * 100
        ax.barh(range(len(s)), s.values, color=cc, alpha=0.88, height=0.62)
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels(s.index, fontsize=9)
        ax.set_xlabel("相对增益重要性 (%)", fontsize=10)
        style_ax(ax, grid_axis="x")
    fig.savefig(os.path.join(CH, "feature_importance.png"), dpi=200)
    plt.close(fig)

    print("PDF图表完成 ->", CH)
    for f in sorted(os.listdir(CH)):
        print("  ", f)


if __name__ == "__main__":
    main()
