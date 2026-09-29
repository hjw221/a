"""
结果分析: 汇总4个消融变体 + 基准, Bootstrap置信区间, 图表(中文), 报告表。
所有数字均来自真实走查运行, 无任何模拟/估算(点差敏感性除外, 已注明)。
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CFG
from data import load_data
from walkforward import build_folds, time_decay_weights
from models import fit_lgb, predict_ens
from backtest import metrics, bootstrap_ci

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results")
CH = os.path.join(RES, "charts")
os.makedirs(CH, exist_ok=True)

VARIANTS = ["legacy", "v2", "fixed_legacy", "fixed_v2"]
LABELS = {"legacy": "legacy23特征+ATR障碍", "v2": "v2特征+ATR障碍",
          "fixed_legacy": "legacy23特征+固定$3.5/$2.0", "fixed_v2": "v2特征+固定$3.5/$2.0"}
COLORS = {"legacy": "#C0392B", "v2": "#2471A3", "fixed_legacy": "#B7950B",
          "fixed_v2": "#7D3C98", "random_matched": "#7F8C8D"}


def load_trades(fs):
    fp = os.path.join(RES, f"trades_{fs}.csv")
    t = pd.read_csv(fp, parse_dates=["entry_time", "exit_time", "signal_time"])
    return t


def collect_baselines(fs):
    """从 checkpoint 聚集各折基准交易日志 (避免CSV被后续运行覆盖)。"""
    logs = {}
    for i in range(24):
        ck = os.path.join(RES, "checkpoints", f"wf_{fs}_fold{i:02d}.pkl")
        if not os.path.exists(ck):
            continue
        with open(ck, "rb") as f:
            d = pickle.load(f)
        for k, v in d.get("base_logs", {}).items():
            logs.setdefault(k, []).append(v)
    return {k: pd.concat(v, ignore_index=True) if v else None for k, v in logs.items()}


def main():
    m5, m1_pack, spread_cost, monthly = load_data(CFG)

    # ============ 1. 汇总对比 + Bootstrap ============
    rows = []
    for fs in VARIANTS:
        t = load_trades(fs)
        m = metrics(t, fs)
        ci_mean, ci_total = bootstrap_ci(t["pnl"].to_numpy(), CFG["bootstrap_iters"], CFG["random_seed"])
        m["label"] = LABELS[fs]
        m["ci_mean"] = ci_mean
        m["ci_total"] = ci_total
        m["trades_per_day"] = m["trades"] / 24 / 21.6
        # 多空拆分
        for d in ["long", "short"]:
            td = t[t["dir"] == d]
            m[f"{d}_trades"] = len(td)
            m[f"{d}_pnl"] = float(td["pnl"].sum())
        rows.append(m)
    comp = pd.DataFrame(rows)

    # 基准: 从checkpoint聚合 (ATR组与固定组各自口径; 同组内always/coin确定性一致)
    base_atr = collect_baselines("legacy")
    base_fixed = collect_baselines("fixed_legacy")
    base_rows = []
    for name in ["always_long", "always_short", "random_coin"]:
        if base_atr.get(name) is not None:
            bm = metrics(base_atr[name], name + "(ATR障碍)")
            bm["label"] = f"{name} (ATR障碍口径)"
            base_rows.append(bm)
    if base_fixed.get("always_long") is not None:
        for name in ["always_long", "always_short", "random_coin"]:
            bm = metrics(base_fixed[name], name + "(固定障碍)")
            bm["label"] = f"{name} (固定障碍口径)"
            base_rows.append(bm)
    rm = base_atr.get("random_matched")
    if rm is not None:
        bm = metrics(rm, "random_matched")
        bm["label"] = "随机同时机同频率(匹配最优组合)"
        base_rows.append(bm)

    # 买入持有(同期金价月收益)
    px = m5["CLOSE"].resample("ME").last().dropna()
    oos = px[(px.index >= "2024-08-01") & (px.index <= "2026-07-31")]
    bh_total = float((oos.iloc[-1] / oos.iloc[0] - 1) * 100)
    bh_monthly = oos.pct_change().dropna() * 100

    # ============ 2. 图1: 权益曲线 ============
    fig, ax = plt.subplots(figsize=(10.5, 6), constrained_layout=True)
    for fs in VARIANTS:
        t = load_trades(fs).sort_values("exit_time")
        ax.plot(t["exit_time"], t["pnl"].cumsum(), label=LABELS[fs], color=COLORS[fs], lw=1.8)
    fp = os.path.join(RES, "trades_baseline_random_matched.csv")
    if rm is not None:
        bt = rm.sort_values("exit_time")
        ax.plot(bt["exit_time"], bt["pnl"].cumsum(), label="随机同时机基准(同频率)", color=COLORS["random_matched"],
                lw=1.2, ls="--", alpha=0.8)
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_title("XAUUSD 走查OOS权益曲线 (2024-08 ~ 2026-07, 24个月, 单位: 美元/1盎司名义)", fontsize=13)
    ax.set_ylabel("累计PnL ($)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)
    fig.savefig(os.path.join(CH, "equity_curves.png"), dpi=150)
    plt.close(fig)

    # ============ 3. 图2: 最优组合逐月PnL ============
    t_leg = load_trades("legacy").sort_values("exit_time")
    mp = t_leg.set_index("exit_time")["pnl"].resample("ME").sum()
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), constrained_layout=True,
                             height_ratios=[1.2, 1])
    colors = ["#27AE60" if v > 0 else "#C0392B" for v in mp.values]
    axes[0].bar(range(len(mp)), mp.values, color=colors, alpha=0.85)
    axes[0].set_xticks(range(len(mp)))
    axes[0].set_xticklabels([d.strftime("%y-%m") for d in mp.index], rotation=60, fontsize=8)
    axes[0].set_title("最优组合(legacy23特征+ATR障碍) 逐月OOS净PnL", fontsize=12)
    axes[0].set_ylabel("月PnL ($)")
    axes[0].grid(alpha=0.25, axis="y")
    axes[0].axhline(0, color="k", lw=0.6)
    axes[1].plot(mp.index, mp.cumsum(), color="#C0392B", lw=2, marker="o", ms=3)
    axes[1].set_title("累计PnL曲线", fontsize=12)
    axes[1].set_ylabel("累计 ($)")
    axes[1].grid(alpha=0.25)
    fig.savefig(os.path.join(CH, "monthly_pnl_winner.png"), dpi=150)
    plt.close(fig)

    # ============ 4. 图3: 逐折AUC与胜率轨迹 ============
    pf = pd.DataFrame(json.load(open(os.path.join(RES, "per_fold_legacy.json"))))
    fig, ax = plt.subplots(figsize=(10.5, 5), constrained_layout=True)
    x = range(len(pf))
    ax.plot(x, pf["auc_oos_long"], marker=".", label="OOS AUC 多头", color="#2471A3")
    ax.plot(x, pf["auc_oos_short"], marker=".", label="OOS AUC 空头", color="#C0392B")
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax2 = ax.twinx()
    wr = pf["ml"].apply(lambda m: 100 * m["win_rate"] if m["trades"] else np.nan)
    ax2.bar(x, wr, alpha=0.18, color="#F39C12", label="月胜率%")
    ax2.set_ylabel("胜率 (%)")
    ax2.set_ylim(0, 80)
    ax.set_xticks(x)
    ax.set_xticklabels(pf["oos_month"], rotation=60, fontsize=8)
    ax.set_ylabel("AUC")
    ax.set_title("最优组合: 逐月OOS AUC 与 胜率 (真实走查)", fontsize=12)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper right")
    ax.grid(alpha=0.25)
    fig.savefig(os.path.join(CH, "auc_winrate_trajectory.png"), dpi=150)
    plt.close(fig)

    # ============ 5. 图4: 金价与ATR背景 ============
    atr = m5["CLOSE"].rolling(288).std() * np.sqrt(288)  # 近似
    fig, ax = plt.subplots(figsize=(10.5, 4.5), constrained_layout=True)
    ax.plot(m5.index, m5["CLOSE"], lw=0.8, color="#B7950B", label="XAUUSD收盘价")
    ax.set_ylabel("价格 ($)")
    ax2 = ax.twinx()
    roll_rng = (m5["HIGH"].rolling(288).max() - m5["LOW"].rolling(288).min())
    ax2.plot(m5.index, roll_rng, lw=0.6, color="#5D6D7E", alpha=0.8, label="24h滚动波动区间($)")
    ax2.set_ylabel("24h波动区间 ($)")
    ax.axvspan(pd.Timestamp("2024-08-01"), pd.Timestamp("2026-07-31"), color="#C0392B", alpha=0.07)
    ax.text(pd.Timestamp("2024-09-01"), m5["CLOSE"].max() * 0.98, "OOS验证区", color="#C0392B", fontsize=10)
    ax.set_title("背景: 金价2022-2026翻倍, 波动率放大~4倍 (固定$障碍失效, ATR障碍自适应)", fontsize=12)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)
    fig.savefig(os.path.join(CH, "price_atr_context.png"), dpi=150)
    plt.close(fig)

    # ============ 6. 特征重要性 (重训末折最优组合模型, 真实增益) ============
    print("[analyze] 重训末折模型获取特征重要性...")
    folds = build_folds(m5, CFG)
    f23 = folds[-1]
    pack = pickle.load(open(os.path.join(BASE, "cache", "pack_legacy.pkl"), "rb"))
    F, lab = pack["F"], pack["lab"]
    tuned = json.load(open(os.path.join(RES, "tuned_params.json")))["legacy"]
    fit_r = f23["fit_rows"]
    w = time_decay_weights(m5.index, fit_r, CFG["sample_halflife_days"])
    Xf = F.iloc[fit_r].to_numpy(np.float32)
    imp_all = {}
    for direction in ["long", "short"]:
        y = (lab["out_long" if direction == "long" else "out_short"].to_numpy() == 1).astype(np.int8)
        m_lgb, _ = fit_lgb(Xf, y[fit_r], w, tuned[direction]["lgb"],
                           es_rounds=CFG["early_stopping_rounds"], max_rounds=CFG["max_rounds"])
        imp = pd.Series(m_lgb.feature_importance("gain"), index=pack["feats"])
        imp_all[direction] = imp.sort_values(ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5), constrained_layout=True)
    for ax, direction in zip(axes, ["long", "short"]):
        top = imp_all[direction].head(15)[::-1]
        ax.barh(range(len(top)), top.values, color="#2471A3" if direction == "long" else "#C0392B", alpha=0.85)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index, fontsize=9)
        ax.set_title(f"特征增益重要性 前15 ({direction}, 末折模型)", fontsize=11)
        ax.grid(alpha=0.25, axis="x")
    fig.savefig(os.path.join(CH, "feature_importance.png"), dpi=150)
    plt.close(fig)
    json.dump({d: {k: float(v) for k, v in imp_all[d].head(20).items()} for d in imp_all},
              open(os.path.join(RES, "feature_importance.json"), "w"), indent=2)

    # ============ 7. 点差敏感性(近似: point=0.001->0.01, 每笔额外9×月度中位点数×0.001美元) ============
    sens = {}
    for fs in VARIANTS:
        t = load_trades(fs)
        t_m = t["exit_time"].dt.to_period("M")
        extra = t_m.map(lambda p: 9.0 * float(monthly.get(p, 200)) * CFG["point_value"])
        sens[LABELS[fs]] = float(t["pnl"].sum() - extra.sum())

    # ============ 8. 输出汇总JSON + 控制台 ============
    out = {
        "comparison": comp.to_dict(orient="records"),
        "baselines": base_rows,
        "buy_hold_pct": bh_total,
        "spread_sensitivity_point001_total_pnl": sens,
        "winner_monthly_pnl": {str(k): float(v) for k, v in mp.items()},
        "winner_folds": pf.to_dict(orient="records"),
    }
    json.dump(out, open(os.path.join(RES, "analysis.json"), "w"), indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 80)
    print("四组消融对比 (真实OOS 2024-08~2026-07, 含点差成本):")
    show = comp[["label", "trades", "win_rate", "total_pnl", "profit_factor", "sharpe",
                 "max_dd", "long_pnl", "short_pnl"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:,.3f}" if abs(v) < 10 else f"{v:,.1f}"))
    print("\nBootstrap 95% CI (单笔均值 / 总额):")
    for r in rows:
        print(f"  {r['label']}: [{r['ci_mean'][0]:.4f}, {r['ci_mean'][1]:.4f}] / "
              f"[{r['ci_total'][0]:,.0f}, {r['ci_total'][1]:,.0f}]")
    print(f"\n基准: " + " | ".join(f"{b['label']}: {b['total_pnl']:,.0f}$" for b in base_rows))
    print(f"买入持有(同期): {bh_total:+.1f}%")
    print(f"\n点值敏感性(point=0.01时总额): " + "\n  ".join(f"{k}: {v:,.0f}$" for k, v in sens.items()))
    print(f"\n图表 -> {CH}")
    return out


if __name__ == "__main__":
    main()
