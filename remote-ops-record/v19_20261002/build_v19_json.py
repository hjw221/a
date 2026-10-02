#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_v19_json.py — 合成第一轮+第二轮结果 → public/data/v19.json (前端数据包)."""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
R1J = json.load(open(os.path.join(BASE, "results_v19", "v19_r1gate.json")))
R2J = json.load(open(os.path.join(BASE, "results_v19", "v19b_round2.json")))
OUT = os.path.join(os.path.dirname(BASE), "public", "data", "v19.json")

YEARS = ["2022", "2023", "2024", "2025", "2026"]


def slim_summ(s):
    """瘦身 summary: 前端所需字段."""
    return dict(total=s.get("total", 0), trades=s.get("trades", 0),
                avg=s.get("avg", 0), t_stat=s.get("t_stat", 0),
                win_rate=s.get("win_rate", 0), maxdd=s.get("maxdd", 0),
                share2026=s.get("share2026", 0), by_year=s.get("by_year", {}),
                n_by_year=s.get("n_by_year", {}), months=s.get("months", {}))


def arm_row(rec, label, kind):
    s = rec["pess03"]
    return dict(name=rec["name"], label=label, kind=kind,
                total=s["total"], trades=s["trades"], avg=s["avg"],
                t_stat=s["t_stat"], win_rate=s["win_rate"], maxdd=s["maxdd"],
                share2026=s.get("share2026", 0), by_year=s["by_year"],
                n_by_year=s.get("n_by_year", {}), all_pos=rec.get("all_pos", False),
                coverage=rec.get("coverage"), months=s.get("months", {}))


# ---------------- 第一轮 ----------------
base = R1J["arms"]["base_atrmed"]["pess03"]
none_s = R1J["arms"]["base_none"]["pess03"]
r1_grid = [arm_row(r, f"θ_on={r['theta_on']} · 锁{r['min_on']}根", "r1g") for r in R1J["grid"]]
union = R1J["arms"]["union"]["pess03"]
inter = R1J["arms"]["intersect"]["pess03"]
best_r1 = R1J["arms"]["best_r1"]
rb = R1J["route_b"]

round1 = dict(
    oracle=dict(ic_var=R1J["oracle"]["ic_var"], ic_skew=R1J["oracle"]["ic_skew"],
                ic_var_by_year=R1J["oracle"]["ic_var_by_year"],
                r_quantiles=R1J["oracle"]["r_quantiles"]),
    repro=R1J["baseline_repro"],
    base=slim_summ(base), base_none=slim_summ(none_s),
    base_coverage=R1J["arms"]["base_atrmed"]["coverage"],
    grid=r1_grid,
    union=slim_summ(union), union_coverage=R1J["arms"]["union"]["coverage"],
    intersect=slim_summ(inter), intersect_coverage=R1J["arms"]["intersect"]["coverage"],
    best_r1=dict(name=best_r1["name"], **slim_summ(best_r1["pess03"]),
                 coverage=best_r1["coverage"]),
    overlap=R1J["overlap"],
    route_b=dict(**slim_summ(rb["pess03"]), side_total=rb["side_cost"]["total"],
                 freq_per_week=rb["freq_per_week"]),
    oracle_ms_per_bar=R1J["meta"]["oracle_ms_per_bar"],
    oracle_wall_s=R1J["meta"]["oracle_wall_s"],
)

# ---------------- 第二轮 ----------------
h1 = [arm_row(r, f"θ_on={r['theta_on']} · 锁{r['min_on']}根", "h16") for r in R2J["h1_grid"]]
h1L = [arm_row(r, f"θ_on={r['theta_on']} · 锁{r['min_on']}根", "h16L") for r in R2J["h1_long"]]
h2 = [arm_row(r, f"θ_on={r['theta_on']} · 锁{r['min_on']}根", "nc") for r in R2J["h2_grid"]]
h2L = [arm_row(r, f"θ_on={r['theta_on']} · 锁{r['min_on']}根", "ncL") for r in R2J["h2_long"]]
nc_best = max(h2, key=lambda r: r["total"])
h16_best = max(h1, key=lambda r: r["total"])

round2 = dict(
    nature=R2J["meta"]["round"],
    h1_grid=h1, h1_long=h1L, h2_grid=h2, h2_long=h2L,
    intersect_h16=slim_summ(R2J["intersect"]["h16"]["pess03"]),
    intersect_h16_with=R2J["intersect"]["h16"]["with_gate"],
    intersect_nc=slim_summ(R2J["intersect"]["nc"]["pess03"]),
    intersect_nc_with=R2J["intersect"]["nc"]["with_gate"],
    nc_best=nc_best, h16_best=h16_best,
    h3=R2J["h3_entry_cond"],
)

# ---------------- 权益曲线 (月度累计) ----------------
def cum_months(months):
    ks = sorted(months.keys())
    out, acc = [], 0.0
    for k in ks:
        acc += months[k]
        out.append([k, round(acc, 1)])
    return out

equity = dict(
    base=cum_months(base["months"]),
    best_r1=cum_months(best_r1["pess03"]["months"]),
    nc_best=cum_months(nc_best["months"]) if nc_best.get("months") else [],
    route_b=cum_months(rb["pess03"]["months"]),
)

# ---------------- 终局判定 ----------------
v1 = R1J["verdict"]
v2 = R2J["verdict2"]
verdict = dict(
    route_a=dict(
        ok=False,
        pre_registered_rule="R1闸门臂 pess03 > $2,980.2 ∧ 逐年全正 ∧ 笔数≥100",
        best_arm=v1["best_arm"], best_total=v1["best_total"], delta=v1["delta"],
        union_total=v1["union_total"],
        tested_variants=6 + 2,  # 第一轮网格 + union/intersect
    ),
    route_b=dict(
        ok=False, total=v1["route_b_total"], side_total=v1["route_b_side_total"],
        freq_per_week=v1["route_b_freq_per_week"],
        expected_freq="3–5笔/周 (用户预期)", actual_freq=f"{v1['route_b_freq_per_week']}笔/周",
        t_stat=rb["pess03"]["t_stat"], by_year=rb["pess03"]["by_year"],
    ),
    round2=dict(
        h1="4h(用户规格上限)前瞻闸门 10/10 无臂超基线 → 视界假设否定",
        h2=("无学习 nowcast $2,816 > 全部学习前瞻闸门($1,437~$2,448) → "
            "先见在该尺度为负增量; 但仍 < 基线 $2,980"),
        h3="入场时点 R̂ 与单笔 PnL 相关 0.05~0.06 ≈ 0 → 无法按预测能量选交易",
    ),
    root_cause=(
        "时间常数错位: 冠军边际生活在 63 天 regime 尺度(ATR 中位=慢持续状态变量), "
        "而 oracle 可预测视界 ≤ 24h(IC_var 在 24h 归零)。1h/4h 能量先见无论学习与否"
        "都无法替代慢 regime 水平; 学习先见反而引入追逐瞬态的噪声(不敌持续性 nowcast)。"
    ),
    champion_survives="v17 冠军在 2 轮 24 个替换/组合变体下全部存活, 未被任何 ML 闸门击败",
    where_r1_belongs=[
        "执行/择时层 (R1 报告 P0): 原生 1h 视界的入场时机微调与成本控制",
        "多资产慢 regime 扩展: 需先获取 DXY/白银/US10Y 数据 (v18 已披露数据缺口)",
        "储层+RLS 引擎本身健康: IC 逐年稳定 0.25~0.42, 0.98ms/bar, 是可复用的在线学习基建",
    ],
)

doc = dict(
    meta=dict(
        generated=R1J["meta"]["generated"], engine="v19 · R1能量神谕 × v17规则冠军",
        branch="reservoir-engine",
        decision=(
            "路线A为主干(R1能量神谕→v17闸门) + 路线B三大机制(施密特/迟滞/时锁)移植到闸门状态机 "
            "+ 路线B本体作为同框架对照臂 — 用数据而非观点回答 A vs B"
        ),
        decision_reasons=[
            "风险不对称: v17 冠军边际已被证明($2,980/159笔/t=3.13), 闸门融合最坏情况有界; 独立方向模型最坏情况是负期望",
            "任务难度不对称: 预测方差状态(IC 0.343)远易于预测方向(IC 0.020, 毛利$436/4.5y)",
            "v17 消融已预演闸门机制: 规则版 ATR 闸门 +$1,320 且救活 2022/2026 — 机制本身有效, 问题只在替换物",
        ],
        prereg=R1J["meta"]["prereg"],
        cost=R1J["meta"]["cost"],
        oracle_ms_per_bar=R1J["meta"]["oracle_ms_per_bar"],
    ),
    round1=round1, round2=round2, equity=equity, verdict=verdict,
)

with open(OUT, "w") as f:
    json.dump(doc, f, separators=(",", ":"), default=str)
print(f"DONE -> {OUT} ({os.path.getsize(OUT)} bytes)")
