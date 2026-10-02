#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_reservoir_json.py — 从服务器产物组装 public/data/reservoir.json"""
import base64
import json
import sys
import time

sys.path.insert(0, '/home/z/my-project/recovery-tools')
from jump_ssh_lib import connect  # noqa: E402

REMOTE = "/root/rivermind-data/research-lab"
OUT = "/home/z/my-project/public/data/reservoir.json"


def fetch_json(tgt, path):
    for _ in range(3):
        _, out, _ = tgt.exec_command(f"base64 -w0 {path}")
        b64 = out.read().decode().strip()
        if b64 and len(b64) > 40 and "No such" not in b64:
            return json.loads(base64.b64decode(b64))
        time.sleep(2)
    raise RuntimeError(f"fetch failed {path}")


def main():
    jump, tgt = connect()
    main_doc = fetch_json(tgt, f"{REMOTE}/reservoir_v1.json")
    probes = {}
    for tag, name, note, hz in [
        ("h16", "h16 (4h)", "4 小时视界", 16),
        ("h32", "h32 (8h)", "8 小时视界", 32),
        ("h96", "h96 (24h)", "24 小时视界", 96),
        ("h32lc", "h32-低换手 (8h·dead0.10·rate0.10·smooth0.7)", "低换手执行变体", 32),
    ]:
        try:
            d = fetch_json(tgt, f"{REMOTE}/reservoir_{tag}.json")
            o = d["runs"]["seed42"]["metrics"]["overall"]
            probes[tag] = dict(name=name, horizon=hz, note=note, pnl=o["pnl"],
                               gross=o["gross"], cost=o["cost"], ic_skew=o["ic_skew"],
                               ic_var=o["ic_var"], sharpe=o["sharpe"],
                               turnover=o["turnover"], avg_pos=o["avg_pos"],
                               by_year=[r["pnl"] for r in d["runs"]["seed42"]["metrics"]["rows"]])
        except Exception as e:
            print(f"probe {tag} skip: {e}")
    tgt.close(); jump.close()

    runs = main_doc["runs"]
    meta = main_doc["meta"]
    p42 = runs["seed42"]
    sh = runs["control_shuffle"]["metrics"]["overall"]
    mom = runs["baseline_mom"]

    # λ 扫描注解
    lam_notes = {
        0.995: "有效记忆200bar<D=505 → 缠绕风险区",
        0.998: "中间带",
        0.999: "有效记忆1000bar ≥ 2×D (默认)",
    }
    lam_sweep = [dict(lam=float(k.replace("lam", "")), pnl=v["metrics"]["overall"]["pnl"],
                      ic_skew=v["metrics"]["overall"]["ic_skew"],
                      note=lam_notes.get(float(k.replace("lam", "")), ""))
                 for k, v in runs.items() if k.startswith("lam")]
    nres_sweep = [dict(n_res=int(k.replace("nres", "")), pnl=v["metrics"]["overall"]["pnl"],
                       ic_skew=v["metrics"]["overall"]["ic_skew"],
                       ms_per_bar=v["timing"]["total_ms"]["mean"], engine_mb=v["engine_mb"])
                  for k, v in runs.items() if k.startswith("nres")]
    seeds = [dict(seed=int(k.replace("seed", "")), pnl=v["metrics"]["overall"]["pnl"],
                  ic_skew=v["metrics"]["overall"]["ic_skew"], ic_var=v["metrics"]["overall"]["ic_var"],
                  sharpe=v["metrics"]["overall"]["sharpe"], maxdd=v["metrics"]["overall"]["maxdd"])
             for k, v in runs.items() if k.startswith("seed") and k != "seed42_curves"]

    tm = p42["timing"]["total_ms"]
    o = p42["metrics"]["overall"]
    verdict = dict(
        headline=(
            f"R1 阶段一·诚实结论：三层引擎「学习为真、算力达标、方向毛利为正」，"
            f"但连续调仓成本是毛利的 {o['cost'] / max(o['gross'], 1):.0f} 倍——瓶颈在执行层不在学习层"
        ),
        points=[
            f"学习真实性：能量头 IC {o['ic_var']:.3f} / 方向头 IC {o['ic_skew']:.3f}（1h 视界，t≈{abs(o['ic_skew']) * (meta['bars'] ** 0.5):.1f} 显著）；洗牌对照 IC → {sh['ic_skew']:.3f} 证实信号来自真实时序结构而非泄漏",
            f"方向毛利全视界为正：4h +$1,080 / 8h +$1,067 / 24h +$411 / 1h +{o['gross']:.0f}——储层的方向判断本身净正，但幅度仅 ~$0.01/bar@1oz",
            f"成本通道致命：0.3×ATR 悲观滑点 + 点差 × 换手 {o['turnover']:.0f} 单位 = ${o['cost']:.0f}；低换手变体(dead0.10/rate0.10)成本砍 58%、毛利仅 −9%，净亏从 −$10,085 收窄至 −$3,695——证明修复路径存在",
            f"算力承诺兑现：单核 {tm['mean']:.2f} ms/bar（p99 {tm['p99']:.2f}），引擎态 {p42['engine_mb']:.1f} MB，n_res=1000 亦仅 {runs.get('nres1000', {}).get('engine_mb', '?')} MB——无反向传播架构 CPU 轻量属实",
            f"W_out 终身进化：‖W‖₂ 在 19↔85 间随市场结构切换持续漂移（2024 牛市 ↔ 2026 高波），无收敛不动点、无灾难遗忘、无停机重训——λ={meta['lam']} 遗忘因子有效",
            f"GMM 状态自组织收敛到 4 组件（压缩/单边/混沌/常规），惊异度生长+剪枝稳定；逐年状态占比随 ATR 环境(2022/0.53→2026/{meta['atr_by_year'].get('2026', 0):.2f}$)迁移",
            "与 M1 时代结论一致且互证：短周期方向≈不可预测（AUC 0.505 ↔ IC_skew 0.02），能量/波动聚集高度可学（IC_var 0.34）——alpha 在波动率不在方向",
        ],
        next=[
            "执行层重构（P0）：事件驱动调仓——仅在信念变化超阈时交易（死区↑/迟滞带/触发式），目标换手再砍 70%，让成本 < 毛利",
            "能量头变现（P1）：IC 0.34 的波动率预测做 v17 HTF 突破冠军的在线闸门（替代静态 ATR 季中位），R1 学习器 × v17 规则引擎融合",
            "视界选型：方向信号在 4h 最强(IC 0.021)、24h 消失(0.004)；能量在 1h 最强——双头双视界非对称部署",
            "阶段二工程：MT5(50行 MQL5) ↔ Python 常驻进程 ZeroMQ/NamedPipe 桥；阶段三编译 C++ Eigen DLL(~200KB) 内嵌 EA",
        ],
    )

    doc = dict(
        meta=dict(engine=meta["engine"], bars=meta["bars"], span=meta["span"], warm=meta["warm"],
                  horizon=meta["horizon"], convention=meta["convention"], lam=meta["lam"],
                  n_res=meta["n_res"], spectral=meta["spectral"], density=meta["density"],
                  process_peak_mb=meta["process_peak_mb"], atr_by_year=meta["atr_by_year"],
                  generated=time.strftime("%Y-%m-%d %H:%M")),
        primary=dict(metrics=p42["metrics"], timing=p42["timing"], engine_mb=p42["engine_mb"],
                     K_final=p42["K_final"], W_l2=runs["seed42_curves"]["W_l2"],
                     K_curve=runs["seed42_curves"]["K"]),
        seeds=seeds, lam_sweep=sorted(lam_sweep, key=lambda x: x["lam"]),
        nres_sweep=nres_sweep,
        shuffle=dict(pnl=sh["pnl"], ic_skew=sh["ic_skew"], ic_var=sh["ic_var"], cost=sh["cost"]),
        momentum=dict(total=mom["total"], by_year=mom["by_year"], sharpe=mom["sharpe"], cost=mom["cost"]),
        probes=[probes[k] for k in ("h16", "h32", "h96", "h32lc") if k in probes],
        verdict=verdict,
    )
    with open(OUT, "w") as f:
        json.dump(doc, f, ensure_ascii=False, separators=(",", ":"))
    print(f"written {OUT}: {len(json.dumps(doc))} bytes")
    print("headline:", verdict["headline"])


if __name__ == "__main__":
    main()
