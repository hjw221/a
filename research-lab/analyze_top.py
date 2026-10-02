#!/usr/bin/env python3
"""analyze_top.py — 领跑配置逐月结构分析"""
import json, os
os.environ["ML_THREADS"] = "24"
os.chdir("/root/rivermind-data/research-lab")
import stage_c_loop as sc

recs = []
for w in range(4):
    try:
        recs += [json.loads(l) for l in open(f"results_research/worker{w}_results.jsonl")]
    except Exception:
        pass
top = [r for r in recs if r["cfg_id"] == "fd22bb26ce"]
cfg = top[0]["cfg"]
print("cfg:", {k: v for k, v in cfg.items() if k not in ("features", "lgb")}, "| feats:", len(cfg["features"]))
m5, F, *_ = sc.load_all()
sc.ALL_FEATS = [c for c in F.columns if F[c].notna().mean() > 0.95]
r = sc.walkforward(cfg, "2022-08", "2026-07", step=1)
years = {}
for f in r["folds"]:
    y = f["month"][:4]
    years.setdefault(y, {"pnl": 0, "trades": 0})
    years[y]["pnl"] += f["pnl"]; years[y]["trades"] += f["trades"]
for y in sorted(years):
    v = years[y]
    print(f"{y}: PnL {v['pnl']:8.0f} | {v['trades']:6d}笔 | 笔均 {v['pnl']/max(v['trades'],1):.3f}")
print("TOTAL:", round(r["pnl"]), r["trades"], "笔")
# 逐月明细尾部
for f in r["folds"][-6:]:
    print(f"  {f['month']}: {f['pnl']:8.1f} ({f['trades']}笔)")
