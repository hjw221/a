#!/usr/bin/env python3
"""v14_weaksearch.py — Arm 1: 弱年份目标搜索 (补 2022-2025 alpha)

协议与 master_research.py 完全一致 (搜索=全窗 step=3 稀疏折同分布, 决赛=54折全窗),
唯一区别: 选择目标从 [总PnL] 改为 [2023+2024 折 PnL].
动机: 年度集中度审计显示 v13 的 82% 集中在 2026; 要在保住总量>732 的同时压
2026 占比, 数学上必须把 2022-2025 (尤其 2023/2024) 的 alpha 提升 ~3 倍.

worker 模式: 由 master 经 subprocess 拉起 (WORKER_ID 环境变量)
master 模式: python3 v14_weaksearch.py --master
冠军标准: 决赛(54折) 2023>0 且 2024>0 且 total>0, 按 min(2023,2024) 最大选.
"""
import os, sys, json, time, random, subprocess
import numpy as np
import stage_c_loop as sc

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results_weak")
os.makedirs(RES, exist_ok=True)
N_WORKERS = int(os.environ.get("N_WORKERS", "4"))
N_ROUNDS = int(os.environ.get("N_ROUNDS", "12"))
REPEATS = int(os.environ.get("REPEATS", "2"))
WEAK_YEARS = ("2023", "2024")

def say(msg):
    sc.say(msg)

def weak_score(folds):
    return float(sum(f["pnl"] for f in folds if str(f["month"])[:4] in WEAK_YEARS))

def weak_pos_ratio(folds):
    wf = [f["pnl"] for f in folds if str(f["month"])[:4] in WEAK_YEARS]
    return float(np.mean([p > 0 for p in wf])) if wf else 0.0

# ---------------- worker ----------------
def load_top_weak():
    tops = []
    for fn in os.listdir(RES):
        if fn.endswith("_results.jsonl"):
            try:
                with open(os.path.join(RES, fn)) as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            if rec.get("weak2324", -1e9) > 0 and rec.get("pnl", -1e9) > -200:
                                tops.append(rec)
                        except Exception:
                            pass
            except Exception:
                pass
    tops.sort(key=lambda r: -r["weak2324"])
    return [t["cfg"] for t in tops[:5]]

def worker_main():
    wid = int(os.environ.get("WORKER_ID", "0"))
    seed = int(os.environ.get("SEED", "7777")) + wid * 1000
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)
    m5, F, *_ = sc.load_all()
    sc.ALL_FEATS = [c for c in F.columns if F[c].notna().mean() > 0.95]
    print(f"[W{wid}] v14 弱年份搜索 worker | 因子池 {len(sc.ALL_FEATS)} | {N_ROUNDS}轮", flush=True)
    logf = os.path.join(RES, f"worker{wid}_results.jsonl")
    best = {"weak2324": -1e9}
    for rd in range(N_ROUNDS):
        t0 = time.time()
        if rd < 4 or pyrng.random() < 0.35:
            cfg = sc.sample_cfg(rng)
        else:
            top = load_top_weak()
            cfg = sc.mutate(pyrng.choice(top), rng) if top else sc.sample_cfg(rng)
        r = sc.walkforward(cfg, "2022-08", "2026-07", step=3)
        wk = weak_score(r.get("folds", []))
        rec = {"worker": wid, "round": rd, "cfg_id": sc.cfg_id(cfg),
               "weak2324": wk, "weak_pos_ratio": weak_pos_ratio(r.get("folds", [])),
               "pnl": r["pnl"], "trades": r["trades"], "sharpe": r["sharpe"],
               "plr": r["plr"], "pos_fold_ratio": r["pos_fold_ratio"],
               "cfg": cfg, "dt": time.strftime("%F %T")}
        with open(logf, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if wk > best["weak2324"]:
            best = rec
        print(f"[W{wid}] R{rd} {rec['cfg_id']} weak2324 {wk:.0f} | total {r['pnl']:.0f} "
              f"({r['trades']}笔) [{time.time()-t0:.0f}s]", flush=True)
    with open(os.path.join(RES, f"worker{wid}_best.json"), "w") as f:
        json.dump(best, f)
    say(f"v14弱年 W{wid} 完成: best weak2324 {best['weak2324']:.0f} / total {best.get('pnl', 0):.0f}")

# ---------------- master ----------------
def run_workers(batch):
    procs = []
    for w in range(N_WORKERS):
        e = dict(os.environ)
        e.update(WORKER_ID=str(w), N_ROUNDS=str(N_ROUNDS), ML_THREADS="7",
                 SEED=str(7777 + batch * 137))
        log = open(os.path.join(RES, f"worker{w}_batch{batch}.out"), "a")
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__)], cwd=BASE,
                             env=e, stdout=log, stderr=log)
        procs.append(p)
    for p in procs:
        p.wait()

def collect():
    recs = []
    for fn in os.listdir(RES):
        if fn.endswith("_results.jsonl"):
            with open(os.path.join(RES, fn)) as f:
                for line in f:
                    try:
                        recs.append(json.loads(line))
                    except Exception:
                        pass
    return recs

def by_year(folds):
    d = {}
    for f in folds:
        y = int(str(f["month"])[:4])
        d[y] = d.get(y, 0.0) + float(f["pnl"])
    return {int(k): round(v, 1) for k, v in sorted(d.items())}

def master_main():
    sc.ALL_FEATS = None
    say(f"v14 弱年份搜索启动: {N_WORKERS}x{N_ROUNDS} x {REPEATS}批, 目标=2023+2024折PnL, 决赛54折")
    champion = None
    for batch in range(REPEATS):
        t0 = time.time()
        say(f"v14 weak batch {batch+1}/{REPEATS} 搜索中 ({N_WORKERS}x{N_ROUNDS})")
        run_workers(batch)
        recs = collect()
        ok = [r for r in recs if r.get("weak2324", -1e9) > 50
              and r.get("weak_pos_ratio", 0) >= 0.5
              and r.get("pnl", -1e9) > -200]
        ok.sort(key=lambda r: -r["weak2324"])
        seen, cands = set(), []
        for r in ok:
            if r["cfg_id"] not in seen:
                seen.add(r["cfg_id"]); cands.append(r)
            if len(cands) >= 3:
                break
        print(f"[master] batch {batch}: {len(recs)} 结果, {len(cands)} 弱年份候选 "
              f"({time.time()-t0:.0f}s)", flush=True)
        say(f"v14 weak batch{batch+1}: {len(recs)}配置 {len(cands)}候选")
        if not cands:
            continue
        best_final = None
        for c in cands:
            fr = sc.final_eval(c["cfg"])
            by = by_year(fr.get("folds", []))
            print(f"[master]   决赛 {c['cfg_id']}: 搜索weak2324 {c['weak2324']:.0f} -> "
                  f"决赛 total {fr['pnl']:.0f} | 逐年 {by}", flush=True)
            say(f"v14决赛 {c['cfg_id'][:6]}: total {fr['pnl']:.0f} | 23:{by.get(2023,0):.0f} "
                f"24:{by.get(2024,0):.0f} 26:{by.get(2026,0):.0f}")
            with open(os.path.join(RES, "finals_weak.jsonl"), "a") as f:
                f.write(json.dumps({"batch": batch, "cfg_id": c["cfg_id"], "cfg": c["cfg"],
                                    "search": {k: c.get(k) for k in ("weak2324", "weak_pos_ratio",
                                                                     "pnl", "trades")},
                                    "final": {k: fr.get(k) for k in ("pnl", "trades", "sharpe",
                                                                     "plr", "win_rate")},
                                    "by_year": by,
                                    "final_folds": fr.get("folds", [])}) + "\n")
            key = min(by.get(2023, -1e9), by.get(2024, -1e9))
            if best_final is None or key > best_final[0]:
                best_final = (key, c, fr, by)
        key, c, fr, by = best_final
        if by.get(2023, -1e9) > 0 and by.get(2024, -1e9) > 0 and fr["pnl"] > 0:
            champion = {"cfg": c["cfg"], "final": {k: fr.get(k) for k in ("pnl", "trades", "sharpe",
                                                                          "plr", "win_rate")},
                        "by_year": by, "final_folds": fr.get("folds", []),
                        "search_weak2324": c["weak2324"]}
            with open(os.path.join(RES, "weak_champion.json"), "w") as f:
                json.dump(champion, f, indent=2)
            say(f"🏆 v14弱年份book达成: 23:{by[2023]:.0f} 24:{by[2024]:.0f} total {fr['pnl']:.0f}")
            print("[master] WEAK CHAMPION!", flush=True)
            break
        else:
            say(f"v14弱年batch{batch+1}: 最优 23:{by.get(2023,0):.0f} 24:{by.get(2024,0):.0f} 未双正, 继续")
    if champion is None:
        say("v14弱年份预算耗尽, 未找到 2023/24 双正 book (结果见 finals_weak.jsonl)")
        print("[master] no weak champion", flush=True)

def main():
    if "--master" in sys.argv or os.environ.get("V14_ROLE") == "master":
        master_main()
    else:
        worker_main()

if __name__ == "__main__":
    main()
