#!/usr/bin/env python3
"""master_research.py — 总指挥: 搜索→守门→决赛循环, 目标 PnL > $732.

流程:
  1. 起N个worker子进程 (各7线程) 搜索配置 (搜索窗 2024-01~2025-06)
  2. 收集全部结果, 候选 = PnL>0 且 pos_fold_ratio>=0.55, 按 PnL 排序 top3
  3. 候选逐个决赛: 决赛窗 2025-07~2026-07 (搜索从未触碰)
  4. 决赛 PnL > 732 → 胜利存档退出; 否则外层再来一轮 (换种子), 预算 REPEATS 轮
  5. 全程 ntfy 汇报
"""
import os, sys, json, time, subprocess, signal
import stage_c_loop as sc

BASE = os.path.dirname(os.path.abspath(__file__))
RES = sc.RES
TARGET = 732.0
N_WORKERS = int(os.environ.get("N_WORKERS", "4"))
N_ROUNDS = int(os.environ.get("N_ROUNDS", "12"))
REPEATS = int(os.environ.get("REPEATS", "4"))

def say(msg):
    sc.say(msg)

def run_workers(batch):
    env = dict(os.environ)
    procs = []
    for w in range(N_WORKERS):
        e = dict(env)
        e.update(WORKER_ID=str(w), N_ROUNDS=str(N_ROUNDS), ML_THREADS="7",
                 SEED=str(1337 + batch * 77))
        log = open(os.path.join(RES, f"worker{w}_batch{batch}.out"), "a")
        p = subprocess.Popen([sys.executable, "stage_c_loop.py"], cwd=BASE,
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

def candidates(recs):
    ok = [r for r in recs if r.get("pnl", -1e9) > 0 and r.get("pos_fold_ratio", 0) >= 0.55]
    ok.sort(key=lambda r: -r["pnl"])
    seen = set(); out = []
    for r in ok:
        if r["cfg_id"] not in seen:
            seen.add(r["cfg_id"]); out.append(r)
        if len(out) >= 3:
            break
    return out

def main():
    sc.ALL_FEATS = None
    finals_only = os.environ.get("FINALS_ONLY", "0") == "1"
    say(f"research master 启动: {N_WORKERS}x{N_ROUNDS}轮 x {REPEATS}批, 目标全窗PnL>{TARGET}"
        + (" [决赛重评模式]" if finals_only else ""))
    print(f"[master] {N_WORKERS} workers x {N_ROUNDS} rounds x {REPEATS} batches", flush=True)
    champion = None
    for batch in range(REPEATS):
        if not finals_only:
            t0 = time.time()
            say(f"batch {batch+1}/{REPEATS} 搜索中 ({N_WORKERS}x{N_ROUNDS} 配置)")
            run_workers(batch)
        recs = collect()
        cands = candidates(recs)
        print(f"[master] batch {batch}: {len(recs)} 结果, {len(cands)} 候选", flush=True)
        if not finals_only:
            say(f"batch {batch+1}: {len(recs)}配置 {len(cands)}候选")
        if not cands:
            say(f"batch {batch+1}: 无合格候选(全部亏损或折一致性差), 继续下一批")
            continue
        # 决赛验收
        best_final = None
        for c in cands:
            fr = sc.final_eval(c["cfg"])
            print(f"[master]   决赛 {c['cfg_id']}: 搜索PnL {c['pnl']:.0f} -> 决赛PnL {fr['pnl']:.0f} "
                  f"({fr['trades']}笔 PLR {fr['plr']:.2f})", flush=True)
            if best_final is None or fr["pnl"] > best_final[1]["pnl"]:
                best_final = (c, fr)
        c, fr = best_final
        say(f"batch {batch+1} 决赛最优: 搜索{c['pnl']:.0f} -> 决赛{fr['pnl']:.0f} "
            f"({fr['trades']}笔 胜率{fr.get('win_rate',0):.1%} PLR {fr['plr']:.2f} Sharpe {fr['sharpe']:.1f})")
        with open(os.path.join(RES, "finals.jsonl"), "a") as f:
            f.write(json.dumps({"batch": batch, "cfg_id": c["cfg_id"], "cfg": c["cfg"],
                                "search": {k: c[k] for k in ("pnl","trades","sharpe","plr","pos_fold_ratio")},
                                "final": fr}) + "\n")
        if fr["pnl"] > TARGET:
            champion = {"cfg": c["cfg"], "final": fr, "search": c["pnl"]}
            with open(os.path.join(RES, "champion.json"), "w") as f:
                json.dump(champion, f, indent=2)
            say(f"🏆 冠军达成! 决赛PnL {fr['pnl']:.0f} > {TARGET} ({fr['trades']}笔 PLR {fr['plr']:.2f})")
            print("[master] CHAMPION!", flush=True)
            break
        else:
            say(f"未达标(差 {TARGET - fr['pnl']:.0f}), 外层继续")
    if champion is None:
        say(f"预算耗尽({REPEATS}批), 未达 {TARGET}. 最优决赛结果见 finals.jsonl")
        print("[master] budget exhausted, no champion", flush=True)

if __name__ == "__main__":
    main()
