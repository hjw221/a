#!/bin/bash
# run_v14.sh — v14 年度均衡双臂驱动
# Arm 2: champ 几何美元钳制 Pareto (5臂, 26线程, ~40-70min)
# Arm 1: 弱年份目标搜索 (4 worker x 7线程 x 12轮 x 2批, ~1.5-3h)
cd "$(dirname "$0")"
mkdir -p results_v14 results_weak
export NTFY_TOPIC=xauusd-qv7m2zk9-res
say() { curl -s --max-time 20 -X POST "https://ntfy.sh/$NTFY_TOPIC" -d "$1" -H "Title: v14" >/dev/null 2>&1 || true; }

say "v14 双臂启动: caparms(5臂美元钳制) -> weaksearch(4x12x2弱年份目标)"

ML_THREADS=26 python3 v14_caparms.py > results_v14/caparms.out 2>&1
RC1=$?
say "caparms 完成 rc=$RC1 (results_v14/caparms.json)"

N_WORKERS=4 N_ROUNDS=12 REPEATS=2 python3 v14_weaksearch.py --master > results_weak/master.out 2>&1
RC2=$?
say "v14 全部完成 rc=$RC2 (caparms.json + weak_champion.json / finals_weak.jsonl)"
