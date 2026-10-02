#!/bin/bash
# 串行执行全部三阶段, ntfy 汇报关键节点。被 bootstrap.sh 以 nohup 启动。
BASE=/root/rivermind-fs/xauusd
RES_TOPIC="xauusd-qv7m2zk9-res"
cd $BASE

say() { curl -s --max-time 30 -X POST "https://ntfy.sh/$RES_TOPIC" -d "$1" -H "Title: pipeline" >/dev/null 2>&1 || true; }

say "=== pipeline start $(date '+%F %T') ==="

echo "[master] phase1: v3bal_ens 重跑" | tee -a logs/master.log
say "phase1 start: 重跑冠军 v3bal_ens"
if bash run_phase1_v3bal.sh >> logs/master.log 2>&1; then
  say "phase1 done: $(grep -A2 '服务器重跑' logs/02_wf_v3bal_ens.log | tail -2 | tr '\n' ' ')"
else
  say "PHASE1 FAILED - 见 logs/01_prep_v3bal.log / 02_wf_v3bal_ens.log"
  exit 1
fi

echo "[master] phase2: m1sc_ad prep+tune" | tee -a logs/master.log
say "phase2 start: m1sc_ad 特征标签+重新调参"
if bash run_phase2_m1sc_prep.sh >> logs/master.log 2>&1; then
  say "phase2 done: tune 完成 (tuned_params_m1.json)"
else
  say "PHASE2 FAILED - 见 logs/03_prep_m1sc_ad.log / 04_tune_m1sc_ad.log"
  exit 1
fi

echo "[master] phase3: 四臂并行" | tee -a logs/master.log
say "phase3 start: 四臂并行走查 (A0/A1/A2/A3)"
bash run_phase3_arms.sh >> logs/master.log 2>&1
say "phase3 done (退出码 $?), 汇总见 master.log"

# ---- 结果打包上传 ----
cd $BASE/v2_ens
tar czf /tmp/results_bundle.tar.gz \
  results/summary_*.json results/per_fold_*.json results/trades_*.csv \
  results/tuned_params*.json ../logs/*.log 2>/dev/null
SZ=$(du -h /tmp/results_bundle.tar.gz | cut -f1)
say "results bundle ready (${SZ}), uploading..."
curl -s --max-time 600 -T /tmp/results_bundle.tar.gz "https://ntfy.sh/$RES_TOPIC" -d "results bundle ${SZ}" || say "UPLOAD FAILED"
say "=== pipeline ALL DONE $(date '+%F %T') ==="
