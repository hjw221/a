#!/bin/bash
# Phase 2: m1sc_ad 重建 — prep(特征+标签) -> tune(首训练窗内重新调参, 原超参已随驱动丢失)
set -e
BASE=/root/rivermind-fs/xauusd
LOGS=$BASE/logs
cd $BASE/v2_ens
mkdir -p $LOGS
CSV=$BASE/data/XAUUSDc_M1_202201022305_202606262057.csv

echo "[phase2] prep m1sc_ad pack (s1-34特征 + 自适应障碍标签) ..."
python3 run_m1.py --stage prep --csv $CSV >> $LOGS/03_prep_m1sc_ad.log 2>&1
echo "[phase2] prep done, see logs/03_prep_m1sc_ad.log"

echo "[phase2] tune (12配置x2内折, 首训练窗2022-01~2024-07, ~1h) ..."
python3 run_m1.py --stage tune --csv $CSV >> $LOGS/04_tune_m1sc_ad.log 2>&1
echo "[phase2] tune done -> results/tuned_params_m1.json"
cat $BASE/v2_ens/results/tuned_params_m1.json | python3 -c "import json,sys; d=json.load(sys.stdin)['m1sc_ad']; print('long lgb:', d['long']['lgb']); print('long xgb:', d['long']['xgb'])"
