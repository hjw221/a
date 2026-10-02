#!/bin/bash
# Phase 1: 重跑冠军 v3bal_ens (24个OOS月, 与本地历史结果严格同口径)
# prep 重建 pack_v3bal (v3-34特征+平衡几何标签) -> wf 走查 v3bal_ens (6成员AUC加权/方向)
set -e
BASE=/root/rivermind-fs/xauusd
LOGS=$BASE/logs
cd $BASE/v2_ens
mkdir -p $LOGS

echo "[phase1] prep v3bal ..."
python3 run_all.py --stage prep --features v3bal >> $LOGS/01_prep_v3bal.log 2>&1
echo "[phase1] prep done, see logs/01_prep_v3bal.log"

echo "[phase1] wf v3bal_ens (24 folds) ..."
python3 run_all.py --stage wf --features v3bal_ens >> $LOGS/02_wf_v3bal_ens.log 2>&1
echo "[phase1] wf done, see logs/02_wf_v3bal_ens.log"

echo "[phase1] summary:"
python3 - << 'EOF'
import json
s = json.load(open("results/summary_v3bal_ens.json"))
m = s["ml"]
print(f"v3bal_ens 服务器重跑: {m['trades']}笔 胜率{m['win_rate']:.3f} PLR {m['plr']:.2f} "
      f"PnL ${m['total_pnl']:.1f} Sharpe {m['sharpe']:.2f}")
print("本地历史基准:        1611笔 胜率0.316 PLR 2.57 PnL $1113.5 Sharpe 1.74")
EOF
