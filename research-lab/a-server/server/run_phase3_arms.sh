#!/bin/bash
# Phase 3: m1sc_ad 续训四臂 (预注册, 共享pack/超参/折, 严格消融)
#   A0 m1sc_ad           基线重建 (单lgb+xgb)
#   A1 m1sc_ad_ens       3种子集成 (v3bal制胜牌)
#   A2 m1sc_ad_ens6      6种子深集成 (12成员/方向)
#   A3 m1sc_ad_ens_hl180 近因加权 (半衰期270->180)
# 并行执行, 各臂日志独立, checkpoint断点续跑
BASE=/root/rivermind-fs/xauusd
LOGS=$BASE/logs
cd $BASE/v2_ens
mkdir -p $LOGS
CSV=$BASE/data/XAUUSDc_M1_202201022305_202606262057.csv

for v in m1sc_ad m1sc_ad_ens m1sc_ad_ens6 m1sc_ad_ens_hl180; do
  echo "[phase3] launch $v"
  nohup python3 run_m1.py --stage wf --variant $v --csv $CSV \
    >> $LOGS/05_wf_$v.log 2>&1 &
  echo "  pid $!"
done
echo "[phase3] all launched. 监控: tail -f $LOGS/05_wf_*.log"
echo "等待全部完成..."
wait
echo "[phase3] all done. 汇总:"
python3 - << 'EOF'
import json, os
print(f"{'变体':<20}{'笔数':>6}{'胜率':>8}{'PLR':>7}{'PnL$':>9}{'Sharpe':>8}")
for v in ["m1sc_ad", "m1sc_ad_ens", "m1sc_ad_ens6", "m1sc_ad_ens_hl180"]:
    fp = f"results/summary_{v}.json"
    if os.path.exists(fp):
        m = json.load(open(fp))["ml"]
        print(f"{v:<20}{m['trades']:>6}{m['win_rate']:>8.3f}{m['plr']:>7.2f}"
              f"{m['total_pnl']:>9.1f}{m['sharpe']:>8.2f}")
print("原始m1sc_ad历史基准:  6496笔 0.352 2.05 732.7 2.29 (原超参丢失, 本轮为重新调参)")
EOF
