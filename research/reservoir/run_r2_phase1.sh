#!/usr/bin/env bash
# R2 phase-1 improvement arm grid (runs sequentially in background)
cd /home/z/my-project/research/reservoir
export OMP_NUM_THREADS=1
CSV=../../research-lab/data.csv
OUT=/tmp/r2arms
mkdir -p $OUT

run() {  # run <label> <extra flags...>
  local label=$1; shift
  if [ -f $OUT/$label.json ]; then echo "skip $label (exists)"; return; fi
  echo "=== ARM $label : $@ ==="
  python3 reservoir_engine2.py --csv $CSV --out $OUT/$label.json --label $label "$@" 2>&1 | rg "PnL=|IC_multi|sentinel|DONE|Error|Traceback" 
}

run a1_log            --target log
run a2_leaks          --leaks 0.1,0.3,0.9
run a3_feat8          --n-feat 8
run a12_log_leaks     --target log --leaks 0.1,0.3,0.9
run a13_log_feat8     --target log --n-feat 8
run a23_leaks_feat8   --leaks 0.1,0.3,0.9 --n-feat 8
run a123_all          --target log --leaks 0.1,0.3,0.9 --n-feat 8
run a123_shuffle      --target log --leaks 0.1,0.3,0.9 --n-feat 8 --shuffle
echo "PHASE1_ALL_DONE"
