#!/bin/bash
# 环境自检 + 依赖安装 (幂等, 可重复执行)
set -e
echo "== python =="; python3 --version
echo "== 依赖 =="
python3 - << 'EOF' || pip3 install "lightgbm==4.5.0" "xgboost==2.1.3" "pandas>=2.0" "numpy>=1.26" "scikit-learn>=1.5" "numba>=0.60"
import lightgbm, xgboost, pandas, numpy, sklearn, numba
print("lgb", lightgbm.__version__, "| xgb", xgboost.__version__, "| pd", pandas.__version__,
      "| np", numpy.__version__, "| nb", numba.__version__)
EOF
python3 -c "import lightgbm, xgboost, pandas, numpy, sklearn, numba; print('deps OK:', lightgbm.__version__, xgboost.__version__)"
echo "== 资源 =="
nproc; free -g | head -2; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no gpu visible"
df -h /root | tail -1
echo "== setup done =="
