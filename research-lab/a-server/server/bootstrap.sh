#!/bin/bash
# bootstrap.sh — 3090 服务器一键启动 (自包含, 幂等, 可重复执行)
# 功能: 依赖检查 -> GitHub/镜像拉取载荷 -> 数据还原(md5校验) -> 启动三阶段管线 + ntfy 代理
# 由用户在 Termux SSH 会话粘贴一条命令触发。
set -u
BASE=/root/rivermind-fs/xauusd
RES_TOPIC="xauusd-qv7m2zk9-res"
RAW_GZ="$BASE/data/XAUUSDc_M1_202201022305_202606262057.csv"
MD5_EXPECT="f6b0d44ded471ed408505deb03a69720"
REPO="https://github.com/hjw221/a.git"
MIRROR_PREFIX="https://ghfast.top/"
WORK=/root/rivermind-fs/.bootstrap

say() { curl -s --max-time 30 -X POST "https://ntfy.sh/$RES_TOPIC" -d "$1" -H "Title: bootstrap" >/dev/null 2>&1 || true; }

echo "[boot] $(date '+%F %T') bootstrap v1.2"
say "bootstrap start $(uname -m), $(nproc) cores"

mkdir -p "$BASE" "$WORK" "$BASE/logs"

# ---------- 1) 依赖 ----------
need=0
command -v git >/dev/null || { apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq git curl >/dev/null 2>&1; need=1; }
command -v git >/dev/null || { echo "FATAL: git 安装失败"; say "FATAL: no git"; exit 1; }
command -v python3 >/dev/null || { apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq python3 python3-pip >/dev/null 2>&1; }
command -v python3 >/dev/null || { echo "FATAL: python3 安装失败"; say "FATAL: no python3"; exit 1; }

python3 - << 'PYEOF' 2>/dev/null || PIPFAIL=1
import lightgbm, xgboost, pandas, numpy, sklearn, numba
print("deps ok:", lightgbm.__version__, xgboost.__version__)
PYEOF
if [ "${PIPFAIL:-0}" = "1" ]; then
  echo "[boot] 安装 python 依赖 (~2分钟)..."
  pip3 install -q "lightgbm==4.5.0" "xgboost==2.1.3" "pandas>=2.0" "numpy>=1.26" "scikit-learn>=1.5" "numba>=0.60" 2>&1 | tail -2
  python3 -c "import lightgbm, xgboost, numba; print('deps installed:', lightgbm.__version__, xgboost.__version__)" \
    || { say "FATAL: pip deps failed"; echo "FATAL: pip deps failed"; exit 1; }
fi
echo "[boot] 依赖 OK"

# ---------- 2) 拉取载荷 (GitHub 直连 -> ghfast.top -> gh-proxy.com 镜像 fallback) ----------
# v1.2 fix: v1.1 中 `git clone ... | tail -1` 的退出码是 tail 的(永远为 0),
#           导致镜像 fallback 永远不会执行、直连失败时直接误报 FATAL。
#           现改为先捕获 git 真实退出码再回显末行日志。
cd "$WORK"
try_clone() {
  rm -rf a-server
  if out=$(git clone --depth 1 -b server "$1" a-server 2>&1); then
    echo "$out" | tail -1
    return 0
  fi
  echo "$out" | tail -1
  return 1
}
SRC=""
if [ -d a-server/.git ]; then
  echo "[boot] 已有 clone, pull 更新..."
  (cd a-server && git fetch origin server --depth 1 && git reset --hard origin/server) \
    && SRC="$WORK/a-server" || { echo "[boot] pull 失败, 重新 clone..."; rm -rf a-server; }
fi
if [ -z "${SRC:-}" ] && [ ! -d a-server/.git ]; then
  echo "[boot] git clone (server 分支, shallow; 直连 -> 镜像 fallback)..."
  if try_clone "$REPO"; then SRC="$WORK/a-server"
  elif try_clone "${MIRROR_PREFIX}${REPO}"; then SRC="$WORK/a-server"
  elif try_clone "https://gh-proxy.com/${REPO}"; then SRC="$WORK/a-server"
  else SRC=""; fi
fi
[ -n "${SRC:-}" ] && [ -f "$SRC/server/v2_ens/run_all.py" ] || { echo "FATAL: 载荷拉取失败"; say "FATAL: clone failed (direct+mirror)"; exit 1; }
echo "[boot] 载荷就绪: $SRC/server"

# ---------- 3) 数据还原 + md5 校验 ----------
mkdir -p "$BASE/data"
if [ ! -f "$RAW_GZ" ] || [ "$(md5sum "$RAW_GZ" | cut -d' ' -f1)" != "$MD5_EXPECT" ]; then
  echo "[boot] gunzip 原始CSV (~1分钟)..."
  gunzip -c "$SRC/server/data/XAUUSDc_M1_202201022305_202606262057.csv.gz" > "$RAW_GZ"
  ACT=$(md5sum "$RAW_GZ" | cut -d' ' -f1)
  [ "$ACT" = "$MD5_EXPECT" ] || { echo "FATAL: md5 mismatch ($ACT)"; say "FATAL: md5 mismatch $ACT"; exit 1; }
fi
echo "[boot] 数据校验 OK ($(du -h "$RAW_GZ" | cut -f1))"

# ---------- 4) 部署代码 (幂等 rsync 式覆盖) ----------
mkdir -p "$BASE/v2_ens/results"
cp -f "$SRC/server/v2_ens/"*.py "$BASE/v2_ens/"
cp -f "$SRC/server/v2_ens/results/"*.json "$BASE/v2_ens/results/" 2>/dev/null || true
cp -f "$SRC/server/"*.sh "$SRC/server/agent.py" "$BASE/"
mkdir -p "$BASE/v2_ens/results/historical_reference"
cp -f "$SRC/server/v2_ens/results/historical_reference/"* "$BASE/v2_ens/results/historical_reference/" 2>/dev/null || true
chmod +x "$BASE/"*.sh
# config.py 数据路径已内置服务器绝对路径, 校验一下
grep -q "/root/rivermind-fs" "$BASE/v2_ens/config.py" || sed -i \
  's|"data_path": .*|"data_path": "/root/rivermind-fs/xauusd/data/XAUUSDc_M1_202201022305_202606262057.csv",|' \
  "$BASE/v2_ens/config.py"
grep -q "rivermind-fs" "$BASE/v2_ens/run_m1.py" || sed -i \
  's|RAW_CSV_DEFAULT = .*|RAW_CSV_DEFAULT = "/root/rivermind-fs/xauusd/data/XAUUSDc_M1_202201022305_202606262057.csv"|' \
  "$BASE/v2_ens/run_m1.py"
echo "[boot] 代码部署 OK"

# ---------- 5) 启动 ntfy 代理 (幂等: 已在线则跳过) ----------
if ! pgrep -f "agent.py" >/dev/null; then
  nohup python3 "$BASE/agent.py" >> "$BASE/logs/agent.log" 2>&1 &
  echo "[boot] agent.py 已启动 pid $!"
else
  echo "[boot] agent.py 已在运行"
fi

# ---------- 6) 启动主管线 (幂等: 已在跑则跳过) ----------
if ! pgrep -f "run_master.sh" >/dev/null; then
  nohup bash "$BASE/run_master.sh" >> "$BASE/logs/master.log" 2>&1 &
  echo "[boot] run_master.sh 已启动 pid $!"
  say "pipeline launched"
else
  echo "[boot] run_master.sh 已在运行"
fi

echo "[boot] 全部就绪。日志: $BASE/logs/ ; ntfy 监控: $RES_TOPIC"
say "bootstrap DONE $(date '+%F %T') - pipeline running"
