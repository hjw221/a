"""xauusd_ml_scratch — 从零构建的 1m 管线, 全局配置。

方法论(用户指令, 最高优先级):
  阶段1 ML清洗+特征工程 -> 找出哪些数据有用、哪些波段扣成本后能吃到
  阶段2 ML因子挖掘
  阶段3 模型训练(只能放最后)

口径共识(沿用上一会话, 代码与旧管线 xauusd_ml_v2 零复用):
  - 数据: 原始M1, 2023-01 起
  - 成本: 单边点差 $0.03, commission 0
  - 切分: 核心训练=2023, 内部验证=2024H1(唯一消融试验田), OOS=2024-07后本轮绝不触碰
"""
from pathlib import Path

ROOT = Path('/home/z/my-project')
SRC_CSV = ROOT / 'upload/5_extracted/XAUUSDc_M1_202201022305_202606262057.csv'
DATA_DIR = ROOT / 'download/xauusd_data'           # 重建的月度文件(数据层)
PKG = ROOT / 'download/xauusd_ml_scratch'          # 本管线
CACHE = PKG / 'cache'
RESULTS = PKG / 'results'

# ---- 成本 ----
COST_SIDE = 0.03            # 单边点差 $
COST_RT = 2 * COST_SIDE     # 往返成本 $0.06

# ---- 时间切分 ----
DATA_START = '2023-01-01'          # 数据宇宙起点
WARMUP_START = '2022-12-01'         # 特征回看暖机(1个月)
TRAIN_END = '2023-12-31 23:59:59'   # 核心训练段 = 2023 全年
VAL_START = '2024-01-01'            # 内部验证段 = 2024H1
VAL_END = '2024-06-30 23:59:59'
OUTCOME_PAD_END = '2024-07-15 23:59:59'  # 仅为完成val末笔交易的前瞻结局(不参与任何拟合/选择)
OOS_START = '2024-07-01'             # 样本外, 本轮绝不触碰

# ---- 阶段1b 几何标定网格(仅用2023数据) ----
KE_GRID = [1.5, 2.0, 2.5, 3.0]   # TP = kE * ATR60($)
PLR_GRID = [2.0, 2.5]           # SL = TP / PLR
H_GRID = [120, 240, 360]        # 超时horizon(分钟)
ATR_WIN = 60                     # ATR窗口(分钟), 因果

# ---- 波动Regime(②消融用, 切点只从2023分布冻结) ----
REGIME_Q = [0.3333, 0.6667]      # ATR60三分位

# ---- δ收紧消融网格(②) ----
DELTA_GRID = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]  # 阈值=0.50+delta
REGIME_VARIANTS = ['all', 'no_low', 'mid', 'mid_high', 'high']

SEED = 42
NJOBS = 2
