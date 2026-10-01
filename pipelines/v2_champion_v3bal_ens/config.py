"""
XAUUSD ML v2 — 全局配置
所有关键参数集中在此, 与原代码散落各处的硬编码不同。
单位约定:
  - 价格: 美元 (XAUUSD 报价)
  - 时间: MT5 服务器时间 (与CSV一致)
  - point: XAUUSDc 报价最小变动 = 0.001 (3位小数报价), 可配置
"""

CFG = {
    # ---------- 数据 ----------
    "data_path": "/home/z/my-project/upload/5_extracted/XAUUSDc_M1_202201022305_202606262057.csv",
    "resample_rule": "5min",          # M1 -> M5 (计算切片: 数据量/5, 2核CPU友好; 本地可改 "1min")
    "point_value": 0.001,             # 1个SPREAD点 = 0.001美元 (XAUUSDc 3位小数报价; 保守可改0.01, 见敏感性分析)
    "spread_impute": "month_median",  # 2022-23点差缺失 -> 用当月非零中位数, 仍无则前后月最近的中位数
    "spread_floor_mult": 3.0,         # SL距离下限 = 3×当月中位数点差成本(防止障碍被点差吞没)

    # ---------- 三重障碍 (ATR自适应, 已用真实数据在首训练窗标定并冻结) ----------
    # 标定依据 (scripts/calibrate_barriers.py, 训练窗2022-01~2024-07真实数据):
    #   ATR24h(M5) 中位数 $1.22 | (2.9,1.657) = 精确复现原 $3.5/$2.0 几何 (TP率25.5%, 超时23.7%)
    #   默认 (2.0,1.143): TP率33.0%, 超时8.4%, 持仓中位26分钟, 盈亏比保持1.75
    "horizon_m1": 120,                # 垂直障碍: 120根M1 = 2小时
    "atr_window_m5": 288,             # ATR窗口: 288根M5 = 24小时交易时间 (波动率基准)
    "tp_atr_mult": 2.0,               # TP = 2.0 × ATR24h   (2022-24中位 $2.45; 2026中位 $11.8)
    "sl_atr_mult": 1.1429,            # SL = 1.1429 × ATR24h (2022-24中位 $1.40; 2026中位 $6.7)
    "sl_floor_usd": 0.30,             # SL 绝对下限(美元), 与 3×点差 floor 取更大者
    # "plr" = 3×障碍几何族: 盈亏比导向, scripts/calibrate_barriers_v3.py 在首训练窗
    # (2022-01~2024-07, 无未来信息) 网格45点标定。随机入场口径已含点差与SL同K线悲观规则。
    "plr_geometry": {                 # 主几何: 可学习性与PLR的平衡点
        "tp_atr_mult": 4.0, "sl_atr_mult": 1.0, "horizon_m1": 360,   # 6h垂直障碍
        # 训练窗真实口径: PLR 3.158, avgW $4.75/avgL $1.50, TP率 L/S 19.6/18.7%,
        # 超时2.3%, EV L/S -$0.160/-$0.206, 持仓中位32分钟
    },
    "plr_geometry_aggr": {            # 激进几何: 网格按PLR最大化的规则选点
        "tp_atr_mult": 4.0, "sl_atr_mult": 0.8, "horizon_m1": 360,
        # 训练窗真实口径: PLR 3.835, avgW $4.78/avgL $1.25, TP率 L/S 16.7/16.0%,
        # 超时1.6%, EV L/S -$0.161/-$0.197, 持仓中位23分钟
    },
    "balance_geometry": {              # 平衡几何: 胜率下限约束内PLR最大化 (v4标定)
        # scripts/calibrate_barriers_v4.py, 同一首训练窗(无未来信息):
        # 约束 随机TP率 L>=28% S>=26% (v3aggr实现胜率18.4%被反馈过低),
        "tp_atr_mult": 3.0, "sl_atr_mult": 1.1429, "horizon_m1": 360,
        # 训练窗真实口径: PLR 2.165, avgW $3.66/avgL $1.69, TP率 L/S 28.0/26.6%,
        # 超时1.0%, EV L/S -$0.157/-$0.217, 持仓中位34分钟
    },
    "wr_floor": 0.30,                  # plr_wr阈值目标的胜率下限 (内部验证段实测胜率)

    # ---------- 走查验证 (用户指定: 24-36个月训练, 验证月滚动并入) ----------
    "initial_train_months": 31,       # 首折训练窗 31 个月 (2022-01 ~ 2024-07), 在24-36范围内
    "max_train_months": 36,           # 窗口上限: 扩到36个月后转滚动(剔最旧月), 永远满足"以往24-36个月"
    "first_oos_month": "2024-08",     # 首个OOS月
    "last_oos_month": "2026-07",      # 最后OOS月 (数据实际到2026-07-17)
    "purge_extra_m5": 6,              # 训练窗右端额外embargo (M5根数), 加在标签horizon之上

    # ---------- 模型 ----------
    "models": ["lgb", "xgb"],         # 集成: LightGBM + XGBoost 概率均值
    "sample_halflife_days": 270,      # 时间衰减样本权重半衰期 (适应金价新 regime)
    "early_stopping_rounds": 50,
    "max_rounds": 600,
    "tune": {
        "n_configs": 24,              # 随机搜索配置数 (每方向)
        "inner_folds": 2,             # 内部purged CV折数 (只用训练窗数据!)
        "subsample_step": 2,          # 调参时每隔2根M5取1根 (加速, 只影响调参不影响最终模型)
    },

    # ---------- 阈值校准 (只在训练窗内部的尾段, 绝不触碰OOS) ----------
    "inner_val_frac": 0.20,           # 训练窗尾部20%做内部验证段(带purge gap)
    "threshold_grid": [0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65, 0.675, 0.70,
                       0.725, 0.75, 0.775, 0.80, 0.85],
    "min_trades_inner": 150,          # 内部验证段最少交易数(阈值候选约束)
    "threshold_objective": "mean_pnl_x_sqrtN",  # 均值PnL×sqrt(交易数): 兼顾质量与样本量, 抗过拟合

    # ---------- 回测模拟器 ----------
    "max_concurrent": 1,              # 单持仓 (与EA行为一致)
    "cooldown_m1": 10,                # 平仓后冷却10根M1再允许新信号
    "entry_slippage_usd": 0.0,        # 额外滑点(美元), 点差之外

    # ---------- 基准对照 ----------
    "random_seed": 42,
    "bootstrap_iters": 10000,

    # ---------- 消融 ----------
    "ablation": "legacy24",           # 原24特征同流程对比 (量化特征升级的真实增益)
}
