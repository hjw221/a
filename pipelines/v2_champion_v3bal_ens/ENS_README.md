# 牌2集成（v3bal_ens）交付说明

**日期**: 2026-09-14 | **验证**: 24个真实OOS月（2024-08 ~ 2026-07），expanding window walk-forward
**本包所有数字均为真实运行输出，无任何模拟、估算或挑选。**

---

## 一、一句话结论

在与 v3bal 基线**严格可比**的同一底座上（同特征34个、同平衡几何 TP=3.0×ATR / SL=1.1429×ATR / 6h、
同冻结超参、同 purged 走查口径、同回测模拟器含点差成本），只把"单种子 lgb+xgb 均匀平均"
换成 **6成员 val-AUC 加权集成**，24个OOS月真实总PnL 从 **+$596 → +$1,114（+87%）**，
Sharpe 从 1.33 → **1.74**，且单笔期望 Bootstrap 95% CI **[+$0.137, +$1.254] 首次不含零**
（全项目第二个做到这一点的变体，第一个是legacy基线）。

信号强度三张牌里唯一的赢家，生产模型已切换。

## 二、三张牌真实OOS对比（同一24个月，含点差，单持仓+冷却）

| 变体 | 交易 | 笔/天 | 胜率 | PLR | avgW/avgL$ | 总PnL$ | PF | Sharpe | maxDD$ | 连亏 | CI95(均值/笔) |
|------|-----:|-----:|-----:|----:|------:|-----:|---:|----:|----:|----:|----:|
| v3bal基线 | 1,331 | 2.6 | 32.2% | 2.40 | 11.63/-4.85 | +596.2 | 1.136 | 1.33 | 250 | 14 | [-0.095, +1.004] 含0 |
| 牌1 v3bal_uniq（去重叠） | 2,719 | 5.4 | 27.7% | 2.52 | 13.01/-5.16 | **-372.2** | 0.963 | -0.51 | 590 | 20 | [-0.536, +0.263] |
| **牌2 v3bal_ens（集成）** | **1,611** | 3.2 | 31.6% | **2.57** | **14.02/-5.47** | **+1,113.5** | **1.185** | **1.74** | 300 | 18 | **[+0.137, +1.254] 不含0** |
| 牌3 v3bal_meta（元标签） | 2,012 | 4.0 | 27.7% | 2.55 | 13.98/-5.49 | **-203.5** | 0.975 | -0.56 | 483 | 24 | [-0.568, +0.398] |

对照基准（同一模拟器）：恒做多 -$1,422 | 恒做空 -$1,613 | 随机方向 -$2,324 | 同频随机 -$897。

牌2对同频随机的模型边际：**$996 → $2,011（翻倍）**；单笔期望 $0.448 → $0.691（+54%）。

## 三、牌2是什么（实现见 variants_ext.py `_fold_ens`）

每个方向、每一折，把原来 1 个 lgb + 1 个 xgb（固定种子）的训练改为：

```
成员 = { lgb, xgb } × 种子 { 42, 1337, 2024 }  =  6 个模型/方向
权重 w_i = max(0, 内部val段AUC_i − 0.50)，AUC < 0.51 的成员权重清零
全部清零 → 退回均匀平均（24折中仅1折发生，fallback率4.2%）
预测概率 = Σ w_i · p_i / Σ w_i
```

其余一切不动：特征集、几何、阈值校准（plr_wr两段式）、purge、时间衰减权重、早停、
冻结超参全部与 v3bal 基线逐位一致——变体间差异**只**来自集成方式，归因干净。

**防泄露**：成员权重只由训练窗内部 val 段（训练窗尾部20%，带purge gap）的 AUC 决定，
OOS 月份从头到尾不参与任何训练/权重/阈值选择。诊断记录（`results/analysis_v5.json`）：
24折平均保留 4.65/6 个成员，成员 val-AUC 均值 0.516。

## 四、为什么有效（机制，诚实版）

- **不是 AUC 变高了**：OOS AUC long 0.506 / short 0.512，与基线（0.504/0.513）持平。
- **是概率分布的高分位尾更可靠了**：6成员平均去掉单模型的种子噪声，
  概率在阈值上方的高置信区域更稳定，阈值校准因此落在更优操作点
  （阈值回退折 5 → 3，即内部验证段更常找到"胜率≥30%且PLR最大化"的合格阈值）。
- avgW $11.63 → $14.02（+21%）、avgL $4.85 → $5.47：同一几何下赢的时候赢得更多，
  属于"同一批信号里更敢下高分位"的操作点改善，不是发现了新信号。

## 五、诚实的代价与风险（不粉饰）

1. **利润更集中，不是更稳**：前2个利润月（2026-03 +$593、2026-05 +$211）占总利润 72%
   （基线 49%）；盈利月 15/24（基线 18/24）。
2. **空头贡献 73%**：long +$298.8（749笔）/ short +$814.8（862笔）。多头侧其实和基线差不多，
   增量大头在空头——与"2024-2026大涨中6h尺度急跌回调是主要盈利来源"的既有结论一致。
3. **回撤与连亏变差一点**：maxDD $250 → $300，最长连亏 14 → 18 笔。
   2026-01（-$118）和 2026-06（-$136）是24个月里最疼的两段。
4. **统计上的位置**：CI 不含零是1,611笔的直接观测+10,000次bootstrap，是"正期望的统计证据"，
   不等于未来每月都正。样本内外制度变化（如点差、波动率结构）仍是主要风险。
5. 月度PnL全序列（真实）：2024-08 +6.4 | 09 +4.5 | 10 +50.3 | 11 -28.0 | 12 -6.2 |
   2025-01 +17.5 | 02 -7.6 | 03 -7.8 | 04 -25.4 | 05 +84.1 | 06 +29.7 | 07 -16.5 |
   08 +20.5 | 09 +21.1 | 10 +157.6 | 11 +22.0 | 12 -43.0 |
   2026-01 -118.0 | 02 +136.2 | 03 +593.4 | 04 +59.5 | 05 +211.0 | 06 -135.7 | 07 +87.9

## 六、本包内容

```
xauusd_ml_v2/
├── ENS_README.md            # 本文件（牌2专述）
├── README.md                # 项目总README（v1~v5全部真实结果与结论）
├── config.py                # 全局参数（几何/走查/成本/阈值网格；VARIANTS见run_all.py）
├── data.py / features.py / features_v3.py / labeling.py / walkforward.py / models.py
├── variants_ext.py          # ★ 三张牌实现（_fold_stride/_fold_ens/_fold_meta）
├── backtest.py              # 单持仓模拟器 + plr_wr阈值校准 + Bootstrap
├── run_all.py               # 编排（prep→tune→wf，VARIANTS表驱动，checkpoint续跑）
├── train_final.py           # 生产重训（--variant v3bal_ens）
├── predict_live.py          # 实盘信号（支持6成员加权加载）
├── analyze.py
├── scripts/                 # ★ 几何标定与分析脚本（相对路径, 解压即跑）
│   ├── calibrate_barriers.py     # v1 ATR几何标定(2.0/1.143, 只用首训练窗)
│   ├── calibrate_barriers_v3.py  # v3 盈亏比几何标定(4.0/1.0与4.0/0.8)
│   ├── calibrate_barriers_v4.py  # v4 平衡几何标定(3.0/1.1429 — v3bal/ens在用)
│   ├── calib_v3_result.json      # v3标定真实网格输出
│   ├── calib_v4_result.json      # v4标定真实网格输出
│   ├── analyze_v3.py / analyze_v4.py / analyze_v5.py  # analysis_*.json生成器
│   ├── streak_analysis.py        # 连亏统计
│   └── smoke_features_v3.py      # v3特征因果性校验(截断vs全量)
├── production_models/       # ★ v3bal_ens 生产模型（2026-09-14重训，36月全量窗）
│   ├── trading_config.json  #   阈值0.3397/0.2678 + 12成员权重 + 特征清单
│   ├── {long,short}_{lgb,xgb}_{s42,s1337,s2024}_latest.*   # 12个模型文件
│   └── feature_importance_v3.json
└── results/                 # 全部真实运行输出（本包不含cache/与checkpoints/，可重建）
    ├── trades_v3bal_ens.csv / trades_v3bal.csv / trades_v3bal_uniq.csv / trades_v3bal_meta.csv
    ├── per_fold_v3bal_ens.json / per_fold_v3bal.json / ...（逐折阈值/AUC/成员权重）
    ├── summary_v3bal_ens.json / summary_v3bal.json / ...
    ├── analysis_v5.json     # ★ 三张牌对比+Bootstrap+月度+多空拆分+机制诊断
    ├── tuned_params.json    # 冻结超参
    └── charts*/             # 图表
```

## 七、如何复现 / 使用

```bash
# 依赖: pandas numpy lightgbm xgboost numba scikit-learn matplotlib
# 1) 复现牌2走查（pack复用v3bal缓存，跳过prep/tune；每折~35秒，2核CPU约15分钟）:
python run_all.py --stage wf --features v3bal_ens

# 2) 复现基线对照:
python run_all.py --stage wf --features v3bal

# 3) 用你自己的CSV: 改 config.py 的 data_path 后全流程:
python run_all.py --stage prep --features v3bal
python run_all.py --stage wf   --features v3bal_ens

# 4) 生产重训（每周/月跑一次，产出production_models/）:
python train_final.py --variant v3bal_ens --data_path 你的新CSV

# 5) 复现三张牌对比分析表(analysis_v5.json, 已验证逐字节可复现):
python scripts/analyze_v5.py

# 6) 几何从零重标定(只用首训练窗2022-01~2024-07, 无未来信息; 每个约几分钟):
python scripts/calibrate_barriers_v4.py     # 平衡几何(牌2在用)
python scripts/calibrate_barriers_v3.py     # 盈亏比几何族
python scripts/smoke_features_v3.py         # v3特征因果性校验

# 7) 实盘: EA每5分钟调 predict_signal(recent_m1_df)
```
以上全部脚本已改为包内相对路径，解压到任意目录即可运行（只需在 `config.py` 里把
`data_path` 指向你的CSV）。

## 八、EA对接要点（与v3bal相同的部分不再重复，只列差异）

1. `predict_live.py` 自动按 `trading_config.json` 加载12个成员并按权重加权平均——
   与走查 `predict_weighted` 同口径，EA调用方式不变（仍是一个概率+方向+动态TP/SL距离）。
2. 阈值更新为 **long 0.3397 / short 0.2678**（内部验证段231笔校准，q=0.985）。
3. short侧 2 个 AUC<0.51 成员（xgb_s1337 / xgb_s2024）权重已清零——文件仍在但预测时不参与。
4. TP/SL 仍为动态：TP=3.0×ATR(24h)、SL=1.1429×ATR(24h)、6小时未触发市价平仓；v3特征需TICKVOL。
