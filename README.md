# XAUUSD M1 量化模型项目 — 全量归档

> 仓库所有数字均为真实运行输出（真实走查 / 真实回测），无模拟、无估算、无挑选。
> 归档日期：2026-09-29

## 一、这个仓库是什么

XAUUSD（黄金）M1 数据的量化交易模型项目完整归档，包含：

1. **原始数据**：2022-01 ~ 2026-07 全部 M1 K线（55 个月度 CSV，约 160 万根）
2. **三代模型管线**：全部代码、生产模型、结果与报告
3. **结论**：旧冠军 `v3bal_ens` 仍是唯一有统计显著正期望的模型；从零重走的路线未超越它（判定依据见下表）

## 二、目录结构

```
data/m1/                        # 55 个月度 M1 CSV (time,open,high,low,close,volume)，2022-01 ~ 2026-07
pipelines/
  v2_champion_v3bal_ens/         # 【冠军】v2/v3 管线 + v3bal_ens 生产模型（12成员AUC加权集成）
                                #   含 mt5_package/ (MT5部署包) + ENS_README.md (交付说明)
  scratch_threestage/           # 从零三段式管线（ML清洗→因子挖掘→训练），2024H1 内部验证 + Regime/δ 消融
  v4_m1_ignition/               # v4 "M1点火-延续" 从零路线（已证伪）
scripts/                        # 全部分析/标定脚本 (v3~v7, scalp, m1, v4_* 系列)
packages/                       # 三个交付 zip（v3bal_ens 两个时点备份 + MT5 部署包）
```

## 三、核心对比：旧冠军 vs 从零新模型（OOS 2024-08 ~ 2026-07，同一验证窗，含成本）

| 模型 | 笔数 | 胜率 | 盈亏比PLR | AUC (OOS) | 总PnL | Sharpe | 单笔CI95 | 判定 |
|---|---:|---:|---:|---|---:|---:|---|---|
| **v3bal_ens（旧冠军）** | 1,611 | 31.6% | **2.57** | L 0.504 / S 0.513 | **+$1,113.5** | **1.74** | [+$0.14, +$1.25] **不含0** | 唯一存活 |
| scratch 三段式单模型 | — | — | 全交易期望 **-$0.046/笔**(2024H1, t=-2.87) | val L 0.5043 / S 0.5006 | 未获准烧OOS | — | — | **淘汰** |
| scratch s08 全量 | 2,729 | 41.4% | 1.49 | 未记录 | +$870.3 | 0.60 | [-$0.24, +$0.87] 含0 | 不显著 |
| v4 M1点火 | 3,246 | 32.9% | 1.93 / 1.63(真实点差) | pooled 0.512 | **-$321 / -$1,368** | -1.02 / -3.38 | 含0 | 亏损，证伪 |
| m1sc_ad (v2框架M1) | 6,496 | 35.2% | 2.05 | L 0.519 / S 0.516 | +$732.7 | 2.29 | [+$0.05, +$0.18] 不含0 | PLR/PnL 均低于冠军 |
| scalp1 | 6,386 | 39.2% | 0.94 | ~0.50 | -$1,276.9 | -7.49 | 含0 | 证伪 |

**判定（按"盈亏比低且AUC低则停"的判据）**：从零路线最终态（scratch 三段式单模型）
盈亏比有效为负（全交易期望显著为负）、AUC 0.504/0.501 低于旧冠军 0.504/0.513 —— **双双更低，从零路线终止**。

冠军 v3bal_ens 的生产模型（12 成员 LightGBM/XGBoost 集成 + `trading_config.json`）位于
`pipelines/v2_champion_v3bal_ens/production_models/`，MT5 部署包位于 `pipelines/v2_champion_v3bal_ens/mt5_package/`。

## 四、口径备忘

- **成本假设**：v2/s8/scatch 系列按固定单边点差 $0.03（往返 $0.06）；v4 与 m1 系列含"数据驱动真实点差"口径
  （SPREAD 列实测：2025 中位 $0.16/边、2026 $0.28/边）——真实点差口径下所有高频路线全部恶化。
- **几何**：冠军 v3bal_ens = TP 3.0×ATR / SL 1.1429×ATR / 6h 超时；scratch = 3.0/2.0/360min(M1)；v4 = (3.0/1.5)×ATR1440/H120。
- **验证纪律**：purged 月度滚动 walk-forward；scratch 路线内部验证段 2024H1，OOS 2024-07 后（s08/v4/m1sc 已消耗该窗口）。
- **失败教训**（详见各管线 README）：高频微障碍（±$0.20，点差占障碍 7.5%+）、DXY 特征、uniq 订单流去重叠、
  meta 元标签、2022-23 点差为 0 直接当 0 成本等均为已证伪路线。

## 五、恢复与复现

```bash
# 环境：Python 3.12 + lightgbm/xgboost/pandas/numpy；数据在 data/m1/
# 1. 复现冠军 v3bal_ens 的实盘推理
cd pipelines/v2_champion_v3bal_ens && python predict_live.py          # 需最近 16000 根 M1
# 2. 复现走查（约数小时，2核CPU）
python run_all.py                                                     # 详见 README.md
# 3. scratch 三段式（含 ② Regime×δ 消融、① 相关性诊断）
cd ../scratch_threestage && python stage1_clean.py && python stage1_swing.py \
  && python stage2_factors.py && python stage3_train.py
# 4. v4 路线
cd ../v4_m1_ignition && cat summary.md
```

## 六、模型演进时间线

| 日期 | 事件 |
|---|---|
| 2026-09-05 | v2 管线建成：ATR 自适应障碍 + purged WF；legacy+ATR 基线 +$1,281/Sharpe 1.13 |
| 2026-09-08 | v3 特征工程（34因子）+ 盈亏比几何标定：PLR 最高 5.09（胜率 18.4%） |
| 2026-09-14 | 平衡几何 v3bal（32.2%/2.40）→ 三张牌对比 → **v3bal_ens 冠军**（+$1,113.5/Sharpe 1.74，CI 首次不含0） |
| 2026-09-27 | m1sc/m1sc_ad/scalp1/geo4 实验 + v4 "M1点火-延续" 从零路线（-$321 证伪） |
| 2026-09-28 | scratch 三段式从零管线 s01-s08（OOS +$870 但 CI 含0，不显著） |
| 2026-09-29 | ② Regime 门控 × δ 收紧消融（2024H1）：模型无资格烧 OOS，瓶颈在核心方向信号 |
| 2026-09-29 | **终局判定：从零路线终止，v3bal_ens 仍为冠军；全量归档至本仓库** |
