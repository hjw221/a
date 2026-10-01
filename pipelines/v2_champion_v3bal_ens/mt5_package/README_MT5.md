# XauV3BalEnsEA — v3bal_ens 的 MT5 复现包

把 Python 管线（`xauusd_ml_v2`）里的生产变体 **v3bal_ens** 完整搬进 MT5：
6成员×2方向 LightGBM/XGBoost AUC 加权集成 → **ONNX 端内推理（无需 Python）**，
交易口径 1:1 对标 Python 走查回测（ATR 自适应三重障碍 / 6h 超时 / 10 根 M1 冷却 / 单持仓）。

> **v1.01（2026-09-15，基于 MT5 build 5833 实测反馈修复）**：
> ① v1.00 的 `OnnxSetInputShape/OutputShape` 少传 index 参数，**无法编译**（你跑的旧 ex5 不是当前源码）；
> ② v1.00 用 `FileReadString` 读 Python 写出的 `\n` 行尾 CSV 不可靠，**24 折索引被读成"1折 ( ~ )"空表 → 全部月份"不在走查范围"→ 零交易**；v1.01 已改二进制解析（兼容 `\n`/`\r\n`/`\r`/BOM）+ 硬校验（坏表直接 INIT_FAILED 并打印文件头 hex 诊断）；
> ③ fold_00~09 目录名补零（v1.00 会拼成 `fold_0` 找不到，2024-08~2025-05 十个月静默 HOLD）。
> **请务必删旧 ex5、用本包 mq5 重编译**；models 目录建议整目录覆盖重拷（CSV 已全部转 `\r\n`）。

## 0. 你要对比的基准（Python 真实 OOS 结果，24 个月 2024-08~2026-07）

| 指标 | 数值 |
|---|---|
| 交易数 | **1,611** |
| 胜率 | 31.6% |
| 盈亏比 PLR | 2.57 |
| 总 PnL（0.01 手） | **+$1,113.5** |
| Sharpe | 1.74（CI95 不含 0） |
| maxDD | $300 |
| 最长连亏 | 18 笔 |

逐笔明细：`reference/trades_v3bal_ens_python.csv`（1,611 行）。

若你的经纪商 M5 历史不够 2024-08 回溯（Exness 实测 XAUUSDc 的 M5 只从 2025-01-01 起），用下表对应子区间对照（同样出自 1,611 笔真实日志，见 `reference/mt5_subset_reference.csv`）：

| 可测区间 | 笔数 | 胜率 | PLR | 总PnL(0.01手) |
|---|---|---|---|---|
| 全 24 个月 2024-08~2026-07 | 1,611 | 31.6% | 2.57 | +$1,113.5 |
| 19 个月 2025-01~2026-07 | 1,364 | 31.5% | 2.59 | +$1,086.5 |
| 7 个月 2026-01~2026-07 | 633 | 32.2% | 2.60 | +$817.2 |
| 2 个月 2026-06~2026-07（快速验证） | 185 | 28.1% | 2.42 | **-$47.8** |

注意 2026-06~07 在 Python 侧是**亏损段**（2026-06 单月 -$135.7）——拿它做链路验证反而更苛刻：EA 应复现出相近的亏损形态（交易数/胜率/PLR 同向），而不是"碰巧赚钱"。

## 1. 安装

1. **EA**：把 `XauV3BalEnsEA.mq5` 放进终端 `MQL5\Experts\XauV3Ens\`，MetaEditor 打开并编译（F7）。需要 MT5 build ≥ 3980（2023 年 4 月后任意版本，ONNX 支持）。
2. **模型**：把本包 `models` 整个目录复制到 **`终端\Common\Files\XauV3Ens\`**（注意：是 Common，不是 MQL5\Files）。**从 v1.00 包升级的请整目录覆盖重拷**（CSV 行尾已修）。最终结构：
   ```
   Common\Files\XauV3Ens\
     wf\wf_index.csv            ← 24折索引(月份→阈值)
     wf\fold_00..fold_23\       ← 每折: members.csv + selftest.csv + 12个.onnx
     prod\index.csv + members.csv + selftest.csv + 10个.onnx
   ```
   （EA 也支持放在 `MQL5\Files\XauV3Ens\` 作后备。）
3. 共 233 个 ONNX 模型约 57MB；只装 wf 或只装 prod 均可（WF 模式不需要 prod，反之亦然）。

## 2. 回测设置（对标 walk-forward 结果用）

| 项 | 值 | 原因 |
|---|---|---|
| 品种 | XAUUSD（三位小数报价，point=0.001） | 训练数据口径；两位报价经纪商成本语义不同，EA 启动会打警告 |
| 周期 | **M5** | 信号周期 |
| 日期 | **2024.08.01 ~ 2026.08.01**；M5 历史不够回溯时改 **2025.01.01 ~ 2026.08.01**，用 19 个月子区间对照（见上表） | 24 个 OOS 月；Exness 实测 XAUUSDc M5 历史仅从 2025-01 起，2024-08~12 可能无法回测。首月需 ~11 交易日 M5 预热，若测试起点=历史起点则该月前 ~12 天不评估（Python 侧仅 20 笔/-$0.9，影响可忽略） |
| 撮合模式 | **每笔分时 基于 真实报价（Every tick based on real ticks）**；快速链路验证可先用 **1分钟 OHLC**（快约一个量级，障碍触发粒度近似） | 障碍判定保真 |
| 入金/币种 | USD 账户；**USC 美分账户可用**：合约=100oz 时报表金额 **÷100 ≈ Python 美元口径**（0.01手=1oz，$1 波动=$1=100USC）。EA 启动日志会打印账户币种/合约规模辅助确认 | PnL 口径 |
| EA 参数 | 默认（MODE_WF + 虚拟障碍 + 0.01 手） | 对齐 Python |
| 佣金 | 0 | Python 回测未计佣金（有佣金则自行在对比时扣） |

- 首月 2024-08 需要 ~11 个交易日的 M5 历史预热（回测器会自动取测试起点前的历史，无需手调）。
- 若你的经纪商 XAUUSD 历史与训练数据（另一经纪商导出的 M1）不同源，**这是最大的结果差异来源**——价格路径、点差、跳价节奏都不同。
- 服务器时间需为 EET（UTC+2 冬 / UTC+3 夏，即"纽约收盘对齐"型）。时间特征（hour_sin、伦敦/纽约时段）按服务器时间计算，与训练 CSV 同口径；若你的经纪商服务器时区不同，时段特征会整体偏移。

## 3. 运行后怎么核对（三级自检，全部已内置）

1. **模型自检（自动）**：每次按月切换折时，EA 用该折 `selftest.csv` 里存的"真实特征行 + 期望概率"跑一遍 ONNX 集成，日志打印
   `[self-test] pL=…(期望…) => PASS`。**24 折全部 PASS 才说明模型链路无损**；FAIL 则该折不交易。
2. **特征对拍（可选，强烈建议跑一次）**：参数 `InpDumpFeatures=true`，EA 把每根收盘 M5 的 34 特征写到 `Common\Files\XauV3Ens\features_EA.csv`。把回测区间设为 2026-07-17 当天，导出后与 `reference/features_reference_python.csv`（Python 侧真值，末 120 根 M5）逐行比对：
   ```
   diff features_EA.csv reference/features_reference_python.csv   (或用Excel/表格软件对比)
   ```
   时间戳相同的行、34 列数值应在 1e-6 量级一致（double 舍入差除外）。任何一列系统性不一致 = 特征移植 bug，请反馈。
3. **逐笔对拍（最终验收）**：回测跑完后看 `Common\Files\XauV3Ens\trades_EA.csv`，与 `reference/trades_v3bal_ens_python.csv` 对比 `signal_time / dir / outcome`。**同一数据源下应高度重合**；不同经纪商数据会自然漂移，方向与节奏一致即可。删除旧日志文件再跑，避免追加混淆。

## 4. 两种模式

- **MODE_WF（默认，回测对标用）**：按信号 K线的日历月自动切换 24 折模型 + 每折独立阈值（与走查验证完全一致）。2024-08~2026-07 之外的月份不交易（日志提示）。**这就是产生上表 1,611 笔结果的原始口径。**
- **MODE_PROD（前向/实盘用）**：单一生产集成（2023-08~2026-07 三十六个月训练，阈值 long 0.3397 / short 0.2678）。⚠️ 不要拿它在 2024-2026 上回测跟 OOS 表对比——那段数据在它的训练窗内，属于样本内，结果必然虚高。

## 5. 已验证的等价性（本机真实跑出来的，不是声称）

- **24 折模型重训逐笔一致**：重跑 v3bal_ens 走查，成员 AUC / 权重 / 阈值 / 逐笔交易日志与原 checkpoint **零差异**，合计 1,611 笔 = 已报告数（`models/dump_report_*.json`）。
- **233 个 ONNX 模型数值对拍**：onnxruntime 输出 vs 原生 LightGBM/XGBoost 文件重载预测，**233/233 PASS（容差 1e-6，实测 ~3e-8）**，含 NaN 缺失分支（`models/onnx_verification.json`）。
- **特征公式 1:1 移植**：34 特征逐条对照 `features_v3.py`（含 pandas 特有语义：样本 std、平盘沿用方向的 run_len、等值取最新的极值距离、2880 根布林带宽中位数）。对拍方法见上节第 2 条。

## 6. 预期差异（诚实清单——MT5 结果不会和 Python 一模一样）

| 差异源 | 方向 | 说明 |
|---|---|---|
| 障碍检查粒度 | EA 略有利 | Python 用 M1 柱高低点+同柱 SL 优先（悲观）；MT5 真实 tick 逐笔触发，更精细、少悲观偏差 |
| 点差 | 不定 | Python 用训练数据当月中位点差逐笔扣费；MT5 用真实点差（你的经纪商点差水平决定） |
| 数据源 | **最大噪声** | 你的 MT5 历史与训练 CSV 不同经纪商 → 价格/跳价/成交节奏不同 |
| 隔夜利息 | EA 略不利 | Python 未计 swap；MT5 按品种 swap 计（持仓≤6h，仅跨三倍 swap 日有感） |
| 滑点 | EA 略不利 | Python 假设 0 滑点；真实 tick 撮合会有微滑点 |
| spread_rel 特征 | 微小 | MT5 M5 柱的 spread 字段 vs 训练用 5 根 M1 点差中位数 |
| 超时平仓时点 | 1 tick | Python 取第 360 根 M1 收盘价；EA 在其后首个 tick 市价平 |

**判定建议**：先看 24 折 self-test 全 PASS + 特征对拍通过（链路正确性），再看交易数/胜率/PLR 是否在 ±10-15% 内同向（数据源差异的正常范围）。PnL 绝对值对点差和 swap 敏感，Sharpe/胜率/PLR 更稳。

## 7. 文件清单

```
mt5_package/
  XauV3BalEnsEA.mq5              ← EA本体 (编译它)
  models/
    wf/  (24折, 223个onnx)  prod/ (10个onnx)
    wf_manifest.json / prod_manifest.json   ← 机器可读manifest(人看)
    onnx_verification.json     ← 233模型对拍报告
    dump_report_*.json         ← 24折重训一致性报告
  reference/
    trades_v3bal_ens_python.csv  ← Python 1611笔逐笔明细(对拍基准)
    features_reference_python.csv ← 特征真值120行(对拍基准)
  tools/                        ← 复现脚本(导出ONNX/重跑dump等, 在原环境用)
  README_MT5.md
```

## 8. 故障排查

- **编译报 OnnxSetInputShape 参数不匹配**：v1.01 已改为正确三参数签名 `OnnxSetInputShape(handle, 0, shape)` / `OnnxSetOutputShape(handle, 0, shape)`（v1.00 少传 index 参数编不过；若你的 build 仍报错把日志发回来）。
- **日志出现 `WF索引: 1折 ( ~ )` 或全部月份"不在走查范围"零交易**：v1.00 的 CSV 行尾 bug（Python 写 `\n` 行尾，MQL5 FileReadString 按 `\r\n` 设计解析不可靠，实测 24 折被读成 1 条空记录）。v1.01 已改二进制解析 + 硬校验：表坏了会直接 INIT_FAILED 并打印文件头 hex，不会无声空跑。本包 CSV 已全部转 `\r\n`。
- **日志 `文件缺失`**：models 目录没放对，检查 `Common\Files\XauV3Ens\wf\wf_index.csv` 是否存在。
- **self-test FAIL**：OnnxRun 参数顺序探测异常 → 设 `InpRunOrderForce=1`（输出在前）或 `2`（输入在前）再跑。
- **一直 warmup / 无交易**：确认 M5 图表品种正确、历史已下载；WF 模式日期在 2024-08~2026-07 内。
- **打印 `point≠0.001 警告`**：换三位报价的 XAUUSD 品种（多数 ECN 是三位），否则点差成本语义偏离训练假设。

## 9. 风险声明（与主 README 一致，不因换平台而消失）

前 2 个月占 24 个月利润的 72%、空头贡献 73%、最长连亏 18 笔、maxDD $300（0.01 手）。CI95 虽不含 0，但这是单一品种单一 regime 的历史统计，不是未来承诺。0.1 手对应 ~$11,135 / maxDD ~$3,000，建议账户 ≥ 3×maxDD。
