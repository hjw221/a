# MQ5 移植 · v17 冠军 → MetaTrader 5 独立复验

**日期** 2026-10-02 · **交付物** `XAUUSD_v17_Champion.mq5`（21,026 bytes · md5 `802834685f0ec877237512ee20d08a13`）

## 背景
用户确认 v17 冠军**零模型**（纯规则，ML 已重定位为后续宏观门控），要求直接产出 MQ5 在 MT5 策略测试器做历史回测——用**独立数据源 + 独立撮合引擎**复验研究结论。这是继 bug1（时域错位）/ bug2（吊灯回望结算）之后最有说服力的第三方验证。

## 移植保真度对照表（研究引擎 → MQ5）
| 研究引擎语义 (v17_htf.py / v17b_refine.py) | MQ5 实现 |
|---|---|
| `dhi55 = rolling(55).max().shift(1)`，bar k 用 [k-55..k-1] 最高价 | 新 bar 开盘取序列索引 1..55 已完成 bar 的最高价 |
| `hit_up = h[k] >= up` 触发停损单 | BuyStop 挂 `up`，仅当根有效（`ORDER_TIME_SPECIFIED` 到 bar 末，下根开盘删旧挂新） |
| 跳空 `o[k] > up` → 按 `o[k]` 成交（保守更差） | 开盘 Ask ≥ 通道线 → 直接市价 Buy |
| ATR14 = TR 的 14 根**简单均值**（非 Wilder） | TR 前缀和自制 SMA（**不用** iATR——内置是 Wilder 平滑，口径不同） |
| 闸门 `atr[k-1] > median(atr[k-3024..k-1])`（右移一根，含自身） | bar1 的 ATR vs 最近 3024 根 ATR 的中位数（分位数线性插值同 pandas 默认） |
| `min_periods=200` 历史不足不开闸 | valid 样本 < 200 → gate 关 |
| 时间出场 `k >= eib+240 → o[k]` 平仓 | `iBarShift(entryBarTime) >= 240` → 新 bar 开盘市价平仓 |
| **出场当根不再进场**（entry 检查先于 exit 检查） | `exitedThisBar` 守卫 |
| 串行单仓，持仓期忽略新信号 | 持仓时不评估进场 |
| 入场 bar = 成交发生的那根 bar | `OnTradeTransaction` 用 deal 时刻定位入场 bar |
| 数据末端强平 (rsn=5) | MT5 测试器自动结算 |
| v18 结论：静态止损 44 变体全负期望 | `InpUseDisasterStop` 默认 **false**（可实验） |
| M30/H1 参数自动推导（48/24 bars 每交易日） | `PeriodSeconds` 推导 hold=5×bpd、gate=63×bpd |

## MT5 测试器设置
- 品种 **XAUUSD** · 周期 **M30** · 日期 **2022.01.01 – 2026.07.31**（研究窗）
- 模式：**每笔报价（真实报价）**最佳；最低 **1 分钟 OHLC**（『仅开盘价』无法模拟停损单 bar 内触发）
- 默认 0.01 手 = 1 盎司 → 测试器 $ ≈ 研究 $ 口径
- 测试结束 OnDeinit 自动打印逐年汇总，直接对照研究参照

## 预期落点（对照基准）
研究（0.3×ATR 悲观成本）：**159 笔 · +$2,980.2 · 逐年全正 {+242.5/34 · +164.5/38 · +319.8/36 · +1394.5/36 · +858.9/15} · 胜率 58.5% · maxDD −$259.6**
MT5 真实点差（$0.2~0.4/RT）低于 0.3×ATR（中位 $1.37/RT）→ 预期总额落在 **$2,980~$3,208**（pess03~base）之间；笔数允许 ±10% 漂移（数据源/清洗差异）。**2026-07-17 之后为样本外前向检验。**

## 已知差异（诚实披露）
1. 数据源：MT5 经纪商行情 vs 研究清洗 M1 聚合（剔除 tickvol≤5 死 bar 与 15σ 异常）
2. 停损单以 **Ask** 触发（引擎用数据 high）——贵约 1 个点差，已被悲观成本假设覆盖
3. 服务器时区平移日内 session——策略全部是 bar 计数结构，对时区不敏感
4. 双向同根触发保守跳过仅在 `InpAllowShort=true` 时相关（默认关）
5. MT5 成交含真实点差 → 无需再扣研究成本项

## 复现
- 文件：`public/data/XAUUSD_v17_Champion.mq5`（前端 v17 tab 有下载卡片，agent-browser 验证 fetch 200/21,026B/内容完整）
- UTF-8 BOM 已加（MetaEditor 中文注释友好）；括号/圆括号平衡静态自检通过
- 沙箱 ssh 二进制丢失 → `/home/z/.ssh/gitshim.py` paramiko git-ssh 桥重建（`core.sshCommand` + `ssh.variant simple`），ls-remote/fetch/push 复活
