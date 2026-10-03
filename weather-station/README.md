# weather-station · UC4 跨资产天气预警站

> 常驻后台计算服务：四台 R2 冠军储层引擎（ESN + 指数遗忘 RLS + 流式 GMM）
> 流式计算三资产波动"天气"，实时推送预警。打包自研究项目沙箱
> （原部署位 `mini-services/weather`，port 3005），本目录为**独立可部署版**。

## 一、它是什么

- **引擎**：`reservoir_engine2.py`（Python 研究版）的 TypeScript 逐行忠实移植，
  `champion-params.json` 为冠军配置（nres50 · λ=0.9999 · log 目标 · 8 维输入 ·
  seed42 矩阵，按 Python RNG 消耗顺序同源导出）。
- **四台引擎**：XAU-1h / XAU-4h（h16 训练）/ DXY-1h / XAG-1h，各自预测未来方差比。
- **预警体系**：
  - 1h/4h 天气等级 ☀️ calm / ⛅ normal / 🌬️ windy / ⛈️ storm（施密特滞回去抖）
  - UC3 断路器（d2>25 → 未来 1h 方差 ≈5.9× 中位，俄乌夜 2022-02-24 复核过）
  - 白银传输预警（R_xag≥2.0 → 白银领先黄金 ~120min，传输 IC 0.55 因果干净）
- **运行形态**：启动回放三资产 4.5 年 M1（~11 万个 M15 槽，约 6~10s）构建引擎状态
  → 常驻每 30s 轮询数据文件，文件增长即视为实时行情追加，增量喂入。
  生产环境把 `src/data-paths.json` 指向实时落盘的行情文件，即得真正的常驻预警。

## 二、移植保真度（TS vs Python 研究引擎，回放 4.5 年全历史）

| 引擎 | 视界 | IC (TS) | IC (Python) |
|---|---|---:|---:|
| XAU-1h | 未来 1h 方差比 | 0.4917 | 0.4907 |
| XAU-4h | 未来 4h 方差比 | 0.5205 | 0.5188 |
| DXY-1h | 1h | 0.4931 | 0.4915 |
| XAG-1h | 1h | 0.5192 | 0.5185 |

十分位校准 0.454→2.624 单调（高预测 → 高实现，无校准断裂）。
回放速度 ~50µs/槽（113k 槽 ≈6s，Bun 单核）。

## 三、快速开始

```bash
# 0) 需要 Bun (>=1.0)。三份 M1 CSV 放入 service/data/（格式与获取方式见 service/data/README.md）
cd weather-station/service

# 1) 装依赖（零第三方依赖，仅类型定义）+ 启动
bun install
bun run dev        # 热重载, port 3005 — 回放 ~10s 后进入常驻轮询

# 2) 移植保真度自检（可选）：回放后打印 IC 对照研究档案
bun run verify
```

启动后：

- `GET /state` — 当前天气 + 历史(240) + 预警(60) + IC + 十分位校准
- `GET /history?n=2880` — 深历史
- `GET /health` — 存活/预热进度
- `WebSocket ws://host:3005/` — 连接即推 `init`，此后每槽推 `bar`、就绪推 `ready`

## 四、目录结构

```
weather-station/
├── README.md                  # 本文件
├── service/                   # 常驻计算服务 (Bun, port 3005, 零第三方运行时依赖)
│   ├── package.json           #   bun run dev | start | verify
│   ├── tsconfig.json
│   ├── README.md              #   引擎内部结构与预警阈值细节
│   ├── data/README.md         #   ★ 数据文件格式/获取方式（CSV 不入库，自备）
│   └── src/
│       ├── index.ts           #   入口: 回放→常驻轮询→HTTP/WS
│       ├── pipeline.ts        #   M1→M15 聚合 + 回放/增量喂入
│       ├── features.ts        #   8 维输入特征 (log-ret/ATR比/动量/时段sin-cos…)
│       ├── engine.ts          #   储层+RLS+GMM 引擎 (TS 移植, 双bug已修)
│       ├── weather.ts         #   天气站: 等级滞回/断路器/白银传输/IC统计
│       ├── champion-params.json  # 冠军配置 (seed42 矩阵, 37KB)
│       └── data-paths.json    #   三资产数据路径 (默认 data/*.csv 相对路径)
└── frontend/
    └── weather-tab.tsx        # 实时仪表盘 tab (React 客户端组件, 见 frontend/README.md)
```

## 五、接实时行情

服务不关心数据从哪来，只看文件是否变长：

1. **MT5 落盘**：EA 每 M1 收线把 K 线 append 到 CSV（格式见 `service/data/README.md`）
   → 30s 内自动增量入引擎；
2. **任意采集器**：python/beaver 等落盘同格式文件即可；
3. **改轮询周期**：`src/index.ts` 顶部 `POLL_MS = 30_000`。

## 六、前端仪表盘

`frontend/weather-tab.tsx` 是本项目主面板的「气象站」tab（WebSocket 经网关
`/?XTransformPort=3005` 接入 + 5s HTTP 轮询兜底）。独立部署时把组件拷入任意
React 项目，依赖 framer-motion / lucide-react / shadcn-ui（card·badge·scroll-area·separator），
并把 WebSocket URL 从网关形式改为直连 `ws://host:3005/`（组件内常量一处）。
细节见 `frontend/README.md`。

## 七、诚实条款（研究级）

- R2 档案曾记录「嵌合引擎 IC 0.491→0.5363」，经严格因果 lag 消融**证伪**为
  asof 未来信息伪影（lag30→0.536 / lag60→0.587 泄漏曲线；严格因果 0.4915≈单资产）。
  跨资产的真实价值 = 白银传输预警（因果干净）+ 三资产能量监控——本站即按此部署。
- IC 全部为回放口径（4.5 年历史 OOS 序贯计算，非样本外新数据）；
  引擎为方差/能量预测器，**不构成交易信号**，用于风控与仓位"呼吸"参考。
- 预警阈值（天气分档 / d2>25 断路器 / R_xag≥2.0 传输）来自研究期统计
  （详见 server 分支 `remote-ops-record/r2_20261003/`），换经纪商/品种需重标定。
