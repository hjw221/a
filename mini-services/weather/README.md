# weather · UC4 跨资产天气预警站 (常驻后台计算服务)

用户指令: "UC4目前来说就是做个'天气预警器'比较合适吧？有空做个脚本常驻后台计算"

## 是什么

四台 R2 冠军储层引擎 (ESN 50 神经元 + 指数遗忘 RLS λ=0.9999 + 流式 GMM, nres50/log/8维,
seed42 矩阵与 Python 研究引擎同源导出) 的 TypeScript 忠实移植, 常驻后台流式计算:

| 引擎 | 视界 | IC (TS 实现) | IC (Python 研究) |
|---|---|---|---|
| XAU-1h | 未来 1h 方差比 | 0.4917 | 0.4907 |
| XAU-4h (h16 训练) | 未来 4h 方差比 | 0.5205 | 0.5188 |
| DXY-1h | 1h | 0.4931 | 0.4915 |
| XAG-1h | 1h | 0.5192 | 0.5185 |

预警体系: 1h/4h 天气等级 (☀️⛅🌬️⛈️, 施密特滞回去抖) · UC3 断路器 (d2>25 → 未来1h方差≈5.9×中位)
· 白银传输预警 (R_xag≥2.0 → 白银领先黄金 ~120min, 传输 IC 0.55 因果干净)。

## 运行

```bash
cd mini-services/weather
bun run dev        # 热重载, port 3005 — 启动回放 ~10s 后进入常驻轮询
bun run verify     # 回放后打印 IC 对照研究档案 (移植保真度验证)
```

- 启动: 回放三资产 4.5 年 M1 历史 (~113k 个 M15 槽, ~6s) 构建引擎状态
- 常驻: 每 30s 轮询数据文件 (`src/data-paths.json`), 文件增长即视为实时行情追加, 增量喂入
  → 生产环境把路径换成实时落盘的行情文件即得真正的常驻预警
- 暴露: `GET /state` (当前天气+历史+预警+IC) · `GET /history?n=` · `GET /health` · WebSocket 推送

前端: 主面板「气象站 · UC4 天气预警器」tab (`src/components/weather-tab.tsx`),
浏览器经网关 `/?XTransformPort=3005` 连接 (WS + 5s 轮询兜底)。

## 诚实修正 (重要)

R2 档案的「嵌合引擎 IC 0.491→0.5363」经严格因果 lag 消融证伪为 asof 未来信息伪影
(lag30→0.536, lag60→0.587, lag120→0.539 泄漏曲线; 严格因果 0.4915≈单资产)。
跨资产的真实价值 = 白银传输预警 + 三资产能量监控 — 本站即按此部署。
另: XAU 专用 h16 训练引擎把 4h 预报 IC 提到 0.5188 (真实增益, 非跨资产)。

## 文件

- `src/engine.ts` — 储层引擎移植 (GMM/RLS 逐行对应 reservoir_engine2.py; 修复过两个移植 bug:
  cfg 解构 undefined + spawn 分支熵越界 — 均已验证)
- `src/pipeline.ts` — M1 CSV 流式清洗+M15 重聚合 (与 load_m15 同口径)
- `src/features.ts` — 8 维因果特征流 (含 h4/h16 双视界标签量)
- `src/weather.ts` — 预警站编排 (槽合并/等级/滞回/预警/IC 配对)
- `src/index.ts` — HTTP/WS 服务 + 回放 + 30s 尾部轮询
- `src/champion-params.json` — seed42 冠军矩阵 (Python numpy 导出, 每引擎独立 RNG 流)
