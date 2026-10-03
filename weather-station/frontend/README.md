# frontend/ — 实时仪表盘 tab

`weather-tab.tsx` — 「气象站 · UC4 天气预警器」React 客户端组件（本项目主面板第 6 tab）。

## 功能

- 四引擎卡片：XAU-1h / XAU-4h / DXY / XAG — 当前 R（方差比预测）、天气等级 emoji、
  GMM 簇 tag、能量条；
- 预警流（最近 60 条）：weather / outlook / breaker / silver / system 五类着色；
- IC 对照表（TS 实测 vs Python 研究档案）+ 十分位校准表；
- 连接状态机：connecting → live(WS) / polling(5s HTTP 兜底) / offline。

## 依赖

React 18+ · framer-motion · lucide-react · shadcn-ui（card / badge / scroll-area / separator）。

## 接入

1. 拷入你的 React/Next.js 项目（如 `src/components/weather-tab.tsx`），渲染 `<WeatherTab />`；
2. **本项目沙箱网关环境**（单端口暴露，经 Caddy 转发）已内置正确写法：
   WebSocket 连 `/?XTransformPort=3005`，HTTP 轮询 `/api/state?XTransformPort=3005` 同理——
   组件开头的常量即此形式，开箱即用；
3. **独立部署/直连环境**：把组件内 WS/HTTP URL 改为 `ws://host:3005/` 与
   `http://host:3005/state`（各一处常量）。

> 组件为纯展示层，无任何引擎逻辑；所有数字来自服务的 `/state` 与 WS 推送。
