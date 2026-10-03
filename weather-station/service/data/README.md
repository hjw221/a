# data/ — 三资产 M1 行情文件（不入库，自备）

服务启动时回放这三个文件构建引擎状态，常驻后每 30s 轮询文件尾部增量。
把三份 CSV 放到本目录（`weather-station/service/data/`），或编辑
`src/data-paths.json` 指向任意绝对/相对路径（相对路径按 `bun` 运行时的工作目录解析，
推荐在 `service/` 目录下启动）。

## 需要的文件

| 键 | 文件名 | 内容 | 本仓库沙箱实测尺寸 |
|---|---|---|---|
| xau | `XAUUSD_M1.csv` | 黄金 M1，2022-01 ~ 2026-07 | ~106 MB |
| dxy | `DXYm_M1.csv` | 美元指数 M1（Exness 符号 DXYm） | ~100 MB |
| xag | `XAGUSDc_M1.csv` | 白银 M1（Exness 符号 XAGUSDc） | ~89 MB |

## CSV 格式（MetaTrader 5 "导出分隔符 Tab" 标准格式）

三资产统一格式：首行表头 `<DATE>	<TIME>	<OPEN>	<HIGH>	<LOW>	<CLOSE>	<TICKVOL>	<VOL>	<SPREAD>`，
其后每分钟一行，Tab 分隔，日期 `2022.01.02`、时间 `23:05:00`：

```
2022.01.02	23:05:00	1830.615	1830.684	1829.688	1829.763	1141	0	0
```

服务只解析 OPEN/HIGH/LOW/CLOSE/TICKVOL 前五列，多余列自动忽略；
CRLF/LF 行尾均可；不完整尾行自动缓冲等待补齐（增量轮询安全）。

## 获取方式

1. **MT5 直接导出**：终端 → 图表（M1 周期）→ 文件 → 保存为…（分隔符选 Tab），
   或用脚本批量导出（符号：XAUUSD、DXYm、XAGUSDc，经纪商不同符号后缀可能不同）。
2. **本项目的下载通道**：研究过程中 DXYm/XAGUSDc 来自 filester.me 公开盘
   （POST `/v2/api/public/download` 换 CDN token 下载），详见研究档案 `remote-ops-record/`
   （server 分支）。黄金 M1 亦可从同源获取后按上方格式合并为单文件。

> 没有跨资产数据也能跑：`src/index.ts` 的合并槽逻辑允许某资产缺行（`g.xau` 等可选），
> 但至少要有 XAU 才能出天气等级；三资产齐全才有白银传输预警与全能量监控。
