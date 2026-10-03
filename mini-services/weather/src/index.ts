/**
 * index.ts — UC4 跨资产天气预警器 · 常驻后台计算服务 (port 3005)
 * ====================================================================
 * 用户指令: "UC4目前来说就是做个'天气预警器'比较合适吧？有空做个脚本常驻后台计算"
 *
 * 启动: 回放三资产全历史 (R2 冠军储层引擎 TS 移植) → 构建天气状态与预警
 * 常驻: 每 30s 轮询数据文件 — 文件增长即视为实时行情追加, 增量喂入引擎
 *       (生产环境把 CSV 路径换成实时落盘的行情文件即得真正的常驻预警)
 * 暴露: HTTP /state /history /health + WebSocket 推送 (Bun 原生)
 *
 * 用法: bun run dev          (热重载)
 *       bun run verify       (回放后打印 IC 对照研究档案)
 */
import { WeatherStation, pearson, decileCalibration } from './weather'
import { M1ToM15Pipeline, replayFile } from './pipeline'
import type { M15Bar } from './pipeline'

const PORT = 3005
const VERIFY = process.argv.includes('--verify')
const POLL_MS = 30_000

interface Paths { xau: string; dxy: string; xag: string }
const paths = (await Bun.file(new URL('./data-paths.json', import.meta.url).pathname).json()) as Paths

// ---------------- 全局状态 ----------------
const station = new WeatherStation()
const pipes = { xau: new M1ToM15Pipeline(), dxy: new M1ToM15Pipeline(), xag: new M1ToM15Pipeline() }
const fileBytes: Record<string, number> = { xau: 0, dxy: 0, xag: 0 }
const tailBuf: Record<string, string> = { xau: '', dxy: '', xag: '' }
let status: 'warming' | 'ready' = 'warming'
let progress = ''
const t0 = Date.now()
const clients = new Set<WebSocket>()

function broadcast(obj: unknown) {
  const s = JSON.stringify(obj)
  for (const ws of clients) if (ws.readyState === 1) ws.send(s)
}

// ---------------- 回放与合并 ----------------
async function replayAll() {
  const loadStart = Date.now()
  for (const k of ['dxy', 'xag', 'xau'] as const) {
    progress = `loading ${k}`
    const { rows, bytes } = await replayFile(paths[k], pipes[k])
    fileBytes[k] = bytes
    console.log(`[weather] ${k}: ${rows.toLocaleString()} M1 rows -> ${pipes[k].bars.length.toLocaleString()} M15 bars (${((Date.now() - loadStart) / 1000).toFixed(0)}s)`)
  }
  // 合并槽
  const barsOf = { xau: pipes.xau.bars, dxy: pipes.dxy.bars, xag: pipes.xag.bars }
  const idx = { xau: 0, dxy: 0, xag: 0 }
  const stepStart = Date.now()
  let slots = 0
  for (;;) {
    const s = Math.min(
      idx.xau < barsOf.xau.length ? barsOf.xau[idx.xau].slot : Infinity,
      idx.dxy < barsOf.dxy.length ? barsOf.dxy[idx.dxy].slot : Infinity,
      idx.xag < barsOf.xag.length ? barsOf.xag[idx.xag].slot : Infinity,
    )
    if (!Number.isFinite(s)) break
    const g: { xau?: M15Bar; dxy?: M15Bar; xag?: M15Bar } = {}
    if (idx.xau < barsOf.xau.length && barsOf.xau[idx.xau].slot === s) { g.xau = barsOf.xau[idx.xau]; idx.xau++ }
    if (idx.dxy < barsOf.dxy.length && barsOf.dxy[idx.dxy].slot === s) { g.dxy = barsOf.dxy[idx.dxy]; idx.dxy++ }
    if (idx.xag < barsOf.xag.length && barsOf.xag[idx.xag].slot === s) { g.xag = barsOf.xag[idx.xag]; idx.xag++ }
    station.stepSlot(s, g)
    slots++
    if (slots % 20000 === 0) console.log(`[weather] merge ${slots.toLocaleString()} slots...`)
  }
  const secs = (Date.now() - stepStart) / 1000
  const perBar = ((Date.now() - stepStart) * 1000 / Math.max(slots, 1)).toFixed(3)
  console.log(`[weather] replay done: ${slots.toLocaleString()} slots in ${secs.toFixed(0)}s (~${perBar}µs/slot)`)
  station.pushAlert({
    t: station.currentState()?.t ?? '', kind: 'system', asset: 'SYS',
    msg: `回放完成: ${slots.toLocaleString()} 个15分钟槽 · 引擎就绪 · 进入常驻轮询(30s)`,
  })
  status = 'ready'
  broadcast({ type: 'ready', state: fullState() })
}

// ---------------- 增量尾部轮询 ----------------
async function pollTail() {
  if (status !== 'ready') return
  let newSlots = 0
  for (const k of ['dxy', 'xag', 'xau'] as const) {
    const f = Bun.file(paths[k])
    const size = f.size
    if (size <= fileBytes[k]) continue
    const chunk = await f.slice(fileBytes[k], size).text()
    fileBytes[k] = size
    tailBuf[k] += chunk
    const lines = tailBuf[k].split('\n')
    tailBuf[k] = lines.pop() ?? ''
    const pipe = pipes[k]
    for (const line of lines) {
      const r = parse(line)
      if (r) pipe.feed(r.tMin, r.open, r.high, r.low, r.close, r.tickvol)
    }
  }
  // 处理新完结的槽 (从 station.lastSlot+1 起)
  const barsOf = { xau: pipes.xau.bars, dxy: pipes.dxy.bars, xag: pipes.xag.bars }
  const idx = { xau: 0, dxy: 0, xag: 0 }
  // 从各自数组找 > lastSlot 的起点 (尾部小扫描)
  for (const k of ['xau', 'dxy', 'xag'] as const) {
    const arr = barsOf[k]
    let i = arr.length - 1
    while (i >= 0 && arr[i].slot > station.lastSlot) i--
    idx[k] = i + 1
  }
  for (;;) {
    const s = Math.min(
      idx.xau < barsOf.xau.length ? barsOf.xau[idx.xau].slot : Infinity,
      idx.dxy < barsOf.dxy.length ? barsOf.dxy[idx.dxy].slot : Infinity,
      idx.xag < barsOf.xag.length ? barsOf.xag[idx.xag].slot : Infinity,
    )
    if (!Number.isFinite(s)) break
    const g: { xau?: M15Bar; dxy?: M15Bar; xag?: M15Bar } = {}
    if (idx.xau < barsOf.xau.length && barsOf.xau[idx.xau].slot === s) { g.xau = barsOf.xau[idx.xau]; idx.xau++ }
    if (idx.dxy < barsOf.dxy.length && barsOf.dxy[idx.dxy].slot === s) { g.dxy = barsOf.dxy[idx.dxy]; idx.dxy++ }
    if (idx.xag < barsOf.xag.length && barsOf.xag[idx.xag].slot === s) { g.xag = barsOf.xag[idx.xag]; idx.xag++ }
    const st = station.stepSlot(s, g)
    newSlots++
    broadcast({ type: 'bar', state: st })
  }
  if (newSlots > 0) console.log(`[weather] live-tail: +${newSlots} slots`)
}

function parse(line: string): { tMin: number; open: number; high: number; low: number; close: number; tickvol: number } | null {
  if (line.length < 20 || line.startsWith('<')) return null
  const t1 = line.indexOf('\t')
  if (t1 < 0) return null
  const dateStr = line.slice(0, t1)
  const rest = line.slice(t1 + 1)
  const t2 = rest.indexOf('\t')
  const timeStr = rest.slice(0, t2)
  const cols = rest.slice(t2 + 1).split('\t')
  if (cols.length < 5) return null
  const y = +dateStr.slice(0, 4), mo = +dateStr.slice(5, 7), d = +dateStr.slice(8, 10)
  const h = +timeStr.slice(0, 2), mi = +timeStr.slice(3, 5)
  const open = parseFloat(cols[0]), high = parseFloat(cols[1]),
        low = parseFloat(cols[2]), close = parseFloat(cols[3]), tickvol = parseFloat(cols[4])
  if (!Number.isFinite(open) || !Number.isFinite(close)) return null
  return { tMin: Math.floor(Date.UTC(y, mo - 1, d, h, mi) / 60000), open, high, low, close, tickvol }
}

// ---------------- HTTP/WS ----------------
function fullState() {
  const cur = station.currentState()
  const ic = {
    xau_1h: +pearson(station.xau.icPairs).toFixed(4),
    xau_4h: +pearson(station.xau4h.icPairs).toFixed(4),
    dxy: +pearson(station.dxy.icPairs).toFixed(4),
    xag: +pearson(station.xag.icPairs).toFixed(4),
  }
  return {
    status,
    progress,
    meta: {
      generated: '2026-10-03',
      engine: 'R2 冠军储层引擎 TS 移植 (nres50 · λ0.9999 · log目标 · 8维 · seed42 矩阵同源)',
      research_ref: { xau_1h: 0.4907, xau_4h: 0.5188, dxy: 0.4915, xag: 0.5185 },
      ic_note: '嵌合0.5363经严格因果lag消融证伪(asof未来信息伪影, lag60→0.587泄漏曲线); 跨资产真实价值=白银传输预警(因果干净IC0.55)+三资产能量监控',
      warm_bars: 2000,
      bars: { xau: station.xau.engineBars, xau4h: station.xau4h.engineBars, dxy: station.dxy.engineBars, xag: station.xag.engineBars },
      span: { first: station.firstT, last: cur?.t },
    },
    current: cur,
    history: station.history.slice(-240),
    alerts: station.alerts.slice(-60).reverse(),
    ic,
    alertCounts: station.alertCounts,
    decile: decileCalibration(station.xau4h.icPairs),
    uptime_s: Math.round((Date.now() - t0) / 1000),
  }
}

const server = Bun.serve({
  port: PORT,
  fetch(req, srv) {
    if (srv.upgrade(req)) return
    const url = new URL(req.url)
    const p = url.pathname
    const headers = { 'content-type': 'application/json; charset=utf-8' } as const
    if (p === '/health') {
      return new Response(JSON.stringify({ ok: true, status, progress, uptime_s: Math.round((Date.now() - t0) / 1000) }), { headers })
    }
    if (p === '/history') {
      const n = Math.min(Math.max(parseInt(url.searchParams.get('n') ?? '2880', 100), 10), 2880)
      return new Response(JSON.stringify({ history: station.history.slice(-n) }), { headers })
    }
    return new Response(JSON.stringify(fullState()), { headers })
  },
  websocket: {
    open(ws) {
      clients.add(ws)
      ws.send(JSON.stringify({ type: 'init', state: fullState() }))
    },
    close(ws) { clients.delete(ws) },
    message(ws, msg) {
      const s = String(msg)
      if (s === 'ping') ws.send('pong')
    },
  },
})

console.log(`[weather] UC4 天气预警站 listening on :${PORT} (status=${status})`)

// ---------------- 主流程 ----------------
await replayAll()
if (VERIFY) {
  console.log('==== IC 验证 (vs 研究档案) ====')
  console.log(`  XAU 1h   : ${pearson(station.xau.icPairs).toFixed(4)}  (python 0.4907)`)
  console.log(`  XAU 4h   : ${pearson(station.xau4h.icPairs).toFixed(4)}  (python h16 训练 0.5188)`)
  console.log(`  DXY      : ${pearson(station.dxy.icPairs).toFixed(4)}  (python 0.4915)`)
  console.log(`  XAG      : ${pearson(station.xag.icPairs).toFixed(4)}  (python 0.5185)`)
  console.log('  XAU 4h 十分位校准:', JSON.stringify(decileCalibration(station.xau4h.icPairs)))
  console.log(`  预警总数: ${station.alerts.length} (见 /state alerts)`)
}
setInterval(pollTail, POLL_MS)
