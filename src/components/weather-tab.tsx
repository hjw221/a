'use client'

/**
 * weather-tab.tsx — UC4 跨资产天气预警站 (第6 tab · 实时)
 * 连接 mini-services/weather (port 3005): WebSocket 推送 + HTTP 轮询兜底
 * 四台储层引擎: XAU-1h / XAU-4h / DXY / XAG — IC 全部与研究引擎对齐 (±0.002)
 */
import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { Activity, CloudSun, FlaskConical, Radio, ShieldAlert, Thermometer } from 'lucide-react'

interface AssetState { R: number; skew: number; d2: number; level: number; tag: number; pct: number; K: number }
interface WeatherState { t: string; slot: number; xau: AssetState; xau4h: AssetState; dxy: AssetState; xag: AssetState }
interface Alert { t: string; kind: string; asset: string; msg: string; R?: number; d2?: number }
interface StationState {
  status: string; progress: string
  meta: {
    generated: string; engine: string
    research_ref: Record<string, number>
    ic_note: string
    warm_bars: number
    bars: Record<string, number>
    span: { first?: string; last?: string }
  }
  current: WeatherState | null
  history: WeatherState[]
  alerts: Alert[]
  ic: Record<string, number>
  alertCounts: Record<string, number>
  decile: { d: string; mean: number; n: number }[]
  uptime_s: number
}

const LEVELS = [
  { name: 'calm', emoji: '☀️', cls: 'text-sky-300', ring: 'border-sky-800/60 bg-sky-950/30' },
  { name: 'normal', emoji: '⛅', ring: 'border-zinc-700 bg-zinc-900/40', cls: 'text-zinc-200' },
  { name: 'windy', emoji: '🌬️', cls: 'text-amber-300', ring: 'border-amber-800/60 bg-amber-950/30' },
  { name: 'storm', emoji: '⛈️', cls: 'text-red-300', ring: 'border-red-800/60 bg-red-950/30' },
]
const KIND_CLS: Record<string, string> = {
  weather: 'text-sky-300', outlook: 'text-zinc-300', breaker: 'text-red-300', silver: 'text-amber-300', system: 'text-emerald-300',
}
const KIND_ICON: Record<string, React.ReactNode> = {
  weather: <CloudSun className="h-3 w-3" aria-hidden />, outlook: <CloudSun className="h-3 w-3" aria-hidden />,
  breaker: <ShieldAlert className="h-3 w-3" aria-hidden />, silver: <Activity className="h-3 w-3" aria-hidden />,
  system: <Radio className="h-3 w-3" aria-hidden />,
}

export function WeatherTab() {
  const [state, setState] = useState<StationState | null>(null)
  const [conn, setConn] = useState<'connecting' | 'live' | 'polling' | 'offline'>('connecting')
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    let poll: ReturnType<typeof setInterval> | null = null
    let closed = false
    const fetchState = async () => {
      try {
        const r = await fetch('/?XTransformPort=3005', { cache: 'no-store' })
        if (!r.ok) throw new Error()
        setState(await r.json())
        setConn(c => (c === 'live' ? c : 'polling'))
      } catch {
        if (!closed) setConn('offline')
      }
    }
    // WebSocket (Bun 原生; 经网关 ?XTransformPort=3005)
    try {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${proto}://${location.host}/?XTransformPort=3005`)
      wsRef.current = ws
      ws.onopen = () => { if (!closed) setConn('live') }
      ws.onmessage = ev => {
        try {
          const m = JSON.parse(ev.data as string)
          if (m.type === 'init' || m.type === 'ready') setState(m.state as StationState)
          else if (m.type === 'bar') {
            setState((prev: StationState | null) => {
              if (!prev || !m.state) return prev
              const st = m.state as WeatherState
              const hist = [...prev.history.slice(-239), st]
              return { ...prev, current: st, history: hist }
            })
          }
        } catch { /* ignore */ }
      }
      ws.onerror = () => { if (!closed) setConn(c => (c === 'live' ? c : 'connecting')) }
      ws.onclose = () => { if (!closed) setConn(c => (c === 'live' ? 'polling' : c)) }
    } catch { /* ws 不可用 → 轮询 */ }
    fetchState()
    poll = setInterval(fetchState, 5000)
    return () => { closed = true; if (poll) clearInterval(poll); wsRef.current?.close() }
  }, [])

  if (!state) {
    return (
      <div className="p-6 text-sm text-zinc-500">
        {conn === 'offline'
          ? '气象站服务离线 — 请启动 mini-services/weather (bun run dev, port 3005)'
          : conn === 'connecting' ? '正在连接气象站…' : '加载中…'}
      </div>
    )
  }
  const cur = state.current
  const hist = state.history
  const connBadge = conn === 'live' ? { txt: 'LIVE · WebSocket', cls: 'border-emerald-800 bg-emerald-950/60 text-emerald-300' }
    : conn === 'polling' ? { txt: 'POLLING · 5s', cls: 'border-sky-800 bg-sky-950/50 text-sky-300' }
    : { txt: conn.toUpperCase(), cls: 'border-amber-800 bg-amber-950/40 text-amber-300' }

  return (
    <div className="space-y-4">
      {/* 标题 */}
      <div className="flex flex-wrap items-center gap-2">
        <Thermometer className="h-4 w-4 text-sky-400" aria-hidden />
        <span className="text-sm font-semibold text-zinc-100">UC4 · 跨资产天气预警站</span>
        <Badge variant="outline" className={connBadge.cls}>{connBadge.txt}</Badge>
        <Badge variant="outline" className="border-zinc-700 bg-zinc-900 text-zinc-400">
          {state.status === 'ready' ? `常驻轮询 · as of ${cur?.t ?? '—'}` : `warming: ${state.progress}`}
        </Badge>
      </div>
      <p className="text-xs leading-relaxed text-zinc-500">
        四台 R2 冠军储层引擎 (nres50 · λ0.9999 · log 目标 · seed42 矩阵同源) 常驻后台流式计算 — 回放 4.5 年历史构建状态,
        每 30s 轮询数据文件增量喂入 (文件追加即实时行情)。{state.meta.engine}
      </p>

      {/* 天气面板 */}
      {cur && (
        <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
          <WeatherCard title="XAU · 1h 预报" a={cur.xau} accent />
          <WeatherCard title="XAU · 4h 展望" a={cur.xau4h} accent />
          <WeatherCard title="DXY · 美元" a={cur.dxy} />
          <WeatherCard title="XAG · 白银" a={cur.xag} silver />
        </div>
      )}

      {/* 预警计数 + IC */}
      <div className="grid gap-3 lg:grid-cols-2">
        <Card className="border-zinc-800 bg-zinc-950/60">
          <CardHeader className="pb-2"><CardTitle className="flex items-center gap-2 text-sm text-zinc-300"><ShieldAlert className="h-3.5 w-3.5 text-red-400" aria-hidden />4.5 年回放累计预警</CardTitle></CardHeader>
          <CardContent>
            <div className="grid grid-cols-4 gap-2 text-center text-xs">
              {[
                { k: 'weather', label: '天气转换', v: state.alertCounts.weather ?? 0 },
                { k: 'outlook', label: '4h展望转换', v: state.alertCounts.outlook ?? 0 },
                { k: 'breaker', label: '断路器 d2>25', v: state.alertCounts.breaker ?? 0 },
                { k: 'silver', label: '白银传输', v: state.alertCounts.silver ?? 0 },
              ].map(x => (
                <div key={x.k} className="rounded border border-zinc-800 bg-zinc-900/60 p-2">
                  <div className="text-[10px] text-zinc-500">{x.label}</div>
                  <div className="mt-1 font-mono text-base font-semibold text-zinc-200">{x.v.toLocaleString()}</div>
                </div>
              ))}
            </div>
            <p className="mt-2 text-[11px] leading-relaxed text-zinc-500">
              断路器 (UC3): d2&gt;25 → 未来 1h 方差 ≈ 5.9×中位 (R2 档案 n=114)。白银传输 (UC4): R_xag≥2.0 →
              白银领先黄金 ~120min (传输 IC 0.55, 因果干净) — XAU 未来 2-4h 方差上行风险。
            </p>
          </CardContent>
        </Card>
        <Card className="border-zinc-800 bg-zinc-950/60">
          <CardHeader className="pb-2"><CardTitle className="flex items-center gap-2 text-sm text-zinc-300"><Activity className="h-3.5 w-3.5 text-emerald-400" aria-hidden />IC 验证 · TS 移植 vs 研究引擎</CardTitle></CardHeader>
          <CardContent>
            <table className="w-full text-xs">
              <thead className="text-zinc-500"><tr className="text-left"><th className="py-1">引擎</th><th className="py-1 text-right">TS 实现</th><th className="py-1 text-right">Python 研究</th><th className="py-1 text-right">偏差</th></tr></thead>
              <tbody className="font-mono text-zinc-300">
                {Object.entries(state.ic).map(([k, v]) => {
                  const ref = state.meta.research_ref[k] ?? 0
                  return (
                    <tr key={k}>
                      <td className="py-1">{k}</td>
                      <td className="py-1 text-right">{v.toFixed(4)}</td>
                      <td className="py-1 text-right text-zinc-500">{ref.toFixed(4)}</td>
                      <td className={`py-1 text-right ${Math.abs(v - ref) <= 0.003 ? 'text-emerald-400' : 'text-amber-400'}`}>{(v - ref >= 0 ? '+' : '')}{(v - ref).toFixed(4)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
            <p className="mt-2 text-[11px] text-zinc-500">XAU-4h 十分位校准: d1 {state.decile[0]?.mean ?? '—'} → d10 {state.decile[9]?.mean ?? '—'} ({state.decile.length >= 10 ? ((state.decile[9]?.mean / state.decile[0]?.mean)).toFixed(1) : '—'}× 价差, 单调)</p>
          </CardContent>
        </Card>
      </div>

      {/* 历史曲线 */}
      <Card className="border-zinc-800 bg-zinc-950/60">
        <CardHeader className="pb-2"><CardTitle className="text-sm text-zinc-300">能量预报历史 · 最近 {hist.length} 个 15 分钟槽</CardTitle></CardHeader>
        <CardContent>
          <EnergyChart hist={hist} />
          <div className="mt-1 flex flex-wrap gap-3 text-[10px] text-zinc-500">
            <span className="flex items-center gap-1"><i className="h-0.5 w-3 bg-sky-400" />XAU 1h</span>
            <span className="flex items-center gap-1"><i className="h-0.5 w-3 bg-emerald-400" />XAU 4h</span>
            <span className="flex items-center gap-1"><i className="h-0.5 w-3 bg-amber-400" />XAG</span>
            <span className="flex items-center gap-1"><i className="h-px w-3 border-t border-dashed border-red-500" />风暴线 R=2.0</span>
          </div>
        </CardContent>
      </Card>

      {/* 预警流 */}
      <Card className="border-zinc-800 bg-zinc-950/60">
        <CardHeader className="pb-2"><CardTitle className="text-sm text-zinc-300">预警流 (最近 {state.alerts.length} 条 · 滞回去抖)</CardTitle></CardHeader>
        <CardContent>
          <ScrollArea className="max-h-72">
            <div className="space-y-1">
              {state.alerts.map((a, i) => (
                <div key={i} className="flex items-start gap-2 rounded border border-zinc-800/60 bg-zinc-900/40 px-2 py-1.5 text-xs">
                  <span className="mt-0.5 shrink-0 text-zinc-600">{a.t}</span>
                  <span className={`mt-0.5 shrink-0 ${KIND_CLS[a.kind] ?? 'text-zinc-400'}`}>{KIND_ICON[a.kind]}</span>
                  <span className="leading-relaxed text-zinc-300">{a.msg}</span>
                </div>
              ))}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      {/* 诚实披露 */}
      <Card className="border-amber-900/50 bg-amber-950/10">
        <CardHeader className="pb-2"><CardTitle className="flex items-center gap-2 text-sm text-amber-300"><FlaskConical className="h-3.5 w-3.5" aria-hidden />诚实修正 · 嵌合 0.5363 的 asof 伪影</CardTitle></CardHeader>
        <CardContent className="space-y-1.5 text-xs leading-relaxed text-zinc-400">
          <p>
            R2 档案的「嵌合引擎 IC 0.491→0.5363」经严格因果检验证伪: 其 asof 允许 DXY/XAG bar 进入 XAU 标签窗口的首 15 分钟
            (lag 消融: lag30→0.536, lag60→0.587, lag120→0.539 — 泄漏曲线)。严格因果 (只用决策时刻已收盘 bar) 下嵌合增益归零
            (0.4915 ≈ 单资产) — shuffle 对照无法区分泄漏与真实领先 (两者都依赖时间对齐), lag 消融才是决定性对照。
          </p>
          <p>
            跨资产的真实价值: 白银→黄金传输预警 (XAG 能量领先 XAU 未来 4h 方差, 传输分析无窗口重叠, IC 0.55) +
            三资产各自能量/惊异度监控 — 本站即按此部署。另: XAU 专用 h16 训练引擎把 4h 预报 IC 提到 <span className="font-mono text-emerald-300">0.5188</span> (真实增益, 非跨资产)。
          </p>
        </CardContent>
      </Card>

      <Separator className="bg-zinc-800" />
      <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-[10px] text-zinc-600">
        引擎 bar 数: {Object.entries(state.meta.bars).map(([k, v]) => `${k} ${v.toLocaleString()}`).join(' · ')} · 数据窗 {state.meta.span.first} → {state.meta.span.last} · uptime {Math.floor(state.uptime_s / 60)}min
      </motion.p>
    </div>
  )
}

function WeatherCard({ title, a, accent, silver }: { title: string; a: AssetState; accent?: boolean; silver?: boolean }) {
  const lv = LEVELS[a.level] ?? LEVELS[1]
  return (
    <div className={`rounded-lg border p-4 ${lv.ring}`}>
      <div className="flex items-center justify-between">
        <span className={`text-xs font-medium ${accent ? 'text-zinc-300' : 'text-zinc-400'}`}>{title}</span>
        <span className="text-2xl" role="img" aria-label={lv.name}>{lv.emoji}</span>
      </div>
      <div className="mt-2 flex items-baseline gap-2">
        <span className={`font-mono text-2xl font-bold ${lv.cls}`}>{a.R.toFixed(2)}</span>
        <span className="text-[10px] text-zinc-500">R̂ (方差比)</span>
      </div>
      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-zinc-500">
        <span>d2 <span className={`font-mono ${a.d2 > 25 ? 'text-red-400' : 'text-zinc-300'}`}>{a.d2.toFixed(1)}</span></span>
        <span>K <span className="font-mono text-zinc-300">{a.K}</span></span>
        <span>vol分位 <span className="font-mono text-zinc-300">{(a.pct * 100).toFixed(0)}%</span></span>
        <span className={silver && a.R >= 2 ? 'text-amber-300' : ''}>{lv.name}</span>
      </div>
      {a.d2 > 25 && (
        <div className="mt-2 rounded bg-red-950/60 px-2 py-1 text-[10px] text-red-300">⚠ 断路器: 未来1h方差≈5.9×中位</div>
      )}
      {silver && a.R >= 2 && (
        <div className="mt-2 rounded bg-amber-950/60 px-2 py-1 text-[10px] text-amber-300">⚠ 传输预警: 白银领先 → XAU 2-4h方差风险</div>
      )}
    </div>
  )
}

function EnergyChart({ hist }: { hist: WeatherState[] }) {
  if (hist.length < 2) return <div className="text-xs text-zinc-600">历史积累中…</div>
  const W = 800, H = 140, PAD = 4
  const maxY = Math.max(2.2, ...hist.map(h => Math.max(h.xau.R, h.xau4h.R, h.xag.R))) * 1.05
  const x = (i: number) => PAD + (i / (hist.length - 1)) * (W - 2 * PAD)
  const y = (v: number) => H - PAD - (v / maxY) * (H - 2 * PAD)
  const line = (sel: (h: WeatherState) => number) =>
    hist.map((h, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(sel(h)).toFixed(1)}`).join(' ')
  const yTick = [0.7, 1.2, 2.0]
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-36 w-full" role="img" aria-label="energy history">
      {yTick.map(v => (
        <g key={v}>
          <line x1={PAD} x2={W - PAD} y1={y(v)} y2={y(v)} stroke={v === 2 ? 'rgb(239 68 68 / 0.5)' : 'rgb(63 63 70 / 0.5)'} strokeDasharray={v === 2 ? '4 3' : ''} strokeWidth={1} />
          <text x={PAD + 2} y={y(v) - 2} fill="rgb(113 113 122)" fontSize={8}>{v}</text>
        </g>
      ))}
      <path d={line(h => h.xag.R)} fill="none" stroke="rgb(251 191 36)" strokeWidth={1.4} opacity={0.85} />
      <path d={line(h => h.xau4h.R)} fill="none" stroke="rgb(52 211 153)" strokeWidth={1.6} />
      <path d={line(h => h.xau.R)} fill="none" stroke="rgb(56 189 248)" strokeWidth={1.6} />
      <text x={W - PAD - 66} y={H - 6} fill="rgb(113 113 122)" fontSize={8}>{hist[0]?.t.slice(5, 10)} → {hist[hist.length - 1]?.t.slice(5, 16)}</text>
    </svg>
  )
}
