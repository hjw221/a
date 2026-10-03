'use client'

import { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Button } from '@/components/ui/button'
import { ReservoirTab } from '@/components/reservoir-tab'
import type { ReservoirData } from '@/components/reservoir-tab'
import { V19Tab } from '@/components/v19-tab'
import type { V19Data } from '@/components/v19-tab'
import { R2Tab } from '@/components/r2-tab'
import type { R2Data } from '@/components/r2-tab'
import {
  AlertTriangle, CheckCircle2, Crosshair, Download, FlaskConical, Gauge, Layers,
  LineChart, ShieldCheck, Timer, TrendingUp, XCircle,
} from 'lucide-react'

/* ================= types ================= */
interface Summary {
  total: number; trades: number; avg: number; std: number; t_stat: number
  win_rate: number; p10: number; p90: number; worst: number; best: number
  by_year: Record<string, number>; n_by_year: Record<string, number>
  share2026: number; sharpe: number; maxdd: number
  med_cost: number; med_atr: number; total54: number
  months: Record<string, number>
}
interface Trade {
  entry_time: string; exit_time: string; side: 'L' | 'S'
  entry_px: number; exit_px: number; cost: number; pnl: number
  hold_bars: number; reason: string; atr_entry: number
}
interface Runner { spec: Record<string, string | number>; base: Summary; pess03: Summary; pess05: Summary }
interface Champ extends Runner { trades: Trade[] }
interface Finalist {
  tf: string; entry: string; dir: string; gate: string; exit: string
  total: number; n: number; avg: number; sh26: number; dd: number
  wr: number; plr: number; by: Record<string, number>
  ladder: boolean; plateau: number | null
}
interface V18MatrixRow {
  stop: string; total: number; trades: number; avg: number; worst: number; p05: number
  hit: number; hit_rate: number; maxdd: number; all_pos: boolean
  by_year: Record<string, number>; ladder: number[]; share2026: number
}
interface V18OverlayRow {
  stop: string; total: number; n_hit: number; hit_rate: number; worst: number
  avoided: number; killed: number; net: number
}
interface V18Data {
  meta: {
    generated: string; branch: string; question: string; answer: string
    engine: string; repro: string
    base: { total: number; trades: number; worst: number; by_year: Record<string, number> }
  }
  matrix: V18MatrixRow[]
  overlay: V18OverlayRow[]
  counterfactual: Record<string, { engine_stopped: number; matched: number; saved_sum: number; cost_sum: number; net: number }>
  mae: {
    p: Record<string, number>; max: number; corr_pnl: number
    hist: Record<string, number>; deep6_n: number; deep6_pnl: number; deep6_pos_n: number
  }
  extreme: { stop: string; total: number; trades: number; worst: number; n_sl_hit: number; all_pos: boolean }[]
  time_variant: { stop: string; total: number; trades: number; worst: number; n_sl_hit: number; all_pos: boolean }[]
  cooldown: { stop: string; total: number; trades: number; worst: number; n_sl_hit: number; all_pos: boolean }[]
  verdicts: { j1: string; j2: string; j3: string; h_cool: string; roots: string[]; implications: string[] }
}
interface V17Data {
  meta: {
    generated: string; window: string; engine: string; cost_note: string
    tf_stats: Record<string, { bars: number; atr_med: number; atr_p90: number }> & { spread_rt_med: number }
  }
  champion: Champ
  runners: Runner[]
  ablation: Record<string, Omit<Summary, 'months' | 'total54'>>
  finalists: Finalist[]
  benchmarks: Record<string, { total?: number; trades?: number; avg?: number; note: string }>
  v17_initial: { note: string; chandelier_verdict: string }
}

/* ================= helpers ================= */
const fmtUSD = (v: number, digits = 0) =>
  `${v < 0 ? '−' : ''}$${Math.abs(v).toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits })}`
const pct = (v: number) => `${v.toFixed(1)}%`
const DIR_LABEL: Record<string, string> = { long: '只做多', short: '只做空', both: '多空双向' }
const GATE_LABEL: Record<string, string> = { none: '无闸门', atrmed: 'ATR>季度中位', atrp30: 'ATR>季度p30' }
const ENTRY_LABEL: Record<string, string> = {
  don20s: 'Donchian-20 停损单', don40s: 'Donchian-40 停损单', don55s: 'Donchian-55 停损单',
  don20: 'Donchian-20 收盘', don40: 'Donchian-40 收盘', don55: 'Donchian-55 收盘',
  'kelt20_2.0': 'Keltner 2.0×ATR', 'kelt20_2.5': 'Keltner 2.5×ATR',
}
const EXIT_LABEL: Record<string, string> = {
  t1d: '持1日', t2d: '持2日', t3d: '持3日', t5d: '持5日', t8d: '持8日', t13d: '持13日',
  tur10: '海龟10反破', tur20: '海龟20反破',
  t5d_ch8: '持5日+8ATR灾难停', t13d_ch8: '持13日+8ATR灾难停',
}
const YEARS = ['2022', '2023', '2024', '2025', '2026']
const rise = (i: number) => ({ initial: { opacity: 0, y: 14 }, animate: { opacity: 1, y: 0 }, transition: { delay: 0.05 * i, duration: 0.4 } })

/* ================= sub components ================= */
function KpiCard({ icon, label, value, sub, tone = 'default' }: {
  icon: React.ReactNode; label: string; value: string; sub?: string
  tone?: 'default' | 'pos' | 'warn'
}) {
  const toneCls = tone === 'pos' ? 'text-emerald-400' : tone === 'warn' ? 'text-amber-400' : 'text-zinc-100'
  return (
    <motion.div {...rise(0)}>
      <Card className="border-zinc-800 bg-zinc-900/60">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
            {icon}{label}
          </div>
          <div className={`mt-2 font-mono text-2xl font-semibold tabular-nums ${toneCls}`}>{value}</div>
          {sub && <div className="mt-1 text-xs text-zinc-500">{sub}</div>}
        </CardContent>
      </Card>
    </motion.div>
  )
}

function YearBars({ by, n }: { by: Record<string, number>; n?: Record<string, number> }) {
  const maxAbs = Math.max(...YEARS.map((y) => Math.abs(by[y] ?? 0)), 1)
  return (
    <div className="grid grid-cols-5 gap-2 sm:gap-3" role="img" aria-label="年度盈亏">
      {YEARS.map((y) => {
        const v = by[y] ?? 0
        const h = Math.max((Math.abs(v) / maxAbs) * 100, 2)
        return (
          <div key={y} className="flex flex-col items-center gap-1.5">
            <span className={`font-mono text-xs font-semibold tabular-nums ${v >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
              {fmtUSD(v)}
            </span>
            <div className="flex h-24 w-full items-end justify-center rounded bg-zinc-900/60">
              <div
                className={`w-3/5 rounded-sm ${v >= 0 ? 'bg-emerald-500/80' : 'bg-rose-500/80'} sm:w-1/2`}
                style={{ height: `${h}%` }}
              />
            </div>
            <span className="text-[11px] text-zinc-500">{y}{n?.[y] ? ` · ${n[y]}笔` : ''}</span>
          </div>
        )
      })}
    </div>
  )
}

function EquityChart({ months }: { months: Record<string, number> }) {
  const pts = useMemo(() => {
    const keys = Object.keys(months).sort()
    return keys.reduce<{ k: string; v: number }[]>((acc, k) => {
      const prev = acc.length > 0 ? acc[acc.length - 1].v : 0
      acc.push({ k, v: prev + months[k] })
      return acc
    }, [])
  }, [months])
  if (pts.length < 2) return null
  const W = 860, H = 200, PAD = 8
  const vals = pts.map((p) => p.v)
  const min = Math.min(...vals, 0), max = Math.max(...vals)
  const x = (i: number) => PAD + (i / (pts.length - 1)) * (W - 2 * PAD)
  const y = (v: number) => H - PAD - ((v - min) / (max - min || 1)) * (H - 2 * PAD)
  const line = pts.map((p, i) => `${x(i).toFixed(1)},${y(p.v).toFixed(1)}`).join(' ')
  const area = `${PAD},${H - PAD} ${line} ${W - PAD},${H - PAD}`
  const yearMarks = [2023, 2024, 2025, 2026].map((yr) => {
    const i = pts.findIndex((p) => p.k.startsWith(String(yr)))
    return i >= 0 ? { i, yr } : null
  }).filter(Boolean)
  return (
    <div className="w-full">
      <svg viewBox={`0 0 ${W} ${H}`} className="h-44 w-full sm:h-52" role="img" aria-label="月度累计盈亏曲线" preserveAspectRatio="none">
        <polygon points={area} fill="rgb(16 185 129 / 0.12)" />
        <line x1={PAD} x2={W - PAD} y1={y(0)} y2={y(0)} stroke="rgb(113 113 122 / 0.4)" strokeDasharray="3 4" strokeWidth="1" />
        <polyline points={line} fill="none" stroke="rgb(52 211 153)" strokeWidth="2" strokeLinejoin="round" />
        {yearMarks.map((m) => (
          <line key={m!.yr} x1={x(m!.i)} x2={x(m!.i)} y1={PAD} y2={H - PAD} stroke="rgb(113 113 122 / 0.35)" strokeWidth="1" strokeDasharray="2 4" />
        ))}
      </svg>
      <div className="mt-1 flex justify-between text-[10px] text-zinc-600">
        {yearMarks.map((m) => <span key={m!.yr}>{m!.yr}</span>)}
        <span>终值 {fmtUSD(vals[vals.length - 1])}</span>
      </div>
    </div>
  )
}

function LadderRow({ name, s, note }: { name: string; s: Summary; note: string }) {
  const allPos = YEARS.every((y) => (s.by_year[y] ?? 0) > 0)
  return (
    <div className="grid grid-cols-[5.5rem_1fr] items-center gap-3 rounded-lg border border-zinc-800 bg-zinc-900/40 p-3 sm:grid-cols-[5.5rem_1fr_auto]">
      <div>
        <div className="font-mono text-sm font-semibold">{name}</div>
        <div className="text-[10px] text-zinc-500">{note}</div>
      </div>
      <div className="flex items-baseline gap-2">
        <span className={`font-mono text-lg font-semibold tabular-nums ${allPos ? 'text-emerald-400' : 'text-amber-400'}`}>
          {fmtUSD(s.total)}
        </span>
        <span className="text-xs text-zinc-500">笔均 {fmtUSD(s.avg, 2)}</span>
        <div className="ml-auto hidden gap-1 sm:flex">
          {YEARS.map((y) => (
            <span key={y} className={`rounded px-1.5 py-0.5 font-mono text-[10px] tabular-nums ${
              (s.by_year[y] ?? 0) >= 0 ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`}>
              {y.slice(2)} {(s.by_year[y] ?? 0) >= 0 ? '+' : '−'}{Math.abs(Math.round(s.by_year[y] ?? 0))}
            </span>
          ))}
        </div>
      </div>
      <Badge variant="outline" className={`col-span-2 justify-self-start sm:col-span-1 sm:justify-self-end ${
        allPos ? 'border-emerald-800 text-emerald-400' : 'border-amber-800 text-amber-400'}`}>
        {allPos ? '五年全正' : '有负年份'}
      </Badge>
    </div>
  )
}

function FinalistsTable({ list }: { list: Finalist[] }) {
  return (
    <ScrollArea className="max-h-96 rounded-md border border-zinc-800">
      <table className="w-full text-left text-xs">
        <thead className="sticky top-0 z-10 bg-zinc-900 shadow-[0_1px_0_0_rgb(63_63_70)]">
          <tr className="text-[10px] uppercase tracking-wide text-zinc-500">
            <th className="px-2 py-2">#</th><th className="px-2 py-2">TF</th><th className="px-2 py-2">进场</th>
            <th className="px-2 py-2">方向</th><th className="px-2 py-2">闸门</th><th className="px-2 py-2">出场</th>
            <th className="px-2 py-2 text-right">总PnL</th><th className="px-2 py-2 text-right">笔数</th>
            <th className="px-2 py-2 text-right">笔均</th><th className="px-2 py-2 text-right">26占比</th>
            <th className="px-2 py-2 text-center">阶梯</th><th className="px-2 py-2 text-center">平台</th>
          </tr>
        </thead>
        <tbody className="font-mono tabular-nums">
          {list.map((r, i) => (
            <tr key={i} className={`border-t border-zinc-800/60 ${i === 0 ? 'bg-emerald-500/5 ring-1 ring-inset ring-emerald-700/40' : ''}`}>
              <td className="px-2 py-1.5 text-zinc-500">{i + 1}</td>
              <td className="px-2 py-1.5 font-sans font-medium">{r.tf}</td>
              <td className="px-2 py-1.5 font-sans">{ENTRY_LABEL[r.entry] ?? r.entry}</td>
              <td className="px-2 py-1.5 font-sans">{DIR_LABEL[r.dir] ?? r.dir}</td>
              <td className="px-2 py-1.5 font-sans">{GATE_LABEL[r.gate] ?? r.gate}</td>
              <td className="px-2 py-1.5 font-sans">{EXIT_LABEL[r.exit] ?? r.exit}</td>
              <td className="px-2 py-1.5 text-right text-emerald-400">{fmtUSD(r.total)}</td>
              <td className="px-2 py-1.5 text-right text-zinc-300">{r.n}</td>
              <td className="px-2 py-1.5 text-right text-zinc-300">${r.avg.toFixed(2)}</td>
              <td className="px-2 py-1.5 text-right text-zinc-400">{r.sh26.toFixed(1)}%</td>
              <td className="px-2 py-1.5 text-center">{r.ladder
                ? <CheckCircle2 className="mx-auto h-3.5 w-3.5 text-emerald-400" aria-label="成本阶梯通过" />
                : <XCircle className="mx-auto h-3.5 w-3.5 text-zinc-600" aria-label="成本阶梯未过" />}</td>
              <td className="px-2 py-1.5 text-center text-zinc-300">{r.plateau === null ? '—' : `${(r.plateau * 100).toFixed(0)}%`}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </ScrollArea>
  )
}

function TradesTable({ trades }: { trades: Trade[] }) {
  return (
    <ScrollArea className="max-h-96 rounded-md border border-zinc-800">
      <table className="w-full text-left text-xs">
        <thead className="sticky top-0 z-10 bg-zinc-900 shadow-[0_1px_0_0_rgb(63_63_70)]">
          <tr className="text-[10px] uppercase tracking-wide text-zinc-500">
            <th className="px-2 py-2">进场</th><th className="px-2 py-2">出场</th>
            <th className="px-2 py-2">向</th><th className="px-2 py-2 text-right">进价</th>
            <th className="px-2 py-2 text-right">出价</th><th className="px-2 py-2 text-right">成本</th>
            <th className="px-2 py-2 text-right">PnL</th><th className="px-2 py-2 text-right">持有</th>
          </tr>
        </thead>
        <tbody className="font-mono tabular-nums">
          {trades.map((t, i) => (
            <tr key={i} className="border-t border-zinc-800/60">
              <td className="whitespace-nowrap px-2 py-1.5 text-zinc-400">{t.entry_time}</td>
              <td className="whitespace-nowrap px-2 py-1.5 text-zinc-400">{t.exit_time}</td>
              <td className="px-2 py-1.5">
                <span className={`rounded px-1 py-0.5 text-[10px] font-semibold ${t.side === 'L' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`}>
                  {t.side === 'L' ? '多' : '空'}
                </span>
              </td>
              <td className="px-2 py-1.5 text-right text-zinc-300">{t.entry_px.toFixed(2)}</td>
              <td className="px-2 py-1.5 text-right text-zinc-300">{t.exit_px.toFixed(2)}</td>
              <td className="px-2 py-1.5 text-right text-zinc-500">{t.cost.toFixed(2)}</td>
              <td className={`px-2 py-1.5 text-right font-semibold ${t.pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                {t.pnl >= 0 ? '+' : '−'}{Math.abs(t.pnl).toFixed(2)}
              </td>
              <td className="px-2 py-1.5 text-right text-zinc-500">{t.hold_bars}bar</td>
            </tr>
          ))}
        </tbody>
      </table>
    </ScrollArea>
  )
}

function AblationCard({ ablation }: { ablation: Record<string, Omit<Summary, 'months' | 'total54'>> }) {
  const order = ['both/none', 'both/atrmed', 'long/none', 'long/atrp30', 'long/atrmed']
  const labels: Record<string, string> = {
    'both/none': '双向 · 无闸门', 'both/atrmed': '双向 · ATR闸门', 'long/none': '只多 · 无闸门',
    'long/atrp30': '只多 · p30闸门', 'long/atrmed': '只多 · ATR闸门 ★冠军',
  }
  const max = Math.max(...order.map((k) => ablation[k]?.total ?? 0), 1)
  return (
    <div className="space-y-2">
      {order.map((k) => {
        const a = ablation[k]
        if (!a) return null
        const isChamp = k === 'long/atrmed'
        return (
          <div key={k} className={`grid grid-cols-[7.5rem_1fr_5rem] items-center gap-2 rounded-md p-2 ${isChamp ? 'bg-emerald-500/10 ring-1 ring-inset ring-emerald-700/50' : 'bg-zinc-900/40'}`}>
            <span className="text-xs text-zinc-300">{labels[k]}</span>
            <div className="h-5 overflow-hidden rounded-sm bg-zinc-800/60">
              <motion.div
                initial={{ width: 0 }} animate={{ width: `${(a.total / max) * 100}%` }} transition={{ duration: 0.6, ease: 'easeOut' }}
                className={`h-full rounded-sm ${isChamp ? 'bg-emerald-500' : 'bg-emerald-700/70'}`}
              />
            </div>
            <span className="text-right font-mono text-xs font-semibold tabular-nums text-emerald-400">{fmtUSD(a.total)}</span>
          </div>
        )
      })}
      <p className="pt-1 text-xs leading-relaxed text-zinc-500">
        同一 M30/Donchian-55/持5日几何下的两个开关消融：<b className="text-zinc-300">只做多 +$1,327</b>（空头在悲观成本下为负贡献）、
        <b className="text-zinc-300">ATR 闸门再 +$1,320</b>（主要救活 2026：无闸门时 2026 为 −$670）。闸门 = 规则版波动率 Regime Gate，即路线 A 的预演。
      </p>
    </div>
  )
}

/* ================= v18 分支A: 静态灾难止损实验室 ================= */
function StopMatrixTable({ rows, baseTotal }: { rows: V18MatrixRow[]; baseTotal: number }) {
  return (
    <ScrollArea className="max-h-96 rounded-md border border-zinc-800">
      <table className="w-full text-left text-xs">
        <thead className="sticky top-0 z-10 bg-zinc-900 shadow-[0_1px_0_0_rgb(63_63_70)]">
          <tr className="text-[10px] uppercase tracking-wide text-zinc-500">
            <th className="px-2 py-2">止损线</th><th className="px-2 py-2 text-right">总PnL</th>
            <th className="px-2 py-2 text-right">vs base</th><th className="px-2 py-2 text-right">笔数</th>
            <th className="px-2 py-2 text-right">最差单笔</th><th className="px-2 py-2 text-right">p05</th>
            <th className="px-2 py-2 text-right">触发</th><th className="px-2 py-2 text-right">maxDD</th>
            <th className="px-2 py-2 text-center">逐年</th>
          </tr>
        </thead>
        <tbody className="font-mono tabular-nums">
          {rows.map((r) => {
            const d = r.total - baseTotal
            const isBase = r.stop === 'none'
            return (
              <tr key={r.stop} className={`border-t border-zinc-800/60 ${isBase ? 'bg-emerald-500/5 ring-1 ring-inset ring-emerald-700/40' : ''}`}>
                <td className="px-2 py-1.5 font-sans font-medium text-zinc-200">{isBase ? '无止损 ★base' : r.stop}</td>
                <td className={`px-2 py-1.5 text-right font-semibold ${isBase ? 'text-emerald-400' : 'text-zinc-300'}`}>{fmtUSD(r.total)}</td>
                <td className={`px-2 py-1.5 text-right ${d >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>{isBase ? '—' : fmtUSD(d)}</td>
                <td className="px-2 py-1.5 text-right text-zinc-400">{r.trades}</td>
                <td className={`px-2 py-1.5 text-right ${r.worst >= -130 ? 'text-amber-300' : 'text-rose-400'}`}>{fmtUSD(r.worst)}</td>
                <td className="px-2 py-1.5 text-right text-zinc-500">{fmtUSD(r.p05)}</td>
                <td className="px-2 py-1.5 text-right text-zinc-400">{r.hit > 0 ? `${r.hit} (${(r.hit_rate * 100).toFixed(0)}%)` : '—'}</td>
                <td className="px-2 py-1.5 text-right text-zinc-500">{fmtUSD(r.maxdd)}</td>
                <td className="px-2 py-1.5 text-center">
                  {r.all_pos ? <CheckCircle2 className="mx-auto h-3.5 w-3.5 text-emerald-400" aria-label="逐年全正" />
                    : <XCircle className="mx-auto h-3.5 w-3.5 text-rose-400" aria-label="有负年份" />}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </ScrollArea>
  )
}

function MaeHistogram({ mae }: { mae: V18Data['mae'] }) {
  const buckets = Object.entries(mae.hist)
    .map(([lo, n]) => ({ lo: parseFloat(lo), n }))
    .sort((a, b) => a.lo - b.lo)
  const maxN = Math.max(...buckets.map((b) => b.n), 1)
  return (
    <div>
      <div className="space-y-1" role="img" aria-label="MAE 分布直方图">
        {buckets.map((b) => (
          <div key={b.lo} className="grid grid-cols-[3.2rem_1fr_1.6rem] items-center gap-2">
            <span className="text-right font-mono text-[10px] text-zinc-500">{b.lo}–{b.lo + 1}×</span>
            <div className="h-4 overflow-hidden rounded-sm bg-zinc-900/60">
              <motion.div
                initial={{ width: 0 }} animate={{ width: `${(b.n / maxN) * 100}%` }}
                transition={{ duration: 0.5, ease: 'easeOut' }}
                className={`h-full rounded-sm ${b.lo >= 6 ? 'bg-rose-500/70' : 'bg-emerald-700/70'}`}
              />
            </div>
            <span className="font-mono text-[10px] tabular-nums text-zinc-400">{b.n}</span>
          </div>
        ))}
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2 text-[11px] text-zinc-500 sm:grid-cols-4">
        <span>p50 <b className="text-zinc-300">{mae.p['50']}×</b></span>
        <span>p90 <b className="text-zinc-300">{mae.p['90']}×</b></span>
        <span>p99 <b className="text-zinc-300">{mae.p['99']}×</b></span>
        <span>max <b className="text-zinc-300">{mae.max}×</b></span>
      </div>
    </div>
  )
}

function OverlayBars({ rows }: { rows: V18OverlayRow[] }) {
  const maxAbs = Math.max(...rows.flatMap((r) => [r.avoided, r.killed]), 1)
  return (
    <div className="space-y-1.5">
      {rows.map((r) => (
        <div key={r.stop} className="grid grid-cols-[3.4rem_1fr] items-center gap-2">
          <span className="font-mono text-[11px] text-zinc-300">{r.stop}</span>
          <div className="space-y-0.5">
            <div className="grid grid-cols-2 gap-1">
              <div className="flex justify-end">
                <motion.div
                  initial={{ width: 0 }} animate={{ width: `${(r.avoided / maxAbs) * 100}%` }}
                  transition={{ duration: 0.5 }} className="h-2.5 rounded-l-sm bg-emerald-600/80"
                />
              </div>
              <div>
                <motion.div
                  initial={{ width: 0 }} animate={{ width: `${(r.killed / maxAbs) * 100}%` }}
                  transition={{ duration: 0.5 }} className="h-2.5 rounded-r-sm bg-rose-600/80"
                />
              </div>
            </div>
            <div className="flex justify-between font-mono text-[9px] tabular-nums text-zinc-500">
              <span className="text-emerald-500">救 +{Math.round(r.avoided)}</span>
              <span className={r.net >= 0 ? 'text-emerald-500' : 'text-rose-500'}>net {fmtUSD(r.net)}</span>
              <span className="text-rose-500">杀 −{Math.round(r.killed)}</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

function V18Lab({ v18 }: { v18: V18Data }) {
  const base = v18.meta.base
  const best = v18.matrix.filter((r) => r.stop !== 'none').reduce((a, b) => (b.total > a.total ? b : a))
  const bestTvt = v18.time_variant.reduce((a, b) => (b.total > a.total ? b : a))
  const bestCd = v18.cooldown.reduce((a, b) => (b.total > a.total ? b : a))
  return (
    <div className="space-y-6">
      {/* 结论横幅 */}
      <motion.div {...rise(0)}>
        <Card className="border-rose-900/60 bg-rose-950/20">
          <CardContent className="p-4 sm:p-6">
            <div className="flex flex-wrap items-start gap-3">
              <XCircle className="mt-0.5 h-5 w-5 shrink-0 text-rose-400" aria-hidden />
              <div className="min-w-0 flex-1">
                <h2 className="text-base font-semibold text-zinc-100">分支 A · 静态灾难止损 — <span className="text-rose-400">否定性结论</span></h2>
                <p className="mt-1.5 text-sm leading-relaxed text-zinc-400">{v18.meta.question}</p>
                <p className="mt-1 text-sm leading-relaxed text-zinc-300"><b className="text-rose-400">答：</b>{v18.meta.answer}</p>
                <p className="mt-2 text-xs leading-relaxed text-zinc-500">
                  假设检验全景：入场锁定 Entry−k×ATR（k=2~20）与入场时刻 Donchian 下轨（N=10/20/40/55），
                  外加时变版（「快速破位才斩」）与止损+冷却版（「被斩=假突破确认」），共 <b className="text-zinc-300">44 个变体</b>全部无法保留 $2,500+。
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* KPI 行 */}
      <div className="grid min-w-0 grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <KpiCard icon={<Crosshair className="h-3.5 w-3.5" />} label="base (无止损)" value={fmtUSD(base.total)} sub={`${base.trades} 笔 · 5年全正`} tone="pos" />
        <KpiCard icon={<AlertTriangle className="h-3.5 w-3.5" />} label="最好止损臂" value={fmtUSD(best.total)} sub={`${best.stop} · 保留 ${(best.total / base.total * 100).toFixed(0)}%`} tone="warn" />
        <KpiCard icon={<XCircle className="h-3.5 w-3.5" />} label="保住 $2,500 的变体" value="0 / 12" sub="全宽度 × 全结构 × 全时变" tone="warn" />
        <KpiCard icon={<ShieldCheck className="h-3.5 w-3.5" />} label="base 尾部(本就封着)" value={fmtUSD(base.worst)} sub="p05 −$89 · maxDD −$260" />
        <KpiCard icon={<Gauge className="h-3.5 w-3.5" />} label="MAE 中位" value={`${v18.mae.p['50']}×ATR`} sub="正常回踩即 5.7 倍 ATR" tone="warn" />
        <KpiCard icon={<LineChart className="h-3.5 w-3.5" />} label="止损触发率" value="36–80%" sub="atr8.0 仍有 36% 被扫" tone="warn" />
      </div>

      {/* 矩阵 + overlay */}
      <div className="grid min-w-0 gap-6 lg:grid-cols-5">
        <section className="min-w-0 lg:col-span-3" aria-label="止损矩阵">
          <Card className="h-full border-zinc-800 bg-zinc-900/40">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-zinc-300">止损矩阵 · 引擎口径（0.3×ATR 悲观成本）</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <StopMatrixTable rows={v18.matrix} baseTotal={base.total} />
              <p className="text-xs leading-relaxed text-zinc-500">
                静态止损线在进场瞬间由（进场价, 入场 ATR / 入场时刻下轨）锁定，全程不上移——用户的「呼吸空间」设计已如实实现。
                即便如此：<b className="text-zinc-300">每一条止损线都在扫正常动量回踩</b>。宽止损（10~20×ATR）虽极少触发（11~38 次），
                但 worst 反而恶化（−$189 → −$373）：斩仓释放仓位后，突破信号在震荡下跌中反复再进场，制造新交易与新尾部。
              </p>
            </CardContent>
          </Card>
        </section>
        <section className="min-w-0 lg:col-span-2" aria-label="纯保险分析">
          <Card className="h-full border-zinc-800 bg-zinc-900/40">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-zinc-300">纯保险 overlay · base 同 159 笔零耦合</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <OverlayBars rows={v18.overlay} />
              <p className="text-xs leading-relaxed text-zinc-500">
                在<b className="text-zinc-300">完全相同的 159 笔</b>上叠加止损（不改变交易集合）：绿色 = 止损砍掉的亏损（保险赔付），
                红色 = 止损误杀的盈利。<b className="text-rose-400">killed &gt; avoided 全率成立</b>——
                每 $1 的赔付代价 $1.3~$1.7 的误杀。这是静态止损负期望的最干净证据（不受再进场路径干扰）。
              </p>
            </CardContent>
          </Card>
        </section>
      </div>

      {/* MAE + 三大根因 */}
      <div className="grid min-w-0 gap-6 lg:grid-cols-5">
        <section className="min-w-0 lg:col-span-2" aria-label="MAE 分布">
          <Card className="h-full border-zinc-800 bg-zinc-900/40">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-zinc-300">MAE 分布 · 「黑天鹅阈值」不存在</CardTitle>
            </CardHeader>
            <CardContent>
              <MaeHistogram mae={v18.mae} />
              <Separator className="my-3 bg-zinc-800" />
              <p className="text-xs leading-relaxed text-zinc-500">
                base 每笔的最大不利偏移（ATR 单位）：0~21× 连续覆盖、无分离带。MAE&gt;6× 的 74 笔里仍有
                <b className="text-zinc-300"> {v18.mae.deep6_pos_n} 笔最终盈利</b>（深回踩后回血）。
                corr(MAE, 终局 PnL) = {v18.mae.corr_pnl}——深回踩确实偏亏，但 −0.52 的相关性不足以把「止损」变成正期望保险。
              </p>
            </CardContent>
          </Card>
        </section>
        <section className="min-w-0 lg:col-span-3" aria-label="三大根因">
          <Card className="h-full border-zinc-800 bg-zinc-900/40">
            <CardHeader className="pb-2"><CardTitle className="text-sm text-zinc-300">为什么失败 — 三大根因</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              {v18.verdicts.roots.map((r, i) => (
                <div key={i} className="flex gap-3 rounded-lg border border-zinc-800 bg-zinc-900/40 p-3">
                  <span className="font-mono text-sm font-semibold text-rose-400">{i + 1}</span>
                  <p className="text-sm leading-relaxed text-zinc-400">{r}</p>
                </div>
              ))}
              <div className="grid grid-cols-3 gap-2 pt-1">
                <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-2.5 text-center">
                  <div className="font-mono text-sm font-semibold text-rose-400">{fmtUSD(bestTvt.total)}</div>
                  <div className="mt-0.5 text-[10px] text-zinc-500">时变止损最好<br />（atr4.0@3d，仍 −20%）</div>
                </div>
                <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-2.5 text-center">
                  <div className="font-mono text-sm font-semibold text-rose-400">{fmtUSD(bestCd.total)}</div>
                  <div className="mt-0.5 text-[10px] text-zinc-500">止损+冷却最好<br />（atr8.0+5d，更差）</div>
                </div>
                <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-2.5 text-center">
                  <div className="font-mono text-sm font-semibold text-rose-400">−$120</div>
                  <div className="mt-0.5 text-[10px] text-zinc-500">尾部最好改善<br />代价 −$800 利润</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </section>
      </div>

      {/* 启示 */}
      <section aria-label="启示与下一步">
        <Card className="border-zinc-800 bg-zinc-900/40">
          <CardHeader className="pb-2"><CardTitle className="text-sm text-zinc-300">启示 — 尾部管理的正确位置</CardTitle></CardHeader>
          <CardContent className="grid gap-2.5 md:grid-cols-3">
            {v18.verdicts.implications.map((t, i) => (
              <p key={i} className="flex gap-2 rounded-lg border border-zinc-800 bg-zinc-900/40 p-3 text-sm leading-relaxed text-zinc-400">
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" aria-hidden />
                <span>{t}</span>
              </p>
            ))}
          </CardContent>
        </Card>
      </section>

      {/* 数据现状披露 */}
      <section aria-label="数据现状">
        <Card className="border-amber-900/50 bg-amber-950/10">
          <CardContent className="p-4 text-sm leading-relaxed text-zinc-400">
            <p className="flex gap-2">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" aria-hidden />
              <span><b className="text-amber-400">多资产数据现状</b> — 本地与服务器仅持有 XAUUSD 单一资产数据（1.6M 根 M1）。
              <b className="text-zinc-300">没有 DXY、白银、US10Y、VIX 数据</b>。多资产宏观 Regime 门控（第三阶段）启动前需先获取并验证这些数据源，
              候选方案：免费 MT5 demo 导出 / stooq / FRED（DXY·US10Y·VIX 日频免费）；白银 XAGUSD 可从同源 MT5 取 M15。</span>
            </p>
          </CardContent>
        </Card>
      </section>
    </div>
  )
}

/* ================= page ================= */
export default function Home() {
  const [data, setData] = useState<V17Data | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [v18, setV18] = useState<V18Data | null>(null)
  const [res, setRes] = useState<ReservoirData | null>(null)
  const [v19, setV19] = useState<V19Data | null>(null)
  const [r2, setR2] = useState<R2Data | null>(null)

  useEffect(() => {
    fetch('/data/v17.json')
      .then((r) => { if (!r.ok) throw new Error(String(r.status)); return r.json() as Promise<V17Data> })
      .then(setData)
      .catch((e: unknown) => setErr(e instanceof Error ? e.message : 'load failed'))
    fetch('/data/v18.json')
      .then((r) => { if (!r.ok) throw new Error(String(r.status)); return r.json() as Promise<V18Data> })
      .then(setV18)
      .catch(() => setV18(null))
    fetch('/data/reservoir.json')
      .then((r) => { if (!r.ok) throw new Error(String(r.status)); return r.json() as Promise<ReservoirData> })
      .then(setRes)
      .catch(() => setRes(null))
    fetch('/data/v19.json')
      .then((r) => { if (!r.ok) throw new Error(String(r.status)); return r.json() as Promise<V19Data> })
      .then(setV19)
      .catch(() => setV19(null))
    fetch('/data/r2.json')
      .then((r) => { if (!r.ok) throw new Error(String(r.status)); return r.json() as Promise<R2Data> })
      .then(setR2)
      .catch(() => setR2(null))
  }, [])

  const c = data?.champion
  const s = c?.pess03

  return (
    <div className="flex min-h-screen flex-col bg-zinc-950 text-zinc-100">
      <header className="border-b border-zinc-800/80 bg-zinc-950/90">
        <div className="mx-auto w-full max-w-6xl px-4 py-6 sm:px-6">
          <div className="flex flex-wrap items-center gap-3">
            <FlaskConical className="h-6 w-6 text-emerald-400" aria-hidden />
            <div>
              <h1 className="text-lg font-semibold tracking-tight sm:text-xl">XAUUSD · 量化研究台 <span className="font-mono text-emerald-400">v17 → v18 → R1</span></h1>
              <p className="mt-0.5 text-xs text-zinc-500">HTF 纯规则突破 · 静态止损检验 · Streaming RLS-Reservoir 在线自学习引擎 · {data?.meta.window ?? ''}</p>
            </div>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            <Badge variant="outline" className="border-emerald-800 bg-emerald-950/40 text-emerald-400"><ShieldCheck className="mr-1 h-3 w-3" />正确结算 · 无回望</Badge>
            <Badge variant="outline" className="border-zinc-700 text-zinc-300"><Layers className="mr-1 h-3 w-3" />双引擎逐位互验</Badge>
            <Badge variant="outline" className="border-amber-800 bg-amber-950/40 text-amber-400"><Gauge className="mr-1 h-3 w-3" />0.3×ATR 悲观成本</Badge>
            <Badge variant="outline" className="border-zinc-700 text-zinc-300"><Timer className="mr-1 h-3 w-3" />串行单仓可部署</Badge>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 sm:px-6">
        {err && (
          <Card className="border-rose-900 bg-rose-950/30">
            <CardContent className="p-4 text-sm text-rose-300">数据加载失败：{err}</CardContent>
          </Card>
        )}
        {!data && !err && (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {Array.from({ length: 8 }).map((_, i) => <div key={i} className="h-24 animate-pulse rounded-lg bg-zinc-900/60" />)}
          </div>
        )}

        {data && c && s && (
          <Tabs defaultValue="v17">
            <TabsList className="max-w-full justify-start overflow-x-auto bg-zinc-900">
              <TabsTrigger value="v17" className="data-[state=active]:bg-zinc-800">v17 · HTF 冠军看板</TabsTrigger>
              <TabsTrigger value="v18" className="data-[state=active]:bg-zinc-800">v18 · 分支A 静态止损实验室</TabsTrigger>
              <TabsTrigger value="r1" className="data-[state=active]:bg-zinc-800">R1 · Reservoir 引擎实验室</TabsTrigger>
              <TabsTrigger value="v19" className="data-[state=active]:bg-zinc-800">v19 · 路线A/B 终局审判</TabsTrigger>
              <TabsTrigger value="r2" className="data-[state=active]:bg-zinc-800">R2 · 自我进化×四用途审判</TabsTrigger>
            </TabsList>
            <TabsContent value="v17" className="mt-4">
          <div className="space-y-6">
            {/* 冠军 KPI */}
            <section aria-label="冠军指标">
              <div className="mb-3 flex flex-wrap items-center gap-2">
                <TrendingUp className="h-4 w-4 text-emerald-400" aria-hidden />
                <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-400">冠军 · M30 Donchian-55 停损单 × 只做多 × ATR 闸门 × 持 5 交易日</h2>
              </div>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
                <KpiCard icon={<Crosshair className="h-3.5 w-3.5" />} label="总 PnL (0.3×ATR)" value={fmtUSD(s.total)} sub="基准 $732.7 的 4.1×" tone="pos" />
                <KpiCard icon={<LineChart className="h-3.5 w-3.5" />} label="笔均" value={`$${s.avg.toFixed(2)}`} sub={`t 统计 ${s.t_stat} · 159 笔`} tone="pos" />
                <KpiCard icon={<CheckCircle2 className="h-3.5 w-3.5" />} label="逐年" value="5/5 全正" sub="2022–2026 每年 15–38 笔" tone="pos" />
                <KpiCard icon={<Gauge className="h-3.5 w-3.5" />} label="2026 占比" value={pct(s.share2026)} sub="历史基准为 88.9%" />
                <KpiCard icon={<TrendingUp className="h-3.5 w-3.5" />} label="Sharpe / maxDD" value={`${s.sharpe} / ${fmtUSD(s.maxdd)}`} sub={`胜率 ${pct(s.win_rate * 100)} · PLR 2.19`} />
                <KpiCard icon={<AlertTriangle className="h-3.5 w-3.5" />} label="成本耐受" value="×3 成本存活" sub={`中位成本 $${s.med_cost}/笔 vs 笔均 $${s.avg}`} tone="warn" />
              </div>
            </section>

            {/* MT5 EA 下载 */}
            <section aria-label="MT5 EA 下载">
              <Card className="border-emerald-900/70 bg-emerald-950/20">
                <CardContent className="flex flex-col gap-4 p-4 sm:flex-row sm:items-center sm:justify-between sm:p-5">
                  <div className="min-w-0 space-y-1.5">
                    <div className="flex items-center gap-2">
                      <Download className="h-4 w-4 shrink-0 text-emerald-400" aria-hidden />
                      <h2 className="text-sm font-semibold text-zinc-200">MT5 独立复验 · v17 冠军 EA（<span className="font-mono">XAUUSD_v17_Champion.mq5</span>）</h2>
                    </div>
                    <p className="text-xs leading-relaxed text-zinc-400">
                      纯规则零模型，四条规则逐语义移植（Donchian-55 停损单 · 只做多 · ATR 季中位闸门 · 240 根时间出场 · 串行单仓，含跳空保守成交与“出场当根不再进场”等引擎细节）。
                      默认 <b className="text-zinc-200">0.01 手 = 1 盎司</b>，测试器里的 $ 数字 ≈ 研究报告口径；结束后日志自动打印逐年汇总。
                    </p>
                    <p className="font-mono text-[11px] leading-relaxed text-zinc-500">
                      测试器设置：品种 XAUUSD · 周期 M30 · 2022.01.01–2026.07.31 · 模式「每笔报价（真实报价）」或最低「1 分钟 OHLC」<br />
                      预期落点：约 159±10% 笔 · 逐年全正 · 总额 $2,980–$3,208（研究 pess03–base 区间）· 2026-07-17 之后 = 样本外
                    </p>
                  </div>
                  <Button asChild className="shrink-0 bg-emerald-600 text-white hover:bg-emerald-500">
                    <a href="/data/XAUUSD_v17_Champion.mq5" download="XAUUSD_v17_Champion.mq5">
                      <Download className="mr-2 h-4 w-4" aria-hidden />下载 .mq5（21 KB）
                    </a>
                  </Button>
                </CardContent>
              </Card>
            </section>

            {/* 年度 + 权益 */}
            <div className="grid gap-6 lg:grid-cols-5">
              <section className="lg:col-span-2" aria-label="年度盈亏">
                <Card className="h-full border-zinc-800 bg-zinc-900/40">
                  <CardHeader className="pb-2"><CardTitle className="text-sm text-zinc-300">年度盈亏 · 悲观成本口径</CardTitle></CardHeader>
                  <CardContent>
                    <YearBars by={s.by_year} n={s.n_by_year} />
                    <Separator className="my-4 bg-zinc-800" />
                    <EquityChart months={s.months} />
                  </CardContent>
                </Card>
              </section>

              <section className="lg:col-span-3" aria-label="成本阶梯与消融">
                <Card className="h-full border-zinc-800 bg-zinc-900/40">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm text-zinc-300">成本阶梯（HTF 论题检验）+ 规则消融</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <LadderRow name="base" s={c.base} note="$0.50/RT 固定" />
                    <LadderRow name="pess03" s={c.pess03} note="0.3×ATR/RT 悲观" />
                    <LadderRow name="pess05" s={c.pess05} note="0.5×ATR/RT 极端" />
                    <p className="text-xs leading-relaxed text-zinc-500">
                      base→0.5×ATR 仅回撤 <b className="text-zinc-300">−13.5%</b>，五年在全部三档成本下保持全正——
                      「点差+滑点只占 H1 单笔波动一小部分」的论题被坐实（实际数据点差中位仅 $0.16/RT）。M1 时代同口径下边缘为 $0.095/笔且 δ=0.1 即蒸发。
                    </p>
                    <Separator className="bg-zinc-800" />
                    <AblationCard ablation={data.ablation} />
                  </CardContent>
                </Card>
              </section>
            </div>

            {/* 策略规格 + 逐笔 */}
            <section aria-label="策略与逐笔">
              <Tabs defaultValue="spec">
                <TabsList className="bg-zinc-900">
                  <TabsTrigger value="spec" className="data-[state=active]:bg-zinc-800">策略规格</TabsTrigger>
                  <TabsTrigger value="trades" className="data-[state=active]:bg-zinc-800">逐笔 159 笔</TabsTrigger>
                  <TabsTrigger value="finalists" className="data-[state=active]:bg-zinc-800">28 强全表</TabsTrigger>
                </TabsList>
                <TabsContent value="spec" className="mt-3">
                  <div className="grid gap-3 md:grid-cols-2">
                    <Card className="border-zinc-800 bg-zinc-900/40">
                      <CardHeader className="pb-2"><CardTitle className="text-sm text-zinc-300">交易规则（4 条）</CardTitle></CardHeader>
                      <CardContent className="space-y-2.5 text-sm leading-relaxed text-zinc-400">
                        <p><b className="text-zinc-200">1 · 进场</b> — 55 根 M30 高点通道（海龟 S2）：价格触及通道线上方即停损单成交（跳空按开盘价，保守）；仅做多。</p>
                        <p><b className="text-zinc-200">2 · 闸门</b> — ATR14 高于其自身过去一季度中位数才允许开仓（波动扩张期；避开 2022 型低波绞肉）。</p>
                        <p><b className="text-zinc-200">3 · 出场</b> — 持满 5 个交易日（240 根 M30），第 240 根开盘市价平仓。无止损（尾部风险见披露）。</p>
                        <p><b className="text-zinc-200">4 · 纪律</b> — 串行单仓，持仓期内忽略新信号；成本 = 0.3×ATR/RT 从每笔扣除。</p>
                        <Separator className="bg-zinc-800" />
                        <p className="text-xs text-zinc-500">
                          信号只用已完成 bar；吊灯止损若出现在 k 根，其价位由 k−1 及更早信息决定；同根多事件保守排序（止损优先）；数据末端强平。
                          全部 6,480 个网格配置经纯 Python 与 numba 双引擎逐位互验。
                        </p>
                      </CardContent>
                    </Card>
                    <Card className="border-zinc-800 bg-zinc-900/40">
                      <CardHeader className="pb-2"><CardTitle className="text-sm text-zinc-300">为什么是它 — 三方证据</CardTitle></CardHeader>
                      <CardContent className="space-y-2.5 text-sm leading-relaxed text-zinc-400">
                        <p><b className="text-emerald-400">① 结构</b> — 赢家集中在「突破后持 5 日」：黄金 H1 级突破后动量延续半衰期约 5 个交易日，与央行购金/降息周期的单边宏观叙事一致。</p>
                        <p><b className="text-emerald-400">② 稳健</b> — 三档成本全正 + 邻域平台 50% + 跨 TF（H1 双变体同过全部门）。原始总分更高的 M15 don40s（$3,661）因平台 0% 被预注册规则剔除。</p>
                        <p><b className="text-emerald-400">③ 对照</b> — 同引擎 M1 真实冠军 $2,599/27,265 笔（笔均 $0.095，2024 微负）；HTF 冠军以 1/171 的笔数赚 1.15× 总量，笔均质量 <b className="text-zinc-200">197×</b>。</p>
                        <p className="text-xs text-zinc-500">历史 $732.7 基准本身引擎存疑（v16 披露同款回望写法）且 2026 占 88.9%，此处仅作参照刻度。</p>
                      </CardContent>
                    </Card>
                  </div>
                </TabsContent>
                <TabsContent value="trades" className="mt-3">
                  <TradesTable trades={c.trades} />
                </TabsContent>
                <TabsContent value="finalists" className="mt-3">
                  <p className="mb-2 text-xs text-zinc-500">
                    预注册筛选：0.3×ATR 悲观成本下逐年全正 ∧ 笔均 ≥ $5 ∧ 笔数 ≥ 150 ∧ 2026 占比 ≤ 70%；「阶梯」= base/pess03/pess05 三档全正，「平台」= 邻域参数全正率。★行 = 最终冠军。
                  </p>
                  <FinalistsTable list={data.finalists} />
                </TabsContent>
              </Tabs>
            </section>

            {/* 失败启示录 */}
            <section aria-label="实验发现与披露">
              <Card className="border-zinc-800 bg-zinc-900/40">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm text-zinc-300">实验发现与诚实披露</CardTitle>
                </CardHeader>
                <CardContent className="grid gap-4 text-sm leading-relaxed text-zinc-400 md:grid-cols-2">
                  <div className="space-y-2.5">
                    <p><b className="text-zinc-200">吊灯再次阵亡</b> — 2.5/3/3.5×ATR 追踪止损在 HTF 正确结算下无一进入 top10；「ATR 移动出场」的直觉在 M1 与 HTF 同时被证伪。真正的主导维度是<b className="text-emerald-400">持仓期的时间结构</b>（5 日 ≫ 其他）。</p>
                    <p><b className="text-zinc-200">2022 是突破系统的绞肉年</b> — 无闸门时 210 个初筛配置中 188 个在 2022 为负（金价全年区间震荡）；ATR 闸门（规则版 Regime Gate）在冠军几何上把 2022 从 +$89 提到 +$243、把 2026 从 −$670 救回 +$859，这正是路线 A 的最小可行版本。</p>
                    <p><b className="text-zinc-200">空头为负贡献</b> — 悲观成本下 HTF 空头突破几乎全线为负；「双向波动收割」是 M1 时代的 bug 幻影，HTF 真实结构是「顺势单边」。</p>
                  </div>
                  <div className="space-y-2.5">
                    <p className="flex gap-2"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" aria-hidden />
                      <span><b className="text-amber-400">无止损尾部风险 → 已闭环检验</b> — 冠军纯靠 5 日时间出场，单笔最差 −$168、p10 −$49。v18（分支 A）对 44 个静态灾难止损变体的完整检验证明：价格底线无法在保留 $2,500+ 的前提下封死尾部——时间出场本身就是该结构的灾难止损，尾部管理应上移到组合层仓位规模。详见 v18 tab。</span></p>
                    <p className="flex gap-2"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" aria-hidden />
                      <span><b className="text-amber-400">样本内网格选择</b> — 6,480 配置全样本扫描，平台/阶梯/跨 TF 是缓解而非消除；2026 仅半年数据；159 笔 t=3.13 达标但非厚样本。live 前应前向纸面验证 ≥ 3 个月。</span></p>
                    <p className="flex gap-2"><CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" aria-hidden />
                      <span><b className="text-emerald-400">ML 的位置已让出</b> — 按用户路线：规则基准已立，下一步 ML 不做方向预测，而在多资产（DXY/US10Y/VIX/金银比）上做周/日级 Regime Filter 替换 ATR 闸门。本轮仅完成规则版。</span></p>
                  </div>
                </CardContent>
              </Card>
            </section>

            {/* 跑者 */}
            <section aria-label="H1 变体">
              <div className="mb-3 flex items-center gap-2">
                <Layers className="h-4 w-4 text-zinc-500" aria-hidden />
                <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-400">同门 H1 变体（同过全部门）</h2>
              </div>
              <div className="grid gap-3 md:grid-cols-2">
                {data.runners.map((r) => (
                  <Card key={String(r.spec.name)} className="border-zinc-800 bg-zinc-900/40">
                    <CardHeader className="pb-2">
                      <CardTitle className="font-mono text-xs text-zinc-300">
                        {r.spec.tf} · {ENTRY_LABEL[String(r.spec.entry)]} · {DIR_LABEL[String(r.spec.dir)]} · {GATE_LABEL[String(r.spec.gate)]} · {EXIT_LABEL[String(r.spec.exit)]}
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="flex items-baseline gap-3">
                        <span className="font-mono text-xl font-semibold tabular-nums text-emerald-400">{fmtUSD(r.pess03.total)}</span>
                        <span className="text-xs text-zinc-500">{r.pess03.trades} 笔 · 笔均 ${r.pess03.avg.toFixed(2)} · 26占比 {r.pess03.share2026.toFixed(1)}% · t={r.pess03.t_stat}</span>
                      </div>
                      <div className="mt-3"><YearBars by={r.pess03.by_year} n={r.pess03.n_by_year} /></div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </section>
          </div>
            </TabsContent>
            <TabsContent value="v18" className="mt-4">
              {v18 ? <V18Lab v18={v18} /> : (
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                  {Array.from({ length: 8 }).map((_, i) => <div key={i} className="h-24 animate-pulse rounded-lg bg-zinc-900/60" />)}
                </div>
              )}
            </TabsContent>
            <TabsContent value="r1" className="mt-4">
              {res ? <ReservoirTab data={res} /> : (
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                  {Array.from({ length: 8 }).map((_, i) => <div key={i} className="h-24 animate-pulse rounded-lg bg-zinc-900/60" />)}
                </div>
              )}
            </TabsContent>
            <TabsContent value="v19" className="mt-4">
              {v19 ? <V19Tab data={v19} /> : (
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                  {Array.from({ length: 8 }).map((_, i) => <div key={i} className="h-24 animate-pulse rounded-lg bg-zinc-900/60" />)}
                </div>
              )}
            </TabsContent>
            <TabsContent value="r2" className="mt-4">
              {r2 ? <R2Tab data={r2} /> : (
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                  {Array.from({ length: 8 }).map((_, i) => <div key={i} className="h-24 animate-pulse rounded-lg bg-zinc-900/60" />)}
                </div>
              )}
            </TabsContent>
          </Tabs>
        )}
      </main>

      <footer className="mt-auto border-t border-zinc-800/80 bg-zinc-950/95 pb-[env(safe-area-inset-bottom)]">
        <div className="mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-2 px-4 py-4 text-xs text-zinc-600 sm:px-6">
          <span>v17→v18→R1→v19→R2 · HTF Breakout + Streaming RLS-Reservoir + 能量审判 + 自我进化 · 生成于 {data?.meta.generated ?? ''}</span>
          <span>{data?.meta.cost_note ?? ''}</span>
        </div>
      </footer>
    </div>
  )
}
