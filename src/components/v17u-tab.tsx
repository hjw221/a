'use client'

/**
 * v17u-tab.tsx — v17 终极版看板 (第7 tab)
 * 冠军(159笔/$2,980) + 2条因果能量纪律 → 109笔/$3,389/笔均$31.10/逐年全正/maxDD减半
 * 数据: public/data/v17u.json · MQ5: public/data/XAUUSD_v17_Ultimate.mq5
 */
import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { AlertTriangle, CheckCircle2, Download, FlaskConical, Radar, ShieldCheck, TrendingUp } from 'lucide-react'

interface Summary {
  total: number; trades: number; avg: number; std: number; t_stat: number
  win_rate: number; p10: number; p90: number; worst: number; best: number
  by_year: Record<string, number>; n_by_year: Record<string, number>
  share2026: number; sharpe: number; maxdd: number
  med_cost: number; med_atr: number; total54: number
}
interface Trade {
  entry_time: string; exit_time: string; side: string
  entry_px: number; exit_px: number; cost: number; pnl: number
  hold_bars: number; reason: string; atr_entry: number; nr_entry?: number
}
interface BlockArm { theta: number; block: number; total: number; n: number; avg: number; t: number; wr: number; all_pos: boolean; yr: Record<string, number>; dd: number; worst: number; share26: number }
interface V17UData {
  meta: { generated: string; title: string; question: string; answer: string; engine: string; cost: string; selection_rule: string }
  origin: { source: string; nr_def: string; quintile: { q: number; n: number; avg: number; wr: number }[]; dilution: string }
  champion: {
    spec: { name: string; tf: string; entry: string; dir: string; gate: string; energy: string; block: string; exit: string; serial: boolean }
    base: Summary; pess03: Summary; pess05: Summary
  }
  base_champion: Summary
  grid_block: BlockArm[]
  honesty: string[]
  trades: Trade[]
}

const fmtUSD = (v: number, d = 0) =>
  `${v < 0 ? '−' : ''}$${Math.abs(v).toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d })}`
const YEARS = ['2022', '2023', '2024', '2025', '2026']

export function V17UTab() {
  const [data, setData] = useState<V17UData | null>(null)
  const [err, setErr] = useState(false)
  useEffect(() => {
    fetch('/data/v17u.json').then(r => r.ok ? r.json() : Promise.reject())
      .then(setData).catch(() => setErr(true))
  }, [])
  if (err) return <div className="p-6 text-sm text-zinc-500">v17u.json 加载失败</div>
  if (!data) return <div className="p-6 text-sm text-zinc-500">加载中…</div>
  const s = data.champion.pess03
  const b = data.champion.base
  const bc = data.base_champion
  const maxAvg = Math.max(...data.grid_block.map(a => a.avg))

  return (
    <div className="space-y-4">
      {/* 标题 */}
      <div className="flex flex-wrap items-center gap-2">
        <Radar className="h-4 w-4 text-emerald-400" aria-hidden />
        <span className="text-sm font-semibold text-zinc-100">{data.meta.title}</span>
        <Badge variant="outline" className="border-emerald-800 bg-emerald-950/50 text-emerald-300">6 条规则 · 零 ML</Badge>
        <Badge variant="outline" className="border-zinc-700 bg-zinc-900 text-zinc-400">{data.meta.engine}</Badge>
      </div>
      <p className="text-xs leading-relaxed text-zinc-500">
        {data.meta.answer}。{data.meta.cost}。选择规则: {data.meta.selection_rule}
      </p>

      {/* KPI: 终极版 vs 冠军 */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <Kpi label="总 PnL" value={fmtUSD(s.total)} sub={`冠军 ${fmtUSD(bc.total)} · +${(((s.total - bc.total) / bc.total) * 100).toFixed(1)}%`} tone="pos" />
        <Kpi label="笔均" value={`$${s.avg.toFixed(2)}`} sub={`冠军 $${bc.avg.toFixed(2)} · +${(((s.avg - bc.avg) / bc.avg) * 100).toFixed(0)}%`} tone="pos" />
        <Kpi label="交易数" value={`${s.trades}`} sub={`冠军 ${bc.trades} 笔`} tone="neutral" />
        <Kpi label="t 统计" value={s.t_stat.toFixed(2)} sub={`冠军 ${bc.t_stat.toFixed(2)}`} tone="pos" />
        <Kpi label="胜率" value={`${(s.win_rate * 100).toFixed(1)}%`} sub={`冠军 ${(bc.win_rate * 100).toFixed(1)}%`} tone="pos" />
        <Kpi label="maxDD" value={fmtUSD(s.maxdd)} sub={`冠军 ${fmtUSD(bc.maxdd)} · 减半`} tone="warn" />
      </div>

      {/* 年度对比 */}
      <Card className="border-zinc-800 bg-zinc-950/60">
        <CardHeader className="pb-2"><CardTitle className="flex items-center gap-2 text-sm text-zinc-300"><TrendingUp className="h-3.5 w-3.5" aria-hidden />逐年对比 · 0.3×ATR 悲观成本 (pess03)</CardTitle></CardHeader>
        <CardContent>
          <div className="grid grid-cols-5 gap-2 text-center text-xs">
            {YEARS.map(y => {
              const u = s.by_year[y] ?? 0
              const c = bc.by_year[y] ?? 0
              return (
                <div key={y} className="rounded border border-zinc-800 bg-zinc-900/60 p-2">
                  <div className="text-zinc-500">{y}</div>
                  <div className={`mt-1 font-mono font-semibold ${u > 0 ? 'text-emerald-400' : 'text-red-400'}`}>{fmtUSD(u, 1)}</div>
                  <div className="mt-1 text-[10px] text-zinc-600">冠军 {fmtUSD(c, 0)}</div>
                </div>
              )
            })}
          </div>
          <div className="mt-2 text-[11px] text-zinc-500">
            成本阶梯: base {fmtUSD(b.total)} / pess03 {fmtUSD(s.total)} / pess05 {fmtUSD(data.champion.pess05.total)} — 三档全部逐年全正 (−8.7%)
          </div>
        </CardContent>
      </Card>

      {/* 两条新规则 */}
      <div className="grid gap-3 md:grid-cols-2">
        <Card className="border-zinc-800 bg-zinc-950/60">
          <CardHeader className="pb-2"><CardTitle className="text-sm text-zinc-300">规则 5 · 低能入场过滤 (θ=1.15)</CardTitle></CardHeader>
          <CardContent className="space-y-2 text-xs text-zinc-400">
            <p><span className="text-zinc-200">NR(t) = 最近4根M15对数收益²和 ÷ 同序列EMA96基线</span> — 纯因果无学习。决策时刻 NR &lt; 1.15 才允许挂停损单。</p>
            <p className="text-zinc-500">来源: v19-H3 诊断 — 基线 159 笔入场 NR 五分位断崖:</p>
            <div className="grid grid-cols-5 gap-1 text-center text-[10px]">
              {data.origin.quintile.map(q => (
                <div key={q.q} className="rounded bg-zinc-900/80 p-1.5">
                  <div className="text-zinc-500">Q{q.q}</div>
                  <div className={`font-mono font-semibold ${q.avg >= 25 ? 'text-emerald-400' : q.avg >= 15 ? 'text-amber-400' : 'text-zinc-400'}`}>${q.avg.toFixed(1)}</div>
                  <div className="text-zinc-600">wr {(q.wr * 100).toFixed(0)}%</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
        <Card className="border-zinc-800 bg-zinc-950/60">
          <CardHeader className="pb-2"><CardTitle className="text-sm text-zinc-300">规则 6 · 高能触发跳过 → 32根封锁再武装</CardTitle></CardHeader>
          <CardContent className="space-y-2 text-xs text-zinc-400">
            <p>触发线被触 ∧ 闸门开 ∧ NR≥θ → 跳过本轮突破并封锁 32 根 M30 (16h) 的一切入场, 封锁期内再遇高能触发则顺延。</p>
            <p className="text-zinc-500">机理: 静态 NR 过滤的重跑被「同一能量事件内的追单次级突破」稀释 — 新交易 75 笔笔均仅 $11.5 (保留交易 $35.1)。封锁机制整体消灭追单。</p>
            <div className="rounded border border-zinc-800 bg-zinc-900/60 p-2 text-[10px] leading-relaxed text-zinc-500">
              「均盈利24+」溯源: v19-H3 诊断过滤视图 84笔/$24.52 — 重跑部署口径为 $31.10 (封锁后更优)。
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 封锁机制网格 */}
      <Card className="border-zinc-800 bg-zinc-950/60">
        <CardHeader className="pb-2"><CardTitle className="text-sm text-zinc-300">机制 2 网格 · θ × 封锁根数 (pess03)</CardTitle></CardHeader>
        <CardContent>
          <div className="max-h-72 overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-zinc-950 text-zinc-500">
                <tr className="text-left">
                  <th className="px-2 py-1.5">θ</th><th className="px-2 py-1.5">封锁</th>
                  <th className="px-2 py-1.5 text-right">总量</th><th className="px-2 py-1.5 text-right">笔数</th>
                  <th className="px-2 py-1.5 text-right">笔均</th><th className="px-2 py-1.5 text-right">t</th>
                  <th className="px-2 py-1.5 text-right">胜率</th><th className="px-2 py-1.5 text-right">maxDD</th>
                  <th className="px-2 py-1.5">逐年全正</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {data.grid_block.map((a, i) => {
                  const isChamp = a.theta === 1.15 && a.block === 32
                  return (
                    <tr key={i} className={isChamp ? 'bg-emerald-950/40 text-emerald-300' : 'text-zinc-400'}>
                      <td className="px-2 py-1.5">{a.theta.toFixed(2)}</td>
                      <td className="px-2 py-1.5">{a.block}根{isChamp ? ' ★' : ''}</td>
                      <td className="px-2 py-1.5 text-right">{fmtUSD(a.total, 1)}</td>
                      <td className="px-2 py-1.5 text-right">{a.n}</td>
                      <td className="px-2 py-1.5 text-right">
                        <span className="relative">
                          ${a.avg.toFixed(2)}
                        </span>
                      </td>
                      <td className="px-2 py-1.5 text-right">{a.t.toFixed(2)}</td>
                      <td className="px-2 py-1.5 text-right">{(a.wr * 100).toFixed(1)}%</td>
                      <td className="px-2 py-1.5 text-right">{fmtUSD(a.dd, 0)}</td>
                      <td className="px-2 py-1.5">{a.all_pos ? <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" aria-hidden /> : <AlertTriangle className="h-3.5 w-3.5 text-amber-500" aria-hidden />}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-[11px] text-zinc-500">
            平台: θ∈[0.9,1.25] × 封锁∈[2,96] 全部逐年全正、笔均 $19~34 (封锁 64/96 笔均更高但 n&lt;100 被预注册门槛排除; 封锁 128 崩塌)。
          </p>
        </CardContent>
      </Card>

      {/* MQ5 下载 + 最近交易 */}
      <div className="grid gap-3 md:grid-cols-2">
        <Card className="border-emerald-900/60 bg-emerald-950/20">
          <CardHeader className="pb-2"><CardTitle className="flex items-center gap-2 text-sm text-emerald-300"><Download className="h-3.5 w-3.5" aria-hidden />MT5 独立复验 EA</CardTitle></CardHeader>
          <CardContent className="space-y-3 text-xs text-zinc-400">
            <a
              href="/data/XAUUSD_v17_Ultimate.mq5"
              download="XAUUSD_v17_Ultimate.mq5"
              className="inline-flex items-center gap-2 rounded-md border border-emerald-700 bg-emerald-900/40 px-3 py-2 font-mono text-[11px] text-emerald-200 transition hover:bg-emerald-900/70"
            >
              <Download className="h-3.5 w-3.5" aria-hidden />XAUUSD_v17_Ultimate.mq5
            </a>
            <p>用法: 品种 XAUUSD · 周期 M30 · 日期 2022.01.01–2026.07.31 · 模式「每笔报价(真实报价)」或「1分钟 OHLC」· 0.01 手 = 1 盎司。</p>
            <p>判定: <span className="text-emerald-300">逐年全正 ∧ 笔均 ≥ $24</span> 为复验通过 (预期 $29~33)。2026-07-17 之后 = 真正样本外。InpEnergyFilter=false 可 A/B 回到纯冠军。</p>
            <p className="text-zinc-500">MQ5 保真度已验证: 620根M15滚动EMA vs 全历史EMA — 53,353 决策点 θ 分类 100.0000% 一致, 完整重跑 $3,368.8/108笔/全正。</p>
          </CardContent>
        </Card>
        <Card className="border-zinc-800 bg-zinc-950/60">
          <CardHeader className="pb-2"><CardTitle className="text-sm text-zinc-300">最近 30 笔 (pess03)</CardTitle></CardHeader>
          <CardContent>
            <ScrollArea className="max-h-64">
              <table className="w-full text-[11px]">
                <thead className="text-zinc-500"><tr className="text-left">
                  <th className="py-1">入场</th><th className="py-1">出场</th>
                  <th className="py-1 text-right">入场价</th><th className="py-1 text-right">PnL</th><th className="py-1 text-right">NR</th>
                </tr></thead>
                <tbody className="font-mono text-zinc-400">
                  {data.trades.slice().reverse().map((t, i) => (
                    <tr key={i}>
                      <td className="py-1">{t.entry_time.slice(5)}</td>
                      <td className="py-1">{t.exit_time.slice(5)}</td>
                      <td className="py-1 text-right">{t.entry_px.toFixed(1)}</td>
                      <td className={`py-1 text-right font-semibold ${t.pnl > 0 ? 'text-emerald-400' : 'text-red-400'}`}>{t.pnl > 0 ? '+' : ''}{t.pnl.toFixed(1)}</td>
                      <td className="py-1 text-right text-zinc-500">{t.nr_entry !== undefined ? t.nr_entry.toFixed(2) : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>

      {/* 诚实条款 */}
      <Card className="border-amber-900/50 bg-amber-950/10">
        <CardHeader className="pb-2"><CardTitle className="flex items-center gap-2 text-sm text-amber-300"><FlaskConical className="h-3.5 w-3.5" aria-hidden />诚实条款</CardTitle></CardHeader>
        <CardContent>
          <ul className="space-y-1.5 text-xs text-zinc-400">
            {data.honesty.map((h, i) => (
              <li key={i} className="flex gap-2"><ShieldCheck className="mt-0.5 h-3 w-3 shrink-0 text-amber-500/70" aria-hidden /><span>{h}</span></li>
            ))}
          </ul>
        </CardContent>
      </Card>

      <Separator className="bg-zinc-800" />
      <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-[10px] text-zinc-600">
        研究档案: remote-ops-record/v17u_20261003/ · GitHub reservoir-engine 分支 · 断言链: 基线复现 $2,980.2/159 逐位 ✓ + block=0 复现静态过滤臂 ✓
      </motion.p>
    </div>
  )
}

function Kpi({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone: 'pos' | 'warn' | 'neutral' }) {
  const color = tone === 'pos' ? 'text-emerald-400' : tone === 'warn' ? 'text-amber-400' : 'text-zinc-200'
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-3">
      <div className="text-[10px] uppercase tracking-wider text-zinc-500">{label}</div>
      <div className={`mt-1 font-mono text-lg font-semibold ${color}`}>{value}</div>
      {sub && <div className="mt-0.5 text-[10px] text-zinc-600">{sub}</div>}
    </div>
  )
}
