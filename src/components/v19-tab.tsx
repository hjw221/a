'use client'

import { useMemo } from 'react'
import { motion } from 'framer-motion'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import {
  AlertTriangle, CheckCircle2, Crosshair, FlaskConical, Gauge, ShieldCheck,
  Skull, Split, Timer, XCircle, Zap,
} from 'lucide-react'

/* ================= types ================= */
interface Summ {
  total: number; trades: number; avg: number; t_stat: number
  win_rate: number; maxdd: number; share2026: number
  by_year: Record<string, number>; n_by_year: Record<string, number>
  months?: Record<string, number>
}
interface ArmRowT {
  name: string; label: string; kind: string
  total: number; trades: number; avg: number; t_stat: number
  win_rate: number; maxdd: number; share2026: number
  by_year: Record<string, number>; n_by_year: Record<string, number>
  all_pos: boolean; coverage?: Record<string, number>; months?: Record<string, number>
}
interface H3Row {
  label: string; corr_pnl: number
  r_hi: { n: number; avg: number; wr: number } | null
  r_lo: { n: number; avg: number; wr: number } | null
  r_2022_hi: { n: number; avg: number; wr: number } | null
  r_2022_lo: { n: number; avg: number; wr: number } | null
}
export interface V19Data {
  meta: {
    generated: string; engine: string; branch: string
    decision: string; decision_reasons: string[]
    prereg: string; cost: string; oracle_ms_per_bar: number
  }
  round1: {
    oracle: { ic_var: number; ic_skew: number; ic_var_by_year: Record<string, number>; r_quantiles: Record<string, number> }
    repro: { total: number; expected: number; ok: boolean }
    base: Summ; base_none: Summ
    base_coverage: Record<string, number>
    grid: ArmRowT[]
    union: Summ; union_coverage: Record<string, number>
    intersect: Summ; intersect_coverage: Record<string, number>
    best_r1: { name: string } & Summ & { coverage: Record<string, number> }
    overlap: { p_r1_on: number; p_atr_on: number; p_both: number; p_r1_only: number; p_atr_only: number; p_both_giv_atr: number }
    route_b: Summ & { side_total: number; freq_per_week: number }
    oracle_ms_per_bar: number
  }
  round2: {
    nature: string
    h1_grid: ArmRowT[]; h1_long: ArmRowT[]
    h2_grid: ArmRowT[]; h2_long: ArmRowT[]
    intersect_h16: Summ; intersect_h16_with: string
    intersect_nc: Summ; intersect_nc_with: string
    nc_best: ArmRowT; h16_best: ArmRowT
    h3: { r4: H3Row; r16: H3Row; nr: H3Row }
  }
  equity: { base: [string, number][]; best_r1: [string, number][]; nc_best: [string, number][] }
  verdict: {
    route_a: { ok: boolean; pre_registered_rule: string; best_arm: string; best_total: number; delta: number; union_total: number; tested_variants: number }
    route_b: { ok: boolean; total: number; side_total: number; freq_per_week: number; expected_freq: string; actual_freq: string; t_stat: number; by_year: Record<string, number> }
    round2: { h1: string; h2: string; h3: string }
    root_cause: string
    champion_survives: string
    where_r1_belongs: string[]
  }
}

/* ================= helpers ================= */
const fmtUSD = (v: number, digits = 0) =>
  `${v < 0 ? '−' : ''}$${Math.abs(v).toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits })}`
const YEARS = ['2022', '2023', '2024', '2025', '2026']
const rise = (i: number) => ({ initial: { opacity: 0, y: 14 }, animate: { opacity: 1, y: 0 }, transition: { delay: 0.05 * i, duration: 0.4 } })
const pct = (v: number) => `${(v * 100).toFixed(1)}%`

/* ================= KPI ================= */
function Kpi({ icon, label, value, sub, tone = 'default' }: {
  icon: React.ReactNode; label: string; value: string; sub?: string
  tone?: 'default' | 'pos' | 'warn' | 'neg'
}) {
  const toneCls = tone === 'pos' ? 'text-emerald-400' : tone === 'warn' ? 'text-amber-400' : tone === 'neg' ? 'text-rose-400' : 'text-zinc-100'
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

/* ================= 年度小柱 ================= */
function YearMini({ by, n }: { by: Record<string, number>; n?: Record<string, number> }) {
  const maxAbs = Math.max(...YEARS.map((y) => Math.abs(by[y] ?? 0)), 1)
  return (
    <div className="grid grid-cols-5 gap-1.5" role="img" aria-label="年度盈亏">
      {YEARS.map((y) => {
        const v = by[y] ?? 0
        const h = Math.max((Math.abs(v) / maxAbs) * 100, 4)
        return (
          <div key={y} className="flex flex-col items-center gap-1">
            <span className={`font-mono text-[10px] font-semibold tabular-nums ${v >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
              {v >= 0 ? '+' : '−'}{Math.abs(Math.round(v))}
            </span>
            <div className="flex h-10 w-full items-end justify-center rounded bg-zinc-900/60">
              <div className={`w-3/5 rounded-sm ${v >= 0 ? 'bg-emerald-500/80' : 'bg-rose-500/80'}`} style={{ height: `${h}%` }} />
            </div>
            <span className="text-[9px] text-zinc-600">{y.slice(2)}{n?.[y] ? `·${n[y]}` : ''}</span>
          </div>
        )
      })}
    </div>
  )
}

/* ================= 三线权益曲线 ================= */
function EquityMulti({ series }: { series: { data: [string, number][]; color: string; name: string }[] }) {
  const chart = useMemo(() => {
    const valid = series.filter((s) => s.data.length > 1)
    if (!valid.length) return null
    const N = Math.max(...valid.map((s) => s.data.length))
    const all = valid.flatMap((s) => s.data.map((d) => d[1]))
    const min = Math.min(...all, 0), max = Math.max(...all, 0)
    const W = 860, H = 220, PAD = 10
    const x = (i: number, n: number) => PAD + (i / (n - 1)) * (W - 2 * PAD)
    const y = (v: number) => H - PAD - ((v - min) / (max - min || 1)) * (H - 2 * PAD)
    const paths = valid.map((s) => ({
      color: s.color, name: s.name,
      d: s.data.map((p, i) => `${x(i, s.data.length).toFixed(1)},${y(p[1]).toFixed(1)}`).join(' '),
      last: s.data[s.data.length - 1][1],
    }))
    return { W, H, PAD, min, max, y, paths, N }
  }, [series])
  if (!chart) return null
  return (
    <div className="w-full">
      <svg viewBox={`0 0 ${chart.W} ${chart.H}`} className="h-48 w-full sm:h-56" role="img" aria-label="月度累计盈亏对比" preserveAspectRatio="none">
        <line x1={chart.PAD} x2={chart.W - chart.PAD} y1={chart.y(0)} y2={chart.y(0)} stroke="rgb(113 113 122 / 0.4)" strokeDasharray="3 4" strokeWidth="1" />
        {chart.paths.map((p) => (
          <polyline key={p.name} points={p.d} fill="none" stroke={p.color} strokeWidth="2" strokeLinejoin="round" />
        ))}
      </svg>
      <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-[11px] text-zinc-500">
        {chart.paths.map((p) => (
          <span key={p.name} className="flex items-center gap-1.5">
            <span className="inline-block h-0.5 w-4 rounded" style={{ background: p.color }} />
            {p.name} <span className="font-mono tabular-nums text-zinc-400">{fmtUSD(p.last)}</span>
          </span>
        ))}
      </div>
    </div>
  )
}

/* ================= 臂表格行 ================= */
function ArmRow({ arm, highlight }: { arm: ArmRowT; highlight?: 'base' | 'best' | 'ctrl' }) {
  const border = highlight === 'base' ? 'border-emerald-900/60' : highlight === 'best' ? 'border-rose-900/50' : highlight === 'ctrl' ? 'border-amber-900/50' : 'border-zinc-800'
  return (
    <div className={`grid grid-cols-1 gap-2 rounded-lg border ${border} bg-zinc-900/40 p-3 lg:grid-cols-[13rem_9rem_1fr] lg:items-center`}>
      <div className="min-w-0">
        <div className="truncate font-mono text-xs font-semibold text-zinc-200">{arm.name}</div>
        <div className="text-[10px] text-zinc-500">{arm.label}</div>
      </div>
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5">
        <span className={`font-mono text-lg font-semibold tabular-nums ${arm.total >= 2980 ? 'text-emerald-400' : arm.total >= 2000 ? 'text-amber-400' : 'text-rose-400'}`}>
          {fmtUSD(arm.total)}
        </span>
        <span className="text-[11px] text-zinc-500">{arm.trades}笔 · 均${arm.avg.toFixed(1)} · t={arm.t_stat.toFixed(2)}</span>
        <Badge variant="outline" className={`ml-auto hidden lg:ml-2 lg:inline-flex ${arm.all_pos ? 'border-emerald-800 text-emerald-400' : 'border-rose-800 text-rose-400'}`}>
          {arm.all_pos ? '全正' : '有负年'}
        </Badge>
      </div>
      <div className="min-w-0"><YearMini by={arm.by_year} n={arm.n_by_year} /></div>
    </div>
  )
}

/* ================= H3 行 ================= */
function H3Line({ row }: { row: H3Row }) {
  const cs = (c: { n: number; avg: number; wr: number } | null) =>
    c ? `${fmtUSD(c.avg, 1)} · ${c.n}笔` : '—'
  return (
    <div className="grid grid-cols-[7.5rem_4.5rem_1fr_1fr] items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900/40 p-2.5 text-xs">
      <span className="font-mono font-semibold text-zinc-300">{row.label}</span>
      <span className={`font-mono font-semibold tabular-nums ${Math.abs(row.corr_pnl) < 0.1 ? 'text-zinc-500' : 'text-amber-400'}`}>
        {row.corr_pnl.toFixed(3)}
      </span>
      <span className="text-zinc-400">高能入场: <span className="font-mono tabular-nums">{cs(row.r_hi)}</span></span>
      <span className="text-zinc-400">低能入场: <span className="font-mono tabular-nums">{cs(row.r_lo)}</span></span>
    </div>
  )
}

/* ================= 主组件 ================= */
export function V19Tab({ data }: { data: V19Data }) {
  const r1 = data.round1
  const r2 = data.round2
  const v = data.verdict
  const ovl = r1.overlap

  return (
    <div className="space-y-4">
      {/* ===== 终局横幅 ===== */}
      <motion.div {...rise(0)}>
        <Card className="border-emerald-900/60 bg-zinc-900/60">
          <CardContent className="p-4 sm:p-5">
            <div className="flex flex-wrap items-center gap-2">
              <ShieldCheck className="h-5 w-5 text-emerald-400" aria-hidden />
              <h2 className="text-base font-semibold text-zinc-100">路线A/B 终局审判 — 双双证伪，v17 冠军存活</h2>
              <Badge variant="outline" className="border-emerald-800 text-emerald-400">2 轮 · 24 个变体 · 0 幸存</Badge>
            </div>
            <p className="mt-2.5 text-sm leading-relaxed text-zinc-400">
              事前决策：<b className="text-zinc-200">路线A为主干</b>（R1 能量神谕替换 v17 的 63 天 ATR 中位闸门），
              并把路线B 的三大机制（施密特触发 · 迟滞带 · 最小时锁）移植到闸门状态机；路线B 本体作为同框架对照臂诚实运行。
              预注册判据下路线A <b className="text-rose-400">失败</b>：最佳学习闸门 {fmtUSD(v.route_a.best_total)}（{fmtUSD(v.route_a.delta, 1)} vs 基线），
              且 6/6 出现负年份；第二轮假设检验连 4h 视界与「无学习」对照一并检验后，
              结论收敛为「<b className="text-zinc-200">时间常数错位</b>」——冠军闸门是 63 天慢持续状态变量，
              任何 ≤24h 的能量先见（学习与否）都无法替代它。
            </p>
          </CardContent>
        </Card>
      </motion.div>

      {/* ===== KPI ===== */}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Kpi icon={<Gauge className="h-4 w-4" aria-hidden />} label="冠军基线 · atrmed" value={fmtUSD(r1.base.total)}
             sub={`${r1.base.trades} 笔 · 笔均 $${r1.base.avg.toFixed(2)} · t=${r1.base.t_stat.toFixed(2)} · 复现断言 ${r1.repro.ok ? 'OK' : 'FAIL'}`} tone="pos" />
        <Kpi icon={<Crosshair className="h-4 w-4" aria-hidden />} label="最佳学习闸门 (R1)" value={fmtUSD(r1.best_r1.total)}
             sub={`${r1.best_r1.name} · ${r1.best_r1.trades} 笔 · 笔均 $${r1.best_r1.avg.toFixed(1)} · 2026 转负`} tone="neg" />
        <Kpi icon={<FlaskConical className="h-4 w-4" aria-hidden />} label="最佳无学习对照 (nowcast)" value={fmtUSD(r2.nc_best.total)}
             sub={`${r2.nc_best.name} · 逐年全正 · 仍 < 基线 ${fmtUSD(r1.base.total)}`} tone="warn" />
        <Kpi icon={<Skull className="h-4 w-4" aria-hidden />} label="路线B · 独立事件驱动" value={fmtUSD(v.route_b.total)}
             sub={`${v.route_b.freq_per_week.toFixed(1)} 笔/周 (预期 3–5) · t=${v.route_b.t_stat.toFixed(2)} · 逐年全负`} tone="neg" />
      </div>

      {/* ===== 决策卡 + 权益对比 ===== */}
      <div className="grid gap-4 lg:grid-cols-[1fr_1fr]">
        <motion.div {...rise(1)}>
          <Card className="h-full border-zinc-800 bg-zinc-900/40">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-sm text-zinc-300">
                <Split className="h-4 w-4 text-zinc-500" aria-hidden />事前为什么选路线A（而非B）
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2.5 text-sm leading-relaxed text-zinc-400">
              {data.meta.decision_reasons.map((r, i) => (
                <p key={i} className="flex gap-2">
                  <span className="font-mono text-xs text-emerald-400/70">0{i + 1}</span>
                  <span>{r}</span>
                </p>
              ))}
              <Separator className="bg-zinc-800" />
              <p className="text-xs text-zinc-500">
                对用户方案的唯一改动：路线B 的施密特触发/迟滞带/时锁纪律<b className="text-zinc-300">作用对象从仓位改为闸门</b>
                （闸门在阈值附近闪烁会让 v17 错过恰在闪烁窗口的入场）；路线B 本体原样实现（|ŝ|≥0.40 ∧ R̂≥1.0 触发 · 锁16根 · 能量回落/64根帽出场 · 固定1oz）作为对照臂。
              </p>
            </CardContent>
          </Card>
        </motion.div>
        <motion.div {...rise(1)}>
          <Card className="h-full border-zinc-800 bg-zinc-900/40">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-zinc-300">权益曲线 — 基线 vs 学习闸门 vs 无学习对照（pess03）</CardTitle>
            </CardHeader>
            <CardContent>
              <EquityMulti series={[
                { data: data.equity.base, color: 'rgb(52 211 153)', name: '冠军 atrmed' },
                { data: data.equity.best_r1, color: 'rgb(251 113 133)', name: 'R1 学习闸门' },
                { data: data.equity.nc_best, color: 'rgb(251 191 36)', name: 'nowcast 无学习' },
              ]} />
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* ===== 第一轮 · 预注册检验 ===== */}
      <section aria-label="第一轮预注册检验">
        <div className="mb-2 flex flex-wrap items-center gap-2">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-400">第一轮 · 预注册检验（跑前锁规则）</h2>
          <Badge variant="outline" className="border-zinc-700 text-zinc-400">复现断言 {fmtUSD(r1.repro.total)} = 档案值 ✓</Badge>
        </div>
        <p className="mb-2 text-xs text-zinc-500">{data.meta.prereg}</p>
        <div className="space-y-2">
          <ArmRow arm={{ ...({ name: 'BASE_atrmed', label: 'v17 冠军原样 · ATR>季度中位', kind: 'base' } as ArmRowT), ...r1.base, by_year: r1.base.by_year, n_by_year: r1.base.n_by_year, all_pos: true }} highlight="base" />
          {r1.grid.map((a) => <ArmRow key={a.name} arm={a} highlight={a.name === r1.best_r1.name ? 'best' : undefined} />)}
          <ArmRow arm={{ ...({ name: 'UNION', label: `R1(${r1.best_r1.name}) ∪ atrmed · 地板保护`, kind: 'combo' } as ArmRowT), ...r1.union, by_year: r1.union.by_year, n_by_year: r1.union.n_by_year, all_pos: YEARS.every((y) => (r1.union.by_year[y] ?? 0) > 0) }} />
          <ArmRow arm={{ ...({ name: 'INTERSECT', label: `R1(${r1.best_r1.name}) ∩ atrmed · 最大净化`, kind: 'combo' } as ArmRowT), ...r1.intersect, by_year: r1.intersect.by_year, n_by_year: r1.intersect.n_by_year, all_pos: YEARS.every((y) => (r1.intersect.by_year[y] ?? 0) > 0) }} />
        </div>
      </section>

      {/* ===== 诊断三卡 ===== */}
      <div className="grid gap-4 md:grid-cols-3">
        <motion.div {...rise(1)}>
          <Card className="h-full border-zinc-800 bg-zinc-900/40">
            <CardHeader className="pb-2"><CardTitle className="text-sm text-zinc-300">学习层无罪 — oracle 质量</CardTitle></CardHeader>
            <CardContent className="text-sm text-zinc-400">
              <div className="flex items-baseline gap-3">
                <span className="font-mono text-2xl font-semibold text-emerald-400">{r1.oracle.ic_var.toFixed(3)}</span>
                <span className="text-xs text-zinc-500">IC_var（1h 能量前瞻）</span>
              </div>
              <div className="mt-2 space-y-1 font-mono text-[11px] tabular-nums text-zinc-500">
                {YEARS.map((y) => (
                  <div key={y} className="flex items-center gap-2">
                    <span className="w-9">{y}</span>
                    <div className="h-1.5 flex-1 rounded bg-zinc-800">
                      <div className="h-1.5 rounded bg-emerald-500/70" style={{ width: `${Math.min((r1.oracle.ic_var_by_year[y] ?? 0) / 0.5) * 100}%` }} />
                    </div>
                    <span className="w-10 text-right">{(r1.oracle.ic_var_by_year[y] ?? 0).toFixed(2)}</span>
                  </div>
                ))}
              </div>
              <p className="mt-2 text-xs leading-relaxed text-zinc-500">逐年 0.25~0.42 稳定为真（IC_skew {r1.oracle.ic_skew.toFixed(3)} 与 R1 阶段一一致）。失败不在预测质量，在<b className="text-zinc-300">架构位置</b>。</p>
            </CardContent>
          </Card>
        </motion.div>
        <motion.div {...rise(2)}>
          <Card className="h-full border-zinc-800 bg-zinc-900/40">
            <CardHeader className="pb-2"><CardTitle className="text-sm text-zinc-300">失败签名 — 闸门仅 49% 同意冠军</CardTitle></CardHeader>
            <CardContent className="text-sm text-zinc-400">
              <div className="space-y-2 font-mono text-xs tabular-nums">
                {[
                  ['双双 ON', ovl.p_both, 'bg-emerald-500/70'],
                  ['仅 R1 ON（坏交易来源）', ovl.p_r1_only, 'bg-rose-500/70'],
                  ['仅 atrmed ON（错过赢家）', ovl.p_atr_only, 'bg-amber-500/70'],
                ].map(([label, val, color]) => (
                  <div key={String(label)} className="flex items-center gap-2">
                    <span className="w-40 shrink-0 text-zinc-500">{label}</span>
                    <div className="h-2 flex-1 rounded bg-zinc-800">
                      <div className={`h-2 rounded ${color}`} style={{ width: `${(Number(val) / 0.8) * 100}%` }} />
                    </div>
                    <span className="w-12 text-right text-zinc-300">{pct(Number(val))}</span>
                  </div>
                ))}
              </div>
              <p className="mt-2 text-xs leading-relaxed text-zinc-500">
                R1-only 窗口的交易笔均 <b className="text-rose-400">${r1.best_r1.avg.toFixed(1)}</b>（基线 ${r1.base.avg.toFixed(1)}）；
                atrmed-only 窗口装着 2026 年的大赢家（基线 2026 +$940 → 学习闸门 −$10~−$571）。
              </p>
            </CardContent>
          </Card>
        </motion.div>
        <motion.div {...rise(3)}>
          <Card className="h-full border-zinc-800 bg-zinc-900/40">
            <CardHeader className="pb-2"><CardTitle className="flex items-center gap-2 text-sm text-zinc-300"><Timer className="h-4 w-4 text-zinc-500" aria-hidden />算力承诺兑现</CardTitle></CardHeader>
            <CardContent className="text-sm text-zinc-400">
              <div className="flex items-baseline gap-3">
                <span className="font-mono text-2xl font-semibold text-zinc-100">{data.meta.oracle_ms_per_bar.toFixed(2)}<span className="text-sm text-zinc-500"> ms/bar</span></span>
              </div>
              <p className="mt-2 text-xs leading-relaxed text-zinc-500">
                本地单线程（OMP=1）逐 bar 诚实计时，含 ESN 递推 + GMM 自组织 + RLS 闭环更新全部三层，
                10.7 万根 M15 全程在线无停机。&lt;2ms / &lt;50MB 承诺在融合场景下同样成立。
              </p>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* ===== 第二轮 ===== */}
      <section aria-label="第二轮假设检验">
        <div className="mb-2 flex flex-wrap items-center gap-2">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-400">第二轮 · 事后假设检验（明确标注非预注册）</h2>
          <Badge variant="outline" className="border-amber-800 text-amber-400">post-hoc · 如实全报</Badge>
        </div>
        <div className="grid gap-4 lg:grid-cols-2">
          <motion.div {...rise(0)}>
            <Card className="h-full border-zinc-800 bg-zinc-900/40">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-zinc-300">H1 · 视界假设 — 4h 前瞻（用户规格上限）<span className="ml-2 text-xs text-rose-400">否定</span></CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {r2.h1_grid.map((a) => <ArmRow key={a.name} arm={a} />)}
                {r2.h1_long.map((a) => <ArmRow key={a.name} arm={a} />)}
                <p className="text-xs leading-relaxed text-zinc-500">
                  10/10 无臂超基线；长时锁（48/96根）变体反而全部转负 2026。R1 阶段一早已给出死因：IC_var 在 24h 视界归零——可预测视界物理上够不到 63 天 regime。
                </p>
              </CardContent>
            </Card>
          </motion.div>
          <motion.div {...rise(1)}>
            <Card className="h-full border-amber-900/50 bg-zinc-900/40">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-zinc-300">H2 · 决定性对照 — 无学习 nowcast<span className="ml-2 text-xs text-amber-400">先见为负增量</span></CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {r2.h2_grid.map((a) => <ArmRow key={a.name} arm={a} highlight={a.name === r2.nc_best.name ? 'ctrl' : undefined} />)}
                {r2.h2_long.map((a) => <ArmRow key={a.name} arm={a} />)}
                <p className="text-xs leading-relaxed text-zinc-500">
                  因果 nowcast（当前实现方差/EMA 基线，零学习零先见）同一施密特机器：<b className="text-amber-300">{fmtUSD(r2.nc_best.total)}</b>，
                  超过全部学习前瞻闸门（{fmtUSD(1437)}~{fmtUSD(2448)}）且逐年全正——但仍不敌基线。波动率的短程最优预测是<b className="text-zinc-300">持续性</b>，储层的先见在闸门尺度只添噪声。
                </p>
              </CardContent>
            </Card>
          </motion.div>
        </div>
        <div className="mt-4 grid gap-4 lg:grid-cols-2">
          <motion.div {...rise(0)}>
            <Card className="h-full border-zinc-800 bg-zinc-900/40">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-zinc-300">H3 · 入场条件诊断 — 能否按预测能量「选交易」</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <H3Line row={r2.h3.r4} />
                <H3Line row={r2.h3.r16} />
                <H3Line row={r2.h3.nr} />
                <p className="text-xs leading-relaxed text-zinc-500">
                  基线 159 笔入场时点的预测能量与单笔 PnL 相关仅 0.05~0.06（≈0）；nowcast 条件甚至为负（高当前能量 → 突破笔均更差：${r2.h3.nr.r_hi ? r2.h3.nr.r_hi.avg.toFixed(1) : '—'} vs ${r2.h3.nr.r_lo ? r2.h3.nr.r_lo.avg.toFixed(1) : '—'}）。
                  2022 净化假设同样落空：1h 前瞻下 2022 高/低能入场笔均几乎相同。净化 2022 假突破的功劳属于 63 天 regime 闸门，不属于 1h 能量。
                </p>
              </CardContent>
            </Card>
          </motion.div>
          <motion.div {...rise(1)}>
            <Card className="h-full border-zinc-800 bg-zinc-900/40">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-zinc-300">第二轮 · intersect 保护变体（h16 / nowcast × atrmed）</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-1 gap-2 rounded-lg border border-zinc-800 bg-zinc-900/40 p-3 sm:grid-cols-[11rem_1fr] sm:items-center">
                  <div>
                    <div className="font-mono text-xs font-semibold text-zinc-200">h16 ∩ atrmed</div>
                    <div className="text-[10px] text-zinc-500">{r2.intersect_h16_with}</div>
                  </div>
                  <div className="flex flex-wrap items-baseline gap-x-3">
                    <span className="font-mono text-lg font-semibold tabular-nums text-amber-400">{fmtUSD(r2.intersect_h16.total)}</span>
                    <span className="text-[11px] text-zinc-500">{r2.intersect_h16.trades} 笔 · 笔均 ${r2.intersect_h16.avg.toFixed(2)}</span>
                    <YearMini by={r2.intersect_h16.by_year} n={r2.intersect_h16.n_by_year} />
                  </div>
                </div>
                <div className="grid grid-cols-1 gap-2 rounded-lg border border-zinc-800 bg-zinc-900/40 p-3 sm:grid-cols-[11rem_1fr] sm:items-center">
                  <div>
                    <div className="font-mono text-xs font-semibold text-zinc-200">nowcast ∩ atrmed</div>
                    <div className="text-[10px] text-zinc-500">{r2.intersect_nc_with}</div>
                  </div>
                  <div className="flex flex-wrap items-baseline gap-x-3">
                    <span className="font-mono text-lg font-semibold tabular-nums text-amber-400">{fmtUSD(r2.intersect_nc.total)}</span>
                    <span className="text-[11px] text-zinc-500">{r2.intersect_nc.trades} 笔 · 笔均 ${r2.intersect_nc.avg.toFixed(2)}（最高笔均）</span>
                    <YearMini by={r2.intersect_nc.by_year} n={r2.intersect_nc.n_by_year} />
                  </div>
                </div>
                <p className="text-xs leading-relaxed text-zinc-500">
                  交叉变体全部五年全正、笔均最高达 ${r2.intersect_nc.avg.toFixed(2)}（基线 ${r1.base.avg.toFixed(2)}），
                  但总量仍低于基线——净化是真实存在的，只是它属于慢 regime 而非快能量。intersect 是本轮唯一「接近打平」的家族，作为后续研究线索保留。
                </p>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </section>

      {/* ===== 路线B 卡 ===== */}
      <motion.div {...rise(1)}>
        <Card className="border-rose-900/50 bg-zinc-900/40">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm text-zinc-300">
              <Skull className="h-4 w-4 text-rose-400" aria-hidden />路线B · 独立事件驱动 — 用户规格 vs 真实结果
            </CardTitle>
          </CardHeader>
          <CardContent className="grid gap-4 md:grid-cols-[1fr_1fr]">
            <div className="space-y-2 text-sm text-zinc-400">
              <p><b className="text-zinc-200">用户规格（原样实现）</b> — 施密特触发：|ŝ|≥0.40 ∧ 能量扩张 R̂≥1.0 才开固定 1oz；最小持仓锁 16 根 M15（4h）禁止反向；出场 = 能量回落基线或 64 根时间帽。</p>
              <div className="grid grid-cols-2 gap-2 font-mono text-xs">
                <div className="rounded border border-zinc-800 bg-zinc-900/60 p-2">
                  <div className="text-zinc-500">预期频率</div>
                  <div className="mt-1 text-base text-zinc-300">3–5 笔/周</div>
                </div>
                <div className="rounded border border-rose-900/50 bg-zinc-900/60 p-2">
                  <div className="text-zinc-500">实际频率</div>
                  <div className="mt-1 text-base text-rose-400">{v.route_b.freq_per_week.toFixed(1)} 笔/周</div>
                </div>
                <div className="rounded border border-zinc-800 bg-zinc-900/60 p-2">
                  <div className="text-zinc-500">预期：笔均跨过点差</div>
                  <div className="mt-1 text-base text-zinc-300">→ 正期望</div>
                </div>
                <div className="rounded border border-rose-900/50 bg-zinc-900/60 p-2">
                  <div className="text-zinc-500">实际：笔均</div>
                  <div className="mt-1 text-base text-rose-400">−$1.05 · t=−3.45</div>
                </div>
              </div>
            </div>
            <div className="space-y-2">
              <YearMini by={v.route_b.by_year} n={r1.route_b.n_by_year} />
              <p className="text-xs leading-relaxed text-zinc-500">
                3,126 笔 · 逐年全负 · v17 同屋 RT 成本下 {fmtUSD(v.route_b.total)}；R1 原始逐边成本口径 {fmtUSD(v.route_b.side_total)}。
                |ŝ|≥0.40 的死区并不稀有（方向头有厚尾），事件驱动纪律把换手从连续版的 11,092 单位砍到 3,126 笔——
                但<b className="text-rose-400">纪律救不了没有 alpha 的信号</b>：IC 0.02 的方向预测在任何执行离散化下都无法跨越点差。
                路线B 的正确遗产是其机制（已移植到闸门），不是其本体。
              </p>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* ===== 根因 + 去向 ===== */}
      <div className="grid gap-4 md:grid-cols-2">
        <motion.div {...rise(1)}>
          <Card className="h-full border-zinc-800 bg-zinc-900/40">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-sm text-zinc-300"><XCircle className="h-4 w-4 text-rose-400" aria-hidden />根因 — 时间常数错位</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm leading-relaxed text-zinc-400">
              <p>{v.root_cause}</p>
              <Separator className="bg-zinc-800" />
              <p className="text-xs text-zinc-500">
                v17 的「ATR &gt; 63 天中位」不是等待被更聪明预测器替换的滞后估计量——它是<b className="text-zinc-300">慢持续 regime 状态变量</b>，
                与 5 日持仓的时间结构同频。把它换成毫秒级前瞻，等于用秒表指挥四季耕作。
              </p>
            </CardContent>
          </Card>
        </motion.div>
        <motion.div {...rise(2)}>
          <Card className="h-full border-zinc-800 bg-zinc-900/40">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-sm text-zinc-300"><Zap className="h-4 w-4 text-amber-400" aria-hidden />R1 真 alpha 的正确去向</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm leading-relaxed text-zinc-400">
              {v.where_r1_belongs.map((s, i) => (
                <p key={i} className="flex gap-2">
                  <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" aria-hidden />
                  <span>{s}</span>
                </p>
              ))}
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* ===== 诚实披露 ===== */}
      <motion.div {...rise(2)}>
        <Card className="border-zinc-800 bg-zinc-900/40">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm text-zinc-300">
              <AlertTriangle className="h-4 w-4 text-amber-400" aria-hidden />诚实披露
            </CardTitle>
          </CardHeader>
          <CardContent className="grid gap-4 text-sm leading-relaxed text-zinc-400 md:grid-cols-2">
            <div className="space-y-2">
              <p><b className="text-zinc-200">两轮性质不同</b> — 第一轮为预注册（跑前锁定判据与网格）；第二轮为诊断驱动的事后假设检验，全部 20 个臂结果如实报告，未做任何挑选性呈现。</p>
              <p><b className="text-zinc-200">网格规模与过拟合面</b> — 学习闸门共 12 配置 + 组合 4 + 对照 10。即便最优臂也未超基线，故不存在「挑选赢家」问题；但 intersect 家族的「笔均提升」属事后观察，不可作为部署依据。</p>
            </div>
            <div className="space-y-2">
              <p><b className="text-zinc-200">闸门语义保真</b> — gate30[k] 取最后先于 o30[k] 收盘的 M15 状态，与冠军 gate[k]=(ATR[k−1]&gt;med[k−1]) 同一因果约定；基线臂与 v17 档案逐位复现（$2,980.2 断言通过）。</p>
              <p><b className="text-zinc-200">数据与成本口径</b> — 与 v17/v18/R1 同一 M1 清洗源（2022-01~2026-07）；成本三档同屋。oracle 本地单线程 {data.meta.oracle_ms_per_bar.toFixed(2)} ms/bar。2026 仅半年；所有结论限于样本内。</p>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  )
}
