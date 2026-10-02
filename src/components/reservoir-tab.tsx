'use client'

import { useMemo } from 'react'
import { motion } from 'framer-motion'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Activity, Brain, Cpu, Database, Gauge, GitBranch, LineChart, Zap,
} from 'lucide-react'

/* ================= types ================= */
interface ResRow {
  year: number; pnl: number; gross: number; cost: number; sharpe: number
  maxdd: number; avg_pos: number; turnover: number; ic_skew: number; bars: number
}
interface MsStat { mean: number; p90: number; p99: number; max: number }
interface ResRun {
  overall: {
    pnl: number; gross: number; cost: number; sharpe: number; maxdd: number
    avg_pos: number; turnover: number; ic_skew: number; ic_var: number; plr: number
  }
  rows: ResRow[]
  equity: [string, number][]
  state_frac: Record<string, number[]>
}
interface ReservoirData {
  meta: {
    engine: string; bars: number; span: [string, string]; warm: number; horizon: number
    convention: string; lam: number; n_res: number; spectral: number; density: number
    process_peak_mb: number; atr_by_year: Record<string, number>; generated: string
  }
  primary: {
    metrics: ResRun
    timing: { wall_s: number; bars: number; esn_ms: MsStat; gmm_ms: MsStat; rls_ms: MsStat; total_ms: MsStat }
    engine_mb: number; K_final: number; W_l2: number[]; K_curve: number[]
  }
  seeds: { seed: number; pnl: number; ic_skew: number; ic_var: number; sharpe: number; maxdd: number }[]
  lam_sweep: { lam: number; pnl: number; ic_skew: number; note: string }[]
  nres_sweep: { n_res: number; pnl: number; ic_skew: number; ms_per_bar: number; engine_mb: number }[]
  shuffle: { pnl: number; ic_skew: number; ic_var: number; cost: number }
  momentum: { total: number; by_year: Record<string, number>; sharpe: number; cost: number }
  probes: { name: string; horizon: number; note: string; pnl: number; ic_skew: number; ic_var: number; sharpe: number; cost: number; turnover: number }[]
  verdict: { headline: string; points: string[]; next: string[] }
}

/* ================= helpers ================= */
const fmtUSD = (v: number, digits = 0) =>
  `${v < 0 ? '−' : ''}$${Math.abs(v).toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits })}`
const YEARS = ['2022', '2023', '2024', '2025', '2026']
const rise = (i: number) => ({ initial: { opacity: 0, y: 14 }, animate: { opacity: 1, y: 0 }, transition: { delay: 0.05 * i, duration: 0.4 } })
const STATE_LABEL = ['压缩态', '单边态', '高波混沌', '常规态']
const STATE_COLOR = ['bg-sky-500/80', 'bg-emerald-500/80', 'bg-rose-500/80', 'bg-zinc-500/80']

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

/* ================= charts ================= */
function EquitySVG({ equity }: { equity: [string, number][] }) {
  const pts = useMemo(() => equity.map((e) => e[1]), [equity])
  if (pts.length < 2) return null
  const W = 860, H = 210, PAD = 10
  const min = Math.min(...pts, 0), max = Math.max(...pts, 0)
  const x = (i: number) => PAD + (i / (pts.length - 1)) * (W - 2 * PAD)
  const y = (v: number) => H - PAD - ((v - min) / (max - min || 1)) * (H - 2 * PAD)
  const path = pts.map((v, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ')
  const zero = y(0)
  const y0 = pts[0], yT = pts[pts.length - 1]
  return (
    <div className="overflow-x-auto" role="img" aria-label="流式引擎权益曲线">
      <svg viewBox={`0 0 ${W} ${H}`} className="h-52 w-full min-w-[560px]">
        <line x1={PAD} x2={W - PAD} y1={zero} y2={zero} stroke="#52525b" strokeDasharray="4 4" strokeWidth={1} />
        <path d={path} fill="none" stroke={yT >= y0 ? '#34d399' : '#fb7185'} strokeWidth={1.6} />
      </svg>
      <div className="flex justify-between px-1 text-[10px] text-zinc-600">
        <span>{equity[0][0].slice(0, 10)}</span>
        <span>{equity[Math.floor(equity.length / 2)][0].slice(0, 10)}</span>
        <span>{equity[equity.length - 1][0].slice(0, 10)}</span>
      </div>
    </div>
  )
}

function Sparkline({ data, color = '#34d399', label, sub }: { data: number[]; color?: string; label: string; sub?: string }) {
  if (data.length < 2) return null
  const W = 420, H = 90, PAD = 6
  const min = Math.min(...data), max = Math.max(...data)
  const x = (i: number) => PAD + (i / (data.length - 1)) * (W - 2 * PAD)
  const y = (v: number) => H - PAD - ((v - min) / (max - min || 1)) * (H - 2 * PAD)
  const path = data.map((v, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ')
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-medium text-zinc-400">{label}</span>
        <span className="font-mono text-[11px] tabular-nums text-zinc-500">
          {data[0].toFixed(1)} → {data[data.length - 1].toFixed(1)}{sub ? ` · ${sub}` : ''}
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="mt-1 h-24 w-full" role="img" aria-label={label}>
        <path d={path} fill="none" stroke={color} strokeWidth={1.5} />
      </svg>
    </div>
  )
}

/* ================= main tab ================= */
export function ReservoirTab({ data }: { data: ReservoirData }) {
  const p = data.primary
  const o = p.metrics.overall
  const tm = p.timing?.total_ms
  const byYear = useMemo(() => {
    const m: Record<string, number> = {}
    p.metrics.rows.forEach((r) => { m[String(r.year)] = r.pnl })
    return m
  }, [p.metrics.rows])

  return (
    <div className="space-y-4">
      {/* 结论横幅 */}
      <motion.div {...rise(0)}>
        <Card className="border-amber-900/70 bg-gradient-to-br from-amber-950/30 to-zinc-950">
          <CardContent className="p-4 sm:p-5">
            <div className="flex items-start gap-3">
              <Brain className="mt-0.5 h-5 w-5 shrink-0 text-amber-400" aria-hidden />
              <div className="min-w-0">
                <div className="text-sm font-semibold text-amber-200 sm:text-base">{data.verdict.headline}</div>
                <ul className="mt-2 space-y-1.5">
                  {data.verdict.points.map((pt) => (
                    <li key={pt} className="flex gap-2 text-xs leading-relaxed text-zinc-300">
                      <span className="mt-0.5 text-amber-500/70">▸</span><span className="min-w-0">{pt}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* KPI 行 */}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Kpi icon={<LineChart className="h-3.5 w-3.5" />} label="全期净 PnL (1oz)" value={fmtUSD(o.pnl)} sub={`毛利 ${fmtUSD(o.gross)} − 成本 ${fmtUSD(o.cost)}`} tone={o.pnl >= 0 ? 'pos' : 'neg'} />
        <Kpi icon={<Activity className="h-3.5 w-3.5" />} label="预测技能 IC" value={`方向 ${o.ic_skew.toFixed(3)} · 能量 ${o.ic_var.toFixed(3)}`} sub={`${data.meta.bars.toLocaleString()} 根 M15 · 视界 ${data.meta.horizon} bar`} tone={o.ic_var > 0.2 ? 'pos' : 'default'} />
        <Kpi icon={<Cpu className="h-3.5 w-3.5" />} label="单 bar 耗时 (单核)" value={`${(tm?.mean ?? 0).toFixed(2)} ms`} sub={`p99 ${(tm?.p99 ?? 0).toFixed(2)} ms · 无反向传播`} tone="pos" />
        <Kpi icon={<Database className="h-3.5 w-3.5" />} label="引擎内存" value={`${p.engine_mb.toFixed(1)} MB`} sub={`进程峰值 ${data.meta.process_peak_mb.toFixed(0)} MB (含数据装载)`} tone="pos" />
      </div>

      {/* 三层架构 + 权益曲线 */}
      <div className="grid gap-4 lg:grid-cols-5">
        <motion.div {...rise(1)} className="min-w-0 lg:col-span-2">
          <Card className="h-full border-zinc-800 bg-zinc-900/60">
            <CardHeader className="pb-2"><CardTitle className="flex items-center gap-2 text-sm"><Brain className="h-4 w-4 text-emerald-400" />三层无梯度闭环</CardTitle></CardHeader>
            <CardContent className="space-y-2.5 text-xs">
              {[
                ['L1 · 动态记忆层', `ESN 储层 ${data.meta.n_res} 神经元 · 稀疏度 ${(data.meta.density * 100).toFixed(0)}% · 谱半径 ${data.meta.spectral}`, 'W_res/W_in 初始化后永久锁死，tanh 矩阵乘法提取非线性时间拓扑'],
                ['L2 · 状态自组织层', `Streaming GMM · 终态 ${p.K_final} 组件`, '惊异度(Mahalanobis²)驱动组件自生长，无需预设状态数'],
                ['L3 · 突触自演化层', `遗忘 RLS λ=${data.meta.lam} · W∈R²ˣ${data.meta.n_res + 5}`, '每根 Bar 闭环递推，δ正则+trace上限防协方差缠绕'],
              ].map(([t, s, d]) => (
                <div key={t} className="rounded-md border border-zinc-800 bg-zinc-950/60 p-2.5">
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
                    <span className="font-semibold text-zinc-200">{t}</span>
                    <span className="min-w-0 break-words font-mono text-[10px] text-emerald-400/90">{s}</span>
                  </div>
                  <p className="mt-1 text-[11px] leading-relaxed text-zinc-500">{d}</p>
                </div>
              ))}
              <div className="rounded-md border border-emerald-900/50 bg-emerald-950/20 p-2.5">
                <span className="font-semibold text-emerald-300">主动推理目标</span>
                <p className="mt-1 text-[11px] leading-relaxed text-zinc-400">
                  预测未来 {data.meta.horizon} 根的联合能量分布：相对实现方差 + 方向动量偏斜。无固定止盈止损标签；
                  压缩→爆发时 ŝkew 增大→暴露非对称释放，噪音态流形塌缩自动离场。预测能量做波动率倒数缩放（v18 启示内置）。
                </p>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div {...rise(2)} className="min-w-0 lg:col-span-3">
          <Card className="h-full border-zinc-800 bg-zinc-900/60">
            <CardHeader className="pb-2">
              <CardTitle className="flex flex-wrap items-center justify-between gap-2 text-sm">
                <span className="flex items-center gap-2"><Zap className="h-4 w-4 text-emerald-400" />流式权益曲线（单次通过 · 无重训）</span>
                <Badge variant="outline" className="border-zinc-700 text-[10px] text-zinc-400">{data.meta.span[0].slice(0, 10)} → {data.meta.span[1].slice(0, 10)}</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <EquitySVG equity={p.metrics.equity} />
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* 学习动力学: W 收敛 + GMM 演化 + 状态占比 */}
      <div className="grid gap-4 lg:grid-cols-3">
        <motion.div {...rise(1)} className="min-w-0">
          <Card className="h-full border-zinc-800 bg-zinc-900/60">
            <CardHeader className="pb-2"><CardTitle className="text-sm">W_out 权重范数 ‖W‖₂ 演化</CardTitle></CardHeader>
            <CardContent>
              <Sparkline data={p.W_l2} label="终身进化（非收敛到不动点，持续追踪市场结构）" color="#34d399" />
              <p className="mt-1 text-[11px] leading-relaxed text-zinc-500">
                遗忘因子 λ 使旧行情权重指数淡化——2024 牛市结构被 2026 高波结构持续替换，无灾难性遗忘亦无停机重训。
              </p>
            </CardContent>
          </Card>
        </motion.div>
        <motion.div {...rise(2)} className="min-w-0">
          <Card className="h-full border-zinc-800 bg-zinc-900/60">
            <CardHeader className="pb-2"><CardTitle className="text-sm">GMM 组件数自生长</CardTitle></CardHeader>
            <CardContent>
              <Sparkline data={p.K_curve} label="状态数 K(t)（惊异度触发 · 冷却 96 bar）" color="#38bdf8" />
              <div className="mt-2 space-y-1.5">
                {YEARS.map((y) => {
                  const fr = p.metrics.state_frac[y]
                  if (!fr) return null
                  return (
                    <div key={y} className="flex items-center gap-2">
                      <span className="w-9 shrink-0 font-mono text-[10px] text-zinc-500">{y}</span>
                      <div className="flex h-3.5 flex-1 overflow-hidden rounded-sm">
                        {fr.map((v, k) => <div key={k} className={STATE_COLOR[k]} style={{ width: `${v * 100}%` }} title={`${STATE_LABEL[k]} ${(v * 100).toFixed(0)}%`} />)}
                      </div>
                    </div>
                  )
                })}
                <div className="flex flex-wrap gap-2 pt-1">
                  {STATE_LABEL.map((s, k) => (
                    <span key={s} className="flex items-center gap-1 text-[10px] text-zinc-500">
                      <span className={`inline-block h-2 w-2 rounded-sm ${STATE_COLOR[k]}`} />{s}
                    </span>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
        <motion.div {...rise(3)} className="min-w-0">
          <Card className="h-full border-zinc-800 bg-zinc-900/60">
            <CardHeader className="pb-2"><CardTitle className="text-sm">年度净 PnL</CardTitle></CardHeader>
            <CardContent>
              <div className="grid grid-cols-5 gap-2">
                {YEARS.map((y) => {
                  const v = byYear[y] ?? 0
                  const maxAbs = Math.max(...YEARS.map((yy) => Math.abs(byYear[yy] ?? 0)), 1)
                  return (
                    <div key={y} className="flex flex-col items-center gap-1">
                      <span className={`font-mono text-[10px] font-semibold tabular-nums ${v >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>{fmtUSD(v)}</span>
                      <div className="flex h-20 w-full items-end justify-center rounded bg-zinc-950/70">
                        <div className={`w-3/5 rounded-sm ${v >= 0 ? 'bg-emerald-500/80' : 'bg-rose-500/80'}`} style={{ height: `${Math.max((Math.abs(v) / maxAbs) * 100, 3)}%` }} />
                      </div>
                      <span className="text-[10px] text-zinc-600">{y}</span>
                    </div>
                  )
                })}
              </div>
              <p className="mt-2 text-[11px] leading-relaxed text-zinc-500">
                年度 ATR 环境：{YEARS.map((y) => `${y}/${data.meta.atr_by_year[y]?.toFixed(2) ?? '—'}`).join(' · ')}（M15 $）
              </p>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* 逐年明细表 */}
      <motion.div {...rise(1)} className="min-w-0">
        <Card className="border-zinc-800 bg-zinc-900/60">
          <CardHeader className="pb-2"><CardTitle className="text-sm">逐年明细（净 PnL / 毛利 / 成本 / IC）</CardTitle></CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[640px] text-xs">
                <thead>
                  <tr className="border-b border-zinc-800 text-left text-[10px] uppercase tracking-wider text-zinc-500">
                    <th className="py-2 pr-3">年份</th><th className="py-2 pr-3">净PnL $</th><th className="py-2 pr-3">毛利 $</th>
                    <th className="py-2 pr-3">成本 $</th><th className="py-2 pr-3">Sharpe</th><th className="py-2 pr-3">maxDD</th>
                    <th className="py-2 pr-3">均|pos|</th><th className="py-2 pr-3">换手</th><th className="py-2">IC_skew</th>
                  </tr>
                </thead>
                <tbody className="font-mono tabular-nums">
                  {p.metrics.rows.map((r) => (
                    <tr key={r.year} className="border-b border-zinc-800/50">
                      <td className="py-1.5 pr-3 text-zinc-400">{r.year}</td>
                      <td className={`py-1.5 pr-3 ${r.pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>{fmtUSD(r.pnl, 1)}</td>
                      <td className="py-1.5 pr-3 text-zinc-400">{fmtUSD(r.gross, 1)}</td>
                      <td className="py-1.5 pr-3 text-amber-400/80">{fmtUSD(r.cost, 1)}</td>
                      <td className="py-1.5 pr-3 text-zinc-400">{r.sharpe.toFixed(2)}</td>
                      <td className="py-1.5 pr-3 text-rose-400/80">{fmtUSD(r.maxdd, 0)}</td>
                      <td className="py-1.5 pr-3 text-zinc-400">{r.avg_pos.toFixed(3)}</td>
                      <td className="py-1.5 pr-3 text-zinc-400">{r.turnover.toFixed(0)}</td>
                      <td className={`py-1.5 ${r.ic_skew >= 0 ? 'text-zinc-400' : 'text-rose-400/80'}`}>{r.ic_skew.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* 对照组: 学习真实性 + 基线 + 探针 */}
      <div className="grid gap-4 lg:grid-cols-2">
        <motion.div {...rise(1)} className="min-w-0">
          <Card className="h-full border-zinc-800 bg-zinc-900/60">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-sm"><Gauge className="h-4 w-4 text-emerald-400" />学习真实性对照 & 基线</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-xs">
              <div className="grid grid-cols-3 gap-2">
                {[
                  ['真学习 (seed42)', fmtUSD(o.pnl), `IC ${o.ic_skew.toFixed(3)}/${o.ic_var.toFixed(3)}`, o.pnl >= 0 ? 'pos' : 'neg'],
                  ['洗牌标签对照', fmtUSD(data.shuffle.pnl), `IC ${data.shuffle.ic_skew.toFixed(3)}/${data.shuffle.ic_var.toFixed(3)}`, 'default'],
                  ['动量基线 (无学习)', fmtUSD(data.momentum.total), `Sharpe ${data.momentum.sharpe.toFixed(2)}`, 'default'],
                ].map(([t, v, s, tone]) => (
                  <div key={t as string} className="rounded-md border border-zinc-800 bg-zinc-950/60 p-2.5">
                    <div className="text-[10px] text-zinc-500">{t}</div>
                    <div className={`mt-1 font-mono text-sm font-semibold tabular-nums ${tone === 'pos' ? 'text-emerald-400' : tone === 'neg' ? 'text-rose-400' : 'text-zinc-300'}`}>{v}</div>
                    <div className="mt-0.5 font-mono text-[10px] text-zinc-500">{s}</div>
                  </div>
                ))}
              </div>
              <div className="rounded-md border border-zinc-800 bg-zinc-950/60 p-2.5">
                <div className="mb-1.5 text-[10px] uppercase tracking-wider text-zinc-500">随机种子稳健性</div>
                <div className="grid grid-cols-3 gap-2 font-mono text-[11px] tabular-nums">
                  {data.seeds.map((s) => (
                    <div key={s.seed} className="rounded bg-zinc-900/70 p-2">
                      <div className="text-zinc-500">seed {s.seed}</div>
                      <div className={s.pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}>{fmtUSD(s.pnl, 1)}</div>
                      <div className="text-[10px] text-zinc-500">IC {s.ic_skew.toFixed(3)}/{s.ic_var.toFixed(3)}</div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="rounded-md border border-zinc-800 bg-zinc-950/60 p-2.5">
                <div className="mb-1.5 text-[10px] uppercase tracking-wider text-zinc-500">λ 遗忘因子扫描（协方差缠绕探针）</div>
                <table className="w-full font-mono text-[11px] tabular-nums">
                  <tbody>
                    {data.lam_sweep.map((l) => (
                      <tr key={l.lam} className="border-b border-zinc-800/40 last:border-0">
                        <td className="py-1 pr-2 text-zinc-400">λ={l.lam}</td>
                        <td className={`py-1 pr-2 ${l.pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>{fmtUSD(l.pnl, 1)}</td>
                        <td className="py-1 text-[10px] text-zinc-600">{l.note}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="rounded-md border border-zinc-800 bg-zinc-950/60 p-2.5">
                <div className="mb-1.5 text-[10px] uppercase tracking-wider text-zinc-500">储层规模扫描</div>
                <table className="w-full font-mono text-[11px] tabular-nums">
                  <tbody>
                    {data.nres_sweep.map((n) => (
                      <tr key={n.n_res} className="border-b border-zinc-800/40 last:border-0">
                        <td className="py-1 pr-2 text-zinc-400">n_res={n.n_res}</td>
                        <td className={`py-1 pr-2 ${n.pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>{fmtUSD(n.pnl, 1)}</td>
                        <td className="py-1 pr-2 text-zinc-500">{n.ms_per_bar.toFixed(2)} ms/bar</td>
                        <td className="py-1 text-zinc-500">{n.engine_mb.toFixed(1)} MB</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div {...rise(2)} className="min-w-0">
          <Card className="h-full border-zinc-800 bg-zinc-900/60">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-sm"><GitBranch className="h-4 w-4 text-sky-400" />预测视界敏感性探针</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-xs">
              <div className="overflow-x-auto">
                <table className="w-full min-w-[420px] text-xs">
                  <thead>
                    <tr className="border-b border-zinc-800 text-left text-[10px] uppercase tracking-wider text-zinc-500">
                      <th className="py-2 pr-3">变体</th><th className="py-2 pr-3">净PnL</th><th className="py-2 pr-3">IC 方向/能量</th>
                      <th className="py-2 pr-3">成本</th><th className="py-2">换手</th>
                    </tr>
                  </thead>
                  <tbody className="font-mono tabular-nums">
                    {[{ name: `h${data.meta.horizon} 基准 (1h)`, horizon: data.meta.horizon, note: '', pnl: o.pnl, ic_skew: o.ic_skew, ic_var: o.ic_var, sharpe: o.sharpe, cost: o.cost, turnover: o.turnover },
                      ...data.probes].map((r) => (
                      <tr key={r.name} className="border-b border-zinc-800/50">
                        <td className="py-1.5 pr-3 text-zinc-300">{r.name}</td>
                        <td className={`py-1.5 pr-3 ${r.pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>{fmtUSD(r.pnl, 1)}</td>
                        <td className="py-1.5 pr-3 text-zinc-400">{r.ic_skew.toFixed(3)} / {r.ic_var.toFixed(3)}</td>
                        <td className="py-1.5 pr-3 text-amber-400/80">{fmtUSD(r.cost, 0)}</td>
                        <td className="py-1.5 text-zinc-400">{r.turnover.toFixed(0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-[11px] leading-relaxed text-zinc-500">
                视界探针回答"方向动量在哪个尺度上可学"：1 小时偏斜 ≈ 不可预测，更长视界（4h/8h/24h）检验储层记忆
                是否能捕捉持续数日的趋势结构。低换手变体检验成本通道的敏感性。
              </p>
              <div className="rounded-md border border-sky-900/50 bg-sky-950/20 p-2.5">
                <div className="text-[11px] font-semibold text-sky-300">下一步（阶段二路线）</div>
                <ul className="mt-1.5 space-y-1">
                  {data.verdict.next.map((n) => (
                    <li key={n} className="flex gap-1.5 text-[11px] leading-relaxed text-zinc-400">
                      <span className="mt-0.5 text-sky-500/70">▸</span><span className="min-w-0">{n}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* 部署路线 */}
      <motion.div {...rise(1)} className="min-w-0">
        <Card className="border-zinc-800 bg-zinc-900/60">
          <CardHeader className="pb-2"><CardTitle className="text-sm">纯 CPU 部署路线（引擎实测达标项）</CardTitle></CardHeader>
          <CardContent>
            <div className="grid gap-3 md:grid-cols-3">
              {[
                ['阶段 1 · Python 仿真（本页）', `单核 ${(tm?.mean ?? 0).toFixed(2)} ms/bar · p99 ${(tm?.p99 ?? 0).toFixed(2)} ms`, 'NumPy 流式单次通过 2022–2026，W_out 终身进化无重训'],
                ['阶段 2 · MT5 ↔ Python 桥', 'ZeroMQ / Named Pipe', 'MT5 50 行只管行情抓取与订单发送，引擎常驻 <1% CPU'],
                ['阶段 3 · C++ Eigen DLL', '~200 KB 单文件', '#import 直嵌 EA，断网本地终身运转'],
              ].map(([t, s, d]) => (
                <div key={t} className="rounded-md border border-zinc-800 bg-zinc-950/60 p-3">
                  <div className="text-xs font-semibold text-zinc-200">{t}</div>
                  <div className="mt-1 font-mono text-[10px] text-emerald-400/90">{s}</div>
                  <p className="mt-1 text-[11px] leading-relaxed text-zinc-500">{d}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  )
}
