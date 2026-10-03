'use client'

import { motion } from 'framer-motion'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import {
  Activity, ArrowUpRight, Brain, Crosshair, Gauge, Layers, Link2,
  Radar, ShieldAlert, Sparkles, TrendingUp, Waves, Zap,
} from 'lucide-react'

/* ================= types ================= */
interface ChainStep { step: string; ic: number; h8: number; ms: number; note: string }
interface KeyArm {
  arm: string; ic_var: number; ic_skew: number; ic_h4: number; ic_h8: number
  ic_h16: number; ic_h32: number; pnl: number; gross: number; cost: number
  turn: number; ms: number; mb: number; dec_spread: number | null; seeds: number
}
interface LinkArm { k: string; n: number; pnl: number; per: number; note: string }
interface UseCase {
  id: string; name: string; verdict: string; strength: number; detail: string
}
export interface R2Data {
  meta: {
    generated: string; title: string; headline: string; root_cause: string
    champion: {
      ic: number; ic_h8: number; ic_h16: number; ic_h32: number; ic_skew: number
      ms: number; mb: number | null; pnl_cont: number; pnl_schmitt: number
      turn: number; dec_spread: number; cfg: string
    }
  }
  evolution: ChainStep[]
  key_arms: KeyArm[]
  linkage: {
    note: string
    baseline: { n: number; pnl: number; plr: number; sharpe: number; per_trade: number }
    arms: LinkArm[]
    verdict: string
  }
  usecases: UseCase[]
  crossasset: {
    self_ic: Record<string, number>
    transmission: Record<string, number>
    augmented: Record<string, number>
    note: string
  }
  v3bal_fix: {
    bug: string; layers: string[]
    fixed: { trades: number; pnl: number; plr: number; sharpe: number }
    hist: { trades: number; pnl: number; plr: number; sharpe: number }
    buggy: { trades: number; pnl: number; plr: number }
  }
}

const strengthMeta: Record<number, { label: string; cls: string }> = {
  4: { label: '强证据', cls: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300' },
  3: { label: '中强证据', cls: 'border-teal-500/40 bg-teal-500/10 text-teal-300' },
  2: { label: '尾部证据', cls: 'border-amber-500/40 bg-amber-500/10 text-amber-300' },
}

const fade = (i: number) => ({
  initial: { opacity: 0, y: 10 },
  animate: { opacity: 1, y: 0 },
  transition: { delay: i * 0.05, duration: 0.3 },
})

/* ================= component ================= */
export function R2Tab({ data }: { data: R2Data }) {
  const { meta, evolution, key_arms, linkage, usecases, crossasset, v3bal_fix } = data
  const champ = meta.champion
  const kpis = [
    { icon: Brain, label: 'IC_var @1h', v: champ.ic.toFixed(3), sub: 'v1 0.343 · +43%', good: true },
    { icon: TrendingUp, label: 'IC_var @2h', v: champ.ic_h8.toFixed(3), sub: 'v1 0.351 · +50%', good: true },
    { icon: Zap, label: '单bar耗时', v: `${champ.ms.toFixed(2)}ms`, sub: 'v1 0.478 · 5×提速', good: true },
    { icon: Activity, label: '换手 (施密特)', v: champ.turn.toLocaleString(), sub: 'v1 11,092 · −92%', good: true },
  ]

  return (
    <div className="space-y-6">
      {/* 结论横幅 */}
      <motion.section {...fade(0)} aria-label="R2 结论">
        <Card className="border-emerald-500/30 bg-gradient-to-br from-emerald-500/10 via-zinc-900 to-zinc-900">
          <CardHeader className="pb-2">
            <div className="flex flex-wrap items-center gap-2">
              <Sparkles className="h-4 w-4 text-emerald-400" aria-hidden />
              <CardTitle className="text-base">R1 自我进化成功 — {meta.headline}</CardTitle>
            </div>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-zinc-300">
            <p>
              <Badge variant="outline" className="mr-2 border-emerald-500/40 text-emerald-300">根因</Badge>
              {meta.root_cause}
            </p>
            <p className="text-xs text-zinc-500">
              冠军配置 <span className="font-mono text-zinc-400">{champ.cfg}</span> ·
              35 臂网格选出 · 3 种子稳健 (0.473/0.482/0.491) · 洗牌对照 IC→−0.019 · v1 bitwise 回归 ✓
            </p>
          </CardContent>
        </Card>
      </motion.section>

      {/* KPI */}
      <motion.section {...fade(1)} aria-label="冠军 KPI">
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          {kpis.map((k) => (
            <Card key={k.label} className="border-zinc-800 bg-zinc-900/60">
              <CardContent className="p-4">
                <div className="flex items-center gap-2 text-zinc-500">
                  <k.icon className="h-3.5 w-3.5" aria-hidden />
                  <span className="text-xs font-medium uppercase tracking-wider">{k.label}</span>
                </div>
                <div className="mt-1.5 font-mono text-2xl font-bold text-emerald-300">{k.v}</div>
                <div className="mt-0.5 text-xs text-zinc-500">{k.sub}</div>
              </CardContent>
            </Card>
          ))}
        </div>
      </motion.section>

      {/* 进化链 */}
      <motion.section {...fade(2)} aria-label="进化链">
        <Card className="border-zinc-800 bg-zinc-900/60">
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2">
              <ArrowUpRight className="h-4 w-4 text-emerald-400" aria-hidden />
              <CardTitle className="text-sm">IC 进化链 — 0.343 → 0.491</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[560px] text-left text-xs">
                <thead className="text-zinc-500">
                  <tr className="border-b border-zinc-800">
                    <th className="py-2 pr-3 font-medium">改进步</th>
                    <th className="py-2 pr-3 text-right font-medium">IC @1h</th>
                    <th className="py-2 pr-3 text-right font-medium">IC @2h</th>
                    <th className="py-2 pr-3 text-right font-medium">ms/bar</th>
                    <th className="py-2 font-medium">备注</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {evolution.map((e, i) => (
                    <tr key={e.step} className={`border-b border-zinc-800/50 ${i === evolution.length - 1 ? 'bg-emerald-500/5' : ''}`}>
                      <td className="py-1.5 pr-3 text-zinc-300">{e.step}</td>
                      <td className="py-1.5 pr-3 text-right text-emerald-300">{e.ic.toFixed(4)}</td>
                      <td className="py-1.5 pr-3 text-right text-zinc-400">{e.h8.toFixed(4)}</td>
                      <td className="py-1.5 pr-3 text-right text-zinc-400">{e.ms.toFixed(2)}</td>
                      <td className="py-1.5 font-sans text-zinc-500">{e.note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </motion.section>

      {/* 四用途审判 */}
      <motion.section {...fade(3)} aria-label="四用途重新评估">
        <div className="mb-2 flex items-center gap-2">
          <Radar className="h-4 w-4 text-teal-400" aria-hidden />
          <h3 className="text-sm font-semibold uppercase tracking-wider text-zinc-400">四个部署用途 · 重新评估 (R1 进步后)</h3>
        </div>
        <div className="grid gap-3 md:grid-cols-2">
          {usecases.map((u) => {
            const st = strengthMeta[u.strength] ?? { label: '弱', cls: 'border-zinc-600 text-zinc-400' }
            const icons: Record<string, typeof Waves> = {
              UC1: Layers, UC2: Waves, UC3: ShieldAlert, UC4: Crosshair,
            }
            const Icon = icons[u.id] ?? Gauge
            return (
              <Card key={u.id} className={`border ${u.strength >= 4 ? 'border-emerald-500/30' : 'border-zinc-800'} bg-zinc-900/60`}>
                <CardHeader className="pb-1">
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <Icon className="h-4 w-4 text-teal-300" aria-hidden />
                      <CardTitle className="text-sm">{u.id} · {u.name}</CardTitle>
                    </div>
                    <Badge variant="outline" className={st.cls}>{st.label}</Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-xs leading-relaxed text-zinc-400">{u.detail}</p>
                  <p className="mt-2 text-xs font-medium text-zinc-300">{u.verdict}</p>
                </CardContent>
              </Card>
            )
          })}
        </div>
        <p className="mt-2 text-xs text-zinc-500">
          优先级排序: UC4 跨资产嵌合 (唯一 IC 提升通道) → UC2 呼吸阀 (零成本) → UC3 断路器 (冲击后冻结) → UC1 网格装甲 (尾部规则) → 联动 v3bal_ens (真但小)
        </p>
      </motion.section>

      {/* 跨资产 */}
      <motion.section {...fade(4)} aria-label="跨资产编码器">
        <Card className="border-teal-500/30 bg-zinc-900/60">
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2">
              <Crosshair className="h-4 w-4 text-teal-400" aria-hidden />
              <CardTitle className="text-sm">UC4 · 跨资产编码器 — DXY / XAG → XAUUSD (用户新数据)</CardTitle>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-3 gap-3">
              <div className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-3">
                <div className="text-xs text-zinc-500">引擎跨资产自 IC</div>
                <div className="mt-1 font-mono text-sm">
                  <div className="text-zinc-300">XAU {crossasset.self_ic.xau?.toFixed(3)}</div>
                  <div className="text-zinc-300">DXY {crossasset.self_ic.dxy?.toFixed(3)}</div>
                  <div className="text-teal-300">XAG {crossasset.self_ic.xag?.toFixed(3)}</div>
                </div>
              </div>
              <div className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-3">
                <div className="text-xs text-zinc-500">传输 (滞后120min)</div>
                <div className="mt-1 font-mono text-sm">
                  <div className="text-teal-300">XAG→XAU h16 {crossasset.transmission.xag_lag120_h16?.toFixed(4)}</div>
                  <div className="text-zinc-300">DXY→XAU h16 {crossasset.transmission.dxy_lag120_h16?.toFixed(4)}</div>
                  <div className="text-zinc-500">XAU 自身 h16 {crossasset.transmission.xau_self_h16?.toFixed(4)}</div>
                </div>
              </div>
              <div className="rounded-lg border border-teal-500/30 bg-zinc-950/60 p-3">
                <div className="text-xs text-zinc-500">嵌合引擎 IC_var</div>
                <div className="mt-1 font-mono text-sm">
                  <div className="text-zinc-300">单资产 {crossasset.augmented.single?.toFixed(4)}</div>
                  <div className="text-emerald-300">+R_dxy/R_xag {crossasset.augmented.aug_R_only?.toFixed(4)}</div>
                  <div className="text-emerald-300">+d2 特征 {crossasset.augmented.aug_R_d2?.toFixed(4)}</div>
                  <div className="text-rose-400">shuffle {crossasset.augmented.shuffle?.toFixed(4)}</div>
                </div>
              </div>
            </div>
            <p className="text-xs text-zinc-500">{crossasset.note} · {data.crossasset.augmented.aug_R_d2 && data.crossasset.augmented.single ? `增益 +${((data.crossasset.augmented.aug_R_d2 / data.crossasset.augmented.single - 1) * 100).toFixed(1)}%` : ''} (shuffle 验证通过)</p>
          </CardContent>
        </Card>
      </motion.section>

      {/* v3bal_ens 修复 + 联动 */}
      <motion.section {...fade(5)} aria-label="v3bal_ens 联动">
        <Card className="border-zinc-800 bg-zinc-900/60">
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2">
              <Link2 className="h-4 w-4 text-amber-400" aria-hidden />
              <CardTitle className="text-sm">v3bal_ens 时域修复 × R1 联动</CardTitle>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-3 gap-3 text-center">
              <div className="rounded-lg border border-rose-500/30 bg-zinc-950/60 p-3">
                <div className="text-xs text-zinc-500">bug 版 (9/30)</div>
                <div className="mt-1 font-mono text-lg text-rose-400">{v3bal_fix.buggy.trades}笔</div>
                <div className="font-mono text-xs text-rose-400">${v3bal_fix.buggy.pnl}</div>
              </div>
              <div className="rounded-lg border border-emerald-500/30 bg-zinc-950/60 p-3">
                <div className="text-xs text-zinc-500">修复版 (本次)</div>
                <div className="mt-1 font-mono text-lg text-emerald-300">{v3bal_fix.fixed.trades}笔</div>
                <div className="font-mono text-xs text-emerald-300">+${v3bal_fix.fixed.pnl} · PLR {v3bal_fix.fixed.plr}</div>
              </div>
              <div className="rounded-lg border border-zinc-700 bg-zinc-950/60 p-3">
                <div className="text-xs text-zinc-500">历史基准</div>
                <div className="mt-1 font-mono text-lg text-zinc-300">{v3bal_fix.hist.trades}笔</div>
                <div className="font-mono text-xs text-zinc-300">+${v3bal_fix.hist.pnl} · PLR {v3bal_fix.hist.plr}</div>
              </div>
            </div>
            <p className="text-xs text-zinc-500">{v3bal_fix.bug} · 挖穿两层坑: {v3bal_fix.layers.join(' / ')}</p>
            <Separator />
            <div className="overflow-x-auto">
              <table className="w-full min-w-[520px] text-left text-xs">
                <thead className="text-zinc-500">
                  <tr className="border-b border-zinc-800">
                    <th className="py-2 pr-3 font-medium">联动机制</th>
                    <th className="py-2 pr-3 text-right font-medium">笔数</th>
                    <th className="py-2 pr-3 text-right font-medium">PnL</th>
                    <th className="py-2 pr-3 text-right font-medium">$/笔</th>
                    <th className="py-2 font-medium">判读</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-zinc-800/50 bg-zinc-800/20">
                    <td className="py-1.5 pr-3 text-zinc-200">基线 (flat 1oz)</td>
                    <td className="py-1.5 pr-3 text-right font-mono">{linkage.baseline.n}</td>
                    <td className="py-1.5 pr-3 text-right font-mono">${linkage.baseline.pnl}</td>
                    <td className="py-1.5 pr-3 text-right font-mono">${linkage.baseline.per_trade.toFixed(2)}</td>
                    <td className="py-1.5 text-zinc-500">—</td>
                  </tr>
                  {linkage.arms.map((a) => (
                    <tr key={a.k} className="border-b border-zinc-800/50">
                      <td className="py-1.5 pr-3 text-zinc-300">{a.k}</td>
                      <td className="py-1.5 pr-3 text-right font-mono text-zinc-400">{a.n}</td>
                      <td className="py-1.5 pr-3 text-right font-mono text-zinc-400">${a.pnl}</td>
                      <td className="py-1.5 pr-3 text-right font-mono text-zinc-300">${a.per.toFixed(2)}</td>
                      <td className="py-1.5 text-zinc-500">{a.note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-xs text-amber-300/90">联动结论: {linkage.verdict}</p>
          </CardContent>
        </Card>
      </motion.section>

      {/* 关键对照臂 */}
      <motion.section {...fade(6)} aria-label="关键对照臂">
        <Card className="border-zinc-800 bg-zinc-900/60">
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2">
              <Gauge className="h-4 w-4 text-zinc-400" aria-hidden />
              <CardTitle className="text-sm">关键对照臂 (35 臂全表见归档)</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[640px] text-left text-xs">
                <thead className="text-zinc-500">
                  <tr className="border-b border-zinc-800">
                    <th className="py-2 pr-3 font-medium">臂</th>
                    <th className="py-2 pr-3 text-right font-medium">IC@1h</th>
                    <th className="py-2 pr-3 text-right font-medium">IC@2h</th>
                    <th className="py-2 pr-3 text-right font-medium">IC@4h</th>
                    <th className="py-2 pr-3 text-right font-medium">毛利$</th>
                    <th className="py-2 pr-3 text-right font-medium">换手</th>
                    <th className="py-2 pr-3 text-right font-medium">ms</th>
                    <th className="py-2 text-right font-medium">dec价差</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {key_arms.map((a) => (
                    <tr key={a.arm} className={`border-b border-zinc-800/50 ${a.arm.includes('shuffle') ? 'bg-rose-500/5' : a.arm.includes('champ_s') ? 'bg-emerald-500/5' : ''}`}>
                      <td className="py-1.5 pr-3 text-zinc-300">{a.arm}</td>
                      <td className="py-1.5 pr-3 text-right text-zinc-200">{a.ic_h4?.toFixed(4)}</td>
                      <td className="py-1.5 pr-3 text-right text-zinc-400">{a.ic_h8?.toFixed(4)}</td>
                      <td className="py-1.5 pr-3 text-right text-zinc-400">{a.ic_h16?.toFixed(4)}</td>
                      <td className="py-1.5 pr-3 text-right text-zinc-400">{a.gross.toFixed(0)}</td>
                      <td className="py-1.5 pr-3 text-right text-zinc-400">{Math.round(a.turn)}</td>
                      <td className="py-1.5 pr-3 text-right text-zinc-400">{a.ms.toFixed(2)}</td>
                      <td className="py-1.5 text-right text-zinc-400">{a.dec_spread ? `${a.dec_spread}×` : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-2 text-xs text-zinc-500">
              诚实条款: 冠军参数按 IC 从 35 臂选出 (选择偏差存在), 但 3 种子稳健 + nres-75..300 单调平台 + shuffle 归零;
              独立净亏 −$1,287 (施密特) — 方向头 IC 0.031 仍不足以独立变现, R1 价值在能量/惊异度/跨资产 (三年证据链第四次确认)
            </p>
          </CardContent>
        </Card>
      </motion.section>
    </div>
  )
}
