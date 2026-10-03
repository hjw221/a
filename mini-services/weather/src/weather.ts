/**
 * weather.ts — UC4 跨资产天气预警站 (严格因果版)
 * ==============================================
 * 四台 R2 冠军级储层引擎常驻流式计算:
 *   · XAU  h4  引擎 — 1h 能量预报 (IC 0.4917, TS vs Python 0.4907 ✓)
 *   · XAU  h16 引擎 — 4h 能量预报 (IC 0.5188, Python h16 训练实锤 — 天气头条)
 *   · DXY  h4  引擎 — 美元指数能量 (IC 0.4931)
 *   · XAG  h4  引擎 — 白银能量 (IC 0.5192) → 传输预警源
 *
 * 诚实修正 (2026-10-03 lag 消融): R2 档案的"嵌合引擎 IC 0.5363"经严格因果检验
 * (asof 只用决策时刻已收盘的 bar) 增益归零 (0.4915 ≈ 单资产) — 该数字来自 asof
 * 允许 DXY/XAG bar 进入标签窗口首 15 分钟的未来信息伪影; lag 越大 IC 越高
 * (lag60: 0.587) 直至越出标签窗 (lag120: 0.539) — 泄漏曲线实锤。
 * 跨资产的真实价值 = 白银→黄金传输预警 (XAG 能量领先 XAU 4h 方差, IC 0.55,
 * 该传输分析无窗口重叠, 因果干净) + 三资产各自能量/惊异度监控。
 *
 * 预警体系:
 *   · 1h 天气等级 (R_xau): ☀️<0.7 / ⛅<1.2 / 🌬️<2.0 / ⛈️≥2.0
 *   · 4h 展望等级 (R_xau4h): 同阈值, 更长视界
 *   · UC3 断路器: d2>25 → 未来1h方差≈5.9×中位 (R2 档案 n=114)
 *   · 白银传输预警: R_xag≥2.0 → XAU 未来2-4h方差上行风险
 */
import { ReservoirEngine, CHAMP } from './engine'
import { FeatureStream } from './features'
import type { M15Bar } from './pipeline'

export type Level = 0 | 1 | 2 | 3
export const LEVEL_NAMES = ['calm', 'normal', 'windy', 'storm'] as const
export const LEVEL_EMOJI = ['☀️', '⛅', '🌬️', '⛈️'] as const

export function levelFromR(R: number): Level {
  if (R < 0.7) return 0
  if (R < 1.2) return 1
  if (R < 2.0) return 2
  return 3
}

export interface AssetState {
  R: number
  skew: number
  d2: number
  level: Level
  tag: number
  pct: number
  K: number
}

export interface WeatherState {
  t: string           // ISO (bar open, UTC 帧 = 数据自身时钟)
  slot: number
  xau: AssetState     // 1h 预报
  xau4h: AssetState   // 4h 预报 (h16 训练引擎)
  dxy: AssetState
  xag: AssetState
}

export interface Alert {
  t: string
  kind: 'weather' | 'outlook' | 'breaker' | 'silver' | 'system'
  asset: string
  msg: string
  R?: number
  d2?: number
}

interface DelayedPair { R: number; base: number }

export interface FeedResult {
  stepped: boolean
  u: Float64Array | null
  v4p: number
  r4: number
  base: number
  v16: number
  r16: number
  base16: number
}

class AssetRunner {
  feats = new FeatureStream()
  engine: ReservoirEngine
  R = 0
  skew = 0
  d2 = 0
  tag = -1
  pct = 0
  K = 0
  engineBars = 0
  icPairs: [number, number][] = []
  private delayed: DelayedPair[] = []

  constructor(public name: string, public horizon: 4 | 16, Win: number[][]) {
    this.engine = new ReservoirEngine(CHAMP.Wr, Win, { horizon })
  }

  feedBar(bar: M15Bar): FeedResult {
    const f = this.feats.feed(bar)
    if (!f) return { stepped: false, u: null, v4p: 0, r4: 0, base: 0, v16: 0, r16: 0, base16: 0 }
    const out = this.horizon === 4
      ? this.engine.step(f.u, f.v4p, f.r4, f.base)
      : this.engine.step(f.u, f.v16, f.r16, f.base16)
    this.R = out.R; this.skew = out.skew; this.d2 = out.d2
    this.tag = out.tag; this.pct = out.pct; this.K = out.K
    this.engineBars++
    // IC 延迟配对: y[t] = clip(v_h[t+h]/base[t], 0, 6)
    const base = this.horizon === 4 ? f.base : f.base16
    const vfNow = this.horizon === 4 ? f.v4p : f.v16
    this.delayed.push({ R: out.R, base })
    if (this.delayed.length > this.horizon) {
      const p = this.delayed.shift()!
      const y = Math.max(0, Math.min(6, vfNow / Math.max(p.base, 1e-14)))
      if (this.engineBars > 2040) this.icPairs.push([p.R, y])
    }
    return { stepped: true, u: f.u, v4p: f.v4p, r4: f.r4, base: f.base, v16: f.v16, r16: f.r16, base16: f.base16 }
  }

  state(): AssetState {
    return { R: round3(this.R), skew: round3(this.skew), d2: round2(this.d2),
             level: levelFromR(this.R), tag: this.tag, pct: round3(this.pct), K: this.K }
  }
}

export class WeatherStation {
  xau = new AssetRunner('xau', 4, CHAMP.Win8)
  xau4h = new AssetRunner('xau4h', 16, CHAMP.Win8)
  dxy = new AssetRunner('dxy', 4, CHAMP.Win8)
  xag = new AssetRunner('xag', 4, CHAMP.Win8)
  history: WeatherState[] = []
  alerts: Alert[] = []
  private lastLevel: Level = -1
  private lastOutlook: Level = -1
  private lastXagHot = false
  private breakerOn = new Map<string, boolean>()
  readonly maxHistory = 2880   // ~30 天
  readonly maxAlerts = 240
  firstT: string | null = null
  /** 回放期累计预警计数 (跨整个 4.5 年历史, 不受滚动窗口截断) */
  alertCounts: Record<string, number> = { weather: 0, outlook: 0, breaker: 0, silver: 0 }
  lastSlot = -1

  /** 槽步进: DXY/XAG 先于 XAU (同槽上下文就绪; 全部只用已收盘 bar — 严格因果) */
  stepSlot(slot: number, bars: { xau?: M15Bar; dxy?: M15Bar; xag?: M15Bar }): WeatherState {
    if (bars.dxy) this.dxy.feedBar(bars.dxy)
    if (bars.xag) this.xag.feedBar(bars.xag)
    if (bars.xau) {
      this.xau.feedBar(bars.xau)
      this.xau4h.feedBar(bars.xau)
    }
    this.lastSlot = slot
    if (this.firstT === null) this.firstT = isoFromSlot(slot)
    const st: WeatherState = {
      t: isoFromSlot(slot), slot,
      xau: this.xau.state(), xau4h: this.xau4h.state(),
      dxy: this.dxy.state(), xag: this.xag.state(),
    }
    this.history.push(st)
    if (this.history.length > this.maxHistory) this.history.shift()
    this.evaluateAlerts(st)
    return st
  }

  private evaluateAlerts(st: WeatherState) {
    const t = st.t
    // 1) 1h 天气等级变化 (施密特滞回: 进入用标准阈, 退出放宽 0.15 — 消临界翻转刷屏)
    const lv = hystLevel(st.xau.R, this.lastLevel)
    if (lv !== this.lastLevel) {
      if (this.lastLevel >= 0) {
        this.pushAlert({
          t, kind: 'weather', asset: 'XAU',
          msg: `XAU 1h天气 ${LEVEL_NAMES[this.lastLevel]}${LEVEL_EMOJI[this.lastLevel]} → ${LEVEL_NAMES[lv]}${LEVEL_EMOJI[lv]} (R̂=${st.xau.R.toFixed(2)})`,
          R: st.xau.R,
        })
      }
      this.lastLevel = lv
    }
    // 2) 4h 展望等级变化 (同款滞回)
    const ov = hystLevel(st.xau4h.R, this.lastOutlook)
    if (ov !== this.lastOutlook) {
      if (this.lastOutlook >= 0) {
        this.pushAlert({
          t, kind: 'outlook', asset: 'XAU·4h',
          msg: `XAU 4h展望 ${LEVEL_NAMES[this.lastOutlook]}${LEVEL_EMOJI[this.lastOutlook]} → ${LEVEL_NAMES[ov]}${LEVEL_EMOJI[ov]} (R̂=${st.xau4h.R.toFixed(2)})`,
          R: st.xau4h.R,
        })
      }
      this.lastOutlook = ov
    }
    // 3) UC3 断路器: d2>25
    const probes: [string, { d2: number }][] = [
      ['XAU', st.xau], ['XAU·4h', st.xau4h], ['DXY', st.dxy], ['XAG', st.xag],
    ]
    for (const [name, a] of probes) {
      const prev = this.breakerOn.get(name) ?? false
      const on = prev ? a.d2 > 20 : a.d2 > 25
      if (on && !prev) {
        this.pushAlert({
          t, kind: 'breaker', asset: name,
          msg: `${name} 断路器触发: 惊异度 d2=${a.d2.toFixed(1)} (>25 → 未来1h方差≈5.9×中位, R2档案 n=114)`,
          d2: a.d2,
        })
      }
      this.breakerOn.set(name, on)
    }
    // 4) 白银传输预警: R_xag ≥ 2.0 进入 / < 1.7 退出 (滞回)
    const xagHot = this.lastXagHot ? st.xag.R >= 1.7 : st.xag.R >= 2.0
    if (xagHot && !this.lastXagHot) {
      this.pushAlert({
        t, kind: 'silver', asset: 'XAG→XAU',
        msg: `白银能量暴涨 R̂_xag=${st.xag.R.toFixed(2)} — 白银领先黄金 ~120min (传输IC 0.55, 因果干净), XAU 未来2-4h方差上行风险`,
        R: st.xag.R,
      })
    }
    this.lastXagHot = xagHot
  }

  pushAlert(a: Alert) {
    if (a.kind !== 'system') this.alertCounts[a.kind] = (this.alertCounts[a.kind] ?? 0) + 1
    this.alerts.push(a)
    if (this.alerts.length > this.maxAlerts) this.alerts.shift()
  }

  currentState(): WeatherState | null {
    return this.history.length ? this.history[this.history.length - 1] : null
  }
}

/** IC (Pearson) */
export function pearson(pairs: [number, number][]): number {
  const n = pairs.length
  if (n < 100) return 0
  let sx = 0, sy = 0
  for (const [x, y] of pairs) { sx += x; sy += y }
  const mx = sx / n, my = sy / n
  let sxy = 0, sxx = 0, syy = 0
  for (const [x, y] of pairs) {
    const dx = x - mx, dy = y - my
    sxy += dx * dy; sxx += dx * dx; syy += dy * dy
  }
  return sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : 0
}

/** 十分位校准 (UC2 证据: 预测分位 → 实现方差比; 冠军 7.32×) */
export function decileCalibration(pairs: [number, number][]): { d: string; mean: number; n: number }[] {
  const sorted = [...pairs].sort((a, b) => a[0] - b[0])
  const n = sorted.length
  const out: { d: string; mean: number; n: number }[] = []
  for (let i = 0; i < 10; i++) {
    const lo = Math.floor((i * n) / 10), hi = Math.floor(((i + 1) * n) / 10)
    const seg = sorted.slice(lo, hi)
    if (seg.length < 20) continue
    const mean = seg.reduce((a, p) => a + p[1], 0) / seg.length
    out.push({ d: `d${i + 1}`, mean: round3(mean), n: seg.length })
  }
  return out
}

function isoFromSlot(slot: number): string {
  return new Date(slot * 15 * 60000).toISOString().slice(0, 16).replace('T', ' ')
}

/** 施密特滞回等级: 进入 [0.7,1.2,2.0] / 退出放宽 0.15 — 消除临界翻转刷屏 */
function hystLevel(R: number, prev: Level): Level {
  if (prev < 0) return levelFromR(R)
  for (let l = 3; l > prev; l--) {
    if (R >= [0.7, 1.2, 2.0][l - 1]) return l as Level
  }
  if (prev > 0 && R < [0.55, 1.05, 1.85][prev - 1]) return (prev - 1) as Level
  return prev
}
function round3(x: number): number { return Math.round(x * 1000) / 1000 }
function round2(x: number): number { return Math.round(x * 100) / 100 }
