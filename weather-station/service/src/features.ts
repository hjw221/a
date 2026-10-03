/**
 * features.ts — M15 bar → 8 维因果特征流 (build_feats n_feat=8 的流式移植)
 * f = [atr_ratio, ret_norm, hl_ratio, vol_z, atr_ratio_slow, ret1h_norm, sin_h, cos_h]
 * 同时产出 v4p(rolling4 ret²) / r4(rolling4 ret) / base(EMA96) 供引擎标签闭环。
 * head=400: 前 400 根只养滚动量不喂引擎 (与 build_feats 的 head 切片等价)。
 */

export interface FeatureOut {
  u: Float64Array   // 8 维 (clip ±4, NaN→0)
  v4p: number       // rolling4( ret² ) — h4 引擎标签 vf
  r4: number        // rolling4( ret )  — h4 引擎标签 rf
  base: number      // EMA96( nan_to_num(v4p) ) — h4 引擎标签基线
  v16: number       // rolling16( ret² ) — h16 引擎标签 vf
  r16: number       // rolling16( ret )  — h16 引擎标签 rf
  base16: number    // EMA96( nan_to_num(v16) ) — h16 引擎标签基线
}

const HEAD = 400

class RollingSum {
  private buf: Float64Array
  private n = 0
  private idx = 0
  private sum = 0
  constructor(private w: number) { this.buf = new Float64Array(w) }
  push(v: number): { sum: number; n: number } {
    if (this.n < this.w) this.n++
    else this.sum -= this.buf[this.idx]
    this.buf[this.idx] = v
    this.sum += v
    this.idx = (this.idx + 1) % this.w
    return { sum: this.sum, n: this.n }
  }
}

class RollingMeanStd {
  private buf: Float64Array
  private n = 0
  private idx = 0
  private sum = 0
  private sum2 = 0
  constructor(private w: number, private minP: number) { this.buf = new Float64Array(w) }
  push(v: number): { mean: number; sd: number } | null {
    if (this.n < this.w) this.n++
    else {
      const old = this.buf[this.idx]
      this.sum -= old; this.sum2 -= old * old
    }
    this.buf[this.idx] = v
    this.sum += v; this.sum2 += v * v
    this.idx = (this.idx + 1) % this.w
    if (this.n < this.minP) return null
    const mean = this.sum / this.n
    const varr = (this.sum2 - this.n * mean * mean) / (this.n - 1)
    return { mean, sd: varr > 0 ? Math.sqrt(varr) : 0 }
  }
}

/** 流式中位数: 有序窗 + 二分插入/删除 (等价 pandas rolling median, 偶数取均值) */
class SortedWindow {
  private a: number[] = []
  private q: number[] = []
  constructor(private w: number, private minP: number) {}
  push(v: number): number {
    if (Number.isNaN(v)) return NaN
    if (this.q.length === this.w) {
      const old = this.q.shift()!
      let lo = 0, hi = this.a.length - 1
      while (lo < hi) { const mid = (lo + hi) >> 1; if (this.a[mid] < old) lo = mid + 1; else hi = mid }
      if (this.a[lo] === old) this.a.splice(lo, 1)
    }
    let lo = 0, hi = this.a.length
    while (lo < hi) { const mid = (lo + hi) >> 1; if (this.a[mid] < v) lo = mid + 1; else hi = mid }
    this.a.splice(lo, 0, v)
    this.q.push(v)
    if (this.q.length < this.minP) return NaN
    const n = this.a.length
    return n % 2 === 1 ? this.a[(n - 1) >> 1] : (this.a[n / 2 - 1] + this.a[n / 2]) / 2
  }
}

export class FeatureStream {
  private trSum = new RollingSum(14)
  private medFast = new SortedWindow(288, 96)
  private medSlow = new SortedWindow(1440, 480)
  private volStat = new RollingMeanStd(96, 32)
  private retSum = new RollingSum(4)
  private ret2Sum = new RollingSum(4)
  private retSum16 = new RollingSum(16)
  private ret2Sum16 = new RollingSum(16)
  private baseEma = 0
  private baseEma16 = 0
  private alpha = 2.0 / 97.0
  private prevClose = NaN
  barIdx = 0

  feed(bar: { tMin: number; open: number; high: number; low: number; close: number; tickvol: number }): FeatureOut | null {
    const { tMin, high, low, close, tickvol } = bar
    const pc = this.prevClose
    this.prevClose = close
    // TR / ATR14 (SMA, min=14 → 前13根 NaN)
    const tr = Number.isFinite(pc)
      ? Math.max(high - low, Math.abs(high - pc), Math.abs(low - pc))
      : high - low
    const tR = this.trSum.push(tr)
    const atr = tR.n >= 14 ? tR.sum / 14 : NaN
    // ATR 慢/快中位 (NaN 的 atr 不进窗 — pandas 里 NaN 行 rolling median 输出 NaN 且窗口含 NaN?
    // pandas rolling 在窗口含 NaN 时 median=NaN; 但 ATR 的 NaN 只出现在最前 13 根 (min_periods=96/480
    // 覆盖该段), 窗口一旦满 96/480 就全是有限值 — 与本实现等价)
    const medFast = this.medFast.push(atr)
    const medSlow = this.medSlow.push(atr)
    // ret
    const ret = Number.isFinite(pc) && pc > 0 && close > 0 ? Math.log(close / pc) : NaN
    // tickvol z
    const vs = this.volStat.push(tickvol)
    // r4 / v4p (min=4 → 前3根 NaN → 0)
    const retF = Number.isFinite(ret) ? ret : 0
    const r4s = this.retSum.push(retF)
    const v4s = this.ret2Sum.push(retF * retF)
    const r4 = r4s.n >= 4 ? r4s.sum : NaN
    const v4p = v4s.n >= 4 ? v4s.sum : NaN
    // EMA96 base (nan_to_num: 前3根 v4p=NaN→0)
    const v4p0 = Number.isFinite(v4p) ? v4p : 0
    this.baseEma += this.alpha * (v4p0 - this.baseEma)
    // h16 引擎: rolling16 求和 + 自身 EMA96 基线
    const r16s = this.retSum16.push(retF)
    const v16s = this.ret2Sum16.push(retF * retF)
    const r16 = r16s.n >= 16 ? r16s.sum : NaN
    const v16 = v16s.n >= 16 ? v16s.sum : NaN
    const v16_0 = Number.isFinite(v16) ? v16 : 0
    this.baseEma16 += this.alpha * (v16_0 - this.baseEma16)
    // ---- 8 维特征 ----
    const c = close
    const hrs = (tMin % 1440) / 60
    const u = new Float64Array(8)
    u[0] = div(atr, posMax(medFast, 1e-9))
    u[1] = div(ret, Number.isFinite(atr) ? Math.max(atr / c, 1e-12) : NaN)
    u[2] = div(high - low, Number.isFinite(atr) ? Math.max(atr, 1e-9) : NaN)
    u[3] = vs ? div(tickvol - vs.mean, Math.max(vs.sd, 1e-9)) : NaN
    u[4] = div(atr, posMax(medSlow, 1e-9))
    u[5] = div(r4, Number.isFinite(atr) ? Math.max((atr / c) * 2.0, 1e-12) : NaN)
    u[6] = Math.sin(2 * Math.PI * hrs / 24)
    u[7] = Math.cos(2 * Math.PI * hrs / 24)
    for (let i = 0; i < 8; i++) u[i] = Number.isFinite(u[i]) ? clampv(u[i], -4, 4) : 0
    this.barIdx++
    if (this.barIdx <= HEAD) return null        // head=400
    return {
      u,
      v4p: Number.isFinite(v4p) ? v4p : 0,
      r4: Number.isFinite(r4) ? r4 : 0,
      base: this.baseEma,
      v16: Number.isFinite(v16) ? v16 : 0,
      r16: Number.isFinite(r16) ? r16 : 0,
      base16: this.baseEma16,
    }
  }
}

function div(a: number, b: number): number { return a / b }
/** np.maximum(med, eps): med NaN → NaN (后续 nan_to_num→0) */
function posMax(med: number, eps: number): number {
  return Number.isFinite(med) ? Math.max(med, eps) : NaN
}
function clampv(v: number, lo: number, hi: number): number { return Math.max(lo, Math.min(hi, v)) }
