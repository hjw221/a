/**
 * engine.ts — R2 冠军储层引擎的忠实 TypeScript 移植 (2026-10-03)
 * ====================================================================
 * 来源: research/reservoir/reservoir_engine2.py (champion: nres50 · λ0.9999 ·
 *       log 目标 · 8 维输入 · seed42) — 三层结构逐行对应:
 *   L1 储层: x = tanh(Wr·x + Win·u)  (leak=1.0, 谱半径 0.95, W 永久锁死)
 *   L2 GMM : 8/10 维输入空间上的流式聚类 (d2 惊异度 = 断路器信号)
 *   L3 RLS : 指数遗忘 λ=0.9999, 延迟标签闭环 (φ_{t-4}, y_{t-4}), D=55
 *
 * 储层矩阵 Wr/Win8/Win10 由 Python numpy default_rng(42) 按 run_pass 的
 * 精确消耗顺序导出 (champion-params.json) — 与研究引擎同源同值。
 *
 * 预测: yh = W·φ → R = expm1(clip(yh0, 0, 3))  (能量比: 未来1h方差/基线)
 */
import params from './champion-params.json'

export interface GmmOut {
  g: Float64Array // [d2min/50, volPct, ent, K/16]
  tag: number     // 0低波平静 1高波方向 2高波混沌 3中波
  pct: number
  d2min: number
  K: number
}

/** L2: 流式 GMM — reservoir_engine2.StreamingGMM 逐行移植 */
export class StreamingGMM {
  dim: number
  kMax: number
  spawnD2: number
  spawnCool: number
  wSpawn: number
  lr: number
  wd: number
  sigma0: number
  mu: Float64Array[] = []
  sd: Float64Array[] = []
  w: number[] = []
  t = 0
  lastSpawn = -1e9

  constructor(dim: number, kMax = 16, spawnD2 = 36.0, spawnCool = 96,
              wSpawn = 0.10, lr = 0.02, wd = 0.01, sigma0 = 0.6) {
    this.dim = dim; this.kMax = kMax; this.spawnD2 = spawnD2
    this.spawnCool = spawnCool; this.wSpawn = wSpawn; this.lr = lr
    this.wd = wd; this.sigma0 = sigma0
  }

  private spawn(z: Float64Array, w0: number) {
    this.mu.push(Float64Array.from(z))
    this.sd.push(new Float64Array(this.dim).fill(this.sigma0))
    this.w.push(w0)
    const s = this.w.reduce((a, b) => a + b, 0)
    for (let i = 0; i < this.w.length; i++) this.w[i] /= s
  }

  step(zIn: Float64Array): GmmOut {
    this.t++
    const z = Float64Array.from(zIn)
    for (let i = 0; i < this.dim; i++) z[i] = Math.max(-3.5, Math.min(3.5, z[i]))
    if (this.w.length === 0) {
      this.spawn(z, 1.0)
      const g = new Float64Array([0.0, 0.5, 0.0, 1.0 / 16.0])
      return { g, tag: 3, pct: 0.0, d2min: 0.0, K: 1 }
    }
    const K = this.w.length
    const d2 = new Float64Array(K)
    const ll = new Float64Array(K)
    for (let k = 0; k < K; k++) {
      let s = 0.0, lsum = 0.0
      const muK = this.mu[k], sdK = this.sd[k]
      for (let i = 0; i < this.dim; i++) {
        const dz = (z[i] - muK[i]) / sdK[i]
        s += dz * dz
        lsum += Math.log(sdK[i])
      }
      d2[k] = s
      ll[k] = Math.log(Math.max(this.w[k], 1e-12)) - 0.5 * s - lsum
    }
    // softmax responsibilities
    const r = new Float64Array(K)
    let m = -Infinity
    for (let k = 0; k < K; k++) m = Math.max(m, ll[k])
    let rsum = 0
    for (let k = 0; k < K; k++) { r[k] = Math.exp(ll[k] - m); rsum += r[k] }
    for (let k = 0; k < K; k++) r[k] /= rsum
    let kb = 0
    for (let k = 1; k < K; k++) if (d2[k] < d2[kb]) kb = k
    const d2minRaw = d2[kb]
    if (d2[kb] > this.spawnD2 && this.t - this.lastSpawn > this.spawnCool
        && this.w.length < this.kMax + 4) {
      this.spawn(z, this.wSpawn)
      this.lastSpawn = this.t
    } else {
      // EM 更新 (mu_old 用于方差更新 — 与 Python 逐字一致)
      const muOld = this.mu.map(a => Float64Array.from(a))
      for (let k = 0; k < K; k++) {
        const muK = this.mu[k], sdK = this.sd[k], mo = muOld[k]
        for (let i = 0; i < this.dim; i++) {
          muK[i] += this.lr * r[k] * (z[i] - muK[i])
          const varK = sdK[i] * sdK[i] + this.lr * r[k] * ((z[i] - mo[i]) ** 2 - sdK[i] * sdK[i])
          sdK[i] = Math.sqrt(Math.max(varK, 0.01))
        }
        this.w[k] = (1.0 - this.wd) * this.w[k] + this.wd * r[k]
      }
      let ws = 0
      for (let k = 0; k < K; k++) ws += this.w[k]
      for (let k = 0; k < K; k++) this.w[k] /= ws
      // 剪枝 w<=1e-3 (保留>=2个组件; kb 身份保持)
      const keep = this.w.map(x => x > 1e-3)
      const nKeep = keep.filter(Boolean).length
      if (nKeep >= 2 && nKeep < K) {
        const kept: number[] = []
        for (let k = 0; k < K; k++) if (keep[k]) kept.push(k)
        if (!keep[kb]) {
          kb = 0
          for (const k of kept) if (d2[k] < d2[kept[kb]]) kb = kept.indexOf(k) // kept 内最小 d2
          // 上面循环把 kb 设为 kept 内最小者的下标
        } else {
          kb = kept.indexOf(kb)
        }
        const wsum = kept.reduce((a, k) => a + this.w[k], 0)
        this.mu = kept.map(k => this.mu[k])
        this.sd = kept.map(k => this.sd[k])
        this.w = kept.map(k => this.w[k] / wsum)
      }
    }
    const K2 = this.w.length
    const d2min = Math.min(d2minRaw, 100.0)
    // 波动分位: 组件 mu[0] 在全体组件中的排位 (Python searchsorted 'left' 语义)
    const volLev = this.mu.map(a => a[0]).sort((a, b) => a - b)
    const volKb = this.mu[kb][0]
    let rank = 0
    for (const v of volLev) if (v < volKb) rank++
    const pct = volLev.length > 1 ? rank / Math.max(K2 - 1, 1) : 0
    // 熵 (r 长度 = 更新前组件数; spawn 分支下 K2 可大于 r.length — Python 同款)
    let ent = 0
    for (let k = 0; k < r.length; k++) ent -= r[k] * Math.log(Math.max(r[k], 1e-12))
    ent /= Math.max(Math.log(K2), 1e-9)
    // tag
    const vol = this.mu[kb][0], di = Math.abs(this.mu[kb][1])
    let tag: number
    if (vol < 0.85 && di < 0.8) tag = 0
    else if (vol > 1.35 && di > 1.0) tag = 1
    else if (vol > 1.35) tag = 2
    else tag = 3
    const g = new Float64Array([d2min / 50.0, pct, ent, Math.min(K2, 16) / 16.0])
    return { g, tag, pct, d2min, K: K2 }
  }
}

export interface EngineOut {
  R: number      // 能量比 (未来1h方差/EMA96基线 的预测, canonical)
  skew: number   // 方向头 (弱, IC~0.03 — 仅展示)
  d2: number     // GMM 惊异度 (UC3 断路器: >25 → 未来1h方差5.9×中位)
  pct: number
  tag: number
  K: number
}

export interface EngineCfg {
  nRes: number
  nIn: number
  lam: number
  delta: number
  p0: number
  pCap: number
  horizon: number
}

const DEFAULT_CFG: EngineCfg = { nRes: 50, nIn: 8, lam: 0.9999, delta: 1e-6, p0: 4.0, pCap: 1e4, horizon: 4 }

interface FifoEntry { phi: Float64Array; base: number }

/** L1+L2+L3 完整引擎 (单资产) */
export class ReservoirEngine {
  Wr: Float64Array   // nRes × nRes 行主序
  Win: Float64Array  // nRes × nIn 行主序
  nRes: number
  nIn: number
  D: number          // nRes + 4 + 1
  x: Float64Array
  gmm: StreamingGMM
  P: Float64Array    // D × D
  W: Float64Array    // 2 × D
  cfg: EngineCfg
  private fifo: FifoEntry[] = []
  private rlsSteps = 0

  constructor(Wr: number[][], Win: number[][], cfg: Partial<EngineCfg> = {}) {
    this.cfg = { ...DEFAULT_CFG, ...cfg }
    this.nRes = Wr.length
    this.nIn = Win[0].length
    this.D = this.nRes + 4 + 1
    this.Wr = toFlat(Wr)
    this.Win = toFlat(Win)
    this.x = new Float64Array(this.nRes)
    this.gmm = new StreamingGMM(this.nIn)
    this.P = new Float64Array(this.D * this.D)
    for (let i = 0; i < this.D; i++) this.P[i * this.D + i] = this.cfg.p0
    this.W = new Float64Array(2 * this.D)
  }

  /**
   * 单 bar 步进 — 与 run_pass 循环体逐行对应。
   * @param u   特征向量 (nIn 维, 已 clip)
   * @param v4p 当前 bar 的 rolling4( ret² ) — 用于延迟标签 vf
   * @param r4  当前 bar 的 rolling4( ret ) — 用于延迟标签 rf
   * @param base 当前 bar 的 EMA96(v4p) — 入 FIFO 供 4 bar 后作标签基线
   */
  step(u: Float64Array, v4p: number, r4: number, base: number): EngineOut {
    const { nRes, nIn, D } = this
    // ---- L2 ----
    const gmmOut = this.gmm.step(u)
    // ---- L1: x = tanh(Wr·x + Win·u) (leak 1.0) ----
    const pre = new Float64Array(nRes)
    for (let i = 0; i < nRes; i++) {
      let s = 0
      const row = i * nRes
      for (let j = 0; j < nRes; j++) s += this.Wr[row + j] * this.x[j]
      const rowIn = i * nIn
      for (let j = 0; j < nIn; j++) s += this.Win[rowIn + j] * u[j]
      pre[i] = s
    }
    for (let i = 0; i < nRes; i++) this.x[i] = Math.tanh(pre[i])
    // ---- φ = [x, g, 1] ----
    const phi = new Float64Array(D)
    phi.set(this.x, 0)
    phi.set(gmmOut.g, nRes)
    phi[D - 1] = 1.0
    // ---- L3: 延迟标签闭环 ----
    const hzn = this.cfg.horizon
    if (this.fifo.length === hzn) {
      const old = this.fifo[0]
      this.rlsUpdate(old.phi, old.base, v4p, r4)
    }
    this.fifo.push({ phi, base })
    if (this.fifo.length > hzn) this.fifo.shift()
    // ---- 预测 ----
    const yh0 = dotRow(this.W, 0, phi)
    const yh1 = dotRow(this.W, 1, phi)
    const R = Math.expm1(Math.max(0.0, Math.min(3.0, yh0)))
    return { R, skew: yh1, d2: gmmOut.d2min, pct: gmmOut.pct, tag: gmmOut.tag, K: gmmOut.K }
  }

  private rlsUpdate(phiOld: Float64Array, baseOld: number, v4p: number, r4: number) {
    const D = this.D
    const { lam, delta, pCap } = this.cfg
    const baseS = Math.max(baseOld, 1e-14)
    const vf = Number.isFinite(v4p) ? v4p : 0.0
    const rf = Number.isFinite(r4) ? r4 : 0.0
    // y_old = [log1p(vf/base), rf/sqrt(vf+0.15·base)] (clip 同款)
    const y0 = Math.max(0.0, Math.min(3.0, Math.log1p(vf / baseS)))
    const den = Math.sqrt(vf + 0.15 * baseS)
    const y1 = den > 0 ? Math.max(-2.5, Math.min(2.5, rf / den)) : 0.0
    // RLS
    const Px = new Float64Array(D)
    for (let i = 0; i < D; i++) {
      let s = 0
      const row = i * D
      for (let j = 0; j < D; j++) s += this.P[row + j] * phiOld[j]
      Px[i] = s
    }
    let phiPx = 0
    for (let j = 0; j < D; j++) phiPx += phiOld[j] * Px[j]
    const gden = lam + phiPx
    const k = new Float64Array(D)
    for (let i = 0; i < D; i++) k[i] = Px[i] / gden
    const e0 = y0 - dotRow(this.W, 0, phiOld)
    const e1 = y1 - dotRow(this.W, 1, phiOld)
    for (let d = 0; d < D; d++) {
      this.W[0 * D + d] += e0 * k[d]
      this.W[1 * D + d] += e1 * k[d]
    }
    for (let i = 0; i < D; i++) {
      const row = i * D
      for (let j = 0; j < D; j++) this.P[row + j] -= k[i] * Px[j]
    }
    const inv = 1.0 / lam
    for (let i = 0; i < D * D; i++) this.P[i] *= inv
    for (let i = 0; i < D; i++) this.P[i * D + i] += delta
    // 每 64 步对称化 + trace 上限 (Python: P=(P+P.T)/2; tr>p_cap → 缩放)
    this.rlsSteps++
    if (this.rlsSteps % 64 === 0) {
      for (let i = 0; i < D; i++) {
        for (let j = i + 1; j < D; j++) {
          const v = 0.5 * (this.P[i * D + j] + this.P[j * D + i])
          this.P[i * D + j] = v
          this.P[j * D + i] = v
        }
      }
      let tr = 0
      for (let i = 0; i < D; i++) tr += this.P[i * D + i]
      tr /= D
      if (tr > pCap) {
        const f = pCap / tr
        for (let i = 0; i < D * D; i++) this.P[i] *= f
      }
    }
  }
}

function dotRow(W: Float64Array, row: number, phi: Float64Array): number {
  const D = phi.length
  let s = 0
  for (let j = 0; j < D; j++) s += W[row * D + j] * phi[j]
  return s
}

function toFlat(a: number[][]): Float64Array {
  const rows = a.length, cols = a[0].length
  const out = new Float64Array(rows * cols)
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) out[i * cols + j] = a[i][j]
  return out
}

/** 冠军储层矩阵 (seed42 numpy 导出) */
export const CHAMP = params as {
  meta: Record<string, unknown>
  Wr: number[][]
  Win8: number[][]
  Win10: number[][]
}
