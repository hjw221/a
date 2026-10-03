/**
 * pipeline.ts — M1 CSV → 清洗 → M15 重聚合 (流式, 可增量)
 * =====================================================
 * 与 reservoir_engine2.load_m15 逐字对应的清洗口径:
 *   br = high - low
 *   rng_z = (br - rolling288(br).mean) / rolling288(br).std   [ddof=1]
 *   剔除: (tickvol<=5 且 br<0.01) 或 |rng_z|>15   [前287根 z=NaN → 保留]
 *   M15: label=left closed=left, agg(open first/high max/low min/close last/tickvol sum)
 * 有状态: tail 模式下持续接收新 M1 行 (文件追加即实时喂入)。
 */

export interface M15Bar {
  slot: number      // 15分钟槽序号 (= epoch分钟/15)
  tMin: number      // 槽起始 epoch 分钟
  open: number
  high: number
  low: number
  close: number
  tickvol: number
}

const RING = 288

export class M1ToM15Pipeline {
  private brRing = new Float64Array(RING)
  private ringN = 0
  private ringIdx = 0
  private sumBr = 0
  private sumBr2 = 0
  private curSlot = -1
  private cur: { o: number; h: number; l: number; c: number; v: number } | null = null
  bars: M15Bar[] = []

  /** 喂入一行 M1 (已排序; 返回该行是否触发新 M15 bar 完结) */
  feed(tMin: number, open: number, high: number, low: number, close: number, tickvol: number): M15Bar | null {
    // ---- 清洗 (rolling288 over RAW 流) ----
    const br = high - low
    let keep = true
    if (tickvol <= 5 && br < 0.01) keep = false
    if (keep && this.ringN >= RING) {
      const n = this.ringN
      const mean = this.sumBr / n
      const varr = (this.sumBr2 - n * mean * mean) / (n - 1)
      if (varr > 0) {
        const z = (br - mean) / Math.sqrt(varr)
        if (Math.abs(z) > 15) keep = false
      }
    }
    // rolling 更新 (无论 keep 与否 — pandas 在原始序列上滚动)
    {
      const old = this.brRing[this.ringIdx]
      if (this.ringN < RING) {
        this.ringN++
      } else {
        this.sumBr -= old
        this.sumBr2 -= old * old
      }
      this.brRing[this.ringIdx] = br
      this.sumBr += br
      this.sumBr2 += br * br
      this.ringIdx = (this.ringIdx + 1) % RING
    }
    if (!keep) return null

    // ---- M15 聚合 ----
    const slot = Math.floor(tMin / 15)
    let emitted: M15Bar | null = null
    if (slot !== this.curSlot) {
      if (this.cur) emitted = this.flush()
      this.curSlot = slot
      this.cur = { o: open, h: high, l: low, c: close, v: tickvol }
    } else if (this.cur) {
      const c = this.cur
      if (high > c.h) c.h = high
      if (low < c.l) c.l = low
      c.c = close
      c.v += tickvol
    }
    return emitted
  }

  /** 强制完结当前槽 (数据末尾) */
  flushTail(): M15Bar | null {
    if (!this.cur) return null
    return this.flush()
  }

  private flush(): M15Bar {
    const c = this.cur!
    const bar: M15Bar = {
      slot: this.curSlot, tMin: this.curSlot * 15,
      open: c.o, high: c.h, low: c.l, close: c.c, tickvol: c.v,
    }
    this.bars.push(bar)
    this.cur = null
    return bar
  }
}

/** "2022.01.02\t23:05:00" → epoch 分钟 (UTC 框架, 与数据自身时钟一致) */
export function parseRow(line: string): { tMin: number; open: number; high: number; low: number; close: number; tickvol: number } | null {
  if (line.length < 20) return null
  const t1 = line.indexOf('\t')
  if (t1 < 0) return null
  const dateStr = line.slice(0, t1)
  const rest = line.slice(t1 + 1)
  const t2 = rest.indexOf('\t')
  const timeStr = rest.slice(0, t2)
  const cols = rest.slice(t2 + 1).split('\t')
  if (cols.length < 5) return null
  // date: YYYY.MM.DD
  const y = +dateStr.slice(0, 4), mo = +dateStr.slice(5, 7), d = +dateStr.slice(8, 10)
  const h = +timeStr.slice(0, 2), mi = +timeStr.slice(3, 5)
  const tMin = Math.floor(Date.UTC(y, mo - 1, d, h, mi) / 60000)
  const open = parseFloat(cols[0]), high = parseFloat(cols[1]),
        low = parseFloat(cols[2]), close = parseFloat(cols[3]),
        tickvol = parseFloat(cols[4])
  if (!Number.isFinite(open) || !Number.isFinite(close)) return null
  return { tMin, open, high, low, close, tickvol }
}

/** 整文件回放 (首行表头跳过); 返回 [bars, 处理行数, 文件字节偏移] */
export async function replayFile(path: string, pipe: M1ToM15Pipeline): Promise<{ rows: number; bytes: number }> {
  const file = Bun.file(path)
  const size = file.size
  const stream = file.stream()
  const decoder = new TextDecoder()
  let buf = ''
  let rows = 0
  let first = true
  const reader = stream.getReader()
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      let nl: number
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl).replace(/\r$/, '')
        buf = buf.slice(nl + 1)
        if (first) { first = false; continue }
        const r = parseRow(line)
        if (r) { pipe.feed(r.tMin, r.open, r.high, r.low, r.close, r.tickvol); rows++ }
      }
    }
    // 尾行
    if (buf.length > 0) {
      const r = parseRow(buf.replace(/\r$/, ''))
      if (r) { pipe.feed(r.tMin, r.open, r.high, r.low, r.close, r.tickvol); rows++ }
    }
  } finally {
    reader.releaseLock()
  }
  return { rows, bytes: size }
}
