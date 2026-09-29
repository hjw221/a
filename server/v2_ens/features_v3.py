"""
特征工程 v3 — 盈亏比(PLR)导向重写 (2026-09-08)。

设计动机 (来自v2消融的真实教训):
  - v2把特征从24扩到~70个, OOS反而亏 -$40.5 (legacy24: +$1281.5) => 特征贵精不贵多;
  - 已实现盈亏比由障碍几何决定上限(v3几何: TP=4xATR/SL=1xATR, 随机口径PLR≈3.16),
    特征工程的任务 = 在宽TP几何下把"能走完4xATR的大方向延续"识别出来, 让胜率越过盈亏平衡线。
  - 因此v3只保留四类有明确机制的特征: 趋势质量 / 波动率挤压-爆发 / 突破结构 / 微观确认。

全部特征只使用当前M5收盘及更早的数据 (因果), 缺失值保留NaN交给LGB/XGB原生缺失分支。
周期约定: 1根M5=5分钟; 12=1小时, 48=4小时, 96=8小时, 288=24小时(交易日), 2880=10天。
"""
import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def _bars_since_extreme(a, n, is_max):
    """O(n)单调队列: 距最近n根内最高/最低点过了多少根K线。"""
    out = np.full(len(a), np.nan)
    q = np.empty(len(a), dtype=np.int64)
    head, tail = 0, 0
    for i in range(len(a)):
        while tail > head:
            j = q[tail - 1]
            worse = (a[j] <= a[i]) if is_max else (a[j] >= a[i])
            if worse:
                tail -= 1
            else:
                break
        q[tail] = i
        tail += 1
        while q[head] <= i - n:
            head += 1
        out[i] = i - q[head]
    return out


def _rolling_slope(close, n):
    """n根K线线性回归斜率 (向量化rolling求和)。"""
    v = close.to_numpy(np.float64)
    i = np.arange(len(v), dtype=np.float64)
    iy = pd.Series(i * v, index=close.index)
    sy = close.rolling(n, min_periods=n).sum()
    siy = iy.rolling(n, min_periods=n).sum()
    k = np.arange(len(v), dtype=np.float64)
    koff = np.maximum(k - n + 1, 0.0)
    xbar = (n - 1) / 2.0
    denom = n * (n - 1) * (2 * n - 1) / 6.0 - n * xbar * xbar
    num = siy - (koff + xbar) * sy
    return num / denom


def _signed_run_len(close, cap=10):
    """同向连续收盘根数 (带符号, 平盘沿用前方向), 截断到±cap。"""
    d = close.diff()
    sgn = np.sign(d.to_numpy(np.float64))
    sgn[sgn == 0] = np.nan
    s = pd.Series(sgn).ffill().fillna(0.0)
    grp = (s != s.shift()).cumsum()
    run = s * (grp.groupby(grp).cumcount() + 1)   # 组内序号×方向
    return pd.Series(run.to_numpy(), index=close.index).clip(-cap, cap) / cap


def _efficiency_ratio(close, n):
    """Kaufman效率比: |净位移| / 路径总长, 趋势质量指标 (1=完美直线, 0=纯噪音)。"""
    num = (close - close.shift(n)).abs()
    den = close.diff().abs().rolling(n, min_periods=n).sum()
    return num / den.replace(0.0, np.nan)


def build_features_v3(m5, atr_window=288):
    """v3特征。输入M5 DataFrame, 返回(特征DataFrame, 特征名列表, ATR列)。"""
    f = pd.DataFrame(index=m5.index)
    c, h, l, o = m5["CLOSE"], m5["HIGH"], m5["LOW"], m5["OPEN"]
    v = m5["TICKVOL"].astype(np.float64)

    # ============ 波动率基准 (障碍的锚, 也是特征分母) ============
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_window, min_periods=atr_window).mean()

    # ---- [1] 波动率regime: 挤压-爆发循环 (宽TP几何只在爆发后有效) ----
    f["atr_ratio"] = tr.rolling(24, min_periods=24).mean() / atr          # 短/长波动比
    f["tr_over_atr"] = tr / atr                                           # 当前K线爆发度
    bb_w = 4.0 * c.rolling(96, min_periods=96).std()                      # 8h布林带宽
    f["squeeze"] = bb_w / bb_w.rolling(2880, min_periods=2880).median()   # 带宽/10天中位 (<1=挤压)
    hi96r, lo96r = h.rolling(96, min_periods=96).max(), l.rolling(96, min_periods=96).min()
    hi288r, lo288r = h.rolling(288, min_periods=288).max(), l.rolling(288, min_periods=288).min()
    f["range_ratio"] = (hi96r - lo96r) / (hi288r - lo288r).replace(0.0, np.nan)  # 短程/长程区间压缩

    # ---- [2] ATR归一化动量 (方向核心; 用ATR做分母 => 金价翻倍/波动放大3倍仍可比) ----
    for p in [12, 48, 96, 288]:
        f[f"momn_{p}"] = (c - c.shift(p)) / atr
    f["mom_acc"] = f["momn_12"] - 0.25 * f["momn_48"]                     # 加速度(短动量超出基线部分)

    # ---- [3] 趋势质量: 效率比 + 回归斜率 (宽TP只该在高质量趋势上出手) ----
    f["er_24"] = _efficiency_ratio(c, 24)
    f["er_96"] = _efficiency_ratio(c, 96)
    f["er_288"] = _efficiency_ratio(c, 288)
    f["slope_96"] = _rolling_slope(c, 96) / atr.replace(0.0, np.nan)
    f["slope_288"] = _rolling_slope(c, 288) / atr.replace(0.0, np.nan)

    # ---- [4] 突破结构 (4xATR的大行情多从区间边缘启动) ----
    hi_prev = h.rolling(288, min_periods=288).max().shift(1)              # 前24h最高(不含当前)
    lo_prev = l.rolling(288, min_periods=288).min().shift(1)
    f["dist_hi_288"] = (c - hi_prev) / atr                                # 突破深度(>0=已破24h高)
    f["dist_lo_288"] = (c - lo_prev) / atr
    rng288 = (hi288r - lo288r).replace(0.0, np.nan)
    f["pos_in_range"] = (c - lo288r) / rng288                             # 0=24h最低 1=24h最高
    f["bars_since_hi96"] = _bars_since_extreme(h.to_numpy(np.float64), 96, True)
    f["bars_since_lo96"] = _bars_since_extreme(l.to_numpy(np.float64), 96, False)
    f["run_len"] = _signed_run_len(c)                                     # 连涨/连跌结构

    # ---- [5] 超涨超跌 (ATR归一; 空头侧的燃料) ----
    f["c_dist_s4h"] = (c - c.rolling(96, min_periods=96).mean()) / atr
    f["c_dist_d1"] = (c - c.rolling(288, min_periods=288).mean()) / atr

    # ---- [6] 微观确认: 量能与K线形态 ----
    f["vol_z_288"] = (v - v.rolling(288, min_periods=288).mean()) / \
                     v.rolling(288, min_periods=288).std().replace(0.0, np.nan)
    f["vol_ratio_24_288"] = v.rolling(24, min_periods=24).mean() / v.rolling(288, min_periods=288).mean()
    rng1 = (h - l).replace(0.0, np.nan)
    f["body"] = (c - o) / rng1
    f["upper_shadow"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / rng1
    f["lower_shadow"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / rng1

    # ---- [7] 时间结构 (金价的时段效应显著: 伦敦/纽约流动性与波动分布不同) ----
    hr = m5.index.hour.values + m5.index.minute.values / 60.0
    f["hour_sin"] = np.sin(hr * 2 * np.pi / 24)
    f["hour_cos"] = np.cos(hr * 2 * np.pi / 24)
    f["dow_sin"] = np.sin(m5.index.dayofweek * 2 * np.pi / 7)
    f["dow_cos"] = np.cos(m5.index.dayofweek * 2 * np.pi / 7)
    f["session_london"] = ((m5.index.hour >= 7) & (m5.index.hour < 13)).astype(np.int8)
    f["session_ny"] = ((m5.index.hour >= 13) & (m5.index.hour < 21)).astype(np.int8)

    # ---- [8] 流动性状态 (点差走阔=事件/低流动性, 宽TP几何的天敌) ----
    sp = m5["SPREAD"].astype(np.float64).replace(0.0, np.nan)
    f["spread_rel"] = sp / sp.rolling(288, min_periods=144).median()

    feats = list(f.columns)
    return f.astype(np.float32), feats, atr
