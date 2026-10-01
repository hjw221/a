"""
特征工程 v2 (约70个特征, 全部只用历史数据) + legacy24 (原代码24特征的精确复刻, 用于消融对比)。
相对原代码的升级:
  - 收益/波动率周期从 48 根M5 扩展到 288 根M5 (24小时), 覆盖跨日结构
  - ATR 归一化动量/价格位置 (对金价翻倍、波动率放大3-5倍自适应, 原代码裸pct_change在2024-2026年被波动淹没)
  - RSI / 随机指标 / 回归斜率 (趋势结构)
  - TickVol 微观结构 (原代码有列但从未使用)
  - 点差状态特征 (点差走阔 = 事件/低流动性预警)
  - 亚洲/伦敦/纽约时段标记
缺失值不 dropna 填0 (原做法注入假信息), 保留 NaN 交给 LightGBM/XGBoost 原生缺失分支。
"""
import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def _bars_since_extreme(a, n, is_max):
    """O(n)单调队列: 距最近n根内最高/最低点过了多少根K线。"""
    out = np.full(len(a), np.nan)
    q = np.empty(len(a), dtype=np.int64)  # 存索引的双端队列
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
    """n根K线线性回归斜率, 向量化(rolling求和), 替代逐窗口python调用。"""
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


def _rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    ru = up.rolling(n, min_periods=n).mean()
    rd = dn.rolling(n, min_periods=n).mean()
    rs = ru / rd.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)) / 100.0  # 归一化到0-1


def _slope_norm(close, n, atr):
    """n根K线线性回归斜率 / ATR: 趋势强度(每根K线的平均漂移, 用ATR归一)。"""
    return _rolling_slope(close, n) / atr.replace(0.0, np.nan)


def build_features_v2(m5, atr_window=288):
    """v2特征。输入M5 DataFrame, 返回(特征DataFrame, 特征名列表, ATR列)。"""
    f = pd.DataFrame(index=m5.index)
    c, h, l, o = m5["CLOSE"], m5["HIGH"], m5["LOW"], m5["OPEN"]
    v = m5["TICKVOL"].astype(np.float64)

    # --- 真实波幅 ATR (波动率基准, 障碍缩放的锚) ---
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_window, min_periods=atr_window).mean()
    f["atr"] = atr
    f["atr_ratio"] = tr.rolling(24, min_periods=24).mean() / atr          # 短期/长期波动 regime
    f["tr_over_atr"] = tr / atr                                           # 当前K线爆发度

    # --- 多周期收益 (5m ~ 24h) ---
    for p in [1, 2, 3, 6, 12, 24, 48, 96, 288]:
        f[f"ret_{p}"] = c.pct_change(p)

    # --- 多周期已实现波动率 ---
    r1 = c.pct_change()
    for p in [6, 12, 24, 48, 96, 288]:
        f[f"vol_{p}"] = r1.rolling(p, min_periods=p).std()
    f["vol_ratio"] = r1.rolling(24).std() / r1.rolling(288, min_periods=288).std()

    # --- ATR归一化价格位置 (波动率自适应, 替代裸z-score) ---
    for p, name in [(24, "s1h"), (96, "s4h"), (288, "d1")]:
        sma = c.rolling(p, min_periods=p).mean()
        f[f"c_dist_{name}"] = (c - sma) / atr
    bb_std = r1.rolling(96, min_periods=96).std() * np.sqrt(96)
    f["bb_z_4h"] = (c - c.rolling(96, min_periods=96).mean()) / bb_std.replace(0.0, np.nan)
    hi288, lo288 = h.rolling(288, min_periods=288).max(), l.rolling(288, min_periods=288).min()
    rng = (hi288 - lo288).replace(0.0, np.nan)
    f["pos_in_range"] = (c - lo288) / rng                                  # 0=24h最低 1=24h最高

    # --- K线形态 (与原代码同口径, 但ATR归一) ---
    rng1 = (h - l).replace(0.0, np.nan)
    f["body"] = (c - o) / rng1
    f["upper_shadow"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / rng1
    f["lower_shadow"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / rng1

    # --- 动量指标 ---
    f["rsi_14"] = _rsi(c, 14)
    f["rsi_56"] = _rsi(c, 56)
    lo14, hi14 = l.rolling(14, min_periods=14).min(), h.rolling(14, min_periods=14).max()
    f["stoch_14"] = (c - lo14) / (hi14 - lo14).replace(0.0, np.nan)
    f["slope_96"] = _slope_norm(c, 96, atr)
    f["slope_288"] = _slope_norm(c, 288, atr)

    # --- TickVol 微观结构 ---
    f["vol_z_288"] = (v - v.rolling(288, min_periods=288).mean()) / v.rolling(288, min_periods=288).std().replace(0.0, np.nan)
    f["vol_ratio_24_288"] = v.rolling(24, min_periods=24).mean() / v.rolling(288, min_periods=288).mean()

    # --- 点差状态 (0视为缺失 -> NaN) ---
    sp = m5["SPREAD"].astype(np.float64).replace(0.0, np.nan)
    sp_med = sp.rolling(288, min_periods=144).median()
    f["spread_rel"] = sp / sp_med
    f["spread_abs"] = sp

    # --- 时间结构 ---
    hr = m5.index.hour.values + m5.index.minute.values / 60.0
    f["hour_sin"] = np.sin(hr * 2 * np.pi / 24)
    f["hour_cos"] = np.cos(hr * 2 * np.pi / 24)
    f["dow_sin"] = np.sin(m5.index.dayofweek * 2 * np.pi / 7)
    f["dow_cos"] = np.cos(m5.index.dayofweek * 2 * np.pi / 7)
    f["session_asia"] = ((m5.index.hour >= 0) & (m5.index.hour < 7)).astype(np.int8)
    f["session_london"] = ((m5.index.hour >= 7) & (m5.index.hour < 13)).astype(np.int8)
    f["session_ny"] = ((m5.index.hour >= 13) & (m5.index.hour < 21)).astype(np.int8)

    # --- 距近期极值的K线数 (突破结构记忆, O(n)单调队列) ---
    f["bars_since_hi96"] = _bars_since_extreme(h.to_numpy(np.float64), 96, True)
    f["bars_since_lo96"] = _bars_since_extreme(l.to_numpy(np.float64), 96, False)

    feats = [c_ for c_ in f.columns if c_ != "atr"]
    return f.astype(np.float32), feats, atr


def build_features_legacy(m5):
    """原代码24特征的精确复刻 (M5口径), 仅作消融对照。"""
    f = pd.DataFrame(index=m5.index)
    c, o, h, l = m5["CLOSE"], m5["OPEN"], m5["HIGH"], m5["LOW"]
    feats = []
    for period in [1, 2, 3, 6, 12, 24, 48]:
        f[f"ret_{period}"] = c.pct_change(periods=period)
        feats.append(f"ret_{period}")
        if period > 1:
            f[f"vol_{period}"] = c.pct_change().rolling(window=period).std()
            feats.append(f"vol_{period}")
    f["body"] = (c - o) / o
    f["upper_shadow"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / o
    f["lower_shadow"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / o
    feats += ["body", "upper_shadow", "lower_shadow"]
    f["hour_sin"] = np.sin(m5.index.hour * (2. * np.pi / 24))
    f["hour_cos"] = np.cos(m5.index.hour * (2. * np.pi / 24))
    f["min_sin"] = np.sin(m5.index.minute * (2. * np.pi / 60))
    f["min_cos"] = np.cos(m5.index.minute * (2. * np.pi / 60))
    f["dow_sin"] = np.sin(m5.index.dayofweek * (2. * np.pi / 7))
    f["dow_cos"] = np.cos(m5.index.dayofweek * (2. * np.pi / 7))
    feats += ["hour_sin", "hour_cos", "min_sin", "min_cos", "dow_sin", "dow_cos"]
    sp = m5["SPREAD"].astype(np.float64)  # 原代码直接用, 0就是0
    f["SPREAD"] = sp
    feats.append("SPREAD")
    return f, feats
