"""
特征工程 s1 — M1 剥头皮专用 (Task14, 2026-09-27)。

设计原则 (继承 v3 的教训, 缩放到 M1 节奏):
  - v3 在 M5 上验证有效的设计哲学: 波动率regime / ATR归一动量 / 趋势质量 /
    突破结构 / 超涨超跌 / 微观确认 / 时段 / 点差状态;
  - 全部窗口按 M1 换算: 12=12min, 60=1h, 240=4h, 480=8h, 1440=24h, 14400=10天;
  - 新增 M1 微观结构: 3分钟脉冲 / 15根内上涨占比 (剥头皮的时间尺度);
  - 全部特征只用当前M1收盘及更早数据 (因果), NaN 交给 LGB/XGB 原生缺失分支。

与"老路"的区别: 此前所有变体的特征都在 M5 网格上; s1 是首个 M1 原生特征集。
"""
import numpy as np
import pandas as pd

from features_v3 import _bars_since_extreme, _rolling_slope, _efficiency_ratio


def build_features_s1(m1):
    """输入 M1 DataFrame (OPEN/HIGH/LOW/CLOSE/TICKVOL/SPREAD, DatetimeIndex)。
    返回 (特征DataFrame float32, 特征名列表, ATR1440序列)。"""
    f = pd.DataFrame(index=m1.index)
    c, h, l, o = m1["CLOSE"], m1["HIGH"], m1["LOW"], m1["OPEN"]
    v = m1["TICKVOL"].astype(np.float64)

    # ============ 波动率基准: 24h M1 ATR (特征分母; 障碍另用固定$) ============
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(1440, min_periods=1440).mean()

    # ---- [1] 波动率regime (短/长波动比; 剥头皮只在合适的波动节奏上出手) ----
    f["atr_ratio"] = tr.rolling(30, min_periods=30).mean() / atr
    f["tr_over_atr"] = tr / atr
    bb_w = 4.0 * c.rolling(240, min_periods=240).std()
    f["squeeze"] = bb_w / bb_w.rolling(14400, min_periods=14400).median()
    hi240, lo240 = h.rolling(240, min_periods=240).max(), l.rolling(240, min_periods=240).min()
    hi1440, lo1440 = h.rolling(1440, min_periods=1440).max(), l.rolling(1440, min_periods=1440).min()
    f["range_ratio"] = (hi240 - lo240) / (hi1440 - lo1440).replace(0.0, np.nan)

    # ---- [2] ATR归一动量 (12min/1h/4h/24h) ----
    for p in [12, 60, 240, 1440]:
        f[f"momn_{p}"] = (c - c.shift(p)) / atr
    f["mom_acc"] = f["momn_12"] - 0.25 * f["momn_60"]

    # ---- [3] 趋势质量 ----
    f["er_24"] = _efficiency_ratio(c, 24)
    f["er_240"] = _efficiency_ratio(c, 240)
    f["er_1440"] = _efficiency_ratio(c, 1440)
    f["slope_240"] = _rolling_slope(c, 240) / atr.replace(0.0, np.nan)
    f["slope_1440"] = _rolling_slope(c, 1440) / atr.replace(0.0, np.nan)

    # ---- [4] 突破结构 ----
    hi_prev = h.rolling(1440, min_periods=1440).max().shift(1)
    lo_prev = l.rolling(1440, min_periods=1440).min().shift(1)
    f["dist_hi_1440"] = (c - hi_prev) / atr
    f["dist_lo_1440"] = (c - lo_prev) / atr
    rng1440 = (hi1440 - lo1440).replace(0.0, np.nan)
    f["pos_in_range"] = (c - lo1440) / rng1440
    f["bars_since_hi240"] = _bars_since_extreme(h.to_numpy(np.float64), 240, True)
    f["bars_since_lo240"] = _bars_since_extreme(l.to_numpy(np.float64), 240, False)
    # 连涨连跌 (带符号, cap 10)
    d = c.diff()
    sgn = np.sign(d.to_numpy(np.float64))
    sgn[sgn == 0] = np.nan
    s = pd.Series(sgn).ffill().fillna(0.0)
    grp = (s != s.shift()).cumsum()
    run = s * (grp.groupby(grp).cumcount() + 1)
    f["run_len"] = pd.Series(run.to_numpy(), index=c.index).clip(-10, 10) / 10.0

    # ---- [5] 超涨超跌 ----
    f["c_dist_s1h"] = (c - c.rolling(60, min_periods=60).mean()) / atr
    f["c_dist_d1"] = (c - c.rolling(1440, min_periods=1440).mean()) / atr

    # ---- [6] 微观确认 (M1特有部分) ----
    f["vol_z_1440"] = (v - v.rolling(1440, min_periods=1440).mean()) / \
                      v.rolling(1440, min_periods=1440).std().replace(0.0, np.nan)
    f["vol_ratio_12_1440"] = v.rolling(12, min_periods=12).mean() / v.rolling(1440, min_periods=1440).mean()
    rng1 = (h - l).replace(0.0, np.nan)
    f["body"] = (c - o) / rng1
    f["upper_shadow"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / rng1
    f["lower_shadow"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / rng1
    f["ret_3"] = (c - c.shift(3)) / atr                      # 3分钟脉冲
    f["up_bars_15"] = (c.diff() > 0).astype(np.float64).rolling(15, min_periods=15).mean()

    # ---- [7] 时间结构 ----
    hr = m1.index.hour.values + m1.index.minute.values / 60.0
    f["hour_sin"] = np.sin(hr * 2 * np.pi / 24)
    f["hour_cos"] = np.cos(hr * 2 * np.pi / 24)
    f["dow_sin"] = np.sin(m1.index.dayofweek * 2 * np.pi / 7)
    f["dow_cos"] = np.cos(m1.index.dayofweek * 2 * np.pi / 7)
    f["session_london"] = ((m1.index.hour >= 7) & (m1.index.hour < 13)).astype(np.int8)
    f["session_ny"] = ((m1.index.hour >= 13) & (m1.index.hour < 21)).astype(np.int8)

    # ---- [8] 流动性状态 ----
    sp = m1["SPREAD"].astype(np.float64).replace(0.0, np.nan)
    f["spread_rel"] = sp / sp.rolling(1440, min_periods=720).median()

    feats = list(f.columns)
    return f.astype(np.float32), feats, atr
