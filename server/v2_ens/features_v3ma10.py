"""
特征工程 v3ma10 — v3全部34特征 + MA10块 (2026-09-15, 用户指令: 在v3bal_ens基础上加MA10指标)。

MA10口径 (显式约定, 避免歧义):
  - 周期10根M5收盘的简单移动平均 = 50分钟均价 (模型原生框架是M5, 不是M1/日线);
  - 只用当前M5收盘及更早数据 (rolling窗口右端=当前根, 严格因果);
  - 不把MA10原始价位喂给模型 (金价1800->3300非平稳, 树会按绝对价位分裂=过拟合陷阱,
    v3全部特征都是ATR归一/比值类的尺度不变量), 而是喂3个MA10的"结构化读法":

    ma10_dist  = (close - MA10) / ATR24h     价格偏离其50分钟均值的ATR倍数
                                              (>0=短趋势上方, 突破/回调位置)
    ma10_slope = (MA10 - MA10[-6]) / ATR24h  MA10自身30分钟变化量的ATR倍数
                                              (短趋势方向的平滑读数)
    ma10_run   = 连续收盘位于MA10同一侧的根数(带符号, ±10截断, /10归一)
                                              (趋势vs均价关系的持续性)

机制假设: bal几何TP=3xATR/SL=1.14xATR的大行情须穿越50分钟均价结构;
         v3已有c_dist_s4h(8h均线距离)/momn_12(1h净动量), MA10块补齐"最短端"的
         均线结构信息。若与既有特征高度共线, 增益应接近0 —— 这正是实验要回答的。
"""
import numpy as np
import pandas as pd

from features_v3 import build_features_v3


def _signed_run_vs_level(close, level, cap=10):
    """连续收盘位于level同一侧的根数 (带符号; 平盘沿用前方向), 截断±cap后归一。"""
    sgn = np.sign((close - level).to_numpy(np.float64))
    sgn[sgn == 0] = np.nan
    s = pd.Series(sgn).ffill().fillna(0.0)
    grp = (s != s.shift()).cumsum()
    run = s * (grp.groupby(grp).cumcount() + 1)     # 组内序号×方向
    return pd.Series(run.to_numpy(), index=close.index).clip(-cap, cap) / cap


def build_features_v3ma10(m5, atr_window=288):
    """v3(34) + MA10块(3) = 37特征。返回(特征DataFrame, 特征名列表, ATR列)。"""
    F, feats, atr = build_features_v3(m5, atr_window)
    c = m5["CLOSE"]
    a = atr.replace(0.0, np.nan)
    ma10 = c.rolling(10, min_periods=10).mean()     # 50分钟简单均线(因果)
    f = pd.DataFrame(index=m5.index)
    f["ma10_dist"] = (c - ma10) / a
    f["ma10_slope"] = (ma10 - ma10.shift(6)) / a    # 30分钟MA变化
    f["ma10_run"] = _signed_run_vs_level(c, ma10)
    out = pd.concat([F, f], axis=1)
    return out.astype(np.float32), list(out.columns), atr
