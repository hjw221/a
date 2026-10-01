"""
特征工程 v4 — v3 + DXY跨资产 (路线A, 2026-09-15)。

设计动机 (来自v2/v5/v6的真实教训: 特征贵精不贵多):
  - 底座内三张牌只有集成幸存; 频率拓展证明低分位信号枯竭 => 更高期望只能加【新信息源】;
  - 金价以美元定价, DXY是最直接的宏观驱动 (负相关随regime浮动)。本版加且只加8个
    有明确机制的跨资产特征, 全部在DXY自身M5序列上因果计算, 再对齐到XAU M5时间轴。

对齐与因果 (防泄露关键):
  - DXY与XAUUSD同一MT5服务器导出, 时间戳同源;
  - XAU M5 bar (收盘于T) 的特征只使用 DXY bar (收盘于T) 及更早 => 无前瞻;
  - XAU有bar而DXY缺失的时刻: DXY状态特征前向沿用 (最后已知值), 同时给 dxy_stale
    (距DXY最近一次真实bar过了多少根XAU M5, 归一到24h) 让模型自己识别数据陈旧;
  - 截断vs全量逐单元一致性由 scripts/smoke_features_v4.py 校验。

8个新特征 (34+8=42):
  dxy_momn_48/288 : DXY动量(ATR归一) — 美元走弱=金价顺风
  dxy_er_96       : DXY Kaufman效率比 — 美元趋势质量
  dxy_atr_ratio   : DXY挤压-爆发 (短/长波动比)
  dxy_pos_in_range: DXY在24h区间位置 (美元压顶=金价承压)
  dxy_dist_d1     : DXY偏离24h均值 (美元超涨)
  xau_dxy_relmom  : 金动量+美指动量 (负相关下的背离度: 同向共振regime识别)
  dxy_stale       : DXY数据陈旧度 (0=新鲜, 1=陈旧24h)
"""
import numpy as np
import pandas as pd

from features_v3 import build_features_v3, _efficiency_ratio


def build_features_v4(m5, m5_dxy, atr_window=288):
    """v4特征 = v3全部 + DXY跨资产8个。返回(特征DataFrame, 特征名列表, ATR列)。"""
    F3, feats, atr = build_features_v3(m5, atr_window)
    if m5_dxy is None:
        return F3, feats, atr                      # 无DXY数据 -> 退化为纯v3

    f = F3.copy()
    c = m5_dxy["CLOSE"].astype(np.float64)
    h, l = m5_dxy["HIGH"].astype(np.float64), m5_dxy["LOW"].astype(np.float64)

    # ---- DXY自身序列上的因果计算 (滚动窗口只用过去) ----
    tr_d = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr_d = tr_d.rolling(atr_window, min_periods=atr_window).mean()

    g = pd.DataFrame(index=m5_dxy.index)
    g["dxy_momn_48"] = (c - c.shift(48)) / atr_d
    g["dxy_momn_288"] = (c - c.shift(288)) / atr_d
    g["dxy_er_96"] = _efficiency_ratio(c, 96)
    g["dxy_atr_ratio"] = tr_d.rolling(24, min_periods=24).mean() / atr_d
    hi288 = h.rolling(288, min_periods=288).max()
    lo288 = l.rolling(288, min_periods=288).min()
    g["dxy_pos_in_range"] = (c - lo288) / (hi288 - lo288).replace(0.0, np.nan)
    g["dxy_dist_d1"] = (c - c.rolling(288, min_periods=288).mean()) / atr_d

    # ---- 对齐到XAU M5时间轴: 状态前向沿用 (只携带最后已知值, 无未来信息) ----
    g_al = g.reindex(m5.index).ffill()

    # ---- 数据陈旧度: 距DXY最近一次真实bar过了多少根XAU M5 (归一到24h=288根) ----
    present = m5.index.isin(m5_dxy.index)
    idx = np.arange(len(m5), dtype=np.float64)
    last_present = np.where(present, idx, np.nan)
    stale = idx - pd.Series(last_present).ffill().to_numpy()
    g_al["dxy_stale"] = stale / 288.0

    # ---- 背离度: 金动量 + 美指动量 (负相关下同号=共振, 异号=对冲压力) ----
    g_al["xau_dxy_relmom"] = f["momn_288"].astype(np.float64) + g_al["dxy_momn_288"]

    for col in ["dxy_momn_48", "dxy_momn_288", "dxy_er_96", "dxy_atr_ratio",
                "dxy_pos_in_range", "dxy_dist_d1", "xau_dxy_relmom", "dxy_stale"]:
        f[col] = g_al[col].astype(np.float32)

    feats = list(f.columns)
    return f.astype(np.float32), feats, atr
