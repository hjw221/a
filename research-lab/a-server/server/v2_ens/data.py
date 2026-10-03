"""
数据底座: M1加载 -> 清洗 -> M5重采样 -> 点差缺失填补
关键修正 vs 原代码:
  - 原代码把 2022-23 的 SPREAD=0 直接当 0 成本 -> PnL 高估。此处按月中位数填补。
  - 统一 M1/M5 口径 (原 run_pipeline 用 M5、retrain_weekly 用 M1, 互相打架)。
缓存: 首次处理后 pickle 缓存, 之后秒级加载。
"""
import os
import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def epoch_minutes(idx):
    """DatetimeIndex -> epoch 分钟 int64. pandas 2/3 双安全 (datetime64[s] 中转,
    修复 [us] dtype 下 astype(int64)//1e9//60 的千秒域 bug — v16 同型)."""
    return (idx.astype("datetime64[s]").astype("int64") // 60).to_numpy(np.int64)


def load_raw_m1(csv_path):
    """加载原始M1, 只读需要的列, 控制内存。"""
    df = pd.read_csv(csv_path, sep="\t",
                     usecols=["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>",
                              "<CLOSE>", "<TICKVOL>", "<SPREAD>"])
    df.columns = ["DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE", "TICKVOL", "SPREAD"]
    # 日期字符串拼接解析 1.6M 行: 用唯一日期映射加速
    uniq_dates = df["DATE"].unique()
    dmap = {d: pd.to_datetime(d, format="%Y.%m.%d") for d in uniq_dates}
    dates = df["DATE"].map(dmap)
    secs = df["TIME"].str.slice(0, 2).astype(np.int32) * 3600 + \
           df["TIME"].str.slice(3, 5).astype(np.int32) * 60
    df.index = dates + pd.to_timedelta(secs, unit="s")
    df = df[["OPEN", "HIGH", "LOW", "CLOSE", "TICKVOL", "SPREAD"]].sort_index()
    return df


def resample_m5(m1):
    """M1 -> M5。等于把数据切小5倍 (2核CPU可承受), 时间跨度完整保留。"""
    agg = m1.resample("5min").agg(
        OPEN=("OPEN", "first"), HIGH=("HIGH", "max"), LOW=("LOW", "min"),
        CLOSE=("CLOSE", "last"), TICKVOL=("TICKVOL", "sum"),
        SPREAD=("SPREAD", "median"))
    agg = agg.dropna(subset=["OPEN"])  # 周末/缺口空箱
    return agg


def impute_spread(m5, point_value):
    """
    2022-2023 SPREAD 全为 0 (导出缺失, 并非真零点差)。
    策略: 每月取非零点差中位数; 全零月用时间上最近的可用月中位数 (前向/后向填充)。
    返回: 每根M5的往返点差成本(美元/手单位)序列 + 每月中位数表(诊断用)。
    """
    med = m5["SPREAD"].copy()
    med[med == 0] = np.nan
    monthly = med.groupby(m5.index.to_period("M")).median()
    monthly = monthly.ffill().bfill()
    month_key = m5.index.to_period("M")
    spread_pts = m5["SPREAD"].to_numpy(dtype=np.float64).copy()
    fill_map = monthly.to_dict()
    miss = spread_pts <= 0
    spread_pts[miss] = np.array([fill_map[k] for k in month_key[miss]], dtype=np.float64)
    spread_cost = spread_pts * point_value  # 每笔往返成本(美元)
    return spread_cost, monthly


def load_data(cfg, force=False):
    """主入口: 返回 (m5_df, m1_arrays, spread_cost, monthly_spread)。带pickle缓存。"""
    tag = os.path.basename(cfg["data_path"]).split(".")[0][:30]
    cache_fp = os.path.join(CACHE_DIR, f"prep_{tag}_{cfg['resample_rule']}.pkl")
    if os.path.exists(cache_fp) and not force:
        with open(cache_fp, "rb") as f:
            m5, m1_pack, spread_cost, monthly = pd.read_pickle(f)
        print(f"[data] 缓存加载: {len(m5):,} 根M5 / {len(m1_pack[0]):,} 根M1")
        return m5, m1_pack, spread_cost, monthly

    print("[data] 首次加载M1 CSV (约30-60秒)...")
    m1 = load_raw_m1(cfg["data_path"])
    print(f"[data] M1: {len(m1):,} 行  {m1.index[0]} ~ {m1.index[-1]}")
    m5 = resample_m5(m1)
    print(f"[data] M5重采样完成: {len(m5):,} 行")
    spread_cost, monthly = impute_spread(m5, cfg["point_value"])

    # M1 精度数组 (标签/回测用): time为int64分钟, 便于numba二分
    m1_times = epoch_minutes(m1.index)
    m1_pack = (
        m1_times,
        m1["OPEN"].to_numpy(np.float64), m1["HIGH"].to_numpy(np.float64),
        m1["LOW"].to_numpy(np.float64), m1["CLOSE"].to_numpy(np.float64),
    )
    with open(cache_fp, "wb") as f:
        pd.to_pickle((m5, m1_pack, spread_cost, monthly), f)
    print(f"[data] 点差填补后月度中位数(美元/笔): 首月={monthly.iloc[0]:.3f} 末月={monthly.iloc[-1]:.3f}")
    print(f"[data] 已缓存 -> {cache_fp}")
    return m5, m1_pack, spread_cost, monthly


def m5_times_minutes(m5):
    return epoch_minutes(m5.index)
