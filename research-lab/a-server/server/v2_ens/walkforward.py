"""
走查验证 (purged walk-forward) — 用户指定的方案:
  首折: 以往 ~31个月训练 -> 验证下1个月
  之后: 已验证的月份滚动并入训练 (扩到36个月后转为滚动窗口, 永远满足"以往24-36个月")
防泄露核心:
  - 每折训练窗右端 purge 掉 (标签前瞻2小时 + 6根M5 embargo) 的行:
    原代码此处未purge, 训练标签直接读入测试期未来数据 (这是原代码泄露点#1)
  - 训练窗内再按时间切 80/20 (带purge gap): 前80%拟合模型+早停, 后20%校准阈值
  - OOS月份从头到尾不参与任何训练/调参/阈值选择
"""
import numpy as np
import pandas as pd


def build_folds(m5, cfg):
    """按月生成走查折。返回fold列表(dict)。"""
    purge_bars = int(np.ceil(cfg["horizon_m1"] / 5)) + cfg["purge_extra_m5"]

    months = m5.index.to_period("M")
    uniq_months = months.unique()
    # 每月的行位置范围
    row_of_month = {}
    month_arr = np.asarray(months)
    pos = np.arange(len(m5))
    for m in uniq_months:
        mask = month_arr == m
        idx = pos[mask]
        row_of_month[m] = (idx[0], idx[-1])

    first_oos = pd.Period(cfg["first_oos_month"], "M")
    last_oos = pd.Period(cfg["last_oos_month"], "M")
    oos_list = [m for m in uniq_months if first_oos <= m <= last_oos]
    if not oos_list:
        raise ValueError("OOS月份区间为空, 请检查配置")

    data_start_month = uniq_months[0]
    folds = []
    for m in oos_list:
        # 训练窗: 数据起点到OOS前一个月; 超过max_train_months则从左侧滚动剔除
        train_months = [x for x in uniq_months if data_start_month <= x < m]
        n_train = len(train_months)
        if n_train > cfg["max_train_months"]:
            train_months = train_months[-cfg["max_train_months"]:]
            n_train = cfg["max_train_months"]
        t_lo = row_of_month[train_months[0]][0]
        t_hi = row_of_month[train_months[-1]][1]
        # --- PURGE: 训练窗右端剔除 (标签前瞻 + embargo) ---
        t_hi_eff = t_hi - purge_bars
        train_rows = np.arange(t_lo, t_hi_eff + 1)

        # --- 内部80/20切分 (阈值校准用, 同样带purge gap) ---
        n_val = int(len(train_rows) * cfg["inner_val_frac"])
        val_rows = train_rows[-n_val:]
        fit_rows = train_rows[:-n_val - purge_bars]   # fit与val之间再留purge gap

        oos_rows = np.arange(row_of_month[m][0], row_of_month[m][1] + 1)
        folds.append({
            "oos_month": str(m), "train_months": n_train,
            "window": f"{train_months[0]}~{train_months[-1]}",
            "train_rows": train_rows, "fit_rows": fit_rows, "val_rows": val_rows,
            "oos_rows": oos_rows, "purge_bars": purge_bars})
    return folds


def time_decay_weights(m5_index, rows, halflife_days):
    """样本时间衰减权重: 距窗末每过半衰期权重减半 (适应金价新regime)。"""
    t = (m5_index[rows].astype("int64") // 10**9 / 86400.0).to_numpy()
    t_ref = t[-1]
    age = t_ref - t
    return np.exp(-np.log(2.0) * np.maximum(age, 0) / halflife_days)
