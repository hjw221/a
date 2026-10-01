"""
M1(1分K线) WR-max 几何标定。用户要求: 1分K线下 30笔/天, 只要求胜率(越高越好)。
规则与前几轮一致: 只用首训练窗 2022-01~2024-07 真实数据(右端purge), 无未来信息。
设计:
  - 信号K线 = M1本身(bar_minutes=1), 入场 = 下一根M1开盘;
  - 高胜率几何 = 近TP + 宽SL; TP下限 = 3×点差成本(保证"胜"在美元口径也为赢,
    用户只要胜率, 但我们不让几何退化成"赢单即亏钱"的假胜率);
  - SL下限沿用 3×点差/0.30美元;
  - ATR窗口 = 1440根M1 (≈24小时交易时间, 与M5族ATR288x5同口径);
  - 频率天花板 = 交易分钟数/天 / (平均持仓M1 + 冷却2根), 要求>=40笔/天 (30目标留余量);
  - 选择规则(预注册): 约束(天花板>=40, 超时<=40%, valid>=95%) 内最大化
    min(随机WRL, 随机WRS); 平分(±0.5pp)取PLR高者。
"""
import sys, os, json, itertools
sys.path.insert(0, "/home/z/my-project/download/xauusd_ml_v2")
import numpy as np
import pandas as pd
from config import CFG
from data import load_raw_m1, impute_spread
from labeling import label_all

OUT = "/home/z/my-project/scripts/calib_m1_result.json"
CSV = CFG["data_path"]

COOLDOWN = 2
TP_FLOOR_MULT = 3.0        # TP >= 3×点差成本
ATR_WIN = 1440             # 24h M1
CEILING_MIN = 40.0
TIMEOUT_MAX = 40.0
VALID_MIN = 95.0


def atr_m1(m1):
    h, l, c = m1["HIGH"], m1["LOW"], m1["CLOSE"]
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(ATR_WIN, min_periods=ATR_WIN).mean()


def geometry_stats(m1_pack, bars_t, atr_arr, spread_cost, rows,
                   tp_mult, sl_mult, horizon):
    m1_t, m1_o, m1_h, m1_l, m1_c = m1_pack
    res = label_all(m1_t, m1_o, m1_h, m1_l, m1_c, bars_t[rows], atr_arr[rows],
                    spread_cost[rows], tp_mult, sl_mult, CFG["sl_floor_usd"],
                    CFG["spread_floor_mult"], horizon, 10, 1, TP_FLOOR_MULT)
    entry_idx, out_long, out_short, pnl_long, pnl_short, ebl, ebs, tp_d, sl_d = res
    ok = entry_idx >= 0
    pl, ps = pnl_long[ok], pnl_short[ok]
    pooled = np.concatenate([pl, ps])
    w, l = pooled[pooled > 0], pooled[pooled < 0]
    plr = w.mean() / abs(l.mean()) if len(w) and len(l) else np.nan
    wr_l = float((pl > 0).mean()) if len(pl) else np.nan
    wr_s = float((ps > 0).mean()) if len(ps) else np.nan
    dur = (np.maximum(np.where(ebl >= 0, ebl, entry_idx),
                      np.where(ebs >= 0, ebs, entry_idx)) - entry_idx + 1)[ok]
    mean_dur = float(dur.mean())
    return {
        "tp_mult": tp_mult, "sl_mult": sl_mult, "horizon": horizon,
        "geo_ratio": round(tp_mult / sl_mult, 3),
        "valid%": round(100 * ok.sum() / len(rows), 1),
        "TP%_L": round(100 * (out_long[ok] == 1).mean(), 1),
        "TP%_S": round(100 * (out_short[ok] == 1).mean(), 1),
        "WRrand%_L": round(100 * wr_l, 1), "WRrand%_S": round(100 * wr_s, 1),
        "timeout%": round(100 * (out_long[ok] == 0).mean(), 1),
        "EV_L$": round(float(pl.mean()), 3), "EV_S$": round(float(ps.mean()), 3),
        "PLR": round(float(plr), 3),
        "avgW$": round(float(w.mean()), 3), "avgL$": round(float(abs(l.mean())), 3),
        "dur_med": int(np.median(dur)), "dur_mean": round(mean_dur, 1),
        "freq_ceiling": None,   # 填充于main(需全局交易分钟/天)
    }


def main():
    m1 = load_raw_m1(CSV)
    spread_cost, monthly = impute_spread(m1, CFG["point_value"])
    atr = atr_m1(m1)
    atr_arr = atr.to_numpy(np.float64)
    m1_t = (m1.index.astype("int64") // 10**9 // 60).to_numpy(np.int64)
    m1_pack = (m1_t, m1["OPEN"].to_numpy(np.float64), m1["HIGH"].to_numpy(np.float64),
               m1["LOW"].to_numpy(np.float64), m1["CLOSE"].to_numpy(np.float64))

    months = m1.index.to_period("M")
    win = np.asarray(months >= pd.Period("2022-01")) & np.asarray(months <= pd.Period("2024-07"))
    purge = 90 + 6     # 最大horizon + embargo (M1根)
    last_row = np.where(win)[0][-1] - purge
    rows_train = np.where(win & np.isfinite(atr_arr))[0]
    rows_train = rows_train[rows_train <= last_row]
    n_days = m1.index.normalize().nunique()
    trade_min_per_day = len(m1) / n_days
    print(f"训练窗 2022-01~2024-07(右端purge {purge}根M1): {len(rows_train):,} 根M1 "
          f"| ATR(1440)中位 ${np.median(atr_arr[rows_train]):.3f} "
          f"| 点差中位 ${np.median(spread_cost[rows_train]):.3f} "
          f"| 交易分钟/天 {trade_min_per_day:.0f}")

    grid = list(itertools.product([0.4, 0.6, 0.8, 1.0, 1.3],   # tp_mult
                                  [1.5, 2.0, 3.0, 4.0],        # sl_mult
                                  [30, 45, 60, 90]))           # horizon M1
    rows_fmt = []
    for tp_m, sl_m, hz in grid:
        st = geometry_stats(m1_pack, m1_t, atr_arr, spread_cost, rows_train, tp_m, sl_m, hz)
        st["freq_ceiling"] = round(trade_min_per_day / (st["dur_mean"] + COOLDOWN), 1)
        rows_fmt.append(st)
        print(f"tp{tp_m:4.1f} sl{sl_m:4.1f} hz{hz:3d} | WRL{st['WRrand%_L']:5.1f} "
              f"WRS{st['WRrand%_S']:5.1f} min{min(st['WRrand%_L'], st['WRrand%_S']):5.1f} "
              f"PLR{st['PLR']:5.2f} TO{st['timeout%']:5.1f}% dur{st['dur_med']:3d} "
              f"ceil{st['freq_ceiling']:5.1f} avgW{st['avgW$']:5.2f} avgL{st['avgL$']:6.2f} "
              f"EV{st['EV_L$']:+.2f}/{st['EV_S$']:+.2f}")

    meet = [c for c in rows_fmt if c["freq_ceiling"] >= CEILING_MIN
            and c["timeout%"] <= TIMEOUT_MAX and c["valid%"] >= VALID_MIN]
    if meet:
        best_wr = max(min(c["WRrand%_L"], c["WRrand%_S"]) for c in meet)
        top = [c for c in meet if min(c["WRrand%_L"], c["WRrand%_S"]) >= best_wr - 0.5]
        sel = max(top, key=lambda c: (c["PLR"], -c["horizon"]))
    else:
        sel = None
    out = {"window": "2022-01~2024-07 (purged)", "bar_minutes": 1,
           "atr_window_bars": ATR_WIN, "cooldown_m1": COOLDOWN,
           "tp_floor_spread_mult": TP_FLOOR_MULT, "grid": rows_fmt, "selected": sel}
    if sel:
        print(f"\n[选定] tp{sel['tp_mult']} sl{sel['sl_mult']} hz{sel['horizon']} "
              f"WRL{sel['WRrand%_L']} WRS{sel['WRrand%_S']} PLR{sel['PLR']} "
              f"TO{sel['timeout%']}% ceil{sel['freq_ceiling']} dur_med{sel['dur_med']} "
              f"EV{sel['EV_L$']}/{sel['EV_S$']}")
    else:
        print("\n[警告] 无满足约束的几何点!")
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"已保存 -> {OUT}")


if __name__ == "__main__":
    main()
