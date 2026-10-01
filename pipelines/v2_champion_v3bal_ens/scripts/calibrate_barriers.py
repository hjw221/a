"""
一次性标定脚本: 用【第一个训练窗 2022-01 ~ 2024-07】的真实数据选择ATR障碍乘数。
(标定只用首折训练窗 = 无未来信息; 2026年数据仅打印诊断, 不参与选择)
候选保持原代码 3.5:2.0 = 1.75 的盈亏比。
"""
import sys, os
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG_ROOT)
import numpy as np
import pandas as pd
from data import load_data
from features import build_features_v2
from labeling import label_all
from config import CFG


def hit_stats(m1_pack, m5, m5_t, atr_arr, spread_cost, rows, tp_mult, sl_mult, horizon=120):
    m1_t, m1_o, m1_h, m1_l, m1_c = m1_pack
    r = rows
    res = label_all(m1_t, m1_o, m1_h, m1_l, m1_c, m5_t[r], atr_arr[r], spread_cost[r],
                    tp_mult, sl_mult, CFG["sl_floor_usd"], CFG["spread_floor_mult"], horizon, 10)
    entry_idx, out_long, out_short, pnl_long, pnl_short, ebl, ebs, tp_d, sl_d = res
    ok = entry_idx >= 0
    n = ok.sum()
    dur = (np.where(ebl >= 0, ebl, entry_idx) - entry_idx + 1)[ok]
    return {
        "valid%": 100 * n / len(r),
        "long_TP%": 100 * (out_long[ok] == 1).mean(),
        "long_SL%": 100 * (out_long[ok] == -1).mean(),
        "short_TP%": 100 * (out_short[ok] == 1).mean(),
        "timeout%": 100 * (out_long[ok] == 0).mean(),
        "TP$med": np.median(tp_d[ok]),
        "SL$med": np.median(sl_d[ok]),
        "dur_med": np.median(dur),
        "EV_long$": np.mean(pnl_long[ok]),
        "EV_short$": np.mean(pnl_short[ok]),
    }


def main():
    m5, m1_pack, spread_cost, monthly = load_data(CFG)
    print("\n[1] 构建ATR ...")
    F, feats, atr = build_features_v2(m5, CFG["atr_window_m5"])
    atr_arr = atr.to_numpy(np.float64)
    m5_t = (m5.index.astype("int64") // 10**9 // 60).to_numpy(np.int64)
    months = m5.index.to_period("M")

    # 第一个训练窗 (与走查首折一致)
    win = np.asarray(months >= pd.Period("2022-01")) & np.asarray(months <= pd.Period("2024-07"))
    rows_train = np.where(win & np.isfinite(atr_arr))[0]
    # 2026诊断样本 (不参与选择)
    win26 = np.asarray(months >= pd.Period("2026-01")) & np.asarray(months <= pd.Period("2026-06"))
    rows_26 = np.where(win26 & np.isfinite(atr_arr))[0]

    print(f"\n训练窗行数: {len(rows_train):,} | ATR中位数(训练窗): ${np.median(atr_arr[rows_train]):.2f} | 2026 ATR中位数: ${np.median(atr_arr[rows_26]):.2f}")
    print(f"训练窗点差成本中位数: ${np.median(spread_cost[rows_train]):.3f} | 2026: ${np.median(spread_cost[rows_26]):.3f} (point={CFG['point_value']})")

    cands = [(1.5, 0.857), (2.0, 1.143), (2.5, 1.429), (2.9, 1.657), (3.5, 2.0)]

    print("\n===== 候选ATR乘数 (训练窗 2022-01~2024-07, 真实数据) =====")
    rows_fmt = []
    for tp_m, sl_m in cands:
        s = hit_stats(m1_pack, m5, m5_t, atr_arr, spread_cost, rows_train, tp_m, sl_m)
        rows_fmt.append({"tp_mult": tp_m, "sl_mult": sl_m, **s})
    df = pd.DataFrame(rows_fmt)
    print(df.round(2).to_string(index=False))

    print("\n===== 对照: 原固定美元 TP=3.5 / SL=2.0 =====")
    # 固定美元 = 用ATR的常数倍不可行, 直接改用美元障碍的等价实现: 传入 atr=全1, mult=美元数
    m1_t, m1_o, m1_h, m1_l, m1_c = m1_pack
    ones = np.ones(len(m5_t))
    res = label_all(m1_t, m1_o, m1_h, m1_l, m1_c, m5_t[rows_train], ones[rows_train],
                    spread_cost[rows_train], 3.5, 2.0, CFG["sl_floor_usd"],
                    CFG["spread_floor_mult"], 120, 10)
    entry_idx, out_long, out_short, pnl_long, pnl_short, *_ = res
    ok = entry_idx >= 0
    print(f"训练窗: long_TP={100*(out_long[ok]==1).mean():.1f}%  short_TP={100*(out_short[ok]==1).mean():.1f}%  "
          f"timeout={100*(out_long[ok]==0).mean():.1f}%  EV_long=${np.mean(pnl_long[ok]):.3f}  EV_short=${np.mean(pnl_short[ok]):.3f}")

    res26 = label_all(m1_t, m1_o, m1_h, m1_l, m1_c, m5_t[rows_26], ones[rows_26],
                      spread_cost[rows_26], 3.5, 2.0, CFG["sl_floor_usd"],
                      CFG["spread_floor_mult"], 120, 10)
    entry_idx, out_long, out_short, pnl_long, pnl_short, *_ = res26
    ok = entry_idx >= 0
    print(f"2026年:  long_TP={100*(out_long[ok]==1).mean():.1f}%  short_TP={100*(out_short[ok]==1).mean():.1f}%  "
          f"timeout={100*(out_long[ok]==0).mean():.1f}%  EV_long=${np.mean(pnl_long[ok]):.3f}  EV_short=${np.mean(pnl_short[ok]):.3f}")
    print("(2026数据仅作诊断: 展示固定美元障碍在高波动 regime 的退化, 未参与乘数选择)")

    print("\n===== 点值敏感性 (若point=0.01, 2026点差成本) =====")
    print(f"point=0.001: ${np.median(spread_cost[rows_26]):.3f}/笔   point=0.01: ${np.median(spread_cost[rows_26])*10:.3f}/笔")


if __name__ == "__main__":
    main()
