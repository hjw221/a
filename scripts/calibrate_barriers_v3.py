"""
盈亏比导向的障碍几何标定 (v3)。
严格规则: 只用【第一个训练窗 2022-01 ~ 2024-07】的真实数据选择几何。
  - 三重障碍下, 已实现盈亏比 ≈ (TP-c)/(SL+c), 由几何决定上限;
    特征工程/阈值选择的任务是让胜率在该几何下站得住。
  - 目标: 最大化随机入场口径的多空合并盈亏比 PLR = avg(win)/|avg(loss)|;
    约束: 每侧TP率>=10% (模型有东西可学), 超时<=35% (不稀释PLR), 每侧EV>=-$0.15 (成本不至于吞掉模型优势)。
  - 2026年数据不参与选择。
"""
import sys, os, json, itertools
sys.path.insert(0, "/home/z/my-project/download/xauusd_ml_v2")
import numpy as np
import pandas as pd
from data import load_data
from features import build_features_v2
from labeling import label_all
from config import CFG

OUT = "/home/z/my-project/scripts/calib_v3_result.json"


def geometry_stats(m1_pack, m5_t, atr_arr, spread_cost, rows,
                   tp_mult, sl_mult, horizon):
    m1_t, m1_o, m1_h, m1_l, m1_c = m1_pack
    res = label_all(m1_t, m1_o, m1_h, m1_l, m1_c, m5_t[rows], atr_arr[rows],
                    spread_cost[rows], tp_mult, sl_mult, CFG["sl_floor_usd"],
                    CFG["spread_floor_mult"], horizon, 10)
    entry_idx, out_long, out_short, pnl_long, pnl_short, ebl, ebs, tp_d, sl_d = res
    ok = entry_idx >= 0
    pl, ps = pnl_long[ok], pnl_short[ok]
    pooled = np.concatenate([pl, ps])
    w, l = pooled[pooled > 0], pooled[pooled < 0]
    plr = w.mean() / abs(l.mean()) if len(w) and len(l) else np.nan
    dur = (np.where(ebl >= 0, ebl, entry_idx) - entry_idx + 1)[ok]
    return {
        "tp_mult": tp_mult, "sl_mult": sl_mult, "horizon": horizon,
        "geo_ratio": round(tp_mult / sl_mult, 3),
        "valid%": round(100 * ok.sum() / len(rows), 1),
        "TP%_L": round(100 * (out_long[ok] == 1).mean(), 1),
        "TP%_S": round(100 * (out_short[ok] == 1).mean(), 1),
        "timeout%": round(100 * (out_long[ok] == 0).mean(), 1),
        "EV_L$": round(float(pl.mean()), 3), "EV_S$": round(float(ps.mean()), 3),
        "PLR": round(float(plr), 3),
        "avgW$": round(float(w.mean()), 3), "avgL$": round(float(abs(l.mean())), 3),
        "dur_med": int(np.median(dur)),
    }


def main():
    m5, m1_pack, spread_cost, monthly = load_data(CFG)
    F, feats, atr = build_features_v2(m5, CFG["atr_window_m5"])   # 只为拿ATR
    atr_arr = atr.to_numpy(np.float64)
    m5_t = (m5.index.astype("int64") // 10**9 // 60).to_numpy(np.int64)
    months = m5.index.to_period("M")
    win = np.asarray(months >= pd.Period("2022-01")) & np.asarray(months <= pd.Period("2024-07"))
    rows_train = np.where(win & np.isfinite(atr_arr))[0]
    print(f"训练窗 2022-01~2024-07: {len(rows_train):,} 根M5 | ATR中位 ${np.median(atr_arr[rows_train]):.2f}"
          f" | 点差中位 ${np.median(spread_cost[rows_train]):.3f}")

    grid = list(itertools.product([2.4, 2.8, 3.2, 3.6, 4.0],   # tp_mult
                                  [0.8, 1.0, 1.1429],          # sl_mult
                                  [120, 240, 360]))            # horizon (M1)
    rows_fmt = []
    for tp_m, sl_m, hz in grid:
        s = geometry_stats(m1_pack, m5_t, atr_arr, spread_cost, rows_train, tp_m, sl_m, hz)
        rows_fmt.append(s)
    df = pd.DataFrame(rows_fmt)
    pd.set_option("display.width", 200)
    print("\n===== 候选几何 (训练窗真实数据, 随机入场口径) =====")
    print(df.to_string(index=False))

    # 约束: TP率>=15%(标签可学习), 超时<=35%, EV>=-0.22(与旧几何OOS随机亏损 -$0.11~-$0.19/笔 同量级, 模型的任务就是翻越)
    ok = df[(df["TP%_L"] >= 15) & (df["TP%_S"] >= 15) & (df["timeout%"] <= 35)
            & (df["EV_L$"] >= -0.22) & (df["EV_S$"] >= -0.22)]
    print(f"\n通过约束的候选: {len(ok)} / {len(df)}")
    ok = ok.sort_values("PLR", ascending=False)
    print(ok.to_string(index=False))
    if len(ok) == 0:
        print("无候选通过约束!")
        return
    best = ok.iloc[0]
    print(f"\n>>> 选定: TP={best['tp_mult']}xATR  SL={best['sl_mult']}xATR  horizon={int(best['horizon'])}M1 "
          f"(几何比 {best['geo_ratio']}, 训练窗PLR {best['PLR']}, TP率L/S {best['TP%_L']}/{best['TP%_S']}%, "
          f"超时 {best['timeout%']}%, EV L/S ${best['EV_L$']}/${best['EV_S$']})")
    json.dump({"best": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                        for k, v in best.items()},
               "all": df.to_dict(orient="records")}, open(OUT, "w"), indent=2)
    print(f"完整表 -> {OUT}")


if __name__ == "__main__":
    main()
