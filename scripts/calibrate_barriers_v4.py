"""
胜率下限导向的障碍几何标定 (v4)。
背景: v3aggr(TP=4.0/SL=0.8/6h)真实OOS胜率18.4%, 用户反馈太低。
  - 低胜率是宽TP/窄SL几何的机械结果, 不是模型缺陷;
  - 本脚本在同一首训练窗(2022-01~2024-07, 无未来信息)重标几何:
    约束 随机入场TP率(=胜率地基) >= 下限, 在此之内最大化盈亏比。
  - 依据(首版走查实测): 模型在几何TP率上再加 +2.5~+6pp 成为最终胜率,
    故随机口径TP率>=28% 大致对应实现胜率 31~34%。
规则与v3标定完全一致, 仅约束不同。2026年数据不参与选择。
"""
import sys, os, json, itertools
sys.path.insert(0, "/home/z/my-project/download/xauusd_ml_v2")
import numpy as np
import pandas as pd
from data import load_data
from features import build_features_v2
from labeling import label_all
from config import CFG

OUT = "/home/z/my-project/scripts/calib_v4_result.json"

# 胜率下限: 随机入场TP率(多头/空头)。目标实现胜率 >= ~31%
TP_FLOOR_L, TP_FLOOR_S = 28.0, 26.0


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
    wr = float(len(w) / len(pooled)) if len(pooled) else np.nan
    dur = (np.where(ebl >= 0, ebl, entry_idx) - entry_idx + 1)[ok]
    return {
        "tp_mult": tp_mult, "sl_mult": sl_mult, "horizon": horizon,
        "geo_ratio": round(tp_mult / sl_mult, 3),
        "valid%": round(100 * ok.sum() / len(rows), 1),
        "TP%_L": round(100 * (out_long[ok] == 1).mean(), 1),
        "TP%_S": round(100 * (out_short[ok] == 1).mean(), 1),
        "timeout%": round(100 * (out_long[ok] == 0).mean(), 1),
        "EV_L$": round(float(pl.mean()), 3), "EV_S$": round(float(ps.mean()), 3),
        "PLR": round(float(plr), 3), "WR_rand": round(100 * wr, 1),
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

    # 平衡几何网格: 覆盖 TP率 26~33% 的区域, sl含0.9填补空隙
    grid = list(itertools.product([2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2],   # tp_mult
                                  [0.9, 1.0, 1.1429],                    # sl_mult
                                  [240, 360]))                           # horizon
    # 参考点(不参与选择, 仅frontier展示): 旧几何/主PLR几何/激进几何
    refs = [(2.0, 1.1429, 120), (4.0, 1.0, 360), (4.0, 0.8, 360)]

    rows_fmt = []
    for tp_m, sl_m, hz in grid + refs:
        s = geometry_stats(m1_pack, m5_t, atr_arr, spread_cost, rows_train, tp_m, sl_m, hz)
        s["is_ref"] = (tp_m, sl_m, hz) in refs
        rows_fmt.append(s)
    df = pd.DataFrame(rows_fmt)
    pd.set_option("display.width", 220)
    print("\n===== 候选几何 (训练窗真实数据, 随机入场口径) =====")
    print(df.to_string(index=False))

    # 约束: TP率下限(胜率地基) + 可学习/成本约束(与v3标定一致)
    cand = df[(~df["is_ref"].astype(bool))
              & (df["TP%_L"] >= TP_FLOOR_L) & (df["TP%_S"] >= TP_FLOOR_S)
              & (df["timeout%"] <= 35)
              & (df["EV_L$"] >= -0.22) & (df["EV_S$"] >= -0.22)]
    print(f"\n通过胜率下限约束(TP%_L>={TP_FLOOR_L}, TP%_S>={TP_FLOOR_S})的候选: {len(cand)} / {len(grid)}")
    cand = cand.sort_values("PLR", ascending=False)
    print(cand.to_string(index=False))
    if len(cand) == 0:
        print("无候选通过约束!")
        return
    best = cand.iloc[0]
    print(f"\n>>> 选定平衡几何: TP={best['tp_mult']}xATR  SL={best['sl_mult']}xATR  horizon={int(best['horizon'])}M1 "
          f"(几何比 {best['geo_ratio']}, 训练窗随机PLR {best['PLR']}, 随机TP率L/S {best['TP%_L']}/{best['TP%_S']}%, "
          f"超时 {best['timeout%']}%, EV L/S ${best['EV_L$']}/${best['EV_S$']})")
    def _conv(v):
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, float, np.floating, np.integer)):
            return float(v)
        return v
    json.dump({"best": {k: _conv(v) for k, v in best.items()},
               "tp_floor_l": TP_FLOOR_L, "tp_floor_s": TP_FLOOR_S,
               "all": json.loads(df.to_json(orient="records"))},
              open(OUT, "w"), indent=2)
    print(f"完整表 -> {OUT}")


if __name__ == "__main__":
    main()
