"""
1分钟线几何标定 (m1) — 次任务: 1分k线 >=30笔/天, 纯胜率口径。
规则与v3/v4/v5标定完全一致: 只用首训练窗 2022-01~2024-07 真实数据(无未来信息),
随机入场口径已含点差与同K线SL悲观规则。2026年数据不参与选择。
差异(相对M5标定):
  - 信号/标签在原生M1上: 入场=信号K线收盘后下一根M1开盘(entry_shift_min=1);
  - ATR基准 = 288根M1(4.8小时)的均真实波幅 —— 1分钟量级, 远小于M5口径的24h ATR,
    SL下限 max($0.30, 3x点差) 会频繁兜底, 属于点差主导区, 网格如实呈现;
  - 用户口径"只要求胜率" -> 目标=约束内最大化随机TP率(胜率地基), EV约束放宽到-0.15。
周转约束: 均值口径 tpd_cap >= 35笔/天 (目标30笔/天 + 余量)。
"""
import sys, os, json, itertools
sys.path.insert(0, "/home/z/my-project/download/xauusd_ml_v2")
import numpy as np
import pandas as pd
from data import load_data
from labeling import label_all
from config import CFG

OUT = "/home/z/my-project/scripts/calib_m1_result.json"
EV_FLOOR = -0.15
TPD_CAP = 35.0
M1_PER_DAY = 1380


def main():
    cfg_m1 = dict(CFG, resample_rule="1min")
    m5, m1_pack, spread_cost, monthly = load_data(cfg_m1)
    m1_t, m1_o, m1_h, m1_l, m1_c = m1_pack
    # ATR: 与features_v3同式 (TR.rolling(288).mean()), 在M1口径下=288分钟均真实波幅
    tr = pd.concat([m5["HIGH"] - m5["LOW"],
                    (m5["HIGH"] - m5["CLOSE"].shift(1)).abs(),
                    (m5["LOW"] - m5["CLOSE"].shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(288, min_periods=288).mean()
    atr_arr = atr.to_numpy(np.float64)
    m5_t = (m5.index.astype("int64") // 10**9 // 60).to_numpy(np.int64)
    months = m5.index.to_period("M")
    win = np.asarray(months >= pd.Period("2022-01")) & np.asarray(months <= pd.Period("2024-07"))
    rows_train = np.where(win & np.isfinite(atr_arr))[0]
    print(f"训练窗 2022-01~2024-07: {len(rows_train):,} 根M1 | ATR(288M1)中位 ${np.median(atr_arr[rows_train]):.3f}"
          f" | 点差中位 ${np.median(spread_cost[rows_train]):.3f} (SL下限=max($0.30, 3x点差)≈${3*np.median(spread_cost[rows_train]):.3f})")

    rows_fmt = []
    grid = list(itertools.product([0.5, 1.0, 1.5, 2.0, 3.0],   # tp_mult (x ATR_M1)
                                  [0.5, 0.8, 1.0, 1.5],        # sl_mult
                                  [30, 45, 60]))               # horizon (M1根)
    for tp_m, sl_m, hz in grid:
        res = label_all(m1_t, m1_o, m1_h, m1_l, m1_c, m5_t[rows_train], atr_arr[rows_train],
                        spread_cost[rows_train], tp_m, sl_m, CFG["sl_floor_usd"],
                        CFG["spread_floor_mult"], hz, 10, 1)
        entry_idx, out_long, out_short, pnl_long, pnl_short, ebl, ebs, tp_d, sl_d = res
        ok = entry_idx >= 0
        pl, ps = pnl_long[ok], pnl_short[ok]
        pooled = np.concatenate([pl, ps])
        w, l = pooled[pooled > 0], pooled[pooled < 0]
        plr = w.mean() / abs(l.mean()) if len(w) and len(l) else np.nan
        dur = (np.where(ebl >= 0, ebl, entry_idx) - entry_idx + 1)[ok]
        tpd_cap = M1_PER_DAY / (dur.mean() + 11)
        rows_fmt.append({
            "tp_mult": tp_m, "sl_mult": sl_m, "horizon": hz,
            "geo_ratio": round(tp_m / sl_m, 3),
            "TP%_L": round(100 * (out_long[ok] == 1).mean(), 1),
            "TP%_S": round(100 * (out_short[ok] == 1).mean(), 1),
            "timeout%": round(100 * (out_long[ok] == 0).mean(), 1),
            "EV_L$": round(float(pl.mean()), 3), "EV_S$": round(float(ps.mean()), 3),
            "PLR": round(float(plr), 3) if np.isfinite(plr) else None,
            "WR_rand": round(100 * len(w) / len(pooled), 1),
            "avgW$": round(float(w.mean()), 3) if len(w) else None,
            "avgL$": round(float(abs(l.mean())), 3) if len(l) else None,
            "dur_med": int(np.median(dur)), "dur_mean": round(float(dur.mean()), 1),
            "tpd_cap": round(float(tpd_cap), 1),
            "sl_floor_bind%": round(100 * (sl_d[ok] >= 3 * spread_cost[rows_train][ok] - 1e-9).mean(), 1),
        })
        print(f"  tp={tp_m} sl={sl_m} hz={hz}: WR_rand {rows_fmt[-1]['WR_rand']}% "
              f"PLR {rows_fmt[-1]['PLR']} tpd_cap {rows_fmt[-1]['tpd_cap']} "
              f"EV L/S {rows_fmt[-1]['EV_L$']}/{rows_fmt[-1]['EV_S$']} "
              f"SL下限绑定 {rows_fmt[-1]['sl_floor_bind%']}%")
    df = pd.DataFrame(rows_fmt)
    pd.set_option("display.width", 240)
    print("\n===== 候选几何 (训练窗真实M1数据, 随机入场口径) =====")
    print(df.to_string(index=False))

    cand = df[(df["TP%_L"] >= 50) & (df["TP%_S"] >= 48) & (df["tpd_cap"] >= TPD_CAP)
              & (df["EV_L$"] >= EV_FLOOR) & (df["EV_S$"] >= EV_FLOOR)]
    print(f"\n通过约束(TP%_L>=50, TP%_S>=48, 周转>={TPD_CAP}笔/天, EV>= {EV_FLOOR}): {len(cand)} 个")
    if len(cand):
        cand = cand.sort_values("WR_rand", ascending=False)
        print(cand.to_string(index=False))
        best = cand.iloc[0]
        print(f"\n>>> 选定1分钟几何: TP={best['tp_mult']}xATR_M1 SL={best['sl_mult']}xATR_M1 "
              f"horizon={int(best['horizon'])}M1 | 随机TP率L/S {best['TP%_L']}/{best['TP%_S']}% "
              f"随机PLR {best['PLR']} EV L/S ${best['EV_L$']}/${best['EV_S$']} "
              f"中位持仓 {best['dur_med']}M1 周转上限 {best['tpd_cap']}笔/天 SL下限绑定 {best['sl_floor_bind%']}%")
    else:
        best = None
        print("无候选通过约束! 放宽约束重看上表全量结果再定。")

    def _conv(v):
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, float, np.floating, np.integer)):
            return float(v)
        return v
    json.dump({"best": None if best is None else {k: _conv(v) for k, v in best.items()},
               "ev_floor": EV_FLOOR, "tpd_cap": TPD_CAP,
               "all": json.loads(df.to_json(orient="records"))},
              open(OUT, "w"), indent=2)
    print(f"完整表 -> {OUT}")


if __name__ == "__main__":
    main()
