"""
频率友好几何标定 (v5) — v3bal拓展。
背景: 用户要求 M5 口径 >=10笔/天 且 胜率/盈亏比都要高。
  - v3bal瓶颈诊断(真实OOS日志): 中位持仓30分钟+冷却10分钟 -> 周转上限~30笔/天,
    实际2.6笔/天的约束在信号稀疏(阈值q~0.9) -> 频率靠阈值目标解决(管线侧);
    胜率/盈亏比靠几何解决(本脚本)。
  - 几何物理: 随机行走下 P(TP先到)=SL/(TP+SL), PLR≈TP/SL, 故
    随机TP率36% ↔ 几何比1.78 ↔ PLR~1.75; 随机TP率33% ↔ 比2.03 ↔ PLR~2.0。
    模型实测加成+2.5~6pp(含阈值选择效应) -> 实现36~40%胜率的两个操作点。
  - 与v3/v4标定完全同一规则: 只用首训练窗 2022-01~2024-07 真实数据(无未来信息),
    随机入场口径已含点差与同K线SL悲观规则。2026年数据不参与选择。
两个操作点(都在频率可行区内, 周转上限>=13笔/天):
  freq  (v3freq):  随机TP率 L>=34% S>=32% -> 目标实现胜率 ~39-40%
  freq2 (v3freq2): 随机TP率 L>=31% S>=29% -> 目标实现胜率 ~36%, PLR更高
"""
import sys, os, json, itertools
sys.path.insert(0, "/home/z/my-project/download/xauusd_ml_v2")
import numpy as np
import pandas as pd
from data import load_data
from features import build_features_v2
from labeling import label_all
from config import CFG

OUT = "/home/z/my-project/scripts/calib_v5_result.json"

TP_FLOOR_A = (34.0, 32.0)   # freq:  胜率~40%操作点
TP_FLOOR_B = (30.0, 29.0)   # freq2: 胜率~35%操作点
EV_FLOOR = -0.22            # 与v3/v4标定同一口径(随机入场EV含点差)
TIMEOUT_CAP = 12.0          # 超时占比上限
TPD_CAP = 12.0              # 均值口径周转上限(笔/天)下限约束
M1_PER_DAY = 1380           # 交易日约23小时


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
    dur_med = float(np.median(dur))
    dur_mean = float(dur.mean())
    tpd_cap = M1_PER_DAY / (dur_mean + 11)    # 均值口径: 长timeout尾占用
    return {
        "tp_mult": tp_mult, "sl_mult": sl_mult, "horizon": horizon,
        "geo_ratio": round(tp_mult / sl_mult, 3),
        "TP%_L": round(100 * (out_long[ok] == 1).mean(), 1),
        "TP%_S": round(100 * (out_short[ok] == 1).mean(), 1),
        "timeout%": round(100 * (out_long[ok] == 0).mean(), 1),
        "EV_L$": round(float(pl.mean()), 3), "EV_S$": round(float(ps.mean()), 3),
        "PLR": round(float(plr), 3), "WR_rand": round(100 * wr, 1),
        "avgW$": round(float(w.mean()), 3), "avgL$": round(float(abs(l.mean())), 3),
        "dur_med": int(dur_med), "dur_mean": round(dur_mean, 1),
        "tpd_cap": round(tpd_cap, 1),
    }


def pick(df, floor_l, floor_s, tag):
    cand = df[(~df["is_ref"].astype(bool))
              & (df["TP%_L"] >= floor_l) & (df["TP%_S"] >= floor_s)
              & (df["timeout%"] <= TIMEOUT_CAP)
              & (df["tpd_cap"] >= TPD_CAP)
              & (df["EV_L$"] >= EV_FLOOR) & (df["EV_S$"] >= EV_FLOOR)]
    print(f"\n===== [{tag}] 通过约束(TP%_L>={floor_l}, TP%_S>={floor_s}, 超时<={TIMEOUT_CAP}%, "
          f"周转>={TPD_CAP}笔/天, EV>= {EV_FLOOR}): {len(cand)}个 =====")
    if len(cand) == 0:
        return None
    cand = cand.sort_values(["PLR", "dur_med"], ascending=[False, True])
    print(cand.to_string(index=False))
    return cand.iloc[0]


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

    # 网格: 胜率友好区(随机TP率29~38%) × horizons{120,180,360}
    # (360在胜率友好几何下超时占比<6%, 且PLR更高; 周转由均值口径tpd_cap约束把关)
    grid = list(itertools.product([1.8, 2.0, 2.2, 2.4, 2.6, 2.8],   # tp_mult
                                  [1.1429, 1.3],                    # sl_mult
                                  [120, 180, 360]))                 # horizon
    refs = [(2.0, 1.1429, 120), (3.0, 1.1429, 360)]                 # 默认/bal参考

    rows_fmt = []
    for tp_m, sl_m, hz in grid + refs:
        s = geometry_stats(m1_pack, m5_t, atr_arr, spread_cost, rows_train, tp_m, sl_m, hz)
        s["is_ref"] = (tp_m, sl_m, hz) in refs
        rows_fmt.append(s)
    df = pd.DataFrame(rows_fmt)
    pd.set_option("display.width", 240)
    print("\n===== 候选几何 (训练窗真实数据, 随机入场口径) =====")
    print(df.to_string(index=False))

    best_a = pick(df, *TP_FLOOR_A, "freq/v3freq 目标胜率~40%")
    best_b = pick(df, *TP_FLOOR_B, "freq2/v3freq2 目标胜率~36%")
    for name, best, floor in [("freq", best_a, TP_FLOOR_A), ("freq2", best_b, TP_FLOOR_B)]:
        if best is None:
            print(f">>> [{name}] 无候选通过约束!")
            continue
        print(f">>> [{name}] 选定: TP={best['tp_mult']}xATR SL={best['sl_mult']}xATR "
              f"horizon={int(best['horizon'])}M1 | 几何比 {best['geo_ratio']} "
              f"随机PLR {best['PLR']} 随机TP率L/S {best['TP%_L']}/{best['TP%_S']}% "
              f"超时 {best['timeout%']}% EV L/S ${best['EV_L$']}/${best['EV_S$']} "
              f"中位持仓 {best['dur_med']}M1 周转上限 {best['tpd_cap']}笔/天")

    def _conv(v):
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, float, np.floating, np.integer)):
            return float(v)
        return v
    json.dump({
        "freq": None if best_a is None else {k: _conv(v) for k, v in best_a.items()},
        "freq2": None if best_b is None else {k: _conv(v) for k, v in best_b.items()},
        "tp_floor_freq": TP_FLOOR_A, "tp_floor_freq2": TP_FLOOR_B,
        "ev_floor": EV_FLOOR, "timeout_cap": TIMEOUT_CAP, "tpd_cap": TPD_CAP,
        "all": json.loads(df.to_json(orient="records")),
    }, open(OUT, "w"), indent=2)
    print(f"完整表 -> {OUT}")


if __name__ == "__main__":
    main()
