"""
v7 几何标定 (5分K线): 用户修订目标 = 胜率>=40%(硬), 盈亏比越高越好, >=10笔/天。
v5/v6/v6ad 已失败的真实教训 (OOS复盘):
  - horizon 240/360 分钟时, 所有特征集的 OOS AUC 都降到 ~0.50 (无预测边际),
    实测胜率加成(lift)只有 -1.1~+0.9pp;
  - 唯一有真实 lift 的是 legacy 特征 + horizon 120 (2h): lift +6.2pp, AUC 0.52/0.55。
  => v7 回到 horizon 120 家族, 用几何把随机基率抬到 ~37-38% (v7a) / ~41% (v7b),
     靠 legacy 特征的选择性把实测胜率推过 40%。
规则与前几轮标定一致: 只用首训练窗 2022-01~2024-07 真实数据 (右端按horizon+embargo截断,
比v5标定更严格), 2024-08之后的数据绝不参与。随机口径已含点差与同K线SL悲观规则。
选择规则(预注册): 约束 随机WR(多)>=floorL, (空)>=floorS, 超时<=12%, 冷却5下天花板>=20笔/天
  -> 最大化随机PLR; PLR差<0.05时取horizon更短者(保AUC边际), 再平取天花板更高者。
两个预注册候选:
  v7a: floorL=0.38 floorS=0.36 (赌legacy特征lift +4~6pp -> 实测42~44%, PLR更高)
  v7b: floorL=0.41 floorS=0.39 (保底: 即使lift归零实测也>=41%, PLR较低)
"""
import sys, os, json, itertools
sys.path.insert(0, "/home/z/my-project/download/xauusd_ml_v2")
import numpy as np
import pandas as pd
from data import load_data
from features import build_features_v2
from labeling import label_all
from config import CFG

OUT = "/home/z/my-project/scripts/calib_v7_result.json"

COOLDOWN = 5          # v7冷却(提频: 10->5根M1)
M5_PER_DAY = 271      # 实测 321,636根M5 / (54月x22天)
TRADE_MIN_PER_DAY = M5_PER_DAY * 5
FLOORS = {"v7a": (0.38, 0.36), "v7b": (0.41, 0.39)}
TIMEOUT_MAX = 12.0    # 超时占比上限(%)
CEILING_MIN = 20.0    # 频率天花板下限(笔/天)


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
    wr_l = float((pl > 0).mean()) if len(pl) else np.nan
    wr_s = float((ps > 0).mean()) if len(ps) else np.nan
    dur_l = (np.where(ebl >= 0, ebl, entry_idx) - entry_idx + 1)[ok]
    dur_s = (np.where(ebs >= 0, ebs, entry_idx) - entry_idx + 1)[ok]
    mean_dur = float(np.concatenate([dur_l, dur_s]).mean())
    ceiling = TRADE_MIN_PER_DAY / (mean_dur + COOLDOWN)
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
        "dur_med": int(np.median(np.concatenate([dur_l, dur_s]))),
        "dur_mean": round(mean_dur, 1),
        "freq_ceiling": round(ceiling, 1),
    }


def pick(cands, floorL, floorS):
    meet = [c for c in cands
            if c["WRrand%_L"] >= 100 * floorL and c["WRrand%_S"] >= 100 * floorS
            and c["timeout%"] <= TIMEOUT_MAX and c["freq_ceiling"] >= CEILING_MIN]
    if not meet:
        return None
    # 预注册规则: PLR最大; 差<0.05取horizon短; 再平取ceiling高
    best = sorted(meet, key=lambda c: (-c["PLR"], c["horizon"], -c["freq_ceiling"]))
    top = [c for c in meet if c["PLR"] >= best[0]["PLR"] - 0.05]
    return sorted(top, key=lambda c: (c["horizon"], -c["freq_ceiling"]))[0]


def main():
    m5, m1_pack, spread_cost, monthly = load_data(CFG)
    F, feats, atr = build_features_v2(m5, CFG["atr_window_m5"])   # 只为拿ATR
    atr_arr = atr.to_numpy(np.float64)
    m5_t = (m5.index.astype("int64") // 10**9 // 60).to_numpy(np.int64)
    months = m5.index.to_period("M")
    win = np.asarray(months >= pd.Period("2022-01")) & np.asarray(months <= pd.Period("2024-07"))
    # 右端purge: horizon 240 + 6根embargo = 54根M5, 严格不碰OOS
    purge = int(np.ceil(240 / 5)) + 6
    last_row = np.where(win)[0][-1] - purge
    rows_train = np.where(win & np.isfinite(atr_arr))[0]
    rows_train = rows_train[rows_train <= last_row]
    print(f"训练窗 2022-01~2024-07(右端purge {purge}根M5): {len(rows_train):,} 根M5 "
          f"| ATR中位 ${np.median(atr_arr[rows_train]):.2f} "
          f"| 点差中位 ${np.median(spread_cost[rows_train]):.3f}")

    grid = list(itertools.product([1.5, 1.7, 1.9, 2.1, 2.3],   # tp_mult
                                  [1.0, 1.1429, 1.3, 1.5, 1.75],  # sl_mult
                                  [120, 240]))              # horizon(只保留有AUC边际的短horizon)
    refs = [(2.0, 1.1429, 120), (1.8, 1.0, 360), (1.6, 1.0, 240)]  # 参考: 默认/v5/v6几何

    rows_fmt = []
    for tp_m, sl_m, hz in grid + refs:
        st = geometry_stats(m1_pack, m5_t, atr_arr, spread_cost, rows_train, tp_m, sl_m, hz)
        rows_fmt.append(st)
        print(f"tp{tp_m:4.1f} sl{sl_m:6.4f} hz{hz:3d} | ratio{st['geo_ratio']:5.2f} "
              f"WRL{st['WRrand%_L']:5.1f} WRS{st['WRrand%_S']:5.1f} PLR{st['PLR']:5.2f} "
              f"TO{st['timeout%']:5.1f}% dur{st['dur_med']:4d} ceil{st['freq_ceiling']:5.1f} "
              f"EV{st['EV_L$']:+.2f}/{st['EV_S$']:+.2f}")

    out = {"window": "2022-01~2024-07 (purged)", "cooldown_m1": COOLDOWN, "grid": rows_fmt}
    for name, (fl, fs) in FLOORS.items():
        sel = pick(rows_fmt, fl, fs)
        out[name] = sel
        print(f"\n[{name}] floor L{fl:.2f}/S{fs:.2f} -> "
              + (f"tp{sel['tp_mult']} sl{sel['sl_mult']} hz{sel['horizon']} "
                 f"PLR{sel['PLR']} WRL{sel['WRrand%_L']} WRS{sel['WRrand%_S']} "
                 f"ceil{sel['freq_ceiling']}" if sel else "无可行点!"))
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\n已保存 -> {OUT}")


if __name__ == "__main__":
    main()
