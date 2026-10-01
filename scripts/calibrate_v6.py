"""
v6标定: 胜率40%几何 + ATR/点差regime门控。
机制: 点差成本近似常数($0.13-0.16), 障碍随ATR缩放 -> 静默regime(ATR$0.9)点差占SL 17%,
活跃regime(ATR$4.6)仅3.3%。随机入场EV因此随regime分裂 (v5诊断: 2024全q亏, 2026全q赚)。
门控 = ATR/点差 >= theta (自归一, 跨经纪商可用), 只在成本占比低的时段高频交易。
严格规则: 只用首训练窗2022-01~2024-07真实数据选theta与几何。2026年数据不参与。
几何来源: calib_v5_result.json (TP率下限39/37+频率天花板>=28内PLR最高且平分取天花板高者)。
"""
import sys, os, json
sys.path.insert(0, "/home/z/my-project/download/xauusd_ml_v2")
import numpy as np
import pandas as pd
from data import load_data
from features import build_features_v2
from labeling import label_all
from config import CFG

OUT = "/home/z/my-project/scripts/calib_v6_result.json"


def main():
    m5, m1_pack, spread_cost, monthly = load_data(CFG)
    F, feats, atr = build_features_v2(m5, CFG["atr_window_m5"])
    atr_arr = atr.to_numpy(np.float64)
    m5_t = (m5.index.astype("int64") // 10**9 // 60).to_numpy(np.int64)
    months = m5.index.to_period("M")
    win = np.asarray(months >= pd.Period("2022-01")) & np.asarray(months <= pd.Period("2024-07"))
    rows_train = np.where(win & np.isfinite(atr_arr))[0]

    ratio = atr_arr / np.maximum(spread_cost, 1e-6)   # ATR / 点差成本 (自归一成本占比倒数)

    # ---- 几何: 从v5标定结果按新约束(TP率>=39/37, 天花板>=28, PLR最高, 平分取天花板)重选 ----
    v5 = json.load(open("/home/z/my-project/scripts/calib_v5_result.json"))
    df = pd.DataFrame(v5["all"])
    cand = df[(~df["is_ref"].astype(bool)) & (df["TP%_L"] >= 39) & (df["TP%_S"] >= 37)
              & (df["freq_ceiling"] >= 28) & (df["EV_L$"] >= -0.25) & (df["EV_S$"] >= -0.25)]
    cand = cand.sort_values(["PLR", "freq_ceiling"], ascending=False)
    # PLR差距<0.03视为平分 -> 取频率天花板更高者 (用户频率约束优先)
    top = cand.iloc[0]
    near = cand[cand["PLR"] >= top["PLR"] - 0.03]
    best_geo = near.sort_values("freq_ceiling", ascending=False).iloc[0]
    print("===== 胜率40%几何重选 (约束: 随机TP率L>=39/S>=37, 天花板>=28笔/天) =====")
    print(near.to_string(index=False))
    print(f">>> 选定: TP={best_geo['tp_mult']}xATR SL={best_geo['sl_mult']}xATR hz={int(best_geo['horizon'])}M1 "
          f"(随机TP率 {best_geo['TP%_L']}/{best_geo['TP%_S']}%, PLR {best_geo['PLR']}, 天花板 {best_geo['freq_ceiling']}/天)")

    # ---- 该几何下的随机入场逐笔结果 (训练窗) ----
    res = label_all(m1_pack[0], m1_pack[1], m1_pack[2], m1_pack[3], m1_pack[4],
                    m5_t[rows_train], atr_arr[rows_train], spread_cost[rows_train],
                    best_geo["tp_mult"], best_geo["sl_mult"], CFG["sl_floor_usd"],
                    CFG["spread_floor_mult"], int(best_geo["horizon"]), 10)
    entry_idx, out_long, out_short, pnl_long, pnl_short, ebl, ebs, tp_d, sl_d = res
    ok = entry_idx >= 0
    r = ratio[rows_train][ok]
    pl, ps = pnl_long[ok], pnl_short[ok]
    pooled = np.concatenate([pl, ps])
    ol, os_ = np.concatenate([out_long[ok] == 1] * 1), np.concatenate([out_short[ok] == 1] * 1)

    print("\n===== 训练窗 ATR/点差 分桶 (随机入场口径, 双向合并) =====")
    print(f"{'ratio桶':>14} {'bar占比':>8} {'TP率':>7} {'EV$/笔':>8} {'PLR':>6}")
    qs = [0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    edges = np.quantile(r, qs)
    rows_out = []
    def side_stats(m):
        p = np.concatenate([pl[m], ps[m]])
        w, l = p[p > 0], p[p < 0]
        tp = float(np.concatenate([out_long[ok][m] == 1, out_short[ok][m] == 1]).mean())
        return p, w, l, tp
    for i in range(len(qs) - 1):
        m = (r >= edges[i]) & (r < edges[i + 1]) if i < len(qs) - 2 else (r >= edges[i])
        if m.sum() < 500:
            continue
        p, w, l, tp = side_stats(m)
        ev, plr = p.mean(), (w.mean() / abs(l.mean()) if len(w) and len(l) else np.nan)
        rows_out.append({"bucket": f"q{qs[i]}-{qs[i+1]}", "ratio_lo": float(edges[i]), "ratio_hi": float(edges[i+1]),
                         "share": float(m.mean()), "tp_rate": tp, "ev": float(ev), "plr": float(plr)})
        print(f"ratio[{edges[i]:5.1f},{edges[i+1]:5.1f}] {m.mean()*100:6.1f}% {tp*100:6.1f}% {ev:8.3f} {plr:6.2f}")

    # ---- theta选择: EV由负转正的ratio水平; 覆盖率>=25%保证频率 ----
    theta = None
    ev_out_all = pooled.mean()
    for th in [6, 7, 8, 9, 10, 12, 14, 16]:
        m = r >= th
        if m.sum() < 500:
            continue
        p, w, l, tp = side_stats(m)
        cov = float(m.mean())
        print(f"  theta={th:2d}: 覆盖{cov*100:5.1f}%  TP率{tp*100:5.1f}%  EV${p.mean():+.3f}  "
              f"PLR{(w.mean()/abs(l.mean()) if len(w) and len(l) else float('nan')):5.2f}")
    # 规则: 最小的theta使 门控内EV >= 门控外EV + 0.10 (清晰分离) 且覆盖率>=25%
    for th in [6, 7, 8, 9, 10, 12, 14, 16]:
        m = r >= th
        if m.mean() < 0.25 or m.sum() < 500:
            continue
        p, _, _, _ = side_stats(m)
        if p.mean() >= ev_out_all + 0.10:
            theta = th
            break
    if theta is None:
        theta = 8   # 兜底: 训练窗中位附近
        print(f"  [警告] 无theta达成EV分离>=0.10, 兜底theta=8 (如实报告)")
    m = r >= theta
    p_in, _, _, _ = side_stats(m)
    p_out, _, _, _ = side_stats(~m)
    print(f"\n>>> 选定门控: ATR/点差 >= {theta} (训练窗覆盖 {m.mean()*100:.1f}%, "
          f"门内EV${p_in.mean():+.3f} vs 门外${p_out.mean():+.3f})")

    json.dump({"geometry": {"tp_mult": float(best_geo["tp_mult"]), "sl_mult": float(best_geo["sl_mult"]),
                            "horizon": int(best_geo["horizon"]), "cooldown_m1": 5,
                            "plr_rand": float(best_geo["PLR"]), "tp_rate_rand": [float(best_geo["TP%_L"]), float(best_geo["TP%_S"])],
                            "freq_ceiling": float(best_geo["freq_ceiling"])},
               "gate": {"kind": "atr_spread_ratio", "theta": int(theta),
                        "coverage_train": float(m.mean()),
                        "ev_in": float(p_in.mean()), "ev_out": float(p_out.mean())},
               "buckets": rows_out}, open(OUT, "w"), indent=2)
    print(f"结果 -> {OUT}")


if __name__ == "__main__":
    main()
