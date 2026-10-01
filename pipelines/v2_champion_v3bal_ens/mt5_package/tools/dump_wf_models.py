"""
dump_wf_models.py — 重跑 v3bal_ens 走查24折, 导出每折成员模型 + MT5用manifest。
关键设计:
  ① 阈值/权重以【已有checkpoint的stats】为准(它们产生了已报告的+$1113.5/1611笔);
  ② 重训模型后逐折对拍: 成员AUC / 权重 / 重算阈值 / 逐笔交易日志 vs checkpoint,
     全部一致 => 导出的模型复现已报告结果(确定性验证);
  ③ 每折附 self_test 参考行(真实OOS末行34特征 + 期望概率), 供EA端ONNX自检。
用法: python dump_wf_models.py --folds 0-7   (分块前台跑, 已完成折自动跳过)
"""
import os, sys, json, time, argparse, pickle
import numpy as np
import pandas as pd

PKG = "/home/z/my-project/download/xauusd_ml_v2"
sys.path.insert(0, PKG)

from config import CFG
from data import load_data
from walkforward import build_folds
from run_all import VARIANTS, variant_cfg, load_pack
from variants_ext import train_variant_fold, predict_weighted
from backtest import build_signals, run_backtest, calibrate_threshold

OUT = os.path.join(PKG, "mt5_package", "models", "wf")
CKPT = os.path.join(PKG, "results", "checkpoints")
FEATSET = "v3bal_ens"


def fold_range(spec):
    a, b = spec.split("-")
    return list(range(int(a), int(b) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0-7")
    args = ap.parse_args()
    folds_to_run = fold_range(args.folds)

    t_start = time.time()
    print(f"[dump] 加载数据+特征pack (复用缓存)...")
    m5, m1_pack, spread_cost, monthly = load_data(CFG)
    cfg_v = variant_cfg(FEATSET)
    folds = build_folds(m5, cfg_v)
    pack = load_pack(FEATSET)
    F, feats, lab, valid = pack["F"], pack["feats"], pack["lab"], pack["valid"]
    tuned = json.load(open(os.path.join(PKG, "results", "tuned_params.json")))[VARIANTS[FEATSET][2]]
    extra = VARIANTS[FEATSET][4]
    y_long = (lab["out_long"].to_numpy() == 1).astype(np.int8)
    y_short = (lab["out_short"].to_numpy() == 1).astype(np.int8)
    print(f"[dump] 数据就绪 ({time.time()-t_start:.0f}s), 开始 {folds_to_run}")

    results = []
    for i in folds_to_run:
        fold = folds[i]
        fold_dir = os.path.join(OUT, f"fold_{i:02d}")
        manifest_fp = os.path.join(fold_dir, "fold.json")
        if os.path.exists(manifest_fp):
            prev = json.load(open(manifest_fp))
            results.append({"fold": i, "oos_month": fold["oos_month"],
                            "pass": prev.get("verification", {}).get("trades_match", None),
                            "skipped": True})
            print(f"[dump] fold {i} 已导出, 跳过")
            continue
        t0 = time.time()
        oos_r = fold["oos_rows"]

        # ---- 1) 重训 (与stage_wf完全同口径) ----
        tv = train_variant_fold(extra, F, y_long, y_short, fold, tuned, cfg_v, m5, valid, i)
        mem = tv["members_models"]

        # ---- 2) checkpoint权威参数 ----
        ck = pickle.load(open(os.path.join(CKPT, f"wf_{FEATSET}_fold{i:02d}.pkl"), "rb"))
        st = ck["stats"]
        thr_l = float(st["threshold"]["long"])
        thr_s = float(st["threshold"]["short"])
        ck_w = {d: [float(m2["weight"]) for m2 in st["extra"]["directions"][d]["members"]]
                for d in ["long", "short"]}
        ck_auc = {d: [m2["auc_val"] for m2 in st["extra"]["directions"][d]["members"]]
                  for d in ["long", "short"]}

        # ---- 3) 确定性对拍 ----
        ver = {"fold": i, "oos_month": fold["oos_month"]}
        # 3a 成员AUC/权重
        auc_diffs, w_diffs = [], []
        for d in ["long", "short"]:
            for j, m2 in enumerate(mem[d]):
                my_auc = float(m2["auc_val"]) if np.isfinite(m2["auc_val"]) else 0.0
                ck_a = float(ck_auc[d][j]) if ck_auc[d][j] is not None else 0.0
                auc_diffs.append(abs(my_auc - ck_a))
                w_diffs.append(abs(float(m2["weight"]) - ck_w[d][j]))
        ver["max_auc_diff"] = float(np.max(auc_diffs))
        ver["max_weight_diff"] = float(np.max(w_diffs))
        # 3b 重算阈值
        (rl, rs), _ = calibrate_threshold(tv["pl_val"], tv["ps_val"], fold["val_rows"],
                                          lab, valid, m5, m1_pack, cfg_v)
        ver["thr_recompute_diff"] = [abs(float(rl) - thr_l), abs(float(rs) - thr_s)]
        # 3c 逐笔交易日志
        sig = build_signals(tv["pl_oos"], tv["ps_oos"], thr_l, thr_s)
        log2 = run_backtest(sig, oos_r, lab, valid, m5, m1_pack, cfg_v,
                            probs=(tv["pl_oos"], tv["ps_oos"]), tag=f"ml_{FEATSET}")
        log2["fold"] = fold["oos_month"]
        ck_log = ck["log"]
        ver["trades_re"] = int(len(log2))
        ver["trades_ck"] = int(len(ck_log))
        if len(log2) == len(ck_log) and len(log2) > 0:
            cols_eq = {}
            for col, atol in [("dir", 0), ("outcome", 0), ("pnl", 1e-9),
                              ("prob", 1e-9), ("entry_time", 0), ("exit_time", 0)]:
                if col in ("dir", "outcome"):
                    cols_eq[col] = bool((log2[col].to_numpy() == ck_log[col].to_numpy()).all())
                elif col in ("entry_time", "exit_time"):
                    cols_eq[col] = bool((log2[col].to_numpy() == ck_log[col].to_numpy()).all())
                else:
                    cols_eq[col] = bool(np.allclose(log2[col].to_numpy(), ck_log[col].to_numpy(), atol=atol))
            ver["log_cols_eq"] = cols_eq
            ver["trades_match"] = all(cols_eq.values())
        else:
            ver["trades_match"] = (len(log2) == len(ck_log))
        ver["pass"] = bool(ver["trades_match"] and ver["max_auc_diff"] < 1e-9
                           and max(ver["thr_recompute_diff"]) < 1e-9)

        # ---- 4) 导出模型 (只导 weight>0 成员; save_model自动截到早停最优树) ----
        os.makedirs(fold_dir, exist_ok=True)
        members_out = []
        for d in ["long", "short"]:
            for m2 in mem[d]:
                if float(m2["weight"]) <= 0:
                    continue
                fn = f"{d}_{m2['kind']}_s{m2['seed']}" + (".txt" if m2["kind"] == "lgb" else ".json")
                fp = os.path.join(fold_dir, fn)
                m2["model"].save_model(fp)
                members_out.append({"direction": d, "kind": m2["kind"], "seed": int(m2["seed"]),
                                    "file": fn, "weight": float(m2["weight"]),
                                    "rounds": int(m2["rounds"]),
                                    "auc_val": (None if not np.isfinite(m2["auc_val"]) else float(m2["auc_val"]))})

        # ---- 5) self_test 参考行: OOS末行(特征全finite) ----
        Xo = F.iloc[oos_r]
        fin_mask = np.isfinite(Xo.to_numpy(np.float32)).all(axis=1)
        cand = np.where(fin_mask)[0]
        j = int(cand[-1])
        grow = int(oos_r[j])
        xrow = Xo.iloc[[j]].to_numpy(np.float32)
        wl = [float(m2["weight"]) for m2 in mem["long"]]
        ws = [float(m2["weight"]) for m2 in mem["short"]]
        epl = float(predict_weighted(mem["long"], wl, xrow)[0])
        eps_ = float(predict_weighted(mem["short"], ws, xrow)[0])

        manifest = {
            "fold": i, "oos_month": fold["oos_month"], "window": fold["window"],
            "train_months": fold["train_months"],
            "thr_long": thr_l, "thr_short": thr_s,
            "members": members_out,
            "n_zero_weight_skipped": 12 - len(members_out),
            "self_test": {"bar_time": str(m5.index[grow]),
                          "features": [float(v) for v in xrow[0]],
                          "expected_pl": round(epl, 6), "expected_ps": round(eps_, 6)},
            "verification": ver,
        }
        json.dump(manifest, open(manifest_fp, "w"), indent=2)
        results.append({"fold": i, "oos_month": fold["oos_month"], "pass": ver["pass"],
                        "trades": ver["trades_re"], "trades_ck": ver["trades_ck"]})
        print(f"[dump] fold {i:02d} {fold['oos_month']}: 成员{len(members_out)}个(权重>0) "
              f"AUCdiff={ver['max_auc_diff']:.2e} thrDiff={max(ver['thr_recompute_diff']):.2e} "
              f"交易 {ver['trades_re']}/{ver['trades_ck']} match={ver['trades_match']} "
              f"PASS={ver['pass']} ({time.time()-t0:.0f}s)")

    # ---- 汇总 ----
    out_fp = os.path.join(OUT, f"dump_report_{args.folds.replace('-', '_')}.json")
    json.dump(results, open(out_fp, "w"), indent=2)
    n_pass = sum(1 for r in results if r.get("pass"))
    print(f"\n[dump] 本块 {len(results)}折: PASS {n_pass}/{len(results)} -> {out_fp}")


if __name__ == "__main__":
    main()
