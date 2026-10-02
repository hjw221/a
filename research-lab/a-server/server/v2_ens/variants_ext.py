"""
变体拓展层 (2026-09-14): v3bal信号强度三张牌的防泄露实现。

三张牌 (全部只动训练过程, 不动特征/几何/回测口径, 与v3bal基线严格可比):
  stride (v3bal_uniq): M5事件采样去标签重叠 — 训练事件从"逐根"改为每12根取1
      (相位逐折错开防时段混叠), 叶子最小样本数按采样比例等效缩放。
      动机: 标签horizon=360根M1=72根M5, 逐根训练时每条真实结局被重复计数~72次,
      采样后残余重复~6次, 有效样本数与表面样本数重新对齐。
  ens (v3bal_ens): 多种子成员集成 — 每方向 lgb+xgb × 3种子 = 6成员,
      以训练窗内部val段AUC加权 (w = max(0, AUC-0.50)), AUC<0.51成员权重清零,
      全清时退回均匀平均 (记录fallback)。
  meta (v3bal_meta): 两层元标签 — 主模型与v3bal基线完全一致(同数据/参数/种子,
      概率逐位一致), 其概率top 15% (val分位标定) 为候选入场; 第二层LGB只在候选上
      训练"该不该接" (标签同为三重障碍结局, 只用有效标签行), 交易 = 候选 且
      meta概率过plr_wr阈值。候选训练集用训练窗内2折OOF概率选取(折间purge,
      固定轮数取主模型ES轮数), 避免用自见的概率挑样本。

防泄露: 所有选择(候选分位/成员权重/元模型ES/阈值)只用训练窗内部数据
(fit段 + val段), OOS月零参与; purge口径与既有管线完全一致。
"""
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from models import fit_lgb, fit_xgb, predict_ens
from walkforward import time_decay_weights


# ---------------- 通用小工具 ----------------
def _lgb_predict(m, X):
    return m.predict(X, num_iteration=m.best_iteration if m.best_iteration else None)


def _xgb_predict(m, X):
    return m.predict(xgb.DMatrix(X))


def _auc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if len(y) < 50 or y.sum() < 10 or (len(y) - y.sum()) < 10:
        return float("nan")
    return float(roc_auc_score(y, p))


def scale_tree_params(params, kind, frac):
    """事件采样/子集训练后, 叶子最小样本数按样本比例等效缩放 (保持单位样本正则强度)。"""
    p = dict(params)
    if frac >= 1.0:
        return p
    if kind == "lgb":
        p["min_data_in_leaf"] = max(20, int(round(p.get("min_data_in_leaf", 100) * frac)))
    else:
        p["min_child_weight"] = max(10.0, float(p.get("min_child_weight", 100.0) * frac))
    return p


def _fit_pair(Xf, yf, wf, tuned_dir, Xv, yv, w_val, es, mr):
    """与基线完全相同的 lgb+xgb 概率对 (tuned_dir: {"lgb":..., "xgb":...})。"""
    m_l, r_l = fit_lgb(Xf, yf, wf, tuned_dir["lgb"], Xv, yv, w_val, es, mr)
    m_x, r_x = fit_xgb(Xf, yf, wf, tuned_dir["xgb"], Xv, yv, w_val, es, mr)
    return [("lgb", m_l), ("xgb", m_x)], {"lgb": int(r_l), "xgb": int(r_x)}


# ---------------- 牌1: stride (标签去重叠事件采样) ----------------
def _fold_stride(extra, F, y_long, y_short, fold, tuned, cfg_v, m5, fold_idx,
                 Xv, Xo, w_val, valid):
    fit_r, val_r = fold["fit_rows"], fold["val_rows"]
    stride = int(extra.get("stride", 12))
    phase = (fold_idx * 7) % stride
    rows = fit_r[phase::stride]
    frac = 1.0 / stride
    w_fit = time_decay_weights(m5.index, rows, cfg_v["sample_halflife_days"])
    Xf = F.iloc[rows].to_numpy(np.float32)
    yv_l, yv_s = y_long[val_r], y_short[val_r]
    ens, rounds = {}, {}
    for d, y in [("long", y_long), ("short", y_short)]:
        tuned_s = {"lgb": scale_tree_params(tuned[d]["lgb"], "lgb", frac),
                   "xgb": scale_tree_params(tuned[d]["xgb"], "xgb", frac)}
        yv = yv_l if d == "long" else yv_s
        ens[d], rounds[d] = _fit_pair(Xf, y[rows], w_fit, tuned_s, Xv, yv, w_val,
                                      cfg_v["early_stopping_rounds"], cfg_v["max_rounds"])
    pl_val = predict_ens(ens["long"], Xv)
    ps_val = predict_ens(ens["short"], Xv)
    pl_oos = predict_ens(ens["long"], Xo)
    ps_oos = predict_ens(ens["short"], Xo)
    ex = {"mode": "stride", "stride": stride, "phase": int(phase),
          "n_fit": int(len(rows)), "n_fit_full": int(len(fit_r))}
    return pl_val, ps_val, pl_oos, ps_oos, rounds, ex, \
        (pl_val, ps_val), (pl_oos, ps_oos)


# ---------------- 牌2: ens (多种子AUC加权集成) ----------------
def ensemble_weights(members, drop_below=0.51):
    """w = max(0, auc-0.50), auc<drop_below清零; 全零退回均匀。返回(weights, fallback)。"""
    aucs = np.array([m["auc_val"] if np.isfinite(m["auc_val"]) else 0.0 for m in members])
    w = np.maximum(aucs - 0.50, 0.0)
    w = np.where(aucs >= drop_below, w, 0.0)
    if w.sum() <= 0:
        return np.ones(len(members)) / len(members), True
    return w / w.sum(), False


def predict_weighted(members, w, X):
    ps = []
    for m, wi in zip(members, w):
        if wi <= 0:
            continue
        pv = _lgb_predict(m["model"], X) if m["kind"] == "lgb" else _xgb_predict(m["model"], X)
        ps.append(wi * pv)
    if not ps:
        return np.full(len(X), 0.5)
    return np.sum(np.vstack(ps), axis=0)


def _fold_ens(extra, F, y_long, y_short, fold, tuned, cfg_v, m5, fold_idx,
              Xv, Xo, w_val, valid):
    fit_r, val_r = fold["fit_rows"], fold["val_rows"]
    seeds = list(extra.get("seeds", [42, 1337, 2024]))
    drop_below = float(extra.get("drop_below", 0.51))
    es, mr = cfg_v["early_stopping_rounds"], cfg_v["max_rounds"]
    w_fit = time_decay_weights(m5.index, fit_r, cfg_v["sample_halflife_days"])
    Xf = F.iloc[fit_r].to_numpy(np.float32)
    valid_val = valid[val_r]
    ex = {"mode": "ens", "seeds": [int(s) for s in seeds], "directions": {}}
    probs = {}
    rounds = {}
    for d, y in [("long", y_long), ("short", y_short)]:
        yv = y[val_r]
        members = []
        for seed in seeds:
            for kind in ["lgb", "xgb"]:
                params = dict(tuned[d][kind])
                if kind == "lgb":
                    params.update(seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
                else:
                    params.update(seed=seed)
                if kind == "lgb":
                    m, r = fit_lgb(Xf, y[fit_r], w_fit, params, Xv, yv, w_val, es, mr)
                    pv = _lgb_predict(m, Xv)
                else:
                    m, r = fit_xgb(Xf, y[fit_r], w_fit, params, Xv, yv, w_val, es, mr)
                    pv = _xgb_predict(m, Xv)
                auc = _auc(yv[valid_val], pv[valid_val])
                members.append({"kind": kind, "seed": int(seed), "model": m,
                                "rounds": int(r), "auc_val": auc})
        w, fb = ensemble_weights(members, drop_below)
        probs[(d, "val")] = predict_weighted(members, w, Xv)
        probs[(d, "oos")] = predict_weighted(members, w, Xo)
        rounds[d] = {f"{m['kind']}_s{m['seed']}": m["rounds"] for m in members}
        ex["directions"][d] = {
            "fallback_uniform": bool(fb),
            "n_kept": int((w > 0).sum()), "n_members": len(members),
            "mean_auc_val": float(np.nanmean([m["auc_val"] for m in members])),
            "members": [{"kind": m["kind"], "seed": m["seed"],
                         "auc_val": (None if not np.isfinite(m["auc_val"]) else float(m["auc_val"])),
                         "weight": float(wi)} for m, wi in zip(members, w)]}
    pl_val, ps_val = probs[("long", "val")], probs[("short", "val")]
    pl_oos, ps_oos = probs[("long", "oos")], probs[("short", "oos")]
    return pl_val, ps_val, pl_oos, ps_oos, rounds, ex, \
        (pl_val, ps_val), (pl_oos, ps_oos)


# ---------------- 牌3: meta (两层元标签过滤) ----------------
def _fold_meta(extra, F, y_long, y_short, fold, tuned, cfg_v, m5, fold_idx,
               Xv, Xo, w_val, valid):
    fit_r, val_r, oos_r = fold["fit_rows"], fold["val_rows"], fold["oos_rows"]
    pb = fold["purge_bars"]
    cand_q = float(extra.get("cand_q", 0.85))
    es, mr = cfg_v["early_stopping_rounds"], cfg_v["max_rounds"]
    halflife = cfg_v["sample_halflife_days"]
    valid_fit, valid_val, valid_oos = valid[fit_r], valid[val_r], valid[oos_r]
    w_fit = time_decay_weights(m5.index, fit_r, halflife)
    Xf = F.iloc[fit_r].to_numpy(np.float32)
    ex = {"mode": "meta", "cand_q": cand_q, "directions": {}}
    probs = {}
    rounds = {}
    auc_src = {}

    for d, y in [("long", y_long), ("short", y_short)]:
        yv = y[val_r]
        # 1) 主模型 = 与v3bal基线逐位一致 (同数据/参数/种子/早停)
        primary, r_pri = _fit_pair(Xf, y[fit_r], w_fit, tuned[d], Xv, yv, w_val, es, mr)
        p_val = predict_ens(primary, Xv)
        p_oos = predict_ens(primary, Xo)
        rounds[d] = r_pri

        # 2) 训练窗内2折OOF概率 (折间purge, 固定轮数=主模型轮数, 不碰对方标签)
        n = len(fit_r)
        cut = n // 2
        A, B = fit_r[:cut], fit_r[cut + pb:]
        p_oof = np.full(n, np.nan)
        wA = time_decay_weights(m5.index, A, halflife)
        wB = time_decay_weights(m5.index, B, halflife)
        r_l, r_x = r_pri["lgb"], r_pri["xgb"]
        XB = F.iloc[B].to_numpy(np.float32)
        mA_l, _ = fit_lgb(F.iloc[A].to_numpy(np.float32), y[A], wA, tuned[d]["lgb"],
                          None, None, None, es, r_l)
        mA_x, _ = fit_xgb(F.iloc[A].to_numpy(np.float32), y[A], wA, tuned[d]["xgb"],
                          None, None, None, es, r_x)
        p_oof[cut + pb:] = 0.5 * (_lgb_predict(mA_l, XB) + _xgb_predict(mA_x, XB))
        XA = F.iloc[A].to_numpy(np.float32)
        mB_l, _ = fit_lgb(XB, y[B], wB, tuned[d]["lgb"], None, None, None, es, r_l)
        mB_x, _ = fit_xgb(XB, y[B], wB, tuned[d]["xgb"], None, None, None, es, r_x)
        p_oof[:cut] = 0.5 * (_lgb_predict(mB_l, XA) + _xgb_predict(mB_x, XA))

        # 3) 候选规则: 主模型概率 top cand_q (val分位, 与OOS同一模型实例)
        if valid_val.sum() > 50:
            thr_cand = float(np.quantile(p_val[valid_val], cand_q))
        else:
            thr_cand = 0.5
        cand_val = valid_val & (p_val >= thr_cand)
        cand_oos = valid_oos & (p_oos >= thr_cand)
        ok_oof = np.isfinite(p_oof) & valid_fit
        q_oof = float(np.quantile(p_oof[ok_oof], cand_q)) if ok_oof.sum() > 50 else thr_cand
        cand_fit = ok_oof & (p_oof >= q_oof)

        # 4) 第二层元模型 (只在候选上训练; 只用有效标签行)
        n_fit_cand = int(cand_fit.sum())
        n_pos_cand = int(y[fit_r][cand_fit].sum())
        feasible = (n_fit_cand >= 500 and n_pos_cand >= 50
                    and cand_val.sum() >= 100 and yv[cand_val].sum() >= 20)
        info = {"thr_cand": thr_cand, "n_cand_fit": n_fit_cand,
                "n_cand_val": int(cand_val.sum()), "n_cand_oos": int(cand_oos.sum()),
                "n_pos_cand_fit": n_pos_cand,
                "auc_primary_val": _auc(yv[valid_val], p_val[valid_val]),
                "auc_primary_oos": _auc(y[oos_r][valid_oos], p_oos[valid_oos])}
        if feasible:
            frac = n_fit_cand / max(1, len(fit_r))
            rows_meta = fit_r[cand_fit]
            Xmeta = F.iloc[rows_meta].to_numpy(np.float32)
            ymeta = y[rows_meta]
            wmeta = time_decay_weights(m5.index, rows_meta, halflife)
            Xv_c = F.iloc[val_r[cand_val]].to_numpy(np.float32)
            yv_c = yv[cand_val]
            params_meta = scale_tree_params(tuned[d]["lgb"], "lgb", frac)
            # 元模型固定300轮, 不用早停: ES集=val候选仅~5k行, AUC噪声(±0.01)远大于
            # 单轮增量, 冒烟实测会塌缩到1~2轮(常数预测器)。此修复基于训练窗内部
            # 诊断(候选集val AUC≈0.50 + rounds=1), 未使用任何OOS信息。
            META_ROUNDS = 300
            m_meta, r_meta = fit_lgb(Xmeta, ymeta, wmeta, params_meta,
                                     None, None, None, es, META_ROUNDS)
            pm_val = _lgb_predict(m_meta, Xv_c)
            p_eff_val = np.zeros(len(val_r))
            p_eff_val[cand_val] = pm_val
            p_eff_oos = np.zeros(len(oos_r))
            if cand_oos.sum() > 0:
                Xo_c = F.iloc[oos_r[cand_oos]].to_numpy(np.float32)
                p_eff_oos[cand_oos] = _lgb_predict(m_meta, Xo_c)
            info.update({
                "fallback": False, "meta_rounds": int(r_meta),
                "frac_meta_train": float(frac),
                "auc_meta_val": _auc(yv_c, pm_val),
                "auc_meta_oos": _auc(y[oos_r][cand_oos], p_eff_oos[cand_oos])
                if cand_oos.sum() >= 50 else float("nan")})
        else:
            # 该方向退回基线行为 (主模型概率直接过阈值), 如实记录
            p_eff_val = p_val.copy()
            p_eff_oos = p_oos.copy()
            info.update({"fallback": True, "auc_meta_val": float("nan"),
                         "auc_meta_oos": float("nan")})
        probs[(d, "val")], probs[(d, "oos")] = p_eff_val, p_eff_oos
        auc_src[(d, "val")], auc_src[(d, "oos")] = p_val, p_oos
        ex["directions"][d] = {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                               for k, v in info.items()}

    pl_val, ps_val = probs[("long", "val")], probs[("short", "val")]
    pl_oos, ps_oos = probs[("long", "oos")], probs[("short", "oos")]
    return pl_val, ps_val, pl_oos, ps_oos, rounds, ex, \
        (auc_src[("long", "val")], auc_src[("short", "val")]), \
        (auc_src[("long", "oos")], auc_src[("short", "oos")])


# ---------------- 入口 ----------------
def train_variant_fold(extra, F, y_long, y_short, fold, tuned, cfg_v, m5, valid, fold_idx):
    """extra变体的单折训练。返回dict; probs均与本段行号对齐。"""
    val_r, oos_r = fold["val_rows"], fold["oos_rows"]
    Xv = F.iloc[val_r].to_numpy(np.float32)
    Xo = F.iloc[oos_r].to_numpy(np.float32)
    w_val = np.ones(len(val_r))
    mode = extra.get("mode")
    args = (extra, F, y_long, y_short, fold, tuned, cfg_v, m5, fold_idx, Xv, Xo, w_val, valid)
    if mode == "stride":
        pl_val, ps_val, pl_oos, ps_oos, rounds, ex, av, ao = _fold_stride(*args)
    elif mode == "ens":
        pl_val, ps_val, pl_oos, ps_oos, rounds, ex, av, ao = _fold_ens(*args)
    elif mode == "meta":
        pl_val, ps_val, pl_oos, ps_oos, rounds, ex, av, ao = _fold_meta(*args)
    else:
        raise ValueError(f"未知mode: {mode}")
    return {"pl_val": pl_val, "ps_val": ps_val, "pl_oos": pl_oos, "ps_oos": ps_oos,
            "rounds": rounds, "ex_stats": ex,
            "auc_val_src": av, "auc_oos_src": ao}
