"""
模型层: LightGBM + XGBoost 概率集成, 时间衰减样本权重, 早停,
内部purged CV随机搜索调参 (不依赖optuna, 2核机器友好)。
防泄露: 调参只用首折训练窗内部数据, 选定后全程冻结;
        早停的验证段也在训练窗内部, OOS月份从未参与任何选择。
"""
import os
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score


# ---------------- 基础拟合 ----------------
def fit_lgb(X, y, w, params, X_val=None, y_val=None, w_val=None, es_rounds=50, max_rounds=600):
    p = dict(params)
    p.update(objective="binary", metric="auc", verbosity=-1, num_threads=int(os.environ.get("ML_THREADS", "2")))
    dtrain = lgb.Dataset(X, label=y, weight=w)
    valid_sets = None
    cbs = [lgb.log_evaluation(0)]
    if X_val is not None:
        dval = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=dtrain)
        valid_sets = [dval]
        cbs.append(lgb.early_stopping(es_rounds, verbose=False))
    model = lgb.train(p, dtrain, num_boost_round=max_rounds, valid_sets=valid_sets, callbacks=cbs)
    rounds = model.best_iteration if valid_sets else max_rounds
    return model, rounds


def fit_xgb(X, y, w, params, X_val=None, y_val=None, w_val=None, es_rounds=50, max_rounds=600):
    p = dict(params)
    p.update(objective="binary:logistic", eval_metric="auc", tree_method="hist",
             nthread=int(os.environ.get("ML_THREADS", "2")), verbosity=0)
    dtrain = xgb.DMatrix(X, label=y, weight=w)
    dval = None
    if X_val is not None:
        dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)
    evals = [(dval, "val")] if dval is not None else None
    if dval is not None:
        model = xgb.train(p, dtrain, num_boost_round=max_rounds, evals=evals,
                          early_stopping_rounds=es_rounds, verbose_eval=False)
        rounds = getattr(model, "best_iteration", max_rounds) or max_rounds
    else:
        model = xgb.train(p, dtrain, num_boost_round=max_rounds)
        rounds = max_rounds
    return model, rounds


def predict_ens(models, X):
    """概率集成: LightGBM与XGBoost输出均值。"""
    ps = []
    for kind, m in models:
        if kind == "lgb":
            ps.append(m.predict(X, num_iteration=m.best_iteration if m.best_iteration else None))
        else:
            ps.append(m.predict(xgb.DMatrix(X)))
    return np.mean(np.vstack(ps), axis=0)


# ---------------- 随机搜索空间 ----------------
def sample_params(rng, kind):
    if kind == "lgb":
        return {
            "learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.15)))),
            "num_leaves": int(rng.integers(15, 96)),
            "min_data_in_leaf": int(rng.integers(100, 1500)),
            "feature_fraction": float(rng.uniform(0.5, 1.0)),
            "bagging_fraction": float(rng.uniform(0.6, 1.0)),
            "bagging_freq": 1,
            "lambda_l1": float(rng.uniform(0, 5)),
            "lambda_l2": float(rng.uniform(0, 10)),
        }
    return {
        "eta": float(np.exp(rng.uniform(np.log(0.01), np.log(0.15)))),
        "max_leaves": int(rng.integers(15, 96)),
        "max_depth": int(rng.integers(3, 9)),
        "min_child_weight": float(rng.uniform(50, 1200)),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),
        "reg_alpha": float(rng.uniform(0, 5)),
        "reg_lambda": float(rng.uniform(0, 10)),
    }


def tune_on_window(Xdf, y, cfg, rows_fit_a, rows_val_a, rows_fit_b, rows_val_b, w_full, seed):
    """
    首折训练窗内部的2段purged时序验证 (只用训练窗数据!):
      段A: 拟合[0,~60%) -> 验证[60%,~85%)   段B: 拟合[0,~85%) -> 验证[85%,100%)
    每个候选配置取两段AUC均值。返回每种模型的最优参数与全部评分(供报告)。
    """
    rng = np.random.default_rng(seed)
    sub = cfg["tune"]["subsample_step"]
    results = {"lgb": [], "xgb": []}
    for kind in ["lgb", "xgb"]:
        for k in range(cfg["tune"]["n_configs"]):
            params = sample_params(rng, kind)
            aucs = []
            for fr, vr in [(rows_fit_a, rows_val_a), (rows_fit_b, rows_val_b)]:
                fr_s, vr_s = fr[::sub], vr[::sub]
                Xf = Xdf.iloc[fr_s].to_numpy(np.float32)
                yf = y[fr_s]
                wf = w_full[fr_s]
                Xv = Xdf.iloc[vr_s].to_numpy(np.float32)
                yv = y[vr_s]
                if yf.sum() < 50 or yv.sum() < 20:
                    continue
                if kind == "lgb":
                    m, _ = fit_lgb(Xf, yf, wf, params, Xv, yv, None,
                                   es_rounds=cfg["early_stopping_rounds"],
                                   max_rounds=cfg["max_rounds"])
                else:
                    m, _ = fit_xgb(Xf, yf, wf, params, Xv, yv, None,
                                   es_rounds=cfg["early_stopping_rounds"],
                                   max_rounds=cfg["max_rounds"])
                p = m.predict(Xv) if kind == "lgb" else m.predict(xgb.DMatrix(Xv))
                aucs.append(roc_auc_score(yv, p))
            if aucs:
                results[kind].append({"params": params, "auc": float(np.mean(aucs)),
                                      "folds": aucs})
        results[kind].sort(key=lambda r: -r["auc"])
    best = {k: (v[0]["params"] if v else None) for k, v in results.items()}
    return best, results
