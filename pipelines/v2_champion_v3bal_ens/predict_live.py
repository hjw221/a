"""
predict_live.py — 实盘信号 (替代原 live_trading_predictor.py)
与原版的关键差异:
  1. 模型 = 按trading_config.json的变体 (默认v3aggr: v3特征34个 + LGB/XGB概率集成)
  2. TP/SL 不再是固定美元 —— 返回 ATR 自适应距离, EA按返回值挂单
  3. 特征用M5口径 (内部聚合): legacy需 ~293根M5, v3需 ~2885根M5(含10天挤压分母窗口)
  4. 缺失特征(数据不足)返回 HOLD, 绝不硬预测 (保留原设计)
  5. v3特征需要TICKVOL, M1入参须含TICKVOL列 (按5根M1求和聚合到M5)
返回: "L"/"S"/"HOLD" + 建议TP/SL距离(美元) + 概率与阈值
"""
import os, sys, json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE, "production_models", "trading_config.json")

_cfg = None
_models = None


def _load():
    global _cfg, _models
    if _cfg is not None:
        return
    import lightgbm as lgb
    import xgboost as xgb
    with open(CONFIG_PATH) as f:
        _cfg = json.load(f)
    _models = {}
    members = _cfg.get("members")
    if members:  # v3bal_ens: 多种子成员AUC加权 (与走查验证predict_weighted同口径)
        for direction in ["long", "short"]:
            lst = []
            for mem in members[direction]:
                fp = os.path.join(BASE, "production_models", mem["file"])
                if mem["kind"] == "lgb":
                    m = lgb.Booster(model_file=fp)
                else:
                    m = xgb.Booster()
                    m.load_model(fp)
                lst.append({"kind": mem["kind"], "model": m,
                            "weight": float(mem["weight"]),
                            "rounds": int(mem.get("rounds", 0) or 0)})
            _models[direction] = lst
        return
    for direction in ["long", "short"]:
        ml = lgb.Booster(model_file=os.path.join(BASE, "production_models", f"{direction}_lgb_latest.txt"))
        mx = xgb.Booster()
        mx.load_model(os.path.join(BASE, "production_models", f"{direction}_xgb_latest.json"))
        _models[direction] = (ml, mx)


def _legacy_features(m5df):
    """与训练完全一致的legacy23特征 (M5口径)。"""
    f = pd.DataFrame(index=m5df.index)
    c, o, h, l = m5df["CLOSE"], m5df["OPEN"], m5df["HIGH"], m5df["LOW"]
    for period in [1, 2, 3, 6, 12, 24, 48]:
        f[f"ret_{period}"] = c.pct_change(periods=period)
        if period > 1:
            f[f"vol_{period}"] = c.pct_change().rolling(window=period).std()
    f["body"] = (c - o) / o
    f["upper_shadow"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / o
    f["lower_shadow"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / o
    f["hour_sin"] = np.sin(m5df.index.hour * (2. * np.pi / 24))
    f["hour_cos"] = np.cos(m5df.index.hour * (2. * np.pi / 24))
    f["min_sin"] = np.sin(m5df.index.minute * (2. * np.pi / 60))
    f["min_cos"] = np.cos(m5df.index.minute * (2. * np.pi / 60))
    f["dow_sin"] = np.sin(m5df.index.dayofweek * (2. * np.pi / 7))
    f["dow_cos"] = np.cos(m5df.index.dayofweek * (2. * np.pi / 7))
    f["SPREAD"] = m5df["SPREAD"].astype(np.float64)
    return f[_cfg["features_list"]]


def _build_features(m5df):
    """按当前模型的特征集分派 (与训练严格同源同实现)。"""
    fs = _cfg.get("feature_set", "legacy23")
    if fs.startswith("v3"):
        from features_v3 import build_features_v3
        F, feats, _ = build_features_v3(m5df, _cfg["atr_window_m5"])
        return F[feats]
    if fs.startswith("v2"):
        from features import build_features_v2
        F, feats, _ = build_features_v2(m5df, _cfg["atr_window_m5"])
        return F[feats]
    return _legacy_features(m5df)


def _atr24(m5df, n):
    c, h, l = m5df["CLOSE"], m5df["HIGH"], m5df["LOW"]
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def predict_signal(recent_m1_df, spread_points_now=None):
    """
    recent_m1_df: 最近的M1 K线 (v3需至少约2885×5=14425根≈10个交易日, 含TICKVOL/SPREAD列),
                  列 OPEN/HIGH/LOW/CLOSE/TICKVOL/SPREAD, datetime索引。
    spread_points_now: 当前实时点差(点数); 缺省用最后一根M5的中位数。
    返回 dict: {"signal": "L"/"S"/"HOLD", "tp_dist": 美元, "sl_dist": 美元,
                "prob_long": float, "prob_short": float, "reason": str}
    EA责任: 按返回的 tp_dist/sl_dist 挂止盈止损, 单持仓, 平仓后冷却10根M1。
    """
    _load()
    import xgboost as xgb
    df = recent_m1_df.copy().sort_index()
    # M1 -> M5 聚合 (与训练数据管线一致; TICKVOL按和)
    agg = {"OPEN": ("OPEN", "first"), "HIGH": ("HIGH", "max"), "LOW": ("LOW", "min"),
           "CLOSE": ("CLOSE", "last"), "SPREAD": ("SPREAD", "median")}
    if "TICKVOL" in df.columns:
        agg["TICKVOL"] = ("TICKVOL", "sum")
    else:
        agg["TICKVOL"] = ("TICKVOL", "size")   # 无TICKVOL时用K线数代理(降级, 结果仅供参考)
    m5 = df.resample("5min").agg(**agg).dropna(subset=["OPEN"])
    need = int(_cfg.get("min_m5_bars", _cfg["atr_window_m5"] + 5))
    if len(m5) < need:
        return {"signal": "HOLD", "reason": f"M5数据不足({len(m5)}/{need}根), 跳过", "tp_dist": None, "sl_dist": None}

    X = _build_features(m5)
    last = X.iloc[-1:]
    if last.isnull().any(axis=1).iloc[0]:
        return {"signal": "HOLD", "reason": "特征缺失, 跳过", "tp_dist": None, "sl_dist": None}

    xv = last.to_numpy(np.float32)
    import xgboost as xgb
    if isinstance(_models["long"], list):   # ens加权成员: p = Σ w_i·p_i (Σw=1)
        def _wprob(members, x):
            ps = []
            for mem in members:
                if mem["weight"] <= 0:
                    continue
                if mem["kind"] == "lgb":
                    r = mem["rounds"]
                    ps.append(mem["weight"] * mem["model"].predict(x, num_iteration=r if r > 0 else None)[0])
                else:
                    ps.append(mem["weight"] * mem["model"].predict(xgb.DMatrix(x))[0])
            return float(np.sum(ps)) if ps else 0.5
        pl = _wprob(_models["long"], xv)
        ps = _wprob(_models["short"], xv)
    else:
        r = _cfg.get("rounds", {})
        rl, rs = int(r.get("long_lgb", 0) or 0), int(r.get("short_lgb", 0) or 0)
        rxl, rxs = int(r.get("long_xgb", 0) or 0), int(r.get("short_xgb", 0) or 0)
        pl = 0.5 * (_models["long"][0].predict(xv, num_iteration=rl or None)[0] +
                    _models["long"][1].predict(xgb.DMatrix(xv), iteration_range=(0, rxl + 1))[0])
        ps = 0.5 * (_models["short"][0].predict(xv, num_iteration=rs or None)[0] +
                    _models["short"][1].predict(xgb.DMatrix(xv), iteration_range=(0, rxs + 1))[0])

    # ATR自适应障碍距离 (与训练/回测同一公式)
    atr = float(_atr24(m5, _cfg["atr_window_m5"]).iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        return {"signal": "HOLD", "reason": "ATR无效", "tp_dist": None, "sl_dist": None}
    sp_pts = spread_points_now if spread_points_now is not None else float(m5["SPREAD"].iloc[-1])
    spread_cost = max(sp_pts, 1.0) * _cfg["point_value"]
    tp_dist = _cfg["tp_atr_mult"] * atr
    sl_dist = max(_cfg["sl_atr_mult"] * atr, _cfg["sl_floor_usd"],
                  _cfg["spread_floor_mult"] * spread_cost)

    thr_l, thr_s = _cfg["long_threshold"], _cfg["short_threshold"]
    long_sig, short_sig = pl > thr_l, ps > thr_s
    if long_sig and short_sig:
        sig = "L" if (pl - thr_l) >= (ps - thr_s) else "S"
    elif long_sig:
        sig = "L"
    elif short_sig:
        sig = "S"
    else:
        sig = "HOLD"
    return {"signal": sig, "tp_dist": round(float(tp_dist), 3), "sl_dist": round(float(sl_dist), 3),
            "prob_long": round(float(pl), 4), "prob_short": round(float(ps), 4),
            "thr_long": thr_l, "thr_short": thr_s, "atr": round(atr, 3),
            "reason": f"pL={pl:.3f}/{thr_l:.3f} pS={ps:.3f}/{thr_s:.3f}"}


if __name__ == "__main__":
    # 用真实数据末端做演示 (取16000根M1: 兼容v3的10天回看窗)
    from config import CFG
    from data import load_raw_m1
    n_take = 16000
    print(f"加载最近{n_take}根真实M1做演示预测...")
    df = load_raw_m1(CFG["data_path"]).tail(n_take)
    r = predict_signal(df)
    print(json.dumps(r, ensure_ascii=False, indent=2))
