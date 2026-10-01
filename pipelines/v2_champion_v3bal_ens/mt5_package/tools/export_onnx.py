"""
export_onnx.py — 把 dump 出的 WF 每折成员 + 生产成员全部转成 ONNX (MT5用)。
每个模型都做数值对拍: onnxruntime 输出 vs 原生库文件重载 predict,
测试行 = 该折 self_test 真实特征行 + 随机行(含NaN)。容差 1e-6。
输出:
  mt5_package/models/wf/fold_XX/*.onnx + fold.json(改指onnx) + wf_manifest.json(总索引)
  mt5_package/models/prod/*.onnx + prod_manifest.json
  mt5_package/models/onnx_verification.json (逐模型对拍报告)
"""
import os, sys, json, glob
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import onnx
import onnxmltools
import onnxruntime as ort
from onnxmltools.convert.common.data_types import FloatTensorType

PKG = "/home/z/my-project/download/xauusd_ml_v2"
WF = os.path.join(PKG, "mt5_package", "models", "wf")
PROD_OUT = os.path.join(PKG, "mt5_package", "models", "prod")
PROD_SRC = os.path.join(PKG, "production_models")
TARGET_OPSET = 15
N_FEATS = 34

rng = np.random.default_rng(20260915)
# 随机对拍行(带NaN/极端值, 覆盖缺失分支)
X_rand = rng.normal(0, 1.5, size=(4, N_FEATS)).astype(np.float32)
X_rand[1, 7] = np.nan
X_rand[2, 20] = np.nan
X_rand[3, 0] = 8.0

verification = {"wf": [], "prod": [], "tol": 1e-6}


def native_predict(kind, path, X):
    if kind == "lgb":
        m = lgb.Booster(model_file=path)
        return m.predict(X)
    m = xgb.Booster()
    m.load_model(path)
    return m.predict(xgb.DMatrix(X))


def convert_one(kind, native_path, onnx_path, X_test):
    """转换单个模型并对拍, 返回 (ok, max_diff)。仅保留float概率输出(删int64 label), 简化MQL5绑定。"""
    if kind == "lgb":
        m = lgb.Booster(model_file=native_path)
    else:
        m = xgb.Booster()
        m.load_model(native_path)
    onx = onnxmltools.convert_xgboost(m, initial_types=[("features", FloatTensorType([None, N_FEATS]))],
                                      target_opset=TARGET_OPSET) if kind == "xgb" else \
        onnxmltools.convert_lightgbm(m, initial_types=[("features", FloatTensorType([None, N_FEATS]))],
                                     target_opset=TARGET_OPSET, zipmap=False)
    # 删非float输出(只留 probabilities float[N,2])
    keep = [o for o in onx.graph.output if o.type.tensor_type.elem_type == onnx.TensorProto.FLOAT]
    assert len(keep) == 1, f"输出异常: {[o.name for o in onx.graph.output]}"
    del onx.graph.output[:]
    onx.graph.output.extend(keep)
    onnxmltools.save_model(onx, onnx_path)
    p_native = native_predict(kind, native_path, X_test)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    assert len(sess.get_outputs()) == 1, "单输出裁剪失败"
    res = sess.run(None, {"features": X_test})
    probs = res[0]
    p_onnx = probs[:, 1] if probs.ndim == 2 else probs
    max_diff = float(np.abs(np.asarray(p_native, dtype=np.float64) -
                            np.asarray(p_onnx, dtype=np.float64)).max())
    return max_diff <= 1e-6, max_diff


def main():
    os.makedirs(PROD_OUT, exist_ok=True)
    wf_index = []

    # ---------- WF: 24折 ----------
    fold_dirs = sorted(glob.glob(os.path.join(WF, "fold_*")))
    print(f"[onnx] WF折目录: {len(fold_dirs)}个")
    for fd in fold_dirs:
        mf = os.path.join(fd, "fold.json")
        man = json.load(open(mf))
        X_self = np.asarray([man["self_test"]["features"]], dtype=np.float32)
        X_test = np.vstack([X_self, X_rand])
        for mem in man["members"]:
            native_fn = mem.get("native_file", mem["file"])
            native_fp = os.path.join(fd, native_fn)
            onnx_fn = native_fn.replace(".txt", ".onnx").replace(".json", ".onnx")
            onnx_fp = os.path.join(fd, onnx_fn)
            ok, md = convert_one(mem["kind"], native_fp, onnx_fp, X_test)
            verification["wf"].append({"fold": man["fold"], "member": native_fn,
                                       "ok": bool(ok), "max_diff": md})
            mem["native_file"] = native_fn
            mem["file"] = onnx_fn
            if not ok:
                print(f"  !! FAIL fold{man['fold']} {native_fn} diff={md:.2e}")
        json.dump(man, open(mf, "w"), indent=2)
        wf_index.append({
            "fold": man["fold"], "oos_month": man["oos_month"],
            "folder": os.path.basename(fd),
            "thr_long": man["thr_long"], "thr_short": man["thr_short"],
            "members": [{"direction": m["direction"], "kind": m["kind"], "file": m["file"],
                         "weight": m["weight"]} for m in man["members"]],
            "self_test": man["self_test"],
        })
        print(f"[onnx] fold {man['fold']:02d} {man['oos_month']}: {len(man['members'])}成员转换完成")
    json.dump({"folds": wf_index}, open(os.path.join(WF, "wf_manifest.json"), "w"), indent=2)

    # ---------- PROD: 生产12成员(权重>0) ----------
    cfg = json.load(open(os.path.join(PROD_SRC, "trading_config.json")))
    # self_test行: 用pack_v3bal的最后一行全finite特征
    import pickle
    with open(os.path.join(PKG, "cache", "pack_v3bal.pkl"), "rb") as f:
        pack = pickle.load(f)
    F = pack["F"]
    fin = np.isfinite(F.to_numpy(np.float32)).all(axis=1)
    gi = int(np.where(fin)[0][-1])
    X_self = F.iloc[[gi]].to_numpy(np.float32)
    prod_members = []
    prod_p = {"long": 0.0, "short": 0.0}
    for d in ["long", "short"]:
        for mem in cfg["members"][d]:
            if float(mem["weight"]) <= 0:
                continue
            native_fp = os.path.join(PROD_SRC, mem["file"])
            onnx_fn = mem["file"].replace(".txt", ".onnx").replace(".json", ".onnx")
            onnx_fp = os.path.join(PROD_OUT, onnx_fn)
            X_test = np.vstack([X_self, X_rand])
            ok, md = convert_one(mem["kind"], native_fp, onnx_fp, X_test)
            verification["prod"].append({"member": mem["file"], "ok": bool(ok), "max_diff": md})
            if not ok:
                print(f"  !! PROD FAIL {mem['file']} diff={md:.2e}")
            # 期望概率: 文件重载原生(与predict_live同口径)
            p = native_predict(mem["kind"], native_fp, X_self)[0]
            prod_p[d] += float(mem["weight"]) * float(p)
            prod_members.append({"direction": d, "kind": mem["kind"], "seed": mem["seed"],
                                 "file": onnx_fn, "native_file": mem["file"],
                                 "weight": float(mem["weight"]), "rounds": mem["rounds"]})
    # 重算两方向(上面循环里pl_sum分别累计)
    prod_manifest = {
        "variant": "v3bal_ens", "mode": "production",
        "train_window": cfg["train_window"],
        "thr_long": float(cfg["long_threshold"]), "thr_short": float(cfg["short_threshold"]),
        "members": prod_members,
        "self_test": {"bar_time": str(F.index[gi]),
                      "features": [float(v) for v in X_self[0]],
                      "expected_pl": round(prod_p.get("long", 0.0), 6),
                      "expected_ps": round(prod_p.get("short", 0.0), 6)},
    }
    json.dump(prod_manifest, open(os.path.join(PROD_OUT, "prod_manifest.json"), "w"), indent=2)
    print(f"[onnx] PROD: {len(prod_members)}成员 -> prod_manifest.json "
          f"(self_test pL={prod_p.get('long', 0):.4f} pS={prod_p.get('short', 0):.4f})")

    # ---------- 汇总 ----------
    all_v = verification["wf"] + verification["prod"]
    n_ok = sum(1 for v in all_v if v["ok"])
    verification["n_total"] = len(all_v)
    verification["n_ok"] = n_ok
    json.dump(verification, open(os.path.join(PKG, "mt5_package", "models", "onnx_verification.json"), "w"), indent=2)
    print(f"\n[onnx] 对拍汇总: {n_ok}/{len(all_v)} 模型 PASS (tol 1e-6)")


if __name__ == "__main__":
    main()
