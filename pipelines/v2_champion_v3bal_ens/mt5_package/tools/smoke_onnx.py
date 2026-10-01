"""ONNX转换冒烟: 生产模型 lgb/xgb 各1个 -> onnxmltools -> onnxruntime 对拍原生predict.
确认: ①数值一致 ②输出形状(label int64 + probs float[N,2]) ③zipmap=False可行"""
import numpy as np, lightgbm as lgb, xgboost as xgb, onnxmltools, onnxruntime as ort
from onnxmltools.convert.common.data_types import FloatTensorType

BASE = "/home/z/my-project/download/xauusd_ml_v2/production_models"
IN_TYPES = [("features", FloatTensorType([None, 34]))]
rng = np.random.default_rng(7)
X = rng.normal(0, 1, size=(5, 34)).astype(np.float32)
X[:, 0] += 1.5  # atr_ratio量级
X[2, 5] = np.nan  # 测试缺失值分支!

# ---- LGB ----
m = lgb.Booster(model_file=f"{BASE}/long_lgb_s42_latest.txt")
p_native = m.predict(X)  # 文件重载, num_iteration=None => 全部树(文件已截到best)
onx = onnxmltools.convert_lightgbm(m, initial_types=IN_TYPES, target_opset=15, zipmap=False)
onnxmltools.save_model(onx, "/tmp/smoke_lgb.onnx")
sess = ort.InferenceSession("/tmp/smoke_lgb.onnx", providers=["CPUExecutionProvider"])
print("LGB inputs:", [(i.name, i.type, i.shape) for i in sess.get_inputs()])
print("LGB outputs:", [(o.name, o.type, o.shape) for o in sess.get_outputs()])
res = sess.run(None, {sess.get_inputs()[0].name: X})
print("LGB outputs实际:", [(type(r).__name__, getattr(r, 'shape', None), getattr(r, 'dtype', None)) for r in res])
p_onnx = res[1][:, 1] if res[1].ndim == 2 else res[1]
print("LGB native:", np.round(p_native, 6))
print("LGB onnx  :", np.round(p_onnx, 6))
print("LGB max|diff| =", np.abs(p_native - p_onnx).max())

# ---- XGB ----
mx = xgb.Booster()
mx.load_model(f"{BASE}/long_xgb_s42_latest.json")
px_native = mx.predict(xgb.DMatrix(X))
onx2 = onnxmltools.convert_xgboost(mx, initial_types=IN_TYPES, target_opset=15, zipmap=False)
onnxmltools.save_model(onx2, "/tmp/smoke_xgb.onnx")
sess2 = ort.InferenceSession("/tmp/smoke_xgb.onnx", providers=["CPUExecutionProvider"])
print("XGB outputs:", [(o.name, o.type, o.shape) for o in sess2.get_outputs()])
res2 = sess2.run(None, {sess2.get_inputs()[0].name: X})
px_onnx = res2[1][:, 1] if res2[1].ndim == 2 else res2[1]
print("XGB native:", np.round(px_native, 6))
print("XGB onnx  :", np.round(px_onnx, 6))
print("XGB max|diff| =", np.abs(px_native - px_onnx).max())
