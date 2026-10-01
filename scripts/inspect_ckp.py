"""检查 v3bal_ens 走查 checkpoint 内容: 是否包含每折训练好的模型 boosters."""
import pickle, sys

CKP = "/home/z/my-project/download/xauusd_ml_v2/results/checkpoints/wf_v3bal_ens_fold00.pkl"

with open(CKP, "rb") as f:
    d = pickle.load(f)

def describe(obj, depth=0, max_depth=3):
    pad = "  " * depth
    if depth > max_depth:
        print(pad + "...")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            t = type(v).__name__
            extra = ""
            if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
                try:
                    extra = f" len={len(v)}"
                except Exception:
                    pass
            print(f"{pad}{k}: {t}{extra}")
            # 深入模型相关键
            if any(s in str(k).lower() for s in ("model", "member", "ens", "booster")):
                describe(v, depth + 1, max_depth)
            elif isinstance(v, dict) and depth < max_depth:
                describe(v, depth + 1, max_depth)
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}list/tuple len={len(obj)}")
        if obj:
            print(f"{pad}  [0]: {type(obj[0]).__name__}")
            if isinstance(obj[0], dict):
                describe(obj[0], depth + 2, max_depth)

print("=== top-level keys ===")
describe(d, 0, 2)
print()
# 专门找模型对象
import lightgbm, xgboost
found = []
def hunt(obj, path="root"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            hunt(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            hunt(v, f"{path}[{i}]")
    else:
        if isinstance(obj, (lightgbm.Booster, xgboost.Booster)):
            found.append((path, type(obj).__name__))
hunt(d)
print(f"=== 找到 {len(found)} 个 booster 模型对象 ===")
for p, t in found[:30]:
    print(f"  {t}: {p}")
