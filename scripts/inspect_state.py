# 摸底: checkpoint内容 / pack结构 / 基线指标 / 交易日志格式
import pickle, json, pandas as pd, numpy as np, sys

BASE = "/home/z/my-project/download/xauusd_ml_v2"

# 1) checkpoint 内容
with open(f"{BASE}/results/checkpoints/wf_v3bal_ens_fold00.pkl", "rb") as f:
    ck = pickle.load(f)
print("== checkpoint keys ==")
if isinstance(ck, dict):
    for k, v in ck.items():
        t = type(v).__name__
        if isinstance(v, np.ndarray):
            print(f"  {k}: ndarray {v.shape} {v.dtype}")
        elif isinstance(v, (pd.DataFrame, pd.Series)):
            print(f"  {k}: {type(v).__name__} {v.shape}")
        elif isinstance(v, dict):
            print(f"  {k}: dict keys={list(v.keys())[:12]}")
        elif isinstance(v, list):
            print(f"  {k}: list len={len(v)} first={type(v[0]).__name__ if v else None}")
        else:
            print(f"  {k}: {t} = {str(v)[:100]}")

# 2) 基线 summary
with open(f"{BASE}/results/summary_v3bal_ens.json") as f:
    sm = json.load(f)
print("\n== summary_v3bal_ens ==")
print(json.dumps(sm, indent=1, ensure_ascii=False)[:2000])

# 3) 交易日志
tr = pd.read_csv(f"{BASE}/results/trades_v3bal_ens.csv")
print("\n== trades_v3bal_ens ==")
print(tr.columns.tolist())
print(tr.head(3).to_string())
print("rows:", len(tr), "| pnl sum:", round(tr.pnl.sum(), 2))

# 4) pack 结构
with open(f"{BASE}/cache/pack_v3bal.pkl", "rb") as f:
    pack = pickle.load(f)
print("\n== pack_v3bal ==")
if isinstance(pack, dict):
    for k, v in pack.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: ndarray {v.shape} {v.dtype}")
        elif isinstance(v, (pd.DataFrame, pd.Series)):
            print(f"  {k}: {type(v).__name__} {v.shape} cols={list(v.columns)[:15] if hasattr(v,'columns') else ''}")
        elif isinstance(v, dict):
            print(f"  {k}: dict keys={list(v.keys())[:12]}")
        elif isinstance(v, (list, tuple)):
            print(f"  {k}: {type(v).__name__} len={len(v)}")
        else:
            print(f"  {k}: {t} = {str(v)[:80]}")
