"""
make_ea_files.py — 从 wf_manifest.json / prod_manifest.json 生成EA直接可读的无头CSV:
  wf/wf_index.csv            : fold,oos_month,thr_long,thr_short
  wf/fold_XX/members.csv     : direction,kind,file,weight
  wf/fold_XX/selftest.csv    : expected_pl,expected_ps,f1..f34 (36列)
  prod/index.csv             : variant,thr_long,thr_short,train_window
  prod/members.csv / prod/selftest.csv 同构
另: 拷贝Python走查交易日志 + 生成特征参考CSV(末120根M5, 供EA特征对拍)。
"""
import os, sys, json, shutil
import numpy as np

PKG = "/home/z/my-project/download/xauusd_ml_v2"
WF = os.path.join(PKG, "mt5_package", "models", "wf")
PROD = os.path.join(PKG, "mt5_package", "models", "prod")
REF = os.path.join(PKG, "mt5_package", "reference")
os.makedirs(REF, exist_ok=True)

CFG = json.load(open(os.path.join(PKG, "production_models", "trading_config.json")))
FEATS = CFG["features_list"]
assert len(FEATS) == 34


def wrow(fp, vals):
    with open(fp, "a") as f:
        f.write(",".join(str(v) for v in vals) + "\r\n")  # v1.01: CRLF for MQL5


def num(x):
    return f"{x:.10g}"


# ---------- WF ----------
man = json.load(open(os.path.join(WF, "wf_manifest.json")))
idx_fp = os.path.join(WF, "wf_index.csv")
open(idx_fp, "w").close()
for fd in man["folds"]:
    wrow(idx_fp, [fd["fold"], fd["oos_month"], num(fd["thr_long"]), num(fd["thr_short"])])
    fdir = os.path.join(WF, fd["folder"])
    mfp = os.path.join(fdir, "members.csv")
    open(mfp, "w").close()
    for m in fd["members"]:
        wrow(mfp, [m["direction"], m["kind"], m["file"], num(m["weight"])])
    st = fd["self_test"]
    sfp = os.path.join(fdir, "selftest.csv")
    open(sfp, "w").close()
    wrow(sfp, [num(st["expected_pl"]), num(st["expected_ps"])] + [num(v) for v in st["features"]])
print(f"[ea-files] wf_index.csv: {len(man['folds'])}折")

# ---------- PROD ----------
pm = json.load(open(os.path.join(PROD, "prod_manifest.json")))
wrow(os.path.join(PROD, "index.csv"), [pm["variant"], num(pm["thr_long"]), num(pm["thr_short"]), pm["train_window"]])
open(os.path.join(PROD, "members.csv"), "w").close()
for m in pm["members"]:
    wrow(os.path.join(PROD, "members.csv"), [m["direction"], m["kind"], m["file"], num(m["weight"])])
st = pm["self_test"]
open(os.path.join(PROD, "selftest.csv"), "w").close()
wrow(os.path.join(PROD, "selftest.csv"), [num(st["expected_pl"]), num(st["expected_ps"])] + [num(v) for v in st["features"]])
print(f"[ea-files] prod: {len(pm['members'])}成员")

# ---------- 参考物 ----------
shutil.copy(os.path.join(PKG, "results", "trades_v3bal_ens.csv"),
            os.path.join(REF, "trades_v3bal_ens_python.csv"))

import pickle
with open(os.path.join(PKG, "cache", "pack_v3bal.pkl"), "rb") as f:
    pack = pickle.load(f)
F = pack["F"]
assert list(F.columns) == FEATS, f"特征顺序不一致!\n{list(F.columns)}\n{FEATS}"
tail = F.tail(120)
fp = os.path.join(REF, "features_reference_python.csv")
with open(fp, "w") as f:
    f.write("time," + ",".join(FEATS) + "\r\n")
    for ts, row in tail.iterrows():
        vals = [ts.strftime("%Y.%m.%d %H:%M")] + [num(v) for v in row.to_numpy()]
        f.write(",".join(vals) + "\r\n")
print(f"[ea-files] 参考特征CSV: {len(tail)}行 ({tail.index[0]} ~ {tail.index[-1]})")
print("[ea-files] 完成")
