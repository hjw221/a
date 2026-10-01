#!/usr/bin/env python
"""把 /home/z/my-project/scripts/ 下属于 xauusd_ml_v2 管线的脚本移植进包内
xauusd_ml_v2/scripts/，并将硬编码绝对路径改为随包可移植的相对路径。
不属于管线的文件(PDF报告产物)不复制。"""
import os
import re
import shutil

SRC = "/home/z/my-project/scripts"
DST = "/home/z/my-project/download/xauusd_ml_v2/scripts"
os.makedirs(DST, exist_ok=True)

PIPE_FILES = [
    "calibrate_barriers.py",      # v1 ATR几何标定 (2.0/1.143)
    "calibrate_barriers_v3.py",   # v3 盈亏比几何标定 (4.0/1.0, 4.0/0.8)
    "calibrate_barriers_v4.py",   # v4 平衡几何标定 (3.0/1.1429 — v3bal/ens在用)
    "calib_v3_result.json",       # v3 标定结果(真实网格输出)
    "calib_v4_result.json",       # v4 标定结果(真实网格输出)
    "analyze_v3.py",              # analysis_v3.json 生成器
    "analyze_v4.py",              # analysis_v4.json 生成器
    "analyze_v5.py",              # analysis_v5.json 生成器(三张牌)
    "streak_analysis.py",         # 连亏统计
    "smoke_features_v3.py",       # v3特征因果性冒烟校验
]

OLD_BASE = "/home/z/my-project/download/xauusd_ml_v2"
# 包内 scripts/xx.py 的包根 = 上两级目录
NEW_BASE = 'os.path.dirname(os.path.dirname(os.path.abspath(__file__)))'

def port(fname):
    src = os.path.join(SRC, fname)
    dst = os.path.join(DST, fname)
    text = open(src, encoding="utf-8").read()
    orig = text

    # 1) sys.path.insert(0, "<OLD_BASE>")
    text = text.replace(
        f'sys.path.insert(0, "{OLD_BASE}")',
        f'_PKG_ROOT = {NEW_BASE}\nsys.path.insert(0, _PKG_ROOT)',
    )
    # 2) BASE = "<OLD_BASE>"
    text = text.replace(
        f'BASE = "{OLD_BASE}"',
        f'BASE = {NEW_BASE}',
    )
    # 3) RES = "<OLD_BASE>/results"  (streak_analysis.py 的写法, 无 sys/os import)
    text = text.replace(
        f'RES = "{OLD_BASE}/results"',
        'import os\nRES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")',
    )
    # 4) OUT = "/home/z/my-project/scripts/calib_*.json" -> 包内 scripts/ 下
    text = re.sub(
        r'OUT = "/home/z/my-project/scripts/(calib_[^"]+)"',
        r'OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "\1")',
        text,
    )
    # 5) sys.path.insert 且无 os import 的文件(smoke_features_v3: import sys, time)
    if "_PKG_ROOT" in text and "import os" not in text and "import sys, os" not in text:
        text = text.replace("import sys, time", "import sys, os, time")

    # 残余绝对路径检查(仅本机路径, 数据路径来自CFG不算)
    leftovers = [ln for ln in text.splitlines() if "/home/z/" in ln]
    shutil.copyfile(src, dst) if text == orig else open(dst, "w", encoding="utf-8").write(text)
    status = "unchanged" if text == orig else "ported"
    return status, leftovers

for f in PIPE_FILES:
    status, leftovers = port(f)
    print(f"{f:32s} {status}" + (f"  LEFTOVER: {leftovers}" if leftovers else ""))
print("done ->", DST)
