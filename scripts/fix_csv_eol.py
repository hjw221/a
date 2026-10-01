#!/usr/bin/env python3
"""把 mt5_package 内全部 .csv 的行尾统一为 CRLF(\r\n)。
原因: Python 生成器写出 \n-only 行尾, MQL5 FileReadString(CSV) 按 \r\n 行终止设计,
在 MT5 build 5833 实测把 24 折 wf_index.csv 解析成 1 条空记录 -> 全部月份 HOLD。
幂等: 已是 CRLF 的文件跳过不重写。"""
import glob
import os

PKG = '/home/z/my-project/download/xauusd_ml_v2/mt5_package'

converted, skipped = [], []
for fp in glob.glob(os.path.join(PKG, '**', '*.csv'), recursive=True):
    raw = open(fp, 'rb').read()
    if b'\r\n' in raw and b'\n' not in raw.replace(b'\r\n', b''):
        skipped.append((fp, len(raw)))          # 已是纯CRLF
        continue
    txt = raw.replace(b'\r\n', b'\n').replace(b'\r', b'\n')   # 归一到LF
    out = txt.replace(b'\n', b'\r\n')                          # 统一转CRLF
    open(fp, 'wb').write(out)
    converted.append((os.path.relpath(fp, PKG), len(raw), len(out)))

print(f"converted: {len(converted)} files")
for rel, a, b in converted:
    print(f"  {rel}  {a}->{b} bytes")
print(f"already CRLF (skipped): {len(skipped)} files")
# 抽查验证
for probe in ['models/wf/wf_index.csv', 'models/wf/fold_00/members.csv',
              'models/prod/index.csv', 'reference/trades_v3bal_ens_python.csv']:
    fp = os.path.join(PKG, probe)
    raw = open(fp, 'rb').read()
    n_lf = raw.count(b'\n')
    n_crlf = raw.count(b'\r\n')
    print(f"[verify] {probe}: LF={n_lf} CRLF={n_crlf} -> {'OK' if n_lf == n_crlf else 'FAIL'}")
