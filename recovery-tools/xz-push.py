#!/usr/bin/env python3
"""xz-push.py — 八路并行 exec 流式推送 xz 分块到目标机.
每块独立文件, 失败可单独重推(块级断点续传).
"""
import sys, os, time, threading, hashlib
sys.path.insert(0, '/home/z/my-project/recovery-tools')
from jump_ssh_lib import connect

PART_DIR = '/tmp'
REMOTE_DIR = '/tmp/xzparts'
CHUNK = 64 * 1024

def md5(b): return hashlib.md5(b).hexdigest()

def stream_part(idx):
    """单线程: 推一个分块, 服务端算 md5 回传."""
    local = f'{PART_DIR}/xzpart_{idx:02d}'
    data = open(local, 'rb').read()
    expect = md5(data)
    try:
        t0 = time.time()
        jump, target = connect()
        stdin, stdout, stderr = target.exec_command(
            f'mkdir -p {REMOTE_DIR}; cat > {REMOTE_DIR}/xzpart_{idx:02d}; md5sum {REMOTE_DIR}/xzpart_{idx:02d}; stat -c %s {REMOTE_DIR}/xzpart_{idx:02d}')
        for j in range(0, len(data), CHUNK):
            stdin.write(data[j:j+CHUNK])
            stdin.flush()
        stdin.channel.shutdown_write()
        out = stdout.read().decode()
        jump.close(); target.close()
        dt = time.time() - t0
        ok = expect in out
        size_ok = str(len(data)) in out
        print(f'  块{idx:02d}: {len(data)/1024:.0f}KB {dt:.0f}s {"OK" if (ok and size_ok) else "FAIL"}', flush=True)
        return ok and size_ok
    except Exception as e:
        print(f'  块{idx:02d}: ERR {e}', flush=True)
        return False

def main():
    parts = sorted(f for f in os.listdir(PART_DIR) if f.startswith('xzpart_'))
    n = len(parts)
    print(f'共 {n} 块, 8 路并行推送...')
    t0 = time.time()
    results = [False] * n
    threads = []
    # 8路同时
    for i in range(n):
        t = threading.Thread(target=lambda i=i: results.__setitem__(i, stream_part(i)))
        t.start(); threads.append(t)
    for t in threads: t.join()
    wall = time.time() - t0
    ok = sum(results)
    total_mb = sum(os.path.getsize(f'{PART_DIR}/p') for p in parts) / 1048576 if False else 17.25
    print(f'完成: {ok}/{n} 块, wall {wall:.0f}s')
    if ok < n:
        print('失败块:', [i for i, r in enumerate(results) if not r])
        print('重跑本脚本前先删掉失败块: python3 - <<P ... 或直接再跑(覆盖式)')
    sys.exit(0 if ok == n else 1)

if __name__ == '__main__':
    main()
