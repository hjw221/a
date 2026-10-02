#!/usr/bin/env python3
"""stream-upload.py — 经跳板用 exec 通道流式上传(SFTP卡死时的替代路径, ~22KB/s).
用法: python3 stream-upload.py 本地文件 远程路径
"""
import sys, time, hashlib
sys.path.insert(0, '/home/z/my-project/recovery-tools')
from jump_ssh_lib import connect

CHUNK = 64 * 1024

def main():
    local, remote = sys.argv[1], sys.argv[2]
    data = open(local, 'rb').read()
    md5 = hashlib.md5(data).hexdigest()
    print(f"流式上传 {local} ({len(data)} bytes, md5 {md5}) -> {remote}")
    jump = target = None
    try:
        jump, target = connect()
        stdin, stdout, stderr = target.exec_command(
            f'cat > {remote}; md5sum {remote}')
        t0 = time.time()
        for i in range(0, len(data), CHUNK):
            stdin.write(data[i:i+CHUNK])
            stdin.flush()
        stdin.channel.shutdown_write()
        out = stdout.read().decode()
        dt = time.time() - t0
        print(f"完成: {dt:.0f}s ({len(data)/1024/max(dt,0.1):.1f} KB/s)")
        print(out.strip())
        ok = md5 in out
        print("校验:", "OK" if ok else "MISMATCH!")
        sys.exit(0 if ok else 1)
    finally:
        for c in (target, jump):
            if c:
                try: c.close()
                except Exception: pass

if __name__ == '__main__':
    main()
