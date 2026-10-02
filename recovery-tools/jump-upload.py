#!/usr/bin/env python3
"""jump-upload.py — 经跳板 SFTP 上传文件到目标服务器."""
import sys, time
sys.path.insert(0, '/home/z/my-project/recovery-tools')
from jump_ssh_lib import connect  # noqa

def main():
    local, remote = sys.argv[1], sys.argv[2]
    jump = target = None
    try:
        jump, target = connect()
        sftp = target.open_sftp()
        size = sftp.stat(remote).st_size if _exists(sftp, remote) else 0
        t0 = time.time()
        last = [0]

        def cb(transferred, total):
            if transferred - last[0] >= 4 * 1024 * 1024 or transferred == total:
                last[0] = transferred
                rate = transferred / max(time.time() - t0, 0.1) / 1024 / 1024
                print(f"  {transferred/1048576:.1f}/{total/1048576:.1f} MB ({rate:.2f} MB/s)", flush=True)

        print(f"上传 {local} -> {remote}")
        sftp.put(local, remote, callback=cb)
        dt = time.time() - t0
        print(f"完成: {dt:.0f}s, 平均 {size/1048576/max(dt,0.1):.2f} MB/s" if size else f"完成: {dt:.0f}s")
        sftp.close()
    finally:
        for c in (target, jump):
            if c:
                try: c.close()
                except Exception: pass

def _exists(sftp, path):
    try:
        sftp.stat(path)
        return True
    except IOError:
        return False

if __name__ == '__main__':
    main()
