#!/usr/bin/env python3
"""relay-upload.py — 三段式中继上传: 沙箱 →(SFTP)→ 跳板 →(原生ssh)→ 目标机.
原理: paramiko 双跳隧道被限速时, 跳板上原生 ssh 单跳 TCP 可能不受影响.
"""
import sys, time
sys.path.insert(0, '/home/z/my-project/recovery-tools')
import paramiko
from jump_ssh_lib import TGT_HOST, TGT_PORT, TGT_USER, TGT_PASS

JUMP_TMP = '/tmp/relay_upload.bin'

def main():
    local = sys.argv[1]
    remote = sys.argv[2]
    test_only = len(sys.argv) > 3 and sys.argv[3] == '--test2m'

    # ---- 段1: 沙箱 → 跳板 ----
    jump = paramiko.SSHClient()
    jump.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    jump.connect('35.208.137.216', port=22, username='root', password='Abc147258@', timeout=25)
    sftp = jump.open_sftp()
    size = 2 * 1024 * 1024 if test_only else None
    t0 = time.time()
    if test_only:
        data = open(local, 'rb').read(2 * 1024 * 1024)
        with sftp.open(JUMP_TMP, 'wb') as f:
            f.write(data)
        size = len(data)
    else:
        sftp.put(local, JUMP_TMP)
        size = sftp.stat(JUMP_TMP).st_size
    dt = time.time() - t0
    print(f"[1/3] 沙箱→跳板: {size/1048576:.1f}MB in {dt:.0f}s = {size/1048576/max(dt,0.1):.2f}MB/s")
    sftp.close()

    # ---- 段2: 跳板上准备 askpass ----
    _, o, _ = jump.exec_command(
        f"printf '#!/bin/sh\\necho {TGT_PASS}\\n' > /tmp/.ap.sh && chmod +x /tmp/.ap.sh && ls -la {JUMP_TMP}")
    print("[2/3] 跳板就绪:", o.read().decode().strip().split('\n')[-1])

    # ---- 段3: 跳板 →(原生ssh)→ 目标机 ----
    stream_cmd = (
        f"SSH_ASKPASS=/tmp/.ap.sh SSH_ASKPASS_REQUIRE=force setsid "
        f"ssh -p {TGT_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
        f"-o ConnectTimeout=20 {TGT_USER}@{TGT_HOST} 'cat > {remote}' < {JUMP_TMP}"
    )
    t1 = time.time()
    _, o, e = jump.exec_command(f"time ({stream_cmd}) 2>&1; echo RC=$?", timeout=600)
    out = o.read().decode()
    dt2 = time.time() - t1
    print(f"[3/3] 跳板→目标: {size/1048576:.1f}MB in {dt2:.0f}s = {size/1048576/max(dt2,0.1):.2f}MB/s")
    print(out[-500:] if len(out) > 500 else out)

    # 清理跳板临时文件
    jump.exec_command(f"rm -f {JUMP_TMP} /tmp/.ap.sh")
    jump.close()

if __name__ == '__main__':
    main()
