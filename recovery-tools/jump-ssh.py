#!/usr/bin/env python3
"""jump-ssh.py — 经跳板在目标服务器执行命令.
用法: python3 jump-ssh.py [--timeout N] "命令"
环境变量 TGT_PORT/TGT_PASS 可切机器 (默认新机30133).
"""
import sys, os
sys.path.insert(0, '/home/z/my-project/recovery-tools')
from jump_ssh_lib import connect, run

def main():
    args = sys.argv[1:]
    timeout = 60
    if args and args[0] == "--timeout":
        timeout = int(args[1]); args = args[2:]
    if not args:
        print("用法: jump-ssh.py [--timeout N] '命令'"); sys.exit(2)
    cmd = args[0]
    jump = target = None
    try:
        jump, target = connect()
        out, err, rc = run(target, cmd, timeout=timeout)
        sys.stdout.write(out)
        if err.strip():
            sys.stdout.write("\n[stderr]\n" + err)
        sys.exit(rc)
    finally:
        for c in (target, jump):
            if c:
                try: c.close()
                except Exception: pass

if __name__ == '__main__':
    main()
