#!/usr/bin/env python3
"""git-ssh paramiko 包装器: 让无 openssh-client 的沙箱能 git push/pull over SSH.
用法: GIT_SSH_COMMAND="python3 git-ssh.py" git ...
git 会调用: git-ssh.py [-p port] [user@]host command...
"""
import sys, os, time, threading
import paramiko

KEY_PATH = os.path.expanduser("~/.ssh/id_ed25519_hjw221")


def parse_args(argv):
    port, user, host, cmd = 22, "git", None, ""
    i = 0
    rest = []
    while i < len(argv):
        a = argv[i]
        if a == "-p":
            port = int(argv[i + 1]); i += 2; continue
        if a.startswith("-o") or a.startswith("-i") or a.startswith("-T") or a.startswith("-W"):
            if a in ("-o", "-i") and i + 1 < len(argv):
                i += 2
            else:
                i += 1
            continue
        rest.append(a); i += 1
    if rest:
        host = rest[0]
        if "@" in host:
            user, host = host.split("@", 1)
        if len(rest) > 1:
            cmd = " ".join(rest[1:])
    return port, user, host, cmd


def main():
    port, user, host, cmd = parse_args(sys.argv[1:])
    if not host:
        sys.stderr.write("git-ssh: no host\n"); sys.exit(1)
    key = paramiko.Ed25519Key.from_private_key_file(KEY_PATH)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=user, pkey=key,
                   timeout=30, banner_timeout=30, auth_timeout=30)

    chan = client.get_transport().open_session(timeout=30)
    if cmd:
        chan.exec_command(cmd)
    else:
        chan.exec_command("")  # 仅认证测试

    stdin_fd = sys.stdin.fileno()
    import fcntl
    old_fl = fcntl.fcntl(stdin_fd, fcntl.F_GETFL)
    fcntl.fcntl(stdin_fd, fcntl.F_SETFL, old_fl | os.O_NONBLOCK)

    eof_sent = False
    exit_code = None
    while True:
        # 远端 -> 本地 stdout/stderr
        if chan.recv_ready():
            data = chan.recv(65536)
            if data:
                sys.stdout.buffer.write(data); sys.stdout.buffer.flush()
        if chan.recv_stderr_ready():
            data = chan.recv_stderr(65536)
            if data:
                sys.stderr.buffer.write(data); sys.stderr.buffer.flush()
        # 本地 stdin -> 远端
        if not eof_sent:
            try:
                b = os.read(stdin_fd, 65536)
                if b:
                    chan.sendall(b)
                else:
                    chan.shutdown_write(); eof_sent = True
            except BlockingIOError:
                pass
            except OSError:
                chan.shutdown_write(); eof_sent = True
        # 退出判断
        if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
            # 排空残余
            while chan.recv_ready():
                d = chan.recv(65536)
                if d: sys.stdout.buffer.write(d)
            while chan.recv_stderr_ready():
                d = chan.recv_stderr(65536)
                if d: sys.stderr.buffer.write(d)
            sys.stdout.buffer.flush(); sys.stderr.buffer.flush()
            exit_code = chan.recv_exit_status()
            break
        if chan.closed and chan.eof_received and not chan.recv_ready():
            exit_code = chan.recv_exit_status() if chan.exit_status_ready() else 0
            break
        time.sleep(0.002)
    fcntl.fcntl(stdin_fd, fcntl.F_SETFL, old_fl)
    client.close()
    sys.exit(exit_code or 0)


if __name__ == "__main__":
    main()
