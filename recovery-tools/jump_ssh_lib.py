#!/usr/bin/env python3
"""jump_ssh_lib.py — 跳板级联 SSH 连接库（沙箱→GCP跳板→gpuhome目标容器）.

环境变量可覆盖默认值:
  JUMP_HOST / JUMP_PORT / JUMP_USER / JUMP_PASS
  TGT_HOST / TGT_PORT / TGT_USER / TGT_PASS
"""
import os
import paramiko

JUMP_HOST = os.environ.get("JUMP_HOST", "35.208.137.216")
JUMP_PORT = int(os.environ.get("JUMP_PORT", "22"))
JUMP_USER = os.environ.get("JUMP_USER", "root")
JUMP_PASS = os.environ.get("JUMP_PASS", "Abc147258@")

# 默认 = 新目标机(30核80G)；旧机器用 TGT_PORT=30181 TGT_PASS=03g2rfg3 覆盖
TGT_HOST = os.environ.get("TGT_HOST", "sx01-ssh.gpuhome.cc")
TGT_PORT = int(os.environ.get("TGT_PORT", "30133"))
TGT_USER = os.environ.get("TGT_USER", "root")
TGT_PASS = os.environ.get("TGT_PASS", "ryq94rge")


def connect(tgt_host=None, tgt_port=None, tgt_user=None, tgt_pass=None, timeout=25):
    """返回 (jump_client, target_client)。失败抛异常。"""
    th = tgt_host or TGT_HOST
    tp = int(tgt_port or TGT_PORT)
    tu = tgt_user or TGT_USER
    tpw = tgt_pass or TGT_PASS

    jump = paramiko.SSHClient()
    jump.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    jump.connect(JUMP_HOST, port=JUMP_PORT, username=JUMP_USER, password=JUMP_PASS,
                 timeout=timeout, banner_timeout=timeout, auth_timeout=timeout)

    # direct-tcpip 通道经跳板到达目标
    chan = jump.get_transport().open_channel(
        "direct-tcpip", (th, tp), ("127.0.0.1", 0), timeout=timeout)

    target = paramiko.SSHClient()
    target.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    target.connect(th, port=tp, username=tu, password=tpw, sock=chan,
                   timeout=timeout, banner_timeout=timeout, auth_timeout=timeout,
                   look_for_keys=False, allow_agent=False)
    return jump, target


def run(target, cmd, timeout=60):
    """在目标机执行命令，返回 (stdout, stderr, exit_code)。"""
    stdin, stdout, stderr = target.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    return out, err, exit_code
