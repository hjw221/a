#!/usr/bin/env python3.13
"""Test GitHub SSH auth via paramiko (no openssh-client in this env).
Tries github.com:22 first, then ssh.github.com:443 (firewall fallback).
"""
import sys
import paramiko

KEY = "/home/z/.ssh/id_ed25519"

for host, port, label in [("github.com", 22, "direct"), ("ssh.github.com", 443, "over-443")]:
    try:
        cli = paramiko.SSHClient()
        cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        cli.connect(host, port=port, username="git", key_filename=KEY,
                    look_for_keys=False, allow_agent=False, timeout=15)
        stdin, stdout, stderr = cli.exec_command("", timeout=15)
        banner = cli.get_transport().remote_version
        # GitHub answers with a greeting on stderr channel for shell-less accounts
        err = stderr.channel.recv_exit_status if False else None
        cli.close()
        print(f"[{label}] {host}:{port} CONNECTED as git, remote={banner}")
        print("AUTH_OK")
        sys.exit(0)
    except Exception as e:
        print(f"[{label}] {host}:{port} FAILED: {type(e).__name__}: {e}")
        try:
            cli.close()
        except Exception:
            pass

print("AUTH_FAIL")
sys.exit(1)
