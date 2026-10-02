#!/usr/bin/env python3
"""ntfy 双向代理 — 服务器端命令执行代理 (仅标准库)。

轮询 CMD_TOPIC, 白名单命令执行, 结果发回 RES_TOPIC。
白名单: ping / status / tail / cat / ls / get / df / ps / gpu
用法: nohup python3 agent.py >> logs/agent.log 2>&1 &
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

# ---- ntfy 通道 (topic 名即鉴权) ----
CMD_TOPIC = "xauusd-qv7m2zk9-cmd"
RES_TOPIC = "xauusd-qv7m2zk9-res"
BASE_URL = "https://ntfy.sh"
BASE = "/root/rivermind-fs/xauusd"          # 工作区根
LOGS = os.path.join(BASE, "logs")

STATE = "/tmp/agent_state.json"


def ntfy_publish(topic, message, title=None):
    req = urllib.request.Request(
        f"{BASE_URL}/{topic}", data=message.encode("utf-8"),
        headers={"Title": title or "srv", "Priority": "default"})
    try:
        urllib.request.urlopen(req, timeout=30).read()
    except Exception as e:                      # noqa: BLE001
        print("publish fail:", e, flush=True)


def ntfy_upload(topic, path, message):
    """curl -T 附件上传 (urllib multipart 繁琐, 直接用 curl)。"""
    subprocess.run(["curl", "-sS", "--max-time", "600", "-T", path,
                    f"{BASE_URL}/{topic}", "-d", message],
                   check=False, timeout=620)


def sh(cmd, timeout=300):
    try:
        r = subprocess.run(["bash", "-c", cmd], capture_output=True,
                           text=True, timeout=timeout)
        out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
        return out.strip() or "(empty)"
    except subprocess.TimeoutExpired:
        return f"(timeout {timeout}s)"
    except Exception as e:                      # noqa: BLE001
        return f"(error: {e})"


ALLOWED_DIRS = [BASE, "/tmp"]


def safe_path(rel):
    p = os.path.realpath(os.path.join(BASE, rel)) if not rel.startswith("/") \
        else os.path.realpath(rel)
    if not any(p.startswith(d) for d in ALLOWED_DIRS):
        return None
    return p


def handle(msg):
    m = msg.strip()
    low = m.lower()
    if low == "ping":
        return sh("uptime && echo OK")
    if low == "status":
        return sh(
            "echo '== procs =='; ps aux | grep -E 'run_all|run_m1|phase' | grep -v grep; "
            f"echo '== last log lines =='; for f in {LOGS}/*.log; do "
            "echo \"--- $f\"; tail -3 \"$f\" 2>/dev/null; done")
    if low == "ps":
        return sh("ps aux | grep -E 'python|cloudflared' | grep -v grep")
    if low in ("df",):
        return sh("df -h /root /tmp; free -g | head -2")
    if low == "gpu":
        return sh("nvidia-smi 2>&1 | head -15 || echo no-gpu")
    if low.startswith("tail "):
        parts = m.split()
        rel = parts[1]
        n = parts[2] if len(parts) > 2 else "40"
        p = safe_path(rel)
        if not p or not os.path.exists(p):
            return f"(bad path: {rel})"
        return sh(f"tail -n {int(n)} '{p}'")
    if low.startswith("ls "):
        p = safe_path(m.split(maxsplit=1)[1])
        if not p:
            return "(bad path)"
        return sh(f"ls -la '{p}' | head -60")
    if low.startswith("cat "):
        p = safe_path(m.split(maxsplit=1)[1])
        if not p or not os.path.exists(p):
            return f"(bad path)"
        if os.path.getsize(p) > 200_000:
            return "(file >200KB, use get)"
        return sh(f"cat '{p}' | head -c 150000")
    if low.startswith("get "):
        rel = m.split(maxsplit=1)[1]
        p = safe_path(rel)
        if not p or not os.path.exists(p):
            return f"(bad path)"
        ntfy_upload(RES_TOPIC, p, f"file: {rel}")
        return f"(uploaded {rel})"
    return "(unknown command; allowed: ping status ps df gpu tail/ls/cat/get)"


def poll():
    ts = 0
    if os.path.exists(STATE):
        try:
            ts = json.load(open(STATE)).get("ts", 0)
        except Exception:                        # noqa: BLE001
            ts = 0
    url = (f"{BASE_URL}/{CMD_TOPIC}/json?poll=1"
           + (f"&since={ts}" if ts else "&since=all"))
    try:
        raw = urllib.request.urlopen(url, timeout=120).read().decode()
    except Exception as e:                      # noqa: BLE001
        print("poll fail:", e, flush=True)
        time.sleep(20)
        return
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except Exception:                        # noqa: BLE001
            continue
        if d.get("event") != "message":
            continue
        m = d.get("message", "")
        t = int(d.get("time", 0))
        if t <= ts or not m:
            continue
        ts = max(ts, t)
        print(f"[cmd] {m}", flush=True)
        out = handle(m)
        if out and not out.startswith("(uploaded"):
            # 分片: ntfy 消息上限 ~4KB
            for i in range(0, min(len(out), 30000), 3500):
                chunk = out[i:i + 3500]
                ntfy_publish(RES_TOPIC, chunk, title=f"reply {i//3500 + 1}")
        json.dump({"ts": ts}, open(STATE, "w"))


if __name__ == "__main__":
    ntfy_publish(RES_TOPIC, f"agent online pid={os.getpid()}", title="agent")
    while True:
        poll()
        time.sleep(5)
