import paramiko
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect("github.com", port=22, username="git", key_filename="/home/z/.ssh/id_ed25519",
            look_for_keys=False, allow_agent=False, timeout=15)
ch = cli.get_transport().open_session()
ch.settimeout(10)
data = b""
try:
    while True:
        b = ch.recv(4096)
        if not b:
            break
        data += b
except Exception:
    pass
print(data.decode(errors="replace"))
cli.close()
