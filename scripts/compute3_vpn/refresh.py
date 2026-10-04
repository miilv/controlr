#!/usr/bin/env python3
"""controlr VPN on compute3: fetch the compute3 subscription profile (vpn-retranslator on
vpn.smrtx.ru, sing-box format) and write ~/controlr-vpn/config.json — an HTTP/SOCKS proxy on
127.0.0.1:18810/18811 (no tun: only processes that set HTTPS_PROXY use it), urltest over the VLESS
nodes. Exit 0 = unchanged, 10 = config changed (caller restarts sing-box), 1 = error (old config kept).
Secrets: sub_token and config.json are 0600; nothing is printed but node names."""
import json, os, subprocess, sys, tempfile, urllib.request
HOME = os.path.expanduser("~/controlr-vpn")
CFG = os.path.join(HOME, "config.json")
token = open(os.path.join(HOME, "sub_token")).read().strip()
url = f"https://vpn.smrtx.ru/sub/{token}?format=singbox"
try:
    req = urllib.request.Request(url, headers={"User-Agent": "sing-box controlr-compute3"})
    # direct connection (no proxy env): the subscription host is reachable from Skoltech
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    sub = json.loads(opener.open(req, timeout=30).read())
except Exception as e:  # noqa: BLE001
    print(f"fetch failed: {type(e).__name__}", file=sys.stderr); sys.exit(1)
nodes = [o for o in sub.get("outbounds", []) if o.get("type") == "vless"]
if not nodes:
    print("no vless nodes in the subscription", file=sys.stderr); sys.exit(1)
tags = [n["tag"] for n in nodes]
cfg = {
    "log": {"level": "warn"},
    "inbounds": [
        {"type": "http", "tag": "http", "listen": "127.0.0.1", "listen_port": 18810},
        {"type": "socks", "tag": "socks", "listen": "127.0.0.1", "listen_port": 18811},
    ],
    "outbounds": [
        {"type": "urltest", "tag": "auto", "outbounds": tags, "url": "https://cp.cloudflare.com/generate_204",
         "interval": "3m", "tolerance": 50, "idle_timeout": "30m"},
        *nodes,
        {"type": "direct", "tag": "direct"},
    ],
    "route": {"rules": [{"ip_is_private": True, "outbound": "direct"}], "final": "auto"},
}
new = json.dumps(cfg, indent=1, ensure_ascii=False)
old = open(CFG).read() if os.path.exists(CFG) else ""
if new == old:
    print(f"unchanged ({len(tags)} nodes)"); sys.exit(0)
fd, tmp = tempfile.mkstemp(dir=HOME, prefix=".config.", suffix=".json")
with os.fdopen(fd, "w") as f:
    f.write(new)
os.chmod(tmp, 0o600)
chk = subprocess.run([os.path.join(HOME, "bin/sing-box"), "check", "-c", tmp], capture_output=True, text=True)
if chk.returncode != 0:
    os.unlink(tmp); print("sing-box check failed:", chk.stderr[:300], file=sys.stderr); sys.exit(1)
os.replace(tmp, CFG)
print(f"updated: {len(tags)} nodes: {', '.join(tags)}"); sys.exit(10)
