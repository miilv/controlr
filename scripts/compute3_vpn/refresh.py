#!/usr/bin/env python3
"""controlr VPN on compute3: fetch the compute3 subscription profile (vpn-retranslator on
vpn.smrtx.ru, sing-box format) and write ~/controlr-vpn/config.json — machine-wide like the dev
box's /etc/sing-box/config.json: tun0 with auto_route, RU/private/torrent direct, everything else
through urltest over the VLESS nodes; plus HTTP/SOCKS on 127.0.0.1:10809/10808 and 18810/18811.
compute3-specific: the router (omniroute) and the subscription host go direct (faster / no
chicken-and-egg); tailscale's addresses are excluded from the tun (its own policy rules win anyway).
Exit 0 = unchanged, 10 = config changed (caller restarts sing-box), 1 = error (old config kept).
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
DIRECT_DOMAINS = [".ru", ".su", ".xn--p1ai", "yandex.ru", "yandex.net", "yandex.com", "yandex.com.tr",
                  "yandexcloud.net", "yandexcloud.com", "yandexcloud.ru",
                  "omniroute.agent-shipyard.com", "smrtx.ru"]       # router direct is faster (net_check)
cfg = {
    "log": {"level": "warn"},
    "dns": {"servers": [{"tag": "cloudflare", "address": "1.1.1.1", "detour": "direct"},
                        {"tag": "cloudflare-secondary", "address": "1.0.0.1", "detour": "direct"}],
            "final": "cloudflare", "strategy": "prefer_ipv4"},
    "inbounds": [
        {"type": "tun", "tag": "tun-in", "interface_name": "tun0",
         "address": ["172.19.0.1/30", "fdfe:dcba:9876::1/126"], "mtu": 1500,
         "auto_route": True, "strict_route": False, "stack": "gvisor",
         "route_exclude_address": ["100.64.0.0/10", "fd7a:115c:a1e0::/48"]},   # tailscale
        {"type": "socks", "tag": "socks", "listen": "127.0.0.1", "listen_port": 10808},
        {"type": "http", "tag": "http", "listen": "127.0.0.1", "listen_port": 10809},
        {"type": "http", "tag": "http-controlr", "listen": "127.0.0.1", "listen_port": 18810},
        {"type": "socks", "tag": "socks-controlr", "listen": "127.0.0.1", "listen_port": 18811},
    ],
    "outbounds": [
        {"type": "urltest", "tag": "auto", "outbounds": tags, "url": "https://cp.cloudflare.com/generate_204",
         "interval": "3m", "tolerance": 50, "idle_timeout": "30m"},
        *nodes,
        {"type": "direct", "tag": "direct"},
        {"type": "block", "tag": "block"},
    ],
    "route": {
        "rules": [
            {"action": "sniff"},
            {"protocol": "dns", "action": "hijack-dns"},
            {"domain_suffix": DIRECT_DOMAINS, "action": "route", "outbound": "direct"},
            {"rule_set": ["geoip-ru"], "action": "route", "outbound": "direct"},
            {"protocol": "bittorrent", "outbound": "direct"},
            {"ip_is_private": True, "outbound": "direct"},
            {"ip_cidr": ["100.64.0.0/10", "fd7a:115c:a1e0::/48"], "outbound": "direct"},
        ],
        "final": "auto",
        "auto_detect_interface": True,
        "rule_set": [{"type": "local", "tag": "geoip-ru", "format": "binary",
                      "path": os.path.join(HOME, "geoip-ru.srs")}],
    },
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
