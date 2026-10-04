#!/usr/bin/env bash
# Network check for the LLM path (no LLM calls, no spend). Run on compute3 (or anywhere with .env):
#   bash scripts/net_check.sh                      # this machine's own route (VPN)
#   HTTPS_PROXY=http://127.0.0.1:18809 bash scripts/net_check.sh   # through the ssh tunnel
# What matters (journal/2026-10-04-luna-latency.md, 2026-10-04-sonnet-latency.md): every router turn
# re-uploads the whole transcript (0.06 MB at turn 0 → ~0.75 MB at turn 16), so UPLOAD bandwidth to
# the router dominates; latency (RTT) matters much less. Healthy: 740 KB upload < 0.75 s (≥ 1 MB/s),
# router first byte < 0.3 s. On 2026-10-04 compute3's VPN gave 42–138 KB/s (5–17 s per 740 KB).
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
: "${OMNIROUTE_BASE_URL:?OMNIROUTE_BASE_URL not set}"
blob=$(mktemp); trap 'rm -f "$blob"' EXIT
python3 - "$blob" <<'PY'
import base64, json, os, sys
s = base64.b64encode(os.urandom(555_000)).decode()           # ~740 KB of base64, like a turn-16 request
json.dump({"model": "cx/does-not-exist-net-check", "messages": [{"role": "user", "content": s}]}, open(sys.argv[1], "w"))
PY
fmt='%{http_code} total %{time_total}s  connect %{time_connect}s  tls %{time_appconnect}s  first-byte %{time_starttransfer}s  upload %{speed_upload} B/s'
echo "egress: $(curl -s -m 10 https://ipinfo.io/json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("city"), d.get("country"), d.get("org"))' 2>/dev/null)"
for i in 1 2 3; do
  echo -n "router 740 KB upload   : "
  curl -s -o /dev/null -m 60 -w "$fmt\n" -H "Authorization: Bearer ${OMNIROUTE_API_KEY:-x}" \
       -H 'content-type: application/json' --data-binary @"$blob" "$OMNIROUTE_BASE_URL/chat/completions"
done
for i in 1 2; do
  echo -n "router GET /models     : "
  curl -s -o /dev/null -m 20 -w "$fmt\n" -H "Authorization: Bearer ${OMNIROUTE_API_KEY:-x}" "$OMNIROUTE_BASE_URL/models"
done
for host in https://chatgpt.com/backend-api/codex/responses https://api.anthropic.com/v1/messages; do
  echo -n "$(echo "$host" | cut -d/ -f3) (no auth): "
  curl -s -o /dev/null -m 20 -w "$fmt\n" -X POST -H 'content-type: application/json' -d '{}' "$host" || echo "unreachable"
done
