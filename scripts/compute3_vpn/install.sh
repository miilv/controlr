#!/usr/bin/env bash
# Install / re-install controlr's VPN proxy on compute3 (run FROM the dev box). No tun, no system
# routing change: sing-box serves an HTTP proxy on compute3:127.0.0.1:18810 (SOCKS 18811) that only
# processes with HTTPS_PROXY use. Nodes come from compute3's own subscription profile on the dev box's
# vpn-retranslator (https://vpn.smrtx.ru/sub/<token>; turn compute3 off there). RUNBOOK §compute3 network.
#   scripts/compute3_vpn/install.sh <file with the compute3 profile token>
set -euo pipefail
HOST="${CONTROLR_HOST:-compute3}"
TOKEN_FILE="${1:?usage: install.sh <token file>}"
DIR="$(cd "$(dirname "$0")" && pwd)"
ssh "$HOST" 'mkdir -p ~/controlr-vpn/bin && chmod 700 ~/controlr-vpn'
scp -q "$(command -v sing-box)" "$HOST:controlr-vpn/bin/sing-box"      # same version as the dev box
scp -q "$TOKEN_FILE" "$HOST:controlr-vpn/sub_token"
scp -q "$DIR/refresh.py" "$HOST:controlr-vpn/refresh.py"
scp -q "$DIR"/controlr-vpn.service "$DIR"/controlr-vpn-refresh.service "$DIR"/controlr-vpn-refresh.timer "$HOST:controlr-vpn/"
ssh "$HOST" 'chmod 600 ~/controlr-vpn/sub_token && chmod 700 ~/controlr-vpn/refresh.py && ~/controlr-vpn/refresh.py; \
  sudo cp ~/controlr-vpn/controlr-vpn*.service ~/controlr-vpn/controlr-vpn-refresh.timer /etc/systemd/system/ && \
  sudo systemctl daemon-reload && sudo systemctl enable --now controlr-vpn.service controlr-vpn-refresh.timer && \
  sudo systemctl restart controlr-vpn && sleep 2 && systemctl is-active controlr-vpn'
