#!/usr/bin/env bash
# Install / re-install controlr's VPN on compute3 (run FROM the dev box): sing-box, machine-wide like
# the dev box (tun0 + auto_route; RU / private / torrent / the router direct), plus HTTP/SOCKS proxies
# on 127.0.0.1:10809/10808 and 18810/18811. Nodes come from compute3's own subscription profile on the dev box's
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
scp -q /etc/sing-box/rules/geoip-ru.srs "$HOST:controlr-vpn/geoip-ru.srs"   # the dev box's RU rule set
scp -q "$DIR"/controlr-vpn.service "$DIR"/controlr-vpn-refresh.service "$DIR"/controlr-vpn-refresh.timer "$HOST:controlr-vpn/"
ssh "$HOST" 'chmod 600 ~/controlr-vpn/sub_token && chmod 700 ~/controlr-vpn/refresh.py && ~/controlr-vpn/refresh.py; \
  sudo cp ~/controlr-vpn/controlr-vpn*.service ~/controlr-vpn/controlr-vpn-refresh.timer /etc/systemd/system/ && \
  sudo systemctl daemon-reload && sudo systemctl enable --now controlr-vpn.service controlr-vpn-refresh.timer && \
  sudo systemctl restart controlr-vpn && sleep 2 && systemctl is-active controlr-vpn'
