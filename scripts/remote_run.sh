#!/usr/bin/env bash
# Run a controlr command on compute3 and pull new run dirs back to ./runs.
#   scripts/remote_run.sh run -c configs/sim_waffle.yaml --episodes 3
#   scripts/remote_run.sh bench-latency -c configs/bench/latency.yaml
# If the command's config uses the isaac backend, the Isaac server is started
# first (headless, under Isaac's own python) unless one is already listening,
# and stopped afterwards if we started it. Deploy first: scripts/deploy.sh
set -euo pipefail
HOST="${CONTROLR_HOST:-compute3}"
REMOTE_DIR="${CONTROLR_REMOTE_DIR:-controlr}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ $# -ge 1 ] || { echo "usage: $0 <controlr subcommand> [args...]" >&2; exit 2; }

# Isaac server: launcher + port owned by controlr.robot.isaac (scripts/isaac_server.sh,
# protocol.DEFAULT_PORT); override here if a run uses a non-default port.
ISAAC_PORT="${ISAAC_PORT:-7801}"

# Does the command need Isaac? The config (with --set overrides) is resolved remotely.
needs_isaac=0
case "$1" in run|sweep) needs_isaac=1 ;; esac

ARGS=$(printf ' %q' "$@")
rc=0
ssh "$HOST" bash -s <<REMOTE || rc=$?
set -euo pipefail
cd "\$HOME/$REMOTE_DIR"
export PATH="\$HOME/.local/bin:\$PATH"
set -a; [ -f .env ] && . ./.env; set +a     # e.g. CONTROLR_ISAAC_AUTHKEY for server + client
backend=none
if [ $needs_isaac = 1 ]; then
  backend=\$(.venv/bin/python - $ARGS <<'PY'
import sys
from controlr.cli import build_parser
from controlr.config import load_config
a = build_parser().parse_args(sys.argv[1:])
if a.cmd == "run":
    print(load_config(a.config, a.sets).robot.backend)
else:
    from controlr.bench.sweep import load_sweep
    s = load_sweep(a.sweep)
    print(load_config(s.config, s.set + a.sets).robot.backend)
PY
)
fi
started=0
# (no TCP probe: an unauthenticated connect would hit the server's HMAC handshake)
if [ "\$backend" = isaac ] && ! ss -ltnH "sport = :$ISAAC_PORT" | grep -q .; then
  mkdir -p runs
  echo "starting Isaac server (port $ISAAC_PORT) -> runs/isaac_server.log"
  export CONTROLR_ISAAC_READY_FILE="\$PWD/runs/.isaac_ready"
  rm -f "\$CONTROLR_ISAAC_READY_FILE"
  setsid nohup bash scripts/isaac_server.sh --port $ISAAC_PORT > runs/isaac_server.log 2>&1 &
  srv=\$!
  started=1
  for i in \$(seq 1 300); do
    [ -f "\$CONTROLR_ISAAC_READY_FILE" ] && break
    kill -0 \$srv 2>/dev/null || { echo "Isaac server died, see runs/isaac_server.log"; tail -20 runs/isaac_server.log; exit 1; }
    sleep 2
  done
  [ -f "\$CONTROLR_ISAAC_READY_FILE" ] || { echo "Isaac server not ready after 600 s"; kill -- -\$srv; exit 1; }
fi
rc=0
.venv/bin/controlr $ARGS || rc=\$?
# setsid -> own process group: stops python.sh and its Isaac child, nothing else
if [ \$started = 1 ]; then kill -- -\$srv 2>/dev/null || true; fi
exit \$rc
REMOTE
mkdir -p "$ROOT/runs"
if ssh "$HOST" test -d "$REMOTE_DIR/runs"; then
  rsync -az --exclude isaac_server.log --exclude .isaac_ready "$HOST:$REMOTE_DIR/runs/" "$ROOT/runs/" || true
fi
exit $rc
