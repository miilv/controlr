#!/usr/bin/env bash
# Run a command with the twin-trained segmenter server up, then stop it (one server per job, like the mv3d / SAM ones).
# The server (torch + ultralytics, ~0.3 GB VRAM, ~20 ms per head frame) listens on a per-job unix socket that is
# exported as DOJO_SEG_ADDR, so a perception service started by CMD (DOJO_PERCEPTION_CFG='{"seg_propose": "on"}')
# finds it. Without the server the flag is a no-op.
#   SEG_PY=~/controlr-seg/.venv/bin/python SEG_WEIGHTS=~/dojo-data/segtrain/models/v1/best.pt \
#   SEG_CLASSES=~/dojo-data/segtrain/models/v1/classes.json scripts/perception/seg/with_seg_server.sh CMD [ARGS...]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${SEG_PY:?python with torch + ultralytics}" "${SEG_WEIGHTS:?best.pt}" "${SEG_CLASSES:?classes.json}"
SOCK="${SEG_SOCK:-${XDG_RUNTIME_DIR:-/tmp}/dojo-seg-$$.sock}"
LOG="${SEG_SERVER_LOG:-/dev/null}"
rm -f "$SOCK"
( cd "$HERE" && YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$(dirname "$SEG_WEIGHTS")/ultra_cfg}" exec "$SEG_PY" -m dojo.perception.seg_twin serve \
    --sock "$SOCK" --weights "$SEG_WEIGHTS" --classes "$SEG_CLASSES" --conf "${SEG_CONF:-0.25}" --max-hours "${SEG_MAX_HOURS:-6}" ) \
    > "$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; rm -f "$SOCK"' EXIT
for _ in $(seq 120); do [[ -S "$SOCK" ]] && break; kill -0 $SRV 2>/dev/null || { echo "[with_seg_server] server died (log: $LOG)" >&2; exit 1; }; sleep 0.5; done
[[ -S "$SOCK" ]] || { echo "[with_seg_server] no socket after 60 s" >&2; exit 1; }
export DOJO_SEG_ADDR="$SOCK"
"$@"
