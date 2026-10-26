#!/usr/bin/env bash
# CPU-only held-pose evaluation on compute3 (no GPU, no lock): the perception venv with CPU silhouettes, nice, in a
# scope (MemoryMax 8G, CPUQuota = 100 % x workers); aborts below 8 GB MemAvailable.
#   scripts/perception/held_eval.sh <out.jsonl> [held_eval run args ...]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$1"; shift
W="${HELD_WORKERS:-4}"
export DOJO_CATALOG="${DOJO_CATALOG:-$HOME/dojo-perception-data/catalog}" DOJO_X5_DIR="${DOJO_X5_DIR:-$HOME/dojo-perception-data/catalog/robot/x5}"
export PYTHONPATH="$HERE" DOJO_GL=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="${HELD_PY:-$HOME/dojo-perception/.venv/bin/python}"
avail_gb() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
[[ "$(avail_gb)" -ge 20 ]] || { echo "[held_eval] MemAvailable $(avail_gb) GB < 20: not starting" >&2; exit 3; }
systemd-run --user --scope -q -p MemoryMax=8G -p CPUQuota=$((100 * W))% nice -n 10 \
  "$PY" -m dojo.perception.held_eval run --out "$OUT" --workers "$W" "$@" &
child=$!
trap 'kill -TERM "$child" 2>/dev/null; wait "$child"; exit 143' TERM INT
while kill -0 "$child" 2>/dev/null; do
  if [[ "$(avail_gb)" -lt 8 ]]; then echo "[held_eval] MemAvailable < 8 GB: aborting" >&2; kill -TERM "$child"; fi
  sleep 5
done
wait "$child"
