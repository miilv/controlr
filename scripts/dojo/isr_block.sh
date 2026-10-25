#!/usr/bin/env bash
# A BLOCK of practice episodes on isr-lab in ONE coord gpu.lock hold (<= 2 h; each Isaac boot costs ~1 min):
#   scripts/dojo/isr_block.sh BLOCK ETA_MIN SPEC
# SPEC: one episode per line, "ID TASK [bridge_episode.sh options]" (blank lines and # comments skipped); seed 0
# layout 9 are fixed (practice). Before EACH episode: the inbox / gpu.reserve gate and the disk floor
# (ISR_DISK_FLOOR_GB, default 7) -- any failure ends the block (rc 76 / 77) and releases the lock. After each
# episode: isr_prune.py (server PNGs) and isr_summary.py (one row -> $DOJO_DATA/bridge/rows.jsonl).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ $# -eq 3 ]] || { echo "usage: isr_block.sh BLOCK ETA_MIN SPEC" >&2; exit 2; }
BLOCK="$1" ETA="$2" SPEC="$(realpath "$3")"
export CR="$HERE"
source "$HERE/scripts/dojo/isr_env.sh"
export ISR_MAX_S="${ISR_MAX_S:-7200}"
if [[ "${ISR_IN_JOB:-0}" != 1 ]]; then
  ISR_IN_JOB=1 exec "$HERE/scripts/dojo/isr_job.sh" "$BLOCK" "$ETA" "$0" "$BLOCK" "$ETA" "$SPEC"
fi
COORD=/home/anywherevla/data2/coord
OUT="$DOJO_DATA/bridge/runs"; ROWS="$DOJO_DATA/bridge/rows.jsonl"; PY="$HERE/.venv/bin/python"
# Isaac's shader / texture caches under our shim HOME are NEVER pruned on isr-lab: rebuilding them spiked the box to
# ~20 cores (lead's rule, 2026-10-26). Below the disk floor the gate stops the block instead.
gate() {
  local m; m="$(find "$COORD/inbox/ilia" -maxdepth 1 -type f 2>/dev/null | head -1)"
  [[ -z "$m" ]] || { echo "[block] unprocessed inbox message $m: stopping the block" >&2; return 76; }
  if [[ -s "$COORD/gpu.reserve" ]]; then echo "[block] gpu.reserve is set: stopping the block" >&2; return 76; fi
  local free_gb; free_gb=$(df -BG --output=avail /home/anywherevla/data2 | tail -1 | tr -dc 0-9)
  (( free_gb >= ${ISR_DISK_FLOOR_GB:-7} )) || { echo "[block] only ${free_gb} GB free (< ${ISR_DISK_FLOOR_GB:-7}): stopping" >&2; return 77; }
  return 0
}
rc=0
while read -r line; do
  [[ -z "${line// }" || "$line" == \#* ]] && continue
  read -r -a f <<< "$line"
  ID="${f[0]}" TASK="${f[1]}"; ARGS=("${f[@]:2}")
  for x in "${ARGS[@]}"; do case "$x" in --seed|--layout|--fake) echo "[block] $x is fixed (practice s0 L9)" >&2; exit 2 ;; esac; done
  gate || { rc=$?; break; }
  mkdir -p "$OUT/$ID"
  echo "[block] $(date -u +%T) $ID $TASK ${ARGS[*]}"
  DOJO_AGENTIC_LOG="$OUT/$ID/agentic" "$HERE/scripts/dojo/bridge_episode.sh" --id "$ID" --task "$TASK" --layout 9 --seed 0 \
    --out "$OUT" "${ARGS[@]}" > "$OUT/$ID.episode.log" 2>&1
  erc=$?
  echo "[block] $(date -u +%T) $ID rc=$erc: $(grep -a '^\[bridge\] '"$ID"': end=' "$OUT/$ID.episode.log" | tail -1 | cut -c1-200)"
  "$PY" "$HERE/scripts/dojo/isr_prune.py" "$ID" || true
  "$PY" "$HERE/scripts/dojo/isr_summary.py" "$ID" --jsonl "$ROWS" || true
  df -h /home/anywherevla/data2 | tail -1
done < <(sed 's/\r$//' "$SPEC")
exit "$rc"
