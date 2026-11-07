#!/usr/bin/env bash
# CLASSIFY3: physics-only GT replay of recorded classify camera rows on compute3 (shared physicalai account).
#   scripts/perception/classify3/replay_job.sh TASK JOBS.jsonl OUT_DIR
# One light lock (a free one of light..light4), systemd user scope 8G / 600 %, physics-only (no rendering), the
# full action list (--full) so the replay's final score can be checked against the recorded row. Our-slice dev ids only
# (m95_replay refuses others through paths.is_offslice_rendered / assert_not_eval).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TASK="$1"; JOBS="$2"; OUT="$3"
export RD_HOME="${RD_HOME:-$HOME/dojo-robodojo}" DOJO_DATA="${DOJO_DATA:-$HOME/dojo-data}"
export DOJO_CATALOG="${DOJO_CATALOG:-$HOME/dojo-data/skills4/catalog:$HOME/dojo-data/skills3/catalog:$HOME/dojo-perception-data/catalog}"
free=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
[[ "$free" -ge 10 ]] || { echo "[classify3] disk ${free} GB < 10" >&2; exit 6; }
LOCK=""
for L in "$HOME/dojo-gpu-light.lock" "$HOME/dojo-gpu-light2.lock" "$HOME/dojo-gpu-light3.lock" "$HOME/dojo-gpu-light4.lock"; do
  exec 9>>"$L"
  if flock -n 9; then LOCK="$L"; break; fi
  exec 9>&-
done
[[ -n "$LOCK" ]] || { echo "[classify3] no free light lock" >&2; exit 75; }
printf 'pid=%s stream=CLASSIFY3 start=%s cmd=replay %s\n' "$$" "$(date -u +%FT%TZ)" "$TASK" > "$LOCK"
export SIMLAB_GPU=0 SIMLAB_MIN_FREE_MB=0
exec systemd-run --user --scope -q -p MemoryMax=8G -p CPUQuota=600% \
  "$HERE/scripts/simlab/rdpy.sh" -u "$HERE/scripts/perception/m95_replay.py" --task "$TASK" --jobs "$JOBS" \
  --out-dir "$OUT" --every 10 --full
