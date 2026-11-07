#!/usr/bin/env bash
# IMIT2 G3 panel job on compute3 (shared physicalai account), from this deployed copy:
#   scripts/skills/imit2_g3.sh <run_name> <seed_list> [extra g3 run args]
#   e.g. scripts/skills/imit2_g3.sh base 100002,100004-100009,100020-100029 --skill-attr imitate_pairing=one_to_one
# Admission (checked before each chunk): GPU memory used <= IMIT2_GPU_MAX_MIB (24 GiB), free disk >= 10 GB.
# Light locks only, systemd scope 8G / 600 % (g3.sh), physics-only. Output ~/dojo-data/imit2/g3/<run>_<commit>/.
# Our-slice dev ids only (100000-09, 100020-29): anything else is refused.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="$1"; LIST="$2"; shift 2
COMMIT="$(cut -c1-7 "$HERE/.deployed_commit" 2>/dev/null || echo nocommit)"
OUT="$HOME/dojo-data/imit2/g3/${RUN}_${COMMIT}"
LOGD="$HOME/dojo-data/imit2/logs"; mkdir -p "$LOGD" "$OUT"
export DOJO_STREAM=IMIT2 DOJO_GPU_LOCKS="$HOME/dojo-gpu-light.lock $HOME/dojo-gpu-light2.lock $HOME/dojo-gpu-light3.lock $HOME/dojo-gpu-light4.lock"
export G3_MIN_MEM_GB=20 G3_MEMORY_MAX=8G G3_ABORT_MEM_GB=8
export DOJO_CATALOG="${DOJO_CATALOG:-$HOME/dojo-data/skills4/catalog:$HOME/dojo-data/skills3/catalog:$HOME/dojo-perception-data/catalog}"
GPU_MAX="${IMIT2_GPU_MAX_MIB:-24576}"
seeds=()
IFS=',' read -ra parts <<< "$LIST"
for p in "${parts[@]}"; do
  if [[ "$p" == *-* ]]; then a="${p%-*}"; b="${p#*-}"; for ((s=a; s<=b; s++)); do seeds+=("$s"); done
  else seeds+=("$p"); fi
done
for s in "${seeds[@]}"; do
  if ! { [[ "$s" -ge 100000 && "$s" -le 100009 ]] || [[ "$s" -ge 100020 && "$s" -le 100029 ]]; }; then
    echo "[imit2_g3] $s is outside our measurement slices (100000-09, 100020-29): refused" >&2; exit 2
  fi
done
admit() {
  local used free
  for i in $(seq 240); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    free=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
    if [[ "$used" -le "$GPU_MAX" && "$free" -ge 10 ]]; then return 0; fi
    [[ $((i % 10)) -eq 1 ]] && echo "[imit2_g3] waiting: GPU used ${used} MiB (max ${GPU_MAX}), disk ${free} GB" >&2
    sleep 30
  done
  echo "[imit2_g3] admission timed out" >&2; return 1
}
chunk=()
run_chunk() {
  [[ ${#chunk[@]} -eq 0 ]] && return 0
  admit || exit 75
  "$HERE/scripts/skills/g3.sh" run --task imitate_sorting_sequence --seeds "$(IFS=,; echo "${chunk[*]}")" --out "$OUT" "$@" \
      >> "$LOGD/g3_${RUN}_${COMMIT}.log" 2>&1
  chunk=()
}
for s in "${seeds[@]}"; do
  chunk+=("$s")
  if [[ ${#chunk[@]} -ge ${IMIT2_CHUNK:-5} ]]; then run_chunk "$@"; sleep 90; fi
done
run_chunk "$@"
echo "[imit2_g3] done $RUN: $OUT"
