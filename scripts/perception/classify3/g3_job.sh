#!/usr/bin/env bash
# CLASSIFY3 G3 panel job on compute3 (shared physicalai account), from this deployed copy:
#   scripts/perception/classify3/g3_job.sh <task> <run_name> <seed_list> [extra g3 run args]
# Light locks light..light4 only (g3.sh takes a free one), systemd scope 8G / 600 % (g3.sh), physics-only. Admission
# before the job: GPU memory used <= 24 GiB, free disk >= 10 GB. Output ~/dojo-data/classify3/g3/<run>_<commit>/.
# Our-slice dev ids only (100000-09, 100020-29): anything else is refused.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TASK="$1"; RUN="$2"; LIST="$3"; shift 3
COMMIT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["head"])' "$HERE/.deployed.json" 2>/dev/null || echo nocommit)"
OUT="$HOME/dojo-data/classify3/g3/${RUN}_${COMMIT}"
LOGD="$HOME/dojo-data/classify3/logs"; mkdir -p "$LOGD" "$OUT"
export DOJO_STREAM=CLASSIFY3 DOJO_GPU_LOCKS="$HOME/dojo-gpu-light.lock $HOME/dojo-gpu-light2.lock $HOME/dojo-gpu-light3.lock $HOME/dojo-gpu-light4.lock"
export G3_MIN_MEM_GB=20 G3_MEMORY_MAX=8G G3_ABORT_MEM_GB=8
export DOJO_CATALOG="${DOJO_CATALOG:-$HOME/dojo-data/skills4/catalog:$HOME/dojo-data/skills3/catalog:$HOME/dojo-perception-data/catalog}"
IFS=',' read -ra parts <<< "$LIST"
for p in "${parts[@]}"; do
  if [[ "$p" == *-* ]]; then a="${p%-*}"; b="${p#*-}"; else a="$p"; b="$p"; fi
  for ((s=a; s<=b; s++)); do
    if ! { [[ "$s" -ge 100000 && "$s" -le 100009 ]] || [[ "$s" -ge 100020 && "$s" -le 100029 ]]; }; then
      echo "[classify3_g3] $s is outside our slices (100000-09, 100020-29): refused" >&2; exit 2
    fi
  done
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
free=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
if [[ "$used" -gt 24576 || "$free" -lt 10 ]]; then echo "[classify3_g3] refused: GPU used ${used} MiB, disk ${free} GB" >&2; exit 75; fi
timeout -k 60 2400 "$HERE/scripts/skills/g3.sh" run --task "$TASK" --seeds "$LIST" --out "$OUT" "$@" >> "$LOGD/g3_${RUN}_${COMMIT}.log" 2>&1
echo "[classify3_g3] done $RUN rc=$?: $OUT"
