#!/usr/bin/env bash
# Genuine Isaac 5.1 renders of D0 layouts on Misha's 4090 box (isr-lab, user anywherevla). His rules:
# ONE Isaac instance on the box (wait until his GPU work is gone or COORDINATION.md says "GPU free for Ilia"),
# everything nice, <= 4 processes of ours, write ONLY under ~/data2/ilia, his install used read-only.
#   ~/data2/ilia/controlr-simlab/scripts/simlab/rerender_isr.sh task [task ...]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ILIA=/home/anywherevla/data2/ilia
COORD=/home/anywherevla/data2/bench/COORDINATION.md
export RD_HOME=/home/anywherevla/data2/controlr-robodojo DOJO_DATA="$ILIA/dojo-data"
export HOME="$ILIA/home" XDG_CACHE_HOME="$ILIA/home/.cache" XDG_DATA_HOME="$ILIA/home/.local/share"   # home partition is full
export SIMLAB_MIN_FREE_MB=11000 OMP_NUM_THREADS=4
mkdir -p "$HOME" "$DOJO_DATA" "$ILIA/d0_51/logs"
gpu_free() {   # no other compute app on the GPU, or Misha's explicit go-ahead
  [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')" ]] && return 0
  grep -q "GPU free for Ilia" "$COORD" 2>/dev/null && return 0
  return 1
}
echo "[$(date -u +%FT%H:%MZ)] Ilia side: waiting for the GPU (one Isaac 5.1 instance, nice, <=4 procs, ~/data2/ilia only) to render D0 layouts: $*" >> "$COORD"
for t in "$@"; do
  until gpu_free; do sleep 60; done
  echo "[$(date -u +%FT%H:%MZ)] Ilia side: START Isaac 5.1 render task $t (pid $$)" >> "$COORD"
  nice -n 10 "$HERE/scripts/simlab/rdpy.sh" -m dojo.simlab.rerender_d0 --src "$ILIA/d0_src" --out "$ILIA/d0_51" \
      --task "$t" > "$ILIA/d0_51/logs/$t.log" 2>&1
  rc=$?
  echo "$(date -u +%T) $t rc=$rc $(tr '\r' '\n' < "$ILIA/d0_51/logs/$t.log" | grep -ac '^\[rerender\]') layouts" >> "$ILIA/d0_51/logs/progress.txt"
  echo "[$(date -u +%FT%H:%MZ)] Ilia side: DONE task $t (rc $rc); GPU released" >> "$COORD"
  sleep 90
done
echo "[$(date -u +%FT%H:%MZ)] Ilia side: FINISHED all D0 renders; no processes of ours left" >> "$COORD"
