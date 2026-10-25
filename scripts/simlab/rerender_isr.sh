#!/usr/bin/env bash
# Genuine Isaac 5.1 renders of D0 layouts on Misha's 4090 box (isr-lab; we SHARE the Unix user anywherevla).
# Rules: start ONLY when ~/data2/bench/GPU_GO_ILIA exists AND no Isaac process is on the GPU; delete the file
# when done; START/DONE lines in COORDINATION.md; ONE Isaac instance; nice; <= 4 processes; write only under
# ~/data2/ilia; his install read-only. Every process we start carries DOJO_ILIA=1 and has a pidfile in
# ~/data2/ilia/run/; never pkill/pgrep by pattern (stop with scripts/simlab/ilia_stop.sh).
#   ~/data2/ilia/controlr-simlab/scripts/simlab/rerender_isr.sh task [task ...]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ILIA=/home/anywherevla/data2/ilia
BENCH=/home/anywherevla/data2/bench
GO="$BENCH/GPU_GO_ILIA"; COORD="$BENCH/COORDINATION.md"; RUN="$ILIA/run"
export DOJO_ILIA=1
export RD_HOME=/home/anywherevla/data2/controlr-robodojo DOJO_DATA="$ILIA/dojo-data"
export HOME="$ILIA/home" XDG_CACHE_HOME="$ILIA/home/.cache" XDG_DATA_HOME="$ILIA/home/.local/share"   # home partition is full
export SIMLAB_MIN_FREE_MB=11000 OMP_NUM_THREADS=4
export SIMLAB_KIT_ARGS="--portable-root $ILIA/kitroot"   # Kit's data/logs/cache outside Misha's install
mkdir -p "$HOME" "$DOJO_DATA" "$ILIA/d0_51/logs" "$ILIA/kitroot" "$RUN"
echo $$ > "$RUN/rerender_isr.pid"
isaac_on_gpu() {   # any compute app whose command line looks like Isaac / Kit / RoboDojo's eval client
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' '); do
    tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -qiE 'isaac|kit/|eval_client|SimulationApp' && return 0
  done
  return 1
}
go() { [[ -f "$GO" ]] && ! isaac_on_gpu; }
echo "[$(date -u +%FT%H:%MZ)] Ilia side: driver pid $$ waiting for $GO + no Isaac on the GPU (tasks: $*)" >> "$COORD"
for t in "$@"; do
  until go; do sleep 60; done
  echo "[$(date -u +%FT%H:%MZ)] Ilia side: START Isaac 5.1 render task $t (driver pid $$)" >> "$COORD"
  nice -n 10 "$HERE/scripts/simlab/rdpy.sh" -m dojo.simlab.rerender_d0 --src "$ILIA/d0_src" --out "$ILIA/d0_51" \
      --task "$t" > "$ILIA/d0_51/logs/$t.log" 2>&1 &
  child=$!; echo "$child" > "$RUN/rerender_$t.pid"
  wait "$child"; rc=$?
  rm -f "$RUN/rerender_$t.pid"
  echo "$(date -u +%T) $t rc=$rc $(tr '\r' '\n' < "$ILIA/d0_51/logs/$t.log" | grep -ac '^\[rerender\]') layouts" >> "$ILIA/d0_51/logs/progress.txt"
  echo "[$(date -u +%FT%H:%MZ)] Ilia side: DONE task $t (rc $rc), GPU released" >> "$COORD"
  sleep 90
done
rm -f "$GO" "$RUN/rerender_isr.pid"
echo "[$(date -u +%FT%H:%MZ)] Ilia side: FINISHED all D0 renders; removed $GO; no processes of ours left" >> "$COORD"
