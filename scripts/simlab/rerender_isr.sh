#!/usr/bin/env bash
# Genuine Isaac 5.1 renders of D0 layouts on Misha's 4090 box (isr-lab; we SHARE the Unix user anywherevla).
# Rules: GPU only through the shared coord lock (scripts/simlab/isr_gpu.sh: ~/data2/coord/gpu.lock + gpu.owner,
# respect gpu.reserve, <= 2 h per hold); START/DONE lines in COORDINATION.md; ONE Isaac instance; nice; <= 4
# processes; write only under
# ~/data2/ilia; his install read-only. Every process we start carries DOJO_ILIA=1 and has a pidfile in
# ~/data2/ilia/run/; never pkill/pgrep by pattern (stop with scripts/simlab/ilia_stop.sh).
#   ~/data2/ilia/controlr-simlab/scripts/simlab/rerender_isr.sh task [task ...]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ILIA=/home/anywherevla/data2/ilia
BENCH=/home/anywherevla/data2/bench
COORD="$BENCH/COORDINATION.md"; RUN="$ILIA/run"
export DOJO_ILIA=1
export RD_HOME=/home/anywherevla/data2/controlr-robodojo DOJO_DATA="$ILIA/dojo-data"
export HOME="$ILIA/home" XDG_CACHE_HOME="$ILIA/home/.cache" XDG_DATA_HOME="$ILIA/home/.local/share"   # home partition is full
export SIMLAB_MIN_FREE_MB=11000 OMP_NUM_THREADS=4
export SIMLAB_KIT_ARGS="--portable-root $ILIA/kitroot"   # Kit's data/logs/cache outside Misha's install
mkdir -p "$HOME" "$DOJO_DATA" "$ILIA/d0_51/logs" "$ILIA/kitroot" "$RUN"
echo $$ > "$RUN/rerender_isr.pid"
source "$HERE/scripts/simlab/isr_gpu.sh"
isr_log "driver pid $$ queued on $ISR_LOCK (tasks: $*)"
for t in "$@"; do
  isr_run "rerender_$t" 30 "$HERE/scripts/simlab/rdpy.sh" -m dojo.simlab.rerender_d0 --src "$ILIA/d0_src" \
      --out "$ILIA/d0_51" --task "$t" > "$ILIA/d0_51/logs/$t.log" 2>&1
  rc=$?
  echo "$(date -u +%T) $t rc=$rc $(tr '\r' '\n' < "$ILIA/d0_51/logs/$t.log" | grep -ac '^\[rerender\]') layouts" >> "$ILIA/d0_51/logs/progress.txt"
done
rm -f "$RUN/rerender_isr.pid"
isr_log "FINISHED all D0 renders; no processes of ours left"
