#!/usr/bin/env bash
# Continuation driver on Misha's box (same rules as rerender_isr.sh): optionally wait for an already
# running task process of ours, render the remaining D0 tasks in real Isaac 5.1, then the contact-rich G2b
# replays (push / pick_place / stack: nominal + perturbation set, rendered AND physics-only) inside the
# same GPU_GO_ILIA window; delete the GO file at the end.
#   rerender_isr2.sh [--wait-pid PID --wait-task TASK] task [task ...]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ILIA=/home/anywherevla/data2/ilia
BENCH=/home/anywherevla/data2/bench
GO="$BENCH/GPU_GO_ILIA"; COORD="$BENCH/COORDINATION.md"; RUN="$ILIA/run"
export DOJO_ILIA=1
export RD_HOME=/home/anywherevla/data2/controlr-robodojo DOJO_DATA="$ILIA/dojo-data"
export HOME="$ILIA/home" XDG_CACHE_HOME="$ILIA/home/.cache" XDG_DATA_HOME="$ILIA/home/.local/share"
export SIMLAB_MIN_FREE_MB=11000 OMP_NUM_THREADS=4 SIMLAB_KIT_ARGS="--portable-root $ILIA/kitroot"
mkdir -p "$RUN" "$ILIA/d0_51/logs" "$ILIA/g2b/logs"
echo $$ > "$RUN/rerender_isr.pid"
isaac_on_gpu() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' '); do
    grep -qaiE 'isaac|kit/|eval_client|SimulationApp' "/proc/$p/cmdline" 2>/dev/null && return 0
  done
  return 1
}
go() { [[ -f "$GO" ]] && ! isaac_on_gpu; }
log_coord() { echo "[$(date -u +%FT%H:%MZ)] Ilia side: $*" >> "$COORD"; }
if [[ "${1:-}" == --wait-pid ]]; then
  wp="$2" wt="$4"; shift 4
  while [[ -d /proc/$wp ]]; do sleep 20; done
  rm -f "$RUN/rerender_$wt.pid"
  echo "$(date -u +%T) $wt rc=? $(tr '\r' '\n' < "$ILIA/d0_51/logs/$wt.log" | grep -ac '^\[rerender\]') layouts" >> "$ILIA/d0_51/logs/progress.txt"
  log_coord "DONE task $wt, GPU released (continuation driver pid $$)"
  sleep 90
fi
run_job() {   # name, then the command; one Isaac process, nice, pidfile
  local name="$1"; shift
  until go; do sleep 60; done
  log_coord "START $name (driver pid $$)"
  nice -n 10 "$@" &
  local child=$!; echo "$child" > "$RUN/$name.pid"
  wait "$child"; local rc=$?
  rm -f "$RUN/$name.pid"
  log_coord "DONE $name (rc $rc), GPU released"
  sleep 90
  return $rc
}
for t in "$@"; do
  run_job "rerender_$t" "$HERE/scripts/simlab/rdpy.sh" -m dojo.simlab.rerender_d0 --src "$ILIA/d0_src" --out "$ILIA/d0_51" \
      --task "$t" > "$ILIA/d0_51/logs/$t.log" 2>&1
  echo "$(date -u +%T) $t rc=$? $(tr '\r' '\n' < "$ILIA/d0_51/logs/$t.log" | grep -ac '^\[rerender\]') layouts" >> "$ILIA/d0_51/logs/progress.txt"
done
for mode in rendered physics; do
  extra=(); [[ "$mode" == physics ]] && extra=(--physics-only)
  for sc in push pick_place stack; do
    run_job "g2b_${sc}_$mode" "$HERE/scripts/simlab/rdpy.sh" -m dojo.simlab.g2 replay --scenario "$sc" --out "$ILIA/g2b" \
        --tag "isr4090_$mode" --no-planner --perturb "${extra[@]}" > "$ILIA/g2b/logs/${sc}_$mode.log" 2>&1
    echo "$(date -u +%T) g2b $sc $mode rc=$?" >> "$ILIA/g2b/logs/progress.txt"
  done
done
rm -f "$GO" "$RUN/rerender_isr.pid"
log_coord "FINISHED (D0 genuine-5.1 renders + G2b replays); removed $GO; no processes of ours left"
