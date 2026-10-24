#!/usr/bin/env bash
# Twin throughput bench (deliverable 5), compute3-safe:
#   scripts/simlab/twin_bench.sh [task]       (physics-only; results ~/dojo-data/simlab/twin_bench/)
# Waits for MemAvailable > 25 GB and load1 < 20; everything runs inside ONE systemd user scope
# (MemoryMax=20G, CPUQuota=1200% = 12 cores) and under ONE GPU lock held here (children skip it).
# In-process num_envs 1 -> 4 -> 8 (-> 16 only if n=8 peaked under 12 GB RSS), then 2 and 4 separate
# single-env processes in parallel.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK="${1:-stack_blocks}"
export RD_HOME="${RD_HOME:-$HOME/dojo-robodojo}" DOJO_DATA="${DOJO_DATA:-$HOME/dojo-data}"
OUT="$DOJO_DATA/simlab/twin_bench"; mkdir -p "$OUT"
G2="$DOJO_DATA/simlab/g2_c3b"
log() { echo "$(date -u +%T) $*" >> "$OUT/progress.txt"; }
wait_ok() {
  for i in $(seq 240); do
    avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); load=$(cut -d' ' -f1 /proc/loadavg)
    if [[ "$avail" -ge 25 ]] && awk -v l="$load" 'BEGIN{exit !(l < 20)}'; then return 0; fi
    [[ $((i % 6)) -eq 1 ]] && log "waiting: MemAvailable ${avail} GB, load ${load}"
    sleep 20
  done
  return 1
}
LOCK="${DOJO_GPU_LOCK:-$HOME/dojo-gpu.lock}"
exec 9>>"$LOCK"
until flock -n 9; do sleep 30; done
printf 'pid=%s stream=simlab start=%s cmd=twin_bench %s\n' "$$" "$(date -u +%FT%TZ)" "$TASK" > "$LOCK"
export SIMLAB_GPU=0          # the lock is held here
scoped() { systemd-run --user --scope -q -p MemoryMax=20G -p CPUQuota=1200% "$@"; }
one() {   # $1 = num_envs, $2 = tag
  SIMLAB_BENCH_TAG="$2" scoped "$HERE/scripts/simlab/rdpy.sh" -m dojo.simlab.twin --task "$TASK" --num-envs "$1" \
      --steps 100 --out "$OUT" --physics-only --g2-dir "$G2" > "$OUT/bench_${TASK}_n$1$2.log" 2>&1
  log "n=$1$2 rc=$? $(tr '\r' '\n' < "$OUT/bench_${TASK}_n$1$2.log" | grep -a '^\[twin\]')"
}
for n in 1 4 8; do wait_ok || { log "gave up waiting"; exit 1; }; one "$n" ""; done
rss8=$(python3 -c "import json;print(json.load(open('$OUT/bench_${TASK}_n8.json'))['max_rss_gb'])" 2>/dev/null || echo 99)
if awk -v r="$rss8" 'BEGIN{exit !(r < 12)}'; then wait_ok && one 16 ""; else log "skip n=16 (n=8 rss ${rss8} GB)"; fi
for N in 2 4; do       # N separate single-env processes sharing the same 12-core / 20 GB scope
  wait_ok || exit 1
  scoped bash -c "for i in \$(seq $N); do SIMLAB_BENCH_TAG=_proc\${i}of$N '$HERE/scripts/simlab/rdpy.sh' -m dojo.simlab.twin \
      --task $TASK --num-envs 1 --steps 100 --out '$OUT' --physics-only > '$OUT/bench_${TASK}_n1_proc\${i}of$N.log' 2>&1 & done; wait"
  log "separate x$N: $(for i in $(seq $N); do tr '\r' '\n' < "$OUT/bench_${TASK}_n1_proc${i}of$N.log" | grep -a '^\[twin\]' | grep -o '[0-9.]* env-steps/s total'; done | tr '\n' ' ')"
done
log DONE
