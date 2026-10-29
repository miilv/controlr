#!/usr/bin/env bash
# One seg_render job (genuine Isaac 5.1, RoboDojo's python) under compute4's sharing rules:
#   - waits (polls every 15 s, up to SEG_WAIT_S) until fewer than SEG_MAX_APPS GPU apps and SEG_MAX_RENDERED rendered
#     processes (apps >= 4 GiB) run, >= 10 GB VRAM and
#     >= 16 GB RAM are free and the lock is free; with SEG_REQUIRE_IDLE=1 (default) also until no other GPU app holds
#     >= 1 GiB (Mikhail's starter treats such a GPU as busy: a pilot only slips into a gap of theirs);
#   - holds the GPU lock (SEG_GPU_LOCK, else DOJO_GPU_LOCK, else our ~/dojo-ours-gpu.lock) through rdpy.sh for at most SEG_MAX_S
#     seconds (default 600: boot + layouts until SEG_MAX_S - 200 s + close), in a 16 GB / 12-core scope.
#   scripts/perception/seg/render_job.sh TASK [seg_render args...]     (--out defaults to ~/dojo-data/segtrain/render)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TASK="$1"; shift
LOCK="${SEG_GPU_LOCK:-${DOJO_GPU_LOCK:-$HOME/dojo-ours-gpu.lock}}"   # never compute4's ~/dojo-gpu.lock (Mikhail's)
MAX_S="${SEG_MAX_S:-600}"
WAIT_S="${SEG_WAIT_S:-14400}"
MAX_APPS="${SEG_MAX_APPS:-4}"
IDLE="${SEG_REQUIRE_IDLE:-1}"
MAX_RENDERED="${SEG_MAX_RENDERED:-99}"   # rendered (Isaac) processes on the box, ours included: apps holding >= 4 GiB
OUT="${SEG_OUT:-$HOME/dojo-data/segtrain/render}"
LOG="$HOME/dojo-data/segtrain/logs"
mkdir -p "$LOG" "$OUT"
source "$HOME/dojo-env.sh"
t0=$(date +%s)
while true; do
  big=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '$1 >= 1024' | wc -l)
  napps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  avail=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  vfree=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  nrend=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '$1 >= 4096' | wc -l)
  [[ "$IDLE" == 1 ]] || big=0
  if [[ "$big" -eq 0 && "$napps" -lt "$MAX_APPS" && "$nrend" -lt "$MAX_RENDERED" && "$avail" -ge 16 && "$vfree" -ge 10240 ]] \
      && flock -n "$LOCK" true; then break; fi
  if (( $(date +%s) - t0 > WAIT_S )); then echo "$(date -u +%FT%TZ) $TASK gave up waiting" >> "$LOG/render_jobs.txt"; exit 75; fi
  sleep 15
done
echo "$(date -u +%FT%TZ) $TASK start (waited $(( $(date +%s) - t0 )) s) $*" >> "$LOG/render_jobs.txt"
cd "$HERE"
DOJO_GPU_LOCK="$LOCK" SIMLAB_LOCK_WAIT_S=30 SIMLAB_QUEUE_LOCK="" timeout -k 30 "$MAX_S" \
  systemd-run --user --scope -q -p MemoryMax=16G -p CPUQuota=1200% \
  "$HERE/scripts/simlab/rdpy.sh" -m dojo.simlab.seg_render --task "$TASK" --out "$OUT" --budget-s $(( MAX_S - 200 )) "$@" \
  > "$LOG/render_${TASK}_$(date -u +%m%dT%H%M%S).log" 2>&1
rc=$?
echo "$(date -u +%FT%TZ) $TASK end rc=$rc ($(( $(date +%s) - t0 )) s)" >> "$LOG/render_jobs.txt"
exit $rc
