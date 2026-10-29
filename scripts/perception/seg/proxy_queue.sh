#!/usr/bin/env bash
# The segmenter's Isaac 6.0 PROXY training set (compute3): per task, physics-only 5.1 states of new layout draws
# (seg_render --physics-only), then RGB (JPEG) + instance masks from the 6.0 proxy (render_d0_proxy.py --any-dirs
# --instance), each phase one job under our heavy lock (SEG_GPU_LOCK / DOJO_GPU_LOCK, default ~/dojo-ours-gpu.lock).
# Before every job: fewer than SEG_MAX_APPS GPU apps on the box (the cap counts everyone's), >= 10 GB VRAM and
# >= 20 GB RAM free, the lock free, and the data dir under SEG_CAP_MB (compute3's disk is 97 % full). A 90 s yield
# follows every job. ~/dojo-data/segtrain/STOP stops the queue.
#   scripts/perception/seg/proxy_queue.sh "--ids 100030-100049 --variants 1 --poses 4" task1 task2 ...
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ARGS="$1"; shift
LOCK="${SEG_GPU_LOCK:-${DOJO_GPU_LOCK:-$HOME/dojo-ours-gpu.lock}}"   # never compute4's ~/dojo-gpu.lock (Mikhail's)
MAX_APPS="${SEG_MAX_APPS:-4}"
CAP_MB="${SEG_CAP_MB:-3000}"
OUT="${SEG_OUT:-$HOME/dojo-data/segtrain/proxy60}"
LOG="$HOME/dojo-data/segtrain/logs"
mkdir -p "$LOG" "$OUT"
export RD_HOME="${RD_HOME:-$HOME/dojo-robodojo}" DOJO_DATA="${DOJO_DATA:-$HOME/dojo-data}"
note() { echo "$(date -u +%FT%TZ) $*" >> "$LOG/proxy_jobs.txt"; }

wait_slot() {
  while true; do
    [[ -f "$HOME/dojo-data/segtrain/STOP" ]] && { note "STOP file"; exit 0; }
    used=$(du -sm "$OUT" | cut -f1)
    [[ "$used" -ge "$CAP_MB" ]] && { note "stop: $used MB >= $CAP_MB MB"; exit 0; }
    napps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    vfree=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    avail=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
    if [[ "$napps" -lt "$MAX_APPS" && "$vfree" -ge 10240 && "$avail" -ge 20 ]] && flock -n "$LOCK" true; then return 0; fi
    sleep 20
  done
}

cd "$HERE"
for t in "$@"; do
  wait_slot
  note "$t states start"
  DOJO_GPU_LOCK="$LOCK" SIMLAB_LOCK_WAIT_S=60 SIMLAB_QUEUE_LOCK="" SIMLAB_MIN_FREE_MB=6000 timeout -k 30 1800 \
    systemd-run --user --scope -q -p MemoryMax=16G -p CPUQuota=800% \
    "$HERE/scripts/simlab/rdpy.sh" -m dojo.simlab.seg_render --task "$t" --out "$OUT" --physics-only --budget-s 1500 $ARGS \
    > "$LOG/proxy_${t}_states.log" 2>&1
  note "$t states rc=$? $(grep -ac '^\[seg\].*: ok' "$LOG/proxy_${t}_states.log") layouts"
  sleep "${SEG_YIELD_S:-90}"
  wait_slot
  note "$t render start"
  DOJO_GPU_LOCK="$LOCK" SIMLAB_LOCK_WAIT_S=60 SIMLAB_QUEUE_LOCK="" timeout -k 30 1800 \
    "$HERE/scripts/simlab/isaac6.sh" dojo/simlab/render_d0_proxy.py --root "$OUT" --tasks "$t" --any-dirs --instance \
    --jpeg 92 > "$LOG/proxy_${t}_render.log" 2>&1
  note "$t render rc=$? $(grep -ac '^\[render\].*frames in' "$LOG/proxy_${t}_render.log") layouts, $(du -sm "$OUT" | cut -f1) MB"
  sleep "${SEG_YIELD_S:-90}"
done
note "queue done"
