#!/usr/bin/env bash
# One segmenter training run under the box's GPU rules: waits (20 s polls) until fewer than SEG_MAX_APPS GPU apps run
# (everyone's), fewer than SEG_MAX_RENDERED heavy ones (>= 4 GiB), >= SEG_MIN_VRAM_MB (8 GB) VRAM and >= 16 GB RAM are
# free and the lock is free, then holds the lock (SEG_GPU_LOCK) for the
# run, in a 16 GB / 8-core scope. train.py throttles to the box's power (--power-target / --power-max); this script
# also logs nvidia-smi's power.draw every 30 s to <run>/power_box.csv.
#   SEG_GPU_LOCK=~/dojo-gpu-heavy2.lock scripts/perception/seg/train_job.sh PYTHON NAME DATA_YAML [train.py args...]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="$1"; NAME="$2"; DATA="$3"; shift 3
LOCK="${SEG_GPU_LOCK:?set SEG_GPU_LOCK}"
MAX_APPS="${SEG_MAX_APPS:-4}"
MAX_RENDERED="${SEG_MAX_RENDERED:-99}"   # heavy processes on the box (apps >= 4 GiB), this run counts as one of them
MIN_VRAM_MB="${SEG_MIN_VRAM_MB:-8192}"
SEG="${SEG_DIR:-$HOME/dojo-data/segtrain}"
RUN="$SEG/runs/$NAME"
mkdir -p "$RUN" "$SEG/logs"
note() { echo "$(date -u +%FT%TZ) $*" >> "$SEG/logs/train_jobs.txt"; }
while true; do
  [[ -f "$SEG/STOP" ]] && { note "$NAME: STOP file"; exit 0; }
  napps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  vfree=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  avail=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  nrend=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '$1 >= 4096' | wc -l)
  if [[ "$napps" -lt "$MAX_APPS" && "$nrend" -lt "$MAX_RENDERED" && "$vfree" -ge "$MIN_VRAM_MB" && "$avail" -ge 16 ]] \
      && flock -n "$LOCK" true; then break; fi
  sleep 20
done
( while true; do echo "$(date -u +%FT%TZ),$(nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' '),$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" >> "$RUN/power_box.csv"; sleep 30; done ) &
LOGGER=$!
trap 'kill $LOGGER 2>/dev/null' EXIT
note "$NAME: start ($LOCK)"
cd "$SEG/models"      # Ultralytics' AMP check loads yolo11n.pt from the cwd (put it there: no download mid-run)
exec 9>>"$LOCK"
flock -w 60 9 || { note "$NAME: lost the lock race"; exit 75; }
printf 'pid=%s stream=seg-S start=%s cmd=train %s\n' "$$" "$(date -u +%FT%TZ)" "$NAME" > "$LOCK"
YOLO_CONFIG_DIR="$SEG/ultra_cfg" systemd-run --user --scope -q -p MemoryMax=16G -p CPUQuota=800% \
  "$PY" "$HERE/scripts/perception/seg/train.py" --data "$DATA" --project "$SEG/runs" --name "$NAME" "$@" > "$RUN/train.log" 2>&1
rc=$?
note "$NAME: end rc=$rc"
exit $rc
