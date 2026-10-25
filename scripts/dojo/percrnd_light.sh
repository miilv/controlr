#!/usr/bin/env bash
# percrnd_light.sh <cmd...>: run one perception R&D job on a compute3 light GPU lane.
# Lane rules: one free ~/dojo-gpu-light{,2,3,4}.lock (flock), start only at MemAvailable >= 20 GB, MemoryMax 8G,
# nice, abort if MemAvailable drops below 8 GB. Never touches other processes.
set -uo pipefail
avail_gb() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
[ "$(avail_gb)" -ge "${PERC_MIN_MEM_GB:-20}" ] || { echo "MemAvailable $(avail_gb) GB < 20: not starting"; exit 3; }
for L in "$HOME/dojo-gpu-light.lock" "$HOME/dojo-gpu-light2.lock" "$HOME/dojo-gpu-light3.lock" "$HOME/dojo-gpu-light4.lock"; do
  exec 9>>"$L"
  if flock -n 9; then
    echo "lane $L"
    systemd-run --user --scope -q -p MemoryMax="${PERC_MEMORY_MAX:-8G}" nice -n 10 "$@" &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
      if [ "$(avail_gb)" -lt "${PERC_ABORT_MEM_GB:-8}" ]; then echo "MemAvailable < 8 GB: aborting our job $pid"; kill "$pid"; fi
      sleep 5
    done
    wait "$pid"; rc=$?
    echo "DONE rc=$rc"; exit $rc
  fi
  exec 9>&-
done
echo "no free light lane"; exit 4
