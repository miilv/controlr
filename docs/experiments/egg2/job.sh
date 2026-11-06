#!/bin/bash
set -euo pipefail
source ~/dojo-env.sh
cd ~/controlr-egg2
name=$1; seed=$2; mode=$3
source scripts/dojo/disk_guard.sh
disk_guard EGG2
exec 8>> ~/dojo-data/egg2/admission.lock
for ((attempt=0;attempt<720;attempt++)); do
 flock 8
 slot=''
 for candidate in "$HOME"/dojo-gpu-light{,2,3,4}.lock; do
  exec 9>> "$candidate"
  if flock -n 9; then slot=$candidate; break; fi
  exec 9>&-
 done
 used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
 ram=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
 load=$(cut -d' ' -f1 /proc/loadavg)
 if [[ -n "$slot" ]] && ((used <= 20480 && ram >= 12288)) && awk "BEGIN{exit !($load <= 26)}"; then
  printf 'pid=%s stream=egg2 seed=%s mode=%s\n' "$$" "$seed" "$mode" > "$slot"
  printf '%s seed=%s mode=%s vram=%s ram=%s load=%s slot=%s\n' "$(date -u +%FT%TZ)" "$seed" "$mode" "$used" "$ram" "$load" "$slot" >> ~/dojo-data/egg2/admissions.log
  flock -u 8
  export SIMLAB_GPU=0
  extra=(); [[ "$mode" == on ]] && extra=(--skill-attr egg_reach=search)
  exec systemd-run --user --scope -q -p MemoryMax=8G -p CPUQuota=400% scripts/simlab/rdpy.sh -u ~/dojo-data/egg2/measure.py run --task fill_egg_holder --seeds "$seed" --layouts ~/dojo-data/egg2/dev10 --trace --out "$HOME/dojo-data/egg2/$name" "${extra[@]}"
 fi
 exec 9>&-
 flock -u 8
 sleep 15
done
exit 75
