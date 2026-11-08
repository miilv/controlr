#!/usr/bin/env bash
# IMIT2 audit replays on compute3: one physics-only m95_replay job on a free light lock (admission GPU <= 24 GiB used,
# disk >= 10 GB), 8 G / 600 % scope, nice; zero model calls.  usage: audit_run.sh JOBS OUTDIR TAG
set -uo pipefail
JOBS=$1; OUT=$2; TAG=$3; CODE=$HOME/controlr-imit2-audit-code
export DOJO_DATA=$HOME/controlr-imit2-cam-data/imit2-cam-1 RD_HOME=$HOME/dojo-robodojo SIMLAB_GPU=0 SIMLAB_MIN_FREE_MB=0
export DOJO_CATALOG=$HOME/dojo-data/skills4/catalog:$HOME/dojo-data/skills3/catalog:$HOME/dojo-perception-data/catalog
mkdir -p "$OUT"
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  free=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
  if [[ "$used" -le 24576 && "$free" -ge 10 ]]; then
    for L in $HOME/dojo-gpu-light.lock $HOME/dojo-gpu-light2.lock $HOME/dojo-gpu-light3.lock $HOME/dojo-gpu-light4.lock; do
      exec 9>>"$L"
      if flock -n 9; then
        printf 'pid=%s stream=IMIT2 audit=%s start=%s\n' "$$" "$TAG" "$(date -u +%FT%TZ)" > "$L"
        echo "[audit] $TAG took $L (GPU ${used} MiB, disk ${free} GB)"
        nice -n 10 systemd-run --user --scope -q -p MemoryMax=8G -p CPUQuota=600% \
          bash "$CODE/scripts/simlab/rdpy.sh" -u "$CODE/scripts/perception/m95_replay.py" --task imitate_sorting_sequence \
          --jobs "$JOBS" --out-dir "$OUT" --every 4 --full
        rc=$?; echo "[audit] $TAG rc=$rc"; exit $rc
      fi
      exec 9>&-
    done
  fi
  sleep 30
done
