#!/usr/bin/env bash
# Load + settle every stored dev layout of the given tasks in the physics-only twin (RoboDojo's own reset
# and stability check). One scope (20G / 12 cores), GPU lock via rdpy.sh.
#   scripts/simlab/settle_check.sh task [task ...]   -> ~/dojo-data/simlab/settle/<task>/settle_summary.json
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RD_HOME="${RD_HOME:-$HOME/dojo-robodojo}" DOJO_DATA="${DOJO_DATA:-$HOME/dojo-data}" SIMLAB_MIN_FREE_MB="${SIMLAB_MIN_FREE_MB:-6000}"
for t in "$@"; do
  systemd-run --user --scope -q -p MemoryMax=20G -p CPUQuota=1200% "$HERE/scripts/simlab/rdpy.sh" -m dojo.simlab.runner \
      --task "$t" --layouts $(ls "$DOJO_DATA"/simlab/layouts/"$t"/1000*.json | sort) --out "$DOJO_DATA/simlab/settle/$t" \
      --physics-only > "$DOJO_DATA/simlab/settle_$t.log" 2>&1
  echo "$(date -u +%T) $t rc=$?" >> "$DOJO_DATA/simlab/settle/progress.txt"
done
