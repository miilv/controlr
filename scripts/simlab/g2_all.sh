#!/usr/bin/env bash
# G2a: record + same-process replays + perturbation sweep, then a fresh-process replay, per scenario.
#   scripts/simlab/g2_all.sh OUT [extra g2 args, e.g. --physics-only]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$1"; shift
mkdir -p "$OUT/logs"
for sc in pick_place stack push; do
  "$HERE/scripts/simlab/rdpy.sh" -m dojo.simlab.g2 record --scenario "$sc" --out "$OUT" "$@" > "$OUT/logs/${sc}_record.log" 2>&1
  echo "$(date -u +%T) $sc record rc=$?" >> "$OUT/logs/progress.txt"
  "$HERE/scripts/simlab/rdpy.sh" -m dojo.simlab.g2 replay --scenario "$sc" --out "$OUT" --tag new_process --no-planner "$@" > "$OUT/logs/${sc}_replay.log" 2>&1
  echo "$(date -u +%T) $sc replay rc=$?" >> "$OUT/logs/progress.txt"
done
echo DONE >> "$OUT/logs/progress.txt"
