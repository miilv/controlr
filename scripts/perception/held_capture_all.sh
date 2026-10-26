#!/usr/bin/env bash
# All held-object captures, one task after the other on ONE light lane (90 s between jobs):
#   scripts/perception/held_capture_all.sh "task:caps:seed seed ..." ...
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for spec in "$@"; do
  IFS=: read -r task caps seeds <<< "$spec"
  echo "$(date -u +%FT%TZ) start $task caps=$caps seeds=$seeds"
  HP_MAX_CAPS="$caps" "$HERE/scripts/perception/held_capture.sh" "$task" $seeds
  echo "$(date -u +%FT%TZ) done $task rc=$?"
  sleep "${HELD_PAUSE_S:-90}"
done
