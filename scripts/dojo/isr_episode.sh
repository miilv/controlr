#!/usr/bin/env bash
# One PRACTICE strict full-system episode on isr-lab (seed 0 layout 9 only, robot-harness AGENTS.md rule 1) through
# the official RoboDojo eval client, under the coord GPU protocol (scripts/dojo/isr_job.sh):
#   scripts/dojo/isr_episode.sh ID TASK [bridge_episode.sh options: --policy --perception --verify --config ...]
# - seed and layout are fixed here (0 / 9) and cannot be passed: nothing else is practice;
# - refuses when ~/data2 has less than ISR_DISK_FLOOR_MB (default 7168; Mikhail's floor is 6 GB) free;
# - DOJO_AGENTIC_LOG = the run dir's agentic/ (agentic perceiver transcripts + frames);
# - the run dir is $DOJO_DATA/bridge/runs/ID; never reads robodojo_result/, _result.json or dev.jsonl.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ $# -ge 2 ]] || { echo "usage: isr_episode.sh ID TASK [bridge_episode.sh options]" >&2; exit 2; }
ID="$1" TASK="$2"; shift 2
for x in "$@"; do
  case "$x" in --seed|--layout|--fake) echo "isr_episode.sh: $x is fixed here (practice seed 0 layout 9)" >&2; exit 2 ;; esac
done
export CR="$HERE"
source "$HERE/scripts/dojo/isr_env.sh"
free_mb=$(df -BM --output=avail /home/anywherevla/data2 | tail -1 | tr -dc 0-9); free_gb=$(awk -v m="$free_mb" 'BEGIN {printf "%.1f", m / 1024}')
(( free_mb >= ${ISR_DISK_FLOOR_MB:-7168} )) || { echo "isr_episode.sh: only ${free_gb} GB free on data2 (< ${ISR_DISK_FLOOR_MB:-7168} MB)" >&2; exit 77; }
OUT="$DOJO_DATA/bridge/runs"
mkdir -p "$OUT/$ID"
export DOJO_AGENTIC_LOG="$OUT/$ID/agentic"
exec "$HERE/scripts/dojo/isr_job.sh" "$ID" "${ISR_ETA_MIN:-30}" "$HERE/scripts/dojo/bridge_episode.sh" --id "$ID" \
  --task "$TASK" --layout 9 --seed 0 --out "$OUT" "$@"
