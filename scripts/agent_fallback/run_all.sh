#!/usr/bin/env bash
# Sequential queue on ONE light lock: scripts/agent_fallback/run_all.sh LOCK ROOT CHUNK TASK...
# Each task's layouts 100000-04 (settled) in chunks of CHUNK layouts per job (job.sh: scope, timeout -k 60 1800).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCK="$1"; ROOT="$2"; CHUNK="$3"; shift 3
AG="${AFB_AGENT:?set AFB_AGENT to the AgentConfig JSON}"
L=~/dojo-data/simlab/layouts
for T in "$@"; do
  mapfile -t LS < <(ls $L/$T/10000[0-4].settled.json 2>/dev/null)
  for ((i=0; i<${#LS[@]}; i+=CHUNK)); do
    "$HERE/scripts/agent_fallback/job.sh" "$LOCK" "$T" "$ROOT" "$AG" "${LS[@]:i:CHUNK}"
  done
done
echo "[afb] queue done $(date -u +%FT%TZ)" >> "$ROOT/jobs.log"
