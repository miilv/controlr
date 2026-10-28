#!/usr/bin/env bash
# One light-lane job: the agent-fallback ceiling on one task's layouts, under a light GPU lock, a scope and a timeout.
#   scripts/agent_fallback/job.sh LOCK TASK OUTROOT AGENT_JSON LAYOUT...
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "$HOME/dojo-env.sh" ]]; then source "$HOME/dojo-env.sh"; else
  export RD_HOME="$HOME/dojo-robodojo" RD_ROOT="$HOME/dojo-robodojo/RoboDojo" DOJO_DATA="$HOME/dojo-data" \
    DOJO_PERCEPTION_DATA="$HOME/dojo-perception-data" OMNI_KIT_ACCEPT_EULA=YES; fi
LOCK="$1"; TASK="$2"; ROOT="$3"; AG="$4"; shift 4
OUT="$ROOT/$TASK"; mkdir -p "$OUT"
export DOJO_GPU_LOCK="$LOCK" SIMLAB_MIN_FREE_MB="${SIMLAB_MIN_FREE_MB:-3000}" DOJO_STREAM=reviewer
export DOJO_AGENT_DOTENV="$HOME/controlr/.env"
export DOJO_CATALOG="$DOJO_DATA/skills3/catalog:$DOJO_DATA/skills2/catalog:$DOJO_PERCEPTION_DATA/catalog"
unset OMNIROUTE_BASE_URL OMNIROUTE_API_KEY
avail() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
for i in $(seq 60); do [[ "$(avail)" -ge 12 ]] && break; sleep 20; done
[[ "$(avail)" -ge 10 ]] || { echo "[afb] MemAvailable $(avail) GB < 10: skip $TASK" >&2; exit 75; }
cd "$HERE"
# wait for the light lock OUTSIDE the job's timeout (a lane held by another stream must not eat the 30 min)
until flock -n "$LOCK" true; do sleep 30; done
timeout -k 60 1800 systemd-run --user --scope -q -p MemoryMax=8G -p CPUQuota=600% \
  "$HERE/scripts/simlab/rdpy.sh" -u scripts/agent_fallback/g3_agent.py --task "$TASK" --out "$OUT" --agent "$AG" \
  --layouts "$@" > "$OUT/job.log" 2>&1
rc=$?
echo "[afb] $TASK rc=$rc $(date -u +%FT%TZ)" | tee -a "$ROOT/jobs.log"
sleep 30
exit $rc
