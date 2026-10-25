#!/usr/bin/env bash
# Deploy this worktree to compute3:~/<dir> (default dojo-skills3) and stamp the commit next to the code
# (g3 writes it into every episode record and the run summary). Refuses a dirty tree.
#   scripts/skills/deploy_skills3.sh [dir]
# Runs: DOJO_STREAM=skills3 G3_LOGD=~/dojo-data/skills3/logs ~/dojo-skills3/scripts/skills/g3_chunks.sh <task> ...
exec "$(dirname "$0")/deploy_skills2.sh" "${1:-dojo-skills3}"
