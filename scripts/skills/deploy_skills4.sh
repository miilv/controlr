#!/usr/bin/env bash
# Deploy this worktree to compute3:~/<dir> (default dojo-skills4) and stamp the commit next to the code
# (g3 writes it into every episode record and the run summary). Refuses a dirty tree.
#   scripts/skills/deploy_skills4.sh [dir]
# Runs: DOJO_STREAM=skills4 G3_LOGD=~/dojo-data/skills4/logs ~/dojo-skills4/scripts/skills/g3_chunks.sh <task> ...
exec "$(dirname "$0")/deploy_skills2.sh" "${1:-dojo-skills4}"
