#!/usr/bin/env bash
# Deploy this worktree to compute3:~/<dir> (default dojo-skills9) and stamp the commit next to the code
# (g3 writes it into every episode record and the run summary). Refuses a dirty tree.
#   scripts/skills/deploy_skills9.sh [dir]
# Runs: DOJO_STREAM=skills9 ~/dojo-skills9/scripts/skills/g3_light.sh <task> <first> <last> <run> [g3 args]
exec "$(dirname "$0")/deploy_skills2.sh" "${1:-dojo-skills9}"
