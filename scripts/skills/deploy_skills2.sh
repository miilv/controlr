#!/usr/bin/env bash
# Deploy this worktree to compute3:~/<dir> (default dojo-skills2) and record the commit next to the code
# (g3 writes it into every episode record and the run summary). Refuses a dirty tree.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DIR="${1:-dojo-skills2}"
cd "$ROOT"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "[deploy_skills2] uncommitted changes: commit first (runs must map to a commit)" >&2; exit 1
fi
scripts/deploy.sh "$DIR" > /dev/null
git rev-parse HEAD | ssh "${CONTROLR_HOST:-compute3}" "cat > ~/$DIR/.deployed_commit"
echo "[deploy_skills2] $(git rev-parse --short HEAD) -> ${CONTROLR_HOST:-compute3}:~/$DIR"
