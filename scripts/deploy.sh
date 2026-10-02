#!/usr/bin/env bash
# Sync this repo to compute3 and (re)build its uv venv.
#   scripts/deploy.sh [REMOTE_DIR]        (default ~/controlr)
# The .env (API key) is never committed; it is copied only if missing remotely.
set -euo pipefail
HOST="${CONTROLR_HOST:-compute3}"
REMOTE_DIR="${1:-controlr}"           # relative to the remote home
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

rsync -az --delete \
  --exclude .git --exclude .venv --exclude 'research/repos' --exclude runs \
  --exclude __pycache__ --exclude '*.egg-info' --exclude .pytest_cache --exclude .env \
  "$ROOT/" "$HOST:$REMOTE_DIR/"

if [ -f "$ROOT/.env" ] && ! ssh "$HOST" test -f "$REMOTE_DIR/.env"; then
  scp -q "$ROOT/.env" "$HOST:$REMOTE_DIR/.env"
  ssh "$HOST" chmod 600 "$REMOTE_DIR/.env"
  echo "copied .env"
fi

ssh "$HOST" bash -s -- "$REMOTE_DIR" <<'REMOTE'
set -euo pipefail
cd "$HOME/$1"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -q -e ".[dev]"
echo "deployed to $PWD ($(.venv/bin/python --version))"
REMOTE
