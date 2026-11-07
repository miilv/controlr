#!/usr/bin/env bash
# Explicit paths; no shared install writes, credentials or host defaults.
set -euo pipefail
ROOT= MODELS= UV= PYTHON=3.11
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT=$2; shift 2;; --models) MODELS=$2; shift 2;;
    --uv) UV=$2; shift 2;; --python) PYTHON=$2; shift 2;;
    *) echo 'usage: setup.sh --root ABS --models ABS [--uv ABS] [--python 3.11]'; exit 2;;
  esac
done
[[ "$ROOT" == /* && "$MODELS" == /* ]] || { echo 'absolute --root and --models required'; exit 2; }
UV=${UV:-$(command -v uv)}
if [[ ${RSI_SETUP_CLEAN:-0} != 1 ]]; then
  exec env -i HOME="$ROOT/home" PATH=/usr/bin:/bin RSI_SETUP_CLEAN=1 bash "$0" \
    --root "$ROOT" --models "$MODELS" --uv "$UV" --python "$PYTHON"
fi
export XDG_CACHE_HOME="$ROOT/cache" UV_CACHE_DIR="$ROOT/cache/uv" UV_PYTHON_INSTALL_DIR="$ROOT/python"
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$MODELS"
REPO="$ROOT/PyRUA-Lean"
REV=e754cf43c822b91f4728777a78071fa4e012c371
if [[ ! -d "$REPO" ]]; then
  git clone --no-checkout --depth 1 https://github.com/DAGroup-PKU/PyRUA-Lean.git "$REPO"
  git -C "$REPO" fetch --depth 1 origin "$REV"
  git -C "$REPO" checkout --detach "$REV"
fi
[[ $(git -C "$REPO" rev-parse HEAD) == "$REV" ]]
[[ -x "$ROOT/venv/bin/python" ]] || "$UV" venv --python-preference only-managed --python "$PYTHON" "$ROOT/venv"
"$UV" pip install -p "$ROOT/venv/bin/python" pip
"$UV" pip install -p "$ROOT/venv/bin/python" torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
cat > "$ROOT/constraints.txt" <<'CONSTRAINTS'
torch==2.8.0
torchvision==0.23.0
numpy>=1.26,<2
opencv-python-headless<4.12
CONSTRAINTS
pin() { "$UV" pip install -p "$ROOT/venv/bin/python" -c "$ROOT/constraints.txt" "$@"; }
git -C "$REPO" fetch --depth 1 origin c6861200bfb190169953d1827260ccf71320eea1
[[ -d "$ROOT/runtime" ]] || git -C "$REPO" worktree add --detach "$ROOT/runtime" c6861200bfb190169953d1827260ccf71320eea1
[[ $(git -C "$ROOT/runtime" rev-parse HEAD) == c6861200bfb190169953d1827260ccf71320eea1 ]]
pin -e "$ROOT/runtime" numpy pillow pyyaml httpx 'mcp>=1.8,<2' uvicorn scipy h5py 'websockets>=14' msgpack msgpack-numpy pydantic 'opencv-python-headless<4.12'
pin 'sam3 @ git+https://github.com/RLinf/sam3.git@c1f6b1e83967c912f675d8975d437ccdc41e17d8' einops pycocotools psutil
export PATH="$ROOT/venv/bin:/usr/bin:/bin"
SHA=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
if [[ ! -f "$MODELS/sam3.pt" ]]; then
  bash "$REPO/integrations/xpolicylab/PyRUA_RSI/download_assets.sh" modelscope
  ASSET="$REPO/integrations/xpolicylab/PyRUA_RSI/checkpoints/sam3.pt"
  printf '%s  %s\n' "$SHA" "$ASSET" | sha256sum -c -
  cp "$ASSET" "$MODELS/sam3.pt.partial"
  mv "$MODELS/sam3.pt.partial" "$MODELS/sam3.pt"
fi
printf '%s  %s\n' "$SHA" "$MODELS/sam3.pt" | sha256sum -c -
python -m pip freeze > "$ROOT/requirements.freeze.txt"
