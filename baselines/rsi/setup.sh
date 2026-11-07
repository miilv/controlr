#!/usr/bin/env bash
# compute7 only; no shared installation writes or credentials.
set -euo pipefail
ROOT="$HOME/controlr-rsi-repro"
export HOME="$ROOT/home" XDG_CACHE_HOME="$ROOT/cache" UV_CACHE_DIR="$ROOT/cache/uv"
mkdir -p "$HOME" "$XDG_CACHE_HOME"
UV=/home/robotics/.local/bin/uv
REPO="$ROOT/PyRUA-Lean"
REV=e754cf43c822b91f4728777a78071fa4e012c371
[[ $(git -C "$REPO" rev-parse HEAD) == "$REV" ]]
$UV venv --python-preference only-managed --python /home/robotics/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/bin/python "$ROOT/venv"
$UV pip install -p "$ROOT/venv/bin/python" pip
$UV pip install -p "$ROOT/venv/bin/python" torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
cat > "$ROOT/constraints.txt" <<'CONSTRAINTS'
torch==2.8.0
torchvision==0.23.0
numpy>=1.26,<2
opencv-python-headless<4.12
CONSTRAINTS
pin() { $UV pip install -p "$ROOT/venv/bin/python" -c "$ROOT/constraints.txt" "$@"; }
# Match the evaluated runtime separately from the newer RSI program bundle.
git -C "$REPO" fetch --depth 1 origin c6861200bfb190169953d1827260ccf71320eea1
[[ -d "$ROOT/runtime" ]] || git -C "$REPO" worktree add --detach "$ROOT/runtime" c6861200bfb190169953d1827260ccf71320eea1
pin -e "$ROOT/runtime" numpy pillow pyyaml httpx 'mcp>=1.8,<2' uvicorn scipy h5py 'websockets>=14' msgpack msgpack-numpy pydantic 'opencv-python-headless<4.12'
pin 'sam3 @ git+https://github.com/RLinf/sam3.git@c1f6b1e83967c912f675d8975d437ccdc41e17d8' einops pycocotools psutil
export PATH="$ROOT/venv/bin:/usr/bin:/bin"
bash "$REPO/integrations/xpolicylab/PyRUA_RSI/download_assets.sh" modelscope
printf '%s  %s\n' 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e "$REPO/integrations/xpolicylab/PyRUA_RSI/checkpoints/sam3.pt" | sha256sum -c -
python -m pip freeze > "$ROOT/requirements.freeze.txt"
