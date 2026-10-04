#!/usr/bin/env bash
# Install RoboDojo (Isaac Sim 5.1 + Isaac Lab + cuRobo + eval assets) under ONE directory on a
# shared host (compute2). Everything — Miniconda, the conda env, pip cache, the RoboDojo clone and
# its assets — lives in $RD_HOME; nothing touches ~/.bashrc or a shared conda. Resumable: each
# step is skipped when its result exists. Run under nohup; log goes to $RD_HOME/install.log.
#
#   RD_HOME=/root/controlr-robodojo bash scripts/robodojo/install.sh [step...]
#   steps: conda clone deps isaacsim isaaclab curobo assets policy_env check   (default: all, in order)
set -euo pipefail
RD_HOME="${RD_HOME:-/root/controlr-robodojo}"
RD_REPO="${RD_REPO:-https://github.com/RoboDojo-Benchmark/RoboDojo.git}"
CONDA="$RD_HOME/miniconda3"
ENV="$CONDA/envs/robodojo"
SRC="$RD_HOME/RoboDojo"
export PIP_CACHE_DIR="$RD_HOME/.cache/pip" TMPDIR="$RD_HOME/.tmp" PIP_USER=0 PYTHONNOUSERSITE=1
export OMNI_KIT_ACCEPT_EULA=YES TERM=xterm-256color
mkdir -p "$RD_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

activate() { source "$CONDA/bin/activate" "$ENV"; }

step_conda() {
  if [[ ! -x "$CONDA/bin/conda" ]]; then
    log "miniconda -> $CONDA"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$TMPDIR/mc.sh"
    bash "$TMPDIR/mc.sh" -b -p "$CONDA"; rm -f "$TMPDIR/mc.sh"
    "$CONDA/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
    "$CONDA/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
  fi
  [[ -x "$ENV/bin/python" ]] || "$CONDA/bin/conda" create -y -p "$ENV" python=3.11
}

step_clone() {
  if [[ ! -d "$SRC/.git" ]]; then
    log "clone RoboDojo"
    git clone --depth 1 "$RD_REPO" "$SRC"
  fi
  cd "$SRC"
  # RoboDojo's own step (install.sh setup_submodules): IsaacLab + cuRobo forks, XPolicyLab.
  for sub in third_party/IsaacLab third_party/curobo XPolicyLab; do
    [[ -n "$(ls -A "$sub" 2>/dev/null)" ]] || git submodule update --init --remote --depth 1 "$sub"
  done
  git rev-parse HEAD > "$RD_HOME/robodojo_commit.txt"
  git submodule status >> "$RD_HOME/robodojo_commit.txt"
}

# The remaining software steps reuse RoboDojo's own install.sh functions (pins included).
run_rd() {
  # install.sh has no --only: source its functions and call the one step.
  activate; cd "$SRC"
  # shellcheck disable=SC1091
  ( set +u; source <(sed -n '/^# ── Entry point/q;p' scripts/install.sh); CURRENT_DIR="$SRC"; "setup_$1" )
}

step_deps()     { run_rd base_deps; }
step_isaacsim() { run_rd isaacsim; }
step_isaaclab() { run_rd isaaclab; }
step_curobo()   { run_rd curobo; }

step_assets() {
  # Every Assets folder except other configs' / seeds' layouts (init_assets.sh's REQUIRED_DIRS
  # is not enough: scenes load Assets/Room too). Add seeds to SPARSE when needed. A sparse git-lfs clone: the LFS batch API fetches ~100 files per
  # request, while per-file downloads (hf download) hit HF's anonymous rate limit after ~650
  # files. The LFS object store is dropped afterwards so the 38.9 GB are stored once.
  local SPARSE=("/Assets/Robots/" "/Assets/Object/" "/Assets/Material/" "/Assets/Room/" "/Assets/Background/"
                "/Assets/Sensor/" "/Assets/Traj/" "/Assets/Eval_Layout/RoboDojo/arx_x5/0/")
  cd "$RD_HOME"
  if [[ ! -d hf_git/.git ]]; then
    log "assets: clone (pointers only)"
    GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 --no-checkout \
      https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo hf_git
  fi
  cd hf_git
  git sparse-checkout init --no-cone
  printf '%s\n' "${SPARSE[@]}" > .git/info/sparse-checkout
  GIT_LFS_SKIP_SMUDGE=1 git checkout -q main
  local inc; inc="$(printf '%s**,' "${SPARSE[@]#/}")"
  for i in $(seq 1 30); do
    log "assets: lfs pull $i"
    git lfs pull --include "${inc%,}" && break
    sleep 300
  done
  rm -rf .git/lfs/objects
  ln -sfn "$RD_HOME/hf_git/Assets" "$SRC/Assets"
  # cuRobo robot configs (Assets/Robots/**/curobo.yml) are generated from *_tmp.yml with
  # absolute paths baked in; without them RoboDojo dies at robot setup.
  activate; cd "$SRC" && python utils/update_embodiment_config_path.py < /dev/null
}

step_policy_env() {
  # XPolicyLab's ws policy server needs websockets>=14, RoboDojo's env pins 12.0: like RoboDojo's
  # own policies, the (no-op) policy server gets its own small env.
  local UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
  [[ -x "$RD_HOME/policy_venv/bin/python" ]] || "$UV" venv -q --python 3.11 "$RD_HOME/policy_venv"
  "$UV" pip install -q --python "$RD_HOME/policy_venv/bin/python" "numpy>=1.23" "pyyaml>=6" \
    "websockets>=14" "msgpack>=1.0.8" "msgpack-numpy>=0.4.8" "pydantic>=2.5" opencv-python-headless h5py
}

step_check() {
  activate; cd "$SRC"
  python - <<'PY'
import importlib
for m in ("isaacsim", "isaaclab", "curobo", "msgpack_numpy", "websockets"):
    importlib.import_module(m); print("ok", m)
PY
  du -sh "$RD_HOME" 2>/dev/null; df -h / | tail -1
}

steps=("$@"); [[ ${#steps[@]} -gt 0 ]] || steps=(conda clone deps isaacsim isaaclab curobo assets policy_env check)
for s in "${steps[@]}"; do log "== $s"; "step_$s"; done
log "done"
