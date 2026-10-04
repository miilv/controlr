#!/usr/bin/env bash
# Install RoboDojo (Isaac Sim 5.1 + Isaac Lab + cuRobo + eval assets) under ONE directory on a
# shared host (compute2). Everything — Miniconda, the conda env, pip cache, the RoboDojo clone and
# its assets — lives in $RD_HOME; nothing touches ~/.bashrc or a shared conda. Resumable: each
# step is skipped when its result exists. Run under nohup; log goes to $RD_HOME/install.log.
#
#   RD_HOME=/root/controlr-robodojo bash scripts/robodojo/install.sh [step...]
#   steps: conda clone deps isaacsim isaaclab curobo assets check   (default: all, in order)
set -euo pipefail
RD_HOME="${RD_HOME:-/root/controlr-robodojo}"
RD_REPO="${RD_REPO:-https://github.com/RoboDojo-Benchmark/RoboDojo.git}"
CONDA="$RD_HOME/miniconda3"
ENV="$CONDA/envs/robodojo"
SRC="$RD_HOME/RoboDojo"
export PIP_CACHE_DIR="$RD_HOME/.cache/pip" TMPDIR="$RD_HOME/.tmp" PIP_USER=0 PYTHONNOUSERSITE=1
export HF_HOME="$RD_HOME/.cache/hf" OMNI_KIT_ACCEPT_EULA=YES TERM=xterm-256color
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
  # Only the folders evaluation needs (init_assets.sh REQUIRED_DIRS). hf download into a plain
  # directory: no git-lfs object store, so the 38.9 GB are stored once.
  activate
  cd "$SRC"
  if [[ ! -d Assets/Eval_Layout ]]; then
    log "assets -> $SRC/Assets"
    pip install -q "huggingface_hub[cli]>=0.30"
    hf download RoboDojo-Benchmark/RoboDojo --repo-type dataset --local-dir "$RD_HOME/hf_assets" \
      --include "Assets/Robots/**" "Assets/Object/**" "Assets/Material/**" "Assets/Eval_Layout/**"
    ln -sfn "$RD_HOME/hf_assets/Assets" Assets
  fi
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

steps=("$@"); [[ ${#steps[@]} -gt 0 ]] || steps=(conda clone deps isaacsim isaaclab curobo assets check)
for s in "${steps[@]}"; do log "== $s"; "step_$s"; done
log "done"
