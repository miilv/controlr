#!/usr/bin/env bash
# Minimal RoboDojo for simlab G2b on a second machine (compute3): the software stack of
# scripts/robodojo/install.sh plus ONLY the assets the G2 tasks need (push_T, stack_blocks:
# robots, one room, the background, sensors, the table/ground materials, blocks, T, pad,
# camera stand), as a sparse git-lfs checkout (~1 GB instead of ~41 GB). No pip cache.
#
#   RD_HOME=~/dojo-robodojo bash scripts/simlab/install_min.sh [software|assets|pin <sha>|check]...
#   (default: software assets check)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RD_HOME="${RD_HOME:-$HOME/dojo-robodojo}"
SRC="$RD_HOME/RoboDojo"
export RD_HOME PIP_NO_CACHE_DIR=1
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
SPARSE=(
  /Assets/Robots/ /Assets/Room/Simple_Room_nolight/ /Assets/Background/brown_photostudio_02_4k.hdr /Assets/Sensor/
  /Assets/Material/material_0122/ /Assets/Material/material_0564/
  /Assets/Object/RoboDojo/Rigid/block/ /Assets/Object/RoboDojo/Rigid/t/ /Assets/Object/RoboDojo/Geometry/t_cushion/
  /Assets/Object/RoboDojo/Geometry/camera_stand/ /Assets/Object/RoboDojo/Clutter/clutter.yml
)

step_software() { bash "$HERE/scripts/robodojo/install.sh" conda clone deps isaacsim isaaclab curobo; }

step_pin() {   # check out the exact RoboDojo commit of the reference install (compute2)
  local sha="$1"; cd "$SRC"
  git fetch -q --depth 1 origin "$sha" && git checkout -q "$sha"
  git submodule update --init --depth 1 third_party/IsaacLab third_party/curobo XPolicyLab || true
  git rev-parse HEAD > "$RD_HOME/robodojo_commit.txt"; git submodule status >> "$RD_HOME/robodojo_commit.txt"
}

step_assets() {
  cd "$RD_HOME"
  [[ -d hf_git/.git ]] || GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 --no-checkout \
      https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo hf_git
  cd hf_git
  git sparse-checkout init --no-cone
  printf '%s\n' "${SPARSE[@]}" > .git/info/sparse-checkout
  GIT_LFS_SKIP_SMUDGE=1 git checkout -q main
  local inc; inc="$(printf '%s**,' "${SPARSE[@]#/}")"; inc="${inc//.hdr\*\*/.hdr}"; inc="${inc//.yml\*\*/.yml}"
  for i in $(seq 1 20); do log "lfs pull $i"; git lfs pull --include "${inc%,}" && break; sleep 120; done
  rm -rf .git/lfs/objects
  ln -sfn "$RD_HOME/hf_git/Assets" "$SRC/Assets"
  if [[ -x "$RD_HOME/miniconda3/envs/robodojo/bin/python" ]]; then
    (cd "$SRC" && "$RD_HOME/miniconda3/envs/robodojo/bin/python" utils/update_embodiment_config_path.py < /dev/null)
  fi
  du -sh "$RD_HOME/hf_git/Assets"
}

step_check() {
  "$RD_HOME/miniconda3/envs/robodojo/bin/python" -c "import isaacsim, isaaclab, curobo; print('ok isaacsim isaaclab curobo')"
  du -sh "$RD_HOME"; df -h "$HOME" | tail -1
}

steps=("$@"); [[ ${#steps[@]} -gt 0 ]] || steps=(software assets check)
while [[ ${#steps[@]} -gt 0 ]]; do
  s="${steps[0]}"; steps=("${steps[@]:1}")
  if [[ "$s" == pin ]]; then step_pin "${steps[0]}"; steps=("${steps[@]:1}"); else log "== $s"; "step_$s"; fi
done
log done
