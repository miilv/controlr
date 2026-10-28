#!/usr/bin/env bash
# Stream C, X2: write_rerun.py with the recorded runs' code and perception environment (integ_eval.sh's, genuine 5.1:
# bg_folds_51, color_calib_51, the frozen calibration set), EGL on this box's GPU, on one light slot (flock -n: a busy
# slot exits 75), nice'd in a user scope. Never the agentic env: refine calls no model.
#   CODE=~/dojo-integ-c4-8dae375 LOCK=~/dojo-gpu-light.lock scripts/perception/bgperc/write_rerun.sh SEQ_ROOT --out OUT.json [...]
#   (SCRIPT=x2_eval.py: the X2 harness in the same environment; PY=: another python, e.g. the recorded runs' venv
#   with CODE= a code export without one)
# Starts only with >= 4 GB free VRAM and <= 3 rendered processes (>= 3 GB) on the GPU (compute4's caps), else exits 75.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../../.." && pwd)"
CODE="${CODE:?the code dir of the recorded runs, with its .venv}"
LOCK="${LOCK:?a light GPU slot lock}"
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
rendered=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '$1>=3000' | wc -l)
if (( free < 4096 || rendered > 3 )); then
  echo "[write_rerun] GPU busy: ${free} MiB free, ${rendered} rendered processes" >&2
  exit 75
fi
PD="${DOJO_PERCEPTION_DATA:-$HOME/dojo-perception-data}"
export DOJO_CATALOG="${DOJO_CATALOG:-$PD/catalog}" DOJO_X5_DIR="${DOJO_X5_DIR:-$PD/catalog/robot/x5}"
export DOJO_BG_FOLDS="${DOJO_BG_FOLDS:-$PD/bg_folds_51}" DOJO_COLOR_CALIB="${DOJO_COLOR_CALIB:-$PD/catalog/color_calib_51.json}"
if [[ -f "$CODE/scripts/dojo/calib_set.sh" ]]; then source "$CODE/scripts/dojo/calib_set.sh"; fi   # (refine reads none of it)
unset AGENTIC_ENV_FILE DOJO_AGENTIC_LOG
export PYOPENGL_PLATFORM=egl PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec flock -n -E 75 "$LOCK" systemd-run --user --scope -q -p MemoryMax=8G -p CPUQuota=400% \
  nice -n 10 "${PY:-$CODE/.venv/bin/python}" -u "$HERE/scripts/perception/bgperc/${SCRIPT:-write_rerun.py}" "$@"
