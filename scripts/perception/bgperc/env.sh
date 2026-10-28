# source me (stream C dev env on a box): the perception python + data, CPU silhouettes (no GPU job)
PD="${DOJO_PERCEPTION_DATA:-$HOME/dojo-perception-data}"
export DOJO_PERCEPTION_DATA="$PD" DOJO_CATALOG="${DOJO_CATALOG:-$PD/catalog}" DOJO_X5_DIR="${DOJO_X5_DIR:-$PD/catalog/robot/x5}"
export DOJO_BG_FOLDS="${DOJO_BG_FOLDS:-$PD/bg_folds}" DOJO_COLOR_CALIB="${DOJO_COLOR_CALIB:-$PD/catalog/color_calib.json}"
if [ "${BGPERC_GL:-0}" = 1 ]; then export DOJO_GL=1 PYOPENGL_PLATFORM=egl; else export DOJO_GL="${DOJO_GL:-0}"; fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-4}"
HERE_C="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$HERE_C${BGPERC_EXTRA_PATH:+:$BGPERC_EXTRA_PATH}"
PY="${BGPERC_PY:-$HOME/controlr-identity/.venv/bin/python}"
export DOJO_AGENTIC_LOG="${DOJO_AGENTIC_LOG:-$HOME/dojo-data/bgperc/agentic_log}"
