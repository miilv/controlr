# isr-lab (Mikhail's 4090 box, anywherevla@100.64.0.2; the Unix user is SHARED with his agents): the environment of
# every bridge process of ours. Source it, then run through scripts/dojo/isr_job.sh (the coord GPU lock).
#   source ~/data2/ilia/bridge/controlr/scripts/dojo/isr_env.sh
# Writes only under ~/data2/ilia. Mikhail's installs (~/data2/bench = robot-harness, ~/data2/controlr-robodojo =
# RoboDojo + Isaac 5.1 + policy_venv) are read-only for us: BENCH and RD_HOME point at our symlink farms over them
# (scripts/dojo/isr_farm.sh), so his unmodified launchers (bin/robot-start-j2 sources HARNESS_ENV) write the strict
# server's run dir, the shim copy and the eval client's result into our tree only.
ILIA=/home/anywherevla/data2/ilia
export DOJO_ILIA=1                                   # marks every process of ours (scripts/simlab/ilia_stop.sh)
export BENCH="$ILIA/bench" RD_HOME="$ILIA/rdhome"
export PYTHONDONTWRITEBYTECODE=1                     # imports through the farms never write __pycache__ into his trees
export CR="${CR:-$ILIA/bridge/controlr}"
# robot-harness launchers: BENCH / RD_HOME from our own env file; the eval client's HOME (Kit caches) in our tree
export HARNESS_ENV="$CR/scripts/dojo/isr_harness_env.sh" HARNESS_SHIM_HOME="$ILIA/home"
# home partition is full: HOME / XDG / temp / caches under ~/data2/ilia
export HOME="$ILIA/home" XDG_CACHE_HOME="$ILIA/home/.cache" XDG_DATA_HOME="$ILIA/home/.local/share" \
  XDG_CONFIG_HOME="$ILIA/home/.config" TMPDIR="$ILIA/tmp" UV_CACHE_DIR="$ILIA/.uv-cache" UV_PYTHON_DOWNLOADS=never
# the SESSION lock bridge_episode.sh / rdpy.sh take on fd 9 (the twin inherits it: SIMLAB_GPU=0 under
# --hold-gpu-lock). It is ours; the shared ~/data2/coord/gpu.lock is held AROUND it by isr_job.sh, so nothing of ours
# ever writes into the coord lock file.
export DOJO_GPU_LOCK="$ILIA/home/dojo-gpu.lock"
export DOJO_DATA="$ILIA/dojo-data" SIMLAB_KIT_ARGS="--portable-root $ILIA/kitroot"
# perception: the task-scoped catalogue subset (no visual.glb) and the 5.1-built background / colour sets (genuine
# 5.1 frames here; journal 2026-10-25-dojo-perception)
export DOJO_PERCEPTION_DATA="$ILIA/bridge/pdata"
export DOJO_CATALOG="$DOJO_PERCEPTION_DATA/catalog" DOJO_X5_DIR="$DOJO_PERCEPTION_DATA/catalog/robot/x5" \
  DOJO_BG_FOLDS="$DOJO_PERCEPTION_DATA/bg_folds_51" DOJO_COLOR_CALIB="$DOJO_PERCEPTION_DATA/catalog/color_calib_51.json"
# omniroute keys (OMNIROUTE_* only, mode 600, owner-approved): the agent (llmio) and the agentic perceiver
export DOJO_AGENT_DOTENV="$ILIA/.env" AGENTIC_ENV_FILE="$ILIA/.env"
# the box hard-resets under all-core load (robot-harness AGENTS.md rule 6): thread pools capped, twin with 4 envs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4 \
  J1_PERCEIVER_THREADS=4 OMNI_KIT_ACCEPT_EULA=YES PYOPENGL_PLATFORM=egl
# the twin's checker reads articulation link poses from physics, as the official rendered env's USD does
# (probe 2026-10-26: updateToUsd is on with the renderer, off in the physics-only kit)
export SIMLAB_USD_SYNC_LINKS=1
mkdir -p "$HOME" "$TMPDIR" "$DOJO_DATA/bridge/runs" "$ILIA/run" "$ILIA/kitroot"
