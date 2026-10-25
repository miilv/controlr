#!/usr/bin/env bash
# Run a script in Isaac Sim 6.0's bundled python on compute3 (render proxy for RoboDojo's 5.1).
#   scripts/simlab/isaac6.sh dojo/simlab/render_d0_proxy.py --root ~/dojo-data/d0 --tasks stack_blocks
# Same environment setup as scripts/isaac_server.sh (bundled NCCL preload); nothing in the shared
# Isaac install is modified. Holds the team GPU lock (~/dojo-gpu.lock) for the life of the process.
set -euo pipefail
SIM_ROOT="${ISAAC_SIM_ROOT:-/home/physicalai/AAAI_MultiAgenticSIM/isaac-sim-6.0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${SIMLAB_GPU:-1}" != 0 ]]; then        # SIMLAB_GPU=0: a parent already holds the lock
  LOCK="${DOJO_GPU_LOCK:-$HOME/dojo-gpu.lock}"
  # Fairness: our several drivers (coverage, d0_ext, demos) queue on a simlab-only lock first, so at most
  # ONE simlab job waits on the team lock at a time (flock wakes waiters at random). Default: on for the
  # simlab deploy dir only; other streams calling these scripts are unaffected.
  QL="${SIMLAB_QUEUE_LOCK:-auto}"
  if [[ "$QL" == auto ]]; then [[ "$HERE" == */dojo-simlab ]] && QL="${DOJO_DATA:-$HOME/dojo-data}/simlab/.queue.lock" || QL=""; fi
  if [[ -n "$QL" ]]; then exec 8>>"$QL"; flock -w "${SIMLAB_LOCK_WAIT_S:-43200}" 8 || exit 75; fi
  exec 9>>"$LOCK"
  if ! flock -n 9; then        # blocking wait: woken on release (30 s polling lost to back-to-back jobs)
    echo "[isaac6] GPU lock $LOCK held by: $(head -c 200 "$LOCK"), waiting..." >&2
    flock -w "${SIMLAB_LOCK_WAIT_S:-43200}" 9 || { echo "[isaac6] gave up waiting for $LOCK" >&2; exit 75; }
  fi
  printf 'pid=%s stream=simlab start=%s cmd=isaac6 %s\n' "$$" "$(date -u +%FT%TZ)" "$*" > "$LOCK"
fi
export CARB_APP_PATH="$SIM_ROOT/kit" ISAAC_PATH="$SIM_ROOT" EXP_PATH="$SIM_ROOT/apps" OMNI_KIT_ACCEPT_EULA=YES
set +u
source "$SIM_ROOT/setup_python_env.sh"
set -u
export LD_PRELOAD="$SIM_ROOT/kit/libcarb.so:$SIM_ROOT/extsDeprecated/omni.isaac.ml_archive/pip_prebundle/nvidia/nccl/lib/libnccl.so.2"
script="$1"; shift
[[ "$script" = /* ]] || script="$HERE/$script"
exec "$SIM_ROOT/kit/python/bin/python3" "$script" "$@"
