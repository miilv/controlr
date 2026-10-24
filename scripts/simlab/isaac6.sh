#!/usr/bin/env bash
# Run a script in Isaac Sim 6.0's bundled python on compute3 (render proxy for RoboDojo's 5.1).
#   scripts/simlab/isaac6.sh dojo/simlab/render_d0_proxy.py --root ~/dojo-data/d0 --tasks stack_blocks
# Same environment setup as scripts/isaac_server.sh (bundled NCCL preload); nothing in the shared
# Isaac install is modified. Holds the team GPU lock (~/dojo-gpu.lock) for the life of the process.
set -euo pipefail
SIM_ROOT="${ISAAC_SIM_ROOT:-/home/physicalai/AAAI_MultiAgenticSIM/isaac-sim-6.0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCK="${DOJO_GPU_LOCK:-$HOME/dojo-gpu.lock}"
exec 9>>"$LOCK"
for i in $(seq 1440); do
  flock -n 9 && break
  [[ $((i % 10)) -eq 1 ]] && echo "[isaac6] GPU lock $LOCK held by: $(tr '\n' ' ' < "$LOCK"), waiting..." >&2
  sleep 30
done
flock -n 9 || { echo "[isaac6] gave up waiting for $LOCK" >&2; exit 75; }
printf 'pid=%s stream=simlab start=%s cmd=isaac6 %s\n' "$$" "$(date -u +%FT%TZ)" "$*" > "$LOCK"
export CARB_APP_PATH="$SIM_ROOT/kit" ISAAC_PATH="$SIM_ROOT" EXP_PATH="$SIM_ROOT/apps" OMNI_KIT_ACCEPT_EULA=YES
set +u
source "$SIM_ROOT/setup_python_env.sh"
set -u
export LD_PRELOAD="$SIM_ROOT/kit/libcarb.so:$SIM_ROOT/extsDeprecated/omni.isaac.ml_archive/pip_prebundle/nvidia/nccl/lib/libnccl.so.2"
script="$1"; shift
[[ "$script" = /* ]] || script="$HERE/$script"
exec "$SIM_ROOT/kit/python/bin/python3" "$script" "$@"
