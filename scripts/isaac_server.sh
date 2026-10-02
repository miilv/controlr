#!/usr/bin/env bash
# Launch the controlr Isaac server inside Isaac Sim 6.0's bundled python (headless).
#
# Mirrors PHANTOM tools/sim/launch_waffles.sh: the stock python.sh overwrites
# LD_PRELOAD and loads an incompatible system NCCL before the bundled torch, so
# we source setup_python_env.sh ourselves and preload the bundled NCCL. Nothing
# in the shared Isaac installation or in PHANTOM is modified.
#
# Env: ISAAC_SIM_ROOT, PHANTOM_ROOT, CONTROLR_ISAAC_AUTHKEY (shared secret),
#      CONTROLR_ISAAC_READY_FILE (written once the socket is listening).
# Args are passed to controlr/robot/isaac/server.py (--port, --scene, --once ...).
set -euo pipefail
SIM_ROOT="${ISAAC_SIM_ROOT:-/home/physicalai/AAAI_MultiAgenticSIM/isaac-sim-6.0}"
PHANTOM_ROOT="${PHANTOM_ROOT:-/home/physicalai/phantom-icra-2027/phantom}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CARB_APP_PATH="$SIM_ROOT/kit" ISAAC_PATH="$SIM_ROOT" EXP_PATH="$SIM_ROOT/apps"
set +u
source "$SIM_ROOT/setup_python_env.sh"
set -u
export LD_PRELOAD="$SIM_ROOT/kit/libcarb.so:$SIM_ROOT/extsDeprecated/omni.isaac.ml_archive/pip_prebundle/nvidia/nccl/lib/libnccl.so.2"
# PHANTOM first (phantom.sim.*, tools.sim.*); controlr's isaac dir is put on
# sys.path by server.py itself (protocol.py / tasks.py are stdlib + numpy).
export PYTHONPATH="$PHANTOM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PHANTOM_ROOT
exec "$SIM_ROOT/kit/python/bin/python3" "$REPO_ROOT/controlr/robot/isaac/server.py" --phantom "$PHANTOM_ROOT" "$@"
