#!/usr/bin/env bash
# Start the SO-101 server on the rig (the machine the arm and the wrist camera are plugged into).
# It runs in LeRobot's own python and needs only controlr/robot/so101/{server,protocol,arm_id}.py.
#   scripts/so101_server.sh                 # real arm + camera (defaults below)
#   scripts/so101_server.sh --fake          # no hardware
# Env: SO101_PYTHON (LeRobot's python), SO101_PORT, SO101_ROBOT_ID, SO101_CAMERA,
#      CONTROLR_SO101_AUTHKEY (shared with the client). Extra args go to server.py.
# Ctrl-C once = freeze the arm in place; twice within 3 s = torque off (the arm falls) + exit.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${SO101_PYTHON:-$HOME/miniconda3/envs/lerobot/bin/python}"
# both CH340s share one by-id name and ttyUSB0/1 swap on re-plug: auto = the server reads every
# /dev/ttyUSB* and picks the follower by its encoders against its calibration file (arm_id.py).
# An explicit port whose motors do not hold follower_v1's calibration is refused.
PORT="${SO101_PORT:-auto}"
ROBOT_ID="${SO101_ROBOT_ID:-follower_v1}"
CAMERA="${SO101_CAMERA:-/dev/v4l/by-id/usb-WebCamera_WebCamera_2502131600-video-index0}"
if [[ " $* " == *" --fake "* ]]; then
  exec "$PY" "$HERE/controlr/robot/so101/server.py" "$@"
fi
if command -v fuser >/dev/null && fuser "$CAMERA" >/dev/null 2>&1; then
  echo "camera $CAMERA is in use by another program (fuser -v $CAMERA); close it first" >&2
  exit 1
fi
exec "$PY" "$HERE/controlr/robot/so101/server.py" --port "$PORT" --robot-id "$ROBOT_ID" --camera "$CAMERA" "$@"
