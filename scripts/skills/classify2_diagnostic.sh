#!/usr/bin/env bash
# Physics only. Private data and fixed diagnostic ids; never launches a camera.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RD_HOME="${RD_HOME:-$HOME/dojo-robodojo}"
export DOJO_DATA="$HOME/controlr-classify2-data"
export DOJO_CATALOG="$HOME/dojo-perception-data/catalog"
export SIMLAB_GPU=0
[[ ! -e "$DOJO_DATA/simlab" ]] || { echo "diagnostic data already exists; refusing overwrite" >&2; exit 2; }
mkdir -p "$DOJO_DATA"
exec 9>>"$HOME/dojo-gpu-light2.lock"
flock -w 60 9
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
free_kb=$(df -Pk "$DOJO_DATA" | awk 'NR==2 {print $4}')
[[ "$used" -le 24576 && "$free_kb" -ge 10485760 ]] || exit 75
printf 'pid=%s stream=classify2\n' "$$" > "$HOME/dojo-gpu-light2.lock"
# Child runs in one bounded resource scope; inherited lock covers its lifetime.
exec systemd-run --user --scope -q -p MemoryMax=8G -p CPUQuota=600% \
  bash "$PWD/scripts/skills/classify2_worker.sh"
