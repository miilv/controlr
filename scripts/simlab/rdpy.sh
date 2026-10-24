#!/usr/bin/env bash
# Run a python module/script in RoboDojo's conda env with RoboDojo + XPolicyLab + this repo importable.
#   scripts/simlab/rdpy.sh -m dojo.simlab.runner --task stack_blocks --layouts ... --out ...
# Env: RD_HOME (default /root/controlr-robodojo). Never cd's into the shared install.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RD_HOME="${RD_HOME:-/root/controlr-robodojo}"
RD="${RD_ROOT:-$RD_HOME/RoboDojo}"
export RD_HOME RD_ROOT="$RD"
export PYTHONPATH="$HERE:$RD:$RD/XPolicyLab:${PYTHONPATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES
# Shared GPU: Isaac needs ~6 GB for one env + RTX; Kit hangs on a dialog when it runs out.
# Wait until SIMLAB_MIN_FREE_MB (default 9000) is free; SIMLAB_MIN_FREE_MB=0 skips the check.
need="${SIMLAB_MIN_FREE_MB:-9000}"
if [[ "$need" -gt 0 ]] && command -v nvidia-smi >/dev/null; then
  for i in $(seq 360); do
    free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    [[ "$free" -ge "$need" ]] && break
    [[ $((i % 6)) -eq 1 ]] && echo "[rdpy] GPU free ${free} MiB < ${need} MiB, waiting..." >&2
    sleep 10
  done
fi
exec "${RD_PYTHON:-$RD_HOME/miniconda3/envs/robodojo/bin/python}" "$@"
