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
# Team rule: ONE GPU-heavy job per machine. Every GPU run holds flock on $DOJO_GPU_LOCK
# (default ~/dojo-gpu.lock; /root/dojo-gpu.lock on compute2) for the life of the python process
# (fd 9 is inherited through exec), with "pid stream start" written into the file.
# SIMLAB_GPU=0 (CPU-only tools: gen_layouts, check_assets, pytest) skips the lock and the VRAM check.
if [[ "${SIMLAB_GPU:-1}" != 0 ]]; then
  LOCK="${DOJO_GPU_LOCK:-$HOME/dojo-gpu.lock}"
  # Fairness: our several drivers (coverage, d0_ext, demos) queue on a simlab-only lock first, so at most
  # ONE simlab job waits on the team lock at a time (flock wakes waiters at random). Default: on for the
  # simlab deploy dir only; other streams calling these scripts are unaffected.
  QL="${SIMLAB_QUEUE_LOCK:-auto}"
  if [[ "$QL" == auto ]]; then [[ "$HERE" == */dojo-simlab ]] && QL="${DOJO_DATA:-$HOME/dojo-data}/simlab/.queue.lock" || QL=""; fi
  if [[ -n "$QL" ]]; then exec 8>>"$QL"; flock -w "${SIMLAB_LOCK_WAIT_S:-43200}" 8 || exit 75; fi
  exec 9>>"$LOCK"
  if ! flock -n 9; then        # blocking wait: woken on release (30 s polling lost to back-to-back jobs)
    echo "[rdpy] GPU lock $LOCK held by: $(head -c 200 "$LOCK"), waiting..." >&2
    flock -w "${SIMLAB_LOCK_WAIT_S:-43200}" 9 || { echo "[rdpy] gave up waiting for $LOCK" >&2; exit 75; }
  fi
  printf 'pid=%s stream=simlab start=%s cmd=%s\n' "$$" "$(date -u +%FT%TZ)" "$*" > "$LOCK"
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
fi
exec "${RD_PYTHON:-$RD_HOME/miniconda3/envs/robodojo/bin/python}" "$@"
