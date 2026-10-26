#!/usr/bin/env bash
# Isaac 6.0 proxy render of the held-object frames (render_d0_proxy over <root>/<task>/<seed>/frame_*.json), on the
# LIGHT lanes (the lead's rule for this stream, 2026-10-26):
#   scripts/perception/held_render.sh <root> <task> [<task> ...]
# - one isaac6 job of ours at a time (~/dojo-data/perc_held/.render.lock);
# - holds TWO free light locks (~/dojo-gpu-light{,2,3,4}.lock) for its lifetime; waits (never preempts);
# - starts only at MemAvailable >= 24 GB and free VRAM >= 10 GB; scope MemoryMax 12G, CPUQuota 600 %;
# - aborts its own job below 8 GB MemAvailable. Stages are kept (--keep-stage): delete them when done.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="$1"; shift
avail_gb() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
vram_free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }
mkdir -p "$HOME/dojo-data/perc_held"
exec 7>>"$HOME/dojo-data/perc_held/.render.lock"
flock -n 7 || { echo "[held_render] another render of ours is running" >&2; exit 75; }
LOCKS=("$HOME/dojo-gpu-light.lock" "$HOME/dojo-gpu-light2.lock" "$HOME/dojo-gpu-light3.lock" "$HOME/dojo-gpu-light4.lock")
t0=$(date +%s); got=()
# all four lanes are polled every 0.5 s by a small helper on fds this shell holds (a flock on an inherited fd stays
# held after the helper exits): blocking on one lane or polling every 15 s lost every release to the g3 chains
fds=()
for L in "${LOCKS[@]}"; do exec {fd}>>"$L"; fds+=("$fd:$L"); done
picked=$(python3 - "$(( ${HELD_RENDER_WAIT_S:-14400} ))" "${fds[@]}" <<'PY'
import fcntl, os, sys, time
limit, specs = float(sys.argv[1]), sys.argv[2:]
fds = [int(s.split(":", 1)[0]) for s in specs]
path = {int(s.split(":", 1)[0]): s.split(":", 1)[1] for s in specs}
got, t0 = [], time.time()
while len(got) < 2 and time.time() - t0 < limit:
    for fd in fds:
        if fd in got:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            got.append(fd)
            with open(path[fd], "w") as f:
                f.write(f"pid={os.getppid()} stream=perc_held start={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                        "cmd=isaac6 render (waiting for a 2nd light slot)\n")
        except OSError:
            pass
        if len(got) >= 2:
            break
    time.sleep(0.5)
print(" ".join(map(str, got)))
PY
)
for g in "${fds[@]}"; do
  fd="${g%%:*}"
  if [[ " $picked " == *" $fd "* ]]; then got+=("$g"); else eval "exec ${fd}>&-"; fi
done
if [[ ${#got[@]} -lt 2 ]]; then
  for g in "${got[@]}"; do eval "exec ${g%%:*}>&-"; done
  echo "[held_render] gave up waiting for lanes" >&2; exit 75
fi
until [[ "$(avail_gb)" -ge 24 && "$(vram_free_mb)" -ge 10000 ]]; do
  (( $(date +%s) - t0 > ${HELD_RENDER_WAIT_S:-14400} )) && { echo "[held_render] gave up waiting for memory" >&2; exit 75; }
  sleep 15
done
for g in "${got[@]}"; do
  printf 'pid=%s stream=perc_held start=%s cmd=isaac6 render_d0_proxy %s (2 light slots)\n' "$$" "$(date -u +%FT%TZ)" "$*" > "${g#*:}"
done
echo "[held_render] took ${got[*]#*:}; MemAvailable $(avail_gb) GB, VRAM free $(vram_free_mb) MB" >&2
export SIMLAB_GPU=0             # the locks are held here
systemd-run --user --scope -q -p MemoryMax=12G -p CPUQuota=600% \
  "$HERE/scripts/simlab/isaac6.sh" dojo/simlab/render_d0_proxy.py --root "$ROOT" --tasks "$@" --keep-stage &
child=$!
trap 'kill -TERM "$child" 2>/dev/null; wait "$child"; exit 143' TERM INT
while kill -0 "$child" 2>/dev/null; do
  if [[ "$(avail_gb)" -lt 8 ]]; then
    echo "[held_render] MemAvailable $(avail_gb) GB < 8: aborting our job" >&2
    kill -TERM "$child" 2>/dev/null; sleep 10; kill -KILL "$child" 2>/dev/null
    wait "$child"; exit 77
  fi
  sleep 5
done
wait "$child"
