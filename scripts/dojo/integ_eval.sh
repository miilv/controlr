#!/usr/bin/env bash
# PROXY end-to-end eval of dojo.runtime.RuntimePolicy on compute3 (one GPU job):
#   scripts/dojo/integ_eval.sh --task T --layouts L1.json ... [--preset NAME] [--mode gt|perception] [--verify off|bundle]
#        [--twin-envs 8] [--out DIR] [--config JSON] [--max-minutes 25]
# --preset (dojo/runtime/presets.py, e.g. submit_v1) supplies mode / verify unless given explicitly.
# --renderer isaac51 (a "full" box where Isaac 5.1 renders, e.g. compute4): episodes are RoboDojo EvalEnvs with RTX
# (dojo.runtime.eval's RenderedEpisode), no 6.0 render service; perception defaults to the 5.1 backgrounds / colours.
# Processes, all in ONE systemd user scope (24G / 14 cores with a twin or perception, else 16G / 12 cores):
#   render service (Isaac 6.0), [twin service (physics-only 5.1 Twin, --verify bundle)],
#   [perception service (controlr venv, EGL; --mode perception)], the driver (dojo.runtime.eval, RoboDojo python).
# Team rules: starts only with MemAvailable >= 25 GB (INTEG_MIN_MEM_GB; + free VRAM INTEG_MIN_FREE_MB; waits <= 10 min),
# holds $DOJO_GPU_LOCK (default ~/dojo-gpu.lock; compute5-8: Integration-3's ~/dojo-gpu-heavy2.lock) (blocking flock -w)
# for its lifetime (keep a job <= 30 min: --max-minutes), every child runs with SIMLAB_GPU=0; services stopped
# on exit; after releasing the lock it yields 90 s (INTEG_YIELD_S) before returning. Exit 3: a service never got
# ready (no episode ran), 75: resources / the lock never came: the queue runner retries such a job once.
# bash reads a script while it runs: a deploy that rewrites this file in place (cp, an editor) would make a
# running copy execute a mix of old and new lines (2026-10-26: the runner re-ran a finished job with mangled
# args, then died on a syntax error). The whole body is one brace group, read before any of it runs.
{
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RD_HOME="${RD_HOME:-$HOME/dojo-robodojo}" DOJO_DATA="${DOJO_DATA:-$HOME/dojo-data}"
OUT="$DOJO_DATA/integ/runs/$(date -u +%Y%m%dT%H%M%SZ)"
MODE="" VERIFY="" PRESET="" RENDERER=isaac60 TWIN_ENVS=8 TASK="" args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --mode) MODE="$2"; args+=("$1" "$2"); shift 2 ;;
    --verify) VERIFY="$2"; args+=("$1" "$2"); shift 2 ;;
    --preset) PRESET="$2"; args+=("$1" "$2"); shift 2 ;;
    --renderer) RENDERER="$2"; args+=("$1" "$2"); shift 2 ;;
    --twin-envs) TWIN_ENVS="$2"; shift 2 ;;
    --task) TASK="$2"; args+=("$1" "$2"); shift 2 ;;
    *) args+=("$1"); shift ;;
  esac
done
[[ -n "$TASK" ]] || { echo "--task is required" >&2; exit 2; }
for a in "${args[@]}"; do                       # D-C10: layout ids >= 100100 are reserved for joint paired runs
  b=${a##*/}; id=${b%%.*}
  if [[ "$a" == *.json && "$id" =~ ^[0-9]+$ ]] && (( id >= 100100 )); then
    echo "[integ] $a: a reserved layout (>= 100100, D-C10): refused" >&2; exit 2
  fi
done
if [[ -n "$PRESET" ]]; then        # the services follow the preset's mode / verify (explicit flags win, as in eval)
  read -r P_MODE P_VERIFY < <(python3 -c 'import runpy, sys; p = runpy.run_path(sys.argv[1])["PRESETS"][sys.argv[2]]
print(p.get("mode", "gt"), p.get("verify", "off"))' "$HERE/dojo/runtime/presets.py" "$PRESET") \
    || { echo "[integ] unknown preset $PRESET" >&2; exit 2; }
  MODE="${MODE:-$P_MODE}" VERIFY="${VERIFY:-$P_VERIFY}"
fi
MODE="${MODE:-gt}" VERIFY="${VERIFY:-off}"
mkdir -p "$OUT"
avail() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
vram() { command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ' || echo 999999; }
# resources before a start: MemAvailable >= INTEG_MIN_MEM_GB (25; 16 on the boxes with two heavy lanes) and, when
# INTEG_MIN_FREE_MB > 0, that much free VRAM (a rendered episode needs ~8 GB; Kit hangs when it runs out)
MIN_GB="${INTEG_MIN_MEM_GB:-25}" MIN_MB="${INTEG_MIN_FREE_MB:-0}"
ready() { [[ "$(avail)" -ge "$MIN_GB" ]] && { [[ "$MIN_MB" -le 0 ]] || [[ "$(vram)" -ge "$MIN_MB" ]]; }; }
wait_res() {
  for i in $(seq "${INTEG_RES_TRIES:-30}"); do
    ready && return 0
    [[ $((i % 3)) -eq 1 ]] && echo "[integ] waiting: MemAvailable $(avail) GB (need $MIN_GB), VRAM free $(vram) MiB (need $MIN_MB)" >&2
    sleep 20
  done
  ready
}
wait_res || { echo "[integ] MemAvailable $(avail) GB / VRAM free $(vram) MiB below $MIN_GB GB / $MIN_MB MiB: refusing to start" >&2; exit 75; }
LOCK="${DOJO_GPU_LOCK:-$HOME/dojo-gpu.lock}"; exec 9>>"$LOCK"
echo "[integ] waiting for $LOCK: $(head -c 160 "$LOCK" | tr '\n' ' ')" >&2
flock -w "${INTEG_LOCK_WAIT_S:-10800}" 9 || { echo "[integ] gave up waiting for $LOCK" >&2; exit 75; }
printf 'pid=%s stream=%s job=%s start=%s cmd=integ_eval %s\n' "$$" "${DOJO_STREAM:-integ}" "$(basename "$OUT")" "$(date -u +%FT%TZ)" "${args[*]}" > "$LOCK"
wait_res || { echo "[integ] resources dropped below $MIN_GB GB / $MIN_MB MiB while waiting for the lock" >&2; exit 75; }
key() { python3 -c 'import secrets; print(secrets.token_hex(16))'; }
port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'; }
export SIMLAB_GPU=0 SIMLAB_MIN_FREE_MB=0
export SIMLAB_RENDER_AUTHKEY="$(key)" SIMLAB_RENDER_PORT="$(port)" SIMLAB_RENDER_READY="$OUT/render.ready"
export SIMLAB_TWIN_AUTHKEY="$(key)" SIMLAB_TWIN_PORT="$(port)" SIMLAB_TWIN_READY="$OUT/twin.ready"
export DOJO_PERCEPTION_AUTHKEY="$(key)" DOJO_PERCEPTION_PORT="$(port)" DOJO_PERCEPTION_READY="$OUT/perception.ready"
PD="${DOJO_PERCEPTION_DATA:-$HOME/dojo-perception-data}"
export DOJO_CATALOG="${DOJO_CATALOG:-$PD/catalog}" DOJO_X5_DIR="${DOJO_X5_DIR:-$PD/catalog/robot/x5}"
if [[ "$RENDERER" == isaac51 ]]; then           # genuine 5.1 frames: the 5.1 backgrounds and colour map
  export DOJO_BG_FOLDS="${DOJO_BG_FOLDS:-$PD/bg_folds_51}" DOJO_COLOR_CALIB="${DOJO_COLOR_CALIB:-$PD/catalog/color_calib_51.json}"
else
  export DOJO_BG_FOLDS="${DOJO_BG_FOLDS:-$PD/bg_folds}" DOJO_COLOR_CALIB="${DOJO_COLOR_CALIB:-$PD/catalog/color_calib.json}"
fi
# agentic perceiver (only read when the config sets "agentic"): omniroute keys from the controlr .env, transcripts here
export AGENTIC_ENV_FILE="${AGENTIC_ENV_FILE:-$HOME/controlr/.env}" DOJO_AGENTIC_LOG="${DOJO_AGENTIC_LOG:-$OUT/agentic}"
export OUT HERE MODE VERIFY TWIN_ENVS TASK RENDERER
if [[ "$VERIFY" == bundle || "$MODE" == perception ]]; then MEM=24G CPU=1400%; else MEM=16G CPU=1200%; fi
# the scope runs in the background so a TERM / INT to this script can stop it: a killed integ_eval.sh used to leave
# the services and the driver running, still holding the GPU lock (fd 9) (2026-10-26, imitate_sorting)
UNIT="$(basename "$OUT")"; UNIT="integ-${UNIT//[^A-Za-z0-9_.-]/_}-$$"
systemd-run --user --scope -q --unit="$UNIT" -p MemoryMax=$MEM -p CPUQuota=$CPU bash -c '
  cg=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1)
  cpu0=$(awk "/usage_usec/ {print \$2}" $cg/cpu.stat); t0=$(date +%s.%N)
  svc=""; twin=""; perc=""
  if [[ "$RENDERER" != isaac51 ]]; then
    "$HERE/scripts/simlab/isaac6.sh" dojo/simlab/render_service.py > "$OUT/render_service.log" 2>&1 &
    svc=$!
  fi
  if [[ "$VERIFY" == bundle ]]; then
    "$HERE/scripts/simlab/rdpy.sh" -u -m dojo.simlab.twin_service --task "$TASK" --num-envs "$TWIN_ENVS" > "$OUT/twin_service.log" 2>&1 &
    twin=$!
  fi
  if [[ "$MODE" == perception ]]; then
    ( cd "$HERE" && PYOPENGL_PLATFORM=egl nice "$HERE/.venv/bin/python" -u -m dojo.runtime.perception_service ) > "$OUT/perception_service.log" 2>&1 &
    perc=$!
  fi
  for i in $(seq 200); do
    { [[ -z "$svc" ]] || [[ -f "$SIMLAB_RENDER_READY" ]]; } && { [[ -z "$twin" ]] || [[ -f "$SIMLAB_TWIN_READY" ]]; } \
      && { [[ -z "$perc" ]] || [[ -f "$DOJO_PERCEPTION_READY" ]]; } && break
    sleep 2
  done
  ok=1
  [[ -z "$svc" || -f "$SIMLAB_RENDER_READY" ]] || { echo "[integ] render service not ready: see render_service.log"; ok=0; }
  [[ -z "$twin" || -f "$SIMLAB_TWIN_READY" ]] || { echo "[integ] twin service not ready: see twin_service.log"; ok=0; }
  [[ -z "$perc" || -f "$DOJO_PERCEPTION_READY" ]] || { echo "[integ] perception service not ready: see perception_service.log"; ok=0; }
  [[ "$ok" == 1 ]] || touch "$OUT/services_not_ready"          # exit 3: the runner retries the job once
  if [[ "$ok" == 1 ]]; then
    "$HERE/scripts/simlab/rdpy.sh" -u -m dojo.runtime.eval --out "$OUT" "$@" 2>&1 | tee "$OUT/integ_eval.log" | grep -a "^\[integ\]"
  fi
  python3 - <<PY
import os
from multiprocessing.connection import Client
for p, k, on in (("SIMLAB_RENDER_PORT", "SIMLAB_RENDER_AUTHKEY", os.environ["RENDERER"] != "isaac51"),
                 ("SIMLAB_TWIN_PORT", "SIMLAB_TWIN_AUTHKEY", os.environ["VERIFY"] == "bundle"),
                 ("DOJO_PERCEPTION_PORT", "DOJO_PERCEPTION_AUTHKEY", os.environ["MODE"] == "perception")):
    if not on:
        continue
    try:
        c = Client(("127.0.0.1", int(os.environ[p])), authkey=bytes.fromhex(os.environ[k])); c.send({"op": "close"}); c.recv()
    except Exception as e:
        print(p, "close:", e)
PY
  sleep 5; kill $svc $twin $perc 2>/dev/null; wait $svc $twin $perc 2>/dev/null
  cpu1=$(awk "/usage_usec/ {print \$2}" $cg/cpu.stat); t1=$(date +%s.%N)
  peak=$(cat $cg/memory.peak 2>/dev/null || echo 0)
  python3 -c "import json,sys; c0,c1,t0,t1,pk=map(float,sys.argv[1:6]); json.dump({\"mem_peak_gb\": pk/1e9, \"cpu_cores_avg\": (c1-c0)/1e6/(t1-t0), \"wall_s\": t1-t0, \"mode\": sys.argv[6], \"verify\": sys.argv[7]}, open(sys.argv[8], \"w\"), indent=1)" $cpu0 $cpu1 $t0 $t1 $peak "$MODE" "$VERIFY" "$OUT/scope_stats.json"
  cat "$OUT/scope_stats.json"
' _ "${args[@]}" &
scope_pid=$!
cg_path() { local cg; cg=$(timeout 10 systemctl --user show -p ControlGroup --value "$UNIT.scope" 2>/dev/null); [[ -n "$cg" ]] && echo "/sys/fs/cgroup$cg"; }
kill_scope() {
  # TERM the scope, then SIGKILL whatever is left after INTEG_KILL_GRACE_S (90): a Kit app on a GUI dialog ignores TERM and
  # `systemctl --user stop` then hangs (compute8 2026-10-27: heavy lock 1 held 45 min until a manual kill -9). Every
  # systemctl call is bounded; the cgroup's own process list is the fallback.
  local cg; cg=$(cg_path)
  timeout 10 systemctl --user kill -s TERM "$UNIT.scope" 2>/dev/null; kill "$scope_pid" 2>/dev/null
  for i in $(seq "${INTEG_KILL_GRACE_S:-90}"); do
    [[ -n "$cg" && -s "$cg/cgroup.procs" ]] || break
    sleep 1
  done
  if [[ -n "$cg" && -s "$cg/cgroup.procs" ]]; then
    echo "[integ] scope $UNIT ignored TERM: SIGKILL" >&2
    timeout 10 systemctl --user kill -s KILL "$UNIT.scope" 2>/dev/null
    for p in $(cat "$cg/cgroup.procs" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  fi
  timeout 15 systemctl --user stop "$UNIT.scope" 2>/dev/null || true
}
trap 'echo "[integ] stopping scope $UNIT" >&2; kill_scope' TERM INT
# watchdog: a rendered episode can freeze inside a Kit step (5.1: a zenity "Failed to open display" every 2 min, no step for
# 44 min), where the driver's --episode-max-minutes check never runs. Hung when, with no env-step / episode progress in
# integ_eval.log for
#   INTEG_WATCHDOG_HARD_S (600): always;
#   INTEG_WATCHDOG_S (300): when Kit logged a GUI dialog (zenity / "Failed to open display") since the last progress, or
#     the whole scope burned under INTEG_WATCHDOG_CORES (0.5; planning keeps it busy).
# Then: kill the scope (TERM, SIGKILL after 90 s), an infra_failure row for the frozen layout, exit 4 (the runner re-queues
# the layouts left once).
WD_S="${INTEG_WATCHDOG_S:-300}" WD_HARD_S="${INTEG_WATCHDOG_HARD_S:-600}" WD_CORES="${INTEG_WATCHDOG_CORES:-0.5}" WD_FIRED=""
cg_usage() {
  local cg; cg=$(cg_path)
  if [[ -n "$cg" && -f "$cg/cpu.stat" ]]; then awk '/usage_usec/ {print $2}' "$cg/cpu.stat"; else echo 0; fi
}
progress() {          # "(episodes finished) (last env step)" and the GUI-dialog line count, from the driver's log
  python3 - "$OUT/integ_eval.log" <<'PY' 2>/dev/null
import re, sys
t = open(sys.argv[1], "rb").read().decode("utf-8", "replace")
s = re.findall(r"step: (?:\x1b\[[0-9;]*m)?(\d+) /", t[-40000:])
print(f"{t.count('[integ] ')}:{s[-1] if s else ''}", t.count("Failed to open display") + t.count("zenity"))
PY
}
wd_tok="" wd_zen=0 wd_t=$(date +%s) wd_ct=$(date +%s) wd_cpu=$(cg_usage)
while kill -0 "$scope_pid" 2>/dev/null; do
  sleep 20 & wait $!
  now=$(date +%s)
  if [[ ! -f "$OUT/integ_eval.log" ]]; then wd_t=$now wd_ct=$now wd_cpu=$(cg_usage); continue; fi    # services booting
  read -r tok zen < <(progress)
  if [[ "$tok" != "$wd_tok" ]]; then wd_tok=$tok wd_zen=${zen:-0} wd_t=$now wd_ct=$now wd_cpu=$(cg_usage); continue; fi
  el=$(( now - wd_t )); cel=$(( now - wd_ct ))
  (( el < WD_S )) && continue
  cpu=$(cg_usage)
  cores=$(python3 -c "print(round(($cpu - $wd_cpu) / 1e6 / max(1, $cel), 2))")
  why=""
  if (( el >= WD_HARD_S )); then why="no env step for ${el} s"
  elif (( ${zen:-0} > wd_zen )); then why="no env step for ${el} s, a Kit GUI dialog (zenity) since"
  elif (( cel >= WD_S )) && python3 -c "import sys; sys.exit(0 if $cores < $WD_CORES else 1)"; then
    why="no env step for ${el} s at $cores cores"
  fi
  if [[ -n "$why" ]]; then
    WD_FIRED="$why (at ${tok}), scope CPU $cores cores"
    echo "[integ] watchdog: $WD_FIRED: killing the scope" >&2
    kill_scope
    break
  fi
  if (( cel >= WD_S )); then wd_ct=$now wd_cpu=$cpu; fi   # busy (planning): a new CPU window, the hard limit keeps counting
done
wait "$scope_pid"
trap - TERM INT
if [[ -n "$WD_FIRED" ]]; then
  python3 - "$OUT" "$TASK" "$MODE" "$VERIFY" "$WD_FIRED" "${args[@]}" <<'PY'
import glob, json, os, sys
out, task, mode, verify, why = sys.argv[1:6]
rest = sys.argv[6:]
lays = rest[rest.index("--layouts") + 1:] if "--layouts" in rest else []
lays = lays[:next((i for i, a in enumerate(lays) if a.startswith("--")), len(lays))]
f = os.path.join(out, f"{task}_{mode}_verify-{verify}_episodes.jsonl")
done = {json.loads(ln).get("layout") for g in glob.glob(os.path.join(out, "*_episodes.jsonl")) for ln in open(g) if ln.strip()}
hung = next((lf for lf in lays if lf not in done), None)
if hung:
    with open(f, "a") as fh:
        fh.write(json.dumps({"task": task, "layout": hung, "layout_id": int(os.path.basename(hung).split(".")[0]), "mode": mode,
                             "verify": verify, "success": False, "score": 0.0, "steps_used": 0, "step_lim": 0, "wall_s": 0.0,
                             "category": "infra_failure", "why": f"watchdog: {why}", "agentic": []}) + "\n")
    print(f"[integ] watchdog: {os.path.basename(hung)} marked infra_failure")
PY
  touch "$OUT/watchdog_fired"
fi
# a service's python child can outlive the kill of its wrapper (rdpy.sh / isaac6.sh) and keeps the scope, and the GPU
# lock fd it inherited, alive: 2026-10-26 a twin that never got ready held ~/dojo-gpu.lock for 30 min after its job
[[ -n "$(cg_path)" && -s "$(cg_path)/cgroup.procs" ]] && kill_scope
timeout 15 systemctl --user stop "$UNIT.scope" 2>/dev/null || true
rm -f "$OUT/stage_current.usd"; rm -rf "$OUT/work"          # scratch (37 MB stage export per job; disk is shared)
echo "[integ] results: $OUT"
exec 9>&-
sleep "${INTEG_YIELD_S:-90}"
[[ -f "$OUT/services_not_ready" ]] && exit 3
[[ -f "$OUT/watchdog_fired" ]] && exit 4
exit 0
}
