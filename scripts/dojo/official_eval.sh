#!/usr/bin/env bash
# The OFFICIAL path on our dev layouts: what the leaderboard runs, with our settled layouts (>= 100000) instead of
# Assets/Eval_Layout. One eval client per call, one task, N layouts, one driver per episode:
#   RoboDojo's own eval client     src/eval_client/main.py unchanged (dojo/simlab/dev_eval.py patches only SeedManager
#                                  to read OUR layouts; scripts/simlab/dev_eval.sh), Isaac 5.1 with RTX: reset, step
#                                  accounting, step_lim, success latch, _result.json + mp4s
#   robot-harness's strict J2 shim + server (Mikhail's code, UNMODIFIED, from a pinned tree $BENCH: the launcher's
#                                  md5 checks are repeated here): XPolicyLab policy harness_strict_j2, strict tier B,
#                                  joint_steps; --dev only for the server's own truth log (dev.jsonl), read after
#   ours                           the bridge driver (python -m dojo.bridge.driver: AgentPolicy / RuntimePolicy on a
#                                  preset, unchanged), the perception service (5.1 calibration), the twin it spawns
# Episodes run back to back in ONE eval client: the next driver's `reset` ends the previous episode on the server (the
# shim books it, an unfinished one as FAILED, and the eval client moves to its next layout), the last driver closes.
# The verdict is never read here: rows come from RoboDojo's _result.json after the eval client exits
# (python -m dojo.bridge.official rows OUT).
#
#   scripts/dojo/official_eval.sh --task T --layouts "100000 100001 ..." [--policy dojo.agent.policy:AgentPolicy]
#       [--preset submit_v1] [--config JSON] [--twin-envs 8] [--id NAME] [--out DIR] [--episode-max-min 25]
#
# Env (defaults for a dojo GPU box, ~/dojo-env.sh): RD_SRC (the box's RoboDojo install, read-only; default $RD_HOME),
#   OFFICIAL_RDHOME (our farm over it, scripts/dojo/rd_farm.sh; default ~/dojo-official-rdhome), BENCH (a robot-harness
#   tree at a pinned commit, with a COMMIT file; default ~/dojo-official-rh), DOJO_DATA, DOJO_PERCEPTION_DATA,
#   DOJO_GPU_LOCK (the HEAVY lock, held for the whole call; keep a call <= 30 min: ~5 short or ~3 long episodes).
# Perception uses the genuine-5.1 set (bg_folds_51, color_calib_51) unless DOJO_BG_FOLDS / DOJO_COLOR_CALIB are set.
# Everything runs in one systemd user scope (24G / 1400 %); only PIDs this script started are stopped.
# bash reads a script while it runs: the body is one brace group, read before any of it runs (integ_eval.sh's note).
{
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${1:-}" != --inner ]]; then
  # outer: the memory gate, the heavy lock, then this script again inside a systemd user scope
  export DOJO_DATA="${DOJO_DATA:-$HOME/dojo-data}"
  avail() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
  for i in $(seq 30); do
    [[ "$(avail)" -ge "${DOJO_MIN_AVAIL_GB:-25}" ]] && break
    [[ "$i" -eq 30 ]] && { echo "[official] MemAvailable $(avail) GB too low: refusing to start" >&2; exit 75; }
    echo "[official] MemAvailable $(avail) GB, waiting..." >&2; sleep 20
  done
  LOCK="${DOJO_GPU_LOCK:-$HOME/dojo-gpu.lock}"; exec 9>>"$LOCK"
  echo "[official] waiting for $LOCK: $(head -c 160 "$LOCK" | tr '\n' ' ')" >&2
  flock -w "${OFFICIAL_LOCK_WAIT_S:-10800}" 9 || { echo "[official] gave up waiting for $LOCK" >&2; exit 75; }
  printf 'pid=%s stream=official start=%s cmd=official_eval %s\n' "$$" "$(date -u +%FT%TZ)" "$*" > "$LOCK"
  UNIT="official-$(date -u +%H%M%S)-$$"
  systemd-run --user --scope -q --unit="$UNIT" -p MemoryMax="${OFFICIAL_MEM:-24G}" -p CPUQuota="${OFFICIAL_CPU:-1400%}" \
    bash "$HERE/scripts/dojo/official_eval.sh" --inner "$@" &
  SCOPE=$!
  trap 'kill -TERM "$SCOPE" 2>/dev/null' TERM INT
  wait "$SCOPE"; RC=$?
  flock -u 9; exec 9>&-
  sleep "${OFFICIAL_YIELD_S:-90}"              # the lock rule: yield before anyone of ours re-takes it
  exit "$RC"
fi
shift   # --inner

TASK="" LAYOUT_IDS="" POLICY="dojo.agent.policy:AgentPolicy" PRESET="submit_v1" CONFIG='{}' TWIN_ENVS=8 NAME="" OUT=""
EP_MAX_MIN=25
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;        --layouts) LAYOUT_IDS="$2"; shift 2 ;;
    --policy) POLICY="$2"; shift 2 ;;    --preset) PRESET="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;    --twin-envs) TWIN_ENVS="$2"; shift 2 ;;
    --id) NAME="$2"; shift 2 ;;          --out) OUT="$2"; shift 2 ;;
    --episode-max-min) EP_MAX_MIN="$2"; shift 2 ;;
    *) echo "[official] unknown option $1" >&2; exit 2 ;;
  esac
done
[[ -n "$TASK" && -n "$LAYOUT_IDS" ]] || { echo "[official] --task and --layouts are required" >&2; exit 2; }
export DOJO_DATA="${DOJO_DATA:-$HOME/dojo-data}"
RD_SRC="${RD_SRC:-${RD_HOME:-$HOME/dojo-robodojo}}"
FARM="${OFFICIAL_RDHOME:-$HOME/dojo-official-rdhome}"
export BENCH="${BENCH:-$HOME/dojo-official-rh}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ID="${NAME:-off-${TASK:0:12}-$STAMP}"
OUT="${OUT:-$DOJO_DATA/official/runs/$ID}"
mkdir -p "$OUT/layouts/$TASK" "$OUT/eps"
LOG="$OUT/official_eval.log"
log() { echo "[official $(date -u +%T)] $*" | tee -a "$LOG" >&2; }
PY="$HERE/.venv/bin/python"
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill -TERM -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null; done; wait 2>/dev/null; }
trap cleanup EXIT
trap 'exit 143' TERM INT          # (a TERM waits for the running foreground driver; to stop everything at once:
                                  #  systemctl --user kill official-<...>.scope, the unit this call runs in)

# 0. our layouts only (settled records >= 100000), staged so dev_eval.py sees exactly these, in id order
LAYOUTS_SRC="$DOJO_DATA/simlab/layouts/$TASK"
IDS=()
for lid in $(tr ' ,' '\n\n' <<< "$LAYOUT_IDS" | grep -v '^$' | sort -n); do
  [[ "$lid" =~ ^[0-9]+$ && "$lid" -ge 100000 ]] || { log "layout id $lid: dev layouts are >= 100000"; exit 2; }
  f="$LAYOUTS_SRC/$lid.settled.json"
  [[ -f "$f" && "$f" != *Eval_Layout* ]] || { log "no settled dev layout $f"; exit 2; }
  cp "$f" "$OUT/layouts/$TASK/"
  IDS+=("$lid")
done
N=${#IDS[@]}
log "task $TASK, $N layouts: ${IDS[*]}; policy $POLICY preset $PRESET config $CONFIG; out $OUT"

# 1. our RoboDojo farm + robot-harness's shim in it (his launcher's own md5 checks, constants read from his launcher)
bash "$HERE/scripts/dojo/rd_farm.sh" "$RD_SRC" "$FARM" >> "$LOG" 2>&1 || { log "rd_farm failed"; exit 1; }
L2="$BENCH/harness/robodojo_launch_j2.sh"
[[ -f "$L2" && -f "$BENCH/harness/robodojo_server_j2.py" ]] || { log "BENCH=$BENCH is not a robot-harness tree with J2"; exit 2; }
BASE_MD5="$(grep -o 'BASE_MD5=[0-9a-f]*' "$L2" | cut -d= -f2)" STRICT_MD5="$(grep -o 'STRICT_MD5=[0-9a-f]*' "$L2" | cut -d= -f2)"
[[ "$(md5sum < "$BENCH/harness/rd_shim/deploy.py" | cut -c1-32)" == "$BASE_MD5" ]] || { log "rd_shim/deploy.py md5 != launcher's"; exit 2; }
[[ "$(md5sum < "$BENCH/harness/rd_shim_strict/deploy.py" | cut -c1-32)" == "$STRICT_MD5" ]] || { log "rd_shim_strict/deploy.py md5 != launcher's"; exit 2; }
POL="$FARM/RoboDojo/XPolicyLab/policy/harness_strict_j2"
mkdir -p "$POL"
cp "$BENCH/harness/rd_shim_strict_v2/"{__init__.py,deploy.py,model.py,deploy.yml} "$POL/"
cp "$BENCH/harness/rd_shim_strict/deploy.py" "$POL/strict_deploy.py"
cp "$BENCH/harness/rd_shim/deploy.py" "$POL/base_deploy.py"
SPY="$FARM/policy_venv/bin/python"
"$PY" - "$OUT/provenance.json" "$BENCH" "$HERE" "$POL" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, bench, here, pol = sys.argv[1:5]
md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()
d = {"robot_harness_commit": (Path(bench) / "COMMIT").read_text().strip() if (Path(bench) / "COMMIT").exists() else None,
     "controlr": json.loads((Path(here) / ".deployed.json").read_text()) if (Path(here) / ".deployed.json").exists() else None,
     "shim_md5": {p.name: md5(p) for p in sorted(Path(pol).glob("*.py"))},
     "server_md5": {n: md5(Path(bench) / "harness" / n) for n in ("robodojo_server_j2.py", "robodojo_server_j1.py",
                                                                   "robodojo_server_strict.py", "robodojo_server.py")}}
Path(out).write_text(json.dumps(d, indent=1))
PY

# 2. robot-harness's strict J2 server (his launcher's environment; the J1 perceiver = our adapter, unused on the local
#    perception route, warmed CPU-only)
RUNDIR="$BENCH/run/$ID"; mkdir -p "$RUNDIR"
printf '{"id": "%s", "kind": "robodojo", "strict": true, "j1": true, "j2": true, "privileged": false, "dev": true, "ready": false, "started_t": %s}\n' \
  "$ID" "$(date +%s)" > "$RUNDIR/meta.json"
export HARNESS_SHIM_ADDR="$RUNDIR/shim.sock" HARNESS_SHIM_AUTHKEY="$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
export HARNESS_PRIVILEGED=0 HARNESS_DEV=1 HARNESS_STRICT=1
export HARNESS_SHIM_IDLE_S="$("$SPY" "$BENCH/harness/ask_pause.py" shim-idle)" || { log "ask_pause shim-idle failed"; exit 2; }
export J1_PERCEIVER="${J1_PERCEIVER:-dojo.perception.j1_adapter:perceive}" J1_PERCEIVER_PATH="${J1_PERCEIVER_PATH:-$HERE}"
export J1_PERCEIVER_PYTHON="${J1_PERCEIVER_PYTHON:-$PY}" J1_ORIG_HOME="$HOME" J1_PERCEIVER_CPU_ONLY="${J1_PERCEIVER_CPU_ONLY:-1}"
export J1_HELD_FALLBACK="${J1_HELD_FALLBACK:-no}"          # no operator here: never wait for one
export PYTHONDONTWRITEBYTECODE=1
( exec setsid nice -n 5 "$SPY" "$BENCH/harness/robodojo_server_j2.py" --id "$ID" --strict --dev --reserve 20 \
    --idle-timeout 3600 9>&- ) > "$RUNDIR/server.log" 2>&1 < /dev/null &
PIDS+=($!)
for _ in $(seq 120); do [[ -S "$BENCH/run/$ID.sock" ]] && break; sleep 0.5; done
[[ -S "$BENCH/run/$ID.sock" ]] || { log "server did not start: $(tail -5 "$RUNDIR/server.log")"; exit 1; }
log "strict J2 server up ($ID)"

# 3. RoboDojo's eval client on our staged layouts (dev_eval.sh: ws policy server + dev_eval.py; we hold the lock)
( cd "$HERE" && RD_HOME="$FARM" RD_ROOT="$FARM/RoboDojo" DOJO_GPU_LOCK_HELD=1 SIMLAB_DEV_LAYOUT_SUFFIX=.settled exec setsid \
    bash scripts/simlab/dev_eval.sh --task "$TASK" --policy harness_strict_j2 --eval-num "$N" \
    --layouts "$OUT/layouts" --out "$OUT/eval" 9>&- ) > "$OUT/dev_eval.log" 2>&1 < /dev/null &
EC=$!; PIDS+=($EC)
log "eval client pid $EC"

# 4. our perception service (genuine 5.1 calibration)
PD="${DOJO_PERCEPTION_DATA:-$HOME/dojo-perception-data}"
export DOJO_CATALOG="${DOJO_CATALOG:-$PD/catalog}" DOJO_X5_DIR="${DOJO_X5_DIR:-$PD/catalog/robot/x5}"
export DOJO_BG_FOLDS="${DOJO_BG_FOLDS:-$PD/bg_folds_51}" DOJO_COLOR_CALIB="${DOJO_COLOR_CALIB:-$PD/catalog/color_calib_51.json}"
export AGENTIC_ENV_FILE="${AGENTIC_ENV_FILE:-$HOME/controlr/.env}" DOJO_AGENT_DOTENV="${DOJO_AGENT_DOTENV:-$HOME/controlr/.env}"
export DOJO_AGENTIC_LOG="${DOJO_AGENTIC_LOG:-$OUT/agentic}"
key() { python3 -c 'import secrets; print(secrets.token_hex(16))'; }
port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'; }
export DOJO_PERCEPTION_AUTHKEY="$(key)" DOJO_PERCEPTION_PORT="$(port)" DOJO_PERCEPTION_READY="$OUT/perception.ready"
( cd "$HERE" && DOJO_ILIA=1 PYOPENGL_PLATFORM=egl exec setsid nice "$PY" -u -m dojo.runtime.perception_service 9>&- ) \
  > "$OUT/perception_service.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 150); do [[ -f "$DOJO_PERCEPTION_READY" ]] && break; sleep 2; done
[[ -f "$DOJO_PERCEPTION_READY" ]] || { log "perception service not ready: see perception_service.log"; exit 1; }
log "perception service up"

# 5. one driver per episode, in the eval client's layout order (ids ascending); the calibration fold of a dev layout is
#    the integ_eval one (crc32(id) % 2: the D0 half its backgrounds must not come from), so the pair compares like for like
k=0; RC=0
for lid in "${IDS[@]}"; do
  k=$((k + 1))
  FIN=(); [[ "$k" -eq "$N" ]] && FIN=(--close)
  fold="$(python3 -c 'import sys, zlib; print(zlib.crc32(sys.argv[1].encode()) % 2)' "$lid")"
  EPO="$OUT/eps/$lid"; mkdir -p "$EPO"
  log "episode $k/$N (layout $lid, fold $fold): driver ${FIN[*]}"
  ( cd "$HERE" && DOJO_ILIA=1 DOJO_AGENT_LOG="$EPO/agent" exec timeout --signal=TERM "$((EP_MAX_MIN * 60 + 600))" \
      "$PY" -u -m dojo.bridge.driver --server-id "$ID" --bench "$BENCH" --policy "$POLICY" --preset "$PRESET" \
      --config "$CONFIG" --perception local --verify bundle --twin spawn --twin-envs "$TWIN_ENVS" --hold-gpu-lock \
      --calib-fold "$fold" --keep-frames 20 --wait-episode 900 --out "$EPO" "${FIN[@]}" 9>&- ) >> "$EPO/driver.log" 2>&1
  rc=$?
  [[ "$rc" -eq 0 ]] || { RC=$rc; log "driver rc=$rc (layout $lid): see $EPO/driver.log"; }
  grep -a "^\[bridge\] $ID:" "$EPO/driver.log" | tail -1 | tee -a "$LOG" >&2
  kill -0 "$EC" 2>/dev/null || { log "eval client exited after $k episodes"; break; }
done
# a driver that died before --close leaves the server up: close it (RoboDojo books an open episode as failed)
if [[ -S "$BENCH/run/$ID.sock" ]]; then
  "$PY" -c 'import sys; from dojo.bridge.client import RobotServerClient as C; r = C(sys.argv[1], bench=sys.argv[2]).call("close"); print("close ok" if r.get("ok") else "close: " + str(r.get("error"))[:200])' \
    "$ID" "$BENCH" 2>&1 | tee -a "$LOG" >&2
fi

# 6. the eval client writes _result.json when its last episode is booked; then stop what we started
for _ in $(seq 180); do kill -0 "$EC" 2>/dev/null || break; sleep 2; done
kill -0 "$EC" 2>/dev/null && log "eval client still running after 360 s: stopping it"
RES="$(find "$OUT/eval" -name _result.json 2>/dev/null | head -1)"
[[ -n "$RES" ]] && cp "$RES" "$OUT/_result.json" && log "RoboDojo result -> $OUT/_result.json" || log "no _result.json under $OUT/eval"
mkdir -p "$OUT/server_run" && cp -r "$RUNDIR/." "$OUT/server_run/" 2>/dev/null
( cd "$HERE" && "$PY" -m dojo.bridge.official rows "$OUT" ) 2>&1 | tee -a "$LOG" >&2
log "done rc=$RC"
exit "$RC"
}
