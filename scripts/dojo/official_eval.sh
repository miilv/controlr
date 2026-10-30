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
#       (rendered measurement uses 100000-09; 100030-49 are the training block and >= 100100 reserved: both refused)
#       [--preset submit_v2] [--config JSON] [--twin-envs 8] [--id NAME] [--out DIR] [--episode-max-min 25]
#       [--candidate CANDIDATE.txt] [--setenv K=V]... [--seg V] [--glyph M] [--glyph-kong M] [--dry-run]
#   --candidate: Integration's CANDIDATE.txt verbatim (its non-comment lines are this script's arguments, split on
#   whitespace exactly as lane2.sh splits a queue line); with it the policy defaults to RuntimePolicy (integ_eval's).
#   --seg / --glyph / --glyph-kong: side servers started the way lane2.sh starts them (model dirs under
#   DOJO_SEG_MODELS, default ~/dojo-data/segtrain/models; this deploy's .venv); one that does not come up stops the call
#   BEFORE robot-harness's server or the eval client starts (fail loud: a scored run never degrades silently).
#   --setenv K=V: set for OUR processes only (perception service, side servers' clients, drivers), never for robot-
#   harness's. --dry-run: print the resolved policy / RuntimeConfig / env / side-server weights (md5) as JSON and exit
#   (no lock, nothing started; scripts/dojo/candidate_parity.py, scripts/dojo/freeze_manifest.py).
#   --raw-ok (dev only, default off): a dev id with no <id>.settled.json runs from its raw <id>.json (make_toast and the
#   conveyors have raw dev records only, as their measurement sets); one call is all settled or all raw.
#   --sealed-block DIR --manifest FILE --lock LOCKFILE: a sealed joint block (robot-harness D-C19; dojo.simlab.sealed):
#   only the ids the manifest lists for the task (all >= 100100), each md5-verified before it is staged into this run's
#   layout dir (nowhere else; logs name id + md5 + pass / fail only); raw records where the manifest says
#   "settled": false; robot-harness's server WITHOUT --dev (no truth log, HARNESS_DEV=0); the lock is required (no
#   default: the joint run's lock is chosen per D-C19 scheduling: ~/coord/dc19.lock per episode). Before anything starts it
#   takes the run-once claim, an O_EXCL marker per (task, id) in $COORD_DIR/scored/<block>/ (COORD_DIR required); a
#   second claim refuses. Manifest schema dc19-sealed/1 (tranches, ids 100100-100119; tranche B only once pre-registered).
#   Every other call keeps refusing ids >= 100100.
#
# Env (defaults for a dojo GPU box, ~/dojo-env.sh): RD_SRC (the box's RoboDojo install, read-only; default $RD_HOME),
#   OFFICIAL_RDHOME (our farm over it, scripts/dojo/rd_farm.sh; default ~/dojo-official-rdhome), BENCH (a robot-harness
#   tree at a pinned commit, with a COMMIT file; default ~/dojo-official-rh), DOJO_DATA, DOJO_PERCEPTION_DATA,
#   DOJO_GPU_LOCK (the HEAVY lock, held for the whole call; keep a call <= 30 min: ~5 short or ~3 long episodes).
# Perception uses the genuine-5.1 set (bg_folds_51, color_calib_51) unless DOJO_BG_FOLDS / DOJO_COLOR_CALIB are set,
# and ONE frozen calibration set in it for every episode (DOJO_CALIB_SET, scripts/dojo/calib_set.sh).
# Everything runs in one systemd user scope (24G / 1400 %); only PIDs this script started are stopped.
# bash reads a script while it runs: the body is one brace group, read before any of it runs (integ_eval.sh's note).
{
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# the router key: one source per process. This script names the file (step 4: AGENTIC_ENV_FILE / DOJO_AGENT_DOTENV),
# so the environment must not carry OMNIROUTE_* as well (checked before the lock wait)
[[ -z "${OMNIROUTE_BASE_URL:-}${OMNIROUTE_API_KEY:-}" ]] || {   # values never printed
  echo "[official] OMNIROUTE_* is set and this script names the key file (AGENTIC_ENV_FILE / DOJO_AGENT_DOTENV): unset OMNIROUTE_BASE_URL" \
    "OMNIROUTE_API_KEY (one source per process: docs/ENVIRONMENT.md)" >&2; exit 2; }
for _a in "$@"; do [[ "$_a" == --dry-run && "${1:-}" != --inner ]] && exec bash "$HERE/scripts/dojo/official_eval.sh" --inner "$@"; done
if [[ "${1:-}" != --inner ]]; then
  # outer: the memory gate, the heavy lock, then this script again inside a systemd user scope
  _sealed="" _lock="" _prev=""                 # a sealed block names its lock (no default): checked before any wait
  for _a in "$@"; do [[ "$_prev" == --sealed-block ]] && _sealed="$_a"; [[ "$_prev" == --lock ]] && _lock="$_a"; _prev="$_a"; done
  [[ -z "$_sealed" || -n "$_lock" ]] || { echo "[official] --sealed-block needs an explicit --lock FILE (no default)" >&2; exit 2; }
  export DOJO_DATA="${DOJO_DATA:-$HOME/dojo-data}"
  avail() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
  for i in $(seq 30); do
    [[ "$(avail)" -ge "${DOJO_MIN_AVAIL_GB:-25}" ]] && break
    [[ "$i" -eq 30 ]] && { echo "[official] MemAvailable $(avail) GB too low: refusing to start" >&2; exit 75; }
    echo "[official] MemAvailable $(avail) GB, waiting..." >&2; sleep 20
  done
  # our own heavy lock (never compute4's ~/dojo-gpu.lock = ~/coord/gpu.lock: Mikhail's alone)
  LOCK="${_lock:-${DOJO_GPU_LOCK:-$HOME/dojo-ours-gpu.lock}}"; exec 9>>"$LOCK"
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

TASK="" LAYOUT_IDS="" POLICY="dojo.agent.policy:AgentPolicy" PRESET="submit_v2" CONFIG='{}' TWIN_ENVS=8 NAME="" OUT=""
EP_MAX_MIN=25 CANDIDATE="" POLICY_SET="" DRY="" SEG="" GLYPH="" GLYPHK="" SETENV=() SEALED_DIR="" MANIFEST="" LOCK_ARG=""
RAW_OK=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;        --layouts) LAYOUT_IDS="$2"; shift 2 ;;
    --policy) POLICY="$2"; POLICY_SET=1; shift 2 ;;    --preset) PRESET="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;    --twin-envs) TWIN_ENVS="$2"; shift 2 ;;
    --id) NAME="$2"; shift 2 ;;          --out) OUT="$2"; shift 2 ;;
    --episode-max-min) EP_MAX_MIN="$2"; shift 2 ;;
    --candidate)                         # its non-comment lines, split on whitespace like lane2.sh's queue line
      [[ -z "$CANDIDATE" && -f "$2" ]] || { echo "[official] --candidate: one readable file (got $2)" >&2; exit 2; }
      CANDIDATE="$2"; read -ra _ct <<< "$(grep -v '^[[:space:]]*#' "$2" | tr '\n' ' ')"; shift 2; set -- "${_ct[@]}" "$@" ;;
    --setenv)
      [[ "$2" =~ ^[A-Z_][A-Z0-9_]*= ]] || { echo "[official] --setenv K=V (got $2)" >&2; exit 2; }
      [[ "$2" =~ ^(OMNIROUTE_|ANTHROPIC_|OPENAI_|OPENROUTER_)|(KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*= ]] && {
        echo "[official] --setenv ${2%%=*}: credentials never travel as arguments (docs/ENVIRONMENT.md)" >&2; exit 2; }
      SETENV+=("$2"); shift 2 ;;
    --seg) SEG="$2"; shift 2 ;;          --glyph) GLYPH="$2"; shift 2 ;;
    --glyph-kong) GLYPHK="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --raw-ok) RAW_OK=1; shift ;;
    --sealed-block) SEALED_DIR="$2"; shift 2 ;;    --manifest) MANIFEST="$2"; shift 2 ;;
    --lock) LOCK_ARG="$2"; shift 2 ;;              # taken by the outer part
    *) echo "[official] unknown option $1" >&2; exit 2 ;;
  esac
done
[[ -n "$CANDIDATE" && -z "$POLICY_SET" ]] && POLICY="dojo.runtime.policy:RuntimePolicy"   # integ_eval's policy
if [[ -n "$SEALED_DIR$MANIFEST" ]]; then     # a sealed joint block: its own rules, nothing of the dev path's
  [[ -d "$SEALED_DIR" && -f "$MANIFEST" ]] || { echo "[official] --sealed-block DIR and --manifest FILE are both required" >&2; exit 2; }
  [[ -n "$DRY" || -n "$LOCK_ARG" ]] || { echo "[official] --sealed-block needs an explicit --lock FILE (no default)" >&2; exit 2; }
  [[ "${HARNESS_DEV:-0}" == 0 && "${HARNESS_PRIVILEGED:-0}" == 0 ]] || {
    echo "[official] --sealed-block refuses HARNESS_DEV / HARNESS_PRIVILEGED (no dev or debug path in a scored block)" >&2; exit 2; }
fi
MODELS="${DOJO_SEG_MODELS:-$HOME/dojo-data/segtrain/models}"
SIDE=()                                  # "kind|model|ENVVAR..." of the side servers this call starts
[[ -n "$SEG" ]] && SIDE+=("seg|$SEG|DOJO_SEG_ADDR")
[[ -n "$GLYPH" ]] && SIDE+=("glyph|$GLYPH|DOJO_GLYPH_ADDR DOJO_GLYPH_ADDR_DIGIT")
[[ -n "$GLYPHK" ]] && SIDE+=("glyphk|$GLYPHK|DOJO_GLYPH_ADDR_KONG")
if [[ -n "$DRY" ]]; then                 # the resolved call as JSON, nothing started
  _pd="${DOJO_PERCEPTION_DATA:-$HOME/dojo-perception-data}"      # step 0's perception data, as it will resolve
  export DRY_PERCEPTION_ENV="$(printf '{"DOJO_CATALOG": "%s", "DOJO_X5_DIR": "%s", "DOJO_BG_FOLDS": "%s", "DOJO_COLOR_CALIB": "%s", "DOJO_CALIB_SET": "%s"}' \
    "${DOJO_CATALOG:-$_pd/catalog}" "${DOJO_X5_DIR:-$_pd/catalog/robot/x5}" "${DOJO_BG_FOLDS:-$_pd/bg_folds_51}" \
    "${DOJO_COLOR_CALIB:-$_pd/catalog/color_calib_51.json}" "${DOJO_CALIB_SET:-frozen_d0_100040_49}")"
  ( cd "$HERE" && "$HERE/.venv/bin/python" - "$POLICY" "$PRESET" "$CONFIG" "$MODELS" "$CANDIDATE" "$TWIN_ENVS" \
      "${#SIDE[@]}" "${SIDE[@]}" "${SETENV[@]}" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
from dojo.bridge.driver import build_config
policy, preset, config, models, cand, twin_envs, n_side = sys.argv[1:8]
side, setenv = sys.argv[8:8 + int(n_side)], sys.argv[8 + int(n_side):]
md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest() if Path(p).is_file() else None
FILES = {"seg": ("best.pt", "classes.json"), "glyph": ("model.pt",), "glyphk": ("model.pt",)}
servers = {}
for s in side:
    kind, model, envs = s.split("|")
    servers[kind] = {"model": model, "env": envs.split(), "files": {f: {"path": str(Path(models) / model / f),
                     "md5": md5(Path(models) / model / f)} for f in FILES[kind]}}
cfg = build_config(preset, config, "bundle", "spawn")
print(json.dumps({"launcher": "official_eval.sh", "candidate": cand or None,
                  "candidate_md5": md5(cand) if cand else None, "policy": policy, "preset": preset,
                  "config_arg": json.loads(config), "runtime_config": cfg.__dict__,
                  "setenv": dict(kv.split("=", 1) for kv in setenv), "side_servers": servers,
                  "perception_env": json.loads(os.environ["DRY_PERCEPTION_ENV"]),
                  "driver_args": {"perception": "local", "verify": "bundle", "twin": "spawn", "twin_envs": int(twin_envs)},
                  "grip": "strict v2 server: eef.<arm>.ee_joint_state_official (D-C11), no DOJO_GRIP_SOURCE needed"},
                 sort_keys=True, default=str))
PY
  ) || exit 2
  exit 0
fi
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
SUFFIX=.settled
if [[ -n "$SEALED_DIR" ]]; then          # the sealed block: manifest ids only, md5-verified, staged here only
  ST="$( cd "$HERE" && "$PY" -m dojo.simlab.sealed stage --dir "$SEALED_DIR" --manifest "$MANIFEST" --task "$TASK" \
          --ids "$LAYOUT_IDS" --dest "$OUT/layouts" 2>&1 )"; rc=$?
  echo "$ST" | tee -a "$LOG" >&2
  [[ "$rc" -eq 0 ]] || { log "sealed block refused: nothing started"; exit 2; }
  # run once (D-C19): one O_EXCL marker per (task, id) in $COORD_DIR/scored/<block>/, before anything else starts
  CL="$( cd "$HERE" && "$PY" -m dojo.simlab.sealed claim --coord-dir "${COORD_DIR:?sealed mode needs COORD_DIR}" \
          --manifest "$MANIFEST" --task "$TASK" --ids "$LAYOUT_IDS" --run-id "$ID" 2>&1 )"; rc=$?
  echo "$CL" | tee -a "$LOG" >&2
  [[ "$rc" -eq 0 ]] || { log "sealed block: run-once claim refused: nothing started"; exit 2; }
  [[ "$( cd "$HERE" && "$PY" -m dojo.simlab.sealed settled --manifest "$MANIFEST" --task "$TASK" )" == 1 ]] || SUFFIX=""
  mapfile -t IDS < <(tr ' ,' '\n\n' <<< "$LAYOUT_IDS" | grep -v '^$' | sort -n)
  LAYOUT_IDS=""                          # the dev loop below stages nothing
fi
for lid in $(tr ' ,' '\n\n' <<< "$LAYOUT_IDS" | grep -v '^$' | sort -n); do
  [[ "$lid" =~ ^[0-9]+$ && "$lid" -ge 100000 ]] || { log "layout id $lid: dev layouts are >= 100000"; exit 2; }
  [[ "$lid" -lt 100100 ]] || { log "layout id $lid: ids >= 100100 are reserved for joint paired runs (D-C10)"; exit 2; }
  if (( lid >= 100010 && lid <= 100019 )) || (( lid >= 100050 && lid <= 100099 )); then
    log "layout id $lid: Mikhail's development slice (D-C10a: never run rendered on our side)"; exit 2
  fi
  if (( lid >= 100030 && lid <= 100049 )); then
    log "layout id $lid: training / priors / calibration block (D-C10c: 100030-49, never evaluated)"; exit 2
  fi
  f="$LAYOUTS_SRC/$lid.settled.json"
  if [[ ! -f "$f" && -n "$RAW_OK" && -f "$LAYOUTS_SRC/$lid.json" ]]; then
    f="$LAYOUTS_SRC/$lid.json"; HAVE_RAW=1          # --raw-ok: this task's dev records are raw (make_toast, conveyors)
  else
    HAVE_SETTLED=1
  fi
  [[ -f "$f" && "$f" != *Eval_Layout* ]] || { log "no settled dev layout $f"; exit 2; }
  cp "$f" "$OUT/layouts/$TASK/"
  IDS+=("$lid")
done
if [[ -n "${HAVE_RAW:-}" ]]; then
  [[ -z "${HAVE_SETTLED:-}" ]] || { log "--raw-ok: one call is all settled or all raw records"; exit 2; }
  SUFFIX=""
  log "--raw-ok: raw dev records (no settled ones for $TASK)"
fi
if [[ "$TASK" == imitate_sorting_sequence && -z "$SEALED_DIR" ]]; then   # its support-arm demo, one per id (demo_gen)
  for lid in "${IDS[@]}"; do
    d="$DOJO_DATA/simlab/traj/$TASK/$lid.pkl"
    [[ -f "$d" ]] || { log "no demo $d (demo_gen): the support arm would stay home, RoboDojo flags the env unstable"; exit 2; }
  done
fi
N=${#IDS[@]}
log "task $TASK, $N layouts: ${IDS[*]}; policy $POLICY preset $PRESET config $CONFIG; out $OUT"
# perception data (genuine 5.1 calibration), checked before anything starts
PD="${DOJO_PERCEPTION_DATA:-$HOME/dojo-perception-data}"
export DOJO_CATALOG="${DOJO_CATALOG:-$PD/catalog}" DOJO_X5_DIR="${DOJO_X5_DIR:-$PD/catalog/robot/x5}"
export DOJO_BG_FOLDS="${DOJO_BG_FOLDS:-$PD/bg_folds_51}" DOJO_COLOR_CALIB="${DOJO_COLOR_CALIB:-$PD/catalog/color_calib_51.json}"
source "$HERE/scripts/dojo/calib_set.sh"         # DOJO_CALIB_SET: ONE frozen calibration set for every episode
calib_set_check || exit 2
# scorer side: never evaluate a layout the calibration set was built from (its manifest)
( cd "$HERE" && "$PY" -c 'import sys; from dojo.perception.calib_set import load_manifest, overlap
m = load_manifest(sys.argv[1]); bad = [i for i in sys.argv[2:] if overlap(m, i)]
sys.exit("[official] calibration set built from layouts %s: refusing to evaluate them" % bad if bad else 0)' \
  "$DOJO_BG_FOLDS/$DOJO_CALIB_SET" "${IDS[@]}" ) || exit 2

# 0b. the candidate's side servers (lane2.sh's commands), BEFORE anything of robot-harness's starts: one that does not
#     come up ends the call here (no episode booked, no silent degrade); their addresses reach only our processes (step 4)
SIDE_ENV=()
for s in "${SIDE[@]}"; do
  IFS='|' read -r kind model envs <<< "$s"
  sock="${XDG_RUNTIME_DIR:-/tmp}/dojo-off-$$-$kind.sock"; rm -f "$sock"
  if [[ "$kind" == seg ]]; then
    [[ -f "$MODELS/$model/best.pt" && -f "$MODELS/$model/classes.json" ]] || { log "seg model $MODELS/$model missing"; exit 2; }
    ( cd "$HERE" && YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$MODELS/ultra_cfg}" exec setsid nice "$PY" -u -m dojo.perception.seg_twin serve \
        --sock "$sock" --weights "$MODELS/$model/best.pt" --classes "$MODELS/$model/classes.json" --conf 0.25 \
        --max-hours "${OFFICIAL_SIDE_MAX_HOURS:-2}" 9>&- ) > "$OUT/$kind.log" 2>&1 < /dev/null &
  else
    [[ -f "$MODELS/$model/model.pt" ]] || { log "$kind model $MODELS/$model/model.pt missing"; exit 2; }
    ( cd "$HERE" && PYTHONPATH="$HERE" exec setsid nice "$PY" -u -m dojo.perception.glyph_read serve --sock "$sock" \
        --model "$MODELS/$model/model.pt" --max-hours "${OFFICIAL_SIDE_MAX_HOURS:-2}" 9>&- ) > "$OUT/$kind.log" 2>&1 < /dev/null &
  fi
  PIDS+=($!)
  for _ in $(seq 60); do [[ -S "$sock" ]] && break; kill -0 "${PIDS[-1]}" 2>/dev/null || break; sleep 2; done
  [[ -S "$sock" ]] || { log "$kind server ($model) did not come up: $(tail -3 "$OUT/$kind.log" | tr '\n' ' ')"; exit 1; }
  for v in $envs; do SIDE_ENV+=("$v=$sock"); done
  log "$kind server up ($model)"
done

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
SEALED_INFO="$MANIFEST|${IDS[*]}" "$PY" - "$OUT/provenance.json" "$BENCH" "$HERE" "$POL" "$CANDIDATE" "$MODELS" "${SIDE[@]}" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
out, bench, here, pol, cand, models = sys.argv[1:7]
man, sealed_ids = os.environ.get("SEALED_INFO", "|").split("|", 1)
md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()
FILES = {"seg": ("best.pt", "classes.json"), "glyph": ("model.pt",), "glyphk": ("model.pt",)}
d = {"robot_harness_commit": (Path(bench) / "COMMIT").read_text().strip() if (Path(bench) / "COMMIT").exists() else None,
     "controlr": json.loads((Path(here) / ".deployed.json").read_text()) if (Path(here) / ".deployed.json").exists() else None,
     "shim_md5": {p.name: md5(p) for p in sorted(Path(pol).glob("*.py"))},
     "server_md5": {n: md5(Path(bench) / "harness" / n) for n in ("robodojo_server_j2.py", "robodojo_server_j1.py",
                                                                   "robodojo_server_strict.py", "robodojo_server.py")},
     "candidate": {"path": cand, "md5": md5(cand), "text": Path(cand).read_text()} if cand else None,
     "side_servers": {s.split("|")[0]: {"model": s.split("|")[1], "md5": {f: md5(Path(models) / s.split("|")[1] / f)
                                                                          for f in FILES[s.split("|")[0]]}}
                      for s in sys.argv[7:]},
     "sealed_block": {"manifest": man, "manifest_md5": md5(man), "ids": sealed_ids.split(), "server_dev": False}
     if man else None}
Path(out).write_text(json.dumps(d, indent=1))
PY

# 2. robot-harness's strict J2 server (his launcher's environment; the J1 perceiver = our adapter, unused on the local
#    perception route, warmed CPU-only)
RUNDIR="$BENCH/run/$ID"; mkdir -p "$RUNDIR"
DEVF=(--dev); DEVJ=true; HDEV=1
[[ -n "$SEALED_DIR" ]] && { DEVF=(); DEVJ=false; HDEV=0; }     # a sealed block: no truth log, no dev path
printf '{"id": "%s", "kind": "robodojo", "strict": true, "j1": true, "j2": true, "privileged": false, "dev": '"$DEVJ"', "ready": false, "started_t": %s}\n' \
  "$ID" "$(date +%s)" > "$RUNDIR/meta.json"
export HARNESS_SHIM_ADDR="$RUNDIR/shim.sock" HARNESS_SHIM_AUTHKEY="$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
export HARNESS_PRIVILEGED=0 HARNESS_DEV="$HDEV" HARNESS_STRICT=1
export HARNESS_SHIM_IDLE_S="$("$SPY" "$BENCH/harness/ask_pause.py" shim-idle)" || { log "ask_pause shim-idle failed"; exit 2; }
export J1_PERCEIVER="${J1_PERCEIVER:-dojo.perception.j1_adapter:perceive}" J1_PERCEIVER_PATH="${J1_PERCEIVER_PATH:-$HERE}"
export J1_PERCEIVER_PYTHON="${J1_PERCEIVER_PYTHON:-$PY}" J1_ORIG_HOME="$HOME" J1_PERCEIVER_CPU_ONLY="${J1_PERCEIVER_CPU_ONLY:-1}"
export J1_HELD_FALLBACK="${J1_HELD_FALLBACK:-no}"          # no operator here: never wait for one
export PYTHONDONTWRITEBYTECODE=1
( exec setsid nice -n 5 "$SPY" "$BENCH/harness/robodojo_server_j2.py" --id "$ID" --strict "${DEVF[@]}" --reserve 20 \
    --idle-timeout 3600 9>&- ) > "$RUNDIR/server.log" 2>&1 < /dev/null &
PIDS+=($!)
for _ in $(seq 120); do [[ -S "$BENCH/run/$ID.sock" ]] && break; sleep 0.5; done
[[ -S "$BENCH/run/$ID.sock" ]] || { log "server did not start: $(tail -5 "$RUNDIR/server.log")"; exit 1; }
log "strict J2 server up ($ID)"

# 3. RoboDojo's eval client on our staged layouts (dev_eval.sh: ws policy server + dev_eval.py; we hold the lock)
( cd "$HERE" && RD_HOME="$FARM" RD_ROOT="$FARM/RoboDojo" DOJO_GPU_LOCK_HELD=1 SIMLAB_DEV_LAYOUT_SUFFIX="$SUFFIX" SIMLAB_SEALED_MANIFEST="${MANIFEST:+$(readlink -f "$MANIFEST")}" exec setsid \
    bash scripts/simlab/dev_eval.sh --task "$TASK" --policy harness_strict_j2 --eval-num "$N" \
    --layouts "$OUT/layouts" --out "$OUT/eval" 9>&- ) > "$OUT/dev_eval.log" 2>&1 < /dev/null &
EC=$!; PIDS+=($EC)
log "eval client pid $EC"

# 4. our perception service (genuine 5.1 calibration)
# the router key for the agent (llmio) and the agentic perceiver: ONE named file each (default the box's ~/controlr/.env,
# OMNIROUTE_* only; llmio has no home-dir fallback any more). Exported only now, after robot-harness's processes started
export AGENTIC_ENV_FILE="${AGENTIC_ENV_FILE:-$HOME/controlr/.env}" DOJO_AGENT_DOTENV="${DOJO_AGENT_DOTENV:-$HOME/controlr/.env}"
export DOJO_AGENTIC_LOG="${DOJO_AGENTIC_LOG:-$OUT/agentic}"
key() { python3 -c 'import secrets; print(secrets.token_hex(16))'; }
port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'; }
# the candidate's environment and side-server addresses: from here on, i.e. for our perception service and drivers only
for kv in "${SETENV[@]}" "${SIDE_ENV[@]}"; do export "$kv"; done
[[ ${#SETENV[@]} -gt 0 || ${#SIDE_ENV[@]} -gt 0 ]] && log "env for our processes: $(for kv in "${SETENV[@]}" "${SIDE_ENV[@]}"; do printf '%s ' "${kv%%=*}"; done)"
export DOJO_PERCEPTION_AUTHKEY="$(key)" DOJO_PERCEPTION_PORT="$(port)" DOJO_PERCEPTION_READY="$OUT/perception.ready"
( cd "$HERE" && DOJO_ILIA=1 PYOPENGL_PLATFORM=egl exec setsid nice "$PY" -u -m dojo.runtime.perception_service 9>&- ) \
  > "$OUT/perception_service.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 150); do [[ -f "$DOJO_PERCEPTION_READY" ]] && break; sleep 2; done
[[ -f "$DOJO_PERCEPTION_READY" ]] || { log "perception service not ready: see perception_service.log"; exit 1; }
log "perception service up"

# 5. one driver per episode, in the eval client's layout order (ids ascending); the perception calibration is the
#    service's one frozen set (DOJO_CALIB_SET), the same for every episode (nothing derived from the layout id)
k=0; RC=0
for lid in "${IDS[@]}"; do
  k=$((k + 1))
  FIN=(); [[ "$k" -eq "$N" ]] && FIN=(--close)
  EPO="$OUT/eps/$lid"; mkdir -p "$EPO"
  log "episode $k/$N (layout $lid, calibration $DOJO_CALIB_SET): driver ${FIN[*]}"
  ( cd "$HERE" && DOJO_ILIA=1 DOJO_AGENT_LOG="$EPO/agent" exec timeout --signal=TERM "$((EP_MAX_MIN * 60 + 600))" \
      "$PY" -u -m dojo.bridge.driver --server-id "$ID" --bench "$BENCH" --policy "$POLICY" --preset "$PRESET" \
      --config "$CONFIG" --perception local --verify bundle --twin spawn --twin-envs "$TWIN_ENVS" --hold-gpu-lock \
      --keep-frames 20 --wait-episode 900 --out "$EPO" "${FIN[@]}" 9>&- ) >> "$EPO/driver.log" 2>&1
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
