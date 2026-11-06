#!/usr/bin/env bash
# integ-c4 lane v2 (Integration stream, compute4, from 2026-10-28 00:00Z).
# Private compute4 camera lanes. LANE_REPO must name a merged deployment.
# lane2.sh heavy1 ~/dojo-integ-c4-heavy1.lock (heavy2 analogue).
# Queue entries use the established whitespace-only parser and compact JSON.
# STOP / STOP.LANE stop dispatch; HTTP 429 stops all lanes.
{
set -uo pipefail
LANE=$1 LOCK=$2
S=${LANE_STATE:-$HOME/dojo-integ-c4-state}; Q=$S/queue2.txt; RUNS=${LANE_RUNS:-$HOME/dojo-data/integ-c4/runs2}; REPO=${LANE_REPO:?set LANE_REPO to the merged deployment}
STREAM=${LANE_STREAM:-integ-c4}
JOB_MAX=${LANE_JOB_MAX_S:-1980}   # 33 min: boot + a 25-min episode cap + teardown (28 min while R2 ran)
mkdir -p "$RUNS"; touch "$Q" "$Q.lock"
source "${LANE_ENV_FILE:-$HOME/dojo-env.sh}"
export DOJO_STREAM=$STREAM DOJO_MIN_FREE_GB=50 INTEG_MIN_FREE_MB=16223 INTEG_MIN_MEM_GB=16 INTEG_YIELD_S=0 INTEG_LOCK_WAIT_S=120
export DOJO_DATA=${LANE_DATA:-$DOJO_DATA}
# 00:55Z (lead): ~/dojo-perception-data/bg_folds_51 has no scene/ backgrounds (belt, imitate's basket, organize's cushion
# ran against the plain table); ours, md5-verified, with compute3's scene/ dirs. Checked in each job's perception service.
export DOJO_BG_FOLDS="${LANE_BG_FOLDS:-$HOME/dojo-data/bg_folds_51_ilia}"
[[ -d "$DOJO_BG_FOLDS/fold0/scene" ]] || { echo "$(date -u +%FT%TZ) [$1] no $DOJO_BG_FOLDS/fold0/scene: refusing to start"; exit 2; }
INO=$(stat -L -c %i "$LOCK")
if [[ "$STREAM" == integ-c3 ]]; then
  [[ "$LOCK" == "$HOME/dojo-gpu.lock" && ! -L "$LOCK" && "$LANE" == heavy1 ]] || { echo "compute3 single heavy slot required"; exit 2; }
elif [[ "$STREAM" == integ-c6 ]]; then
  [[ "$LOCK" == "$HOME/dojo-gpu.lock" || "$LOCK" == "$HOME/dojo-gpu-heavy2.lock" ]] && [[ ! -L "$LOCK" ]] || { echo "compute6 heavy slot required"; exit 2; }
else
  [[ "$LOCK" == "$HOME/dojo-integ-c4-$LANE.lock" && ! -L "$LOCK" ]] || { echo "private lane lock required"; exit 2; }
fi
stopped() { [[ -e "$S/STOP" || -e "$S/STOP.$LANE" ]]; }
log() { echo "$(date -u +%FT%TZ) [$LANE] $*"; }
ours() { grep -qs "$STREAM" "/proc/$1/cgroup"; }
others() { local n=0 p; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do ours "$p" || n=$((n+1)); done; echo "$n"; }
mine() {   # our running rendered jobs (other lanes')
  local n=0 f; for f in "$S"/running.*; do [[ -e "$f" ]] && kill -0 "$(cat "$f")" 2>/dev/null && n=$((n+1)); done; echo "$n"
}
sides() {  # the BIG GPU side servers (SAM / mv3d) other lanes run: GPU jobs for the box total, not rendered lanes
  local n=0 f; for f in "$S"/side.*; do [[ -e "$f" && "$f" != "$S/side.$LANE" ]] && n=$((n + $(cat "$f"))); done; echo "$n"
}
# caps (file lane2.caps "OURS TOTAL", re-read per job; 00:35Z lead: power limits lifted, after R2 up to 3 rendered lanes of
# ours + <= 2 other GPU jobs on the box): our running jobs + 1 <= OURS and GPU apps outside our scopes + ours + 1 <= TOTAL
# lead 29 Oct 03:2xZ: TOTAL counts others + our rendered lanes + only the BIG side servers (SAM / mv3d, > 2 GB); the
# per-job seg / glyph servers (~0.5-0.9 GB) do not count. VRAM guard instead: no job start with < VRAM_MIN_MIB free.
shared_ok() {   # KIND...: each shared side server's PID alive and answering an info ping within 15 s
  local k
  for k in "$@"; do
    kill -0 "$(cat "$S/shared/$k.pid" 2>/dev/null)" 2>/dev/null || return 1
    timeout 15 "$SHARED_PY" "$S/ping_side.py" "$S/shared/$k.sock" >/dev/null 2>&1 || return 1
  done
}
SHARED_PY="${SHARED_PY:-$REPO/.venv/bin/python}"
vram_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -dc 0-9; }
cap_ok() {   # rendered of ours + 1 <= OURS; others + ours + big side servers + 1 <= TOTAL; free VRAM >= VRAM_MIN_MIB
  local o m d c v; o=$(others); m=$(mine); d=$(sides); v=$(vram_free); read -r c < "$S/lane2.caps" 2>/dev/null; set -- ${c:-1 4}
  local ours=${1:-1}
  local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [[ "$used" =~ ^[0-9]+$ ]] || return 1
  local max_used=16384
  [[ "$STREAM" == integ-c3 ]] && max_used=14336
  (( used <= max_used && o + m + d + 1 <= ${2:-4} && m + 1 <= ours && ${v:-0} >= 16223 ))
}
# The sole allowed coordination file. No other coord paths are accessed.
boot_gate() {
  [[ "$STREAM" == integ-c6 || "$STREAM" == integ-c3 ]] && return 0  # ours alone; never accesses compute4's coordination protocol
  local epoch iso who now
  while true; do
    epoch=0 iso= who=
    if [[ -e "$HOME/coord/lane_gate.stamp" ]]; then
      read -r epoch iso who < "$HOME/coord/lane_gate.stamp" || true
      [[ "$epoch" =~ ^[0-9]+$ ]] || { log "invalid lane_gate.stamp: waiting"; sleep 15; continue; }
    fi
    now=$(date +%s)
    if (( now - epoch >= 120 )) && cap_ok; then
      printf '%s %s ilia-ab\n' "$now" "$(date -u +%FT%TZ)" > "$HOME/coord/lane_gate.stamp"
      log "boot gate claimed: $now ilia-ab"
      return 0
    fi
    log "boot gate: last start $((now - epoch)) s ago ($who); waiting"
    stopped && return 1
    sleep 15
  done
}
take_lock() { exec 8>>"$LOCK"; until flock -w 60 8; do stopped && { exec 8>&-; return 1; }; done; return 0; }
drop_lock() { flock -u 8 2>/dev/null; exec 8>&-; }
front() { flock "$Q.lock" bash -c '{ printf "%s\n" "$2"; cat "$1"; } > "$1.tmp" && mv "$1.tmp" "$1"' _ "$Q" "$1"; }
# GPU side servers that live only inside one job's lock window (SAM, mv3d): units[] / pids[]; side.$LANE counts them
units=() pids=() big=0
side_start() {   # KIND SOCK ENVVAR WAIT_S CMD...: start in its own integ-c4_ scope, wait for the socket, export ENVVAR=SOCK
  local kind=$1 sock=$2 var=$3 wait_s=$4 u; shift 4
  k=$((k + 1)); u="${STREAM}_${kind}_${LANE}_$$_$k"; rm -f "$sock"
  ( cd "$repo" && exec systemd-run --user --scope -q --unit="$u" -p MemoryMax=16G -p CPUQuota=400% "$@" ) \
    > "$RUNS/$name.$kind.log" 2>&1 7>&- 8>&- < /dev/null &
  units+=("$u"); pids+=($!); [[ "$kind" == sam || "$kind" == mv3d ]] && big=$((big + 1)); echo "$big" > "$S/side.$LANE"
  local i; for ((i = 0; i < wait_s / 2; i++)); do [[ -S "$sock" ]] && break; kill -0 "${pids[-1]}" 2>/dev/null || break; sleep 2; done
  [[ -S "$sock" ]] || return 1
  jenv+=("$var=$sock"); log "$kind server up ($(( $(date +%s) - t0 )) s)"
}
side_stop() {
  local u p cg
  for u in "${units[@]}"; do
    timeout 30 systemctl --user stop "$u.scope" 2>/dev/null
    cg=$(timeout 10 systemctl --user show -p ControlGroup --value "$u.scope" 2>/dev/null)
    [[ -n "$cg" && -s "/sys/fs/cgroup$cg/cgroup.procs" ]] && for p in $(cat "/sys/fs/cgroup$cg/cgroup.procs"); do kill -9 "$p" 2>/dev/null; done
  done
  for p in "${pids[@]}"; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done
  units=() pids=() big=0; rm -f "$S/side.$LANE"
}
rows() {   # RUN_DIR: "<n finished non-infra rows> <summary of each row>"
  python3 - "$1" <<'PY'
import glob, json, sys
ok, out = 0, []
for f in glob.glob(sys.argv[1] + "/*_episodes.jsonl"):
    for ln in open(f):
        if not ln.strip():
            continue
        r = json.loads(ln)
        ag = r.get("agentic") or []
        calls = sum(int(a.get("calls") or 0) for a in ag if isinstance(a, dict))
        if r.get("category") != "infra_failure" and not r.get("skipped"):
            ok += 1
        out.append(f"{r.get('layout_id')}: success {r.get('success')} score {r.get('score')} cat {r.get('category') or 'ok'} "
                   f"steps {r.get('steps_used')}/{r.get('step_lim')} wall {r.get('wall_s')} s calls {calls}")
print(ok, "; ".join(out) or "no row")
PY
}
share() { return; }
log "lane2 up (pid $$, lock $LOCK inode $INO, repo ${REPO#$HOME/}, share <= $(cat "$S/lane2.share" 2>/dev/null || echo 720) s, job cap $JOB_MAX s)"
# per-box catalogue gate (#285, lead 28 Oct): any gap in the perception catalogue fails the lane loudly; logs catalog_md5
GATE_REPO="${LANE_GATE_REPO:-$REPO}"
if ! gate_out=$(PY="$GATE_REPO/.venv/bin/python" bash "$GATE_REPO/scripts/dojo/catalog_gate.sh" 2>&1); then
  log "CATALOG GATE FAILED (gate ${GATE_REPO#$HOME/}): lane not started"; printf '%s\n' "$gate_out" | tail -8 | sed "s/^/  [gate] /"
  exit 1
fi
log "catalog gate OK: $(grep -m1 '^catalog_md5' <<<"$gate_out")"
k=0
while true; do
  stopped && { log "STOP file: exiting"; break; }
  [[ -s "$Q" ]] || { log "queue empty"; break; }
  # Serialize admission until the previous job has loaded its render/perception services.
  exec 7>>"$S/wave-start.lock"
  until flock -w 15 7; do stopped && break; done
  t=0
  until cap_ok; do
    (( t % 600 == 0 )) && log "cap: others $(others), ours $(mine), big side servers $(sides), free VRAM $(vram_free) MiB (used must be <= 16384 MiB): waiting"
    stopped && break
    sleep 30; t=$((t + 30))
  done
  stopped && { log "STOP file: exiting"; break; }
  tw=$(date +%s)
  take_lock || { log "STOP file while waiting for the shared lock: exiting"; break; }
  waited=$(( $(date +%s) - tw ))
  # Recheck after the lane lock wait; do not consume a queue row on resource failure.
  if ! cap_ok; then drop_lock; flock -u 7; exec 7>&-; sleep 15; continue; fi
  source "$REPO/scripts/dojo/disk_guard.sh"
  disk_guard "[wave lane]" || { drop_lock; exit 6; }
  boot_gate || { drop_lock; break; }
  if [[ "$STREAM" == integ-c3 ]]; then
    log "shared-account preflight"; who; nvidia-smi --query-gpu=memory.used,memory.free,power.draw --format=csv
  fi
  line=$(flock "$Q.lock" bash -c 'head -1 "$1"; sed -i 1d "$1"' _ "$Q")
  [[ -z "$line" ]] && { drop_lock; log "queue empty"; break; }
  name=${line%% *}; args=${line#* }; repo="$REPO"; sam=""; mv3d=""; seg=""; glyph=""; glyphk=""; jenv=(); toks=(); keep=()
  read -ra toks <<<"$args"
  for ((i = 0; i < ${#toks[@]}; i++)); do
    case "${toks[i]}" in
      --repo) repo=${toks[i+1]/#\~/$HOME}; i=$((i + 1)) ;;
      --setenv) jenv+=("${toks[i+1]/\~/$HOME}"); i=$((i + 1)) ;;
      --sam) sam=${toks[i+1]}; i=$((i + 1)) ;;
      --mv3d) mv3d=1 ;;
      --seg) seg=${toks[i+1]}; i=$((i + 1)) ;;   # the model version dir under ~/dojo-data/segtrain/models (v1, v2b)
      --glyph) glyph=${toks[i+1]}; i=$((i + 1)) ;;   # S's digit / glyph reader: model dir under ~/dojo-data/segtrain/models (digit_v2)
      --glyph-kong) glyphk=${toks[i+1]}; i=$((i + 1)) ;;   # W #284: make_kong tiles' reader (kong_v1b) -> DOJO_GLYPH_ADDR_KONG
      *) keep+=("${toks[i]}") ;;
    esac
  done
  args="${keep[*]}"
  if [[ "$repo" != "$REPO" ]] && ! [[ "$(realpath "$repo")" == "$HOME"/* && -f "$repo/scripts/dojo/integ_eval.sh" ]]; then
    drop_lock; log "bad --repo $repo: dropped $name"; continue
  fi
  printf 'pid=%s stream=%s job=%s start=%s cmd=lane2 %s\n' "$$" "$STREAM" "$name" "$(date -u +%FT%TZ)" "$args" > "$LOCK"
  t0=$(date +%s)
  if [[ -n "$sam" ]]; then
    side_start sam "$S/sam.$LANE.sock" DOJO_SAM_ADDR 240 \
      "$repo/.venv/bin/python" -u -m dojo.perception.sam_masks serve --sock "$S/sam.$LANE.sock" --models "$sam" --max-hours 1 \
      || { side_stop; drop_lock; log "SAM server not ready (see $name.sam.log): $name put back, lane stops"; front "$line"; touch "$S/STOP.$LANE"; break; }
  fi
  # box-level shared side servers (lead 29 Oct, after #301): $S/shared.on + the same model -> the shared socket, no per-job
  # server (~2.1 GB VRAM back per lane); the lane pings them before and checks PID + ping after the job (shared_ok)
  shared_kinds=()
  if [[ -e "$S/shared.on" ]]; then
    [[ "$seg" == "$(cat "$S/shared/seg.model" 2>/dev/null)" ]] && { jenv+=("DOJO_SEG_ADDR=$S/shared/seg.sock"); shared_kinds+=(seg); seg=""; }
    [[ -n "$glyph" && "$glyph" == "$(cat "$S/shared/glyph.model" 2>/dev/null)" ]] && { jenv+=("DOJO_GLYPH_ADDR=$S/shared/glyph.sock" "DOJO_GLYPH_ADDR_DIGIT=$S/shared/glyph.sock"); shared_kinds+=(glyph); glyph=""; }
    [[ -n "$glyphk" && "$glyphk" == "$(cat "$S/shared/glyphk.model" 2>/dev/null)" ]] && { jenv+=("DOJO_GLYPH_ADDR_KONG=$S/shared/glyphk.sock"); shared_kinds+=(glyphk); glyphk=""; }
    if (( ${#shared_kinds[@]} )) && ! shared_ok "${shared_kinds[@]}"; then
      side_stop; drop_lock; log "SHARED SIDE DOWN before $name (${shared_kinds[*]}): put back, every lane stops"; front "$line"; touch "$S/STOP"; break
    fi
    shared_pids=$(for k in "${shared_kinds[@]}"; do cat "$S/shared/$k.pid"; done | tr '\n' ' ')
  fi
  if [[ -n "$seg" ]]; then         # S's twin-trained segmenter (#220 / #234; = with_seg_server.sh's server), model dir $seg,
    # run from the JOB's own pinned deploy venv (+ ultralytics), never another stream's dir (lead, 28 Oct 20:20Z)
    side_start seg "$S/seg.$LANE.sock" DOJO_SEG_ADDR 120 env YOLO_CONFIG_DIR="$S/ultra_cfg" \
      "$repo/.venv/bin/python" -u -m dojo.perception.seg_twin serve --sock "$S/seg.$LANE.sock" \
      --weights "$HOME/dojo-data/segtrain/models/$seg/best.pt" --classes "$HOME/dojo-data/segtrain/models/$seg/classes.json" \
      --conf 0.25 --max-hours 1 \
      || { side_stop; drop_lock; log "seg server not ready (see $name.seg.log): $name put back, lane stops"; front "$line"; touch "$S/STOP.$LANE"; break; }
  fi
  if [[ -n "$glyph" ]]; then       # S's glyph reader (#277, dojo.perception.glyph_read; ~1 GB VRAM) -> DOJO_GLYPH_ADDR, from the job's deploy
    side_start glyph "$S/glyph.$LANE.sock" DOJO_GLYPH_ADDR 120 env PYTHONPATH="$repo" \
      "$repo/.venv/bin/python" -u -m dojo.perception.glyph_read serve --sock "$S/glyph.$LANE.sock" \
      --model "$HOME/dojo-data/segtrain/models/$glyph/model.pt" --max-hours 1 \
      || { side_stop; drop_lock; log "glyph server not ready (see $name.glyph.log): $name put back, lane stops"; front "$line"; touch "$S/STOP.$LANE"; break; }
    jenv+=("DOJO_GLYPH_ADDR_DIGIT=$S/glyph.$LANE.sock")   # #284's explicit digit address (DOJO_GLYPH_ADDR stays: older shas)
  fi
  if [[ -n "$glyphk" ]]; then      # W's #284 routing: mahjong tiles -> DOJO_GLYPH_ADDR_KONG only (S's kong_v1b), numerals stay on --glyph
    side_start glyphk "$S/glyphk.$LANE.sock" DOJO_GLYPH_ADDR_KONG 120 env PYTHONPATH="$repo" \
      "$repo/.venv/bin/python" -u -m dojo.perception.glyph_read serve --sock "$S/glyphk.$LANE.sock" \
      --model "$HOME/dojo-data/segtrain/models/$glyphk/model.pt" --max-hours 1 \
      || { side_stop; drop_lock; log "glyph-kong server not ready (see $name.glyphk.log): $name put back, lane stops"; front "$line"; touch "$S/STOP.$LANE"; break; }
  fi
  if [[ -n "$mv3d" ]]; then        # VGGT-1B (bf16 trunk) + DINOv2-base features: DOJO_EMBED_MODELS must hold dinov2-base
    side_start mv3d "$S/mv3d.$LANE.sock" DOJO_MV3D_ADDR 600 env DOJO_EMBED_MODELS="$HOME/dojo-data/models/embed" \
      "$repo/.venv/bin/python" -u -m dojo.perception.mv3d serve --sock "$S/mv3d.$LANE.sock" --max-hours 1 \
      || { side_stop; drop_lock; log "mv3d server not ready (see $name.mv3d.log): $name put back, lane stops"; front "$line"; touch "$S/STOP.$LANE"; break; }
  fi
  task=$(sed -n 's/.*--task \([^ ]*\).*/\1/p' <<<"$args")
  for f in "$S"/start.*; do          # two Isaacs starting one SDF task together can fail to cook it (DEVELOPMENT pitfall)
    [[ -e "$f" && "$f" != "$S/start.$LANE" ]] || continue
    read -r ot oe < "$f"
    if [[ "$ot" == "$task" ]] && (( $(date +%s) - oe < 75 )); then
      d=$(( 75 - ($(date +%s) - oe) )); log "stagger: $task started on ${f##*.} $(( $(date +%s) - oe )) s ago: waiting $d s"; sleep "$d"
    fi
  done
  echo "$task $(date +%s)" > "$S/start.$LANE"
  {   # the catalogue this job reads (W's overlay, 28 Oct): index.json md5 + the overlay manifest's when there is one
    cat_dir="${DOJO_CATALOG:-$HOME/dojo-perception-data/catalog}"
    printf '{"catalog_dir": "%s", "index_md5": "%s", "manifest_md5": "%s"}\n' "$cat_dir" \
      "$(md5sum "$cat_dir/index.json" 2>/dev/null | cut -c1-32)" "$(cat "$cat_dir"/*manifest* 2>/dev/null | md5sum | cut -c1-32)"
  } > "$RUNS/$name.catalog"
  # Retain provenance even if the watchdog emits a synthetic infrastructure row without eval's `code` field.
  if [[ -f "$repo/.deployed.json" ]]; then
    mkdir -p "$RUNS/$name"
    cp "$repo/.deployed.json" "$RUNS/$name/provenance.json"
  fi
  echo $$ > "$S/running.$LANE"; echo "$name" > "$S/job.$LANE"
  log "start $name (lock after ${waited} s, repo ${repo#$HOME/}${jenv[*]:+, env ${jenv[*]}}, gpu others $(others), ours $(mine), $(nvidia-smi --query-gpu=power.draw,memory.used,memory.free --format=csv,noheader | head -1 | sed 's/, / \/ /g'), MemAvailable $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) GB)"
  # shellcheck disable=SC2086
  env -C "$repo" "${jenv[@]}" DOJO_GPU_LOCK="$S/inner.$LANE.lock" bash scripts/dojo/integ_eval.sh --out "$RUNS/$name" $args \
    > "$RUNS/$name.log" 2>&1 7>&- 8>&- < /dev/null &
  jp=$!; killed=""; bgchk=""; admitted=0
  while kill -0 "$jp" 2>/dev/null; do
    sleep 15
    if [[ -z "$bgchk" && -f "$RUNS/$name/perception.ready" ]]; then   # the backgrounds the job's perception service really uses
      pp=""; for p in $(pgrep -f "dojo[.]runtime[.]perception_service"); do
        tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qxF "DOJO_PERCEPTION_READY=$RUNS/$name/perception.ready" && { pp=$p; break; }
      done
      bg=$( [[ -n "$pp" ]] && tr '\0' '\n' < "/proc/$pp/environ" | sed -n 's/^DOJO_BG_FOLDS=//p')
      addr_bad=""
      for kv in "${jenv[@]}"; do      # side-server addresses (DOJO_SAM_ADDR / DOJO_MV3D_ADDR / DOJO_SEG_ADDR / DOJO_GLYPH_ADDR) must reach it too
        [[ "$kv" == DOJO_*_ADDR=* || "$kv" == DOJO_*_ADDR_*=* ]] || continue
        tr '\0' '\n' < "/proc/$pp/environ" 2>/dev/null | grep -qxF "$kv" || addr_bad="$addr_bad ${kv%%=*}"
      done
      if [[ -n "$pp" && "$bg" == "$DOJO_BG_FOLDS" && -z "$addr_bad" ]]; then bgchk=ok; echo "$bg" > "$RUNS/$name.bg"
        log "$name: perception service $pp uses DOJO_BG_FOLDS=$bg$(for kv in "${jenv[@]}"; do [[ "$kv" == DOJO_*_ADDR=* || "$kv" == DOJO_*_ADDR_*=* ]] && printf ', %s' "${kv%%=*}"; done)"
      else bgchk=bad; killed=bg; touch "$S/STOP"; log "BG MISMATCH in $name (service ${pp:-not found}: '${bg}'; side addr missing:${addr_bad:- none}): stopping the job and every lane"; kill -TERM "$jp"; break; fi
    fi
    if (( admitted == 0 && $(date +%s) - t0 >= 150 )) && [[ "$bgchk" == ok ]]; then
      log "VRAM admission released after service readiness and 150 s startup window"
      flock -u 7; exec 7>&-; admitted=1
    fi
    if grep -rqs "HTTP 429" "$RUNS/$name/agentic" "$RUNS/$name/perception_service.log"; then
      killed=429; touch "$S/STOP"; log "CLAUDE 429 in $name: stopping the job and every lane"; kill -TERM "$jp"; break
    fi
    if (( $(date +%s) - t0 > JOB_MAX )); then killed=wall; log "$name past the ${JOB_MAX}-s job cap: TERM"; kill -TERM "$jp"; break; fi
  done
  wait "$jp"; rc=$?
  if (( admitted == 0 )); then flock -u 7; exec 7>&-; fi
  side_stop
  if (( ${#shared_kinds[@]} )) && { [[ "$(for k in "${shared_kinds[@]}"; do cat "$S/shared/$k.pid"; done | tr '\n' ' ')" != "$shared_pids" ]] || ! shared_ok "${shared_kinds[@]}"; }; then
    for x in "$RUNS/$name" "$RUNS/$name".*; do [[ -e "$x" ]] && mv "$x" "${x/$name/$name.sidedown}"; done
    rm -f "$S/running.$LANE" "$S/job.$LANE"; drop_lock
    log "SHARED SIDE DOWN during $name (${shared_kinds[*]}): run dir -> .sidedown (not a row), job put back, every lane stops"
    front "$line"; touch "$S/STOP"; break
  fi
  rm -f "$S/running.$LANE" "$S/job.$LANE"
  drop_lock; tr=$(date +%s)
  read -r nok summ < <(rows "$RUNS/$name")
  log "done $name rc=$rc$([[ -n "$killed" ]] && echo " killed=$killed") lock $(( tr - t0 )) s | $summ"
  if [[ "$killed" == 429 ]] || grep -rqs "HTTP 429" "$RUNS/$name/agentic" "$RUNS/$name/perception_service.log"; then
    echo "$(date -u +%FT%TZ) $name agentic unavailable (429)" >> "$S/429.marks"; touch "$S/STOP"
    log "429 marked; lanes stopped"; break
  fi
  if [[ "${nok:-0}" -eq 0 && "$rc" -eq 75 && -z "$killed" ]]; then   # resources (RAM / VRAM) never came: not the job's fault
    front "$line"; log "rc 75 (resources): $name put back at the front"; sleep 120; continue
  fi
  if [[ "${nok:-0}" -eq 0 ]]; then
    if [[ "$name" != *__retry* ]]; then front "${name}__retry ${line#* }"; log "requeued once as ${name}__retry (no finished episode)"
    else log "DROPPED: $name finished no episode twice"; fi
  fi
  share $(( tr - t0 ))
done
rm -f "$S/running.$LANE" "$S/job.$LANE"
exit
}
