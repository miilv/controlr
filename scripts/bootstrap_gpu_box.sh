#!/usr/bin/env bash
# One-command bootstrap of a new GPU box for the dojo streams, run from the LOCAL box over ssh:
#   RoboDojo (Isaac Sim 5.1 + IsaacLab + cuRobo + XPolicyLab) at compute3's pinned commits + every Asset the dev tasks
#   need, controlr main + venv (perception extras), the perception catalogue + shape caches, our dev layouts + D0
#   references (relayed from compute3 through this box: nothing is stored here), smoke tests, GPU-lock conventions.
#
#   scripts/bootstrap_gpu_box.sh HOST [--role full|physics] [--dry-run] [--only S1,S2] [--skip S1,S2] [--force S1,S2]
#        [--assets needed|all] [--ref origin/main] [--source compute3] [--min-disk-gb 300] [--prefix DIR] [--ignore-preflight]
#
#   --role full     (default) Isaac Sim 5.1 renders here: driver 570/580 required; render + official dev_eval smokes
#   --role physics  physics-only box (like compute3): no 5.1 rendering, those two smokes are skipped
#   --dry-run       print every command (local and remote) and touch nothing; no ssh
#   --only preflight  the read-only preflight alone (nvidia-smi, df, free, ...): writes nothing on HOST
#   --prefix DIR    big disk not under $HOME (e.g. /data): ~/dojo-robodojo, ~/dojo-data, ~/dojo-perception-data -> DIR/...
#
# Steps (resumable, idempotent: a re-run skips what is done and re-attaches to what is still running on HOST;
# the remote ones run detached, so a dropped ssh connection loses nothing):
#   preflight conventions code venv robodojo assets data catalog catalog_post assets_post
#   smoke_twin smoke_render smoke_dev_eval smoke_perception ready
# Runbook: docs/RUNBOOK-new-gpu-box.md. Remote half: scripts/infra/box_steps.sh. Logs: runs/bootstrap/<host>/ here,
# ~/dojo-bootstrap/logs/<step>.log on HOST.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBDIR="$ROOT/scripts/infra"

HOST="" ROLE=full DRY=0 ONLY="" SKIP="" FORCE="" ASSETS=needed REF=origin/main SOURCE=compute3 MIN_DISK_GB=300 IGNORE_PF=0 PREFIX=""
usage() { sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --role) ROLE="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --only) ONLY="$2"; shift 2 ;;
    --skip) SKIP="$2"; shift 2 ;;
    --force) FORCE="$2"; shift 2 ;;
    --assets) ASSETS="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --min-disk-gb) MIN_DISK_GB="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --ignore-preflight) IGNORE_PF=1; shift ;;
    -h|--help) usage 0 ;;
    -*) echo "unknown option $1" >&2; usage 2 ;;
    *) [[ -z "$HOST" ]] || { echo "one HOST only" >&2; exit 2; }; HOST="$1"; shift ;;
  esac
done
[[ -n "$HOST" ]] || usage 2
[[ "$ROLE" == full || "$ROLE" == physics ]] || { echo "--role full|physics" >&2; exit 2; }
[[ "$ASSETS" == needed || "$ASSETS" == all ]] || { echo "--assets needed|all" >&2; exit 2; }

ALL_STEPS=(preflight conventions code venv robodojo assets data catalog catalog_post assets_post
           smoke_twin smoke_render smoke_dev_eval smoke_perception ready)
declare -A DEPS=(
  [venv]="code" [robodojo]="code" [catalog_post]="venv catalog" [assets_post]="robodojo assets"
  [smoke_twin]="assets_post data venv" [smoke_render]="assets_post data venv" [smoke_dev_eval]="assets_post data venv"
  [smoke_perception]="catalog_post data venv"
)
declare -A STATE=() T0=() SECS=()
BR=dojo-bootstrap                      # on HOST, relative to its home
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=8
          -o ControlMaster=auto -o "ControlPath=/tmp/.cr-boot-%C" -o ControlPersist=15m)
T_START=$(date +%s)
if [[ "$DRY" == 1 ]]; then SSHC=ssh; else SSHC="ssh $(printf '%q ' "${SSH_OPTS[@]}")"; fi

# ----------------------------------------------------------------------------------------------- helpers
ts() { date -u +%H:%M:%S; }
log() { if [[ "$DRY" == 1 ]]; then echo "  # $*"; else echo "[$(ts)] $*"; fi; }
RS() { if [[ "$DRY" == 1 ]]; then echo "  + $1"; return 0; fi; log "+ $1"; bash -c "set -o pipefail; $1"; }  # local snippet
# shellcheck disable=SC2029  # remote commands are built here on purpose
rssh() { ssh "${SSH_OPTS[@]}" "$HOST" "$@"; }
# shellcheck disable=SC2029
sssh() { ssh "${SSH_OPTS[@]}" "$SOURCE" "$@"; }
in_list() { [[ ",$2," == *",$1,"* ]]; }
want() {   # does this run include step $1?
  [[ -n "$ONLY" ]] && ! in_list "$1" "$ONLY" && return 1
  [[ -n "$SKIP" ]] && in_list "$1" "$SKIP" && return 1
  [[ "$ROLE" != full && ( "$1" == smoke_render || "$1" == smoke_dev_eval ) ]] && return 1
  return 0
}
blocked_by() {   # a dependency that failed (or was blocked) in this run
  local d
  for d in ${DEPS[$1]:-}; do
    [[ "${STATE[$d]:-}" == failed || "${STATE[$d]:-}" == blocked ]] && { echo "$d"; return 0; }
  done
  return 1
}
header() { echo; if [[ "$DRY" == 1 ]]; then echo "### $1"; else log "=== $1"; fi; }

# ----------------------------------------------------------------------------------------------- remote steps
remote_dry() {   # print the commands a remote step would run (the remote library in DRY_RUN mode, run here)
  local d; d="$(mktemp -d)"
  # shellcheck disable=SC2088  # a literal ~: printed commands name the box's paths
  ( cd "$d" && DRY_RUN=1 HOME='~' BOX='~/dojo-bootstrap' LIB_REMOTE='~/dojo-bootstrap/lib' ROLE="$ROLE" ASSETS="$ASSETS" \
      CODE_SHA="$CODE_SHA" MIN_DISK_GB="$MIN_DISK_GB" PREFIX="$PREFIX" bash "$LIBDIR/box_steps.sh" "$@" )
  rmdir "$d" 2>/dev/null || true
}
launch() {   # start remote step $1 detached unless done / running
  local s="$1" st
  if [[ "$DRY" == 1 ]]; then
    echo "  + ssh $HOST 'setsid nohup bash $BR/lib/box_steps.sh run $s >> $BR/logs/$s.log 2>&1 < /dev/null &'"
    remote_dry run "$s"; STATE[$s]="done"; return 0
  fi
  # --force, and the steps that re-check every run (shape caches, READY): drop the marker first
  if in_list "$s" "$FORCE" || in_list "$s" "catalog_post,ready"; then rssh "rm -f $BR/state/$s.done"; fi
  st="$(rssh "bash $BR/lib/box_steps.sh status $s" | awk '{print $2}')"
  case "$st" in
    done) log "$s: done already (skip; --force $s to redo)"; STATE[$s]="done"; return 0 ;;
    running) log "$s: already running on $HOST, attaching" ;;
    *) rssh "setsid nohup bash $BR/lib/box_steps.sh run $s >> $BR/logs/$s.log 2>&1 < /dev/null &"
       log "$s: started on $HOST (log ~/$BR/logs/$s.log)" ;;
  esac
  STATE[$s]=running; T0[$s]=$(date +%s)
}
wait_steps() {   # until every given running step is done / failed
  [[ "$DRY" == 1 ]] && return 0
  local pend=() s
  for s in "$@"; do [[ "${STATE[$s]:-}" == running ]] && pend+=("$s"); done
  local nap=3 last_print=0
  while [[ ${#pend[@]} -gt 0 ]]; do
    sleep "$nap"; [[ "$nap" -lt 20 ]] && nap=$((nap + 3))
    local out; out="$(rssh "bash $BR/lib/box_steps.sh poll ${pend[*]}" 2>/dev/null)" || { log "poll failed (network?), retrying"; continue; }
    local still=() now st secs rest
    now=$(date +%s)
    while read -r s st secs rest; do
      [[ -n "$s" ]] || continue
      case "$st" in
        done) STATE[$s]="done"; SECS[$s]="$secs"; log "$s: done in ${secs} s" ;;
        failed) STATE[$s]=failed; SECS[$s]="$secs"; log "$s: FAILED after ${secs} s; last lines of ~/$BR/logs/$s.log:"
                rssh "tail -n 40 $BR/logs/$s.log" | sed 's/^/    | /' ;;
        todo) if [[ $((now - T0[$s])) -gt 90 ]]; then STATE[$s]=failed; log "$s: never started on $HOST:"; rssh "tail -n 20 $BR/logs/$s.log" | sed 's/^/    | /'
              else still+=("$s"); fi ;;
        *) still+=("$s")
           [[ $((now - last_print)) -ge 60 ]] && log "  $s running $(( (now - T0[$s]) / 60 )) min | $rest" ;;
      esac
    done <<< "$out"
    [[ $((now - last_print)) -ge 60 ]] && last_print=$now
    pend=("${still[@]}")
  done
}
run_remote() {   # launch + wait, honouring --only/--skip and failed dependencies
  local s="$1" b
  want "$s" || { STATE[$s]=skipped; return 0; }
  if b="$(blocked_by "$s")"; then STATE[$s]=blocked; log "$s: blocked by $b"; return 0; fi
  header "$s"; launch "$s"; wait_steps "$s"
}

# ----------------------------------------------------------------------------------------------- local steps
relay() {   # SRC_DIR DST_DIR (relative to the homes of $SOURCE / $HOST): incremental copy streamed through this box
  local src="$1" dst="$2"
  local lst="find . -type f -printf '%P\\t%s\\t%T@\\n'"
  local norm="awk -F'\\t' -v OFS='\\t' '{\$3 = int(\$3); print}' | LC_ALL=C sort"
  if [[ "$DRY" == 1 ]]; then
    echo "  + comm -23 <(ssh $SOURCE \"cd $src && $lst\" | $norm) <(ssh $HOST \"mkdir -p $dst && cd $dst && $lst\" | $norm) | cut -f1 > need"
    printf '%s\n' "  + tr '\\n' '\\0' < need | ssh $SOURCE \"cd $src && nice -n 10 tar --null -T - -cf -\" | ssh $HOST \"cd $dst && tar -xf -\""
    return 0
  fi
  local tmp; tmp="$(mktemp -d)"
  sssh "cd $src && $lst" | bash -c "$norm" > "$tmp/src" || { log "relay: cannot list $SOURCE:$src"; rm -rf "$tmp"; return 1; }
  rssh "mkdir -p $dst && cd $dst && $lst" | bash -c "$norm" > "$tmp/dst" || { rm -rf "$tmp"; return 1; }
  LC_ALL=C comm -23 "$tmp/src" "$tmp/dst" > "$tmp/need"
  local n bytes; n=$(wc -l < "$tmp/need"); bytes=$(awk -F'\t' '{s += $2} END {printf "%.0f", s / 1e6}' "$tmp/need")
  if [[ "$n" -eq 0 ]]; then log "relay $src: up to date ($(wc -l < "$tmp/src") files)"; rm -rf "$tmp"; return 0; fi
  log "relay $SOURCE:$src -> $HOST:$dst: $n files, $bytes MB"
  local t=$SECONDS
  cut -f1 "$tmp/need" | tr '\n' '\0' | sssh "cd $src && nice -n 10 tar --null -T - -cf -" | rssh "cd $dst && tar -xf -" \
    || { log "relay $src: transfer failed (re-run resumes)"; rm -rf "$tmp"; return 1; }
  rssh "cd $dst && $lst" | bash -c "$norm" > "$tmp/dst"
  local left; left=$(LC_ALL=C comm -23 "$tmp/src" "$tmp/dst" | wc -l)
  rm -rf "$tmp"
  [[ "$left" -eq 0 ]] || { log "relay $src: $left files still differ"; return 1; }
  log "relay $src: done, $bytes MB in $((SECONDS - t)) s"
}
DATA_DIRS=(dojo-data/simlab/layouts dojo-data/simlab/traj dojo-data/simlab/g2_c3b dojo-data/d0 dojo-data/d0_51 dojo-data/d0_ext
           dojo-data/skills2/catalog dojo-data/skills3/catalog)
CATALOG_DIRS=(dojo-perception-data/catalog dojo-perception-data/bg_folds dojo-perception-data/bg_folds_51
              dojo-perception-data/task_config dojo-perception-data/looks)
# overrides (space-separated, relative to the homes), e.g. for a partial copy or a test
[[ -n "${BOOT_DATA_DIRS:-}" ]] && read -r -a DATA_DIRS <<< "$BOOT_DATA_DIRS"
[[ -n "${BOOT_CATALOG_DIRS:-}" ]] && read -r -a CATALOG_DIRS <<< "$BOOT_CATALOG_DIRS"
relay_group() {   # NAME DIR...: one local step
  local name="$1" d rc=0; shift
  for d in "$@"; do relay "$d" "$d" || rc=1; done
  log "$name: $( [[ "$rc" == 0 ]] && echo complete || echo "INCOMPLETE")"
  return "$rc"
}

# ----------------------------------------------------------------------------------------------- main
if [[ "$DRY" != 1 ]]; then
  LOGDIR="$ROOT/runs/bootstrap/${HOST//[^A-Za-z0-9._-]/_}"; mkdir -p "$LOGDIR"
  LOG="$LOGDIR/$(date -u +%Y%m%dT%H%M%SZ).log"
  exec > >(tee -a "$LOG") 2>&1
  log "bootstrap $HOST role=$ROLE assets=$ASSETS ref=$REF source=$SOURCE (local log $LOG)"
else
  echo "# DRY RUN: bootstrap $HOST role=$ROLE assets=$ASSETS ref=$REF source=$SOURCE; nothing is executed"
fi
cat <<EOF

Estimated wall time (compute3 measurements, 2026-10-24/25; unknown networks may be 2x slower):
  step            downloads                                  wall        runs
  preflight       -                                          < 1 min
  code + venv     ~0.4 GB wheels (usd-core, opencv, scipy)   1-3 min
  robodojo        ~10-14 GB (torch cu128, isaacsim 5.1) -> 24 GB env      10-20 min  } in parallel
  assets          $( [[ "$ASSETS" == all ]] && echo "40.6 GB, 7.1k LFS files (all but eval data) " || echo "13.5 GB, 2.75k LFS files (pins list)      ")  $( [[ "$ASSETS" == all ]] && echo "30-50 min" || echo "10-15 min")  } (HF ~15-25 MB/s)
  data            1.9 GB layouts + D0 + D0-5.1 + skills catalogues  2-6 min    } relayed from $SOURCE
  catalog         2.9 GB perception catalogue + backgrounds      2-6 min    } (~50 MB/s from compute3)
  catalog_post    -                                          < 1 min
  assets_post     -                                          1-2 min
  smoke_twin      -                                          2-4 min (first Kit boot)
  smoke_render    -                                          5-15 min (first RTX boot builds the shader cache)
  smoke_dev_eval  -                                          4-8 min
  smoke_perception -                                         1-3 min (runs during the installs)
  total           ~45-60 min ($( [[ "$ROLE" == full ]] && echo "role full" || echo "role physics: ~35 min"))
EOF

# code ref (resolved here; dry runs never touch the network)
if [[ "$DRY" != 1 ]] && [[ "$REF" == origin/* ]] && { want code || want venv; }; then
  git -C "$ROOT" fetch -q origin "${REF#origin/}" || log "WARN: git fetch failed; using the local $REF"
fi
CODE_SHA="$(git -C "$ROOT" rev-parse --verify -q "$REF^{commit}" || echo "<sha of $REF>")"

# 1. preflight: read-only, the script is streamed (nothing written to HOST)
if want preflight; then
  header "preflight ($HOST, read-only)"
  if [[ "$DRY" == 1 ]]; then
    echo "  + ssh $HOST 'ROLE=$ROLE MIN_DISK_GB=$MIN_DISK_GB PREFIX=$PREFIX bash -s -- preflight' < scripts/infra/box_steps.sh"
    remote_dry preflight; STATE[preflight]="done"
  else
    rssh "ROLE=$ROLE MIN_DISK_GB=$MIN_DISK_GB PREFIX=$PREFIX bash -s -- preflight" < "$LIBDIR/box_steps.sh"; rc=$?
    if [[ "$rc" -eq 255 ]]; then STATE[preflight]=failed; log "ssh to $HOST failed"; exit 1; fi
    if [[ "$rc" -ne 0 ]]; then
      STATE[preflight]=failed
      [[ "$IGNORE_PF" == 1 ]] || { log "NO-GO: fix the FAIL lines (or --ignore-preflight / --role physics / --min-disk-gb)"; exit 1; }
      log "NO-GO ignored (--ignore-preflight)"; STATE[preflight]="done"
    else STATE[preflight]="done"; fi
    if want data || want catalog; then
      sssh true || { log "source $SOURCE unreachable (data / catalog relays need it)"; exit 1; }
    fi
  fi
fi
remaining=0; for s in "${ALL_STEPS[@]:1}"; do want "$s" && remaining=1; done
[[ "$remaining" == 1 ]] || { log "preflight only: nothing written on $HOST"; exit "$( [[ "${STATE[preflight]:-}" == "done" ]] && echo 0 || echo 1)"; }

# 2. push the remote library + this run's config (the library comes from THIS checkout, the code from $REF)
header "push scripts/infra -> ~/$BR/lib, config"
RS "tar -C $LIBDIR --exclude __pycache__ -cf - . | $SSHC $HOST 'mkdir -p $BR/state $BR/logs && rm -rf $BR/lib.new && mkdir $BR/lib.new && tar -xf - -C $BR/lib.new && rm -rf $BR/lib && mv $BR/lib.new $BR/lib'"
CFG="ROLE=\${ROLE:-$ROLE}
ASSETS=\${ASSETS:-$ASSETS}
CODE_SHA=\${CODE_SHA:-$CODE_SHA}
MIN_DISK_GB=\${MIN_DISK_GB:-$MIN_DISK_GB}
PREFIX=\${PREFIX:-$PREFIX}"
if [[ "$DRY" == 1 ]]; then echo "  + ssh $HOST 'cat > $BR/config.env'   # ROLE=$ROLE ASSETS=$ASSETS CODE_SHA=$CODE_SHA MIN_DISK_GB=$MIN_DISK_GB PREFIX=$PREFIX"
else printf '%s\n' "$CFG" | rssh "cat > $BR/config.env" || { log "cannot write to $HOST"; exit 1; }; fi

run_remote conventions

# 3. code: git archive of $REF streamed into ~/dojo-bootstrap/code.new, installed as ~/controlr
if want code; then
  header "code ($REF = ${CODE_SHA:0:12})"
  st=todo
  [[ "$DRY" == 1 ]] || st="$(rssh "bash $BR/lib/box_steps.sh status code" | awk '{print $2}')"
  if [[ "$st" == "done" ]] && ! in_list code "$FORCE"; then log "code: ${CODE_SHA:0:12} deployed already"; STATE[code]="done"
  else
    RS "git -C $ROOT archive --format=tar $CODE_SHA | $SSHC $HOST 'rm -rf $BR/code.new && mkdir -p $BR/code.new && tar -xf - -C $BR/code.new'" \
      && launch code && wait_steps code || STATE[code]=failed
  fi
else STATE[code]=skipped; fi

# 4. in parallel: venv, robodojo, assets (remote, detached) + data / catalog relays (here, streamed)
header "parallel: venv robodojo assets (remote) + data catalog (relays from $SOURCE)"
PAR=()
for s in venv robodojo assets; do
  want "$s" || { STATE[$s]=skipped; continue; }
  if b="$(blocked_by "$s")"; then STATE[$s]=blocked; log "$s: blocked by $b"; continue; fi
  launch "$s"; PAR+=("$s")
done
RELAY_PIDS=()
for g in data catalog; do
  want "$g" || { STATE[$g]=skipped; continue; }
  if [[ "$DRY" == 1 ]]; then
    echo "  # $g"; if [[ "$g" == data ]]; then relay_group data "${DATA_DIRS[@]}"; else relay_group catalog "${CATALOG_DIRS[@]}"; fi
    STATE[$g]="done"
  else
    if [[ "$g" == data ]]; then dirs=("${DATA_DIRS[@]}"); else dirs=("${CATALOG_DIRS[@]}"); fi
    ( set -o pipefail; relay_group "$g" "${dirs[@]}" 2>&1 | sed -u "s/^/[$g] /" ) &
    RELAY_PIDS+=("$g:$!"); STATE[$g]=running
  fi
done
wait_relays() {
  local e
  for e in "${RELAY_PIDS[@]}"; do
    if wait "${e#*:}"; then STATE[${e%%:*}]="done"; else STATE[${e%%:*}]=failed; log "${e%%:*} relay FAILED (re-run resumes)"; fi
  done
  RELAY_PIDS=()
}
# perception is usable first: venv + catalogue -> shape caches -> perception smoke, while Isaac installs
wait_steps venv
wait_relays
run_remote catalog_post
run_remote smoke_perception
wait_steps "${PAR[@]}"
run_remote assets_post
run_remote smoke_twin
run_remote smoke_render
run_remote smoke_dev_eval
run_remote ready

# ----------------------------------------------------------------------------------------------- summary
echo
if [[ "$DRY" == 1 ]]; then echo "# end of dry run"; exit 0; fi
log "summary for $HOST (role $ROLE), $(( ($(date +%s) - T_START) / 60 )) min wall:"
bad=0
for s in "${ALL_STEPS[@]}"; do
  st="${STATE[$s]:-not run}"
  [[ "$st" == failed || "$st" == blocked ]] && bad=1
  printf '  %-17s %-8s %s\n' "$s" "$st" "${SECS[$s]:+${SECS[$s]} s}"
done
rssh "for f in $BR/state/smoke_*.result; do [ -f \"\$f\" ] && echo \"  \$(basename \$f .result): \$(cat \$f)\"; done" 2>/dev/null
if [[ "$bad" == 0 ]]; then log "box ready: ssh $HOST cat ~/dojo-robodojo/READY"
else log "NOT ready: fix the failed step, then re-run the same command (done steps are skipped)"; fi
exit "$bad"
