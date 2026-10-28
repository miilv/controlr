#!/usr/bin/env bash
# One-command bootstrap of a new GPU box for the dojo streams, run from the LOCAL box over ssh:
#   RoboDojo (Isaac Sim 5.1 + IsaacLab + cuRobo + XPolicyLab) at compute3's pinned commits + every Asset the dev tasks
#   need, controlr main + venv (perception extras) + .env (OMNIROUTE_* only), the perception catalogue + shape caches,
#   our dev layouts + D0 references, on role physics compute3's Isaac Sim 6.0 render build, GPU-lock / scope
#   conventions, ~/dojo-box.json, smoke tests. Several boxes at once: scripts/bootstrap_fleet.sh.
#
#   scripts/bootstrap_gpu_box.sh HOST [--role auto|full|physics] [--dry-run] [--only S1,S2] [--skip S1,S2] [--force S1,S2]
#        [--probe full|basic] [--assets needed|all] [--assets-from hf|compute3|HOST] [--with-isaac6] [--ref origin/main]
#        [--source compute3] [--env-file PATH | --no-secrets] [--min-disk-gb 300] [--prefix DIR] [--ignore-preflight]
#        [--shared-home]
#
#   --role auto     (default) the preflight reads the driver: 570/580 -> full, anything else (595) -> physics; and if the
#                   5.1 render smoke then fails on a "full" box, the box falls back to physics by itself
#   --role full     Isaac Sim 5.1 renders here: official dev_eval, the bridge, 5.1 frames (driver 570/580 required)
#   --role physics  like compute3: 5.1 physics-only (twin, G3) + compute3's Isaac Sim 6.0 build for the proxy_eval render
#   --dry-run       print every command (local and remote) and touch nothing; no ssh
#   --only preflight  the read-only preflight alone: writes nothing on HOST. --probe basic: ONLY nvidia-smi, df, free,
#                   nproc on HOST (another team's box), evaluated here
#   --assets-from   where the 13.5 GB of Assets come from: hf (git-lfs batch API, default), compute3 (relayed through this
#                   box), or a box that has them; in a fleet, peers copy from whoever has them already
#   --env-file      the .env whose OMNIROUTE_* lines go to HOST:~/controlr/.env (mode 600; default: the main checkout's
#                   .env). Values are never printed or logged. --no-secrets: skip it
#   --prefix DIR    big disk not under $HOME (e.g. /data): ~/dojo-robodojo, ~/dojo-data, ... -> DIR/...
#   --shared-home   HOST's $HOME is shared with other people (compute2's /root): no loginctl enable-linger, no
#                   ~/.config/uv/uv.toml, our pinned uv in ~/dojo-bootstrap/tools/uv (the home's ~/.local/bin/uv stays)
#   --fleet-dir DIR (set by bootstrap_fleet.sh) peer-to-peer fetches coordinated through DIR on this box
#
# Steps (resumable, idempotent: a re-run skips what is done and re-attaches to what is still running on HOST;
# the remote ones run detached, so a dropped ssh connection loses nothing):
#   preflight conventions secrets code venv robodojo assets data catalog isaac6 catalog_post smoke_perception
#   smoke_bridge assets_post smoke_twin smoke_render smoke_dev_eval smoke_proxy_eval ready
# Runbook: docs/RUNBOOK-new-gpu-box.md. Remote half: scripts/infra/box_steps.sh. Logs: runs/bootstrap/<host>/ here
# (+ summary.tsv), ~/dojo-bootstrap/logs/<step>.log on HOST.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBDIR="$ROOT/scripts/infra"

HOST="" ROLE=auto DRY=0 ONLY="" SKIP="" FORCE="" ASSETS=needed REF=origin/main SOURCE=compute3 MIN_DISK_GB=300 IGNORE_PF=0
PREFIX="" PROBE=full ASSETS_FROM=hf WITH_ISAAC6=0 ENV_FILE="" SECRETS=1 FLEET_DIR="" SUMMARY="" P2P=direct SHARED_HOME=0
ISAAC6_SRC="${ISAAC6_SRC:-/home/physicalai/AAAI_MultiAgenticSIM/isaac-sim-6.0}"   # compute3's Isaac Sim 6.0 (isaac6.sh default)
ISAAC6_VERSION="6.0.0-rc.59+release.41464.5f2772bc.gl"
FLEET_WAIT_S="${FLEET_WAIT_S:-5400}"     # a fleet peer waits this long for a holder, then fetches from the origin itself
FLEET_POLL_S="${FLEET_POLL_S:-15}"       # how often a waiting peer looks for a holder
RUNS="${BOOT_RUNS_DIR:-$ROOT/runs/bootstrap}"   # local logs + summary.tsv (tests point it elsewhere)
usage() { sed -n '2,/^set -uo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --role) ROLE="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --only) ONLY="$2"; shift 2 ;;
    --skip) SKIP="$2"; shift 2 ;;
    --force) FORCE="$2"; shift 2 ;;
    --probe) PROBE="$2"; shift 2 ;;
    --assets) ASSETS="$2"; shift 2 ;;
    --assets-from) ASSETS_FROM="$2"; shift 2 ;;
    --with-isaac6) WITH_ISAAC6=1; shift ;;
    --ref) REF="$2"; shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --no-secrets) SECRETS=0; shift ;;
    --min-disk-gb) MIN_DISK_GB="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --ignore-preflight) IGNORE_PF=1; shift ;;
    --shared-home) SHARED_HOME=1; shift ;;
    --fleet-dir) FLEET_DIR="$2"; shift 2 ;;
    --p2p) P2P="$2"; shift 2 ;;
    --summary) SUMMARY="$2"; shift 2 ;;
    -h|--help) usage 0 ;;
    -*) echo "unknown option $1" >&2; usage 2 ;;
    *) [[ -z "$HOST" ]] || { echo "one HOST only (several: scripts/bootstrap_fleet.sh)" >&2; exit 2; }; HOST="$1"; shift ;;
  esac
done
[[ -n "$HOST" ]] || usage 2
[[ "$ROLE" == auto || "$ROLE" == full || "$ROLE" == physics ]] || { echo "--role auto|full|physics" >&2; exit 2; }
[[ "$ASSETS" == needed || "$ASSETS" == all ]] || { echo "--assets needed|all" >&2; exit 2; }
[[ "$PROBE" == full || "$PROBE" == basic ]] || { echo "--probe full|basic" >&2; exit 2; }
[[ "$PROBE" == full || "$ONLY" == preflight ]] || { echo "--probe basic is for --only preflight (it checks too little to install on)" >&2; exit 2; }
[[ "$P2P" == direct || "$P2P" == relay ]] || { echo "--p2p direct|relay" >&2; exit 2; }
if [[ -z "$ENV_FILE" ]]; then          # the main checkout's .env (worktrees have none: CLAUDE.md rule 5)
  ENV_FILE="$ROOT/.env"
  [[ -f "$ENV_FILE" ]] || ENV_FILE="$(cd "$(git -C "$ROOT" rev-parse --git-common-dir 2>/dev/null || echo "$ROOT/.git")/.." && pwd)/.env"
fi

ALL_STEPS=(preflight conventions secrets code venv robodojo assets data catalog isaac6 catalog_post smoke_perception
           smoke_bridge assets_post smoke_twin smoke_render smoke_dev_eval smoke_proxy_eval ready)
FETCH_STEPS=(assets data catalog isaac6)          # copied from an origin (HF / compute3) or, in a fleet, from a peer
declare -A DEPS=(
  [venv]="code" [robodojo]="code" [catalog_post]="venv catalog" [assets_post]="robodojo assets"
  [smoke_perception]="catalog_post data venv" [smoke_bridge]="venv code"
  [smoke_twin]="assets_post data venv" [smoke_render]="assets_post data venv" [smoke_dev_eval]="assets_post data venv"
  [smoke_proxy_eval]="assets_post data venv isaac6"
)
declare -A STATE=() T0=() SECS=() FPID=() DIRECT=()
BR=dojo-bootstrap                      # on HOST, relative to its home
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=8
          -o ControlMaster=auto -o "ControlPath=/tmp/.cr-boot-%C" -o ControlPersist=15m)
# box-to-box ssh (fleet key from bootstrap_fleet.sh; paths relative to the box's home, where rsync starts ssh)
FLEET_SSH="ssh -i $BR/fleet_key -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$BR/fleet_known_hosts -o ConnectTimeout=15 -o ServerAliveInterval=15"
T_START=$(date +%s)
ROLE_AUTO=0 ROLE_REASON="" FELL_BACK=0 PF_GO="" PF_GPU="" PF_DRIVER="" PF_ROLE=""
[[ "$ROLE" == auto ]] && ROLE_AUTO=1
if [[ "$DRY" == 1 ]]; then SSHC=ssh; else SSHC="ssh $(printf '%q ' "${SSH_OPTS[@]}")"; fi
RUN_TMP="$(mktemp -d)"

# ----------------------------------------------------------------------------------------------- helpers
ts() { date -u +%H:%M:%S; }
log() { if [[ "$DRY" == 1 ]]; then echo "  # $*"; else echo "[$(ts)] $*"; fi; }
RS() { if [[ "$DRY" == 1 ]]; then echo "  + $1"; return 0; fi; log "+ $1"; bash -c "set -o pipefail; $1"; }  # local snippet
# shellcheck disable=SC2029  # remote commands are built here on purpose
rssh() { ssh "${SSH_OPTS[@]}" "$HOST" "$@"; }
# shellcheck disable=SC2029
hssh() { local h="$1"; shift; ssh "${SSH_OPTS[@]}" "$h" "$@"; }
in_list() { [[ ",$2," == *",$1,"* ]]; }
want() {   # does this run include step $1?
  [[ -n "$ONLY" ]] && ! in_list "$1" "$ONLY" && return 1
  [[ -n "$SKIP" ]] && in_list "$1" "$SKIP" && return 1
  case "$1" in
    smoke_render|smoke_dev_eval) [[ "$ROLE" == full ]] ;;
    isaac6|smoke_proxy_eval) [[ "$ROLE" == physics || "$WITH_ISAAC6" == 1 ]] ;;
    secrets) [[ "$SECRETS" == 1 ]] ;;
    *) return 0 ;;
  esac
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
      CODE_SHA="$CODE_SHA" MIN_DISK_GB="$MIN_DISK_GB" PREFIX="$PREFIX" SHARED_HOME="$SHARED_HOME" bash "$LIBDIR/box_steps.sh" "$@" )
  rmdir "$d" 2>/dev/null || true
}
push_config() {   # this run's config -> HOST:~/dojo-bootstrap/config.env (sourced by every remote step)
  local cfg="ROLE=\${ROLE:-$ROLE}
ROLE_REASON=\${ROLE_REASON:-'${ROLE_REASON//\'/}'}
BOX_ALIAS=\${BOX_ALIAS:-'${HOST//\'/}'}
ASSETS=\${ASSETS:-$ASSETS}
CODE_SHA=\${CODE_SHA:-$CODE_SHA}
MIN_DISK_GB=\${MIN_DISK_GB:-$MIN_DISK_GB}
PREFIX=\${PREFIX:-$PREFIX}
SHARED_HOME=\${SHARED_HOME:-$SHARED_HOME}"
  if [[ "$DRY" == 1 ]]; then echo "  + ssh $HOST 'cat > $BR/config.env'   # ROLE=$ROLE ASSETS=$ASSETS CODE_SHA=$CODE_SHA MIN_DISK_GB=$MIN_DISK_GB PREFIX=$PREFIX SHARED_HOME=$SHARED_HOME BOX_ALIAS=$HOST"
  else printf '%s\n' "$cfg" | rssh "cat > $BR/config.env"; fi
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
  want "$s" || { STATE[$s]="${STATE[$s]:-skipped}"; return 0; }
  if b="$(blocked_by "$s")"; then STATE[$s]=blocked; log "$s: blocked by $b"; return 0; fi
  header "$s"; launch "$s"; wait_steps "$s"
}

# ----------------------------------------------------------------------------------------------- copies
relay() {   # SRC_HOST SRC_DIR DST_DIR [EXCLUDE...]: incremental copy SRC_HOST:SRC_DIR -> HOST:DST_DIR streamed through this
            # box (nothing stored here). Dirs relative to the homes unless absolute; EXCLUDEs are pruned (never listed).
  local sh="$1" src="$2" dst="$3" e prune=""; shift 3
  for e in "$@"; do prune+=" -path ./$e -prune -o"; done
  local lst="find .$prune \\( -type f -o -type l \\) -printf '%P\\t%s\\t%T@\\n'"
  local dirs="find . -mindepth 1$prune -type d -printf '%P\\n'"
  local norm="awk -F'\\t' -v OFS='\\t' '{\$3 = int(\$3); print}' | LC_ALL=C sort"
  if [[ "$DRY" == 1 ]]; then
    echo "  + comm -23 <(ssh $sh \"cd $src && $lst\" | $norm) <(ssh $HOST \"mkdir -p $dst && cd $dst && $lst\" | $norm) | cut -f1 > need"
    printf '%s\n' "  + tr '\\n' '\\0' < need | ssh $sh \"cd $src && nice -n 10 tar --null -T - -cf -\" | ssh $HOST \"cd $dst && tar -xf -\""
    printf '%s\n' "  + ssh $sh \"cd $src && $dirs\" | ssh $HOST \"cd $dst && xargs -r -d '\\n' mkdir -p\"   # empty dirs too"
    return 0
  fi
  local tmp; tmp="$(mktemp -d)"
  hssh "$sh" "cd $src && $lst" | bash -c "$norm" > "$tmp/src" || { log "relay: cannot list $sh:$src"; rm -rf "$tmp"; return 1; }
  rssh "mkdir -p $dst && cd $dst && $lst" | bash -c "$norm" > "$tmp/dst" || { rm -rf "$tmp"; return 1; }
  LC_ALL=C comm -23 "$tmp/src" "$tmp/dst" > "$tmp/need"
  local n bytes; n=$(wc -l < "$tmp/need"); bytes=$(awk -F'\t' '{s += $2} END {printf "%.0f", s / 1e6}' "$tmp/need")
  if [[ "$n" -eq 0 ]]; then log "relay $sh:$src: up to date ($(wc -l < "$tmp/src") files)"; rm -rf "$tmp"; return 0; fi
  log "relay $sh:$src -> $HOST:$dst: $n files, $bytes MB"
  local t=$SECONDS
  cut -f1 "$tmp/need" | tr '\n' '\0' | hssh "$sh" "cd $src && nice -n 10 tar --null -T - -cf -" | rssh "cd $dst && tar -xf -" \
    || { log "relay $src: transfer failed (re-run resumes)"; rm -rf "$tmp"; return 1; }
  hssh "$sh" "cd $src && $dirs" | rssh "cd $dst && xargs -r -d '\\n' mkdir -p" || true
  rssh "cd $dst && $lst" | bash -c "$norm" > "$tmp/dst"
  local left; left=$(LC_ALL=C comm -23 "$tmp/src" "$tmp/dst" | wc -l)
  rm -rf "$tmp"
  [[ "$left" -eq 0 ]] || { log "relay $src: $left files still differ"; return 1; }
  log "relay $src: done, $bytes MB in $((SECONDS - t)) s"
}
direct_ok() {   # can HOST pull straight from fleet peer $1 (fleet key, rsync on both, reachable)? cached per peer
  local sh="$1" a
  [[ -n "$FLEET_DIR" && "$P2P" == direct && "$sh" != "$SOURCE" ]] || return 1
  if [[ "$DRY" == 1 ]]; then echo "  + ssh $HOST \"command -v rsync && $FLEET_SSH -p <port> <user@addr of $sh> 'command -v rsync'\"   # direct box-to-box?"; return 0; fi
  a="$(cat "$FLEET_DIR/addr/$sh" 2>/dev/null || true)"
  [[ -n "$a" && "$a" != relay ]] || return 1
  [[ -n "${DIRECT[$sh]:-}" ]] && { [[ "${DIRECT[$sh]}" == 1 ]]; return; }
  DIRECT[$sh]=0
  if rssh "test -f $BR/fleet_key && command -v rsync >/dev/null && $FLEET_SSH -p ${a##*:} ${a%:*} 'command -v rsync' >/dev/null" 2>/dev/null; then
    DIRECT[$sh]=1
  else log "direct $HOST <- $sh ($a) not possible (key / rsync / network): relaying through this box"; fi
  [[ "${DIRECT[$sh]}" == 1 ]]
}
xfer() {   # SRC_HOST SRC_DIR DST_DIR [EXCLUDE...]: box-to-box rsync over the fleet key when possible, else relay()
  local sh="$1" src="$2" dst="$3"; shift 3
  local a abs e ex="" xl
  for e in "$@"; do ex+=" --exclude /$e"; done
  xl="$BR/logs/xfer_${dst//[^A-Za-z0-9]/_}.log"
  if [[ "$DRY" == 1 && -n "$FLEET_DIR" && "$P2P" == direct && "$sh" != "$SOURCE" ]]; then
    direct_ok "$sh"
    echo "  + ssh $HOST \"mkdir -p $dst && rsync -a --partial --stats$ex -e '$FLEET_SSH -p <port>' <user@addr of $sh>:<abs path of $src>/ $dst/ > $xl 2>&1\""
    echo "  # else (no direct path between the two boxes): relayed through this box"
    relay "$sh" "$src" "$dst" "$@"; return 0
  fi
  if [[ "$DRY" != 1 ]] && direct_ok "$sh"; then
    a="$(cat "$FLEET_DIR/addr/$sh")"
    abs="$(hssh "$sh" "cd $src && pwd -P")" || { log "xfer: no $sh:$src"; return 1; }
    log "direct $sh:$src -> $HOST:$dst (rsync over the fleet key; log ~/$xl)"
    local t=$SECONDS
    # (sh syntax: the box's login shell may be dash; rsync's own exit code decides)
    if rssh "mkdir -p $dst && rsync -a --partial --stats$ex -e '$FLEET_SSH -p ${a##*:}' ${a%:*}:$abs/ $dst/ > $xl 2>&1; rc=\$?; grep -a -E 'Number of regular files transferred|Total transferred file size' $xl; exit \$rc"; then
      log "direct $src: done in $((SECONDS - t)) s"; return 0
    fi
    log "direct copy from $sh failed (~/$xl): relaying through this box instead"
  fi
  relay "$sh" "$src" "$dst" "$@"
}
DATA_DIRS=(dojo-data/simlab/layouts dojo-data/simlab/traj dojo-data/simlab/g2_c3b dojo-data/d0 dojo-data/d0_51 dojo-data/d0_ext
           dojo-data/skills2/catalog dojo-data/skills3/catalog)
CATALOG_DIRS=(dojo-perception-data/catalog dojo-perception-data/bg_folds dojo-perception-data/bg_folds_51
              dojo-perception-data/task_config dojo-perception-data/looks)
# overrides (space-separated, relative to the homes), e.g. for a partial copy or a test
[[ -n "${BOOT_DATA_DIRS:-}" ]] && read -r -a DATA_DIRS <<< "$BOOT_DATA_DIRS"
[[ -n "${BOOT_CATALOG_DIRS:-}" ]] && read -r -a CATALOG_DIRS <<< "$BOOT_CATALOG_DIRS"
HF_EXCL=(.git/lfs Assets/Eval_Layout Assets/Traj/RoboDojo/imitate_sorting_sequence)   # never copied, never listed
I6_DIR=dojo-isaac6/isaac-sim-6.0
isaac6_check() {
  if [[ "$DRY" == 1 ]]; then echo "  + ssh $HOST \"test \\\"\\\$(cat $I6_DIR/VERSION)\\\" = $ISAAC6_VERSION && $I6_DIR/kit/python/bin/python3 -c 'import sys; print(sys.version)'\""; return 0; fi
  # shellcheck disable=SC2016  # $(cat ...) runs on HOST
  rssh "v=\"\$(cat $I6_DIR/VERSION 2>/dev/null)\"; [ \"\$v\" = '$ISAAC6_VERSION' ] || { echo \"isaac6: VERSION '\$v' != $ISAAC6_VERSION\"; exit 1; }; $I6_DIR/kit/python/bin/python3 -c 'import sys; print(\"isaac6 python\", sys.version.split()[0])'"
}
copy_set() {   # STEP SRC_HOST [origin]: copy the STEP's directories from SRC_HOST
  local s="$1" sh="$2" origin="${3:-}" d rc=0
  case "$s" in
    assets) xfer "$sh" dojo-robodojo/hf_git dojo-robodojo/hf_git "${HF_EXCL[@]}" || return 1
            launch assets; wait_steps assets; [[ "${STATE[assets]}" == "done" ]] ;;      # the remote step verifies the copy
    data) for d in "${DATA_DIRS[@]}"; do xfer "$sh" "$d" "$d" || rc=1; done; return "$rc" ;;
    catalog) for d in "${CATALOG_DIRS[@]}"; do xfer "$sh" "$d" "$d" || rc=1; done; return "$rc" ;;
    isaac6) if [[ -n "$origin" ]]; then xfer "$sh" "$ISAAC6_SRC" "$I6_DIR" kit/cache kit/logs || return 1
            else xfer "$sh" "$I6_DIR" "$I6_DIR" || return 1; fi
            isaac6_check ;;
  esac
}
fetch_origin() {   # STEP: from the origin: HF for assets (--assets-from), $SOURCE (compute3) for the rest
  local s="$1"
  case "$s" in
    assets) if [[ "$ASSETS_FROM" == hf ]]; then launch assets; wait_steps assets; [[ "${STATE[assets]}" == "done" ]]
            else copy_set assets "$ASSETS_FROM" origin; fi ;;
    *) copy_set "$s" "$SOURCE" origin ;;
  esac
}
pick_source() {   # STEP -> ORIGIN | WAIT | a fleet host that has it (least busy); under one lock on this box
  local s="$1" f h n best="" bestn=999 seed
  (
    flock 8
    seed="$(cat "$FLEET_DIR/seed/$s" 2>/dev/null || true)"
    [[ "$seed" == "$HOST" ]] && { echo ORIGIN; exit 0; }
    [[ -z "$seed" ]] && { echo "$HOST" > "$FLEET_DIR/seed/$s"; echo ORIGIN; exit 0; }    # nobody fetches it yet: this box
    for f in "$FLEET_DIR/have/$s"/*; do
      [[ -e "$f" ]] || continue
      h="$(basename "$f")"
      [[ "$h" == "$HOST" || -e "$FLEET_DIR/bad/$s/$HOST/$h" ]] && continue
      n="$(find "$FLEET_DIR/busy/$s" -maxdepth 1 -name "$h.*" 2>/dev/null | wc -l)"
      [[ "$n" -lt "$bestn" ]] && { best="$h"; bestn="$n"; }
    done
    if [[ -n "$best" ]]; then mkdir -p "$FLEET_DIR/busy/$s"; touch "$FLEET_DIR/busy/$s/$best.$HOST"; echo "$best"; exit 0; fi
    if [[ -e "$FLEET_DIR/failed/$s/$seed" ]]; then echo "$HOST" > "$FLEET_DIR/seed/$s"; echo ORIGIN; exit 0; fi   # take over
    echo WAIT
  ) 8>>"$FLEET_DIR/.pick.lock"
}
fleet_mark() { [[ -n "$FLEET_DIR" && "$DRY" != 1 ]] || return 0; mkdir -p "$FLEET_DIR/$1/$2"; touch "$FLEET_DIR/$1/$2/$HOST"; }
fetch() {   # STEP: one fetch step; the outcome goes to $RUN_TMP/STEP.state (it runs in the background)
  local s="$1" t0=$SECONDS rc=1 src waited=0 seed=""
  [[ -n "$FLEET_DIR" ]] && seed="$(cat "$FLEET_DIR/seed/$s" 2>/dev/null || true)"
  if [[ "$DRY" == 1 ]]; then
    if [[ -z "$FLEET_DIR" || -z "$seed" || "$seed" == "$HOST" ]]; then
      [[ -n "$FLEET_DIR" ]] && log "fleet: $HOST is the seed for $s: fetch from the origin, then serve the other boxes"
      fetch_origin "$s"
    else
      log "fleet: wait until a fleet box has $s (seed $seed; up to $((FLEET_WAIT_S / 60)) min, then the origin), then copy from the least busy one:"
      copy_set "$s" "<peer>"
    fi
    STATE[$s]="done"; return 0
  fi
  if [[ "$s" == assets ]] && ! in_list assets "$FORCE" && [[ "$(rssh "bash $BR/lib/box_steps.sh status assets" | awk '{print $2}')" == "done" ]]; then
    log "assets: done already (skip; --force assets to redo)"; rc=0
  elif [[ "$s" == isaac6 ]] && ! in_list isaac6 "$FORCE" && isaac6_check >/dev/null 2>&1 && rssh "test -f $BR/state/isaac6.copied"; then
    log "isaac6: $I6_DIR complete already (skip; --force isaac6 to re-check every file)"; rc=0
  elif [[ -z "$FLEET_DIR" ]]; then fetch_origin "$s"; rc=$?
  else
    while true; do
      src="$(pick_source "$s")"
      case "$src" in
        ORIGIN) [[ "$seed" == "$HOST" ]] || log "fleet: fetching $s from the origin (seed $seed failed or none)"
                fetch_origin "$s"; rc=$?; break ;;
        WAIT) [[ $((waited % (20 * FLEET_POLL_S))) == 0 ]] && log "fleet: waiting for a box that has $s (seed $seed), ${waited} s so far"
              if [[ "$waited" -ge "$FLEET_WAIT_S" ]]; then log "fleet: waited ${waited} s for $s: fetching from the origin"
                fetch_origin "$s"; rc=$?; break; fi
              sleep "$FLEET_POLL_S"; waited=$((waited + FLEET_POLL_S)) ;;
        *) log "fleet: copying $s from $src"
           copy_set "$s" "$src"; rc=$?
           rm -f "$FLEET_DIR/busy/$s/$src.$HOST"
           [[ "$rc" == 0 ]] && break
           mkdir -p "$FLEET_DIR/bad/$s/$HOST"; touch "$FLEET_DIR/bad/$s/$HOST/$src"; log "fleet: copy of $s from $src failed; next source" ;;
      esac
    done
  fi
  if [[ "$rc" == 0 ]]; then
    fleet_mark have "$s"
    [[ "$s" == isaac6 ]] && rssh "touch $BR/state/isaac6.copied"
  else
    [[ "$(cat "$FLEET_DIR/seed/$s" 2>/dev/null)" == "$HOST" ]] && fleet_mark failed "$s"
  fi
  echo "$( [[ "$rc" == 0 ]] && echo "done" || echo failed) $((SECONDS - t0))" > "$RUN_TMP/$s.state"
  log "$s: $( [[ "$rc" == 0 ]] && echo "complete" || echo "FAILED (re-run resumes)") after $((SECONDS - t0)) s"
  return "$rc"
}
start_fetch() {   # STEP in the background (prefixed log lines); STATE comes back through wait_fetch
  local s="$1" b
  want "$s" || { STATE[$s]=skipped; return 0; }
  if [[ "$DRY" == 1 ]]; then echo "  # $s"; fetch "$s"; return 0; fi
  ( fetch "$s" 2>&1 | sed -u "s/^/[$s] /" ) &
  FPID[$s]=$!; STATE[$s]=running
}
wait_fetch() {
  local s
  for s in "$@"; do
    [[ -n "${FPID[$s]:-}" ]] || continue
    wait "${FPID[$s]}" 2>/dev/null; unset "FPID[$s]"
    read -r STATE["$s"] SECS["$s"] < "$RUN_TMP/$s.state" 2>/dev/null || STATE[$s]=failed
  done
}

# ----------------------------------------------------------------------------------------------- preflight
pf_facts() {   # "@ key value" lines of a preflight -> PF_* (the rest is printed)
  local line
  while IFS= read -r line; do
    case "$line" in
      "@ gpu "*) PF_GPU="${line#@ gpu }" ;; "@ driver "*) PF_DRIVER="${line#@ driver }" ;; "@ role "*) PF_ROLE="${line#@ role }" ;;
      "@ "*) ;;
      *) echo "$line" ;;
    esac
  done
}
basic_probe() {   # ONLY nvidia-smi, df, free, nproc on HOST (another team's box); the verdict is computed here
  # shellcheck disable=SC2016  # $HOME / $PREFIX expand on HOST
  local cmd='nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader,nounits; echo @@; df -BG --output=avail,target '"${PREFIX:-\"\$HOME\"}"'; echo @@; free -g; echo @@; nproc'
  if [[ "$DRY" == 1 ]]; then echo "  + ssh $HOST '$cmd'   # nothing else runs there; evaluated here"; return 0; fi
  local out rc; out="$(rssh "$cmd" 2>&1)"; rc=$?
  [[ "$rc" -eq 255 ]] && { echo "$out"; return 255; }
  local fail=0 warn=0 sec=0 gpu="" df="" mem="" cpus="" line
  while IFS= read -r line; do
    if [[ "$line" == "@@" ]]; then sec=$((sec + 1)); continue; fi
    case "$sec" in 0) [[ -z "$gpu" ]] && gpu="$line" ;; 1) df+="$line"$'\n' ;; 2) mem+="$line"$'\n' ;; 3) cpus="$line" ;; esac
  done <<< "$out"
  p() { printf '%-5s %-14s %s\n' "$1" "$2" "$3"; [[ "$1" == FAIL ]] && fail=$((fail + 1)); [[ "$1" == WARN ]] && warn=$((warn + 1)); return 0; }
  echo "preflight $HOST $(date -u +%FT%TZ) role=$ROLE (basic probe: nvidia-smi, df, free, nproc only)"
  local name drv vram used major eff="$ROLE"
  if [[ -z "$gpu" || "$gpu" == *failed* || "$gpu" == *"not found"* || "$gpu" != *,* ]]; then p FAIL gpu "nvidia-smi: ${gpu:-no output}"
  else
    IFS=',' read -r name drv vram used <<< "$gpu"
    name="${name# }"; drv="${drv// /}"; vram="${vram// /}"; used="${used// /}"; major="${drv%%.*}"
    PF_GPU="$name"; PF_DRIVER="$drv"
    p PASS gpu "$name, $((vram / 1024)) GB VRAM"
    if [[ " ${BOX_DRIVER_OK:-570 580} " == *" $major "* ]]; then [[ "$ROLE" == auto ]] && eff=full; p PASS driver "$drv (Isaac Sim 5.1 RTX tested on 580)"
    elif [[ "$ROLE" == full ]]; then p FAIL driver "$drv: Isaac Sim 5.1 RTX needs driver 570 / 580 (595 segfaults in librtx.scenedb)"
    else [[ "$ROLE" == auto ]] && eff=physics; p WARN driver "$drv: no 5.1 rendering: role physics"; fi
    [[ "$vram" -lt 12000 ]] && p FAIL vram "${vram} MiB < 12 GB"
    [[ "$used" -gt 2000 ]] && p WARN gpu_busy "${used} MiB used: someone else is on this GPU"
  fi
  local avail mnt; read -r avail mnt <<< "$(sed -n 2p <<< "$df")"; avail="${avail//[!0-9]/}"
  if [[ -z "$avail" ]]; then p FAIL disk "df failed"
  elif [[ "$avail" -lt "$MIN_DISK_GB" ]]; then p FAIL disk "${avail} GB free on $mnt < ${MIN_DISK_GB} GB"
  else p PASS disk "${avail} GB free on $mnt"; fi
  local mt ma; read -r mt ma <<< "$(awk '/^Mem:/ {print $2, $7}' <<< "$mem")"
  if [[ "${mt:-0}" -lt 30 ]]; then p FAIL ram "${mt:-?} GB < 32 GB"
  elif [[ "${ma:-0}" -lt 25 || "$mt" -lt 60 ]]; then p WARN ram "${mt} GB total, ${ma} GB available"
  else p PASS ram "${mt} GB total, ${ma} GB available"; fi
  if [[ "${cpus:-0}" -lt 12 ]]; then p WARN cpu "${cpus:-?} threads < 14"; else p PASS cpu "$cpus threads"; fi
  [[ "$eff" == auto ]] && eff=physics
  PF_ROLE="$eff"
  p INFO unchecked "basic probe: os / glibc, build tools, Vulkan, EGL, network, systemd, inotify not checked (--probe full on our own boxes)"
  echo
  if [[ "$fail" -eq 0 ]]; then echo "GO ($warn warnings, basic probe)"; return 0; fi
  echo "NO-GO ($fail failed, $warn warnings)"; return 1
}
run_preflight() {   # -> PF_GO, PF_ROLE; exit codes: 0 GO, 1 NO-GO, 255 ssh
  local rc
  if [[ "$PROBE" == basic ]]; then basic_probe; rc=$?
  elif [[ "$DRY" == 1 ]]; then
    echo "  + ssh $HOST 'ROLE=$ROLE MIN_DISK_GB=$MIN_DISK_GB PREFIX=$PREFIX bash -s -- preflight' < scripts/infra/box_steps.sh"
    remote_dry preflight; rc=0
  else
    rssh "ROLE=$ROLE MIN_DISK_GB=$MIN_DISK_GB PREFIX=$PREFIX bash -s -- preflight" < "$LIBDIR/box_steps.sh" > "$RUN_TMP/pf.out" 2>&1; rc=$?
    pf_facts < "$RUN_TMP/pf.out"
    [[ "$rc" -eq 255 ]] && log "ssh to $HOST failed"
  fi
  if [[ "$rc" -eq 0 ]]; then PF_GO=GO; elif [[ "$rc" -eq 255 ]]; then PF_GO=SSH-FAIL; else PF_GO=NO-GO; fi
  return "$rc"
}

# ----------------------------------------------------------------------------------------------- secrets
secrets() {   # OMNIROUTE_* lines of the local .env -> HOST:~/controlr/.env (mode 600). Values never printed or logged.
  local f="$ENV_FILE" pat='^(export[[:space:]]+)?OMNIROUTE_[A-Z0-9_]+='
  local put="umask 077 && mkdir -p controlr && cat > controlr/.env.new && chmod 600 controlr/.env.new && mv controlr/.env.new controlr/.env"
  if [[ ! -f "$f" ]]; then log "secrets: no $f: skipped (the agent stream needs OMNIROUTE_* in ~/controlr/.env)"; STATE[secrets]=skipped; return 0; fi
  local names; names="$(grep -oE "${pat%=}" "$f" | sed -E 's/^export[[:space:]]+//' | sort -u | paste -sd' ' -)"
  if [[ -z "$names" ]]; then log "secrets: no OMNIROUTE_* in $f: skipped"; STATE[secrets]=skipped; return 0; fi
  if [[ "$DRY" == 1 ]]; then
    echo "  + grep -E '$pat' $f | ssh $HOST '$put'   # keys: $names (values never printed)"
    echo "  + ssh $HOST 'stat -c %a controlr/.env; grep -c . controlr/.env'"
    STATE[secrets]="done"; return 0
  fi
  if grep -E "$pat" "$f" | rssh "$put" && [[ "$(rssh 'stat -c %a controlr/.env')" == 600 ]]; then
    log "secrets: $names -> $HOST:~/controlr/.env (mode 600)"; STATE[secrets]="done"
  else log "secrets: FAILED to write $HOST:~/controlr/.env"; STATE[secrets]=failed; fi
}

# ----------------------------------------------------------------------------------------------- summary
write_summary() {   # runs/bootstrap/<host>/summary.tsv (+ --summary FILE): what bootstrap_fleet.sh tabulates
  [[ "$DRY" == 1 || -z "${LOGDIR:-}" ]] && return 0
  local s f="$LOGDIR/summary.tsv" bad=0
  for s in "${ALL_STEPS[@]}"; do
    [[ "${STATE[$s]:-}" == failed || "${STATE[$s]:-}" == blocked ]] || continue
    [[ "$s" == smoke_render && "$FELL_BACK" == 1 ]] && continue      # the box fell back to physics: not a defect
    bad=1
  done
  {
    printf 'host\t%s\ngpu\t%s\ndriver\t%s\nrole\t%s\nrole_reason\t%s\ngo\t%s\nwall_s\t%s\nready\t%s\nlog\t%s\n' \
      "$HOST" "${PF_GPU:-?}" "${PF_DRIVER:-?}" "$ROLE" "$ROLE_REASON" "${PF_GO:-?}" "$(( $(date +%s) - T_START ))" \
      "$( [[ "$bad" == 0 && "${STATE[ready]:-}" == "done" ]] && echo yes || echo no)" "${LOG:-}"
    for s in "${ALL_STEPS[@]}"; do printf 'step\t%s\t%s\t%s\n' "$s" "${STATE[$s]:-not run}" "${SECS[$s]:-}"; done
  } > "$f.new" && mv "$f.new" "$f"
  [[ -n "$SUMMARY" ]] && cp "$f" "$SUMMARY"
  return 0
}
# shellcheck disable=SC2317,SC2329  # the EXIT trap calls it (SC2317 on shellcheck < 0.11)
on_exit() {
  local s
  for s in "${!FPID[@]}"; do kill "${FPID[$s]}" 2>/dev/null; done
  if [[ -n "$FLEET_DIR" && "$DRY" != 1 ]]; then   # a seed that stops before its fetches finish hands them over
    for s in "${FETCH_STEPS[@]}"; do
      [[ "$(cat "$FLEET_DIR/seed/$s" 2>/dev/null)" == "$HOST" && ! -e "$FLEET_DIR/have/$s/$HOST" ]] && fleet_mark failed "$s"
    done
  fi
  write_summary
  rm -rf "$RUN_TMP"
}
trap on_exit EXIT

# ----------------------------------------------------------------------------------------------- main
if [[ "$DRY" != 1 ]]; then
  LOGDIR="$RUNS/${HOST//[^A-Za-z0-9._-]/_}"; mkdir -p "$LOGDIR"
  LOG="$LOGDIR/$(date -u +%Y%m%dT%H%M%SZ).log"
  exec > >(tee -a "$LOG") 2>&1
  log "bootstrap $HOST role=$ROLE assets=$ASSETS (from $ASSETS_FROM) ref=$REF source=$SOURCE${FLEET_DIR:+ fleet=$FLEET_DIR} (local log $LOG)"
else
  echo "# DRY RUN: bootstrap $HOST role=$ROLE assets=$ASSETS (from $ASSETS_FROM) ref=$REF source=$SOURCE${FLEET_DIR:+ fleet}; nothing is executed"
  [[ "$ROLE" == auto ]] && echo "# role auto: the preflight decides (driver 570/580 -> full, else physics); this dry run shows role full (--role physics for the other)"
fi
[[ "$DRY" == 1 && "$ROLE" == auto ]] && ROLE=full
if [[ "$ONLY" != preflight ]]; then cat <<EOF

Estimated wall time (compute3 measurements, 2026-10-24/25; unknown networks may be 2x slower):
  step             downloads                                          wall
  preflight        -                                                  < 1 min
  conventions      uv 20 MB                                           < 1 min
  secrets, code    10 MB                                              < 1 min
  venv             ~0.4 GB wheels (usd-core, opencv, scipy)           1-3 min    }
  robodojo         ~10-14 GB (torch cu128, isaacsim 5.1) -> 24 GB env  10-20 min  } in parallel
  assets           $( [[ "$ASSETS" == all ]] && echo "40.6 GB (all but eval data)   " || echo "13.5 GB, 2.75k LFS files      ") HF $( [[ "$ASSETS" == all ]] && echo "30-50" || echo "10-15") min / a peer 1-5 min }
  data             1.9 GB layouts + D0 + D0-5.1 + skills catalogues   2-6 min    } relayed from $SOURCE
  catalog          2.9 GB perception catalogue + backgrounds          2-6 min    } (or a fleet peer)
  isaac6           physics only: 31 GB Isaac Sim 6.0 from $SOURCE       10-15 min / a peer 2-6 min }
  catalog_post     -                                                  < 1 min
  smoke_perception -                                                  1-3 min (during the installs)
  smoke_bridge     -                                                  < 1 min (CPU)
  assets_post      -                                                  1-2 min
  smoke_twin       -                                                  2-4 min (first Kit boot)
  smoke_render     full only                                          5-15 min (first RTX boot builds the shader cache)
  smoke_dev_eval   full only                                          4-8 min
  smoke_proxy_eval physics only                                       3-8 min (first 6.0 RTX boot)
  total            ~45-60 min (either role; the critical path is robodojo + the Isaac smokes)
EOF
fi

# code ref (resolved here; dry runs never touch the network)
if [[ "$DRY" != 1 ]] && [[ "$REF" == origin/* ]] && { want code || want venv; } && [[ "$ONLY" != preflight ]]; then
  git -C "$ROOT" fetch -q origin "${REF#origin/}" || log "WARN: git fetch failed; using the local $REF"
fi
CODE_SHA="$(git -C "$ROOT" rev-parse --verify -q "$REF^{commit}" || echo "<sha of $REF>")"

# 1. preflight: read-only, the script is streamed (nothing written to HOST); also decides role auto
if want preflight || [[ "$ROLE" == auto ]]; then
  header "preflight ($HOST, read-only$( [[ "$PROBE" == basic ]] && echo ", basic probe")$( want preflight || echo ": only to pick the role"))"
  run_preflight; rc=$?
  STATE[preflight]="done"
  if [[ "$rc" -eq 255 ]]; then STATE[preflight]=failed; exit 1; fi
  if [[ "$ROLE" == auto ]]; then
    ROLE="${PF_ROLE:-physics}"; ROLE_REASON="auto: driver ${PF_DRIVER:-unknown}"
    # an earlier run fell back to physics after a failed 5.1 render smoke: keep that (--role full retries 5.1)
    if [[ "$DRY" != 1 && "$ROLE" == full && "$ONLY" != preflight ]] \
        && rssh "grep -qs 'render smoke failed' $BR/config.env" 2>/dev/null; then
      ROLE=physics; ROLE_REASON="auto: driver ${PF_DRIVER:-?}, but the Isaac Sim 5.1 render smoke failed earlier -> physics"
    fi
    log "role auto -> $ROLE ($ROLE_REASON)"
  fi
  if [[ "$rc" -ne 0 ]] && want preflight; then
    STATE[preflight]=failed
    if [[ "$IGNORE_PF" != 1 ]]; then
      log "NO-GO: fix the FAIL lines (or --ignore-preflight / --role physics / --min-disk-gb / --prefix)"; exit 1
    fi
    log "NO-GO ignored (--ignore-preflight)"; STATE[preflight]="done"
  fi
  want preflight || STATE[preflight]=skipped
fi
remaining=0; for s in "${ALL_STEPS[@]:1}"; do want "$s" && remaining=1; done
[[ "$remaining" == 1 ]] || { log "preflight only: nothing written on $HOST"; exit "$( [[ "${STATE[preflight]:-}" == "done" ]] && echo 0 || echo 1)"; }
if [[ "$DRY" != 1 ]] && { want data || want catalog || want isaac6 || [[ "$ASSETS_FROM" == "$SOURCE" ]]; }; then
  if ! hssh "$SOURCE" true; then
    [[ -n "$FLEET_DIR" ]] || { log "source $SOURCE unreachable (data / catalog / isaac6 relays need it)"; exit 1; }
    log "WARN: source $SOURCE unreachable: this box can still copy data / catalog / isaac6 from fleet peers"
  fi
fi

# 2. push the remote library + this run's config (the library comes from THIS checkout, the code from $REF)
header "push scripts/infra -> ~/$BR/lib, config"
RS "tar -C $LIBDIR --exclude __pycache__ -cf - . | $SSHC $HOST 'mkdir -p $BR/state $BR/logs && rm -rf $BR/lib.new && mkdir $BR/lib.new && tar -xf - -C $BR/lib.new && rm -rf $BR/lib && mv $BR/lib.new $BR/lib'" \
  || { log "cannot write to $HOST"; exit 1; }
push_config || { log "cannot write to $HOST"; exit 1; }

run_remote conventions

# 3. secrets (.env, OMNIROUTE_* only), code: git archive of $REF streamed into ~/dojo-bootstrap/code.new, installed as ~/controlr
if want secrets; then header "secrets (OMNIROUTE_* of $ENV_FILE -> ~/controlr/.env, values never shown)"; secrets
else STATE[secrets]=skipped; fi
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

# 4. in parallel: venv, robodojo (remote, detached) + the fetches: assets, data, catalog, isaac6 (HF / compute3 / fleet peers)
header "parallel: venv robodojo (remote) + fetches assets data catalog$( want isaac6 && echo " isaac6") (origin: HF / $SOURCE; fleet peers when there are)"
PAR=()
for s in venv robodojo; do
  want "$s" || { STATE[$s]=skipped; continue; }
  if b="$(blocked_by "$s")"; then STATE[$s]=blocked; log "$s: blocked by $b"; continue; fi
  launch "$s"; PAR+=("$s")
done
for s in "${FETCH_STEPS[@]}"; do start_fetch "$s"; done

# perception and the bridge are usable first (venv + catalogue), while Isaac installs
wait_steps venv
run_remote smoke_bridge
wait_fetch catalog
run_remote catalog_post
wait_fetch data
run_remote smoke_perception
wait_steps "${PAR[@]}"
wait_fetch assets
run_remote assets_post
run_remote smoke_twin
run_remote smoke_render
if [[ "${STATE[smoke_render]:-}" == failed && "$ROLE_AUTO" == 1 ]]; then
  FELL_BACK=1; ROLE=physics; ROLE_REASON="auto: driver ${PF_DRIVER:-?}, but the Isaac Sim 5.1 render smoke failed -> physics"
  log "role -> physics: $ROLE_REASON; adding the Isaac Sim 6.0 install"
  push_config; rssh "rm -f $BR/state/conventions.done"; run_remote conventions     # ~/dojo-env.sh / DOJO_BOX.md for physics
  start_fetch isaac6
fi
run_remote smoke_dev_eval
wait_fetch isaac6
run_remote smoke_proxy_eval
run_remote ready

# ----------------------------------------------------------------------------------------------- summary
echo
if [[ "$DRY" == 1 ]]; then echo "# end of dry run"; exit 0; fi
write_summary
log "summary for $HOST (role $ROLE${ROLE_REASON:+, $ROLE_REASON}), $(( ($(date +%s) - T_START) / 60 )) min wall:"
for s in "${ALL_STEPS[@]}"; do printf '  %-17s %-8s %s\n' "$s" "${STATE[$s]:-not run}" "${SECS[$s]:+${SECS[$s]} s}"; done
rssh "for f in $BR/state/smoke_*.result; do [ -f \"\$f\" ] && echo \"  \$(basename \$f .result): \$(cat \$f)\"; done" 2>/dev/null
if [[ "$(sed -n 's/^ready\t//p' "$LOGDIR/summary.tsv")" == yes ]]; then log "box ready: ssh $HOST cat ~/dojo-box.json ~/dojo-robodojo/READY"; exit 0; fi
log "NOT ready: fix the failed step, then re-run the same command (done steps are skipped)"
exit 1
