#!/usr/bin/env bash
# Bootstrap several GPU boxes at once: scripts/bootstrap_gpu_box.sh per host, in parallel, with peer-to-peer copies.
#
#   scripts/bootstrap_fleet.sh [--dry-run] [--seed HOST] [--p2p direct|relay] [--addr HOST=user@ip[:port]]...
#        [--keep-fleet-key] [bootstrap_gpu_box.sh options...] HOST[:full|:physics] HOST ...
#   scripts/bootstrap_fleet.sh --only preflight [--probe basic] HOST ...   # read-only, the GO / NO-GO table only
#   scripts/bootstrap_fleet.sh --remove-fleet-key HOST ...
#
# 1. Preflight of every host in parallel (read-only): GO / NO-GO, GPU, driver, role (auto: driver 570/580 -> full,
#    else physics). NO-GO hosts are reported and left alone.
# 2. --p2p direct (default): an ephemeral ed25519 "fleet key" (runs/bootstrap/fleet_key here, never in git) goes to
#    every GO box (~/dojo-bootstrap/fleet_key + one `restrict` line in ~/.ssh/authorized_keys), so boxes rsync from
#    each other directly. It is removed again at the end (--keep-fleet-key keeps it). Where a box cannot reach another
#    (NAT, ProxyJump), that copy is relayed through this box instead. --p2p relay: always through this box.
# 3. Every GO host's bootstrap in parallel. The seed (--seed, default the first GO host) fetches the Assets from HF and
#    data / catalogue from compute3; the first physics box fetches Isaac Sim 6.0 from compute3. Every other box waits
#    until SOME box has a set and copies it from the least busy holder; a box that has finished a copy serves the
#    next ones. If a seed fails, the next box that asks takes over its fetch. So HF is asked once, not once per box.
# 4. A table: host, GPU, driver, role, go/no-go, every step OK/FAIL, wall time, ready; ~/dojo-box.json of every
#    box collected into one file. Logs: runs/bootstrap/fleet-<stamp>/ (one log per host) + runs/bootstrap/<host>/.
#
# Options passed to every host: --role --only --skip --force --assets --assets-from --ref --source --env-file
# --no-secrets --min-disk-gb --prefix --with-isaac6 --ignore-preflight. HOST:full / HOST:physics overrides --role.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOOT="$ROOT/scripts/bootstrap_gpu_box.sh"
DRY=0 SEED="" P2P=direct KEEP_KEY=0 REMOVE_KEY=0 ROLE=auto ONLY="" PROBE=full REF=origin/main
RUNS="${BOOT_RUNS_DIR:-$ROOT/runs/bootstrap}"   # local logs (tests point it elsewhere)
TICK_S="${FLEET_TICK_S:-10}"
HOSTS=() PASS=()
declare -A HROLE=() ADDR_OVR=()
usage() { sed -n '2,/^set -uo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --seed) SEED="$2"; shift 2 ;;
    --p2p) P2P="$2"; shift 2 ;;
    --addr) ADDR_OVR["${2%%=*}"]="${2#*=}"; shift 2 ;;
    --keep-fleet-key) KEEP_KEY=1; shift ;;
    --remove-fleet-key) REMOVE_KEY=1; shift ;;
    --role) ROLE="$2"; shift 2 ;;
    --only) ONLY="$2"; PASS+=("$1" "$2"); shift 2 ;;
    --probe) PROBE="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --skip|--force|--assets|--assets-from|--source|--env-file|--min-disk-gb|--prefix) PASS+=("$1" "$2"); shift 2 ;;
    --no-secrets|--with-isaac6|--ignore-preflight) PASS+=("$1"); shift ;;
    -h|--help) usage 0 ;;
    -*) echo "unknown option $1" >&2; usage 2 ;;
    *) h="${1%%:*}"; HOSTS+=("$h"); [[ "$1" == *:* ]] && HROLE[$h]="${1#*:}"; shift ;;
  esac
done
[[ ${#HOSTS[@]} -gt 0 ]] || usage 2
[[ "$P2P" == direct || "$P2P" == relay ]] || { echo "--p2p direct|relay" >&2; exit 2; }
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=8
          -o ControlMaster=auto -o "ControlPath=/tmp/.cr-boot-%C" -o ControlPersist=15m)
# shellcheck disable=SC2029
hssh() { local h="$1"; shift; ssh "${SSH_OPTS[@]}" "$h" "$@"; }
ts() { date -u +%H:%M:%S; }
log() { if [[ "$DRY" == 1 ]]; then echo "# $*"; else echo "[$(ts)] [fleet] $*"; fi; }
role_of() { echo "${HROLE[$1]:-$ROLE}"; }
KEY="$RUNS/fleet_key"
# (sed -i replaces the file atomically: a failure leaves authorized_keys as it was, never truncated)
UNKEY="sed -i '/ dojo-fleet-key\$/d' .ssh/authorized_keys; rm -f dojo-bootstrap/fleet_key dojo-bootstrap/fleet_known_hosts"
remove_key() {   # HOST...
  local h
  for h in "$@"; do
    if [[ "$DRY" == 1 ]]; then echo "  + ssh $h \"$UNKEY\""; continue; fi
    if hssh "$h" "test -f .ssh/authorized_keys && { $UNKEY; }; true"; then log "$h: fleet key removed"
    else log "$h: could not remove the fleet key (ssh?): scripts/bootstrap_fleet.sh --remove-fleet-key $h"; fi
  done
}
if [[ "$REMOVE_KEY" == 1 ]]; then remove_key "${HOSTS[@]}"; exit 0; fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ "$DRY" == 1 ]]; then FD="$(mktemp -d)"; trap 'rm -rf "$FD"' EXIT
else FD="$RUNS/fleet-$STAMP"; fi
mkdir -p "$FD"/{logs,pf,seed,addr,have,busy,failed,bad,summary}
T_START=$(date +%s)
if [[ "$DRY" == 1 ]]; then echo "# DRY RUN: fleet of ${#HOSTS[@]} (${HOSTS[*]}), p2p $P2P; nothing is executed"
else log "fleet of ${#HOSTS[@]}: ${HOSTS[*]} (p2p $P2P); logs $FD"; fi

# ------------------------------------------------------------------------------------------------- 1. preflights
declare -A GO=() GPU=() DRV=() R=()
if [[ "$DRY" == 1 ]]; then
  echo; echo "### 1. preflight of every host, in parallel (read-only)"
  for h in "${HOSTS[@]}"; do
    echo "  + scripts/bootstrap_gpu_box.sh $h --only preflight --probe $PROBE --role $(role_of "$h") --summary $FD/pf/$h.tsv > logs/$h.preflight.log &"
    GO[$h]=GO; R[$h]="$(role_of "$h")"; [[ "${R[$h]}" == auto ]] && R[$h]=full; GPU[$h]="?"; DRV[$h]="?"
  done
  echo "  # role auto: driver 570/580 -> full, else physics; this dry run shows auto as full"
else
  pids=()
  for h in "${HOSTS[@]}"; do
    pp=(); for ((i = 0; i < ${#PASS[@]}; i++)); do   # the preflight needs only the disk gate's options
      case "${PASS[$i]}" in --min-disk-gb|--prefix) pp+=("${PASS[$i]}" "${PASS[$((i + 1))]}") ;; esac; done
    bash "$BOOT" "$h" --only preflight --probe "$PROBE" --role "$(role_of "$h")" "${pp[@]}" --summary "$FD/pf/$h.tsv" > "$FD/logs/$h.preflight.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}" 2>/dev/null
  for h in "${HOSTS[@]}"; do
    f="$FD/pf/$h.tsv"
    GO[$h]="$(sed -n 's/^go\t//p' "$f" 2>/dev/null)"; GO[$h]="${GO[$h]:-SSH-FAIL}"
    GPU[$h]="$(sed -n 's/^gpu\t//p' "$f" 2>/dev/null)"; DRV[$h]="$(sed -n 's/^driver\t//p' "$f" 2>/dev/null)"
    R[$h]="$(sed -n 's/^role\t//p' "$f" 2>/dev/null)"; R[$h]="${R[$h]:-$(role_of "$h")}"
  done
  echo
  printf '%-20s %-26s %-12s %-8s %s\n' host gpu driver role preflight
  for h in "${HOSTS[@]}"; do
    printf '%-20s %-26s %-12s %-8s %s\n' "$h" "${GPU[$h]:-?}" "${DRV[$h]:-?}" "${R[$h]}" "${GO[$h]}"
    [[ "${GO[$h]}" == GO ]] || sed -n '/^FAIL/p' "$FD/logs/$h.preflight.log" | sed 's/^/    /'
  done
fi
if [[ "$ONLY" == preflight ]]; then   # preflights only: nothing is written on any host (no fleet key either)
  log "preflight only: nothing written on ${HOSTS[*]}; logs $FD/logs"
  for h in "${HOSTS[@]}"; do [[ "${GO[$h]}" == GO ]] || exit 1; done; exit 0
fi
[[ "$PROBE" == full ]] || { log "--probe basic is for --only preflight"; exit 2; }
OK_HOSTS=()
for h in "${HOSTS[@]}"; do
  if [[ "${GO[$h]}" == GO ]] || { [[ " ${PASS[*]} " == *" --ignore-preflight "* && "${GO[$h]}" == NO-GO ]]; }; then OK_HOSTS+=("$h"); fi
done
[[ ${#OK_HOSTS[@]} -gt 0 ]] || { log "no host is GO: nothing to do (fix the FAIL lines above; logs $FD/logs)"; exit 1; }

# ------------------------------------------------------------------------------------------------- 2. plan: seeds, addresses, fleet key
[[ -z "$SEED" || " ${OK_HOSTS[*]} " == *" $SEED "* ]] || { log "--seed $SEED is not a GO host"; exit 1; }
SEED="${SEED:-${OK_HOSTS[0]}}"
I6SEED=""
[[ "${R[$SEED]}" == physics ]] && I6SEED="$SEED"
if [[ -z "$I6SEED" ]]; then for h in "${OK_HOSTS[@]}"; do [[ "${R[$h]}" == physics ]] && { I6SEED="$h"; break; }; done; fi
for s in assets data catalog; do echo "$SEED" > "$FD/seed/$s"; done
[[ -n "$I6SEED" ]] && echo "$I6SEED" > "$FD/seed/isaac6"     # none: the first box that needs it (a 5.1 fallback) claims it
for h in "${OK_HOSTS[@]}"; do
  if [[ -n "${ADDR_OVR[$h]:-}" ]]; then a="${ADDR_OVR[$h]}"; [[ "$a" == *:* ]] || a="$a:22"
  else
    cfg="$(ssh -G "$h" 2>/dev/null)"
    if grep -qiE '^(proxyjump|proxycommand) ' <<< "$cfg" && ! grep -qiE '^proxyjump none' <<< "$cfg"; then a=relay
    else a="$(awk '$1 == "user" {u = $2} $1 == "hostname" {n = $2} $1 == "port" {p = $2} END {print u "@" n ":" p}' <<< "$cfg")"; fi
  fi
  [[ "$a" =~ ^[^@[:space:]]+@[^:[:space:]]+:[0-9]+$ ]] || a=relay     # unknown: copies to / from this host go through here
  echo "$a" > "$FD/addr/$h"
done
echo
[[ "$DRY" == 1 ]] && echo "### 2. plan"
log "seed: $SEED fetches Assets from HF$( [[ " ${PASS[*]} " == *" --assets-from "* ]] && echo " (or --assets-from)") and data / catalogue from compute3; ${I6SEED:-no physics box} fetches Isaac Sim 6.0 from compute3; the others copy from whoever has a set"
for h in "${OK_HOSTS[@]}"; do log "  $h: role ${R[$h]}, reachable by peers as $(cat "$FD/addr/$h")"; done
USE_KEY=0
if [[ "$P2P" == direct && ${#OK_HOSTS[@]} -gt 1 ]]; then
  USE_KEY=1
  if [[ "$DRY" == 1 ]]; then
    echo "  + ssh-keygen -q -t ed25519 -N '' -C dojo-fleet-key -f $KEY   # once; runs/ is not in git"
    for h in "${OK_HOSTS[@]}"; do
      echo "  + ssh $h \"umask 077 && mkdir -p .ssh dojo-bootstrap && { grep -qF '<key body>' .ssh/authorized_keys || { sed -i '/ dojo-fleet-key\$/d' .ssh/authorized_keys && echo 'restrict <$KEY.pub>' >> .ssh/authorized_keys; }; } && cat > dojo-bootstrap/fleet_key && chmod 600 dojo-bootstrap/fleet_key\" < $KEY   # the private key: stdin only, never printed"
    done
  else
    mkdir -p "$(dirname "$KEY")"
    [[ -f "$KEY" ]] || ssh-keygen -q -t ed25519 -N '' -C dojo-fleet-key -f "$KEY" || { log "ssh-keygen failed: p2p relay"; USE_KEY=0; }
    chmod 600 "$KEY" 2>/dev/null
    pub="$(cat "$KEY.pub" 2>/dev/null)"; kbody="${pub#* }"; kbody="${kbody%% *}"
    for h in "${OK_HOSTS[@]}"; do
      [[ "$USE_KEY" == 1 ]] || break
      # append only (an older fleet key line is dropped with sed -i first); the private key travels on stdin
      if hssh "$h" "umask 077 && mkdir -p .ssh dojo-bootstrap && touch .ssh/authorized_keys && { grep -qF '$kbody' .ssh/authorized_keys || { sed -i '/ dojo-fleet-key\$/d' .ssh/authorized_keys && echo 'restrict $pub' >> .ssh/authorized_keys; }; } && cat > dojo-bootstrap/fleet_key && chmod 600 dojo-bootstrap/fleet_key" < "$KEY"; then
        log "$h: fleet key installed"
      else log "$h: fleet key NOT installed: copies to / from $h are relayed"; echo relay > "$FD/addr/$h"; fi
    done
  fi
fi

# ------------------------------------------------------------------------------------------------- 3. bootstraps in parallel
# one code commit for the whole fleet (and one git fetch here, not one per host racing on the same repo)
if [[ "$DRY" != 1 && "$REF" == origin/* ]]; then
  git -C "$ROOT" fetch -q origin "${REF#origin/}" || log "WARN: git fetch failed; using the local $REF"
fi
SHA="$(git -C "$ROOT" rev-parse --verify -q "$REF^{commit}" || echo "$REF")"
[[ "$DRY" == 1 ]] || log "code for every box: $REF = ${SHA:0:12}"
PASS+=(--ref "$( [[ "$DRY" == 1 ]] && echo "$REF" || echo "$SHA")")
echo
[[ "$DRY" == 1 ]] && echo "### 3. every GO host's bootstrap, in parallel (each one's dry run follows)"
declare -A PID=()
for h in "${OK_HOSTS[@]}"; do
  args=("$h" --role "${R[$h]}" --fleet-dir "$FD" --p2p "$P2P" --summary "$FD/summary/$h.tsv" "${PASS[@]}")
  # the role was decided by the preflight: keep "auto" for hosts that were auto, so a failed 5.1 render falls back
  [[ "$(role_of "$h")" == auto ]] && args[2]=auto
  if [[ "$DRY" == 1 ]]; then
    echo; echo "########## $h: scripts/bootstrap_gpu_box.sh ${args[*]} --dry-run"
    # the dry run of each host: the role the preflight would pick (auto shows as full)
    a2=("${args[@]}"); a2[2]="${R[$h]}"
    bash "$BOOT" "${a2[@]}" --dry-run | sed 's/^/  /'
  else
    bash "$BOOT" "${args[@]}" > "$FD/logs/$h.log" 2>&1 &
    PID[$h]=$!
    log "$h: bootstrap started (log $FD/logs/$h.log)"
  fi
done
if [[ "$DRY" != 1 ]]; then
  last=0
  while true; do
    running=0
    for h in "${!PID[@]}"; do kill -0 "${PID[$h]}" 2>/dev/null && running=$((running + 1)); done
    [[ "$running" == 0 ]] && break
    if [[ $(( $(date +%s) - last )) -ge 120 ]]; then
      last=$(date +%s); log "$running running, $(( (last - T_START) / 60 )) min:"
      for h in "${OK_HOSTS[@]}"; do
        kill -0 "${PID[$h]}" 2>/dev/null || { echo "    $h: finished"; continue; }
        echo "    $h: $(grep -a -v '^\s*$' "$FD/logs/$h.log" | tail -1 | cut -c1-140)"
      done
    fi
    sleep "$TICK_S"
  done
  for h in "${!PID[@]}"; do wait "${PID[$h]}" 2>/dev/null; done
fi

# ------------------------------------------------------------------------------------------------- 4. table
if [[ "$DRY" == 1 ]]; then
  echo; echo "### 4. table (runs/bootstrap/fleet-<stamp>/table.txt), dojo-box.json of every box -> boxes.json"
  [[ "$USE_KEY" == 1 && "$KEEP_KEY" != 1 ]] && { echo "### 5. fleet key removal"; remove_key "${OK_HOSTS[@]}"; }
  echo "# end of fleet dry run"; exit 0
fi
STEPS=(preflight conventions secrets code venv robodojo assets data catalog isaac6 catalog_post smoke_perception
       smoke_bridge assets_post smoke_twin smoke_render smoke_dev_eval smoke_proxy_eval ready)
ABBR=(pf conv sec code venv rd ast data cat i6 cpost perc brdg apost twin r51 deval prox ready)
cell() { case "$1" in done) echo OK ;; failed) echo FAIL ;; blocked) echo blk ;; skipped|"not run"|"") echo - ;; running) echo run ;; *) echo "$1" ;; esac; }
{
  printf '%-18s %-10s %-11s %-8s %-6s' host gpu driver role go
  for a in "${ABBR[@]}"; do printf ' %-5s' "$a"; done
  printf ' %6s %s\n' wall ready
  for h in "${HOSTS[@]}"; do
    f="$FD/summary/$h.tsv"; [[ -f "$f" ]] || f="$FD/pf/$h.tsv"
    g="$(sed -n 's/^gpu\t//p' "$f" 2>/dev/null)"; g="${g#NVIDIA }"; g="${g#GeForce }"
    printf '%-18s %-10s %-11s %-8s %-6s' "$h" "${g:0:10}" "$(sed -n 's/^driver\t//p' "$f" 2>/dev/null | cut -c1-11)" \
      "$(sed -n 's/^role\t//p' "$f" 2>/dev/null)" "$(sed -n 's/^go\t//p' "$f" 2>/dev/null)"
    for s in "${STEPS[@]}"; do printf ' %-5s' "$(cell "$(awk -F'\t' -v s="$s" '$1 == "step" && $2 == s {print $3}' "$f" 2>/dev/null)")"; done
    w="$(sed -n 's/^wall_s\t//p' "$f" 2>/dev/null)"
    printf ' %5sm %s\n' "$(( ${w:-0} / 60 ))" "$(sed -n 's/^ready\t//p' "$f" 2>/dev/null)"
  done
  echo "legend: OK / FAIL / blk = blocked by a failed dependency / - = not part of this role or run. Steps: ${STEPS[*]}"
  for h in "${HOSTS[@]}"; do
    r="$(sed -n 's/^role_reason\t//p' "$FD/summary/$h.tsv" 2>/dev/null)"; [[ -n "$r" ]] && echo "  $h: $r"
  done
} | tee "$FD/table.txt"
echo "[" > "$FD/boxes.json"; first=1
for h in "${OK_HOSTS[@]}"; do
  j="$(hssh "$h" 'cat ~/dojo-box.json' 2>/dev/null)" || continue
  [[ "$first" == 1 ]] || echo "," >> "$FD/boxes.json"; first=0
  echo "$j" >> "$FD/boxes.json"
done
echo "]" >> "$FD/boxes.json"
log "every box's ~/dojo-box.json: $FD/boxes.json; wall $(( ($(date +%s) - T_START) / 60 )) min"
[[ "$USE_KEY" == 1 && "$KEEP_KEY" != 1 ]] && remove_key "${OK_HOSTS[@]}"
bad=0
for h in "${HOSTS[@]}"; do [[ "$(sed -n 's/^ready\t//p' "$FD/summary/$h.tsv" 2>/dev/null)" == yes ]] || bad=1; done
exit "$bad"
