#!/usr/bin/env bash
# System layer for a CLEAN Ubuntu GPU box (the root half, before scripts/bootstrap_gpu_box.sh), run from the LOCAL
# box over ssh. One or several hosts; each runs in parallel with its own log.
#
#   SUDO_PASS=... scripts/provision_box.sh [--user U] [--ip IP] [--driver 580] [--no-reboot] [--dry-run] HOST [HOST...]
#
#   HOST          an ssh alias, or with --ip a new alias written to ~/.ssh/config (HostName IP, User U)
#   --user U      the box's sudo user (default: haptics)
#   --driver N    NVIDIA branch (default 580: Isaac Sim 5.1 renders on 570/580; 595 crashes it)
#   SUDO_PASS     the user's sudo password (env, or prompted once); SSH_PASS (default SUDO_PASS) logs in once to
#                 install this box's ~/.ssh/id_ed25519.pub. Neither is ever printed, logged or put on a command line:
#                 they travel over ssh stdin only.
#
# What it does on HOST (idempotent; a re-run skips what is done):
#   1. our ssh key into ~/.ssh/authorized_keys (password login once, then key only)
#   2. apt: the HWE kernel meta + Canonical-SIGNED prebuilt NVIDIA modules for it
#      (linux-generic-hwe-<ver>, linux-modules-nvidia-<N>-open-generic-hwe-<ver>), then nvidia-driver-<N>-open; no
#      DKMS: a DKMS module is signed by a local key that Secure Boot rejects until enrolled at the console (MokManager);
#      the signed modules load with Secure Boot on. Plus build-essential cmake git git-lfs curl ffmpeg.
#   3. loginctl enable-linger U; fs.inotify.max_user_watches=524288 (persistent)
#   4. reboot if the driver is not live yet (or a newer kernel is waiting), wait, check nvidia-smi + signer
# Then: scripts/bootstrap_gpu_box.sh HOST (one box) or scripts/bootstrap_fleet.sh HOST... (several).
# Logs: runs/provision/<host>.log here.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_NAME=haptics IP="" DRV=580 REBOOT=1 DRY=0 HOSTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --user) USER_NAME="$2"; shift 2 ;;
    --ip) IP="$2"; shift 2 ;;
    --driver) DRV="$2"; shift 2 ;;
    --no-reboot) REBOOT=0; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,/^set -uo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) HOSTS+=("$1"); shift ;;
  esac
done
[[ ${#HOSTS[@]} -gt 0 ]] || { echo "usage: $0 [opts] HOST [HOST...]" >&2; exit 2; }
[[ -n "$IP" && ${#HOSTS[@]} -gt 1 ]] && { echo "--ip takes one HOST" >&2; exit 2; }
PUB="$HOME/.ssh/id_ed25519.pub"; [[ -f "$PUB" ]] || { echo "no $PUB" >&2; exit 2; }
if [[ $DRY -eq 0 && -z "${SUDO_PASS:-}" ]]; then read -r -s -p "sudo password for $USER_NAME: " SUDO_PASS; echo; fi
SSH_PASS="${SSH_PASS:-${SUDO_PASS:-}}"
LOGDIR="$ROOT/runs/provision"; mkdir -p "$LOGDIR"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)

# the root half, streamed to HOST as a file; reads nothing secret (sudo gets the password on stdin)
ROOT_SCRIPT='
set -x
export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a
. /etc/os-release; VER="$VERSION_ID"; DRV="$1"; U="$2"
case "$VER" in 22.04|24.04) ;; *) echo "UNSUPPORTED Ubuntu $VER"; exit 3 ;; esac
for i in $(seq 1 60); do fuser /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock >/dev/null 2>&1 || break; echo "dpkg busy, waiting"; sleep 10; done
apt-get update -q
apt-get install -y -q "linux-generic-hwe-$VER" "linux-modules-nvidia-$DRV-open-generic-hwe-$VER" || exit 4
apt-get install -y -q "nvidia-driver-$DRV-open" build-essential cmake git git-lfs curl ffmpeg || exit 5
dpkg -l "nvidia-dkms-$DRV-open" 2>/dev/null | grep -q ^ii && apt-get purge -y -q "nvidia-dkms-$DRV-open"
loginctl enable-linger "$U"
echo "fs.inotify.max_user_watches=524288" > /etc/sysctl.d/90-dojo-inotify.conf; sysctl -p /etc/sysctl.d/90-dojo-inotify.conf
NEWEST=$(ls /lib/modules | grep -E "^[0-9]" | sort -V | tail -1)   # /lib/modules can hold non-kernel dirs ("kernel")
echo "NEWEST_KERNEL=$NEWEST RUNNING=$(uname -r)"
echo "SIGNER=$(modinfo -k "$NEWEST" -F signer nvidia 2>/dev/null)"
echo "SECUREBOOT=$(mokutil --sb-state 2>/dev/null | head -1)"
echo ROOT_DONE
'

provision_one() {
  local h="$1" log="$LOGDIR/$1.log"
  exec >>"$log" 2>&1
  echo "=== $(date -u +%FT%TZ) provision $h user=$USER_NAME driver=$DRV"
  if [[ -n "$IP" ]] && ! grep -q "^Host $h\$" "$HOME/.ssh/config" 2>/dev/null; then
    printf '\nHost %s\n    HostName %s\n    User %s\n    IdentityFile ~/.ssh/id_ed25519\n' "$h" "$IP" "$USER_NAME" >> "$HOME/.ssh/config"
  fi
  if [[ $DRY -eq 1 ]]; then echo "[dry] key, apt (driver $DRV, signed modules), linger, inotify, reboot if needed"; return 0; fi
  if ! ssh "${SSH_OPTS[@]}" "$h" true 2>/dev/null; then
    command -v sshpass >/dev/null || { echo "FAIL key: no key access and no sshpass"; return 1; }
    local pub; pub="$(cat "$PUB")"
    SSHPASS="$SSH_PASS" sshpass -e ssh -o StrictHostKeyChecking=accept-new -o PubkeyAuthentication=no "$h" \
      "umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; grep -qxF '$pub' ~/.ssh/authorized_keys || echo '$pub' >> ~/.ssh/authorized_keys" \
      || { echo "FAIL key: password login"; return 1; }
    ssh "${SSH_OPTS[@]}" "$h" true || { echo "FAIL key: still no key access"; return 1; }
  fi
  echo "key OK"
  printf '%s' "$ROOT_SCRIPT" | ssh "${SSH_OPTS[@]}" "$h" 'cat > /tmp/dojo_provision_root.sh'
  printf '%s\n' "$SUDO_PASS" | ssh "${SSH_OPTS[@]}" "$h" \
    "sudo -S -p '' bash /tmp/dojo_provision_root.sh $DRV $USER_NAME > /tmp/dojo_provision_root.log 2>&1; tail -6 /tmp/dojo_provision_root.log"
  local out; out="$(ssh "${SSH_OPTS[@]}" "$h" 'cat /tmp/dojo_provision_root.log')"
  grep -q ROOT_DONE <<<"$out" || { echo "FAIL apt (see HOST:/tmp/dojo_provision_root.log)"; return 1; }
  grep -q "SIGNER=Canonical" <<<"$out" || echo "WARN: the newest kernel's nvidia module is not Canonical-signed (Secure Boot may reject it)"
  local newest running live
  newest="$(sed -n 's/^NEWEST_KERNEL=\([^ ]*\).*/\1/p' <<<"$out" | tail -1)"
  running="$(ssh "${SSH_OPTS[@]}" "$h" uname -r)"
  live="$(ssh "${SSH_OPTS[@]}" "$h" 'nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1')"
  [[ "$newest" =~ ^[0-9] ]] || newest="$running"   # unparsed: do not reboot on it
  if [[ "${live%%.*}" != "$DRV" || "$newest" != "$running" ]]; then
    if [[ $REBOOT -eq 0 ]]; then echo "NEEDS REBOOT (driver '$live', kernel $running -> $newest)"; return 0; fi
    echo "rebooting (driver '$live', kernel $running -> $newest)"
    printf '%s\n' "$SUDO_PASS" | ssh "${SSH_OPTS[@]}" "$h" "sudo -S -p '' systemctl reboot" || true
    sleep 60
    local t=0; until ssh "${SSH_OPTS[@]}" "$h" true 2>/dev/null; do sleep 15; t=$((t+15)); [[ $t -gt 900 ]] && { echo "FAIL: not back after reboot"; return 1; }; done
    live="$(ssh "${SSH_OPTS[@]}" "$h" 'nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1')"
  fi
  local gpu; gpu="$(ssh "${SSH_OPTS[@]}" "$h" 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1')"
  if [[ "${live%%.*}" == "$DRV" ]]; then echo "READY $h: $gpu, driver $live, kernel $(ssh "${SSH_OPTS[@]}" "$h" uname -r)"
  else echo "FAIL driver: nvidia-smi says '$live'"; return 1; fi
}

rc=0; pids=()
for h in "${HOSTS[@]}"; do provision_one "$h" & pids+=($!); done
for i in "${!pids[@]}"; do wait "${pids[$i]}" || rc=1; echo "${HOSTS[$i]}: $(grep -E '^(READY|FAIL|NEEDS|WARN|\[dry\])' "$LOGDIR/${HOSTS[$i]}.log" | tail -2 | tr '\n' ' ')"; done
[[ $rc -eq 0 ]] && echo "next: scripts/bootstrap_fleet.sh ${HOSTS[*]}"
exit $rc
