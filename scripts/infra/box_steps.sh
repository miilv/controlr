#!/usr/bin/env bash
# Remote half of scripts/bootstrap_gpu_box.sh. Runs ON the GPU box: the driver streams it for the preflight
# (nothing is written to the box) and pushes it to ~/dojo-bootstrap/lib/ for every other step.
#
#   box_steps.sh preflight                 read-only checks -> PASS/WARN/FAIL lines, "@ key value" facts, GO / NO-GO (exit 0 / 1)
#   box_steps.sh run STEP                  run one step; marker state/STEP.done (fingerprint) on success
#   box_steps.sh status STEP...            done | running | failed | todo, one line per step
#   box_steps.sh poll STEP...              status + the last line of each step's log (for the driver)
#   box_steps.sh steps                     the step names
#   box_steps.sh venv DIR                  DIR/.venv with the perception extras (for a stream's deploy dir)
#   box_steps.sh boxjson                   (re)write ~/dojo-box.json: role, specs, capabilities, locks, scopes, smokes
#   box_steps.sh manifest DIR [NAME [REF_MD5 REF_HOST]]
#                                          md5 manifest of DIR (every file, sorted path + md5): prints it; with NAME also
#                                          state/manifest_NAME.{md5,json} (+ whether it equals REF_HOST's), in ~/dojo-box.json
#
# Config: env vars, or $BOX/config.env written by the driver (ROLE, ASSETS, CODE_SHA, MIN_DISK_GB, PREFIX, BOX_ALIAS,
# ROLE_REASON, SHARED_HOME). DRY_RUN=1: print every command instead of running it; every "is it done already?" check answers no.
# Roles: full = Isaac Sim 5.1 renders here (driver 570/580): official dev_eval, the bridge, 5.1 frames;
#        physics = like compute3: 5.1 physics-only (twin, G3) + an Isaac Sim 6.0 render install (proxy_eval);
#        auto (preflight only) = full on driver 570/580, else physics.
# Layout on the box (compute3 conventions, dojo/README.md): ~/dojo-robodojo (RoboDojo + Isaac Sim 5.1, read-only
# for streams), ~/dojo-isaac6 (Isaac Sim 6.0, role physics), ~/controlr (main), ~/dojo-data/<stream>/,
# ~/dojo-perception-data/, ~/dojo-gpu.lock + ~/dojo-gpu-light{,2,3,4}.lock, ~/dojo-env.sh, ~/dojo-box.json.
set -uo pipefail

LIB="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo .)"
DRY="${DRY_RUN:-0}"
BOX="${BOX:-$HOME/dojo-bootstrap}"
# shellcheck source=/dev/null
[[ "$DRY" != 1 && -f "$BOX/config.env" ]] && source "$BOX/config.env"
ROLE="${ROLE:-full}"                       # full | physics (| auto: the preflight picks one from the driver)
ROLE_REASON="${ROLE_REASON:-}"             # why this role (the driver writes it: "driver 580", "5.1 render smoke failed", ...)
BOX_ALIAS="${BOX_ALIAS:-}"                 # the ssh name the driver used for this box
ASSETS="${ASSETS:-needed}"                 # needed: pins/robodojo_assets.txt (13.5 GB); all: every Asset but eval data (40.6 GB)
MIN_DISK_GB="${MIN_DISK_GB:-300}"
PREFIX="${PREFIX:-}"                       # big disk elsewhere: ~/dojo-{robodojo,data,perception-data} become symlinks into it
SHARED_HOME="${SHARED_HOME:-0}"            # 1: this $HOME is shared with other people (compute2's /root): no lingering, no
                                           # ~/.config/uv/uv.toml, our pinned uv in $BOX/tools/uv instead of ~/.local/bin
CODE_SHA="${CODE_SHA:-unknown}"
RD_HOME="${RD_HOME:-$HOME/dojo-robodojo}"
SRC="$RD_HOME/RoboDojo"
CONDA="$RD_HOME/miniconda3"
ENVP="$CONDA/envs/robodojo"
RDPY="$ENVP/bin/python"
HFG="$RD_HOME/hf_git"
ISAAC6_HOME="${ISAAC6_HOME:-$HOME/dojo-isaac6}"
ISAAC6="$ISAAC6_HOME/isaac-sim-6.0"                  # role physics: compute3's Isaac Sim 6.0 build, copied (scripts/simlab/isaac6.sh)
ISAAC6_VERSION="6.0.0-rc.59+release.41464.5f2772bc.gl"   # compute3's isaac-sim-6.0/VERSION (the proxy-render calibration was fit on it)
CODE="${CODE_DIR:-$HOME/controlr}"
VENV="$CODE/.venv"
DATA="${DOJO_DATA:-$HOME/dojo-data}"
PERC="${DOJO_PERCEPTION_DATA:-$HOME/dojo-perception-data}"
SMOKE="$DATA/infra/smoke"
PINS="$LIB/pins"                                      # read here
ASSETS_LIST="${ASSETS_PINS:-$PINS/robodojo_assets.txt}"   # ASSETS_PINS: another list (tests, a partial install)
CALIB_PINS="${CALIB_SETS_PINS:-$PINS/calib_sets.txt}"      # CALIB_SETS_PINS: another pin file (tests)
LIBX="${LIB_REMOTE:-$LIB}"                           # the same files as named in commands (dry runs print the box path)
HF_URL=https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo
HF_REV=269457a3b42d8d0ae69898a601f2c1487c2f826e      # HF dataset main on 2026-10-25 = compute3's hf_git HEAD
UV_VERSION=0.12.1                                     # compute3's uv
GIT_LFS_VERSION=3.4.1                                 # compute3's git-lfs (only if the box has none)
DRIVER_OK="${BOX_DRIVER_OK:-570 580}"                 # Isaac Sim 5.1 RTX: 580 renders (isr-lab 4090), 595 segfaults
STEPS=(conventions code venv robodojo assets assets_post catalog_post calib_sets smoke_twin smoke_render smoke_dev_eval
       smoke_proxy_eval smoke_perception smoke_bridge ready)
LIGHT_LOCKS=(dojo-gpu-light.lock dojo-gpu-light2.lock dojo-gpu-light3.lock dojo-gpu-light4.lock)   # physics-only light lane slots
# systemd-run --user --scope settings of the team (dojo/README.md, proxy_eval.sh, g3.sh); exported by ~/dojo-env.sh
SCOPE_BUNDLE="-p MemoryMax=24G -p CPUQuota=1400%"     # proxy_eval --bundle / integ_eval / a bridge episode: ONE GPU job
SCOPE_PROXY="-p MemoryMax=16G -p CPUQuota=1200%"      # proxy_eval without --bundle (episode env + 6.0 render service)
SCOPE_BATCH="-p MemoryMax=20G -p CPUQuota=1200%"      # presettle / d0_ext / looks / demos / settle_check
SCOPE_G3="-p MemoryMax=10G -p CPUQuota=600%"          # g3 on the heavy lane
SCOPE_LIGHT="-p MemoryMax=8G -p CPUQuota=600%"        # one light-lane slot (physics-only, no render service)
MIN_AVAIL_GB=25                                       # start a heavy job only with >= 25 GB MemAvailable (light lane: 20)
if [[ "$SHARED_HOME" == 1 ]]; then UV_DIR="$BOX/tools/uv"; else UV_DIR="$HOME/.local/bin"; fi   # where our pinned uv lives
export PATH="$UV_DIR:$HOME/.local/bin:$PATH" UV_PYTHON_PREFERENCE=only-managed GIT_TERMINAL_PROMPT=0
export RD_HOME DOJO_DATA="$DATA" OMNI_KIT_ACCEPT_EULA=YES ISAAC_SIM_ROOT="$ISAAC6"

# --------------------------------------------------------------------------------------------- helpers
ts() { date -u +%H:%M:%S; }
log() { if [[ "$DRY" == 1 ]]; then echo "    # $*"; else echo "[$(ts)] $*"; fi; }
dnote() { [[ "$DRY" != 1 ]] || echo "    # $*"; }      # explains a conditional branch in a dry run
die() { echo "[$(ts)] ERROR: $*" >&2; exit 1; }
q() {   # shell-quote for display; leaves plain words (paths with ~, = and :) readable
  local out=() a
  for a in "$@"; do
    if [[ "$a" =~ ^[A-Za-z0-9_@%+=:,./~^-]+$ ]]; then out+=("$a"); else out+=("'${a//\'/\'\\\'\'}'"); fi
  done
  echo "${out[*]}"
}
x() {   # run one command (or print it in a dry run)
  if [[ "$DRY" == 1 ]]; then echo "    + $(q "$@")"; return 0; fi
  echo "[$(ts)] + $(q "$@")"; "$@"
}
xs() {  # run one shell snippet with pipes / redirects (or print it)
  if [[ "$DRY" == 1 ]]; then echo "    + $1"; return 0; fi
  echo "[$(ts)] + $1"; bash -c "set -o pipefail; $1"
}
is() { [[ "$DRY" != 1 ]] && "$@"; }                # idempotency check: always "not yet" in a dry run
have() { [[ "$DRY" != 1 ]] && command -v "$1" >/dev/null 2>&1; }
hash12() { cat "$@" 2>/dev/null | sha256sum | cut -c1-12; }
pin_sha() { awk -v p="$1" '($2 == p) {s=$1; sub(/^[+ -]/, "", s); print s}' "$PINS/robodojo_commit.txt"; }
rd_commit() { head -1 "$PINS/robodojo_commit.txt" | tr -d ' +'; }
note() { [[ "$DRY" == 1 ]] && return 0; mkdir -p "$BOX/state"; echo "$*" > "$BOX/state/$CUR.result"; log "RESULT $*"; }

# fingerprints: a step is done when its marker holds the current one (a pin / commit change re-runs it)
fp_conventions() { echo "v2 $ROLE ${PREFIX:-home}$( [[ "$SHARED_HOME" == 1 ]] && echo " shared")"; }
fp_code() { echo "$CODE_SHA"; }
fp_venv() { echo "py3.12 $(hash12 "$PINS/controlr-venv.txt")"; }
fp_robodojo() { echo "$(hash12 "$PINS/robodojo_commit.txt") uv$UV_VERSION"; }
fp_assets() { echo "$ASSETS $HF_REV $(hash12 "$ASSETS_LIST")"; }
fp_assets_post() { echo "$(fp_robodojo) $(fp_assets)"; }
fp_catalog_post() { echo "always"; }
fp_calib_sets() { echo "$(hash12 "$CALIB_PINS") ${CALIB_EXTRA_ROOTS:-}"; }
fp_smoke_twin() { echo "$CODE_SHA $(fp_assets_post)"; }
fp_smoke_render() { fp_smoke_twin; }
fp_smoke_dev_eval() { fp_smoke_twin; }
fp_smoke_proxy_eval() { echo "$(fp_smoke_twin) $ISAAC6_VERSION"; }
fp_smoke_bridge() { echo "$CODE_SHA $(fp_venv)"; }
fp_smoke_perception() { echo "$CODE_SHA $(fp_venv)"; }
fp_ready() { echo "always"; }

# --------------------------------------------------------------------------------------------- preflight
# Read-only: nvidia-smi, df, /proc, command -v, ldconfig -p, systemctl --user show-environment, curl HEADs.
# Writes nothing on the box. "@ key value" lines are facts for the driver (role, GPU, driver, disk, RAM, CPU).
PF_FAIL=0 PF_WARN=0
pf() {  # LEVEL CHECK DETAIL
  printf '%-5s %-14s %s\n' "$1" "$2" "$3"
  [[ "$1" == FAIL ]] && PF_FAIL=$((PF_FAIL + 1))
  [[ "$1" == WARN ]] && PF_WARN=$((PF_WARN + 1))
  return 0
}
preflight() {
  if [[ "$DRY" == 1 ]]; then
    local c
    for c in "hostname" "nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader,nounits" \
             "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits" "df -BG --output=avail ~" \
             "grep -E 'MemTotal|MemAvailable' /proc/meminfo" "nproc" "lscpu" "tailscale status --self --peers=false" \
             "ldd --version" "cat /etc/os-release" "cat /proc/sys/fs/inotify/max_user_watches" "df -BG /dev/shm" \
             "ls /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json" "ldconfig -p | grep libEGL_nvidia" \
             "command -v git git-lfs curl tar flock setsid python3 ffmpeg rsync systemd-run" "systemctl --user show-environment" \
             "curl -sS -o /dev/null -w '%{http_code}' --max-time 10 <huggingface.co pypi.nvidia.com download.pytorch.org github.com repo.anaconda.com astral.sh>"; do
      echo "    + $c"
    done
    return 0
  fi
  echo "preflight $(hostname) $(date -u +%FT%TZ) role=$ROLE (read-only)"
  # GPU + driver -> the role (auto: full where Isaac Sim 5.1 RTX renders, physics elsewhere)
  local gpu name drv vram used major eff="$ROLE"
  if ! gpu="$(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader,nounits 2>&1 | sed -n 1p)" \
      || [[ -z "$gpu" || "$gpu" == *"failed"* || "$gpu" == *"No devices"* ]]; then
    pf FAIL gpu "nvidia-smi: ${gpu:-no output}"
  else
    IFS=',' read -r name drv vram used <<< "$gpu"
    name="${name# }"; drv="${drv// /}"; vram="${vram// /}"; used="${used// /}"; major="${drv%%.*}"
    pf PASS gpu "$name, $((vram / 1024)) GB VRAM ($(nvidia-smi -L 2>/dev/null | wc -l) GPU)"
    echo "@ gpu $name"; echo "@ vram_mib $vram"; echo "@ driver $drv"
    if [[ " $DRIVER_OK " == *" $major "* ]]; then
      [[ "$ROLE" == auto ]] && eff=full
      pf PASS driver "$drv (Isaac Sim 5.1 RTX tested on 580)$( [[ "$ROLE" == auto ]] && echo ": role auto -> full")"
    elif [[ "$ROLE" == full ]]; then
      pf FAIL driver "$drv: Isaac Sim 5.1 RTX needs driver ${DRIVER_OK// / or }; 595 segfaults in librtx.scenedb at boot (compute3). Ask for 580, or --role physics / auto"
    else
      [[ "$ROLE" == auto ]] && eff=physics
      pf WARN driver "$drv: not ${DRIVER_OK// / or }, so no 5.1 rendering$( [[ "$ROLE" == auto ]] && echo ": role auto -> physics") (5.1 physics-only + Isaac Sim 6.0 render, like compute3)"
    fi
    if [[ "$vram" -lt 12000 ]]; then pf FAIL vram "${vram} MiB < 12 GB"
    elif [[ "$vram" -lt 20000 ]]; then pf WARN vram "${vram} MiB < 20 GB (bundle = episode env + twin + render)"
    fi
    local apps; apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | grep -c . || true)"
    if [[ "$used" -gt 2000 || "$apps" -gt 0 ]]; then pf WARN gpu_busy "${used} MiB used, ${apps} compute process(es): someone else is on this GPU"
    else pf PASS gpu_busy "${used} MiB used, no compute processes"; fi
  fi
  # disk: the install root ($HOME, or --prefix) + room for caches in $HOME (uv, Kit shader / extension caches)
  local root="${PREFIX:-$HOME}" avail
  if [[ -n "$PREFIX" && ! ( -d "$PREFIX" && -w "$PREFIX" ) ]]; then pf FAIL disk "--prefix $PREFIX is not a writable directory"
  else
    avail="$(df -BG --output=avail "$root" 2>/dev/null | tail -1 | tr -dc 0-9)"
    echo "@ disk_gb ${avail:-0}"
    if [[ -z "$avail" ]]; then pf FAIL disk "df failed on $root"
    elif [[ "$avail" -lt "$MIN_DISK_GB" ]]; then pf FAIL disk "${avail} GB free on $(df --output=target "$root" | tail -1) < ${MIN_DISK_GB} GB$( [[ -z "$PREFIX" ]] && echo " (a bigger mount? --prefix DIR)")"
    else pf PASS disk "${avail} GB free on $(df --output=target "$root" | tail -1)"; fi
    if [[ -n "$PREFIX" ]]; then
      avail="$(df -BG --output=avail "$HOME" 2>/dev/null | tail -1 | tr -dc 0-9)"
      [[ "${avail:-0}" -lt 30 ]] && pf WARN home_disk "${avail} GB free in $HOME < 30 GB (venv, uv / Kit caches stay there)"
    fi
  fi
  # RAM / CPU
  local mem_t mem_a cpus model
  mem_t=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1048576 ))
  mem_a=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1048576 ))
  echo "@ ram_gb $mem_t"
  if [[ "$mem_t" -lt 30 ]]; then pf FAIL ram "${mem_t} GB < 32 GB"
  elif [[ "$mem_a" -lt 25 || "$mem_t" -lt 60 ]]; then pf WARN ram "${mem_t} GB total, ${mem_a} GB available (bundle scopes want 24 GB, jobs start at >= 25 GB available)"
  else pf PASS ram "${mem_t} GB total, ${mem_a} GB available"; fi
  cpus="$(nproc)"; model="$(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -1)"
  echo "@ cpus $cpus"
  if [[ "$cpus" -lt 12 ]]; then pf WARN cpu "$cpus threads ($model) < 14 (the bundle scope's quota)"
  else pf PASS cpu "$cpus threads ($model)"; fi
  # OS
  local os glibc
  os="$(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown}")"
  glibc="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$')"
  if awk -v g="${glibc:-0}" 'BEGIN{split(g,a,"."); exit !(a[1]*100+a[2] >= 235)}'; then pf PASS os "$os, glibc $glibc, kernel $(uname -r)"
  else pf FAIL os "$os, glibc ${glibc:-?} < 2.35 (Isaac Sim 5.1 wheels)"; fi
  # tailscale
  if command -v tailscale >/dev/null 2>&1; then
    local self; self="$(tailscale status --self --peers=false 2>/dev/null | head -1 | awk '{print $1, $2}')"
    if [[ -n "$self" ]]; then pf PASS tailscale "$self"; else pf WARN tailscale "installed, not logged in / not running"; fi
  else
    pf WARN tailscale "not installed (team access is via the tailnet)"
  fi
  # tools
  local t miss=() opt=()
  for t in git curl tar flock setsid nohup python3 sha256sum gcc g++ make cmake; do command -v "$t" >/dev/null 2>&1 || miss+=("$t"); done
  for t in git-lfs ffmpeg rsync bzip2; do command -v "$t" >/dev/null 2>&1 || opt+=("$t"); done
  if [[ ${#miss[@]} -gt 0 ]]; then pf FAIL tools "missing: ${miss[*]} (IsaacLab / cuRobo build: sudo apt-get install -y build-essential cmake git curl python3)"
  elif [[ ${#opt[@]} -gt 0 ]]; then pf WARN tools "missing: ${opt[*]} (bootstrap falls back: git-lfs/ffmpeg static builds; apt is nicer)"
  else pf PASS tools "git $(git --version | awk '{print $3}'), git-lfs, ffmpeg, rsync, python3 $(python3 -V 2>&1 | awk '{print $2}')"; fi
  local sp; sp="$(python3 -V 2>&1 | awk '{print $2}')"
  [[ "$sp" == *rc* ]] && pf WARN python "system python3 $sp is a release candidate (3.11.0rc1 segfaults torch.jit/SAM3): every venv here uses uv-managed CPython"
  # graphics stack: Vulkan ICD (Kit RTX), EGL (perception GL)
  local icd=""
  for f in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json; do [[ -f "$f" ]] && icd="$f"; done
  if [[ -n "$icd" ]]; then pf PASS vulkan "$icd"
  elif [[ "$eff" == full ]]; then pf FAIL vulkan "no NVIDIA Vulkan ICD (container without graphics capability?): Kit RTX will not start"
  else pf WARN vulkan "no NVIDIA Vulkan ICD: the Isaac Sim 6.0 render service (proxy_eval) will not start either; twin / G3 are fine"; fi
  # (no `grep -q` after a pipe here: under pipefail the writer gets SIGPIPE and the check fails)
  if [[ "$(PATH="$PATH:/sbin:/usr/sbin" ldconfig -p 2>/dev/null | grep -c libEGL_nvidia)" -gt 0 ]]; then pf PASS egl "libEGL_nvidia present (perception GL)"
  else pf WARN egl "no libEGL_nvidia: perception falls back to the slow CPU path (no card digits)"; fi
  # kernel limits
  local w shm
  w="$(cat /proc/sys/fs/inotify/max_user_watches 2>/dev/null || echo 0)"
  if [[ "$w" -lt 524288 ]]; then pf WARN inotify "max_user_watches $w < 524288: Kit logs ENOSPC (harmless; sudo sysctl fs.inotify.max_user_watches=524288)"
  else pf PASS inotify "max_user_watches $w"; fi
  shm="$(df -BG --output=size /dev/shm 2>/dev/null | tail -1 | tr -dc 0-9)"
  [[ -n "$shm" && "$shm" -lt 8 ]] && pf WARN shm "/dev/shm ${shm} GB < 8 GB (docker default?)"
  if XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}" systemctl --user show-environment >/dev/null 2>&1; then pf PASS systemd "user manager up (systemd-run --user scopes for proxy_eval / integ_eval)"
  else pf WARN systemd "no systemd user manager: proxy_eval.sh / integ_eval.sh scopes will not start (loginctl enable-linger)"; fi
  # network: the install pulls from these
  local u code bad=()
  [[ "${PREFLIGHT_NET:-1}" == 0 ]] && { pf INFO network "not checked (PREFLIGHT_NET=0)"; }
  [[ "${PREFLIGHT_NET:-1}" == 0 ]] || for u in https://huggingface.co https://pypi.nvidia.com https://download.pytorch.org/whl/cu128 https://github.com https://repo.anaconda.com https://astral.sh; do
    code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 10 "$u" 2>/dev/null || true)"
    [[ "$code" =~ ^[23] ]] || bad+=("${u#https://}=${code:-timeout}")
  done
  if [[ ${#bad[@]} -gt 0 ]]; then pf FAIL network "unreachable: ${bad[*]}"
  elif [[ "${PREFLIGHT_NET:-1}" != 0 ]]; then pf PASS network "huggingface, pypi.nvidia, pytorch, github, anaconda, astral"; fi
  # resumable state
  [[ -d "$RD_HOME" || -d "$BOX" ]] && pf INFO resume "existing $( [[ -d "$RD_HOME" ]] && echo "$RD_HOME ") $( [[ -d "$BOX" ]] && echo "$BOX") (steps skip what is done)"
  [[ "$eff" == auto ]] && eff=physics          # no GPU / driver read: never claim 5.1 rendering
  echo "@ role $eff"
  echo
  if [[ "$PF_FAIL" -eq 0 ]]; then echo "GO ($PF_WARN warnings)"; return 0; fi
  echo "NO-GO ($PF_FAIL failed, $PF_WARN warnings)"; return 1
}

# --------------------------------------------------------------------------------------------- steps
step_conventions() {   # dojo/README.md machine conventions: dirs, GPU locks, env file, scopes, uv policy, ~/dojo-box.json
  if [[ -n "$PREFIX" ]]; then   # the big directories live on --prefix, under the usual names in $HOME
    local d
    for d in dojo-robodojo dojo-data dojo-perception-data dojo-isaac6; do
      x mkdir -p "$PREFIX/$d"
      if is test -L "$HOME/$d" && [[ "$(readlink -f "$HOME/$d")" == "$(readlink -f "$PREFIX/$d")" ]]; then continue; fi
      is test -e "$HOME/$d" && die "$HOME/$d exists and is not a link to $PREFIX/$d: move it there first"
      x ln -s "$PREFIX/$d" "$HOME/$d"
    done
  fi
  ensure_uv
  x mkdir -p "$DATA"/{simlab,perception,skills,integ,agent,bridge,infra} "$PERC" "$BOX/state" "$BOX/logs"
  # the team's GPU locks: ONE heavy job (Isaac with rendering, a bundle, a model) holds ~/dojo-gpu.lock; up to four
  # physics-only light jobs (g3 light lane, twin probes) hold one ~/dojo-gpu-light*.lock slot each
  x touch "$HOME/dojo-gpu.lock" "${LIGHT_LOCKS[@]/#/$HOME/}" "$DATA/simlab/.queue.lock"
  # uv must never pick a system interpreter (Ubuntu 22.04's python3.11 is 3.11.0rc1: SAM3 / torch.jit segfault)
  if [[ "$SHARED_HOME" == 1 ]]; then
    log "shared home: ~/.config/uv/uv.toml left alone (UV_PYTHON_PREFERENCE=only-managed comes from ~/dojo-env.sh and every step)"
  elif ! is grep -qs 'python-preference' "$HOME/.config/uv/uv.toml"; then
    x mkdir -p "$HOME/.config/uv"
    xs "printf 'python-preference = \"only-managed\"\\n' >> $HOME/.config/uv/uv.toml"
  fi
  dnote "as root only: /root/controlr-robodojo -> dojo-robodojo (the compute2 default RD_HOME)"
  if [[ "$HOME" == /root || "$DRY" == 1 ]] && ! is test -e /root/controlr-robodojo; then
    x ln -s dojo-robodojo /root/controlr-robodojo       # the compute2 default RD_HOME of rdpy.sh / dev_eval.sh / paths.py
  fi
  # systemd-run --user scopes outlive the ssh session only with lingering (proxy_eval.sh / integ_eval.sh / g3.sh need them)
  if [[ "$SHARED_HOME" == 1 ]]; then
    dnote "shared home: no loginctl (a system setting of someone else's machine)"
    is test -e "/var/lib/systemd/linger/$(id -un)" \
      || log "shared home: lingering stays off: a --user scope lives while the script that started it does (integ_eval.sh / proxy_eval.sh wait for theirs)"
  elif ! is test -e "/var/lib/systemd/linger/$(id -un)"; then
    dnote "only when lingering is off; no sudo: works where polkit allows it, else a WARN (loginctl enable-linger as admin)"
    xs "loginctl enable-linger $(id -un) 2>/dev/null || echo 'WARN: loginctl enable-linger failed: scopes need an open session (ask an admin)'"
  fi
  local envf="$HOME/dojo-env.sh" i6line
  if [[ "$ROLE" == physics ]]; then
    i6line="export ISAAC_SIM_ROOT=\"$ISAAC6\"    # Isaac Sim 6.0 (compute3's build): scripts/simlab/isaac6.sh, proxy_eval.sh's render service"
  else
    i6line="# ISAAC_SIM_ROOT: no Isaac Sim 6.0 here (role full renders with 5.1; dev_eval.sh is the reference)"
  fi
  if [[ "$DRY" == 1 ]]; then echo "    + cat > $envf   # DOJO_BOX_ROLE RD_HOME RD_ROOT DOJO_DATA DOJO_GPU_LOCK DOJO_GPU_LIGHT_LOCKS DOJO_SCOPE_* ISAAC_SIM_ROOT DOJO_CATALOG ..."
  else
    cat > "$envf" <<EOF
# dojo GPU box environment (scripts/bootstrap_gpu_box.sh, role $ROLE). Source it before running repo scripts here:
#   source ~/dojo-env.sh && cd ~/dojo-<stream> && scripts/simlab/rdpy.sh -m ...
export DOJO_BOX_ROLE="$ROLE"              # full: Isaac Sim 5.1 renders; physics: 5.1 physics-only + Isaac Sim 6.0 render (~/dojo-box.json)
export RD_HOME="$RD_HOME"                 # RoboDojo + Isaac Sim 5.1 (read-only for streams)
export RD_ROOT="$SRC"
export DOJO_DATA="$DATA"                  # data: \$DOJO_DATA/<stream>/
export DOJO_PERCEPTION_DATA="$PERC"
export DOJO_CATALOG="$PERC/catalog"       # perception catalogue (skills also: \$DOJO_DATA/skills2/catalog, skills3/catalog)
export DOJO_X5_DIR="$PERC/catalog/robot/x5"
$i6line
# GPU locks (flock): ONE heavy GPU job per machine holds DOJO_GPU_LOCK (rdpy.sh / isaac6.sh / dev_eval.sh /
# proxy_eval.sh / integ_eval.sh / bridge_episode.sh take it); <= 30 min per hold, 90 s yield before re-taking.
# Light lane: up to 4 physics-only jobs, one slot each: DOJO_GPU_LOCKS="\$DOJO_GPU_LIGHT_LOCKS" scripts/skills/g3.sh ...
export DOJO_GPU_LOCK="$HOME/dojo-gpu.lock"
export DOJO_GPU_LIGHT_LOCKS="${LIGHT_LOCKS[*]/#/$HOME/}"
# systemd-run --user --scope -q \$DOJO_SCOPE_BUNDLE <cmd>   (start a heavy job only with >= \$DOJO_MIN_AVAIL_GB GB MemAvailable)
export DOJO_SCOPE_BUNDLE="$SCOPE_BUNDLE"
export DOJO_SCOPE_PROXY="$SCOPE_PROXY"
export DOJO_SCOPE_BATCH="$SCOPE_BATCH"
export DOJO_SCOPE_G3="$SCOPE_G3"
export DOJO_SCOPE_LIGHT="$SCOPE_LIGHT"
export DOJO_MIN_AVAIL_GB=$MIN_AVAIL_GB
# genuine Isaac 5.1 calibration (5.1-rendering box), per run:
#   DOJO_D0=\$DOJO_DATA/d0_51 DOJO_BG_FOLDS=\$DOJO_PERCEPTION_DATA/bg_folds_51 DOJO_COLOR_CALIB=\$DOJO_CATALOG/color_calib_51.json
export UV_PYTHON_PREFERENCE=only-managed
export OMNI_KIT_ACCEPT_EULA=YES
export PATH="$( [[ "$SHARED_HOME" == 1 ]] && echo "$UV_DIR:")\$HOME/.local/bin:\$PATH"
# systemd-run --user needs the user manager's runtime dir; tailscale ssh and su sessions do not export it
[ -n "\${XDG_RUNTIME_DIR:-}" ] || [ ! -d "/run/user/\$(id -u)" ] || export XDG_RUNTIME_DIR="/run/user/\$(id -u)"
EOF
    log "wrote $envf"
  fi
  local use
  if [[ "$ROLE" == full ]]; then use="Isaac Sim 5.1 renders here: official dev_eval.sh, the bridge (BRIDGE-RUNBOOK), 5.1 frames, twin, G3, perception."
  else use="5.1 physics-only (twin, G3, RoboDojoRunner(physics_only=True)) + Isaac Sim 6.0 render ($ISAAC6): proxy_eval.sh, perception. NO official dev_eval / bridge here."; fi
  if [[ "$DRY" == 1 ]]; then echo "    + cat > ~/DOJO_BOX.md   # the conventions below"
  else
    cat > "$HOME/DOJO_BOX.md" <<EOF
# This box: dojo GPU box (role $ROLE), set up by controlr scripts/bootstrap_gpu_box.sh

- \`source ~/dojo-env.sh\` first (RD_HOME, DOJO_DATA, GPU locks, scope settings, catalogue paths). Machine facts,
  capabilities and smoke results: \`~/dojo-box.json\` (pick boxes by it); install summary: \`$RD_HOME/READY\`.
- Role $ROLE: $use
- RoboDojo (Isaac Sim 5.1 + IsaacLab + cuRobo + XPolicyLab, pinned like compute3): \`$RD_HOME\` -- read-only for streams.
  Run through \`scripts/simlab/rdpy.sh\` (conda python + the GPU lock).
- Code: \`~/controlr\` = main at deploy time (\`.deployed_commit\`; \`.env\` = OMNIROUTE_* only, mode 600); streams deploy to \`~/dojo-<stream>\`.
- Data: \`~/dojo-data/<stream>/\` (dev layouts \`simlab/layouts\`, D0 \`d0\` / \`d0_51\`), perception: \`~/dojo-perception-data\`.
- GPU locks: ONE heavy job per machine holds \`flock ~/dojo-gpu.lock\` (rdpy.sh, isaac6.sh, dev_eval.sh, proxy_eval.sh,
  integ_eval.sh, bridge_episode.sh do it; the file holds "pid stream start cmd"). Keep a hold <= 30 min, yield 90 s.
  Light lane: up to 4 physics-only jobs, one of \`~/dojo-gpu-light{,2,3,4}.lock\` each (\`DOJO_GPU_LOCKS="\$DOJO_GPU_LIGHT_LOCKS"\`).
- Check \`nvidia-smi\` before starting Isaac or a model; never touch other people's processes (kill by PID, never pkill -f
  over ssh); stop what you started.
- Big jobs in a scope: \`systemd-run --user --scope -q \$DOJO_SCOPE_BUNDLE ...\` (24G / 1400%; proxy 16G / 1200%; batch 20G /
  1200%; g3 10G / 600%; light 8G / 600%); start only with >= $MIN_AVAIL_GB GB MemAvailable.
- Eval data firewall: never read \`Assets/Eval_Layout/**\` or eval-paired \`Assets/Traj\` (they are not even downloaded here).
$( [[ "$SHARED_HOME" == 1 ]] && echo "- SHARED HOME: other people use this \$HOME and machine too. Write only under ~/dojo-*, ~/controlr*; never touch
  other dirs, processes or sessions; no apt / sudo / loginctl / system settings. Our pinned uv: \`$UV_DIR\`.
- NO LINGERING here: a \`systemd-run --user\` scope dies with the last session unless its launcher stays alive. Start
  detached jobs as \`setsid nohup bash -c 'source ~/dojo-env.sh; <cmd>; echo rc=\$?' > LOG 2>&1 < /dev/null &\` (the bash
  stays); g3.sh execs into its scope, so give it \`G3_ABORT_MEM_GB=8\` (it then waits as the scope's parent).")
EOF
  fi
  box_json || log "WARN: ~/dojo-box.json not written now (the ready step writes it again)"
}

box_json() {   # ~/dojo-box.json: what agents read to pick a box (role, specs, capabilities from the smokes, locks, scopes)
  if [[ "$DRY" == 1 ]]; then echo "    + python3 $LIBX/box_json.py > ~/dojo-box.json   # schema dojo-box/1: role, GPU, driver, CPU, RAM, disk, capabilities, locks, scopes, smokes"; return 0; fi
  local scopes=0
  XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}" systemd-run --user --scope -q true >/dev/null 2>&1 && scopes=1
  BOXJ_ROLE="$ROLE" BOXJ_REASON="$ROLE_REASON" BOXJ_ALIAS="$BOX_ALIAS" BOXJ_SCOPES="$scopes" BOXJ_BOX="$BOX" \
  BOXJ_RD="$RD_HOME" BOXJ_CODE="$CODE" BOXJ_DATA="$DATA" BOXJ_PERC="$PERC" BOXJ_I6="$ISAAC6" BOXJ_I6V="$ISAAC6_VERSION" \
  BOXJ_LOCKS="${LIGHT_LOCKS[*]}" BOXJ_SB="$SCOPE_BUNDLE" BOXJ_SP="$SCOPE_PROXY" BOXJ_SBA="$SCOPE_BATCH" BOXJ_SG="$SCOPE_G3" \
  BOXJ_SL="$SCOPE_LIGHT" BOXJ_MIN="$MIN_AVAIL_GB" BOXJ_SHA="$CODE_SHA" BOXJ_RDC="$(rd_commit)" BOXJ_HF="$HF_REV" \
  BOXJ_ASSETS="$ASSETS" BOXJ_ROOT="${PREFIX:-$HOME}" python3 "$LIB/box_json.py" > "$HOME/dojo-box.json.new" \
    && mv "$HOME/dojo-box.json.new" "$HOME/dojo-box.json" && log "wrote ~/dojo-box.json"
}

step_code() {   # the driver extracted \`git archive <ref>\` into $BOX/code.new; install it as ~/controlr
  is test -d "$BOX/code.new" || [[ "$DRY" == 1 ]] || die "no $BOX/code.new (the driver streams it)"
  x mkdir -p "$CODE"
  if [[ "$DRY" == 1 ]] || command -v rsync >/dev/null 2>&1; then
    dnote "without rsync: cp -a (stale files stay)"
    x rsync -a --delete --exclude .venv --exclude .env --exclude runs --exclude .deployed_commit "$BOX/code.new/" "$CODE/"
  else
    xs "cp -a $BOX/code.new/. $CODE/"
  fi
  xs "echo $CODE_SHA > $CODE/.deployed_commit"
  x rm -rf "$BOX/code.new"
}

ensure_uv() {   # uv $UV_VERSION in $UV_DIR (one installer at a time: venv and robodojo run in parallel)
  if is test -x "$UV_DIR/uv" && [[ "$("$UV_DIR/uv" --version 2>/dev/null | awk '{print $2}')" == "$UV_VERSION" ]]; then return 0; fi
  dnote "only if $UV_DIR/uv is not $UV_VERSION"
  if [[ "$SHARED_HOME" == 1 ]]; then   # the home's own ~/.local/bin/uv (and its receipt, shell profiles) stay untouched
    xs "mkdir -p $BOX/state $UV_DIR && flock $BOX/state/.uv.lock sh -c 'curl -LsSf https://astral.sh/uv/$UV_VERSION/install.sh | env UV_UNMANAGED_INSTALL=$UV_DIR sh'"
  else
    xs "mkdir -p $BOX/state && flock $BOX/state/.uv.lock sh -c 'curl -LsSf https://astral.sh/uv/$UV_VERSION/install.sh | env UV_NO_MODIFY_PATH=1 sh'"
  fi
}

make_venv() {   # CHECKOUT: CHECKOUT/.venv = compute3's ~/dojo-perception/.venv (perception extras), on uv-managed CPython 3.12
  local code="$1" venv="$1/.venv"
  ensure_uv
  if is test -x "$venv/bin/python" && "$venv/bin/python" -c 'import sys; assert "/uv/python/" in sys.base_prefix' 2>/dev/null; then
    log "venv exists ($("$venv/bin/python" -V)) on managed CPython"
  else
    dnote "only when the venv is missing or not on uv-managed CPython"
    x rm -rf "$venv"
    x uv venv -q --python 3.12 --python-preference only-managed "$venv"
  fi
  # pyrender pins PyOpenGL 3.1.0, whose texture upload fails under 3.12: the freeze minus pyopengl, then 3.1.9 over it
  x mkdir -p "$BOX/state"
  xs "grep -v -i '^pyopengl==' $LIBX/pins/controlr-venv.txt > $BOX/state/venv-reqs.txt"
  x uv pip install -q --python "$venv/bin/python" -r "$BOX/state/venv-reqs.txt"
  x uv pip install -q --python "$venv/bin/python" --no-deps "$(grep -i '^pyopengl==' "$PINS/controlr-venv.txt")"
  x uv pip install -q --python "$venv/bin/python" --no-deps -e "$code"
  xs "cd $code && $venv/bin/python -c 'import sys, OpenGL, pyrender, trimesh, cv2, pxr, dojo.perception.render, controlr; assert OpenGL.__version__ == \"3.1.9\", OpenGL.__version__; assert \"rc\" not in sys.version.split()[0]; print(\"venv ok\", sys.version.split()[0], \"pyopengl\", OpenGL.__version__)'"
}

step_venv() { make_venv "$CODE"; }

ensure_git_lfs() {
  if have git-lfs || is git lfs version >/dev/null 2>&1; then return 0; fi
  local d="$BOX/tools/git-lfs"
  dnote "only without a system git-lfs: the static release binary"
  if ! is test -x "$d/git-lfs"; then
    x mkdir -p "$d"
    xs "curl -LsSf https://github.com/git-lfs/git-lfs/releases/download/v$GIT_LFS_VERSION/git-lfs-linux-amd64-v$GIT_LFS_VERSION.tar.gz | tar -xz -C $d --strip-components=1"
  fi
  export PATH="$d:$PATH"
}

rd_clone_pinned() {   # RoboDojo + its 3 submodules at exactly compute3's commits (robodojo_commit.txt)
  local rc; rc="$(rd_commit)"
  if ! is test -d "$SRC/.git"; then
    x git init -q "$SRC"
    x git -C "$SRC" remote add origin https://github.com/RoboDojo-Benchmark/RoboDojo.git
  fi
  if ! is test "$(git -C "$SRC" rev-parse HEAD 2>/dev/null)" = "$rc"; then
    x git -C "$SRC" fetch -q --depth 1 origin "$rc"
    x git -C "$SRC" checkout -q --detach "$rc"
  fi
  local sub sha
  for sub in third_party/IsaacLab third_party/curobo XPolicyLab; do
    sha="$(pin_sha "$sub")"
    [[ -n "$sha" ]] || die "no pin for $sub in robodojo_commit.txt"
    if is test "$(git -C "$SRC/$sub" rev-parse HEAD 2>/dev/null)" = "$sha"; then continue; fi
    dnote "$sub -> $sha (compute3's checkout)"
    is test -e "$SRC/$sub/.git" || x git -C "$SRC" submodule update --init --depth 1 "$sub" \
      || x git -C "$SRC" submodule update --init "$sub"
    x git -C "$SRC/$sub" fetch -q --depth 1 origin "$sha"
    x git -C "$SRC/$sub" checkout -q --detach "$sha"
  done
  xs "{ git -C $SRC rev-parse HEAD; git -C $SRC submodule status; } > $RD_HOME/robodojo_commit.txt"
  if [[ "$DRY" != 1 ]]; then        # SHAs must equal compute3's (the describe suffix may differ)
    diff <(awk '{s=$1; sub(/^[+ -]/,"",s); print s, $2}' "$RD_HOME/robodojo_commit.txt") \
         <(awk '{s=$1; sub(/^[+ -]/,"",s); print s, $2}' "$PINS/robodojo_commit.txt") \
      || die "RoboDojo checkout differs from the compute3 pin"
    log "RoboDojo $(cut -c1-8 <<< "$rc") + submodules = compute3 pin"
  fi
}

step_robodojo() {   # Isaac Sim 5.1 + IsaacLab + cuRobo + XPolicyLab in $RD_HOME (scripts/robodojo/install.sh, RoboDojo's own pins)
  ensure_uv
  local inst="$CODE/scripts/robodojo/install.sh"
  is test -f "$inst" || [[ "$DRY" == 1 ]] || die "$inst missing: run the code step first"
  x env RD_HOME="$RD_HOME" bash "$inst" conda
  rd_clone_pinned
  # each of these skips what is already installed (pip show / import checks inside RoboDojo's install functions)
  x env RD_HOME="$RD_HOME" bash "$inst" deps isaacsim isaaclab curobo policy_env
  if ! have ffmpeg && ! is test -x "$ENVP/bin/ffmpeg"; then   # RoboDojo's eval client pipes mp4s through `ffmpeg` on PATH
    dnote "only without a system ffmpeg: imageio-ffmpeg's static binary into the env"
    x "$RDPY" -m pip install -q imageio-ffmpeg==0.6.0
    xs "ln -sfn \"\$($RDPY -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')\" $ENVP/bin/ffmpeg"
  fi
  x env RD_HOME="$RD_HOME" bash "$inst" check
  xs "$RD_HOME/policy_venv/bin/python -c 'import sys; v = sys.version.split()[0]; assert \"rc\" not in v, v; print(\"policy_venv\", v)'"
  # drift report against compute3's env (not enforced: RoboDojo's install pins the stack that matters)
  if [[ "$DRY" == 1 ]]; then echo "    + diff <($RDPY -m pip freeze) $LIBX/pins/robodojo-env.freeze.txt   # drift report vs compute3"
  else
    local d; d="$(diff <("$RDPY" -m pip freeze --disable-pip-version-check 2>/dev/null | grep -v '^-e ' | sort) \
                       <(grep -v '^#' "$PINS/robodojo-env.freeze.txt" | sort) | grep '^[<>]' || true)"
    if [[ -z "$d" ]]; then log "env freeze == compute3 (299 packages)"
    else log "env freeze vs compute3: $(grep -c '^<' <<< "$d") here / $(grep -c '^>' <<< "$d") compute3 lines differ:"; echo "$d" | head -40; fi
    "$RDPY" -V
  fi
}

lfs_includes() {   # sparse lines -> git-lfs --include patterns (folders get **)
  grep -v '^\s*\(#\|$\)' "$1" | sed -e 's#^/##' -e 's#/$#/**#' | paste -sd, -
}

step_assets() {   # RoboDojo HF Assets at the pinned revision: sparse git-lfs (batch API; per-file hf download hits HF's rate limit)
  ensure_git_lfs
  local list="$BOX/state/assets_sparse.txt" applied="$HFG/.dojo_applied_sparse.txt" new="$BOX/state/assets_new.txt"
  x mkdir -p "$BOX/state"
  if [[ "$ASSETS" == all ]]; then
    xs "printf '%s\n' /Assets/ '!/Assets/Eval_Layout/' '!/Assets/Traj/RoboDojo/imitate_sorting_sequence/' > $list"
  else
    xs "grep -v '^\s*\(#\|\$\)' ${ASSETS_PINS:-$LIBX/pins/robodojo_assets.txt} > $list"
  fi
  # a tree copied from a fleet peer or compute3 (bootstrap_gpu_box.sh fetch): no applied list yet (compute3's own
  # checkout), so take its sparse list minus eval data; the pointer check below proves the content is there
  if is test -d "$HFG/Assets" && is test -f "$HFG/.git/info/sparse-checkout" && ! is test -f "$applied" \
      && [[ "$(git -C "$HFG" rev-parse HEAD 2>/dev/null)" == "$HF_REV" ]]; then
    grep -v -e Eval_Layout -e imitate_sorting_sequence "$HFG/.git/info/sparse-checkout" > "$applied"
    log "assets: existing checkout at the pin (copied): its sparse list counts as applied ($(grep -c . "$applied") paths)"
  fi
  local missing=1
  if is test -f "$applied" && [[ "$(git -C "$HFG" rev-parse HEAD 2>/dev/null)" == "$HF_REV" ]] \
      && [[ -z "$(comm -13 <(sort "$applied") <(sort "$list"))" ]]; then
    missing=0; log "assets: every pinned path is applied at $HF_REV (fetched before or copied from a peer): no git / HF calls"
  fi
  dnote "the git / git-lfs part below runs only when some pinned path is not applied yet at $HF_REV"
  if [[ "$missing" == 1 ]] && ! is test -d "$HFG/.git"; then
    x env GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 --no-checkout "$HF_URL" "$HFG"
    x git -C "$HFG" lfs install --local
  fi
  [[ "$missing" == 1 ]] && x git -C "$HFG" sparse-checkout init --no-cone
  # (a copied index carries the peer's stat data: every git command below re-hashes through the LFS clean filter once)
  x cp "$list" "$HFG/.git/info/sparse-checkout"
  if [[ "$missing" == 0 ]]; then :
  elif ! is test "$(git -C "$HFG" rev-parse HEAD 2>/dev/null)" = "$HF_REV" || ! is test -d "$HFG/Assets"; then
    x env GIT_LFS_SKIP_SMUDGE=1 git -C "$HFG" checkout -q main
    if [[ "$DRY" != 1 && "$(git -C "$HFG" rev-parse HEAD 2>/dev/null)" != "$HF_REV" ]]; then
      log "HF main moved past the pin $HF_REV: fetching the pinned revision"
      x git -C "$HFG" fetch -q --depth 1 origin "$HF_REV"
      x env GIT_LFS_SKIP_SMUDGE=1 git -C "$HFG" checkout -q --detach "$HF_REV"
    fi
  fi
  [[ "$missing" == 1 ]] && x env GIT_LFS_SKIP_SMUDGE=1 git -C "$HFG" sparse-checkout reapply
  # only patterns not pulled yet (LFS objects are dropped after each pull, so a full re-pull would re-download)
  if is test -f "$applied"; then xs "comm -13 <(sort $applied) <(sort $list) > $new"; else x cp "$list" "$new"; fi
  if is test ! -s "$new"; then log "assets: nothing new to pull"
  else
    local inc ex=()
    if [[ "$ASSETS" == all ]]; then inc="Assets/**"; ex=(--exclude "Assets/Eval_Layout/**,Assets/Traj/RoboDojo/imitate_sorting_sequence/**")
    elif [[ "$DRY" == 1 ]]; then inc="<$(grep -vc '^\s*\(#\|$\)' "$ASSETS_LIST") patterns from $(basename "$ASSETS_LIST")>"
    else inc="$(lfs_includes "$new")"; fi
    local i ok=0
    for i in $(seq 1 20); do
      log "assets: git lfs pull, attempt $i ($ASSETS)"
      if x git -C "$HFG" lfs pull --include "$inc" "${ex[@]}"; then ok=1; break; fi
      [[ "$DRY" == 1 ]] && { ok=1; break; }
      sleep 120                        # HF rate limit / network: back off, resume (fetched objects are kept until the end)
    done
    [[ "$ok" == 1 ]] || die "git lfs pull failed 20 times"
  fi
  # every checked-out LFS file must be real content now
  if [[ "$DRY" != 1 ]]; then
    # (pipefail off: grep -l exits 1 on the files that are NOT pointers, which is the good case)
    local ptr; ptr="$(set +o pipefail; find "$HFG/Assets" -type f -size -400c -print0 | xargs -0 -r grep -l -m1 '^version https://git-lfs.github.com/spec' 2>/dev/null | wc -l)"
    if [[ "$ptr" -ne 0 ]]; then   # the applied list lied (an interrupted pull, a partial copy): the re-run pulls every pattern
      rm -f "$applied"; die "$ptr LFS pointer files left (download incomplete): re-run the assets step"
    fi
    [[ ! -e "$HFG/Assets/Eval_Layout" && ! -e "$HFG/Assets/Traj/RoboDojo/imitate_sorting_sequence" ]] \
      || die "eval data in the checkout: check $list"
  fi
  x rm -rf "$HFG/.git/lfs/objects"   # stored once (the working tree), like compute3
  xs "{ cat $list; cat $applied 2>/dev/null || true; } | sort -u > $applied.new && mv $applied.new $applied"   # what a peer copy may carry beyond our list stays known
  xs "du -sh $HFG/Assets"
}

step_assets_post() {   # Assets into the RoboDojo tree + cuRobo configs with absolute paths + check_assets
  x ln -sfn "$HFG/Assets" "$SRC/Assets"
  # Assets/Robots/**/curobo.yml are generated from *_tmp.yml with absolute paths baked in (RoboDojo dies at robot setup without them)
  xs "cd $SRC && $RDPY utils/update_embodiment_config_path.py < /dev/null"
  if [[ "$DRY" != 1 ]]; then
    local n; n="$(set +o pipefail; find -L "$SRC/Assets/Robots" -name curobo.yml 2>/dev/null | wc -l)"
    [[ "$n" -gt 0 ]] || die "no curobo.yml generated under Assets/Robots"
    log "curobo.yml: $n"
  fi
  # do the Assets hold every object the 54 task configs can ask for? (known upstream gaps: broom_shovel 1/4, hammer 7 place)
  logrun "$BOX/logs/check_assets.out" 'task-object instances' \
    "cd $CODE && SIMLAB_GPU=0 timeout 900 $(rdpy) -m dojo.simlab.check_assets --json $BOX/state/check_assets.json"
}

step_catalog_post() {   # shape caches (sil.npz + shape_cache.json) for catalogue assets that lack them
  local cat="$PERC/catalog" keys="$BOX/state/precompute_keys.json"
  is test -f "$cat/index.json" || [[ "$DRY" == 1 ]] || die "no catalogue at $cat (the driver relays it from compute3)"
  xs "$VENV/bin/python -c 'import json, pathlib, sys; c = pathlib.Path(sys.argv[1]); k = [\"/\".join(p.parts[-4:-1]) for p in sorted(c.glob(\"*/*/*/geom.npz\")) if not (p.parent / \"shape_cache.json\").exists()]; json.dump(k, open(sys.argv[2], \"w\")); print(len(k), \"assets without shape caches\")' $cat $keys"
  if [[ "$DRY" == 1 ]] || [[ "$(cat "$keys")" != "[]" ]]; then
    xs "cd $CODE && DOJO_CATALOG=$cat $VENV/bin/python -m dojo.perception.catalog precompute --out $cat --keys-json $keys"
  fi
  if [[ "$DRY" != 1 ]]; then
    log "catalogue: $(find "$cat" -name info.json | wc -l) assets, $(find "$cat" -name shape_cache.json | wc -l) shape caches, $(du -sh "$cat" | cut -f1)"
  fi
}

tree_md5() {   # DIR: md5 of its manifest listing (what pins/calib_sets.txt pins)
  (cd "$1" && find . \( -type f -o -type l \) -printf '%P\0' | LC_ALL=C sort -z | xargs -0 -r md5sum) | md5sum | cut -c1-32
}

step_calib_sets() {   # the frozen perception calibration sets (pins/calib_sets.txt), identical on every box and root
  # legality audit 2026-10-28 P1-1: the perception service uses ONE frozen set for every episode (DOJO_CALIB_SET); a
  # set that is missing or differs on one box (compute4's silent missing scene background, 2026-10-28) must fail here
  local dir name want src got root n=0
  while read -r dir name want; do
    [[ -z "$dir" || "$dir" == \#* ]] && continue
    src="$PERC/$dir/$name"
    if [[ "$DRY" == 1 ]]; then
      echo "    + verify $src against $want, copy it into ~/dojo-data/${dir}_ilia ${CALIB_EXTRA_ROOTS:-}"; continue
    fi
    [[ -f "$src/manifest.json" ]] || die "no calibration set $src (the catalog relay brings $dir from the source box)"
    got="$(tree_md5 "$src")"
    [[ "$got" == "$want" ]] || die "$src: md5 $got != pinned $want (pins/calib_sets.txt): a different set"
    for root in "$DATA/${dir}_ilia" ${CALIB_EXTRA_ROOTS:-}; do
      [[ -d "$root" ]] || continue
      if [[ ! -d "$root/$name" || "$(tree_md5 "$root/$name")" != "$want" ]]; then
        rm -rf "$root/$name.new"
        cp -a "$src" "$root/$name.new"
        [[ "$(tree_md5 "$root/$name.new")" == "$want" ]] || die "$root/$name.new: copy differs from $src"
        rm -rf "$root/$name"; mv "$root/$name.new" "$root/$name"
        log "installed $name into $root"
      fi
      n=$((n + 1))
    done
    log "$src: md5 $got ok"
  done < "$CALIB_PINS"
  note "calibration sets ok ($n extra roots)"
}

rdpy() { echo "env RD_HOME=$RD_HOME DOJO_DATA=$DATA nice -n 5 scripts/simlab/rdpy.sh"; }
logrun() {   # LOG PATTERN CMD: run CMD (already under `timeout`) into LOG; show matching lines; keep CMD's rc
  xs "$3 > $1 2>&1; rc=\$?; grep -a -E '$2' $1 | tail -20; [ \$rc = 0 ] || tail -5 $1; exit \$rc"
}
check_py() {   # ARGS < python-script: a result check in the controlr venv; its RESULT line becomes state/STEP.result
  local out rc
  out="$("$VENV/bin/python" - "$@")" && rc=0 || rc=$?        # (no set -e exit before the output is logged)
  echo "$out"
  note "$(grep -a '^RESULT' <<< "$out" | tail -1 | sed 's/^RESULT [a-z_]*: //')$( [[ $rc != 0 ]] && echo ' [FAILED]')"
  return "$rc"
}

step_smoke_twin() {   # physics-only Twin: boot, load (+ settle check), 40-step rollout, snapshot/restore + 40
  # Layouts 100000 / 100001 (D-C10a: >= 100100 is reserved for joint runs, never generated, read or run). The G2
  # contact-restore check replays layout 110007, so it runs only with SMOKE_RESERVED_OK=1 (the joint-run owners' OK).
  local out="$SMOKE/twin" g2=""
  [[ "${SMOKE_RESERVED_OK:-0}" == 1 ]] && g2=" --g2-dir $DATA/simlab/g2_c3b"
  dnote "contact-restore check (--g2-dir, layout 110007) only with SMOKE_RESERVED_OK=1"
  x rm -rf "$out"; x mkdir -p "$out"
  logrun "$out/run.log" '^\[twin\]|Error|error:' \
    "cd $CODE && timeout 1800 $(rdpy) -m dojo.simlab.twin --task stack_blocks --num-envs 1 --steps 40 --physics-only --seeds 100000,100001 --out $out$g2"
  [[ "$DRY" == 1 ]] && return 0
  check_py "$out/bench_stack_blocks_n1.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
stable = all(r["load_first"]["stable"]) and all(r["load_new_layout"]["stable"])
cr = r.get("contact_restore")
msg = (f"RESULT smoke_twin: layouts {r.get('seeds')}, boot {r['boot_s']:.0f} s, layouts stable {stable}, 40 steps at "
       f"{r['steps']['env_steps_per_s']:.1f} env-steps/s, restore+40 {r['segment']['restore_plus_40_steps_s']:.2f} s, ")
if cr is None:       # its G2 record is on reserved layout 110007 (D-C10a)
    print(msg + "contact restore NOT checked (its G2 record is on reserved layout 110007)")
    sys.exit(0 if stable and r["steps"]["n"] == 40 else 1)
print(msg + f"contact restore bitwise {cr.get('bitwise')} (max {cr.get('max_obj_divergence_mm', float('nan')):.4f} mm)")
# bitwise on compute3; another box can differ by float noise (compute4: 0.0001 mm). <= 0.01 mm passes with a WARN
exact = cr.get("bitwise") is True
close = cr.get("max_obj_divergence_mm", 1e9) <= 0.01
if close and not exact:
    print("WARN smoke_twin: contact restore not bitwise but within 0.01 mm (float noise); verify is unaffected")
sys.exit(0 if stable and r["steps"]["n"] == 40 and (exact or close) else 1)
PY
}

step_smoke_render() {   # full Isaac Sim 5.1 RTX: one D0 layout, 3 cameras; compared with isr-lab's genuine 5.1 frame 0
  [[ "$ROLE" == full ]] || { log "role $ROLE: skipped (no 5.1 rendering)"; note "skipped (role $ROLE)"; return 0; }
  local out="$SMOKE/render" ref="$DATA/d0_51/stack_blocks/100000"
  x rm -rf "$out"; x mkdir -p "$out"
  logrun "$out/run.log" '^\[simlab\]|Error|Segmentation|error:' \
    "cd $CODE && timeout 2700 $(rdpy) -m dojo.simlab.runner --task stack_blocks --layouts $ref/layout.json --out $out --images"
  [[ "$DRY" == 1 ]] && return 0
  check_py "$out" "$ref" <<'PY'
import sys
from pathlib import Path
import numpy as np
from PIL import Image
out, ref = Path(sys.argv[1]), Path(sys.argv[2])
ok, msg = True, []
for cam in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
    p = out / f"layout_{cam}.png"
    if not p.exists():
        msg.append(f"{cam} missing"); ok = False; continue
    a = np.asarray(Image.open(p).convert("RGB"), np.float32)
    b = np.asarray(Image.open(ref / f"frame_0_{cam}.png").convert("RGB"), np.float32)
    mad = float(np.abs(a - b).mean()) if a.shape == b.shape else float("nan")
    ok = ok and a.shape == (480, 640, 3) and float(a.std()) > 5
    msg.append(f"{cam} std {a.std():.0f}, MAD vs isr-lab {mad:.1f}")
print("RESULT smoke_render:", "; ".join(msg), "(MAD = mean abs diff to isr-lab's genuine 5.1 frame 0, 0-255)")
sys.exit(0 if ok else 1)
PY
}

reserved_gate() {   # the smokes that replay G2 on layout 110007: D-C10a reserves >= 100100 for joint runs
  [[ "${SMOKE_RESERVED_OK:-0}" == 1 ]] && return 1
  dnote "only with SMOKE_RESERVED_OK=1: it replays G2 on layout 110007, inside the reserved >= 100100 range (D-C10a)"
  [[ "$DRY" == 1 ]] && return 1
  log "skipped: it replays G2 on layout 110007, inside the reserved >= 100100 range (D-C10a; SMOKE_RESERVED_OK=1 with the joint-run owners' OK)"
  note "skipped (reserved layout 110007, D-C10a)"
}

step_smoke_dev_eval() {   # the OFFICIAL harness (RoboDojo main.py via dev_eval.sh) on one dev layout, replaying G2's stack trajectory
  [[ "$ROLE" == full ]] || { log "role $ROLE: skipped (dev_eval needs 5.1 rendering)"; note "skipped (role $ROLE)"; return 0; }
  reserved_gate && return 0
  local lay="$SMOKE/dev_layouts" out="$SMOKE/dev_eval" pol="$SRC/XPolicyLab/policy/dojo_replay"
  x rm -rf "$lay" "$out"; x mkdir -p "$lay/stack_blocks" "$out" "$pol"
  x cp "$DATA/simlab/g2_c3b/stack_layout.json" "$lay/stack_blocks/110007.json"      # stack_blocks seed 110007 (G2 'stack': success)
  x cp "$LIBX/xpl_dojo_replay/__init__.py" "$LIBX/xpl_dojo_replay/model.py" "$LIBX/xpl_dojo_replay/deploy.py" "$LIBX/xpl_dojo_replay/deploy.yml" "$pol/"
  logrun "$out/run.log" '^\[dev_eval\]|dojo_replay|Error|error:' \
    "cd $CODE && DOJO_REPLAY_TRAJ=$DATA/simlab/g2_c3b/stack_traj.json RD_HOME=$RD_HOME DOJO_DATA=$DATA timeout 2700 scripts/simlab/dev_eval.sh --task stack_blocks --policy dojo_replay --eval-num 1 --layouts $lay --out $out"
  [[ "$DRY" == 1 ]] && return 0
  local res; res="$(find "$out" -name _result.json 2>/dev/null | head -1 || true)"
  [[ -n "$res" ]] || die "no _result.json under $out (see eval_client.log / policy_server.log there)"
  check_py "$res" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"RESULT smoke_dev_eval: episodes {r.get('eval_time')}, success_rate {r.get('success_rate')}, score {r.get('score')} "
      f"(expected 1 / 1.0: G2 'stack' replays to success on compute3 and isr-lab)")
if r.get("success_rate") != 1.0:
    print("WARN: the harness ran but the replay did not succeed: compare with G2b (contact divergence) before trusting this box")
sys.exit(0 if r.get("eval_time") == 1 else 1)
PY
}

step_smoke_perception() {   # EGL context + inverse layout on one genuine-5.1 D0 frame (push_T 100000; compute3: T 0.5 mm, pad 0.7 mm)
  local out="$SMOKE/perception"
  local env="cd $CODE && PYOPENGL_PLATFORM=egl DOJO_CATALOG=$PERC/catalog DOJO_X5_DIR=$PERC/catalog/robot/x5 DOJO_D0=$DATA/d0_51 DOJO_BG_FOLDS=$PERC/bg_folds_51 DOJO_COLOR_CALIB=$PERC/catalog/color_calib_51.json"
  local f
  for f in "$DATA/d0_51/push_T" "$PERC/bg_folds_51" "$PERC/catalog/color_calib_51.json" "$PERC/catalog/robot/x5"; do
    is test -e "$f" || [[ "$DRY" == 1 ]] || die "missing $f (data / catalog relays)"
  done
  x rm -rf "$out"; x mkdir -p "$out"
  # a GL context on the NVIDIA EGL driver (Mesa llvmpipe would "work", ~100x slower: not what perception was validated on)
  xs "$env $VENV/bin/python -c 'from dojo.perception.render import gl; g = gl(); assert g is not None, \"no EGL/GL: perception would fall back to the CPU path\"; from OpenGL import GL; v = GL.glGetString(GL.GL_VENDOR).decode(); print(\"EGL:\", v, GL.glGetString(GL.GL_RENDERER).decode()); assert \"${PERCEPTION_GL_VENDOR:-NVIDIA}\" in v, \"EGL is not on the NVIDIA driver\"'"
  logrun "$out/run.log" 'frames|Traceback|Error' \
    "$env timeout 1200 nice $VENV/bin/python -m dojo.perception.eval_g1 --out $out --tasks push_T --episodes 1 --frames 0 --bg-loo --workers 1"
  [[ "$DRY" == 1 ]] && return 0
  check_py "$out/objects.jsonl" <<'PY'
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1])]
err = lambda r: r["xy_err_mm"] if r.get("xy_err_mm") is not None else float("nan")
ok = len(rows) >= 2 and all(r.get("found") and err(r) < 7.0 for r in rows)
print("RESULT smoke_perception:", "; ".join(f"{r['label']} found={r.get('found')} xy {err(r):.1f} mm" for r in rows),
      "(tolerance 7 mm; compute3 reference: t 0.5, target_t 0.7)")
sys.exit(0 if ok else 1)
PY
}

step_smoke_proxy_eval() {   # PROXY dev-eval like compute3: physics-only 5.1 episode + the Isaac Sim 6.0 render service, replay policy
  local out="$SMOKE/proxy_eval" lay="$DATA/simlab/g2_c3b/stack_layout.json" traj="$DATA/simlab/g2_c3b/stack_traj.json"
  if [[ "$DRY" != 1 ]] && ! test -x "$ISAAC6/kit/python/bin/python3"; then
    [[ "$ROLE" == physics ]] && die "no Isaac Sim 6.0 at $ISAAC6 (the isaac6 fetch)"
    log "no Isaac Sim 6.0 here (role $ROLE): skipped"; note "skipped (no Isaac Sim 6.0)"; return 0
  fi
  reserved_gate && return 0
  x rm -rf "$out"; x mkdir -p "$out"
  # proxy_eval.sh: MemAvailable >= 25 GB, the heavy GPU lock, one systemd --user scope (16G / 1200%) for both processes
  logrun "$out/run.log" '^\[proxy_eval\]|Error|error:|refusing|Segmentation' \
    "cd $CODE && ISAAC_SIM_ROOT=$ISAAC6 RD_HOME=$RD_HOME DOJO_DATA=$DATA timeout 2700 scripts/simlab/proxy_eval.sh --task stack_blocks --layouts $lay --policy replay:$traj --out $out"
  [[ "$DRY" == 1 ]] && return 0
  check_py "$out/_result.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
d = next(iter(r.get("details", {}).values()), {})
print(f"RESULT smoke_proxy_eval: episodes {r.get('eval_time')}, success_rate {r.get('success_rate')}, "
      f"steps {d.get('steps_used')}/{d.get('step_lim')}, render {r.get('render_s_median') or float('nan'):.2f} s/obs "
      f"(compute3: success, 263/550 steps, 0.12 s/obs)")
if r.get("success_rate") != 1.0:
    print("WARN: the proxy ran but the replay did not succeed: compare with G2b (contact divergence) before trusting this box")
sys.exit(0 if r.get("eval_time") == 1 and (r.get("render_s_median") or 0) > 0 else 1)
PY
}

step_smoke_bridge() {   # CPU: the bridge's launcher path on the fake strict server (BRIDGE-RUNBOOK §3): RuntimePolicy, 0 LLM calls
  local out="$SMOKE/bridge" id="fk-smoke"
  x rm -rf "$out"; x mkdir -p "$out/bench" "$out/out"
  logrun "$out/run.log" '^\[bridge|Traceback|Error' \
    "cd $CODE && BENCH=$out/bench timeout 900 scripts/dojo/bridge_episode.sh --id $id --fake --policy dojo.runtime.policy:RuntimePolicy --perception server --verify off --out $out/out"
  [[ "$DRY" == 1 ]] && return 0
  check_py "$out/out/$id/bridge.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"RESULT smoke_bridge: end={s.get('end')} steps={s.get('steps_used')}/{s.get('step_lim')} ended={s.get('ended')} "
      f"(fake strict server; a CPU check of our side: the strict episode needs role full + robot-harness)")
sys.exit(0 if s.get("end") in ("ended", "policy_done") else 1)
PY
}

step_ready() {   # READY note for the streams (like compute3's ~/dojo-robodojo/READY)
  if [[ "$DRY" == 1 ]]; then echo "    + cat > $RD_HOME/READY   # GPU, driver, role, pins, smoke results"; box_json; return 0; fi
  local gpu; gpu="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
  {
    echo "RoboDojo install for the dojo streams on $(hostname) ($gpu). Written $(date -u +%FT%TZ) by scripts/bootstrap_gpu_box.sh."
    echo "role       : $ROLE ($( [[ "$ROLE" == full ]] && echo "Isaac Sim 5.1 renders here: dev_eval.sh / official harness / bridge" || echo "PHYSICS-ONLY 5.1 (RoboDojoRunner(..., physics_only=True)) + Isaac Sim 6.0 render: proxy_eval.sh"))${ROLE_REASON:+; $ROLE_REASON}"
    echo "env python : $RDPY ($("$RDPY" -V 2>&1); Isaac Sim 5.1.0, IsaacLab, cuRobo, torch 2.7 cu128)"
    echo "RoboDojo   : $SRC (commit $(rd_commit), submodules as compute3); Assets -> hf_git/Assets ($ASSETS, $(du -sh "$HFG/Assets" 2>/dev/null | cut -f1), HF $HF_REV)"
    echo "controlr   : $CODE @ $(cat "$CODE/.deployed_commit" 2>/dev/null) (venv: perception extras, uv-managed CPython)"
    [[ "$ROLE" == physics ]] && echo "isaac 6.0  : $ISAAC6 ($(cat "$ISAAC6/VERSION" 2>/dev/null || echo missing); ISAAC_SIM_ROOT in ~/dojo-env.sh)"
    echo "run with   : source ~/dojo-env.sh; ~/dojo-<stream>/scripts/simlab/rdpy.sh -m <module> ...   (GPU lock ~/dojo-gpu.lock)"
    local s
    for s in smoke_twin smoke_render smoke_dev_eval smoke_proxy_eval smoke_perception smoke_bridge; do
      echo "$(printf '%-17s' "$s"): $(cat "$BOX/state/$s.result" 2>/dev/null || echo "not run")"
    done
    echo "read-only for streams; conventions: ~/DOJO_BOX.md; machine facts + capabilities: ~/dojo-box.json"
  } > "$RD_HOME/READY"
  cat "$RD_HOME/READY"
  box_json
  python3 -c 'import json, os; c = json.load(open(os.path.expanduser("~/dojo-box.json")))["capabilities"]; print("capabilities:", ", ".join(k for k, v in c.items() if v) or "none")' 
}

manifest() {   # DIR [NAME [REF_MD5 REF_HOST]]: "md5  path" for every file under DIR (sorted, C locale); read-only without NAME
  local dir="$1" name="${2:-}" ref="${3:-}" refh="${4:-}"
  [[ -d "$dir" ]] || die "manifest: no $dir"
  if [[ -z "$name" ]]; then
    (cd "$dir" && find . \( -type f -o -type l \) -printf '%P\0' | LC_ALL=C sort -z | xargs -0 -r md5sum); return
  fi
  local m="$BOX/state/manifest_$name"
  mkdir -p "$BOX/state"
  if ! (cd "$dir" && find . \( -type f -o -type l \) -printf '%P\0' | LC_ALL=C sort -z | xargs -0 -r md5sum) > "$m.md5.new"; then
    die "manifest: md5sum failed under $dir"
  fi
  mv "$m.md5.new" "$m.md5"
  python3 - "$m.md5" "$dir" "$ref" "$refh" > "$m.json.new" <<'PY' && mv "$m.json.new" "$m.json"
import hashlib, json, os, sys, time
f, d, ref, refh = sys.argv[1:5]
raw = open(f, "rb").read()
out = {"dir": d, "files": raw.count(b"\n"), "bytes": sum(os.path.getsize(os.path.join(d, ln.split(b"  ", 1)[1].decode()))
       for ln in raw.splitlines() if b"  " in ln), "md5": hashlib.md5(raw).hexdigest(),
       "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
if ref:
    out["ref"] = {"host": refh or None, "md5": ref, "equal": ref == out["md5"]}
print(json.dumps(out))
PY
  cat "$m.json"
}

# --------------------------------------------------------------------------------------------- dispatch
status_of() {
  local s="$1"
  if [[ -f "$BOX/state/$s.pid" ]] && kill -0 "$(cat "$BOX/state/$s.pid")" 2>/dev/null; then echo running
  elif [[ -f "$BOX/state/$s.done" && "$(cat "$BOX/state/$s.done")" == "$("fp_$s")" ]]; then echo "done"   # "always" steps: the driver clears it
  elif [[ -f "$BOX/state/$s.rc" && "$(cat "$BOX/state/$s.rc")" != 0 ]]; then echo failed
  else echo todo; fi
}

run_step() {
  local s="$1" t0 rc
  declare -F "step_$s" >/dev/null || die "unknown step $s (steps: ${STEPS[*]})"
  CUR="$s"
  if [[ "$DRY" == 1 ]]; then "step_$s"; return 0; fi
  mkdir -p "$BOX/state" "$BOX/logs"
  exec 7>"$BOX/state/$s.lock"
  flock -n 7 || die "step $s is already running"
  echo $$ > "$BOX/state/$s.pid"; rm -f "$BOX/state/$s.rc" "$BOX/state/$s.done"
  t0=$(date +%s)
  log "== $s start (role $ROLE, assets $ASSETS)"
  ( set -e; "step_$s" ); rc=$?
  echo "$rc" > "$BOX/state/$s.rc"; echo $(( $(date +%s) - t0 )) > "$BOX/state/$s.secs"
  [[ "$rc" == 0 ]] && "fp_$s" > "$BOX/state/$s.done"
  rm -f "$BOX/state/$s.pid"
  log "== $s $( if [[ "$rc" == 0 ]]; then echo "done"; else echo "FAILED rc=$rc"; fi) in $(( $(date +%s) - t0 )) s"
  return "$rc"
}

cmd="${1:-}"; shift || true
case "$cmd" in
  preflight) preflight < /dev/null ;;
  run) run_step "${1:?step}" ;;
  status) for s in "$@"; do echo "$s $(status_of "$s")"; done ;;
  poll) for s in "$@"; do
          last="$(tail -c 2000 "$BOX/logs/$s.log" 2>/dev/null | tr '\r' '\n' | grep -a -v '^\s*$' | tail -1 | cut -c1-150)"
          echo "$s $(status_of "$s") $(cat "$BOX/state/$s.secs" 2>/dev/null || echo -) ${last}"
        done ;;
  steps) echo "${STEPS[*]}" ;;
  boxjson) CUR=boxjson; box_json && cat "$HOME/dojo-box.json" ;;
  manifest) CUR=manifest; manifest "$@" ;;
  venv) make_venv "$(cd "${1:?checkout dir}" && pwd)" ;;   # a stream's deploy dir (scripts/deploy.sh builds no perception extras)
  fn) f="${1:?function}"; shift; CUR="$f"; "$f" "$@" ;;     # debugging: one internal function (e.g. rd_clone_pinned)
  *) echo "usage: box_steps.sh preflight | run STEP | status STEP... | poll STEP... | steps | venv DIR | boxjson | manifest DIR [NAME [REF_MD5 REF_HOST]]" >&2; exit 2 ;;
esac
