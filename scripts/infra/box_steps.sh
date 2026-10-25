#!/usr/bin/env bash
# Remote half of scripts/bootstrap_gpu_box.sh. Runs ON the GPU box: the driver streams it for the preflight
# (nothing is written to the box) and pushes it to ~/dojo-bootstrap/lib/ for every other step.
#
#   box_steps.sh preflight                 read-only checks -> PASS/WARN/FAIL lines + GO / NO-GO (exit 0 / 1)
#   box_steps.sh run STEP                  run one step; marker state/STEP.done (fingerprint) on success
#   box_steps.sh status STEP...            done | running | failed | todo, one line per step
#   box_steps.sh poll STEP...              status + the last line of each step's log (for the driver)
#   box_steps.sh steps                     the step names
#   box_steps.sh venv DIR                  DIR/.venv with the perception extras (for a stream's deploy dir)
#
# Config: env vars, or $BOX/config.env written by the driver (ROLE, ASSETS, CODE_SHA, MIN_DISK_GB).
# DRY_RUN=1: print every command instead of running it; every "is it done already?" check answers no.
# Layout on the box (compute3 conventions, dojo/README.md): ~/dojo-robodojo (RoboDojo + Isaac Sim 5.1, read-only
# for streams), ~/controlr (main), ~/dojo-data/<stream>/, ~/dojo-perception-data/, ~/dojo-gpu.lock.
set -uo pipefail

LIB="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo .)"
DRY="${DRY_RUN:-0}"
BOX="${BOX:-$HOME/dojo-bootstrap}"
# shellcheck source=/dev/null
[[ "$DRY" != 1 && -f "$BOX/config.env" ]] && source "$BOX/config.env"
ROLE="${ROLE:-full}"                       # full: Isaac Sim 5.1 renders here; physics: physics-only box
ASSETS="${ASSETS:-needed}"                 # needed: pins/robodojo_assets.txt (13.5 GB); all: every Asset but eval data (40.6 GB)
MIN_DISK_GB="${MIN_DISK_GB:-300}"
PREFIX="${PREFIX:-}"                       # big disk elsewhere: ~/dojo-{robodojo,data,perception-data} become symlinks into it
CODE_SHA="${CODE_SHA:-unknown}"
RD_HOME="${RD_HOME:-$HOME/dojo-robodojo}"
SRC="$RD_HOME/RoboDojo"
CONDA="$RD_HOME/miniconda3"
ENVP="$CONDA/envs/robodojo"
RDPY="$ENVP/bin/python"
HFG="$RD_HOME/hf_git"
CODE="${CODE_DIR:-$HOME/controlr}"
VENV="$CODE/.venv"
DATA="${DOJO_DATA:-$HOME/dojo-data}"
PERC="${DOJO_PERCEPTION_DATA:-$HOME/dojo-perception-data}"
SMOKE="$DATA/infra/smoke"
PINS="$LIB/pins"                                      # read here
ASSETS_LIST="${ASSETS_PINS:-$PINS/robodojo_assets.txt}"   # ASSETS_PINS: another list (tests, a partial install)
LIBX="${LIB_REMOTE:-$LIB}"                           # the same files as named in commands (dry runs print the box path)
HF_URL=https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo
HF_REV=269457a3b42d8d0ae69898a601f2c1487c2f826e      # HF dataset main on 2026-10-25 = compute3's hf_git HEAD
UV_VERSION=0.12.1                                     # compute3's uv
GIT_LFS_VERSION=3.4.1                                 # compute3's git-lfs (only if the box has none)
DRIVER_OK="${BOX_DRIVER_OK:-570 580}"                 # Isaac Sim 5.1 RTX: 580 renders (isr-lab 4090), 595 segfaults
STEPS=(conventions code venv robodojo assets assets_post catalog_post smoke_twin smoke_render smoke_dev_eval
       smoke_perception ready)
export PATH="$HOME/.local/bin:$PATH" UV_PYTHON_PREFERENCE=only-managed GIT_TERMINAL_PROMPT=0
export RD_HOME DOJO_DATA="$DATA" OMNI_KIT_ACCEPT_EULA=YES

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
fp_conventions() { echo "v1 $ROLE ${PREFIX:-home}"; }
fp_code() { echo "$CODE_SHA"; }
fp_venv() { echo "py3.12 $(hash12 "$PINS/controlr-venv.txt")"; }
fp_robodojo() { echo "$(hash12 "$PINS/robodojo_commit.txt") uv$UV_VERSION"; }
fp_assets() { echo "$ASSETS $HF_REV $(hash12 "$ASSETS_LIST")"; }
fp_assets_post() { echo "$(fp_robodojo) $(fp_assets)"; }
fp_catalog_post() { echo "always"; }
fp_smoke_twin() { echo "$CODE_SHA $(fp_assets_post)"; }
fp_smoke_render() { fp_smoke_twin; }
fp_smoke_dev_eval() { fp_smoke_twin; }
fp_smoke_perception() { echo "$CODE_SHA $(fp_venv)"; }
fp_ready() { echo "always"; }

# --------------------------------------------------------------------------------------------- preflight
# Read-only: nvidia-smi, df, free, /proc, command -v, curl HEADs. Writes nothing on the box.
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
  # GPU + driver
  local gpu name drv vram used major
  if ! gpu="$(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader,nounits 2>&1 | sed -n 1p)" \
      || [[ -z "$gpu" || "$gpu" == *"failed"* || "$gpu" == *"No devices"* ]]; then
    pf FAIL gpu "nvidia-smi: ${gpu:-no output}"
  else
    IFS=',' read -r name drv vram used <<< "$gpu"
    name="${name# }"; drv="${drv// /}"; vram="${vram// /}"; used="${used// /}"; major="${drv%%.*}"
    pf PASS gpu "$name, $((vram / 1024)) GB VRAM ($(nvidia-smi -L 2>/dev/null | wc -l) GPU)"
    if [[ " $DRIVER_OK " == *" $major "* ]]; then
      pf PASS driver "$drv (Isaac Sim 5.1 RTX tested on 580)"
    elif [[ "$ROLE" == full ]]; then
      pf FAIL driver "$drv: Isaac Sim 5.1 RTX needs driver ${DRIVER_OK// / or }; 595 segfaults in librtx.scenedb at boot (compute3). Ask for 580, or --role physics"
    else
      pf WARN driver "$drv: not ${DRIVER_OK// / or }, so no 5.1 rendering; --role physics is fine"
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
  if [[ "$mem_t" -lt 30 ]]; then pf FAIL ram "${mem_t} GB < 32 GB"
  elif [[ "$mem_a" -lt 25 || "$mem_t" -lt 60 ]]; then pf WARN ram "${mem_t} GB total, ${mem_a} GB available (bundle scopes want 24 GB, jobs start at >= 25 GB available)"
  else pf PASS ram "${mem_t} GB total, ${mem_a} GB available"; fi
  cpus="$(nproc)"; model="$(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -1)"
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
  elif [[ "$ROLE" == full ]]; then pf FAIL vulkan "no NVIDIA Vulkan ICD (container without graphics capability?): Kit RTX will not start"
  else pf WARN vulkan "no NVIDIA Vulkan ICD"; fi
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
  if systemctl --user show-environment >/dev/null 2>&1; then pf PASS systemd "user manager up (systemd-run --user scopes for proxy_eval / integ_eval)"
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
  if sudo -n true 2>/dev/null; then pf INFO sudo "passwordless sudo"; else pf INFO sudo "no passwordless sudo (bootstrap needs none)"; fi
  echo
  if [[ "$PF_FAIL" -eq 0 ]]; then echo "GO ($PF_WARN warnings)"; return 0; fi
  echo "NO-GO ($PF_FAIL failed, $PF_WARN warnings)"; return 1
}

# --------------------------------------------------------------------------------------------- steps
step_conventions() {   # dojo/README.md machine conventions: dirs, the team GPU lock, env file, uv policy
  if [[ -n "$PREFIX" ]]; then   # the big directories live on --prefix, under the usual names in $HOME
    local d
    for d in dojo-robodojo dojo-data dojo-perception-data; do
      x mkdir -p "$PREFIX/$d"
      if is test -L "$HOME/$d" && [[ "$(readlink -f "$HOME/$d")" == "$(readlink -f "$PREFIX/$d")" ]]; then continue; fi
      is test -e "$HOME/$d" && die "$HOME/$d exists and is not a link to $PREFIX/$d: move it there first"
      x ln -s "$PREFIX/$d" "$HOME/$d"
    done
  fi
  ensure_uv
  x mkdir -p "$DATA"/{simlab,perception,skills,integ,agent,infra} "$PERC" "$BOX/state" "$BOX/logs"
  x touch "$HOME/dojo-gpu.lock" "$DATA/simlab/.queue.lock"
  # uv must never pick a system interpreter (Ubuntu 22.04's python3.11 is 3.11.0rc1: SAM3 / torch.jit segfault)
  if ! is grep -qs 'python-preference' "$HOME/.config/uv/uv.toml"; then
    x mkdir -p "$HOME/.config/uv"
    xs "printf 'python-preference = \"only-managed\"\\n' >> $HOME/.config/uv/uv.toml"
  fi
  dnote "as root only: /root/controlr-robodojo -> dojo-robodojo (the compute2 default RD_HOME)"
  if [[ "$HOME" == /root || "$DRY" == 1 ]] && ! is test -e /root/controlr-robodojo; then
    x ln -s dojo-robodojo /root/controlr-robodojo       # the compute2 default RD_HOME of rdpy.sh / dev_eval.sh / paths.py
  fi
  local envf="$HOME/dojo-env.sh"
  if [[ "$DRY" == 1 ]]; then echo "    + cat > $envf   # RD_HOME RD_ROOT DOJO_DATA DOJO_GPU_LOCK DOJO_CATALOG DOJO_X5_DIR ..."
  else
    cat > "$envf" <<EOF
# dojo GPU box environment (scripts/bootstrap_gpu_box.sh). Source it before running repo scripts here:
#   source ~/dojo-env.sh && cd ~/dojo-<stream> && scripts/simlab/rdpy.sh -m ...
export RD_HOME="$RD_HOME"                 # RoboDojo + Isaac Sim 5.1 (read-only for streams)
export RD_ROOT="$SRC"
export DOJO_DATA="$DATA"                  # data: \$DOJO_DATA/<stream>/
export DOJO_GPU_LOCK="$HOME/dojo-gpu.lock"   # ONE GPU-heavy job per machine (flock; rdpy.sh / dev_eval.sh / proxy_eval.sh take it)
export DOJO_PERCEPTION_DATA="$PERC"
export DOJO_CATALOG="$PERC/catalog"       # perception catalogue (skills also: \$DOJO_DATA/skills2/catalog, skills3/catalog)
export DOJO_X5_DIR="$PERC/catalog/robot/x5"
# genuine Isaac 5.1 calibration (5.1-rendering box), per run:
#   DOJO_D0=\$DOJO_DATA/d0_51 DOJO_BG_FOLDS=\$DOJO_PERCEPTION_DATA/bg_folds_51 DOJO_COLOR_CALIB=\$DOJO_CATALOG/color_calib_51.json
export UV_PYTHON_PREFERENCE=only-managed
export OMNI_KIT_ACCEPT_EULA=YES
export PATH="\$HOME/.local/bin:\$PATH"
EOF
    log "wrote $envf"
  fi
  if [[ "$DRY" == 1 ]]; then echo "    + cat > ~/DOJO_BOX.md   # the conventions below"
  else
    cat > "$HOME/DOJO_BOX.md" <<EOF
# This box: dojo GPU box (role $ROLE), set up by controlr scripts/bootstrap_gpu_box.sh

- \`source ~/dojo-env.sh\` first (RD_HOME, DOJO_DATA, DOJO_GPU_LOCK, catalogue paths).
- RoboDojo (Isaac Sim 5.1 + IsaacLab + cuRobo + XPolicyLab, pinned like compute3): \`$RD_HOME\` -- read-only for streams.
  Run through \`scripts/simlab/rdpy.sh\` (conda python + the GPU lock). Status: \`$RD_HOME/READY\`.
- Code: \`~/controlr\` = main at deploy time (\`.deployed_commit\`); streams deploy to \`~/dojo-<stream>\`.
- Data: \`~/dojo-data/<stream>/\` (dev layouts \`simlab/layouts\`, D0 \`d0\` / \`d0_51\`), perception: \`~/dojo-perception-data\`.
- ONE GPU-heavy job per machine: every Isaac / model job holds \`flock ~/dojo-gpu.lock\` (rdpy.sh, isaac6.sh, dev_eval.sh,
  proxy_eval.sh, integ_eval.sh do it; the file holds "pid stream start cmd"). Keep a hold <= 30 min, yield 90 s between jobs.
- Check \`nvidia-smi\` before starting Isaac or a model; never touch other people's processes (kill by PID, never pkill -f
  over ssh); stop what you started.
- Big jobs in a scope: \`systemd-run --user --scope -p MemoryMax=24G -p CPUQuota=1400% ...\`; start only with >= 25 GB available.
- Eval data firewall: never read \`Assets/Eval_Layout/**\` or eval-paired \`Assets/Traj\` (they are not even downloaded here).
EOF
  fi
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

ensure_uv() {   # uv $UV_VERSION in ~/.local/bin (one installer at a time: venv and robodojo run in parallel)
  if is test -x "$HOME/.local/bin/uv" && [[ "$("$HOME/.local/bin/uv" --version 2>/dev/null | awk '{print $2}')" == "$UV_VERSION" ]]; then return 0; fi
  dnote "only if ~/.local/bin/uv is not $UV_VERSION"
  xs "mkdir -p $BOX/state && flock $BOX/state/.uv.lock sh -c 'curl -LsSf https://astral.sh/uv/$UV_VERSION/install.sh | env UV_NO_MODIFY_PATH=1 sh'"
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
  if ! is test -d "$HFG/.git"; then
    x env GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 --no-checkout "$HF_URL" "$HFG"
    x git -C "$HFG" lfs install --local
  fi
  x git -C "$HFG" sparse-checkout init --no-cone
  x cp "$list" "$HFG/.git/info/sparse-checkout"
  if ! is test "$(git -C "$HFG" rev-parse HEAD 2>/dev/null)" = "$HF_REV" || ! is test -d "$HFG/Assets"; then
    x env GIT_LFS_SKIP_SMUDGE=1 git -C "$HFG" checkout -q main
    if [[ "$DRY" != 1 && "$(git -C "$HFG" rev-parse HEAD 2>/dev/null)" != "$HF_REV" ]]; then
      log "HF main moved past the pin $HF_REV: fetching the pinned revision"
      x git -C "$HFG" fetch -q --depth 1 origin "$HF_REV"
      x env GIT_LFS_SKIP_SMUDGE=1 git -C "$HFG" checkout -q --detach "$HF_REV"
    fi
  fi
  x env GIT_LFS_SKIP_SMUDGE=1 git -C "$HFG" sparse-checkout reapply
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
    [[ "$ptr" -eq 0 ]] || die "$ptr LFS pointer files left (download incomplete): re-run the assets step"
    [[ ! -e "$HFG/Assets/Eval_Layout" && ! -e "$HFG/Assets/Traj/RoboDojo/imitate_sorting_sequence" ]] \
      || die "eval data in the checkout: check $list"
  fi
  x rm -rf "$HFG/.git/lfs/objects"   # stored once (the working tree), like compute3
  x cp "$list" "$applied"
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

step_smoke_twin() {   # physics-only Twin: boot, load (+ settle check), 40-step rollout, snapshot/restore + 40, contact-restore bitwise check
  local out="$SMOKE/twin"
  x rm -rf "$out"; x mkdir -p "$out"
  logrun "$out/run.log" '^\[twin\]|Error|error:' \
    "cd $CODE && timeout 1800 $(rdpy) -m dojo.simlab.twin --task stack_blocks --num-envs 1 --steps 40 --physics-only --out $out --g2-dir $DATA/simlab/g2_c3b"
  [[ "$DRY" == 1 ]] && return 0
  check_py "$out/bench_stack_blocks_n1.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
stable = all(r["load_first"]["stable"]) and all(r["load_new_layout"]["stable"])
cr = r.get("contact_restore", {})
print(f"RESULT smoke_twin: boot {r['boot_s']:.0f} s, layouts stable {stable}, 40 steps at {r['steps']['env_steps_per_s']:.1f} env-steps/s, "
      f"restore+40 {r['segment']['restore_plus_40_steps_s']:.2f} s, contact restore bitwise {cr.get('bitwise')}")
sys.exit(0 if stable and r["steps"]["n"] == 40 and cr.get("bitwise") is True else 1)
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

step_smoke_dev_eval() {   # the OFFICIAL harness (RoboDojo main.py via dev_eval.sh) on one dev layout, replaying G2's stack trajectory
  [[ "$ROLE" == full ]] || { log "role $ROLE: skipped (dev_eval needs 5.1 rendering)"; note "skipped (role $ROLE)"; return 0; }
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

step_ready() {   # READY note for the streams (like compute3's ~/dojo-robodojo/READY)
  if [[ "$DRY" == 1 ]]; then echo "    + cat > $RD_HOME/READY   # GPU, driver, role, pins, smoke results"; return 0; fi
  local gpu; gpu="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
  {
    echo "RoboDojo install for the dojo streams on $(hostname) ($gpu). Written $(date -u +%FT%TZ) by scripts/bootstrap_gpu_box.sh."
    echo "role       : $ROLE ($( [[ "$ROLE" == full ]] && echo "Isaac Sim 5.1 renders here: dev_eval.sh / official harness" || echo "PHYSICS-ONLY: RoboDojoRunner(..., physics_only=True)"))"
    echo "env python : $RDPY ($("$RDPY" -V 2>&1); Isaac Sim 5.1.0, IsaacLab, cuRobo, torch 2.7 cu128)"
    echo "RoboDojo   : $SRC (commit $(rd_commit), submodules as compute3); Assets -> hf_git/Assets ($ASSETS, $(du -sh "$HFG/Assets" 2>/dev/null | cut -f1), HF $HF_REV)"
    echo "controlr   : $CODE @ $(cat "$CODE/.deployed_commit" 2>/dev/null) (venv: perception extras, uv-managed CPython)"
    echo "run with   : source ~/dojo-env.sh; ~/dojo-<stream>/scripts/simlab/rdpy.sh -m <module> ...   (GPU lock ~/dojo-gpu.lock)"
    local s
    for s in smoke_twin smoke_render smoke_dev_eval smoke_perception; do
      echo "$(printf '%-17s' "$s"): $(cat "$BOX/state/$s.result" 2>/dev/null || echo "not run")"
    done
    echo "read-only for streams; conventions: ~/DOJO_BOX.md"
  } > "$RD_HOME/READY"
  cat "$RD_HOME/READY"
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
  venv) make_venv "$(cd "${1:?checkout dir}" && pwd)" ;;   # a stream's deploy dir (scripts/deploy.sh builds no perception extras)
  fn) f="${1:?function}"; shift; CUR="$f"; "$f" "$@" ;;     # debugging: one internal function (e.g. rd_clone_pinned)
  *) echo "usage: box_steps.sh preflight | run STEP | status STEP... | poll STEP... | steps | venv DIR" >&2; exit 2 ;;
esac
