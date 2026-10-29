# source me: disk_guard TAG -> 0 when the filesystems holding $HOME and $DOJO_DATA both have >= DOJO_MIN_FREE_GB
# (default 10) free, else a loud message and 1. A full disk on a shared box takes everyone's jobs down (compute3,
# 29 Oct: / at 100 %); a job never starts below the floor. Boxes with a small data disk set their own floor
# (compute1: DOJO_MIN_FREE_GB=2 on ~/data2, next to its lanes' own guard). DOJO_MIN_FREE_GB=0 disables the check.
disk_guard() {
  local tag="${1:-[disk]}" need="${DOJO_MIN_FREE_GB:-10}" p free
  [[ "$need" == 0 ]] && return 0
  for p in "$HOME" "${DOJO_DATA:-$HOME}"; do
    [[ -e "$p" ]] || continue
    free=$(df -BG --output=avail "$p" 2>/dev/null | tail -1 | tr -dc 0-9)
    [[ -n "$free" ]] || continue
    if (( free < need )); then
      echo "$tag DISK GUARD: $(df -h --output=target "$p" | tail -1 | tr -d ' ') has ${free} GB free < DOJO_MIN_FREE_GB=${need}: not starting" >&2
      return 1
    fi
  done
  return 0
}
