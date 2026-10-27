#!/usr/bin/env bash
# A RoboDojo install of our own as a symlink farm over a read-only one (the box's ~/dojo-robodojo), so that an
# XPolicyLab policy dir and the eval client's outputs land in OUR tree and the shared install is never written:
#   DEST/miniconda3, policy_venv, robodojo_commit.txt -> SRC/... (read-only)
#   DEST/RoboDojo/      one symlink per entry of SRC/RoboDojo/ except eval_result/ (ours) and XPolicyLab/ (ours: one
#                       symlink per entry except policy/ (ours: policy dirs are copied in by the caller, e.g.
#                       scripts/dojo/official_eval.sh installs robot-harness's harness_strict_j2 shim there) and
#                       setup_policy_server.py (a byte copy: python puts the REAL directory of a symlinked script on
#                       sys.path, which would import the shared policy/ dir)).
# The same construction as scripts/dojo/isr_farm.sh (isr-lab), for any box. RoboDojo's own path logic keeps the farm
# (pwd-based roots in its shell scripts, abspath(__file__) in global_configs.py). Run Python through it with
# PYTHONDONTWRITEBYTECODE=1 so imports never write __pycache__ into the shared tree. Idempotent.
#   scripts/dojo/rd_farm.sh SRC DEST [--check]
set -euo pipefail
SRC="${1:?SRC (a RoboDojo install: miniconda3/ policy_venv/ RoboDojo/)}" DEST="${2:?DEST}"
[[ -d "$SRC/RoboDojo/XPolicyLab" ]] || { echo "rd_farm: $SRC has no RoboDojo/XPolicyLab" >&2; exit 2; }
R="$DEST"
link() { [[ -L "$2" || -e "$2" ]] || { ln -s "$1" "$2"; echo "link $2 -> $1"; }; }
if [[ "${3:-}" == --check ]]; then
  for p in "$R/RoboDojo/eval_result" "$R/RoboDojo/XPolicyLab/policy"; do
    [[ -d "$p" && ! -L "$p" ]] || { echo "rd_farm: not ours: $p"; exit 1; }
  done
  cmp -s "$SRC/RoboDojo/XPolicyLab/setup_policy_server.py" "$R/RoboDojo/XPolicyLab/setup_policy_server.py" \
    || { echo "rd_farm: setup_policy_server.py differs from the source"; exit 1; }
  echo "farm ok: $R"; exit 0
fi
mkdir -p "$R/RoboDojo/eval_result" "$R/RoboDojo/XPolicyLab/policy"
for e in miniconda3 policy_venv robodojo_commit.txt; do [[ -e "$SRC/$e" ]] && link "$SRC/$e" "$R/$e"; done
for p in "$SRC/RoboDojo/"* "$SRC/RoboDojo/".[!.]*; do
  [[ -e "$p" ]] || continue
  n="$(basename "$p")"
  case "$n" in eval_result|XPolicyLab|__pycache__) continue ;; esac
  link "$p" "$R/RoboDojo/$n"
done
for p in "$SRC/RoboDojo/XPolicyLab/"* "$SRC/RoboDojo/XPolicyLab/".[!.]*; do
  [[ -e "$p" ]] || continue
  n="$(basename "$p")"
  case "$n" in policy|__pycache__) continue ;; setup_policy_server.py)
    [[ -e "$R/RoboDojo/XPolicyLab/$n" ]] || { cp "$p" "$R/RoboDojo/XPolicyLab/$n"; echo "copy $n"; }; continue ;; esac
  link "$p" "$R/RoboDojo/XPolicyLab/$n"
done
for n in __init__.py lib.sh; do [[ -e "$SRC/RoboDojo/XPolicyLab/policy/$n" ]] && link "$SRC/RoboDojo/XPolicyLab/policy/$n" "$R/RoboDojo/XPolicyLab/policy/$n"; done
cmp -s "$SRC/RoboDojo/XPolicyLab/setup_policy_server.py" "$R/RoboDojo/XPolicyLab/setup_policy_server.py" \
  || { echo "rd_farm: setup_policy_server.py differs from the source"; exit 1; }
echo "farm ok: $R"
