#!/usr/bin/env bash
# isr-lab: our own BENCH and RD_HOME under ~/data2/ilia as symlink farms over Mikhail's installs, so that his
# launchers (bin/robot-start-j2 -> harness/robodojo_{start,launch}_j2.sh, unmodified) write ONLY into our tree:
#   ~/data2/ilia/bench   harness/ bin/ skills/ tools/ -> ~/data2/bench/... (read-only); run/ is ours (the strict
#                        server's socket, meta.json, episode.jsonl, obs PNGs, launch logs, the copied RoboDojo result)
#   ~/data2/ilia/rdhome  miniconda3/ policy_venv/ robodojo_commit.txt -> ~/data2/controlr-robodojo/... (read-only);
#                        RoboDojo/ = one symlink per entry of his RoboDojo/ except eval_result/ (ours: the eval client
#                        writes its manifest and result there, relative to the cwd) and XPolicyLab/ (ours, with one
#                        symlink per entry except policy/ (ours: the launcher copies the shim into
#                        policy/harness_strict_j2/) and setup_policy_server.py (a byte copy: python puts the REAL
#                        directory of a symlinked script on sys.path, which would import his policy dir)).
# RoboDojo's own path logic keeps the farm: eval_policy.sh / robodojo.sh use `pwd` (logical) roots, global_configs.py
# uses abspath(__file__) (not realpath). Run with PYTHONDONTWRITEBYTECODE=1 (isr_env.sh) so imports through the farm
# never write __pycache__ into his trees. Idempotent; prints what it created.
#   scripts/dojo/isr_farm.sh [--check]
set -euo pipefail
SRC_BENCH=/home/anywherevla/data2/bench SRC_RD=/home/anywherevla/data2/controlr-robodojo
ILIA=/home/anywherevla/data2/ilia
B="$ILIA/bench" R="$ILIA/rdhome"
link() { [[ -L "$2" || -e "$2" ]] || { ln -s "$1" "$2"; echo "link $2 -> $1"; }; }
if [[ "${1:-}" == --check ]]; then
  for p in "$B/run" "$R/RoboDojo/eval_result" "$R/RoboDojo/XPolicyLab/policy"; do
    [[ -d "$p" && ! -L "$p" ]] || { echo "not ours: $p"; exit 1; }
  done
  echo "farm ok: $B, $R"; exit 0
fi
mkdir -p "$B/run" "$R/RoboDojo/eval_result" "$R/RoboDojo/XPolicyLab/policy"
for e in harness bin skills tools; do [[ -e "$SRC_BENCH/$e" ]] && link "$SRC_BENCH/$e" "$B/$e"; done
# the lock-aware starters append START / DONE to $BENCH/COORDINATION.md: the shared log, append-only
link "$SRC_BENCH/COORDINATION.md" "$B/COORDINATION.md"
for e in miniconda3 policy_venv robodojo_commit.txt; do link "$SRC_RD/$e" "$R/$e"; done
for p in "$SRC_RD/RoboDojo/"* "$SRC_RD/RoboDojo/".[!.]*; do
  [[ -e "$p" ]] || continue
  n="$(basename "$p")"
  case "$n" in eval_result|XPolicyLab) continue ;; esac
  link "$p" "$R/RoboDojo/$n"
done
for p in "$SRC_RD/RoboDojo/XPolicyLab/"* "$SRC_RD/RoboDojo/XPolicyLab/".[!.]*; do
  [[ -e "$p" ]] || continue
  n="$(basename "$p")"
  case "$n" in policy|__pycache__) continue ;; setup_policy_server.py)
    [[ -e "$R/RoboDojo/XPolicyLab/$n" ]] || { cp "$p" "$R/RoboDojo/XPolicyLab/$n"; echo "copy $n"; }; continue ;; esac
  link "$p" "$R/RoboDojo/XPolicyLab/$n"
done
for n in __init__.py lib.sh; do [[ -e "$SRC_RD/RoboDojo/XPolicyLab/policy/$n" ]] && link "$SRC_RD/RoboDojo/XPolicyLab/policy/$n" "$R/RoboDojo/XPolicyLab/policy/$n"; done
cmp -s "$SRC_RD/RoboDojo/XPolicyLab/setup_policy_server.py" "$R/RoboDojo/XPolicyLab/setup_policy_server.py" \
  || { echo "setup_policy_server.py differs from the source"; exit 1; }
echo "farm ok: $B, $R"
