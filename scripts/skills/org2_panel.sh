#!/usr/bin/env bash
# ORG2 registered G3 panel, one lane (docs/experiments/2026-11-07-org2-g3-prereg.md), from this deployed copy:
#   scripts/skills/org2_panel.sh <lane lock file> <id list> <settled layout dir> <B skill attrs file>
# Arm A (defaults) then arm B (the registered attrs) over the same ids, through org2_g3.sh (admission, light lock,
# scope, timeout). Run names panelA / panelB; outputs ~/dojo-data/org2/g3/panel{A,B}_<commit>/.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCK="$1"; IDS="$2"; LAY="$3"; ATTRS="$4"
mapfile -t B < <(tr ' ' '\n' < "$ATTRS" | grep -v '^$')
ORG2_LOCKS="$LOCK" "$HERE/scripts/skills/org2_g3.sh" panelA "$IDS" "$LAY" --trace
sleep 60
ORG2_LOCKS="$LOCK" "$HERE/scripts/skills/org2_g3.sh" panelB "$IDS" "$LAY" --trace "${B[@]}"
echo "[org2_panel] lane $LOCK done"
