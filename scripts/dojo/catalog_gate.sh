#!/usr/bin/env bash
# Per-box catalogue coverage gate (run it before every eval lane / job; it fails loudly): every candidate asset of
# every task in the perception catalogue's tasks.json has geometry, every slot base its support frames, every
# Articulation candidate its links.json (dojo.runtime.catalog_check --vocabularies all; RoboDojo's known-unshipped
# assets are not gaps). Prints the catalogue's index.json md5 (the row field catalog_md5).
#   PY=<a python with numpy> scripts/dojo/catalog_gate.sh [CATALOG]
# A gap means a perception-blind cell (2026-10-28: 143 assets of 15 cells were missing on every box). Fix with
#   python -m dojo.runtime.catalog_check --catalog C --vocabularies all --sync --assets <RoboDojo Assets>
# (into an overlay first; journal 2026-10-28-dojo-plan-none-perception).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CAT="${1:-${DOJO_CATALOG:-$HOME/dojo-perception-data/catalog}}"
PY="${PY:-python3}"
[[ -f "$CAT/index.json" ]] || { echo "CATALOG GATE FAILED: no index.json in $CAT" >&2; exit 2; }
echo "catalog_md5 $(md5sum "$CAT/index.json" | cut -c1-32) ($CAT)"
PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m dojo.runtime.catalog_check --catalog "$CAT" --vocabularies all
rc=$?
if [[ $rc -ne 0 ]]; then
  echo "CATALOG GATE FAILED (rc=$rc): perception cannot see some task objects on this box ($CAT); see above" >&2
  exit 1
fi
echo "catalog gate OK"
