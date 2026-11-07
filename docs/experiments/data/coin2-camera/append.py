"""COIN2-CAM: append the frozen 70-row block to compute4's queue2 (run under flock queue2.txt.lock).

    flock ~/dojo-integ-c4-state/queue2.txt.lock python3 append.py ROWS EXPECTED_SHA256 RECEIPT N_ROWS

Existing rows are never reordered or changed. A queue holding no row (only blank lines) is replaced by the block,
because lane2.sh stops at a blank first line ("queue empty") and would never reach the appended rows. The STOP file is
never touched. Refuses when the block's hash differs from the frozen one, a row lacks the registered route, or a row
name is already queued.
"""
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

rows_p, want, receipt = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
S = Path.home() / "dojo-integ-c4-state"
Q = S / "queue2.txt"
block = rows_p.read_bytes()
assert hashlib.sha256(block).hexdigest() == want, "block hash differs from the frozen one"
rows = [ln for ln in block.decode().splitlines() if ln.strip()]
assert len(rows) == int(sys.argv[4])
assert all(ln.count("--setenv AGENTIC_MODEL=claude/claude-sonnet-5-5") == 1 for ln in rows)
before = Q.read_bytes()
old = [ln for ln in before.decode().splitlines() if ln.strip()]
names_old = {ln.split(" ", 1)[0] for ln in old}
assert not names_old & {ln.split(" ", 1)[0] for ln in rows}, "a row name is already queued"
stop_present = (S / "STOP").exists()
running = sorted(p.name for p in S.glob("running.*"))
if old:
    new = before if before.endswith(b"\n") else before + b"\n"
    new += block
    mode = "appended after existing rows"
else:
    new = block
    mode = "queue held no row (only blank lines): replaced by the block"
tmp = Q.with_name(Q.name + ".coin2cam.tmp")
tmp.write_bytes(new)
os.replace(tmp, Q)
after = Q.read_bytes()
assert after.startswith(before if old else b"") and after.endswith(block)
rec = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "host": "compute4", "stream": "COIN2-CAM",
       "mode": mode, "rows_before": len(old), "rows_after": len([ln for ln in after.decode().splitlines() if ln.strip()]),
       "positions_1_based": [len(old) + 1, len(old) + len(rows)],
       "sha256_before": hashlib.sha256(before).hexdigest(), "bytes_before": len(before),
       "sha256_after": hashlib.sha256(after).hexdigest(), "block_sha256": want,
       "first": rows[0].split(" ", 1)[0], "last": rows[-1].split(" ", 1)[0],
       "route_checked_rows": len(rows), "route": "claude/claude-sonnet-5-5",
       "stop_present": stop_present, "stop_touched": False, "running_lanes": running}
receipt.write_text(json.dumps(rec, indent=1) + "\n")
print(json.dumps(rec))
