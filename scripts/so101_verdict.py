"""Record the operator's verdict for a real-rig episode (the SO-101 has no automatic success
check: DONE ends the episode as ``done_unverified``). Writes ``verdict.json`` into the run dir;
aggregate it, not ``summary.json``'s ``success``.

    uv run python scripts/so101_verdict.py runs/<run_dir> success|partial|fail ["note"]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def main(argv: list[str]) -> None:
    if len(argv) < 2 or argv[1] not in ("success", "partial", "fail"):
        raise SystemExit(__doc__)
    run = Path(argv[0])
    if not (run / "summary.json").exists():
        raise SystemExit(f"{run} is not a run dir (no summary.json)")
    rec = {"verdict": argv[1], "note": " ".join(argv[2:]), "judged_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "outcome": json.loads((run / "summary.json").read_text()).get("outcome")}
    (run / "verdict.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(rec)


if __name__ == "__main__":
    main(sys.argv[1:])
