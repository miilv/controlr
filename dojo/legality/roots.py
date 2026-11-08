"""Path policy for offline tools that read RoboDojo's asset tree (the f-catalogue extraction): roots and files they
may never read under. Kept here, with the rest of the legality tooling, because it has to name what it forbids.

``forbidden_part`` checks every component of a path, so a tool can refuse its STARTING root, not only what it finds
below it.
"""
from __future__ import annotations

import os
from pathlib import Path

FORBIDDEN_PARTS = {"task", "tasks", "config", "configs", "eval_layout", "traj", "coord", "logs", "log", "dc19",
                   "rh-misha", "controlr-dc19", "proc", "environ", "tasks.json", "info.json", "lane.env"}
FORBIDDEN_PREFIX = ("dojo-official-rh", ".env")
FORBIDDEN_SUBSTR = ("scored", "sealed")


def forbidden_part(path) -> str | None:
    """The first component of ``path`` an asset-reading tool may never read under (or open), or None."""
    for part in Path(os.path.expanduser(str(path))).parts:
        q = part.lower()
        if q in FORBIDDEN_PARTS or q.startswith(FORBIDDEN_PREFIX) or any(s in q for s in FORBIDDEN_SUBSTR):
            return part
    return None
