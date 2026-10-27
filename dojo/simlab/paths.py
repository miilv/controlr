"""Where things live on the RoboDojo hosts (overridable by env vars)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

RD_HOME = Path(os.environ.get("RD_HOME", "/root/controlr-robodojo"))
RD_ROOT = Path(os.environ.get("RD_ROOT", str(RD_HOME / "RoboDojo")))
RD_PYTHON = Path(os.environ.get("RD_PYTHON", str(RD_HOME / "miniconda3/envs/robodojo/bin/python")))
ASSETS = RD_ROOT / "Assets"
OBJECTS = ASSETS / "Object" / "RoboDojo"
TASK_CONFIGS = RD_ROOT / "task" / "RoboDojo" / "config"
ENV_CFG = RD_ROOT / "env_cfg"
DATA = Path(os.environ.get("DOJO_DATA", "/root/dojo-data"))
LAYOUTS = DATA / "simlab" / "layouts"

# Eval data: never opened by simlab (rule in dojo/README.md). Code that lists or reads files
# must refuse paths under it.
EVAL_LAYOUT_DIR = ASSETS / "Eval_Layout"

# Our own layout seeds start here; RoboDojo's eval layouts are 0..N per eval seed.
SEED_BASE = 100_000
# layout ids >= RESERVED_MIN_ID are for joint paired runs with robot-harness (D-C10, 2026-10-27): never generated, read
# or run on our side (rendered measurement: 100000-09; G3 / twin dev: 100000-29)
RESERVED_MIN_ID = 100_100


def is_reserved_id(layout_id) -> bool:
    try:
        return int(layout_id) >= RESERVED_MIN_ID
    except (TypeError, ValueError):
        return False


def assert_not_eval(path: str | os.PathLike) -> None:
    # realpath never raises on unreadable parents (CI runs as non-root, RD_HOME is under /root)
    p = Path(os.path.realpath(path))
    ev = Path(os.path.realpath(EVAL_LAYOUT_DIR))
    if p == ev or ev in p.parents or "Eval_Layout" in p.parts:
        raise PermissionError(f"refusing to touch eval data: {p}")


# Assets/Traj demos keyed by something generic (make_kong: push rank 0..3; tic-tac-toe: board cell i_j).
# Every other Traj file is paired with an eval layout (imitate_sorting_sequence/<eval_seed>/<seed>.pkl,
# whose target_label is the answer) and is off limits like Eval_Layout.
GENERIC_TRAJ = ("make_kong", "play_tic_tac_toe")
_FIREWALL_EVENTS = ("open", "os.listdir", "os.scandir", "glob.glob", "glob.glob/2", "shutil.copyfile",
                    "shutil.copytree", "os.rename", "os.remove")
_firewall_on = False


def eval_asset_violation(path) -> str | None:
    """Why ``path`` is off limits, or None: anything under an ``Eval_Layout`` directory, and every file under
    ``Assets/Traj`` except ``Traj/<bench>/<task>/<file>`` of a ``GENERIC_TRAJ`` task (no I/O: abspath only)."""
    try:
        p = os.path.abspath(os.fsdecode(path))
    except TypeError:
        return None                                   # a file descriptor
    parts = Path(p).parts
    if "Eval_Layout" in parts:
        return "eval layouts"
    for i in range(1, len(parts)):
        if parts[i] == "Traj" and parts[i - 1] == "Assets":
            rest = parts[i + 1:]
            if len(rest) == 3 and rest[1] in GENERIC_TRAJ:
                return None
            return "eval-paired demonstrations (Assets/Traj)"
    return None


def install_asset_firewall() -> None:
    """Process-wide and permanent (an audit hook): opening, listing, globbing or copying an off-limits path
    (``eval_asset_violation``) raises PermissionError, whoever calls it, RoboDojo's own code included.
    Installed by every simlab env process (``RoboDojoRunner``, ``dev_eval``)."""
    global _firewall_on
    if _firewall_on:
        return

    def hook(event, args):
        if event in _FIREWALL_EVENTS and args:
            why = eval_asset_violation(args[0])
            if why:
                raise PermissionError(f"asset firewall: {why}: {args[0]}")
    sys.addaudithook(hook)
    _firewall_on = True
