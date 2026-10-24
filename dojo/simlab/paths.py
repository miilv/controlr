"""Where things live on the RoboDojo hosts (overridable by env vars)."""

from __future__ import annotations

import os
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


def assert_not_eval(path: str | os.PathLike) -> None:
    p = Path(path).resolve()
    ev = EVAL_LAYOUT_DIR.resolve() if EVAL_LAYOUT_DIR.exists() else EVAL_LAYOUT_DIR
    if p == ev or ev in p.parents or "Eval_Layout" in p.parts:
        raise PermissionError(f"refusing to touch eval data: {p}")
