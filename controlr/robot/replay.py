"""Replay robot: plays back recorded frames, ignoring the actions.

WHY: latency and prompt-caching benchmarks need realistic, *reproducible*
image sequences without a simulator — the model's replies must not change
what it sees next, or two models/configs could not be compared turn by turn.

Sources (``params["dir"]``):
* a run directory (see docs/ARCHITECTURE.md "Run log"): the first image of
  every user message in ``messages.jsonl`` (or the image whose label equals
  ``params["label"]``), bytes from ``images/<sha>.jpg``; states from
  ``setup.json`` (``state0``) and ``turns.jsonl`` (``state`` after each turn);
  the instruction from ``setup.json``;
* any directory of ``*.jpg|*.jpeg|*.png`` files (natural sort order).

Frame k is shown after the k-th ``execute``; at the end the last frame is
held (or the sequence restarts with ``params["loop"]=true``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from controlr.robot.base import Robot
from controlr.robot.kinematics import UR3Kinematics
from controlr.robot.mock import d435_camera
from controlr.robot.spec import ur3_cb3_spec
from controlr.types import Action, EventLevel, ExecReport, GoalReport, Observation, RobotState, SafetyEvent

_IMG_EXT = {".jpg", ".jpeg", ".png"}


def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            break                                   # truncated last line of a crashed run
    return out


def frames_from_run(run_dir: Path, label: str | None = None) -> list[Path]:
    """Image files of a run directory in turn order (one per user message)."""
    out: list[Path] = []
    for msg in _read_jsonl(run_dir / "messages.jsonl"):
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            continue
        imgs = [p for p in msg["content"] if isinstance(p, dict) and p.get("image_sha")]
        if label is not None:
            imgs = [p for p in imgs if p.get("label") == label] or imgs
        if imgs:
            f = run_dir / "images" / f"{imgs[0]['image_sha']}.jpg"
            if f.exists():
                out.append(f)
    return out


class ReplayRobot(Robot):
    def __init__(self, params: dict | None = None) -> None:
        p = dict(params or {})
        src = p.get("dir") or p.get("path")
        if not src:
            raise ValueError("replay backend needs robot.params.dir (a run dir or an image dir)")
        self.dir = Path(src).expanduser()
        if not self.dir.is_dir():
            raise FileNotFoundError(f"replay dir not found: {self.dir}")
        self.loop = bool(p.get("loop", False))
        self.dt = float(p.get("dt", 1.0))
        self.camera_name = str(p.get("camera", "scene"))
        self.spec = ur3_cb3_spec()
        self.kin = UR3Kinematics(tcp_offset=self.spec.tcp_offset)
        self.is_run = (self.dir / "messages.jsonl").exists()
        if self.is_run:
            self.frames = frames_from_run(self.dir, p.get("label"))
        else:
            self.frames = sorted((f for f in self.dir.iterdir() if f.suffix.lower() in _IMG_EXT),
                                 key=_natural_key)
        if not self.frames:
            raise ValueError(f"no frames found in {self.dir}")
        self._states = self._load_states()
        setup = self._setup()
        self.task_instruction = p.get("instruction") or setup.get("instruction") or \
            "Replay of recorded camera frames (actions do not change the scene)."
        self._i = 0
        self._t = 0.0

    # ------------------------------------------------------------------ data
    def _setup(self) -> dict:
        f = self.dir / "setup.json"
        try:
            return json.loads(f.read_text()) if f.exists() else {}
        except json.JSONDecodeError:
            return {}

    def _load_states(self) -> list[dict | None]:
        """states[k] = recorded state shown with frame k (None: unknown)."""
        if not self.is_run:
            return []
        out: list[dict | None] = [self._setup().get("state0")]
        out += [r.get("state") for r in _read_jsonl(self.dir / "turns.jsonl")]
        return out

    def _home_state(self) -> RobotState:
        q = np.array(self.spec.home_q, dtype=float)
        pos, rv = self.kin.fk(q)
        return RobotState(t=self._t, q=q, tcp_pos=pos, tcp_rotvec=rv,
                          gripper_mm=self.spec.gripper_max_mm, gripper_closed=False)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    # ------------------------------------------------------------------- api
    def reset(self, task_cfg: dict, seed: int | None = None) -> Observation:
        self._i = 0
        self._t = 0.0
        return self.observe()

    def state(self) -> RobotState:
        rec = self._states[self._i] if self._i < len(self._states) else None
        if not isinstance(rec, dict) or rec.get("q") is None:
            return self._home_state()
        try:
            return RobotState(
                t=self._t, q=np.asarray(rec["q"], float), tcp_pos=np.asarray(rec["tcp_pos"], float),
                tcp_rotvec=np.asarray(rec["tcp_rotvec"], float),
                gripper_mm=float(rec.get("gripper_mm") or 0.0),
                gripper_closed=bool(rec.get("gripper_closed")), holding=rec.get("holding"))
        except (KeyError, TypeError, ValueError):
            return self._home_state()

    def observe(self) -> Observation:
        img = np.asarray(Image.open(self.frames[self._i]).convert("RGB"), dtype=np.uint8)
        h, w = img.shape[:2]
        cam = d435_camera(w, h, self.camera_name)
        return Observation(t=self._t, images={self.camera_name: img}, cameras={self.camera_name: cam},
                           state=self.state())

    def execute(self, actions: list[Action]) -> ExecReport:
        before = self.state()
        if self._i + 1 < len(self.frames):
            self._i += 1
        elif self.loop:
            self._i = 0
        self._t += self.dt
        ev = [SafetyEvent(EventLevel.INFO, "replay",
                          f"replay frame {self._i + 1}/{len(self.frames)} (actions are not executed)")]
        return ExecReport(requested=list(actions), executed=list(actions), events=ev,
                          state_before=before, state_after=self.state(), duration_s=self.dt)

    def check_goal(self) -> GoalReport:
        return GoalReport(False, None, "replay: no goal check",
                          {"frame": self._i, "n_frames": len(self.frames)})
