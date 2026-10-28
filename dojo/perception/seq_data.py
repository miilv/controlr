"""DEV: read the per-step frame sequences ``seq_capture`` writes (stream C, continuous perception research).

    seq = Sequence(dir)                    # <out>/<task>_<lid>/ (meta.json, steps.jsonl, frames/)
    fr = seq.frame(step)                   # {"images": {cam: HxWx3 uint8}, "state": {k: array}, "instruction"}
    gt = seq.gt(step)                      # DEV-ONLY GT rows at that step (scorer side, never a policy input)
    ev = seq.events()                      # gripper closes / opens from the replayed actions

GT rows are the scorer's; nothing here is reachable from the runtime.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def _read_image(path: Path) -> np.ndarray:
    import cv2

    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return np.ascontiguousarray(im[:, :, ::-1])


class Sequence:
    def __init__(self, d: str | Path):
        self.dir = Path(d)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.task = self.meta["task"]
        self.layout_id = self.meta["layout_id"]
        self.instruction = self.meta.get("instruction") or ""
        self.rows = [json.loads(ln) for ln in open(self.dir / "steps.jsonl") if ln.strip()]
        self.by_step = {int(r["step"]): r for r in self.rows}
        self.steps = sorted(self.by_step)
        self.fmt = self.meta.get("fmt", "png")

    def state(self, step: int) -> dict:
        return {k: np.asarray(v, float) for k, v in self.by_step[step]["state"].items()}

    def frame(self, step: int, fmt: str | None = None, cams=CAMS) -> dict:
        fmt = fmt or self.fmt
        imgs = {}
        for c in cams:
            p = self.dir / "frames" / f"{step:04d}_{c}.{fmt}"
            if not p.exists() and fmt != self.fmt:
                p = self.dir / "frames" / f"{step:04d}_{c}.{self.fmt}"
            imgs[c] = _read_image(p)
        return {"images": imgs, "state": self.state(step), "instruction": self.instruction}

    def gt(self, step: int) -> list[dict]:
        return self.by_step[step]["gt"]

    def grip(self, step: int) -> dict:
        """The gripper command of the last action before ``step``'s observation, per arm (1 open, 0 closed)."""
        g = self.by_step[step].get("grip_cmds") or []
        if not g:
            return {}
        last = g[-1]
        return {a: float(last[f"{a}_ee_joint_state"]) for a in ("left", "right") if f"{a}_ee_joint_state" in last}

    def events(self, close_thr: float = 0.5, open_thr: float = 0.6) -> list[dict]:
        """Grasps (a closing command, the hand not already closed: the belief's attach rule, as ``GTProbe``) and
        releases (an opening command above ``open_thr`` after a close), with the observation step they belong to.
        With stride 1 the observation at ``step`` is the state right after that command."""
        out, closed, prev = [], {"left": False, "right": False}, {"left": 1.0, "right": 1.0}
        for s in self.steps:
            for a, g in self.grip(s).items():
                if g < close_thr and g < prev[a] - 1e-6 and not closed[a]:
                    closed[a] = True
                    out.append({"kind": "grasp", "arm": a, "step": s})
                elif g > open_thr and closed[a]:
                    closed[a] = False
                    out.append({"kind": "release", "arm": a, "step": s})
                prev[a] = g
        return out


__all__ = ["Sequence", "CAMS"]
