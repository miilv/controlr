"""DEV: full per-step frame sequences of a recorded episode, for continuous-perception research (stream C).

A proxy / rendered run saves only the frames it asked for (frame 0, scan, looks: ``save_look_frames``), but the
runtime's executed action list is complete (``<run>/<tag>_<lid>_actions.json.gz``). ``SeqReplayPolicy`` replays that
list through ``dojo.runtime.eval`` in mode gt (physics-only 5.1 + the 6.0 render service, or genuine 5.1), returning
``DOJO_SEQ_STRIDE`` actions per ``next()``, so the harness renders an observation after every ``stride`` env steps, and
keeps each one: every camera as an image, the official state, and the dev-only GT object rows (scorer side: mode gt
puts them under ``DEV_GT_KEY``; nothing here is a policy).

    DOJO_SEQ_SRC=<run dir with *_actions.json.gz> DOJO_SEQ_OUT=<dir> [DOJO_SEQ_STRIDE=1] [DOJO_SEQ_FMT=jpg|png]
    scripts/dojo/integ_eval.sh --task T --layouts L.json --mode gt --verify off \
        --policy dojo.perception.seq_capture:SeqReplayPolicy

Per episode ``<out>/<task>_<lid>/``: ``frames/<step:04d>_<cam>.<fmt>`` (jpg: quality ``DOJO_SEQ_JPG_Q`` 95, 4:4:4,
~0.3 MB per step for 3 cameras against 1.1 MB as PNG; frame 0 also as PNG) and ``steps.jsonl`` (one row per observation:
step, state, GT rows, the gripper commands of the actions that led to it). The physics is the recorded run's (G3 /
the proxy are bit-deterministic on the same actions); the eval row's ``dev_gt_probe`` of the replay can be compared
with the recorded run's.
"""
from __future__ import annotations

import glob
import gzip
import io
import json
import os
from pathlib import Path

import numpy as np

DEV_GT_KEY = "_dev_gt"


def find_actions(src: str, task: str, lid) -> str | None:
    """The recorded run's executed actions for (task, layout id) under ``src`` (a run dir or a file)."""
    if os.path.isfile(src):
        return src
    hits = sorted(glob.glob(os.path.join(src, f"{task}_*_{lid}_actions.json.gz")))
    return hits[0] if hits else None


def load_actions(path: str) -> list[dict]:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)["actions"]


def encode_image(im: np.ndarray, fmt: str = "png", quality: int = 95) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    if fmt == "jpg":
        Image.fromarray(np.asarray(im, np.uint8)).save(buf, format="JPEG", quality=int(quality), subsampling=0)
    else:
        Image.fromarray(np.asarray(im, np.uint8)).save(buf, format="PNG", compress_level=3)
    return buf.getvalue()


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


class SeqReplayPolicy:
    """Replays a recorded run's actions ``stride`` at a time and keeps every observation (see the module doc)."""

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.src = os.environ.get("DOJO_SEQ_SRC", "")
        self.out = Path(os.environ.get("DOJO_SEQ_OUT", "seq_out"))
        self.stride = max(1, int(os.environ.get("DOJO_SEQ_STRIDE", "1")))
        self.fmt = os.environ.get("DOJO_SEQ_FMT", "jpg")
        self.quality = int(os.environ.get("DOJO_SEQ_JPG_Q", "95"))
        self.max_steps = int(os.environ.get("DOJO_SEQ_MAX_STEPS", "0"))
        self.report: dict = {}

    def reset(self, info: dict) -> None:
        task, lid = info.get("task"), info.get("layout_id")
        path = find_actions(self.src, task, lid)
        if path is None:
            raise FileNotFoundError(f"no recorded actions for {task} {lid} under {self.src!r}")
        self.actions = load_actions(path)
        if self.max_steps:
            self.actions = self.actions[: self.max_steps]
        self.i = 0
        self.dir = self.out / (os.environ.get("DOJO_SEQ_NAME") or f"{task}_{lid}")   # one name per recorded run
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        self._rows = open(self.dir / "steps.jsonl", "w")
        self._last_acts: list[dict] = []
        (self.dir / "meta.json").write_text(json.dumps({"task": task, "layout_id": lid, "actions": path,
                                                         "n_actions": len(self.actions), "stride": self.stride,
                                                         "fmt": self.fmt, "instruction": info.get("instruction"),
                                                         "step_lim": info.get("step_lim")}, indent=1))
        self.report = {"timings": {}, "events": [], "history": [], "records": {}, "looks": [], "result": None,
                       "error": None, "seq": {"dir": str(self.dir), "n_actions": len(self.actions)}}

    def _keep(self, obs: dict) -> None:
        step = int(obs["steps_used"])
        for cam, im in obs["images"].items():
            (self.dir / "frames" / f"{step:04d}_{cam}.{self.fmt}").write_bytes(encode_image(im, self.fmt, self.quality))
            if step == 0 and self.fmt != "png":            # frame 0 lossless too (the runtime's own frame-0 estimate)
                (self.dir / "frames" / f"{step:04d}_{cam}.png").write_bytes(encode_image(im, "png"))
        grip = [{k: float(np.asarray(v).reshape(-1)[0]) for k, v in a.items() if k.endswith("_ee_joint_state")}
                for a in self._last_acts]
        row = {"step": step, "state": _jsonable(obs["state"]), "gt": _jsonable(obs.get(DEV_GT_KEY) or []),
               "grip_cmds": grip}
        self._rows.write(json.dumps(row) + "\n")
        self._rows.flush()

    def next(self, obs: dict) -> dict:
        self._keep(obs)
        if self.i >= len(self.actions):
            self._rows.close()
            return {"done": True}
        acts = self.actions[self.i:self.i + self.stride]
        self.i += len(acts)
        self._last_acts = acts
        return {"actions": [{k: np.asarray(v, np.float32) for k, v in a.items()} for a in acts]}

    def episode_report(self) -> dict:
        try:
            self._rows.close()
        except Exception:  # noqa: BLE001
            pass
        self.report["seq"]["replayed"] = getattr(self, "i", 0)
        return self.report


__all__ = ["SeqReplayPolicy", "find_actions", "load_actions", "encode_image"]
