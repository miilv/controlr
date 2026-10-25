"""The hybrid agent loop: one LLM conversation per episode, each reply one code cell over ``robo``.

Transcript: ``controlr.llm.transcript.Transcript`` (append-only; images JPEG-encoded once and kept as bytes;
cache markers applied per request on Claude routes, the stored transcript never carries them). Turn 0 =
``RUN <nonce>`` + instruction + scene text + the head frame. Every later user turn = the previous cell's result
(deterministic text) + its images. Reasoning effort / extra_body never change within an episode (cache rule).

End: ``finish()`` in a cell, the episode ending (step budget / RoboDojo's success latch), ``max_calls`` replies,
or ``max_nudges`` replies in a row without a code block. The caller runs ``finish()`` if the model never did.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from controlr.llm import Transcript, apply_cache_markers, cache_style_for, encode_image
from dojo.agent.llmio import call_record
from dojo.agent.manual import render_manual
from dojo.agent.robo import Robo
from dojo.agent.session import Session

CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


@dataclass
class AgentConfig:
    model: str = "cx/gpt-6-astra-high"
    extra_body: dict = field(default_factory=dict)
    max_tokens: int = 16000
    max_calls: int = 30
    max_nudges: int = 2
    max_cell_steps: int = 250
    max_cell_contacts: int = 4
    auto_frames: bool = True
    image_px: int = 448              # longer side of attached frames
    first_image_px: int = 640
    jpeg_quality: int = 80
    max_images_per_turn: int = 3
    manual: str = "manual_v2"
    appendix: str = "agent_v2"
    timeout_s: float = 600.0


def system_text(cfg: AgentConfig) -> str:
    app = (Path(__file__).with_name("prompts") / f"{cfg.appendix}.md").read_text()
    app = (app.replace("{max_cell_steps}", str(cfg.max_cell_steps))
           .replace("{max_cell_contacts}", str(cfg.max_cell_contacts)).replace("{max_calls}", str(cfg.max_calls)))
    return render_manual(cfg.manual) + app


def _resize(rgb: np.ndarray, px: int) -> np.ndarray:
    from PIL import Image
    h, w = rgb.shape[:2]
    s = px / max(h, w)
    if s >= 1:
        return rgb
    return np.asarray(Image.fromarray(np.asarray(rgb, np.uint8)).resize((int(round(w * s)), int(round(h * s))),
                                                                        Image.BILINEAR))


def extract_code(text: str) -> str | None:
    m = CODE_RE.findall(text or "")
    return m[-1] if m else None


def format_result(cr, receipts: list[str], state_line: str) -> str:
    out = []
    if receipts:
        out.append("ACTIONS:\n" + "\n".join(f"- {r}" for r in receipts))
    if cr.stdout.strip():
        out.append("STDOUT:\n" + cr.stdout.rstrip())
    if cr.value is not None:
        out.append("VALUE: " + cr.value)
    if cr.error:
        out.append("ERROR: " + cr.error)
    if cr.checkpoint:
        out.append("CHECKPOINT: " + cr.checkpoint + " (the rest of the cell did not run)")
    if not out:
        out.append("(the cell ran; no output)")
    out.append(state_line)
    return "\n".join(out)


class AgentLoop:
    def __init__(self, cfg: AgentConfig, client, robo: Robo, log_dir: Path | None = None, log=print):
        self.cfg, self.client, self.robo = cfg, client, robo
        self.log_dir = Path(log_dir) if log_dir else None
        self.logf = log
        self.calls: list[dict] = []
        self.cells: list[dict] = []
        self.transcript: Transcript | None = None
        self.on_call = None                 # callback(loop) after every LLM call (the policy keeps its report live)

    def _state_line(self) -> str:
        try:
            s = self.robo.state()
            arm = lambda a: f"{'open' if a.gripper > 0.5 else 'closed'}" + (f" holding {a.holding}" if a.holding else "")  # noqa: E731
            return f"STATE: step {s.step}/{s.step_lim} ({s.steps_left} left); left {arm(s.left)}; right {arm(s.right)}"
        except Exception as e:  # noqa: BLE001
            return f"STATE: unavailable ({type(e).__name__})"

    def _images(self, imgs) -> list:
        parts = []
        for label, rgb in imgs[-self.cfg.max_images_per_turn:]:
            parts.append(f"image: {label}")
            parts.append(encode_image(_resize(rgb, self.cfg.image_px), self.cfg.jpeg_quality, label))
        return parts

    def run(self, instruction: str, scene: str, frame0: np.ndarray | None) -> dict:
        cfg = self.cfg
        tr = self.transcript = Transcript(system_text(cfg))
        session = Session({"robo": self.robo, "finish": self.robo.finish})
        nonce = uuid.uuid4().hex[:12]
        parts = [f"RUN {nonce}\nINSTRUCTION: {instruction}\n\n{scene}"]
        if frame0 is not None:
            parts += ["image: cam_head at the start", encode_image(_resize(frame0, cfg.first_image_px),
                                                                   cfg.jpeg_quality, "cam_head start")]
        tr.add_user(parts)
        style = cache_style_for(cfg.model, "auto")
        nudges, outcome = 0, "max_calls"
        for i in range(cfg.max_calls):
            msgs = apply_cache_markers(tr.to_messages(), style)
            res = self.client.complete(cfg.model, msgs, max_tokens=cfg.max_tokens, extra_body=cfg.extra_body or None,
                                       timeout_s=cfg.timeout_s)
            rec = call_record(res, cfg.model)
            rec["i"] = i
            self.calls.append(rec)
            if self.on_call is not None:
                self.on_call(self)
            if res.error or not res.text:
                self.logf(f"[agent] call {i}: error {res.error}")
                if res.error and res.http_status and 400 <= res.http_status < 500 and res.http_status != 429:
                    outcome = "llm_error"
                    break
                tr.add_assistant(res.text or "")
                tr.add_user(["(no reply received; send the next cell)"])
                continue
            tr.add_assistant(res.text)
            code = extract_code(res.text)
            if code is None:
                nudges += 1
                if nudges > cfg.max_nudges:
                    outcome = "no_code"
                    break
                tr.add_user(["Your reply had no ```python``` block. Reply with one code cell; call finish() to end."])
                continue
            nudges = 0
            self.robo.begin_cell()
            t0 = time.perf_counter()
            cr = session.run(code)
            receipts, imgs = self.robo.drain()
            cell = {"i": i, "code": code, "stdout": cr.stdout, "value": cr.value, "error": cr.error,
                    "checkpoint": cr.checkpoint, "finished": cr.finished, "receipts": receipts,
                    "n_images": len(imgs), "cell_s": round(time.perf_counter() - t0, 2),
                    "step": self.robo.ctx.steps_used()}
            self.cells.append(cell)
            self._write_logs()
            self.logf(f"[agent] cell {i}: {len(receipts)} actions, step {cell['step']}"
                      + (f", error {cr.error[:80]}" if cr.error else "") + (", FINISH" if cr.finished else ""))
            if cr.finished:
                outcome = "finished" if self.robo.finished is not None else "episode_over"
                break
            tr.add_user([format_result(cr, receipts, self._state_line())] + self._images(imgs))
        self._write_logs()
        return {"outcome": outcome, "calls": len(self.calls), "cells": len(self.cells),
                "finish": self.robo.finished, "usage": self.usage()}

    def usage(self) -> dict:
        def s(k):
            return int(sum(c.get(k) or 0 for c in self.calls))
        return {"calls": len(self.calls), "errors": sum(1 for c in self.calls if not c["ok"]),
                "prompt_tokens": s("prompt_tokens"), "cached_tokens": s("cached_tokens"),
                "completion_tokens": s("completion_tokens"), "reasoning_tokens": s("reasoning_tokens"),
                "llm_s": round(sum(c.get("t_wall") or 0 for c in self.calls), 1)}

    def _write_logs(self) -> None:
        if self.log_dir is None or self.transcript is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        imgdir = self.log_dir / "images"
        imgdir.mkdir(exist_ok=True)
        for im in self.transcript.images():
            p = imgdir / f"{im.sha}.jpg"
            if not p.exists():
                p.write_bytes(im.jpeg)
        with (self.log_dir / "messages.jsonl").open("w") as f:
            for m in self.transcript.to_log_records():
                f.write(json.dumps(m) + "\n")
        with (self.log_dir / "calls.jsonl").open("w") as f:
            for c in self.calls:
                f.write(json.dumps(c) + "\n")
        with (self.log_dir / "cells.jsonl").open("w") as f:
            for c in self.cells:
                f.write(json.dumps(c, default=str) + "\n")
        (self.log_dir / "agent_config.json").write_text(json.dumps(asdict(self.cfg), indent=1))


__all__ = ["AgentConfig", "AgentLoop", "extract_code", "system_text", "format_result"]
