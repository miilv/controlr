"""The FK agent loop: per step, LLM -> joint target -> constraints -> arm -> photo -> back to start.

Why joint targets (not TCP targets + IK): the idea under test is that a vision LLM, given the
history of {joints, motion, photo} and FK numbers, can itself work out the joint angles that put
the gripper at the object — forward kinematics, as a human with a teleop leader does. The code
only checks and executes (FK for the constraints, IK only for the fixed "approach from above"
and "lift" segments). The reply also carries ``expected_tcp_mm`` so the logs show how well the
model's own kinematics match FK.

Arm access: ``ServerArm`` speaks the SO-101 server protocol directly (``execute`` / ``observe`` /
``move_joints``), converting MODEL angles to the server's LeRobot DEGREES with the joint offsets
exactly like ``So101Robot`` does, with the same stop thresholds. Nothing here talks to the
hardware except through the server, so the fake server (``server.py --fake``) runs the full loop.

Episode log (``<log_dir>/``): ``config.json``, ``steps.jsonl`` (one line per step: decision,
plan, execution, LLM usage), ``images/step_NN.jpg``, ``prompts/step_NN.json`` (the request with
images as labels), ``summary.json``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from controlr.fkagent import geometry as G
from controlr.fkagent import prompt as Pm

# ----------------------------------------------------------------------------------- arm


class ServerArm:
    """Thin client of ``controlr/robot/so101/server.py`` in MODEL degrees."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7811, *, offsets=G.JOINT_OFFSETS_DEG,
                 tcp_speed_mm_s: float = 40.0, track_stop_deg: float = 20.0,
                 current_stop_ma=(400.0, 600.0, 600.0, 400.0, 300.0), stop_ticks: int = 3,
                 jpeg_quality: int = 90, conn=None, connect_timeout_s: float = 10.0) -> None:
        from controlr.robot.so101 import protocol as P

        self.P = P
        self.offsets = np.asarray(offsets, float)
        self.tcp_speed = float(tcp_speed_mm_s)
        self.stop = {"track_stop_rad": math.radians(track_stop_deg),
                     "current_stop_ma": list(current_stop_ma) if current_stop_ma else None,
                     "stop_ticks": int(stop_ticks)}
        self.jpeg_quality = jpeg_quality
        if conn is None:
            from multiprocessing.connection import Client

            deadline = time.monotonic() + connect_timeout_s
            while True:
                try:
                    conn = Client((host, int(port)), authkey=P.authkey_from_env())
                    break
                except OSError as e:
                    if time.monotonic() > deadline:
                        raise ConnectionError(f"no SO-101 server at {host}:{port}: {e}") from e
                    time.sleep(0.3)
        self.conn = conn
        self.info = self._call("info")
        self.fake = bool(self.info.get("fake"))

    def _call(self, op: str, **args):
        self.conn.send(self.P.request(op, **args))
        return self.P.unwrap(self.conn.recv())

    def limits(self, **kw) -> G.Limits:
        return G.Limits.from_server_limits(self.info["limits_rad"], self.offsets, **kw)

    def _model(self, q_rad) -> np.ndarray:
        return np.degrees(np.asarray(q_rad, float)) + self.offsets

    def _server(self, q_deg) -> np.ndarray:
        return np.radians(np.asarray(q_deg, float) - self.offsets)

    def state(self) -> tuple[np.ndarray, float]:
        s = self._call("state")
        return self._model(s["q"]), float(s["gripper"])

    def observe(self) -> tuple[bytes, np.ndarray, float]:
        r = self._call("observe", jpeg_quality=self.jpeg_quality)
        return bytes(r["jpeg"]), self._model(r["state"]["q"]), float(r["state"]["gripper"])

    def _result(self, r: dict) -> dict:
        return {"q_deg": self._model(r["state"]["q"]).round(2).tolist(), "gripper": float(r["state"]["gripper"]),
                "stopped": bool(r.get("stopped")), "stop_reason": r.get("stop_reason") or "",
                "stop_joint": r.get("stop_joint") or "", "gripper_blocked": bool(r.get("gripper_blocked")),
                "wall_s": round(float(r.get("wall_s", 0.0)), 2),
                "peak_current_ma": [round(float(v)) for v in np.asarray(r.get("peak_current_ma", [])).ravel()]}

    def move(self, rows_deg, gripper: float | None = None) -> dict:
        """Follow joint waypoints (the first row = where the arm is; it is skipped)."""
        rows = [np.asarray(r, float) for r in rows_deg]
        pts = [G.tcp_mm(r) for r in rows]
        length = sum(float(np.linalg.norm(b - a)) for a, b in zip(pts[:-1], pts[1:]))
        dur = length / max(self.tcp_speed, 1.0)
        q = np.array([self._server(r) for r in rows[1:]]) if len(rows) > 1 else np.array([self._server(rows[0])])
        g = np.full(len(q), math.nan)
        if gripper is not None:
            g[:] = float(gripper)
        return self._result(self._call("execute", q=q, gripper=g, duration_s=dur, stop=self.stop))

    def set_gripper(self, pct: float) -> dict:
        q, _ = self.state()
        return self._result(self._call("execute", q=np.array([self._server(q)]), gripper=np.array([float(pct)]),
                                        duration_s=0.0, stop=self.stop))

    def move_joints(self, q_deg, gripper: float | None = None, duration_s: float = 4.0) -> dict:
        return self._result(self._call("move_joints", q=self._server(q_deg), gripper=gripper,
                                        duration_s=float(duration_s)))

    def close(self) -> None:
        try:
            self._call("close")
        except Exception:  # noqa: BLE001
            pass
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001
            pass


# ----------------------------------------------------------------------------------- agent

@dataclass
class AgentConfig:
    task: str = "pick up the object on the table"
    model: str = "claude/claude-opus-5-5"
    max_steps: int = 6
    start_q_deg: tuple = G.SURVEY_Q_DEG
    gripper_open_default: float = 35.0
    gripper_open_max: float = 60.0
    gripper_open_min: float = 10.0
    close_pct: float = 0.0
    grasp_margin: float = 2.5          # gripper % above the close command = something between the jaws
    lift_dz_mm: float = 50.0
    long_edge: int = 640
    jacobian: bool = True
    max_tokens: int = 16000
    reply_retries: int = 1
    release_at_end: bool = False       # open the jaws at the start pose after a grasp
    reset_duration_s: float = 5.0
    object_xyz_mm: tuple | None = None # ground truth for the metrics (operator-measured), optional
    fewshot: list = field(default_factory=list)   # sessions (dicts) shown as examples
    fewshot_root: str = ""


def llm_text_usage(res) -> dict:
    u = res.usage
    return {"error": res.error, "finish": res.finish_reason, "t_wall": round(res.timings.t_wall or 0.0, 2),
            "prompt_tokens": getattr(u, "prompt_tokens", None), "completion_tokens": getattr(u, "completion_tokens", None),
            "cache_read": getattr(u, "cache_read_tokens", None), "request_bytes": res.request_bytes}


def ask(llm, model: str, system: str, parts: list, *, max_tokens: int, retries: int = 1,
        log=None) -> tuple[Pm.Decision | None, list[dict], str]:
    """One decision: call, parse, on a bad reply re-ask (``retries`` times) with the parse error.
    Returns (decision or None, per-call records, last reply text)."""
    calls, text, extra = [], "", []
    for attempt in range(retries + 1):
        msgs = Pm.build_messages(system, parts + extra)
        if log is not None:
            log(msgs, attempt)
        res = llm.complete(model, msgs, max_tokens=max_tokens)
        text = res.text or ""
        rec = llm_text_usage(res)
        rec["n_images"] = Pm.count_images(msgs)
        rec["reply"] = text
        calls.append(rec)
        if res.error:
            continue
        try:
            return Pm.parse_reply(text), calls, text
        except Pm.ReplyError as e:
            rec["parse_error"] = str(e)
            extra = [f"\nYour previous reply could not be used ({e}). Reply again, ending with the JSON object."]
    return None, calls, text


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, bytes):
        return f"<{len(o)} bytes>"
    raise TypeError(type(o).__name__)


def redact(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        c = m["content"]
        if isinstance(c, list):
            c = [p if p.get("type") != "image_url" else {"type": "image_url", "image_url": "<image>"} for p in c]
        out.append({"role": m["role"], "content": c})
    return out


class FkAgent:
    def __init__(self, arm, llm, cfg: AgentConfig, log_dir, *, limits: G.Limits | None = None) -> None:
        self.arm, self.llm, self.cfg = arm, llm, cfg
        self.lim = limits or arm.limits()
        self.start_q = np.asarray(cfg.start_q_deg, float)
        self.dir = Path(log_dir)
        (self.dir / "images").mkdir(parents=True, exist_ok=True)
        (self.dir / "prompts").mkdir(exist_ok=True)
        self.steps: list[Pm.Step] = []
        self.system = Pm.system_text(self.lim, max_steps=cfg.max_steps, lift_dz=cfg.lift_dz_mm)
        self._fewshot = (Pm.fewshot_parts(cfg.fewshot, cfg.fewshot_root, long_edge=cfg.long_edge)
                         if cfg.fewshot else [])
        why = G.check_path([self.start_q], self.lim)
        if why:
            raise ValueError(f"the start pose breaks the limits: {why}")
        cfg_d = {k: v for k, v in asdict(cfg).items() if k != "fewshot"}
        cfg_d["fewshot_sessions"] = [s["session_id"] for s in cfg.fewshot]
        cfg_d["limits"] = self.lim.to_dict()
        cfg_d["arm_fake"] = bool(getattr(arm, "fake", False))
        (self.dir / "config.json").write_text(json.dumps(cfg_d, indent=1, default=_jsonable))

    # --------------------------------------------------------------- helpers
    def _log(self, rec: dict) -> None:
        with open(self.dir / "steps.jsonl", "a") as f:
            f.write(json.dumps(rec, default=_jsonable) + "\n")

    def _save_photo(self, idx: int, jpeg: bytes | None, tag: str = "") -> str | None:
        if jpeg is None:
            return None
        name = f"images/step_{idx:02d}{tag}.jpg"
        (self.dir / name).write_bytes(jpeg)
        return name

    def _go_home(self, candidates: list[list], gripper: float | None) -> dict | None:
        """Back to the start pose along the first candidate path (current pose first) that passes
        the checks; None (the arm stays where it is) if none does."""
        for rows in candidates:
            rows = [np.asarray(r, float) for r in rows]
            if len(rows) > 1 and not G.check_path(rows, self.lim):
                return self.arm.move(rows, gripper)
        return None

    def _open_pct(self, d: Pm.Decision) -> float:
        g = self.cfg.gripper_open_default if d.gripper_open_pct is None else d.gripper_open_pct
        return float(np.clip(g, self.cfg.gripper_open_min, self.cfg.gripper_open_max))

    # --------------------------------------------------------------- episode
    def reset(self) -> None:
        r = self.arm.move_joints(self.start_q, self.cfg.gripper_open_default, self.cfg.reset_duration_s)
        if r["stopped"]:
            raise RuntimeError(f"blocked while moving to the start pose: {r['stop_joint']} {r['stop_reason']}")
        jpeg, q, g = self.arm.observe()
        st = Pm.Step(0, "observe", q.round(2).tolist(), g, jpeg)
        self.steps.append(st)
        self._log({"step": 0, "action": "observe", "q_deg": st.q_deg, "gripper": g,
                   "photo": self._save_photo(0, jpeg), **G.pose_summary(q)})

    def run(self) -> dict:
        t0 = time.monotonic()
        if not self.steps:
            self.reset()
        outcome, prev_target, calls_total = "max_steps", None, 0
        last_tcp = None
        for k in range(1, self.cfg.max_steps + 1):
            parts = self._fewshot + Pm.history_parts(self.cfg.task, self.steps, long_edge=self.cfg.long_edge,
                                                     jacobian=self.cfg.jacobian, max_steps=self.cfg.max_steps)

            def dump(msgs, attempt, k=k):
                (self.dir / "prompts" / f"step_{k:02d}_{attempt}.json").write_text(json.dumps(redact(msgs), indent=1))

            d, calls, _ = ask(self.llm, self.cfg.model, self.system, parts, max_tokens=self.cfg.max_tokens,
                              retries=self.cfg.reply_retries, log=dump)
            calls_total += len(calls)
            rec: dict = {"step": k, "llm": calls}
            if d is None:
                rec["action"] = "no_decision"
                self._log(rec)
                outcome = "no_decision"
                break
            rec["decision"] = d.as_dict()
            est = {"xyz_mm": d.object_xyz_mm, "uncertainty_mm": d.uncertainty_mm, "visible": d.visible}
            if d.action == "give_up":
                rec["action"] = "give_up"
                self._log(rec)
                outcome = "give_up"
                break
            plan = G.plan_target(self.start_q, d.q_deg, self.lim, prev_target)
            rec["plan"] = plan.as_dict()
            if not plan.ok:
                st = Pm.Step(k, "rejected", None, requested_q_deg=d.q_deg,
                             notes=[f"target refused, the arm did not move: {plan.reason}"] + plan.notes, estimate=est)
                self.steps.append(st)
                rec["action"] = "rejected"
                self._log(rec)
                continue
            if d.expected_tcp_mm is not None:
                rec["expected_vs_fk_mm"] = round(float(np.linalg.norm(
                    np.subtract(d.expected_tcp_mm, G.tcp_mm(d.q_deg)))), 1)
            opened = self._open_pct(d)
            res = self.arm.move(plan.approach, opened)
            rec["move"] = res
            jpeg, q, g = self.arm.observe()
            notes = list(plan.notes)
            result = ""
            back = [[q] + list(reversed(plan.approach))[1:]]
            stopped = res["stopped"]
            if stopped:
                notes.append(f"the arm hit something ({res['stop_joint']}: {res['stop_reason']}) and stopped")
            grasped = False
            photo2 = None
            if d.action == "grip" and not stopped:
                cr = self.arm.set_gripper(self.cfg.close_pct)
                rec["close"] = cr
                holding = cr["gripper_blocked"] or cr["gripper"] > self.cfg.close_pct + self.cfg.grasp_margin
                if holding:
                    up = G.lift_path(q, self.cfg.lift_dz_mm, self.lim)
                    if up and not G.check_path(up, self.lim):
                        lr = self.arm.move(up)
                        rec["lift"] = lr
                        stopped = lr["stopped"]
                    photo2, q2, g2 = self.arm.observe()
                    grasped = g2 > self.cfg.close_pct + self.cfg.grasp_margin
                    rec["after_lift"] = {"q_deg": q2.round(2).tolist(), "gripper": g2, **G.pose_summary(q2)}
                    result = (f"GRASPED: the jaws stalled at {cr['gripper']:.0f} % and still hold after the lift"
                              if grasped else f"the jaws stalled at {cr['gripper']:.0f} % but the object slipped "
                                              f"out during the lift (gripper {g2:.0f} %)")
                    above = [plan.approach[1]] if len(plan.approach) > 2 else []
                    back = [[q2, self.start_q], [q2] + above + [self.start_q]]
                else:
                    result = f"the jaws closed EMPTY (gripper {cr['gripper']:.0f} %): nothing between the jaw tips"
                    self.arm.set_gripper(opened)
            st = Pm.Step(k, d.action, q.round(2).tolist(), g, jpeg, requested_q_deg=d.q_deg,
                         executed_q_deg=plan.target_q_deg.round(2).tolist(),
                         motion_deg=(plan.target_q_deg - self.start_q).round(2).tolist(), notes=notes,
                         result=result, estimate=est)
            self.steps.append(st)
            last_tcp = G.tcp_mm(q)
            rec.update({"action": d.action, "reached_q_deg": st.q_deg, "gripper": g, **G.pose_summary(q),
                        "photo": self._save_photo(k, jpeg), "photo_after_lift": self._save_photo(k, photo2, "_lift"),
                        "result": result, "notes": notes})
            if self.cfg.object_xyz_mm is not None:
                rec["tcp_to_object_mm"] = round(float(np.linalg.norm(last_tcp - np.asarray(self.cfg.object_xyz_mm))), 1)
                if d.object_xyz_mm is not None:
                    rec["estimate_error_mm"] = round(float(np.linalg.norm(
                        np.subtract(d.object_xyz_mm, self.cfg.object_xyz_mm))), 1)
            if stopped:
                rec["home"] = None
                self._log(rec)
                outcome = "stopped"
                break
            rec["home"] = self._go_home(back, None if grasped else opened)
            self._log(rec)
            if rec["home"] is None or rec["home"]["stopped"]:
                outcome = "stuck"          # no safe way back: the operator takes over
                break
            prev_target = plan.target_q_deg
            if grasped:
                outcome = "grasped"
                break
        if outcome == "grasped" and self.cfg.release_at_end:
            self.arm.set_gripper(self.cfg.gripper_open_default)
        n_motion = sum(1 for s in self.steps if s.action in ("move", "grip"))
        summary = {"outcome": outcome, "success": outcome == "grasped",
                   "steps": len(self.steps) - 1, "motions": n_motion, "llm_calls": calls_total,
                   "grips": sum(1 for s in self.steps if s.action == "grip"),
                   "final_tcp_mm": None if last_tcp is None else last_tcp.round(1).tolist(),
                   "wall_s": round(time.monotonic() - t0, 1), "fewshot": len(self.cfg.fewshot)}
        if self.cfg.object_xyz_mm is not None and last_tcp is not None:
            summary["final_tcp_to_object_mm"] = round(float(np.linalg.norm(last_tcp - np.asarray(self.cfg.object_xyz_mm))), 1)
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=1, default=_jsonable))
        return summary
