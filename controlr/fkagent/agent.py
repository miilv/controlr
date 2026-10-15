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
        self.timing: dict = {}                 # seconds per server op (execute = motion, observe, state, ...)
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
        rq = self.info.get("rest_q")
        self.rest_q_deg = None if rq is None else self._model(rq)

    def _call(self, op: str, **args):
        t0 = time.monotonic()
        self.conn.send(self.P.request(op, **args))
        try:
            return self.P.unwrap(self.conn.recv())
        finally:
            self.timing[op] = self.timing.get(op, 0.0) + (time.monotonic() - t0)

    def timing_since(self, mark: dict) -> dict:
        return {k: round(v - mark.get(k, 0.0), 2) for k, v in self.timing.items() if v - mark.get(k, 0.0) > 0.005}

    def limits(self, **kw) -> G.Limits:
        return G.Limits.from_server_limits(self.info["limits_rad"], self.offsets, **kw)

    def _model(self, q_rad) -> np.ndarray:
        return np.degrees(np.asarray(q_rad, float)) + self.offsets

    def _server(self, q_deg) -> np.ndarray:
        return np.radians(np.asarray(q_deg, float) - self.offsets)

    def state(self) -> tuple[np.ndarray, float]:
        s = self._call("state")
        self.last_temp_c = float(np.nanmax(np.asarray(s.get("temp_c") or [np.nan], float))) \
            if s.get("temp_c") is not None else None
        return self._model(s["q"]), float(s["gripper"])

    def max_temp_c(self) -> float | None:
        self.state()
        return self.last_temp_c

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

    def hold(self) -> None:
        self._call("hold")

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
    gripper_open_default: float = 30.0
    gripper_open_max: float = 35.0     # wider: the moving jaw can knock the wrist camera (operator, 2026-10-15)
    gripper_open_min: float = 10.0
    close_pct: float = 0.0
    grasp_margin: float = 2.5          # gripper % above the close command = something between the jaws
    lift_dz_mm: float = 50.0
    long_edge: int = 640
    fewshot_photos: int = 2            # per example session (the last ones: pre_grasp/grasp/lift...)
    fewshot_edge: int = 320
    history_full_photos: int = 3       # the last N states' photos at long_edge, older ones at 320 px
    jacobian: bool = True
    max_tokens: int = 16000
    extra_body: dict = field(default_factory=dict)   # e.g. {"reasoning_effort": "low"}
    fewshot_stream_stop: bool = True   # close the stream once the JSON object is complete
    reply_retries: int = 1
    release_at_end: bool = False       # open the jaws at the start pose after a grasp
    place_back: bool = True            # (return loop) after a successful lift: put the object back down
    max_temp_c: float = 55.0           # any motor hotter than this: stop the episode
    fold: bool = True                  # start and end every episode in the server's rest (folded) pose,
                                       # rest <-> start along path-checked moves
    carry_min_z_mm: float | None = None  # while holding, the TCP stays this high along every path (e.g. over a bowl rim)
    max_replans: int = 3               # plan_once: re-plans after a failed stage check
    # pixel loop (loop="pixel"): the model points, code computes every pose
    camera: object = None              # visual.Camera (None: stock mount)
    object_z_mm: float = 40.0          # height of the object's visible centre (ray-plane intersection)
    grasp_z_mm: float = 18.0           # TCP height at the grasp
    target_z_mm: float = 0.0           # height of the place target's top (table: 0)
    release_dz_mm: float = 40.0        # release this far above the target's top
    look_z_mm: float = 120.0           # look-from-above pose height before grasping
    photo_size: tuple = (640, 360)     # what the model sees (pixels are scaled to the camera's size)
    loop: str = "return"               # return: back to the start pose after every action (the first design);
                                       # continuous: re-plan from the current pose, start pose only when done
    place_xyz_mm: tuple | None = None  # continuous: where the task asks the object to be put (metrics)
    revisit_mm: float = 10.0           # continuous: a target this close to an earlier one is refused
    place_tol_mm: float = 30.0         # success = placed within this of place_xyz_mm (when given)
    max_boundary_clamps: int = 2       # consecutive targets pulled back at the workspace edge -> stop
    reset_duration_s: float = 5.0
    object_xyz_mm: tuple | None = None # ground truth for the metrics (operator-measured), optional
    fewshot: list = field(default_factory=list)   # sessions (dicts) shown as examples
    fewshot_root: str = ""


def llm_text_usage(res) -> dict:
    u = res.usage
    return {"error": res.error, "finish": res.finish_reason, "t_wall": round(res.timings.t_wall or 0.0, 2),
            "ttft": None if res.timings.ttft is None else round(res.timings.ttft, 2),
            "t_complete": None if res.timings.t_complete is None else round(res.timings.t_complete, 2),
            "reasoning_tokens": getattr(u, "reasoning_tokens", None), "stopped_early": res.stopped_early,
            "prompt_tokens": getattr(u, "prompt_tokens", None), "completion_tokens": getattr(u, "completion_tokens", None),
            "cache_read": getattr(u, "cache_read_tokens", None), "request_bytes": res.request_bytes}


def ask(llm, model: str, system: str, parts: list, *, max_tokens: int, retries: int = 1,
        log=None, extra_body: dict | None = None, early_stop: bool = True) -> tuple[Pm.Decision | None, list[dict], str]:
    """One decision: call, parse, on a bad reply re-ask (``retries`` times) with the parse error.
    Returns (decision or None, per-call records, last reply text)."""
    calls, text, extra = [], "", []
    for attempt in range(retries + 1):
        msgs = Pm.build_messages(system, parts + extra)
        if log is not None:
            log(msgs, attempt)
        res = llm.complete(model, msgs, max_tokens=max_tokens, extra_body=extra_body or None,
                           stop_when=Pm.json_complete if early_stop else None)
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
        self.system = Pm.system_text(self.lim, max_steps=cfg.max_steps, lift_dz=cfg.lift_dz_mm,
                                     gripper_min=cfg.gripper_open_min, gripper_max=cfg.gripper_open_max,
                                     loop=cfg.loop, revisit_mm=cfg.revisit_mm, max_replans=cfg.max_replans)
        self._fewshot = (Pm.fewshot_parts(cfg.fewshot, cfg.fewshot_root, long_edge=cfg.fewshot_edge,
                                          max_photos=cfg.fewshot_photos)
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
        tm = getattr(self.arm, "timing", None)
        if tm is not None:
            rec.setdefault("arm_time_s", self.arm.timing_since(getattr(self, "_tmark", {})))
            self._tmark = dict(tm)
        rec.setdefault("llm_time_s", round(sum(c.get("t_wall") or 0.0 for c in rec.get("llm", [])), 1))
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

    def _lim_for(self, holding: bool, cur_q=None) -> G.Limits:
        """While holding (``carry_min_z_mm``): a raised floor, never above the current height."""
        if not holding or self.cfg.carry_min_z_mm is None:
            return self.lim
        import dataclasses

        z = self.cfg.carry_min_z_mm
        if cur_q is not None:
            z = min(z, float(G.tcp_mm(cur_q)[2]) - 1.0)
        lo = (self.lim.box_lo_mm[0], self.lim.box_lo_mm[1], max(self.lim.box_lo_mm[2], z))
        return dataclasses.replace(self.lim, box_lo_mm=lo)

    def _too_hot(self) -> str:
        f = getattr(self.arm, "max_temp_c", None)
        t = f() if f else None
        if t is not None and t > self.cfg.max_temp_c:
            return f"motor temperature {t:.0f} C > {self.cfg.max_temp_c:.0f} C"
        return ""

    def _open_pct(self, d: Pm.Decision) -> float:
        g = self.cfg.gripper_open_default if d.gripper_open_pct is None else d.gripper_open_pct
        return float(np.clip(g, self.cfg.gripper_open_min, self.cfg.gripper_open_max))

    # --------------------------------------------------------------- episode
    def _fold_path_move(self, rows, gripper, what: str) -> dict:
        why = G.check_path(rows, self.lim)
        if why:
            raise RuntimeError(f"{what}: the path is not safe ({why}); not moving")
        r = self.arm.move(rows, gripper)
        if r["stopped"]:
            raise RuntimeError(f"{what}: blocked ({r['stop_joint']} {r['stop_reason']})")
        return r

    def reset(self) -> None:
        rest = getattr(self.arm, "rest_q_deg", None)
        if self.cfg.fold and rest is not None:
            q, _ = self.arm.state()
            if float(np.max(np.abs(q - rest))) > 3.0:          # fold first
                self._fold_path_move([q, rest], None, "folding to the rest pose")
            self._fold_path_move([np.asarray(rest, float), self.start_q], self.cfg.gripper_open_default,
                                 "rest -> start pose")
        else:
            r = self.arm.move_joints(self.start_q, self.cfg.gripper_open_default, self.cfg.reset_duration_s)
            if r["stopped"]:
                raise RuntimeError(f"blocked while moving to the start pose: {r['stop_joint']} {r['stop_reason']}")
        jpeg, q, g = self.arm.observe()
        st = Pm.Step(0, "observe", q.round(2).tolist(), g, jpeg)
        self.steps.append(st)
        self._log({"step": 0, "action": "observe", "q_deg": st.q_deg, "gripper": g,
                   "photo": self._save_photo(0, jpeg), **G.pose_summary(q)})

    def fold_end(self) -> dict:
        """End of the episode: start pose -> rest (folded), jaws closed, on a checked path."""
        rest = getattr(self.arm, "rest_q_deg", None)
        if not self.cfg.fold or rest is None:
            return {"folded": False, "why": "fold off"}
        q, _ = self.arm.state()
        rows = [q, np.asarray(rest, float)]
        why = G.check_path(rows, self.lim)
        if why and float(np.max(np.abs(q - self.start_q))) > 3.0:
            # not at the start pose: go there first if that is safe
            if not G.check_path([q, self.start_q], self.lim):
                self.arm.move([q, self.start_q], None)
                q, _ = self.arm.state()
                rows = [q, np.asarray(rest, float)]
                why = G.check_path(rows, self.lim)
        if why:
            return {"folded": False, "why": why}
        r = self.arm.move(rows, 2.0)
        return {"folded": not r["stopped"], "result": r}

    def run(self) -> dict:
        summary = self._run_any()
        if summary.get("outcome") not in ("stopped", "error", "stuck"):
            try:
                summary["fold"] = self.fold_end()
            except Exception as e:  # noqa: BLE001
                summary["fold"] = {"folded": False, "why": f"{type(e).__name__}: {e}"}
            (self.dir / "summary.json").write_text(json.dumps(summary, indent=1, default=_jsonable))
        return summary

    def _run_any(self) -> dict:
        try:
            if self.cfg.loop == "plan_once":
                return self._run_plan_once()
            if self.cfg.loop == "pixel":
                return self._run_pixel()
            return self._run_continuous() if self.cfg.loop == "continuous" else self._run()
        except Exception as e:  # noqa: BLE001  (bus / protocol error: hold where it is, report)
            try:
                self.arm.hold()
            except Exception:  # noqa: BLE001
                pass
            import traceback

            self._log({"step": len(self.steps), "action": "error", "error": f"{type(e).__name__}: {e}",
                       "traceback": traceback.format_exc()})
            summary = {"outcome": "error", "success": False, "error": f"{type(e).__name__}: {e}",
                       "steps": len(self.steps) - 1}
            (self.dir / "summary.json").write_text(json.dumps(summary, indent=1))
            return summary

    def _run(self) -> dict:
        t0 = time.monotonic()
        if not self.steps:
            self.reset()
        outcome, prev_target, calls_total = "max_steps", None, 0
        clamps = 0
        self._closest = None
        last_tcp = None
        for k in range(1, self.cfg.max_steps + 1):
            hot = self._too_hot()
            if hot:
                self._log({"step": k, "action": "overheat", "error": hot})
                outcome = "overheat"
                break
            parts = self._fewshot + Pm.history_parts(self.cfg.task, self.steps, long_edge=self.cfg.long_edge,
                                                     jacobian=self.cfg.jacobian, max_steps=self.cfg.max_steps,
                                                     full_photos=self.cfg.history_full_photos)

            def dump(msgs, attempt, k=k):
                (self.dir / "prompts" / f"step_{k:02d}_{attempt}.json").write_text(json.dumps(redact(msgs), indent=1))

            d, calls, _ = ask(self.llm, self.cfg.model, self.system, parts, max_tokens=self.cfg.max_tokens,
                              retries=self.cfg.reply_retries, log=dump,
                              extra_body=self.cfg.extra_body, early_stop=self.cfg.fewshot_stream_stop)
            calls_total += len(calls)
            rec: dict = {"step": k, "llm": calls}
            if d is None:
                rec["action"] = "no_decision"
                self._log(rec)
                outcome = "no_decision"
                break
            rec["decision"] = d.as_dict()
            est = {"xyz_mm": d.object_xyz_mm, "uncertainty_mm": d.uncertainty_mm, "visible": d.visible}
            if d.action == "wait":
                time.sleep(2.0)
                jpeg, q, g = self.arm.observe()
                self.steps.append(Pm.Step(k, "wait", q.round(2).tolist(), g, jpeg, estimate=est,
                                          result="waited at the start pose; new photo"))
                rec.update({"action": "wait", "photo": self._save_photo(k, jpeg)})
                self._log(rec)
                continue
            if d.action == "give_up":
                rec["action"] = "give_up"
                self._log(rec)
                outcome = "give_up"
                break
            plan = G.plan_target(self.start_q, d.q_deg, self.lim, prev_target)
            rec["plan"] = plan.as_dict()
            edge = (not plan.ok) or (plan.clamped and "workspace" in (plan.reason or ""))
            clamps = clamps + 1 if edge else 0
            if clamps >= self.cfg.max_boundary_clamps:
                rec["action"] = "boundary_stop"
                self._log(rec)
                outcome = "boundary"       # repeated pushes against the limits: stop
                break
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
                    lifted = False
                    if up and not G.check_path(up, self.lim):
                        lr = self.arm.move(up)
                        rec["lift"] = lr
                        stopped = lr["stopped"]
                        lifted = not stopped
                    photo2, q2, g2 = self.arm.observe()
                    grasped = g2 > self.cfg.close_pct + self.cfg.grasp_margin
                    rec["after_lift"] = {"q_deg": q2.round(2).tolist(), "gripper": g2, **G.pose_summary(q2)}
                    result = (f"GRASPED: the jaws stalled at {cr['gripper']:.0f} % and still hold after the lift"
                              if grasped else f"the jaws stalled at {cr['gripper']:.0f} % but the object slipped "
                                              f"out during the lift (gripper {g2:.0f} %)")
                    above = [plan.approach[1]] if len(plan.approach) > 2 else []
                    back = [[q2, self.start_q], [q2] + above + [self.start_q]]
                    if self.cfg.place_back and not stopped:
                        # put it back where it was (the scene stays the same for the next episode)
                        if lifted:
                            dn = self.arm.move([q2] + list(reversed(up))[1:])
                            rec["place_down"] = dn
                            stopped = dn["stopped"]
                        if not stopped:
                            rec["place_open"] = self.arm.set_gripper(opened)
                            q3, _ = self.arm.state()
                            back = [[q3] + list(reversed(plan.approach))[1:]]
                            rec["placed_back"] = True
                else:
                    result = f"the jaws closed EMPTY (gripper {cr['gripper']:.0f} %): nothing between the jaw tips"
                    self.arm.set_gripper(opened)
            st = Pm.Step(k, d.action, q.round(2).tolist(), g, jpeg, requested_q_deg=d.q_deg,
                         executed_q_deg=plan.target_q_deg.round(2).tolist(),
                         motion_deg=(plan.target_q_deg - self.start_q).round(2).tolist(), notes=notes,
                         result=result, estimate=est)
            self.steps.append(st)
            last_tcp = G.tcp_mm(q)
            if self.cfg.object_xyz_mm is not None:
                dmin = float(np.linalg.norm(last_tcp - np.asarray(self.cfg.object_xyz_mm)))
                self._closest = dmin if self._closest is None else min(self._closest, dmin)
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
            holding_now = grasped and not rec.get("placed_back")
            rec["home"] = self._go_home(back, None if holding_now else opened)
            self._log(rec)
            if rec["home"] is None or rec["home"]["stopped"]:
                outcome = "stuck"          # no safe way back: the operator takes over
                break
            prev_target = plan.target_q_deg
            if grasped:
                outcome = "grasped"
                break
        if outcome == "grasped" and self.cfg.release_at_end and not self.cfg.place_back:
            self.arm.set_gripper(self.cfg.gripper_open_default)
        n_motion = sum(1 for s in self.steps if s.action in ("move", "grip"))
        summary = {"outcome": outcome, "success": outcome == "grasped",
                   "steps": len(self.steps) - 1, "motions": n_motion, "llm_calls": calls_total,
                   "grips": sum(1 for s in self.steps if s.action == "grip"),
                   "final_tcp_mm": None if last_tcp is None else last_tcp.round(1).tolist(),
                   "wall_s": round(time.monotonic() - t0, 1), "fewshot": len(self.cfg.fewshot)}
        if self.cfg.object_xyz_mm is not None and last_tcp is not None:
            summary["closest_tcp_to_object_mm"] = None if self._closest is None else round(self._closest, 1)
            summary["final_tcp_to_object_mm"] = round(float(np.linalg.norm(last_tcp - np.asarray(self.cfg.object_xyz_mm))), 1)
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=1, default=_jsonable))
        return summary

    # ------------------------------------------------------------ continuous loop
    def _summary(self, outcome: str, t0: float, calls_total: int, last_tcp, extra: dict | None = None) -> dict:
        extra = dict(extra or {})
        perr = extra.get("place_error_xy_mm")
        ok = outcome in ("grasped", "placed") and (perr is None or perr <= self.cfg.place_tol_mm)
        summary = {"outcome": outcome, "success": ok, "loop": self.cfg.loop,
                   "steps": len(self.steps) - 1,
                   "motions": sum(1 for s in self.steps if s.action in ("move", "grip", "place", "grasp", "lift",
                                                                         "carry", "release")),
                   "llm_calls": calls_total, "grips": sum(1 for s in self.steps if s.action == "grip"),
                   "final_tcp_mm": None if last_tcp is None else np.round(last_tcp, 1).tolist(),
                   "wall_s": round(time.monotonic() - t0, 1), "fewshot": len(self.cfg.fewshot), **extra}
        if self.cfg.object_xyz_mm is not None:
            summary["closest_tcp_to_object_mm"] = None if self._closest is None else round(self._closest, 1)
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=1, default=_jsonable))
        return summary

    def _run_continuous(self) -> dict:
        t0 = time.monotonic()
        if not self.steps:
            self.reset()
            self.steps[0].holding = False
        calls_total, clamps, outcome = 0, 0, "max_steps"
        self._closest = None
        visited: list[np.ndarray] = []                 # TCPs of earlier targets (revisit rule)
        holding, grasp_q, grasped_once, last_tcp = False, None, False, None
        placed_at = None
        first = True
        for k in range(1, self.cfg.max_steps + 1):
            hot = self._too_hot()
            if hot:
                self._log({"step": k, "action": "overheat", "error": hot})
                outcome = "overheat"
                break
            cur_q, cur_g = self.arm.state()
            parts = self._fewshot + Pm.history_parts(self.cfg.task, self.steps, long_edge=self.cfg.long_edge,
                                                     jacobian=self.cfg.jacobian, max_steps=self.cfg.max_steps,
                                                     full_photos=self.cfg.history_full_photos)

            def dump(msgs, attempt, k=k):
                (self.dir / "prompts" / f"step_{k:02d}_{attempt}.json").write_text(json.dumps(redact(msgs), indent=1))

            d, calls, _ = ask(self.llm, self.cfg.model, self.system, parts, max_tokens=self.cfg.max_tokens,
                              retries=self.cfg.reply_retries, log=dump,
                              extra_body=self.cfg.extra_body, early_stop=self.cfg.fewshot_stream_stop)
            calls_total += len(calls)
            rec: dict = {"step": k, "llm": calls, "holding_before": holding,
                         "start_q_deg": cur_q.round(2).tolist()}
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
            if d.action == "wait":
                time.sleep(2.0)
                jpeg, q, g = self.arm.observe()
                self.steps.append(Pm.Step(k, "wait", q.round(2).tolist(), g, jpeg, estimate=est, holding=holding,
                                          result="waited without moving; new photo"))
                rec.update({"action": "wait", "photo": self._save_photo(k, jpeg)})
                self._log(rec)
                continue
            notes: list[str] = []

            def refuse(why: str):
                self.steps.append(Pm.Step(k, "rejected", None, requested_q_deg=d.q_deg, estimate=est,
                                          holding=holding, notes=[f"target refused, the arm did not move: {why}"]))
                rec["action"] = "rejected"
                rec["refused"] = why
                self._log(rec)

            if d.action == "place" and not holding:
                refuse("'place' needs the object in the jaws (nothing is held)")
                continue
            if d.action == "grip" and holding:
                refuse("already holding the object: 'place' it")
                continue
            in_place = d.action == "grip" and float(np.max(np.abs(np.subtract(d.q_deg, cur_q)))) < 1.0
            if not in_place:
                tgt = G.tcp_mm(d.q_deg)
                near = [i for i, v in enumerate(visited) if float(np.linalg.norm(v - tgt)) < self.cfg.revisit_mm]
                if near:
                    refuse(f"within {self.cfg.revisit_mm:.0f} mm of an earlier target (do not revisit; "
                           "change the target according to the photos)")
                    clamps += 1
                    if clamps >= self.cfg.max_boundary_clamps + 1:
                        outcome = "revisit"
                        break
                    continue
                plan = G.plan_target(cur_q, d.q_deg, self._lim_for(holding, cur_q), None if first else cur_q,
                                     path=G.continuous_path)
                rec["plan"] = plan.as_dict()
                edge = (not plan.ok) or (plan.clamped and "workspace" in (plan.reason or ""))
                clamps = clamps + 1 if edge else 0
                if clamps >= self.cfg.max_boundary_clamps:
                    rec["action"] = "boundary_stop"
                    self._log(rec)
                    outcome = "boundary"
                    break
                if not plan.ok:
                    refuse(plan.reason)
                    continue
                notes += plan.notes
                if d.expected_tcp_mm is not None:
                    rec["expected_vs_fk_mm"] = round(float(np.linalg.norm(
                        np.subtract(d.expected_tcp_mm, G.tcp_mm(d.q_deg)))), 1)
                opened = None if holding else self._open_pct(d)
                res = self.arm.move(plan.approach, opened)
                rec["move"] = res
                visited.append(G.tcp_mm(plan.target_q_deg))
                first = False
                executed = plan.target_q_deg
                if res["stopped"]:
                    notes.append(f"the arm hit something ({res['stop_joint']}: {res['stop_reason']}) and stopped")
                    rec.update({"action": d.action, "notes": notes})
                    self._log(rec)
                    outcome = "stopped"
                    break
            else:
                executed = cur_q
                opened = self._open_pct(d)
            jpeg, q, g = self.arm.observe()
            result, photo2 = "", None
            act = d.action
            if act in ("grip", "place") and not in_place and plan.clamped:
                # the guard cut the move: no closing / releasing at a point the agent did not choose
                result = (f"the move was CUT by the safety limits ({plan.reason}); the jaws were NOT "
                          f"{'closed' if act == 'grip' else 'opened'}: re-plan from this pose")
                act = "move"
            if act == "grip":
                cr = self.arm.set_gripper(self.cfg.close_pct)
                rec["close"] = cr
                if cr["stopped"]:
                    outcome = "stopped"
                if cr["gripper_blocked"] or cr["gripper"] > self.cfg.close_pct + self.cfg.grasp_margin:
                    up = G.lift_path(q, self.cfg.lift_dz_mm, self.lim)
                    if len(up) > 1 and not G.check_path(up, self.lim):
                        lr = self.arm.move(up)
                        rec["lift"] = lr
                        if lr["stopped"]:
                            outcome = "stopped"
                    photo2, q2, g2 = self.arm.observe()
                    holding = g2 > self.cfg.close_pct + self.cfg.grasp_margin
                    rec["after_lift"] = {"q_deg": q2.round(2).tolist(), "gripper": g2, **G.pose_summary(q2)}
                    if holding:
                        grasp_q, grasped_once = q.copy(), True
                        result = (f"GRASPED: the jaws stalled at {cr['gripper']:.0f} % and hold the object after "
                                  f"the {self.cfg.lift_dz_mm:.0f} mm lift; now 'place' it")
                    else:
                        result = f"the jaws stalled at {cr['gripper']:.0f} % but the object slipped out during the lift"
                        self.arm.set_gripper(opened or self.cfg.gripper_open_default)
                    q, g = q2, g2
                else:
                    result = f"the jaws closed EMPTY (gripper {cr['gripper']:.0f} %): nothing between the jaw tips; reopened"
                    self.arm.set_gripper(opened or self.cfg.gripper_open_default)
            elif act == "place":
                orr = self.arm.set_gripper(self._open_pct(d))
                rec["open"] = orr
                placed_at = G.tcp_mm(q)
                holding = False
                result = "released the object here"
            st = Pm.Step(k, act, q.round(2).tolist(), g, jpeg if photo2 is None else photo2,
                         requested_q_deg=d.q_deg, executed_q_deg=np.round(executed, 2).tolist(),
                         motion_deg=np.round(np.asarray(executed) - cur_q, 2).tolist(), notes=notes, result=result,
                         estimate=est, holding=holding)
            self.steps.append(st)
            last_tcp = G.tcp_mm(executed)
            if self.cfg.object_xyz_mm is not None and act in ("move", "grip") and not rec.get("holding_before"):
                dmin = float(np.linalg.norm(last_tcp - np.asarray(self.cfg.object_xyz_mm)))
                self._closest = dmin if self._closest is None else min(self._closest, dmin)
                rec["tcp_to_object_mm"] = round(dmin, 1)
            rec.update({"action": act, "requested_action": d.action, "reached_q_deg": np.round(q, 2).tolist(), "gripper": g,
                        **G.pose_summary(q), "photo": self._save_photo(k, jpeg),
                        "photo_after_lift": self._save_photo(k, photo2, "_lift"), "result": result, "notes": notes,
                        "holding": holding})
            self._log(rec)
            if outcome == "stopped":
                break
            if act == "place":
                up = G.lift_path(q, self.cfg.lift_dz_mm, self.lim)
                if len(up) > 1 and not G.check_path(up, self.lim):
                    self.arm.move(up, None)
                qn, _ = self.arm.state()
                home = self._go_home([[qn, self.start_q]], self.cfg.gripper_open_default)
                jpeg, _, _ = self.arm.observe()
                self._log({"step": k, "action": "home", "home": home, "photo": self._save_photo(k, jpeg, "_home")})
                outcome = "placed" if home is not None else "stuck"
                break
        extra = {"grasped": grasped_once, "placed_tcp_mm": None if placed_at is None else placed_at.round(1).tolist()}
        if placed_at is not None and self.cfg.place_xyz_mm is not None:
            extra["place_error_xy_mm"] = round(float(np.linalg.norm(placed_at[:2] - np.asarray(self.cfg.place_xyz_mm)[:2])), 1)
        if outcome not in ("placed", "stopped", "error", "stuck"):
            extra["end"] = self._finish_unplaced(holding, grasp_q)
        return self._summary(outcome, t0, calls_total, last_tcp, extra)

    def _finish_unplaced(self, holding: bool, grasp_q) -> str:
        """Task not finished: put a held object back down where it was grasped, then go to the
        start pose along checked paths. Returns what was done."""
        q, _ = self.arm.state()
        did = []
        if holding and grasp_q is not None:
            rows = G.continuous_path(q, grasp_q, self.lim)
            if not G.check_path(rows, self.lim):
                r = self.arm.move(rows, None)
                if r["stopped"]:
                    return "stopped while putting the object back"
                did.append("put the object back down")
            self.arm.set_gripper(self.cfg.gripper_open_default)
            q, _ = self.arm.state()
        up = G.lift_path(q, self.cfg.lift_dz_mm, self.lim)
        if len(up) > 1 and not G.check_path(up, self.lim):
            self.arm.move(up, None)
            q, _ = self.arm.state()
        home = self._go_home([[q, self.start_q]], None)
        did.append("returned to the start pose" if home is not None else "NO safe path home: left in place")
        return "; ".join(did)

    # ------------------------------------------------------------ plan-once loop
    def _move_split(self, target, holding: bool, gripper, first: bool, rec: dict, tag: str) -> tuple[bool, str, list]:
        """Move toward a planned target; a target beyond the per-step limits is reached in up to 4
        sub-moves (each re-checked). Any other cut (workspace, path) = the stage fails."""
        notes: list[str] = []
        moves = rec.setdefault("moves", [])
        target = np.asarray(target, float)
        for sub in range(4):
            cur, _ = self.arm.state()
            if float(np.max(np.abs(target - cur))) < 0.5:
                return True, "", notes
            plan = G.plan_target(cur, target, self._lim_for(holding, cur), None if (first and sub == 0) else cur,
                                 path=G.continuous_path)
            moves.append({"stage": tag, "sub": sub, "plan": plan.as_dict()})
            if not plan.ok:
                return False, f"refused: {plan.reason}", notes
            step_cut = plan.clamped and ("step" in plan.reason or "joint changes" in plan.reason)
            if plan.clamped and not step_cut and plan.reason:
                return False, f"cut by the guard ({plan.reason}); not executed", notes
            res = self.arm.move(plan.approach, gripper)
            moves[-1]["result"] = res
            if res["stopped"]:
                return False, f"STOP: the arm hit something ({res['stop_joint']}: {res['stop_reason']})", notes
            if not plan.clamped:
                return True, "", notes
            notes.append(f"{tag}: split at the per-step limit ({plan.reason})")
        return False, "not reached after 4 sub-moves", notes

    def _stage(self, k: int, name: str, q, g, jpeg, result: str, holding: bool, rec: dict) -> None:
        self.steps.append(Pm.Step(k, name, np.round(q, 2).tolist(), g, jpeg, result=result, holding=holding))
        rec.update({"photo": self._save_photo(k, jpeg, f"_{name}"), "reached_q_deg": np.round(q, 2).tolist(),
                    "gripper": g, **G.pose_summary(q), "result": result, "holding": holding})
        self._log(rec)

    def _ask_plan(self, holding: bool, why: str) -> tuple[Pm.StagePlan | None, list]:
        final = ("\n" + why + "\nGive the plan for the remaining stages. End with the JSON object.")
        parts = self._fewshot + Pm.history_parts(self.cfg.task, self.steps, long_edge=self.cfg.long_edge,
                                                 jacobian=self.cfg.jacobian, max_steps=self.cfg.max_steps, final=final,
                                                 full_photos=self.cfg.history_full_photos)
        calls, extra = [], []
        n = len(list((self.dir / "prompts").glob("plan_*.json")))
        for attempt in range(self.cfg.reply_retries + 1):
            msgs = Pm.build_messages(self.system, parts + extra)
            (self.dir / "prompts" / f"plan_{n:02d}_{attempt}.json").write_text(json.dumps(redact(msgs), indent=1))
            res = self.llm.complete(self.cfg.model, msgs, max_tokens=self.cfg.max_tokens,
                                    extra_body=self.cfg.extra_body or None,
                                    stop_when=Pm.json_complete if self.cfg.fewshot_stream_stop else None)
            rec = llm_text_usage(res)
            rec["reply"] = res.text or ""
            calls.append(rec)
            if res.error:
                continue
            try:
                return Pm.parse_plan(res.text, need_grasp=not holding), calls
            except Pm.ReplyError as e:
                rec["parse_error"] = str(e)
                extra = [f"\nYour previous reply could not be used ({e}). Reply again, ending with the JSON object."]
        return None, calls

    def _run_plan_once(self) -> dict:
        t0 = time.monotonic()
        if not self.steps:
            self.reset()
            self.steps[0].holding = False
        self._closest = None
        holding, grasped_once, last_tcp, placed_at = False, False, None, None
        calls_total, replans, k, waits = 0, -1, 0, 0
        outcome = "replan_budget"
        why = "PLAN the whole task now."
        first = True
        while replans < self.cfg.max_replans:
            hot = self._too_hot()
            if hot:
                self._log({"step": k, "action": "overheat", "error": hot})
                outcome = "overheat"
                break
            plan, calls = self._ask_plan(holding, why)
            calls_total += len(calls)
            replans += 1
            k += 1
            rec: dict = {"step": k, "action": "plan" if replans == 0 else "replan", "llm": calls, "why": why,
                         "holding": holding}
            if plan is None:
                rec["action"] = "no_decision"
                self._log(rec)
                outcome = "no_decision"
                break
            rec["plan"] = plan.as_dict()
            if plan.grasp_q_deg is not None:
                rec["plan_grasp_tcp_mm"] = G.tcp_mm(plan.grasp_q_deg).round(1).tolist()
            if plan.place_q_deg is not None:
                rec["plan_place_tcp_mm"] = G.tcp_mm(plan.place_q_deg).round(1).tolist()
            self._log(rec)
            if plan.action == "give_up":
                outcome = "give_up"
                break
            if plan.action == "wait":
                waits += 1
                replans -= 1
                if waits > 3:
                    outcome = "wait_limit"
                    break
                time.sleep(2.0)
                jpeg, q, g = self.arm.observe()
                self.steps.append(Pm.Step(k, "wait", np.round(q, 2).tolist(), g, jpeg, holding=holding,
                                          result="waited; new photo"))
                why = "You asked to wait; here is a new photo. PLAN the remaining stages now."
                continue
            open_pct = float(np.clip(plan.gripper_open_pct or self.cfg.gripper_open_default,
                                     self.cfg.gripper_open_min, self.cfg.gripper_open_max))
            rel_pct = float(np.clip(plan.release_open_pct or self.cfg.gripper_open_max,
                                    self.cfg.gripper_open_min, self.cfg.gripper_open_max))
            failed = ""
            # 1-2. grasp + lift
            if not holding:
                k += 1
                rec = {"step": k, "stage": "grasp", "target_q_deg": plan.grasp_q_deg}
                ok, why_m, notes = self._move_split(plan.grasp_q_deg, False, open_pct, first, rec, "grasp")
                first = False
                q, g = self.arm.state()
                last_tcp = G.tcp_mm(q)
                if self.cfg.object_xyz_mm is not None:
                    dmin = float(np.linalg.norm(last_tcp - np.asarray(self.cfg.object_xyz_mm)))
                    self._closest = dmin if self._closest is None else min(self._closest, dmin)
                    rec["tcp_to_object_mm"] = round(dmin, 1)
                rec["notes"] = notes
                if not ok:
                    jpeg, q, g = self.arm.observe()
                    rec["check"] = {"stage": "grasp", "ok": False, "why": why_m}
                    self._stage(k, "grasp", q, g, jpeg, f"CHECK FAILED (grasp): could not reach the grasp pose: {why_m}",
                                False, rec)
                    if why_m.startswith("STOP"):
                        outcome = "stopped"
                        break
                    why = f"The grasp stage failed: {why_m}."
                    continue
                jpeg, q, g = self.arm.observe()
                cr = self.arm.set_gripper(self.cfg.close_pct)
                rec["close"] = cr
                stalled = cr["gripper_blocked"] or cr["gripper"] > self.cfg.close_pct + self.cfg.grasp_margin
                rec["check"] = {"stage": "grasp", "ok": bool(stalled), "gripper": cr["gripper"],
                                "gripper_blocked": cr["gripper_blocked"]}
                if not stalled:
                    self.arm.set_gripper(open_pct)
                    self._stage(k, "grasp", q, g, jpeg, f"CHECK FAILED (grasp): the jaws closed EMPTY (gripper "
                                                       f"{cr['gripper']:.0f} %); reopened, the arm is still there", False, rec)
                    why = "The grasp check failed: the jaws closed empty at grasp_q_deg (photo above, taken there before closing)."
                    continue
                self._stage(k, "grasp", q, g, jpeg, f"check ok: the jaws stalled at {cr['gripper']:.0f} %", False, rec)
                k += 1
                rec = {"step": k, "stage": "lift"}
                up = G.lift_path(q, self.cfg.lift_dz_mm, self.lim)
                lifted = False
                if len(up) > 1 and not G.check_path(up, self.lim):
                    lr = self.arm.move(up)
                    rec["lift"] = lr
                    if lr["stopped"]:
                        outcome = "stopped"
                        jpeg, q2, g2 = self.arm.observe()
                        self._stage(k, "lift", q2, g2, jpeg, "STOP during the lift", True, rec)
                        break
                    lifted = True
                jpeg, q2, g2 = self.arm.observe()
                holding = g2 > self.cfg.close_pct + self.cfg.grasp_margin
                dz = float(G.tcp_mm(q2)[2] - G.tcp_mm(q)[2])
                rec["check"] = {"stage": "lift", "ok": bool(holding and lifted), "gripper": g2, "dz_mm": round(dz, 1)}
                if holding:
                    grasped_once = True
                    self._stage(k, "lift", q2, g2, jpeg, f"check ok: still holding (gripper {g2:.0f} %), TCP up "
                                                         f"{dz:.0f} mm", True, rec)
                else:
                    self.arm.set_gripper(open_pct)
                    self._stage(k, "lift", q2, g2, jpeg, f"CHECK FAILED (lift): the object slipped out (gripper "
                                                         f"{g2:.0f} %); jaws reopened", False, rec)
                    why = "The lift check failed: the object was not held after the lift."
                    continue
            # 3. carry
            k += 1
            rec = {"step": k, "stage": "carry", "target_q_deg": plan.place_q_deg}
            ok, why_m, notes = self._move_split(plan.place_q_deg, True, None, False, rec, "carry")
            jpeg, q, g = self.arm.observe()
            still = g > self.cfg.close_pct + self.cfg.grasp_margin
            err = float(np.linalg.norm(G.tcp_mm(q) - G.tcp_mm(plan.place_q_deg)))
            rec["notes"] = notes
            rec["check"] = {"stage": "carry", "ok": bool(ok and still and err < 15.0), "tcp_err_mm": round(err, 1),
                            "holding": bool(still), "why": why_m}
            if not (ok and still and err < 15.0):
                holding = still
                msg = why_m or ("the object was lost on the way" if not still else f"{err:.0f} mm from place_q_deg")
                self._stage(k, "carry", q, g, jpeg, f"CHECK FAILED (carry): {msg}", holding, rec)
                if why_m.startswith("STOP"):
                    outcome = "stopped"
                    break
                why = f"The carry check failed: {msg}."
                continue
            self._stage(k, "carry", q, g, jpeg, f"check ok: at the place pose ({err:.0f} mm)", True, rec)
            # 4. release
            k += 1
            rec = {"step": k, "stage": "release"}
            orr = self.arm.set_gripper(rel_pct)
            placed_at = G.tcp_mm(q)
            up = G.lift_path(q, self.cfg.lift_dz_mm, self.lim)
            if len(up) > 1 and not G.check_path(up, self.lim):
                rec["lift"] = self.arm.move(up, None)
            jpeg, q2, g2 = self.arm.observe()
            opened = orr["gripper"] >= rel_pct - 5.0
            rec["check"] = {"stage": "release", "ok": bool(opened), "gripper": orr["gripper"]}
            holding = not opened
            self._stage(k, "release", q2, g2, jpeg, ("check ok: the jaws opened; photo after lifting off"
                                                     if opened else "CHECK FAILED (release): the jaws did not open"),
                        holding, rec)
            if not opened:
                outcome = "release_failed"
                break
            # 5. home
            qn, _ = self.arm.state()
            home = self._go_home([[qn, self.start_q]], self.cfg.gripper_open_default)
            jpeg, _, _ = self.arm.observe()
            self._log({"step": k + 1, "stage": "home", "home": home, "photo": self._save_photo(k + 1, jpeg, "_home")})
            outcome = "placed" if home is not None else "stuck"
            break
        extra = {"grasped": grasped_once, "replans": max(0, replans),
                 "placed_tcp_mm": None if placed_at is None else placed_at.round(1).tolist()}
        if placed_at is not None and self.cfg.place_xyz_mm is not None:
            extra["place_error_xy_mm"] = round(float(np.linalg.norm(placed_at[:2] - np.asarray(self.cfg.place_xyz_mm)[:2])), 1)
        if outcome not in ("placed", "stopped", "error", "stuck"):
            extra["end"] = self._finish_unplaced(holding, None)
        return self._summary(outcome, t0, calls_total, last_tcp, extra)

    # ------------------------------------------------------------ pixel loop
    def _ask_pixel(self) -> tuple[Pm.PixelReply | None, list]:
        w, h = self.cfg.photo_size
        system = Pm.PIXEL_SYSTEM.format(w=w, h=h)
        parts = Pm.pixel_history_parts(self.cfg.task, self.steps, w=w, h=h, full_photos=self.cfg.history_full_photos)
        calls, extra = [], []
        n = len(list((self.dir / "prompts").glob("px_*.json")))
        for attempt in range(self.cfg.reply_retries + 1):
            msgs = Pm.build_messages(system, parts + extra)
            (self.dir / "prompts" / f"px_{n:02d}_{attempt}.json").write_text(json.dumps(redact(msgs), indent=1))
            res = self.llm.complete(self.cfg.model, msgs, max_tokens=self.cfg.max_tokens,
                                    extra_body=self.cfg.extra_body or None, stop_when=Pm.pixel_json_complete)
            rec = llm_text_usage(res)
            rec["reply"] = res.text or ""
            rec["n_images"] = Pm.count_images(msgs)
            calls.append(rec)
            if res.error:
                continue
            try:
                return Pm.parse_pixel_reply(res.text, w, h), calls
            except Pm.ReplyError as e:
                rec["parse_error"] = str(e)
                extra = [f"\nYour previous reply could not be used ({e}). Reply with the JSON only."]
        return None, calls

    def _px_point(self, q_obs, px, z_mm):
        from controlr.fkagent import visual as V

        cam = self.cfg.camera or V.Camera()
        sx = cam.width / self.cfg.photo_size[0]
        sy = cam.height / self.cfg.photo_size[1]
        return V.pixel_to_point(cam, q_obs, px[0] * sx, px[1] * sy, z_mm)

    def _pose_for(self, xyz, seed):
        from controlr.fkagent import visual as V

        return V.tool_pose_q(xyz, self.lim, seed)

    def _run_pixel(self) -> dict:
        t0 = time.monotonic()
        if not self.steps:
            self.reset()
        holding, phase, look_xy, calls_total = False, "search", None, 0
        searches = [self.start_q, self.start_q + np.array([-25.0, 0, 0, 0, 0]),
                    self.start_q + np.array([25.0, 0, 0, 0, 0])]
        n_search, first, grasped_once, outcome, judged_done = 0, True, False, "max_steps", False
        self._closest = None
        last_tcp = None
        for k in range(1, self.cfg.max_steps + 1):
            hot = self._too_hot()
            if hot:
                self._log({"step": k, "action": "overheat", "error": hot})
                outcome = "overheat"
                break
            rep, calls = self._ask_pixel()
            calls_total += len(calls)
            rec: dict = {"step": k, "llm": calls, "phase_before": phase, "holding_before": holding}
            if rep is None:
                rec["action"] = "no_decision"
                self._log(rec)
                outcome = "no_decision"
                break
            rec["reply"] = rep.as_dict()
            judgement = (f"object_px {rep.object_px}, target_px {rep.target_px}, held={rep.held}, "
                         f"done={rep.done}: {rep.note}")
            # the model's judgement belongs to the photo it looked at (the newest step)
            self.steps[-1].estimate = {**(self.steps[-1].estimate or {}), "judgement": judgement}
            q_obs = np.asarray(self.steps[-1].q_deg, float)
            notes: list[str] = []
            if rep.person:
                time.sleep(2.0)
                jpeg, q, g = self.arm.observe()
                self.steps.append(Pm.Step(k, "wait", q.round(2).tolist(), g, jpeg, result="person in view: held still"))
                rec.update({"action": "wait", "photo": self._save_photo(k, jpeg)})
                self._log(rec)
                continue
            if rep.done:
                judged_done = True
                rec["action"] = "done"
                self._log(rec)
                outcome = "done"
                break
            action, target, gripper, result = None, None, None, ""
            if not holding:
                P = None if rep.object_px is None else self._px_point(q_obs, rep.object_px, self.cfg.object_z_mm)
                if P is None:
                    if n_search >= len(searches):
                        rec["action"] = "give_up"
                        self._log(rec)
                        outcome = "not_found"
                        break
                    action, target = "search", searches[n_search]
                    n_search += 1
                    notes.append("object not visible: looking from another pose")
                else:
                    rec["object_estimate_mm"] = P.round(1).tolist()
                    notes.append(f"object estimate from your pixel: {_v3(P)} mm")
                    near = look_xy is not None and float(np.linalg.norm(P[:2] - look_xy)) < 25.0 and phase == "looked"
                    if not near:
                        qt, pitch = self._pose_for((P[0], P[1], self.cfg.look_z_mm), q_obs)
                        action, target = "look", qt
                        look_xy = P[:2].copy()
                        notes.append(f"look from {self.cfg.look_z_mm:.0f} mm above it (pitch {pitch})")
                    else:
                        qt, pitch = self._pose_for((P[0], P[1], self.cfg.grasp_z_mm), q_obs)
                        action, target, gripper = "grasp", qt, self.cfg.gripper_open_default
                        notes.append(f"grasp at TCP z {self.cfg.grasp_z_mm:.0f} mm (pitch {pitch})")
            else:
                T = None if rep.target_px is None else self._px_point(q_obs, rep.target_px, self.cfg.target_z_mm)
                if T is None:
                    action, target = "look_target", self.start_q
                    notes.append("target not visible: looking from the start pose")
                else:
                    rec["target_estimate_mm"] = T.round(1).tolist()
                    qt, pitch = self._pose_for((T[0], T[1], self.cfg.target_z_mm + self.cfg.release_dz_mm), q_obs)
                    action, target = "place", qt
                    notes.append(f"target estimate from your pixel: {_v3(T)} mm; release {self.cfg.release_dz_mm:.0f} mm "
                                 f"above it (pitch {pitch})")
            rec["action"] = action
            if target is None:
                notes.append("no reachable pose for that point")
                jpeg, q, g = self.arm.observe()
                self.steps.append(Pm.Step(k, action or "none", q.round(2).tolist(), g, jpeg, notes=notes,
                                          result="did not move"))
                rec.update({"notes": notes, "photo": self._save_photo(k, jpeg)})
                self._log(rec)
                continue
            rec["target_q_deg"] = np.round(target, 2).tolist()
            ok, why, mnotes = self._move_split(target, holding, None if holding else (gripper or self.cfg.gripper_open_default),
                                               first, rec, action)
            first = False
            notes += mnotes
            if not ok:
                jpeg, q, g = self.arm.observe()
                self.steps.append(Pm.Step(k, action, q.round(2).tolist(), g, jpeg, notes=notes,
                                          result=f"move failed: {why}"))
                rec.update({"notes": notes, "result": why, "photo": self._save_photo(k, jpeg)})
                self._log(rec)
                if why.startswith("STOP"):
                    outcome = "stopped"
                    break
                phase = "search"
                continue
            jpeg, q, g = self.arm.observe()
            last_tcp = G.tcp_mm(q)
            photo = jpeg
            if action == "look":
                phase = "looked"
            elif action == "grasp":
                cr = self.arm.set_gripper(self.cfg.close_pct)
                rec["close"] = cr
                if cr["gripper_blocked"] or cr["gripper"] > self.cfg.close_pct + self.cfg.grasp_margin:
                    up = G.lift_path(q, 50.0, self.lim)
                    if len(up) > 1 and not G.check_path(up, self.lim):
                        rec["lift"] = self.arm.move(up)
                    photo, q, g = self.arm.observe()
                    holding = g > self.cfg.close_pct + self.cfg.grasp_margin
                    grasped_once |= holding
                    result = (f"jaws stalled at {cr['gripper']:.0f} % (something between them); lifted 50 mm"
                              if holding else "jaws stalled but nothing held after the lift")
                    phase = "carry" if holding else "search"
                else:
                    self.arm.set_gripper(self.cfg.gripper_open_default)
                    up = G.lift_path(q, self.cfg.look_z_mm - self.cfg.grasp_z_mm, self.lim)
                    if len(up) > 1 and not G.check_path(up, self.lim):
                        rec["lift"] = self.arm.move(up)
                    photo, q, g = self.arm.observe()
                    result = "jaws closed EMPTY; reopened and went back up to look again"
                    phase = "search"
            elif action == "place":
                rec["open"] = self.arm.set_gripper(self.cfg.gripper_open_max)
                up = G.lift_path(q, 50.0, self.lim)
                if len(up) > 1 and not G.check_path(up, self.lim):
                    rec["lift"] = self.arm.move(up)
                photo, q, g = self.arm.observe()
                holding = False
                phase = "placed"
                result = "opened the jaws above the target and lifted 50 mm"
            self.steps.append(Pm.Step(k, action, np.round(q, 2).tolist(), g, photo, notes=notes, result=result,
                                      holding=holding))
            rec.update({"notes": notes, "result": result, "reached_q_deg": np.round(q, 2).tolist(), "gripper": g,
                        **G.pose_summary(q), "photo": self._save_photo(k, photo), "holding": holding})
            self._log(rec)
        extra = {"grasped": grasped_once, "model_judged_done": judged_done}
        if outcome != "stopped" and outcome != "overheat":
            extra["end"] = self._finish_unplaced(holding, None) if holding else self._home_only()
        summary = self._summary(outcome, t0, calls_total, last_tcp, extra)
        summary["success"] = judged_done          # the model's judgement from the photos (owner's rule)
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=1, default=_jsonable))
        return summary

    def _home_only(self) -> str:
        q, _ = self.arm.state()
        if float(np.max(np.abs(q - self.start_q))) < 1.0:
            return "at the start pose"
        up = G.lift_path(q, 40.0, self.lim)
        if len(up) > 1 and not G.check_path(up, self.lim):
            self.arm.move(up, None)
            q, _ = self.arm.state()
        home = self._go_home([[q, self.start_q]], None)
        return "returned to the start pose" if home is not None else "NO safe path home: left in place"


def _v3(p) -> str:
    return "(" + ", ".join(f"{float(x):.0f}" for x in p) + ")"
