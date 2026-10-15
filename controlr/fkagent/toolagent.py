"""Tool-calling dialog agent: the model looks at frames and calls tools; tools do all 3D work.

One append-only conversation:
- the system goal;
- FRAME 0 (a user message with a photo, joints, TCP, gripper, blocked flag, elapsed time);
- then, per step: an assistant turn of tool calls, the tool results, and the next FRAME.
Photos older than the newest ``keep_photos`` frames become a text placeholder; their text stays.

Every turn is a single short model call plus at most ONE motion tool (move_to / move_joints /
look / lift / home / grip / wait), then a new frame. Info tools (locate, reachable, tcp) take
milliseconds and need no frame. Every motion goes through the same safety layer as the other
loops: ``plan_target`` checks, ``continuous_path``, splitting at the step limits, the carry floor,
3 mm box margin, server stop rules, and the temperature guard. Completion is the model's call
(``done``), judged from the photos. The episode starts and ends folded (``FkAgent.run``).
"""

from __future__ import annotations

import json
import time

import numpy as np

from controlr.fkagent import geometry as G
from controlr.fkagent import prompt as Pm
from controlr.fkagent.agent import FkAgent, _jsonable

SYSTEM = """\
You control a real SO-101 robot arm (5 joints + a parallel gripper) with tools. GOAL: {task}

The robot:
* Base frame in mm: +x forward (away from the robot), +y to the robot's left, +z up; the table is
  z = 0. TCP = the point between the jaw tips.
* Workspace the TCP may reach: {workspace}.
  Objects outside it cannot be handled: do not chase them, give_up and report their position.
* The only camera rides on the gripper. Each FRAME is a {w}x{h} px photo, (0, 0) top-left; the
  green jaws are at the bottom; a held object is between them. With the tool pointing down,
  image up = farther from the robot.

Tools do ALL geometry (forward kinematics, the calibrated camera, inverse kinematics, safety
checks). You give PIXELS of what you see in the NEWEST frame, plus the height of the point you
pointed at (the object's centre / the target's top), and coarse choices. Never compute joint
angles or coordinates yourself.

Pace: each turn, call at most ONE motion tool (move_to, move_joints, look, lift, home, grip,
wait); after it you get the next FRAME. locate / reachable / tcp are instant and need no frame.
Keep text to one short sentence or none.

A typical pick-and-place:
1. Find both objects; if something is not visible, look(pose) through the look poses.
2. move_to(u, v, height_mm=<object centre height>, above_mm=80): look from above the object.
3. move_to(u, v, height_mm=<centre height>, z=<grasp TCP height + 30>): just above grasp height.
   CHECK the photo: the object must be centred between the two jaw tips. If not, nudge(right_mm,
   up_mm) toward it (small steps, look again). Positions from far-away photos can be off by
   several cm; the photo at this height is the truth.
4. lift(-30) to descend onto it, check again, then grip("close"): blocked=true means held; then lift(50).
5. Put it down: move_to the spot (pixel of a free table spot: height_mm=0, z = the height you grasped
   it at + 2..5 mm; or the CENTRE of a target's top: height_mm=<top height>, above_mm=15). Release LOW,
   never drop it: a round object dropped from higher rolls away. Then grip("open"), then lift(50).
* Jaws: open_pct {omax:.0f} at most; for the plush carrot open up to {wide:.0f} (grip open with
  object="carrot") and grasp it by a narrow part (the tip or the leaves end), not the wide body.
* A free spot for a temporary put-down must be clear of other objects by ~60 mm.
6. When the photos show the task complete, call done(summary). If impossible, give_up(reason).

SAFETY: if a person, a hand or an arm is visible in the newest frame, call wait() and nothing else.
"""


def _num(a: dict, k: str, default=None):
    v = a.get(k, default)
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return default


def tool_specs(look_names: list[str]) -> list[dict]:
    def f(name, desc, props, req=()):
        return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
            "type": "object", "properties": props, "required": list(req)}}}

    px = {"u": {"type": "number", "description": "pixel x in the newest frame"},
          "v": {"type": "number", "description": "pixel y in the newest frame"},
          "height_mm": {"type": "number", "description": "height above the table of the point you pointed at"}}
    return [
        f("locate", "Pixel in the newest frame -> base-frame point (mm) at the given height, and whether "
                    "it is inside the workspace / graspable. Instant.", px, ("u", "v", "height_mm")),
        f("reachable", "Can the TCP reach (x, y, z) mm pointing down? Instant.",
          {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}}, ("x", "y", "z")),
        f("tcp", "Current joints, TCP, gripper. Instant.", {}),
        f("move_to", "MOTION. Move the TCP (tool pointing down as steeply as reachable, approach from above). "
                     "Either a pixel (u, v, height_mm: that point; TCP at height_mm + above_mm, or at z if given) "
                     "or x, y, z in mm.",
          {**px, "above_mm": {"type": "number", "description": "TCP this far above the pointed point (default 0)"},
           "z": {"type": "number", "description": "TCP height in mm (overrides height_mm + above_mm)"},
           "x": {"type": "number"}, "y": {"type": "number"}}),
        f("move_joints", "MOTION. Joint target in degrees [pan, lift, elbow, wrist_flex, wrist_roll] (checked).",
          {"q_deg": {"type": "array", "items": {"type": "number"}, "minItems": 5, "maxItems": 5}}, ("q_deg",)),
        f("look", "MOTION. Go to a fixed look pose to search for objects: " + ", ".join(look_names) + ".",
          {"pose": {"type": "string", "enum": look_names}}, ("pose",)),
        f("lift", "MOTION. Straight up by dz_mm (default 50); negative = straight DOWN (to descend onto an "
                  "object once it is centred between the jaw tips).", {"dz_mm": {"type": "number"}}),
        f("nudge", "MOTION. Small sideways TCP correction in the NEWEST photo's directions, at the same height: "
                   "right_mm > 0 moves the jaws toward what is at the image's right, up_mm > 0 toward what is "
                   "higher in the image. Each at most 40 mm. Use it to centre an object between the jaw tips.",
          {"right_mm": {"type": "number"}, "up_mm": {"type": "number"}}),
        f("grip", "MOTION (jaws). close -> returns blocked (true = something is held); open -> releases / opens "
                  "(open_pct 10..35, default 35; up to the wide cap with object='carrot').",
          {"action": {"type": "string", "enum": ["open", "close"]}, "open_pct": {"type": "number"},
           "object": {"type": "string", "description": "what you will grasp; 'carrot' allows open_pct up to the wide cap"}},
          ("action",)),
        f("home", "MOTION. Back to the start (survey) pose.", {}),
        f("wait", "Do not move; a new frame in ~2 s (use when a person/hand is visible).", {}),
        f("done", "The task is complete as the photos show. Ends the episode.",
          {"summary": {"type": "string"}}, ("summary",)),
        f("give_up", "The task cannot be done (e.g. an object outside the workspace). Ends the episode.",
          {"reason": {"type": "string"}}, ("reason",)),
    ]


MOTION_TOOLS = {"move_to", "move_joints", "look", "lift", "nudge", "grip", "home", "wait"}


class ToolAgent(FkAgent):
    """``llm`` is a ``toolchat.ToolChat`` / ``FakeToolChat``."""

    keep_photos = 6

    def _run_any(self) -> dict:
        try:
            return self._run_tools()
        except Exception as e:  # noqa: BLE001
            import traceback

            try:
                self.arm.hold()
            except Exception:  # noqa: BLE001
                pass
            self._log({"step": len(self.steps), "action": "error", "error": f"{type(e).__name__}: {e}",
                       "traceback": traceback.format_exc()})
            summary = {"outcome": "error", "success": False, "error": f"{type(e).__name__}: {e}"}
            (self.dir / "summary.json").write_text(json.dumps(summary, indent=1))
            return summary

    # ------------------------------------------------------------------ frames
    def _frame(self, note: str = "") -> dict:
        t0 = time.monotonic()
        jpeg, q, g = self.arm.observe()
        self._frame_q = np.asarray(q, float)
        self._n_frame += 1
        k = self._n_frame
        s = G.pose_summary(q)
        text = (f"FRAME {k} (t = {time.monotonic() - self._t0:.1f} s): joints {Pm._v(q)} deg; TCP {Pm._v(s['tcp_mm'], 0)} mm, "
                f"pitch {s['tool_pitch_deg']:.0f} deg; gripper {g:.0f} %; holding (jaws blocked): "
                f"{'yes' if self.holding else 'no'}" + (f". {note}" if note else ""))
        w, _ = self.cfg.photo_size
        img = Pm.image_part(jpeg, long_edge=w, label=f"frame{k}")
        msg = {"role": "user", "content": [{"type": "text", "text": text},
                                           {"type": "image_url", "image_url": {"url": img.data_url}}]}
        self.messages.append(msg)
        self._frame_msgs.append((k, msg))
        if len(self._frame_msgs) > self.keep_photos:          # bound the request: old photos -> placeholder
            kk, old = self._frame_msgs[-self.keep_photos - 1]
            if len(old["content"]) > 1:
                old["content"] = [old["content"][0], {"type": "text", "text": f"[photo of frame {kk} omitted]"}]
        photo = self._save_photo(k, jpeg, "_frame")
        self.steps.append(Pm.Step(k, "frame", np.round(q, 2).tolist(), g, jpeg, holding=self.holding))
        return {"frame": k, "photo": photo, "t_frame": round(time.monotonic() - t0, 2), "text": text}

    # ------------------------------------------------------------------ tools
    def _ws_text(self) -> str:
        lo, hi, L = self.lim.box_lo_mm, self.lim.box_hi_mm, self.lim
        t = f"x {lo[0]:.0f}..{hi[0]:.0f}, y {lo[1]:.0f}..{hi[1]:.0f}, z {lo[2]:.0f}..{hi[2]:.0f} mm"
        if L.side_y_mm is not None:
            t += f"; and |y| up to {L.side_y_mm:.0f} where x <= {L.side_x_max_mm:.0f}"
        return t

    def _locate(self, a: dict):
        u, v, h = _num(a, "u"), _num(a, "v"), _num(a, "height_mm", self.cfg.object_z_mm)
        if u is None or v is None:
            return None, "u and v are required"
        w, hh = self.cfg.photo_size
        if not (0 <= u <= w and 0 <= v <= hh):
            return None, f"pixel ({u:.0f}, {v:.0f}) is outside the {w}x{hh} frame"
        P = self._px_point(self._frame_q, [u, v], h)
        if P is None:
            return None, "that pixel does not look down onto that height"
        return P, ""

    def _reach(self, x, y, z, pitch=None) -> tuple[np.ndarray | None, str, float | None]:
        from controlr.fkagent import visual as V

        lo, hi = np.asarray(self.lim.box_lo_mm), np.asarray(self.lim.box_hi_mm)
        p = np.array([x, y, z], float)
        if not G.in_box(p, self.lim, -self.lim.target_margin_mm):
            return None, f"({x:.0f}, {y:.0f}, {z:.0f}) mm is outside the workspace ({self._ws_text()})", None
        kw = {} if pitch is None else {"pitches": (float(pitch),)}
        q, pt = V.tool_pose_q(p, self.lim, self._frame_q, **kw)
        if q is None:
            return None, f"no reachable tool pose at ({x:.0f}, {y:.0f}, {z:.0f}) mm", None
        return q, "", pt

    def _motion(self, name: str, target_q, rec: dict, gripper=None) -> dict:
        q0, _ = self.arm.state()
        ok, why, notes = self._move_split(target_q, self.holding, gripper, self._first, rec, name)
        if why.startswith("STOP"):
            self._stopped = True
        q, _ = self.arm.state()
        moved = float(np.max(np.abs(q - q0))) > 0.2
        self._first = self._first and not moved
        return {"ok": ok, "why": why, "notes": notes, "moved": moved, "reached_tcp_mm": G.tcp_mm(q).round(1).tolist()}

    def _tool(self, name: str, a: dict, rec: dict) -> dict:
        if "_bad_json" in a:
            return {"ok": False, "why": "arguments are not valid JSON"}
        if name == "tcp":
            q, g = self.arm.state()
            return {"joints_deg": np.round(q, 1).tolist(), **G.pose_summary(q), "gripper_pct": round(g, 1),
                    "holding": self.holding}
        if name == "locate":
            P, why = self._locate(a)
            if P is None:
                return {"ok": False, "why": why}
            self.located.append({"frame": self._n_frame, "px": [a.get("u"), a.get("v")], "xyz_mm": P.round(1).tolist()})
            _, why_g, _ = self._reach(P[0], P[1], self.cfg.grasp_z_mm)
            return {"ok": True, "xyz_mm": P.round(1).tolist(), "in_workspace": not why_g, "why": why_g}
        if name == "reachable":
            q, why, pt = self._reach(_num(a, "x", 0), _num(a, "y", 0), _num(a, "z", 0))
            return {"ok": q is not None, "why": why, "pitch_deg": pt}
        if name == "move_to":
            if a.get("u") is not None and a.get("v") is not None:
                P, why = self._locate(a)
                if P is None:
                    return {"ok": False, "why": why}
                self.located.append({"frame": self._n_frame, "px": [a.get("u"), a.get("v")],
                                     "xyz_mm": P.round(1).tolist()})
                z = _num(a, "z")
                z = (P[2] + (_num(a, "above_mm", 0.0) or 0.0)) if z is None else z
                x, y = float(P[0]), float(P[1])
            else:
                x, y, z = _num(a, "x"), _num(a, "y"), _num(a, "z")
                if None in (x, y, z):
                    return {"ok": False, "why": "give u, v, height_mm or x, y, z"}
            q, why, pt = self._reach(x, y, z)
            if q is None:
                return {"ok": False, "why": why}
            r = self._motion("move_to", q, rec)
            return {**r, "target_mm": [round(x, 1), round(y, 1), round(z, 1)], "pitch_deg": pt}
        if name == "move_joints":
            qd = a.get("q_deg")
            if not isinstance(qd, list) or len(qd) != 5:
                return {"ok": False, "why": "q_deg needs 5 numbers"}
            return self._motion("move_joints", np.asarray(qd, float), rec)
        if name == "look":
            poses = dict(self._looks)
            if a.get("pose") not in poses:
                return {"ok": False, "why": f"unknown pose; use one of {list(poses)}"}
            return self._motion(f"look {a['pose']}", poses[a["pose"]], rec)
        if name == "home":
            return self._motion("home", self.start_q, rec)
        if name == "nudge":
            return self._nudge(_num(a, "right_mm", 0.0) or 0.0, _num(a, "up_mm", 0.0) or 0.0)
        if name == "lift":
            dz = float(np.clip(_num(a, "dz_mm", 50.0) or 50.0, -100.0, 100.0))
            q, _ = self.arm.state()
            up = G.lift_path(q, dz, self._lim_for(self.holding, q))
            if len(up) < 2 or G.check_path(up, self._lim_for(self.holding, q)):
                return {"ok": False, "why": "straight up is not reachable / not safe here"}
            r = self.arm.move(up, None)
            if r["stopped"]:
                self._stopped = True
            q, _ = self.arm.state()
            return {"ok": not r["stopped"], "reached_tcp_mm": G.tcp_mm(q).round(1).tolist()}
        if name == "grip":
            if a.get("action") == "close":
                r = self.arm.set_gripper(self.cfg.close_pct)
                self.holding = bool(r["gripper_blocked"] or r["gripper"] > self.cfg.close_pct + self.cfg.grasp_margin)
                self.grasped_once |= self.holding
                return {"ok": True, "gripper_pct": round(r["gripper"], 1), "blocked": self.holding}
            cap = self.cfg.gripper_open_max
            if str(a.get("object", "")).lower() == "carrot" and self.cfg.gripper_wide_pct:
                cap = max(cap, self.cfg.gripper_wide_pct)    # checked in place: no camera contact up to 55 %
            want = _num(a, "open_pct", self.cfg.gripper_open_max) or self.cfg.gripper_open_max
            pct = float(np.clip(want, self.cfg.gripper_open_min, cap))
            r = self.arm.set_gripper(pct)
            was = self.holding
            self.holding = False
            if was:
                self.released_at = G.tcp_mm(self.arm.state()[0]).round(1).tolist()
            out = {"ok": True, "gripper_pct": round(r["gripper"], 1), "released": was}
            if want > cap:
                out["note"] = f"opening capped at {cap:.0f} %"
            return out
        if name == "wait":
            time.sleep(2.0)
            return {"ok": True}
        return {"ok": False, "why": f"unknown tool {name}"}

    def _nudge(self, right: float, up: float) -> dict:
        """Translate the TCP along the image's right / up directions projected onto the table
        (same height, same tool orientation, straight line, checked)."""
        from controlr.fkagent import visual as V
        from controlr.robot.so101 import kinematics as K

        right, up = float(np.clip(right, -40, 40)), float(np.clip(up, -40, 40))
        if abs(right) < 0.5 and abs(up) < 0.5:
            return {"ok": False, "why": "nothing to do"}
        q, _ = self.arm.state()
        cam = self.cfg.camera or V.Camera()
        Rc = cam.T_base_cam(q)[:3, :3]
        def flat(v):
            v = np.array([v[0], v[1], 0.0])
            n = np.linalg.norm(v)
            return v / n if n > 1e-6 else v
        ex, ey = flat(Rc[:, 0]), flat(-Rc[:, 1])          # image right, image up, on the table
        T = G.fk_T(q)
        p = T[:3, 3] * 1000.0 + right * ex + up * ey
        lim = self._lim_for(self.holding, q)
        if not G.in_box(p, lim, -lim.target_margin_mm):
            return {"ok": False, "why": f"({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}) mm is outside the workspace"}
        lim_rad = np.radians(np.stack([lim.joint_lo_deg, lim.joint_hi_deg], axis=1))
        R = K.reachable_rotation(p / 1000.0, T[:3, :3])       # 5 joints: the tool turns with the pan
        path, frac, why = K.line_path(np.radians(q), p / 1000.0, R, lim_rad)
        rows = [q] + [np.degrees(x) for x in path]
        if frac < 0.99 or len(rows) < 2:
            return {"ok": False, "why": f"not reachable that way ({why or 'IK'})"}
        bad = G.check_path(rows, lim)
        if bad:
            return {"ok": False, "why": bad}
        r = self.arm.move(rows, None)
        if r["stopped"]:
            self._stopped = True
        q2, _ = self.arm.state()
        return {"ok": not r["stopped"], "moved": True, "reached_tcp_mm": G.tcp_mm(q2).round(1).tolist(),
                "moved_mm": [round(right, 1), round(up, 1)]}

    # ------------------------------------------------------------------ loop
    def _run_tools(self) -> dict:
        self._t0 = time.monotonic()
        self.holding, self.grasped_once, self.released_at = False, False, None
        self._first, self._stopped, self._n_frame = True, False, -1
        self._frame_msgs: list = []
        self.located: list = []
        self.messages: list[dict] = []
        self._looks = [("survey", self.start_q)] + [(n.replace(" ", ""), q) for n, q in self._search_poses()]
        self.tools = tool_specs([n for n, _ in self._looks])
        w, h = self.cfg.photo_size
        self.messages.append({"role": "system", "content": SYSTEM.format(
            task=self.cfg.task, workspace=self._ws_text(), w=w, h=h, wide=self.cfg.gripper_wide_pct,
            omax=self.cfg.gripper_open_max)})
        self.reset()                                # fold -> start pose (checked), photo = step 0
        self.steps.clear()
        f0 = self._frame("Start.")
        self._log({"step": 0, "action": "frame", **f0})
        outcome, end_text, calls = "max_steps", "", 0
        t_model_all = []
        for k in range(1, self.cfg.max_steps + 1):
            hot = self._too_hot()
            if hot:
                outcome, end_text = "overheat", hot
                self._log({"step": k, "action": "overheat", "error": hot})
                break
            t_m = time.monotonic()
            rep = self.llm.complete(self.cfg.model, self.messages, self.tools, max_tokens=self.cfg.max_tokens,
                                    extra_body=self.cfg.extra_body or None)
            t_model = time.monotonic() - t_m
            calls += 1
            t_model_all.append(t_model)
            rec: dict = {"step": k, "t_model_s": round(t_model, 2), "usage": rep.usage, "content": rep.content,
                         "tool_calls": [], "llm": [{"t_wall": round(rep.t_wall, 2), "error": rep.error}]}
            (self.dir / "prompts" / f"turn_{k:02d}.json").write_text(json.dumps(
                [m if m["role"] != "user" or isinstance(m["content"], str) else
                 {"role": "user", "content": [c if c["type"] == "text" else {"type": "image_url", "image_url": "<image>"}
                                              for c in m["content"]]} for m in self.messages], indent=1))
            if rep.error:
                rec["action"] = "llm_error"
                self._log(rec)
                outcome, end_text = "llm_error", rep.error
                break
            self.messages.append(rep.raw())
            if not rep.tool_calls:
                self.messages.append({"role": "user", "content": "Call a tool (one motion per turn), or done / give_up."})
                rec["action"] = "no_tool_call"
                self._log(rec)
                continue
            moved, ended = False, None
            for c in rep.tool_calls:
                a = c.args()
                t_t = time.monotonic()
                if c.name in ("done", "give_up"):
                    res = {"ok": True}
                    ended = (c.name, str(a.get("summary") or a.get("reason") or ""))
                elif c.name in MOTION_TOOLS and moved:
                    res = {"ok": False, "why": "skipped: one motion per turn; you get a new frame first"}
                elif ended:
                    res = {"ok": False, "why": "skipped: the episode ended"}
                else:
                    res = self._tool(c.name, a, rec)
                    # a motion that was refused before moving neither uses the turn nor gives a frame
                    moved |= c.name in MOTION_TOOLS and (c.name in ("grip", "wait") or bool(res.get("moved", res.get("ok"))))
                t_tool = time.monotonic() - t_t
                rec["tool_calls"].append({"name": c.name, "args": a, "result": res, "t_tool_s": round(t_tool, 3)})
                self.messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(res, default=_jsonable)})
            rec["action"] = ",".join(t["name"] for t in rec["tool_calls"])
            if self._stopped:
                rec["stopped"] = True
                self._log(rec)
                outcome, end_text = "stopped", "the arm hit something (server STOP)"
                break
            if ended:
                self._log(rec)
                outcome, end_text = ended
                break
            if moved:
                f = self._frame()
                rec.update({"frame": f["frame"], "photo": f["photo"], "t_frame_s": f["t_frame"]})
            self._log(rec)
        # end of the task: never leave an object hanging; go back to the start pose
        end_actions = []
        if outcome not in ("stopped", "error"):
            if self.holding:
                q, _ = self.arm.state()
                p = G.tcp_mm(q)
                qd, _, _ = self._reach(p[0], p[1], self.cfg.grasp_z_mm + 5)
                if qd is not None:
                    self._motion("put_down", qd, {})
                self.arm.set_gripper(self.cfg.gripper_open_default)
                self.holding = False
                end_actions.append("put the held object down")
            end_actions.append(self._home_only())
        summary = {"outcome": {"done": "done", "give_up": "give_up"}.get(outcome, outcome),
                   "success": outcome == "done", "end_text": end_text, "loop": "tools", "model": self.cfg.model,
                   "turns": calls, "frames": self._n_frame, "grasped": self.grasped_once,
                   "released_at_mm": self.released_at, "located": self.located, "end": "; ".join(end_actions),
                   "wall_s": round(time.monotonic() - self._t0, 1),
                   "t_model_median_s": round(float(np.median(t_model_all)), 2) if t_model_all else None}
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=1, default=_jsonable))
        return summary
