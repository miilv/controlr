"""A fake strict robot server: robot-harness's JSON-lines protocol over a Unix socket, on the kinematic env.

For tests and dry runs of the bridge without Isaac (CPU only). It mimics ``harness/robodojo_server_j1.py --strict``
(strict tier B + ``perceive`` + B1 ``held_look``) as far as the bridge sees it:

* socket ``<bench>/run/<id>.sock``, one JSON request per connection, replies ``{"ok", "cmd", "t_wall", "step",
  "steps_left", "ended", ...}``, errors ``{"ok": false, "error", "hint"}``;
* ``info`` (tier ``B-strict: ...``, ``strict.finger_measurement: false``, ``commands``), ``reset`` (takes the next
  queued episode: instruction, ``step_budget``, first ``obs``), ``obs`` (PNGs written to ``save_dir`` or the run
  dir, ``eef`` per arm in the strict server's shape: pos = TCP, quat, flange {pos, quat}, gripper, grip_cmd,
  grip_obs, q, home_dpos_max_m, home_ddeg, at_home, held_look), ``hold``, ``gripper``, ``return_home``,
  ``perceive`` (a J1-schema output from the env's objects; the fake's perceiver), ``held_look`` (B1-style answer
  from the env's attachment), ``log``, ``close``;
* the PROPOSED verb ``joint_steps`` (actions = list of the four official joint-mode keys per env step; ``reserve``;
  ``max_dq``): validated before any step (refused at 0 steps: keys, finiteness, gripper 0..1, joint jump per step
  vs the last command <= max_dq, budget), then one env step per entry, verbatim, stopping when the episode ends.

``leak=[...]`` makes it send banned fields (``objects``, ``finger_q``, ``calib``, ...) so the bridge's strict mode
can be tested; ``tier=`` fakes a non-strict server. ``success_fn(env)`` emulates RoboDojo's success latch (the
episode ends, ``ended`` goes true; the verdict itself never leaves the fake except through ``result()``).

    python -m dojo.bridge.fake_server --id fake-1 --bench /tmp/fakebench --scene stack_blocks   # a CLI for dry runs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import threading
import time
from pathlib import Path

import numpy as np

from dojo.skills.env import MAX_JOINT_STEP, KinematicEnv
from dojo.skills.geometry import T_to_pose7
from dojo.skills.kinematics import ARMS, X5Kin

STRICT_TIER = "B-strict: images + joints + flange + commanded gripper; calib computed from public config"
ACTION_KEYS = ("left_arm_joint_state", "left_ee_joint_state", "right_arm_joint_state", "right_ee_joint_state")
CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def _q_angle_deg(a, b) -> float:
    return math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(a, b))))))


class FakeStrictServer:
    def __init__(self, server_id: str, bench: str | Path, episodes: list, reserve: int = 20, img_hw=(48, 64),
                 leak: tuple = (), tier: str = STRICT_TIER, joint_steps: bool = True, success_fn=None,
                 perceive_noise_m: float = 0.0, seed: int = 0):
        """``episodes``: callables returning a fresh ``KinematicEnv`` (one per RoboDojo episode, in order)."""
        self.id = server_id
        self.bench = Path(bench)
        self.run_dir = self.bench / "run" / server_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.sock_path = str(self.bench / "run" / f"{server_id}.sock")
        self.queue = list(episodes)
        self.reserve = reserve
        self.hw = img_hw
        self.leak = tuple(leak)
        self.tier = tier
        self.has_joint_steps = joint_steps
        self.success_fn = success_fn
        self.noise = perceive_noise_m
        self.rng = np.random.default_rng(seed)
        self.kin = X5Kin()
        self.env: KinematicEnv | None = None
        self.hold_q: dict = {}
        self.hold_g: dict = {}
        self.home_q: dict = {}
        self.home_fl: dict = {}
        self.success = False
        self.episodes: list[dict] = []
        self.requests: list[dict] = []          # (verb, step) log, for tests
        self.n_perceive = 0
        self.closing = False
        self._srv = None
        self._thread = None

    # ------------------------------------------------------------------ facts
    @property
    def commands(self) -> list[str]:
        c = ["info", "reset", "obs", "move_to", "move_delta", "gripper", "hold", "return_home", "reach", "skill",
             "locate", "project", "success", "log", "end", "close", "held_look", "perceive"]
        return c + (["joint_steps"] if self.has_joint_steps else [])

    def _ended(self) -> bool:
        return bool(self.env is not None and (self.env.ended or self.success))

    def _facts(self) -> dict:
        if self.env is None:
            return {"step": 0, "steps_left": 0, "ended": False}
        return {"step": self.env.steps_used, "steps_left": self.env.step_lim - self.env.steps_used, "ended": self._ended()}

    def result(self) -> dict:
        """DEV ONLY (test assertions): the fake's verdict of the current / last episode."""
        return {"success": self.success, "steps_used": self.env.steps_used if self.env else 0}

    # ------------------------------------------------------------------ the env
    def _step(self, act: dict) -> bool:
        env = self.env
        ended = env.step({k: np.asarray(v, np.float32) for k, v in act.items()})
        if not self.success and self.success_fn is not None and self.success_fn(env):
            self.success = True
        return ended or self.success

    def _hold_action(self) -> dict:
        out = {}
        for a in ARMS:
            out[f"{a}_arm_joint_state"] = np.asarray(self.hold_q[a], np.float32)
            out[f"{a}_ee_joint_state"] = np.asarray([self.hold_g[a]], np.float32)
        return out

    def _eef(self) -> dict:
        out = {}
        st = self.env.proprio()
        for a in ARMS:
            fl = np.asarray(st[f"{a}_ee_pose"], float)
            tcp = self.kin.tcp(a, self.env.q[a])[:3, 3]
            hf = self.home_fl[a]
            dpos = float(np.abs(fl[:3] - hf[:3]).max())
            ddeg = _q_angle_deg(fl[3:7], hf[3:7])
            e = {"pos": tcp.round(5).tolist(), "quat": fl[3:7].round(6).tolist(),
                 "flange": {"pos": fl[:3].round(6).tolist(), "quat": fl[3:7].round(6).tolist()},
                 "gripper": self.hold_g[a], "grip_cmd": self.hold_g[a], "grip_obs": float(st[f"{a}_ee_joint_state"][0]),
                 "q": np.asarray(st[f"{a}_arm_joint_state"], float).round(6).tolist(), "home_dpos_max_m": round(dpos, 4),
                 "home_ddeg": round(ddeg, 2), "at_home": bool(dpos <= 0.15 and ddeg <= 20), "held_look": None}
            if "finger_q" in self.leak:
                e["finger_q"] = [0.02, 0.02]
                e["gap_m"] = 0.04
            out[a] = e
        return out

    def _frames(self, cams, save_dir) -> dict:
        from PIL import Image
        d = Path(save_dir) if save_dir else self.run_dir / "obs"
        d.mkdir(parents=True, exist_ok=True)
        h, w = self.hw
        paths = {}
        for i, c in enumerate(cams):
            img = np.zeros((h, w, 3), np.uint8)
            img[..., i] = (self.env.steps_used * 7) % 255           # changes with the step: a "new frame"
            p = d / f"s{self.env.steps_used:04d}_{c}.png"
            Image.fromarray(img).save(p)
            paths[c] = str(p)
        return paths

    # ------------------------------------------------------------------ verbs
    def cmd_info(self, a):
        r = {"kind": "robodojo", "instruction": self.env.instruction if self.env else None,
             "step_lim": self.env.step_lim if self.env else 0, "episode_active": self.env is not None and not self.closing,
             "episode_pending": bool(self.queue), "arms": list(ARMS), "cameras": list(CAMS), "tier": self.tier,
             "reserve": self.reserve, "commands": self.commands}
        if self.tier.startswith("B-strict"):
            r["strict"] = {"calib": "computed (strict_calib.py)", "finger_measurement": False,
                           "held_signal": "held_look (perceiver)", "violations": 0}
        return r

    def cmd_reset(self, a):
        if not self.queue:
            raise _Err("no RoboDojo episode arrived", "the fake's episode queue is empty")
        if self.env is not None:
            self.episodes.append({"outcome": "reset", **self.result()})
        self.env = self.queue.pop(0)()
        self.success = False
        st = self.env.proprio()
        self.hold_q = {x: np.asarray(st[f"{x}_arm_joint_state"], float).copy() for x in ARMS}
        self.hold_g = {x: float(np.asarray(st[f"{x}_ee_joint_state"]).reshape(-1)[0]) for x in ARMS}
        self.home_q = {x: v.copy() for x, v in self.hold_q.items()}
        self.home_fl = {x: np.asarray(st[f"{x}_ee_pose"], float).copy() for x in ARMS}
        obs = self.cmd_obs({"cameras": a.get("cameras")})
        rep = {"instruction": self.env.instruction, "step_budget": self.env.step_lim, "arms": list(ARMS),
               "cameras": list(CAMS), "obs": obs, "note": "task and layout are fixed by robot-start"}
        return rep

    def _need_env(self):
        if self.env is None:
            raise _Err("no active episode", "call reset first")

    def cmd_obs(self, a):
        self._need_env()
        cams = a.get("cameras")
        cams = list(CAMS) if cams is None else [c for c in cams if c in CAMS]
        out = {"images": self._frames(cams, a.get("save_dir")), "eef": self._eef(), "instruction": self.env.instruction}
        if "objects" in self.leak:
            out["objects"] = {o["label"]: {"pos": np.asarray(o["pos"]).tolist()} for o in self.env.objects}
        if "calib" in self.leak:
            out["calib"] = {"cam_head": {"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}}
        return out

    def cmd_joint_steps(self, a):
        self._need_env()
        if not self.has_joint_steps:
            raise _Err("unknown command 'joint_steps'", "")
        acts = a.get("actions")
        if not isinstance(acts, list) or not acts:
            raise _Err("joint_steps needs actions=[{4 joint-mode keys}, ...]", "")
        max_dq = float(a.get("max_dq", MAX_JOINT_STEP)) + 1e-6
        reserve = int(a.get("reserve", self.reserve))
        prev = {x: np.asarray(self.hold_q[x], float) for x in ARMS}
        parsed = []
        for i, act in enumerate(acts):
            if set(act) != set(ACTION_KEYS):
                return self._refused(f"action {i}: keys {sorted(act)}")
            p = {k: np.asarray(act[k], float).reshape(-1) for k in ACTION_KEYS}
            if any(not np.all(np.isfinite(v)) for v in p.values()):
                return self._refused(f"action {i}: non-finite")
            for x in ARMS:
                q, g = p[f"{x}_arm_joint_state"], p[f"{x}_ee_joint_state"]
                if q.size != 6 or g.size != 1 or not 0.0 <= g[0] <= 1.0:
                    return self._refused(f"action {i}: {x} q[{q.size}] g {g.tolist()}")
                dq = float(np.abs(q - prev[x]).max())
                if dq > max_dq:
                    return self._refused(f"action {i}: {x} joint jump {dq:.3f} rad > max_dq {max_dq:.3f}")
                prev[x] = q
            parsed.append(p)
        left = self.env.step_lim - self.env.steps_used
        if len(parsed) > left - reserve:
            return self._refused("budget", hint=f"needs {len(parsed)} steps, {left} left, reserve {reserve}")
        n0 = self.env.steps_used
        ended = self._ended()
        for p in parsed:
            if ended:
                break
            for x in ARMS:
                self.hold_q[x] = p[f"{x}_arm_joint_state"]
                self.hold_g[x] = float(p[f"{x}_ee_joint_state"][0])
            ended = self._step(self._hold_action())
        eef = self._eef()
        track = {x: round(float(np.abs(np.asarray(eef[x]["q"]) - self.hold_q[x]).max()), 4) for x in ARMS}
        return {"steps": self.env.steps_used - n0, "refused": None, "tracking_rad": track,
                "eef": {x: {k: eef[x][k] for k in ("q", "flange", "grip_cmd")} for x in ARMS}}

    def _refused(self, why, hint=""):
        return {"steps": 0, "refused": why, "hint": hint}

    def cmd_hold(self, a):
        self._need_env()
        n0 = self.env.steps_used
        for _ in range(int(a.get("steps", 1))):
            if self._ended() or self._step(self._hold_action()):
                break
        return {"steps": self.env.steps_used - n0}

    def cmd_gripper(self, a):
        self._need_env()
        arm = a.get("arm")
        if arm not in ARMS:
            raise _Err(f"arm must be 'left' or 'right', got {arm!r}", "")
        act = a.get("action", a.get("value"))
        g = 1.0 if act in ("open", None) else 0.0 if act == "close" else float(act)
        n0 = self.env.steps_used
        for _ in range(8):                                  # <= 0.25 of range per step, like the shim
            if abs(g - self.hold_g[arm]) < 1e-6 or self._ended():
                break
            self.hold_g[arm] = float(np.clip(g, self.hold_g[arm] - 0.25, self.hold_g[arm] + 0.25))
            self._step(self._hold_action())
        for _ in range(int(a.get("wait_steps", 3 if g < 0.5 else 0))):
            if self._ended():
                break
            self._step(self._hold_action())
        r = {"arm": arm, "grip_cmd": self.hold_g[arm], "cmd_width_m": round(2 * max(0.0, -0.01 + g * 0.054), 4),
             "steps": self.env.steps_used - n0}
        if "finger_q" in self.leak:
            r["grasp_detected"] = True
        return r

    def cmd_return_home(self, a):
        self._need_env()
        arms = a.get("arms") or list(ARMS)
        n0 = self.env.steps_used
        if a.get("open", True):
            for x in arms:
                self.hold_g[x] = 1.0
            self._step(self._hold_action())
        while not self._ended():
            d = max(float(np.abs(self.home_q[x] - self.hold_q[x]).max()) for x in arms)
            if d < 1e-6:
                break
            for x in arms:
                self.hold_q[x] = self.hold_q[x] + np.clip(self.home_q[x] - self.hold_q[x], -0.08, 0.08)
            self._step(self._hold_action())
        return {"all_home": True, "steps": self.env.steps_used - n0}

    def cmd_perceive(self, a):
        """The fake's perceiver in the J1 schema (label, category, asset, pos, quat, size, confidence, source)."""
        self._need_env()
        self.n_perceive += 1
        objs = []
        for o in self.env.objects:
            p = np.asarray(o["pos"], float) + np.r_[self.rng.normal(0, self.noise, 2), 0.0]
            objs.append({"label": o["label"], "category": o["category"],
                         "asset": f"{o.get('type', 'Rigid')}/{o['category']}/{int(o.get('category_idx', 0)):05d}",
                         "pos": p.round(5).tolist(), "quat": np.asarray(o["quat"], float).round(6).tolist(),
                         "size": (2 * np.asarray(o["half"], float)).round(4).tolist(), "confidence": 0.9,
                         "source": "head" if self.n_perceive == 1 else "fused"})
        frames = self._frames(list(CAMS), self.run_dir / "perceive" / f"c{self.n_perceive:03d}")
        return {"perceived": objs, "regions": [], "needs_look": [], "last_seen": [], "n": self.n_perceive,
                "frames": frames, "perceiver": "fake"}

    def cmd_held_look(self, a):
        self._need_env()
        arm = a.get("arm")
        held = arm in self.env.attached
        return {"arm": arm, "stage": a.get("stage", ""), "held": held, "answer": "yes" if held else "no",
                "source": "B1", "confidence": 0.9}

    def cmd_log(self, a):
        return {"note": str(a.get("note", ""))[:2000]}

    def cmd_close(self, a):
        if self.env is not None:
            self.episodes.append({"outcome": "close", **self.result()})
        self.closing = True
        return {"closed": True}

    # ------------------------------------------------------------------ dispatch + socket
    def handle(self, req: dict) -> dict:
        t0 = time.time()
        cmd = req.get("cmd")
        args = {k: v for k, v in req.items() if k != "cmd"}
        self.requests.append({"cmd": cmd, "step": self.env.steps_used if self.env else 0,
                              "n": len(args.get("actions") or []) if cmd == "joint_steps" else None})
        try:
            fn = getattr(self, f"cmd_{cmd}", None)
            if fn is None or cmd not in self.commands:
                raise _Err(f"unknown command {cmd!r}", f"commands: {self.commands}")
            rep = {"ok": True, **fn(args)}
        except _Err as e:
            rep = {"ok": False, "error": str(e), "hint": e.hint}
        except Exception as e:  # noqa: BLE001
            rep = {"ok": False, "error": f"{type(e).__name__}: {e}", "hint": ""}
        rep.update(cmd=cmd, t_wall=round(time.time() - t0, 4), **self._facts())
        return rep

    def start(self) -> "FakeStrictServer":
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(self.sock_path)
        self._srv.listen(8)
        self._srv.settimeout(0.2)
        self._thread = threading.Thread(target=self._serve, daemon=True, name=f"fake-server-{self.id}")
        self._thread.start()
        return self

    def _serve(self) -> None:
        while not self.closing:
            try:
                c, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            c.settimeout(None)
            buf = b""
            try:
                while not buf.endswith(b"\n"):
                    ch = c.recv(1 << 20)
                    if not ch:
                        break
                    buf += ch
                if buf.strip():
                    rep = self.handle(json.loads(buf.decode()))
                    c.sendall((json.dumps(rep) + "\n").encode())
            finally:
                c.close()
        try:
            self._srv.close()
            os.unlink(self.sock_path)
        except OSError:
            pass

    def stop(self) -> None:
        self.closing = True
        if self._thread is not None:
            self._thread.join(5)


class _Err(Exception):
    def __init__(self, msg, hint=""):
        super().__init__(msg)
        self.hint = hint


# ------------------------------------------------------------------ CLI (dry runs of the bridge driver)
STACK = "Stack the three blocks with different textures."
SCENES = {
    "stack_blocks": (STACK, [{"label": f"block_{i}", "category": "block", "category_idx": 5, "pos": p,
                              "quat": [1, 0, 0, 0], "half": [0.015] * 3}
                             for i, p in enumerate([(0.412, -0.09, 0.78), (-0.26, -0.189, 0.78), (-0.344, -0.1, 0.78)])]),
}


def stacked(env) -> bool:
    p = sorted((np.asarray(o["pos"]) for o in env.objects), key=lambda v: v[2])
    return all(np.linalg.norm(a[:2] - b[:2]) < 0.006 and 0.025 < b[2] - a[2] < 0.035 for a, b in zip(p[:-1], p[1:]))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--scene", default="stack_blocks", choices=sorted(SCENES))
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--step-lim", type=int, default=550)
    a = ap.parse_args(argv)
    instr, rows = SCENES[a.scene]
    srv = FakeStrictServer(a.id, a.bench, [lambda: KinematicEnv(rows, step_lim=a.step_lim, instruction=instr)] * a.episodes,
                           success_fn=stacked if a.scene == "stack_blocks" else None).start()
    print(json.dumps({"server_id": a.id, "sock": srv.sock_path}), flush=True)
    try:
        while not srv.closing:
            time.sleep(0.5)
    finally:
        srv.stop()
    print(f"[fake_server] closed; episodes {srv.episodes}", flush=True)


if __name__ == "__main__":
    main()
