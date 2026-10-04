"""controlr's XPolicyLab policy for RoboDojo: the shim inside RoboDojo's eval client.

Installed as ``XPolicyLab/policy/controlr/`` (``scripts/robodojo/run.sh`` copies this
directory plus ``../protocol.py``). RoboDojo calls ``eval_one_episode(TASK_ENV, model_client)``
once per episode with the sim paused; we connect to ``controlr robodojo-serve`` and serve its
requests (observe / execute / check_goal / done) until it ends the episode. RoboDojo keeps
everything that makes the result official: layouts, ``step_lim``, the success latch, the
result json and the mp4 (frames are appended inside ``get_obs``).

Motion follows the reference LLM adapter RoboDojo itself runs (XPolicyLab
``GPT_6_Astra_Direct_EEF``): each arm target is a flange pose planned by RoboDojo's cuRobo
planner, the joint path is resampled so no joint moves more than ``ARM_STEP_RAD`` per env
step (what the position controller tracks), grippers move after the arms arrive (at most
``GRIP_STEP`` of their range per step), and every env step commands both arms and both
grippers (an arm not commanded holds its last target). No controlr / torch imports here:
CPU-testable with a fake ``TASK_ENV`` (``tests/test_robodojo.py``).
"""

from __future__ import annotations

import copy
import time
import traceback
from multiprocessing.connection import Client
from typing import Any, Callable

import numpy as np

try:                                   # installed as XPolicyLab/policy/controlr/
    from . import protocol as P
except ImportError:                    # imported from the controlr tree (tests)
    from controlr.robot.robodojo import protocol as P  # type: ignore[no-redef]

ARMS = ("left", "right")
# Joint travel one env step delivers cleanly (RoboDojo's tracking, measured by the
# GPT_6_Astra_Direct_EEF adapter: <1 % of moves >100 mm off at 0.054 rad/step).
ARM_STEP_RAD = 0.05
GRIP_STEP = 0.25                       # gripper range fraction per env step (same source)
CV_FROM_USD = np.diag([1.0, -1.0, -1.0, 1.0])   # USD camera (-z forward, y up) -> OpenCV


def _np(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def resample_path(path: np.ndarray, steps: int) -> np.ndarray:
    """``steps`` waypoints along the planned joint path, ending on its target; the planner's
    first point (where the arm already is) is not a waypoint."""
    path = np.asarray(path, float)
    if len(path) < 2:
        return np.repeat(path[-1:], steps, axis=0)
    src = np.linspace(0.0, 1.0, len(path))
    dst = np.linspace(0.0, 1.0, steps + 1)[1:]
    return np.stack([np.interp(dst, src, path[:, j]) for j in range(path.shape[1])], axis=1)


def arm_steps(path: np.ndarray, step_rad: float = ARM_STEP_RAD) -> int:
    """Env steps that keep every joint under ``step_rad`` per step (total variation:
    a cuRobo path can double back)."""
    path = np.asarray(path, float)
    if len(path) < 2:
        return 1
    travel = np.abs(np.diff(path, axis=0)).sum(axis=0)
    return max(1, int(np.ceil(float(travel.max()) / step_rad - 1e-9)))


def default_planner(task_env: Any) -> Callable[[str, np.ndarray, np.ndarray], dict]:
    """RoboDojo's cuRobo planner for one arm (as GPT_6_Astra_Direct_EEF calls it)."""
    manager = task_env.robot_manager

    def plan(arm: str, flange: np.ndarray, q_start: np.ndarray) -> dict:
        robot = manager.get_robot_by_arm_name(f"{arm}_arm")
        planner = manager.planner.get(robot.robot_name)
        if planner is None:
            return {"status": "Unavailable"}
        return planner.plan_path(q_start, np.asarray(flange, float),
                                 real_robot_pose=copy.deepcopy(robot.entity_origin_pose))

    return plan


class Rig:
    """The robot side of one RoboDojo episode (env index 0)."""

    def __init__(self, task_env: Any, planner: Callable | None = None, video: bool = True) -> None:
        self.env = task_env
        self.plan = planner or default_planner(task_env)
        self.video = video                       # get_obs after every env step -> official mp4
        self.hold_q: dict[str, np.ndarray] = {}
        self.hold_g: dict[str, float] = {}
        self._last_state: dict | None = None

    # ------------------------------------------------------------------ facts
    def steps_used(self) -> int:
        cnt = getattr(self.env, "take_action_cnt", [0])
        return int(cnt[0])

    def step_lim(self) -> int:
        return int(getattr(self.env, "step_lim", 0) or 0)

    def ended(self) -> bool:
        return bool(getattr(self.env, "end_flag", [False])[0])

    def success(self) -> bool:
        return self.ended() and bool(getattr(self.env, "success", [False])[0])

    def episode(self, obs: dict) -> dict:
        env = self.env
        seeds = getattr(env, "env_seeds", None)
        layout = None
        try:
            layout = int(seeds[0]) if seeds is not None else None
        except (TypeError, ValueError, IndexError, KeyError):
            pass
        return {"task": str(getattr(env, "task_name", "") or ""), "layout_id": layout,
                "seed": getattr(env, "eval_seed", getattr(env, "seed", None)),
                "instruction": str(obs.get("instruction") or getattr(env, "instruction", [""])[0] or ""),
                "step_lim": self.step_lim(), "arms": list(ARMS),
                "cameras": sorted((obs.get("vision") or {}).keys()),
                "control_hz": float(getattr(getattr(env, "obs_manager", None), "collect_freq", 25.0) or 25.0)}

    # ------------------------------------------------------------- observing
    def _raw_obs(self) -> dict:
        raw = self.env.get_obs()
        st = raw.get("state") or {}
        self._last_state = st
        for arm in ARMS:                             # first observation: hold where we are
            if arm not in self.hold_q and f"{arm}_arm_joint_state" in st:
                self.hold_q[arm] = _np(st[f"{arm}_arm_joint_state"]).reshape(-1)
            if arm not in self.hold_g and f"{arm}_ee_joint_state" in st:
                self.hold_g[arm] = float(_np(st[f"{arm}_ee_joint_state"]).reshape(-1)[0])
        return raw

    def _cameras(self, names: list[str]) -> tuple[dict, dict]:
        K, T = {}, {}
        cm = getattr(self.env, "camera_manager", None)
        if cm is None:
            return K, T
        try:
            order = list(cm.camera_names[0])
        except Exception:  # noqa: BLE001
            return K, T
        for name in names:
            if name not in order:
                continue
            i = order.index(name)
            try:
                K[name] = _np(cm.get_camera_intrinsics(i, 0))
                T[name] = _np(cm.get_camera_extrinsics(i, 0)) @ CV_FROM_USD
            except Exception:  # noqa: BLE001 — overlays are optional; images are not
                pass
        return K, T

    def observe(self) -> dict:
        raw = self._raw_obs()
        vision = raw.get("vision") or {}
        images = {n: np.ascontiguousarray(np.asarray(v["color"])[..., :3]).astype(np.uint8)
                  for n, v in vision.items() if isinstance(v, dict) and "color" in v}
        K, T = self._cameras(sorted(images))
        st = raw.get("state") or {}
        arms = {}
        for arm in ARMS:
            arms[arm] = {"q": _np(st.get(f"{arm}_arm_joint_state", np.zeros(6))).reshape(-1),
                         "flange": _np(st.get(f"{arm}_ee_pose", np.zeros(7))).reshape(-1),
                         "grip": float(_np(st.get(f"{arm}_ee_joint_state", [self.hold_g.get(arm, 1.0)]))
                                       .reshape(-1)[0]),
                         "grip_cmd": float(self.hold_g.get(arm, 1.0))}
        return {"images": images, "K": K, "T_world_cam": T, "arms": arms,
                "instruction": raw.get("instruction"), "steps_used": self.steps_used(),
                "step_lim": self.step_lim(), "ended": self.ended(), "success": self.success()}

    def _measured_q(self, arm: str) -> np.ndarray:
        st = self._last_state or {}
        if f"{arm}_arm_joint_state" in st:
            return _np(st[f"{arm}_arm_joint_state"]).reshape(-1)
        return self.hold_q[arm]

    # -------------------------------------------------------------- execution
    def _action(self) -> dict:
        d = {}
        for arm in ARMS:
            d[f"{arm}_arm_joint_state"] = np.asarray(self.hold_q[arm], np.float32)
            d[f"{arm}_ee_joint_state"] = np.asarray([self.hold_g[arm]], np.float32)
        return d

    def _step_env(self) -> bool:
        """One env step with the current hold targets; True when RoboDojo ended the episode."""
        self.env.take_action(self._action())
        if self.video:
            self._raw_obs()
        return bool(self.env.is_episode_end())

    def execute(self, steps: list[dict], arm_step_rad: float = ARM_STEP_RAD,
                grip_step: float = GRIP_STEP) -> dict:
        t0, n0 = time.perf_counter(), self.steps_used()
        if not self.hold_q:
            self._raw_obs()
        out_steps = []
        ended = self.ended()
        for step in steps:
            if ended:
                break
            rec: dict = {"arms": {}}
            paths: dict[str, np.ndarray] = {}
            for arm, cmd in step.items():
                if arm not in ARMS:
                    rec["arms"][arm] = {"status": "unknown arm"}
                    continue
                if cmd.get("flange") is None:
                    continue
                q0 = self._measured_q(arm)
                try:
                    res = self.plan(arm, np.asarray(cmd["flange"], float), q0)
                except Exception as e:  # noqa: BLE001 — a planner crash is this arm's failure
                    res = {"status": f"error: {type(e).__name__}: {e}"}
                status = str(res.get("status", "Fail"))
                pos = res.get("position")
                if status != "Success" or pos is None or len(_np(pos)) == 0:
                    rec["arms"][arm] = {"status": status, "waypoints": 0}
                    continue
                paths[arm] = _np(pos).reshape(-1, 6)
                rec["arms"][arm] = {"status": "Success"}
            n_arm = max((arm_steps(p, arm_step_rad) for p in paths.values()), default=0)
            grips = {arm: float(np.clip(cmd["grip"], 0.0, 1.0)) for arm, cmd in step.items()
                     if arm in ARMS and cmd.get("grip") is not None}
            n_grip = max((int(np.ceil(abs(g - self.hold_g[arm]) / grip_step - 1e-9)) for arm, g in grips.items()),
                         default=0)
            for arm, path in paths.items():
                rec["arms"][arm]["waypoints"] = n_arm
            way = {arm: resample_path(p, n_arm) for arm, p in paths.items()}
            done = 0
            for i in range(n_arm):
                for arm, w in way.items():
                    self.hold_q[arm] = w[i]
                done += 1
                if self._step_env():
                    ended = True
                    break
            if not ended and n_grip:                  # a gripper already where it was told costs nothing
                for arm, g in grips.items():          # gripper after the arm has arrived
                    self.hold_g[arm] = g
                for _ in range(n_grip):
                    done += 1
                    if self._step_env():
                        ended = True
                        break
            rec["env_steps"] = done
            for arm, path in paths.items():      # run-log diagnostics: plan end vs where the arm is
                rec["arms"][arm]["q_goal"] = [round(float(v), 4) for v in path[-1]]
                rec["arms"][arm]["q_after"] = [round(float(v), 4) for v in self._measured_q(arm)]
            out_steps.append(rec)
        return {"steps": out_steps, "env_steps": self.steps_used() - n0, "ended": self.ended(),
                "success": self.success(), "steps_used": self.steps_used(), "step_lim": self.step_lim(),
                "wall_s": time.perf_counter() - t0,
                "sim_s": (self.steps_used() - n0) / float(getattr(getattr(self.env, "obs_manager", None),
                                                                   "collect_freq", 25.0) or 25.0)}

    def check_goal(self) -> dict:
        return {"success": self.success(), "ended": self.ended(), "steps_used": self.steps_used(),
                "step_lim": self.step_lim()}

    def finish(self) -> None:
        """controlr is done with the episode: one RoboDojo has not ended counts as failed
        (what RoboDojo's own LLM adapter does when it stops early)."""
        if self.env.is_episode_end():
            return
        for i in range(int(getattr(self.env, "num_envs", 1) or 1)):
            self.env.success[i] = False
        self.env.is_episode_end()


def serve(conn, rig: Rig) -> str:
    """Answer controlr's requests until it says ``done``; returns its outcome."""
    while True:
        try:
            req = conn.recv()
        except EOFError:
            return "disconnected"
        t0 = time.perf_counter()
        op, args = req.get("op"), req.get("args") or {}
        try:
            if op == "observe":
                conn.send(P.ok(rig.observe(), t0))
            elif op == "execute":
                conn.send(P.ok(rig.execute(list(args.get("steps") or []),
                                           float(args.get("arm_step_rad") or ARM_STEP_RAD),
                                           float(args.get("grip_step") or GRIP_STEP)), t0))
            elif op == "check_goal":
                conn.send(P.ok(rig.check_goal(), t0))
            elif op == "done":
                conn.send(P.ok({"steps_used": rig.steps_used()}, t0))
                return str(args.get("outcome", ""))
            else:
                conn.send(P.error(f"unknown op {op!r}"))
        except Exception as e:  # noqa: BLE001 — report, keep serving
            conn.send(P.error(f"{type(e).__name__}: {e}", traceback.format_exc()))


def eval_one_episode(TASK_ENV: Any, model_client: Any = None, *, planner: Callable | None = None) -> None:
    del model_client                  # no served model: the controller is controlr
    rig = Rig(TASK_ENV, planner=planner)
    first = rig._raw_obs()
    conn = Client(P.address_from_env(), authkey=P.authkey_from_env())
    try:
        conn.send({"op": "hello", "episode": rig.episode(first), "version": P.PROTOCOL_VERSION})
        outcome = serve(conn, rig)
        print(f"[controlr] episode over: {outcome} (steps {rig.steps_used()}/{rig.step_lim()}, "
              f"success={rig.success()})", flush=True)
    finally:
        try:
            rig.finish()
        finally:
            conn.close()


def eval_one_episode_batch(TASK_ENV: Any, model_client: Any = None) -> None:
    n = int(getattr(TASK_ENV, "num_envs", 1) or 1)
    if n > 1:
        raise RuntimeError(f"controlr runs one conversation per env; RoboDojo started {n} (eval_batch must be false)")
    eval_one_episode(TASK_ENV, model_client)


__all__ = ["Rig", "eval_one_episode", "eval_one_episode_batch", "serve", "resample_path", "arm_steps"]
