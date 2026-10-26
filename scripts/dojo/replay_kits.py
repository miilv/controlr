"""Physics divergence between the RENDERED kit (the eval client's app flags, Isaac 5.1 RTX) and the PHYSICS-ONLY kit
(our twin's): replay one episode's EXACT executed action list open-loop on the same settled dev layout and record every
task object's pose after every env step.

    scripts/simlab/rdpy.sh -u scripts/dojo/replay_kits.py run --layout L.settled.json --actions A_actions.json.gz \\
        --out DIR/a_rtx.json [--physics-only] [--update-to-usd] [--no-fabric-ext] [--obs-every K]
    python3 scripts/dojo/replay_kits.py compare DIR/a_rtx.json DIR/b_phys.json [--actions A_actions.json.gz]

``run`` (RoboDojo's python): ``RoboDojoRunner(physics_only=...)``, ``load(layout)``, then ``step(a)`` for every
action, recording per step the labelled non-Geometry objects (pos + quat wxyz), each arm's measured flange and the
commanded gripper. ``--update-to-usd`` forces ``/physics/updateToUsd`` on after the app starts (the rendered kit's
value); ``--no-fabric-ext`` disables ``omni.physx.fabric`` (loaded only in the physics-only kit). ``compare`` (any
python): max position / orientation divergence over time, the first step above 1 mm / 0.1 deg and the phase of the
action list there (from the commanded gripper and flange height: approach, close, lift, carry, release, ...).
Dev layouts only (the runner's asset firewall).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

ARMS = ("left", "right")


def _load_actions(p: str) -> list[dict]:
    with gzip.open(p, "rt") as f:
        return json.load(f)["actions"]


def run_cmd(a) -> None:
    from dojo.simlab import paths
    from dojo.simlab.runner import RoboDojoRunner
    paths.assert_not_eval(a.layout)
    out = Path(a.out)
    paths.assert_not_eval(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    acts = _load_actions(a.actions)
    run = RoboDojoRunner(a.task, workdir=str(out.parent / f"work_{out.stem}"), physics_only=a.physics_only,
                         need_planner=False)
    import carb
    import omni.kit.app
    st = carb.settings.get_settings()
    em = omni.kit.app.get_app().get_extension_manager()
    if a.update_to_usd:
        st.set_bool("/physics/updateToUsd", True)
    if a.no_fabric_ext and em.is_extension_enabled("omni.physx.fabric"):
        em.set_extension_enabled_immediate("omni.physx.fabric", False)
    info = run.load(json.loads(Path(a.layout).read_text()))
    settings = {k: st.get(k) for k in ("/physics/updateToUsd", "/physics/fabricEnabled", "/physics/updateVelocitiesToUsd")}
    settings["physx_fabric_ext"] = bool(em.is_extension_enabled("omni.physx.fabric"))

    def snap(k: int) -> dict:
        objs = {o["label"]: [float(v) for v in list(o["pos"]) + list(o["quat"])]
                for o in run.objects_gt(0) if o.get("label") and not o.get("clutter") and o.get("type") != "Geometry"}
        s = run.obs_state(0)
        return {"step": k, "objects": objs,
                "flange": {arm: [float(v) for v in np.asarray(s[f"{arm}_ee_pose"]).reshape(-1)[:7]] for arm in ARMS},
                "grip_cmd": {arm: float(np.asarray(s[f"{arm}_ee_joint_state"]).reshape(-1)[0]) for arm in ARMS}}

    frames = [snap(0)]
    save_at = {int(x) for x in (a.save_obs_at or "").split(",") if x.strip()}
    saved = {}

    def save_obs(k: int) -> None:
        """The contract observation after k executed actions (3 RGB PNGs + the official state), as the policy got it."""
        from PIL import Image
        o = run.obs(0)
        d = out.parent / f"{Path(a.layout).name.split('.')[0]}_s{k:04d}"
        d.mkdir(parents=True, exist_ok=True)
        for cam, im in o["images"].items():
            Image.fromarray(np.asarray(im, np.uint8)).save(d / f"{cam}.png")
        (d / "state.json").write_text(json.dumps({kk: np.asarray(v, float).reshape(-1).tolist()
                                                  for kk, v in o["state"].items()}, indent=1))
        saved[k] = {"dir": str(d), "objects_true": frames[-1]["objects"]}

    if 0 in save_at:
        save_obs(0)
    t0 = time.perf_counter()
    ended_at = None
    for k, act in enumerate(acts, 1):
        if a.obs_every and not a.physics_only and (k - 1) % a.obs_every == 0:
            run.obs(0)                              # an official observation (render_for_capture + get_obs) first
        if run.step({kk: np.asarray(v, np.float32) for kk, v in act.items()})[0] and ended_at is None:
            ended_at = k
        frames.append(snap(k))
        if k in save_at and not a.physics_only:
            save_obs(k)
        if ended_at is not None:
            break
    res = {"layout": a.layout, "actions": a.actions, "physics_only": a.physics_only, "update_to_usd": a.update_to_usd,
           "obs_every": a.obs_every,
           "no_fabric_ext": a.no_fabric_ext, "settings": settings, "stable": info["stable"], "steps": len(frames) - 1,
           "ended_at": ended_at, "replay_s": round(time.perf_counter() - t0, 1), "reward": run.reward(0),
           "frames": frames, "saved_obs": saved}
    out.write_text(json.dumps(res))
    print(f"[replay] {out.name}: {res['steps']} steps, ended_at {ended_at}, settings {settings}, reward {res['reward']}",
          flush=True)
    sys.stdout.flush()
    try:
        run.close()
    finally:
        import os
        os._exit(0)


def _quat_deg(q1, q2) -> float:
    d = abs(float(np.dot(np.asarray(q1, float) / np.linalg.norm(q1), np.asarray(q2, float) / np.linalg.norm(q2))))
    return float(np.degrees(2 * np.arccos(min(1.0, d))))


def phases(frames: list[dict]) -> list[str]:
    """Per step: what the action list was doing, from the commanded gripper (1 open .. 0 closed) and the flange z."""
    out = []
    for i, f in enumerate(frames):
        tags = []
        prev = frames[i - 1] if i else f
        for arm in ARMS:
            g, g0 = f["grip_cmd"][arm], prev["grip_cmd"][arm]
            dz = f["flange"][arm][2] - prev["flange"][arm][2]
            moving = np.linalg.norm(np.subtract(f["flange"][arm][:3], prev["flange"][arm][:3])) > 1e-4
            if g < g0 - 1e-3:
                tags.append(f"{arm}:close")
            elif g > g0 + 1e-3:
                tags.append(f"{arm}:release")
            elif moving:
                closed = g < 0.5
                tags.append(f"{arm}:" + ("lift" if closed and dz > 1e-4 else "carry" if closed else
                                         "descend" if dz < -1e-4 else "approach"))
        out.append(",".join(tags) or "hold")
    return out


def compare_cmd(a) -> None:
    A, B = (json.loads(Path(p).read_text()) for p in (a.a, a.b))
    fa, fb = A["frames"], B["frames"]
    n = min(len(fa), len(fb))
    ph = phases(fa)
    per_step, first_mm, first_deg, max_mm, max_deg, worst = [], None, None, 0.0, 0.0, None
    bitwise = len(fa) == len(fb) and all(x["objects"] == y["objects"] for x, y in zip(fa, fb))
    for k in range(n):
        s_mm, s_deg, s_lab = 0.0, 0.0, None
        for lab, pa in fa[k]["objects"].items():
            pb = fb[k]["objects"].get(lab)
            if pb is None:
                continue
            dm = float(np.linalg.norm(np.subtract(pa[:3], pb[:3]))) * 1000
            dd = _quat_deg(pa[3:], pb[3:])
            if dm > s_mm:
                s_mm, s_lab = dm, lab
            s_deg = max(s_deg, dd)
        per_step.append((round(s_mm, 4), round(s_deg, 4), s_lab))
        if first_mm is None and s_mm > 1.0:
            first_mm = k
        if first_deg is None and s_deg > 0.1:
            first_deg = k
        if s_mm > max_mm:
            max_mm, worst = s_mm, k
        max_deg = max(max_deg, s_deg)
    rep = {"a": a.a, "b": b_name(a.b), "steps_a": len(fa) - 1, "steps_b": len(fb) - 1, "bitwise_equal": bitwise,
           "max_mm": round(max_mm, 3), "max_deg": round(max_deg, 3), "worst_step": worst,
           "first_step_over_1mm": first_mm, "first_step_over_0.1deg": first_deg,
           "phase_at_first_1mm": ph[first_mm] if first_mm is not None else None,
           "object_at_first_1mm": per_step[first_mm][2] if first_mm is not None else None,
           "reward_a": A.get("reward"), "reward_b": B.get("reward"), "settings_a": A.get("settings"),
           "settings_b": B.get("settings"),
           "first_nonzero_step": next((k for k, (m, d, _) in enumerate(per_step) if m > 0 or d > 0), None)}
    if first_mm is not None:
        lo = max(0, first_mm - 6)
        rep["around_first_1mm"] = [{"step": k, "mm": per_step[k][0], "deg": per_step[k][1], "obj": per_step[k][2],
                                    "phase": ph[k]} for k in range(lo, min(n, first_mm + 4))]
    print(json.dumps(rep, indent=1))


def b_name(p: str) -> str:
    return p


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--task", default="stack_blocks")
    r.add_argument("--layout", required=True)
    r.add_argument("--actions", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--physics-only", action="store_true")
    r.add_argument("--update-to-usd", action="store_true")
    r.add_argument("--no-fabric-ext", action="store_true")
    r.add_argument("--obs-every", type=int, default=0, help="rendered only: an observation before every k-th step")
    r.add_argument("--save-obs-at", default="", help="rendered only: save the contract observation after these step "
                   "counts (comma list) to <out dir>/<layout>_s<step>/{cam_*.png,state.json}")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args(argv)
    (run_cmd if a.cmd == "run" else compare_cmd)(a)


if __name__ == "__main__":
    main()
