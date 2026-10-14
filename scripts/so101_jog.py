"""First-motion check of the SO-101 rig, no LLM: home, then a small move along +x, +y, +z
(and back) with a frame after each, plus a gripper close/open. Prints the measured TCP after
every step and saves the frames (with the VIEW line burnt in) to OUT_DIR, so a human can check
that the arm goes where the numbers say and the camera mount / VIEW directions are right.

    uv run python scripts/so101_jog.py [OUT_DIR] [--step-mm 20] [--port 7811]

A human stays at the arm's power switch. Ctrl-C here = the arm holds where it is.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from controlr.config import SafetyConfig
from controlr.robot.safety import make_envelope
from controlr.robot.so101.client import So101Robot
from controlr.types import Action, ActionMode


def save(obs, out: Path, name: str) -> None:
    img = Image.fromarray(obs.images["wrist"])
    d = ImageDraw.Draw(img)
    text = (obs.notes[0] if obs.notes else "").replace("VIEW: ", "")
    d.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
    d.text((4, 4), f"{name}: tcp {np.round(obs.state.tcp_pos * 1000).astype(int).tolist()} mm | {text}"[:200],
           fill=(255, 255, 0))
    img.save(out / f"{name}.jpg", quality=90)


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=f"runs/so101_jog_{time.strftime('%Y%m%dT%H%M%S')}")
    ap.add_argument("--step-mm", type=float, default=20.0)
    ap.add_argument("--port", type=int, default=7811)
    ap.add_argument("--tcp-speed", type=float, default=0.03)
    ap.add_argument("--no-gripper", action="store_true")
    ap.add_argument("--workspace", default=None, help="lo_x,lo_y,lo_z,hi_x,hi_y,hi_z in m (narrower than the spec)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    params = {"port": a.port, "reset_duration_s": 5.0}
    if a.workspace:
        w = [float(v) for v in a.workspace.split(",")]
        params.update(workspace_lo=w[:3], workspace_hi=w[3:])
    robot = So101Robot(params, tcp_speed_m_s=a.tcp_speed)
    print("workspace mm:", np.round(np.array(robot.spec.workspace_lo) * 1000).tolist(),
          np.round(np.array(robot.spec.workspace_hi) * 1000).tolist())
    print("server:", {k: robot.info[k] for k in ("fake", "camera", "max_joint_speed")})
    print("limits deg:", np.degrees(robot.limits).round(1).tolist())
    print("home q deg:", np.degrees(robot.home_q).round(1).tolist())
    st = robot.state()
    print("now  q deg:", np.degrees(st.q).round(1).tolist(), "tcp mm", np.round(st.tcp_pos * 1000).tolist())
    input("Arm will move to HOME slowly. Hand on the power switch, then press Enter (Ctrl-C aborts) ")
    try:
        obs = robot.reset({})
        save(obs, out, "00_home")
        print("home tcp mm", np.round(obs.state.tcp_pos * 1000).tolist(), "|", obs.notes[0] if obs.notes else "")
        env = make_envelope(robot.spec, SafetyConfig(max_step_m=0.05, table_clearance_m=0.01))
        env.reset(robot.reference_state())
        s = a.step_mm / 1000.0
        steps = [("x+", (s, 0, 0)), ("x-", (-s, 0, 0)), ("y+", (0, s, 0)), ("y-", (0, -s, 0)),
                 ("z+", (0, 0, s)), ("z-", (0, 0, -s))]
        for i, (name, d) in enumerate(steps, 1):
            before = robot.state()
            acts, events = env.filter([Action(ActionMode.EE_DELTA, d, raw=f"jog {name}")], before)
            rep = robot.execute(acts)
            obs = robot.observe()
            moved = (obs.state.tcp_pos - before.tcp_pos) * 1000
            print(f"{name}: asked {np.round(np.array(d) * 1000).tolist()} mm, moved {np.round(moved, 1).tolist()} mm"
                  f" | events {[e.brief or e.message for e in events + rep.events]}")
            save(obs, out, f"{i:02d}_{name}")
            if rep.stopped:
                print("STOPPED - the arm was blocked; aborting")
                return
        if not a.no_gripper:
            for i, g in ((7, 0.0), (8, robot.gmax_m)):
                rep = robot.execute([Action(None, None, gripper=g, raw="jog grip")])
                obs = robot.observe()
                print(f"gripper -> {g * 1000:.0f} mm: measured {obs.state.gripper_mm:.0f} mm")
                save(obs, out, f"{i:02d}_grip_{int(g * 1000)}")
        print(f"frames in {out}")
    except KeyboardInterrupt:
        print("interrupted: holding")
    finally:
        robot.hold()
        robot.close()


if __name__ == "__main__":
    main(sys.argv[1:])
