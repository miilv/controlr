"""One-call probe of the decision head on a reach task (live calls; OPENROUTER_API_KEY).

Backends: mock (red ball) or isaac (reach target stated relative to the packet / box; run on
compute3 — the client launches the Isaac server and stops it at the end).
For N scenes (random target, random start pose) it asks the decision head ONCE, as on turn 0,
and scores the step against the true direction to the ball — no episodes, ~0.4 s per call, so
question wording / reduction / overlays / levels can be compared cheaply before episodes.

    uv run python scripts/decisions_probe.py -c configs/mock_dec.yaml -n 24 [--set k=v ...] [--tag name]

Per sample: true delta d (TCP -> ball, mm), step s (mm). Scores: cos(s, d) (0 for no motion),
per-axis sign accuracy on axes with |d_axis| >= 15 mm (a 0 step counts wrong), share of
non-motion replies, and the mean of the reduced |step|. Writes runs/probe_<UTC>_<tag>.jsonl.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controlr.config import load_config  # noqa: E402
from controlr.llm.decisions import DecisionsClient  # noqa: E402
from controlr.loop import _user_parts, turn0_text  # noqa: E402
from controlr.llm.transcript import Transcript  # noqa: E402
from controlr.observation.renderers import ObservationRenderer  # noqa: E402
from controlr.prompts.builder import build_system_prompt  # noqa: E402
from controlr.protocol.feedback import format_feedback  # noqa: E402
from controlr.robot import make_robot  # noqa: E402
from controlr.robot.safety import make_envelope  # noqa: E402
from controlr.types import Action, ActionMode  # noqa: E402


def _target(robot) -> np.ndarray:
    sc = getattr(robot, "_scene", None)
    if sc is not None and getattr(sc, "target", None) is not None:        # mock
        return np.asarray(sc.target, float)
    return np.asarray(robot.scene_record()["reach_target"]["point_mm"], float) / 1000.0   # isaac


def sample(cfg, i: int, robot=None, aim_fn=None):
    """(messages, true delta mm, (cameras, ref tcp)) for sample i: reset with seed i, then a random
    start move."""
    robot = robot or make_robot(cfg)
    obs = robot.reset(cfg.task.__dict__ | {"params": dict(cfg.task.params)}, seed=1000 + i)
    ref = obs.state
    rng = np.random.default_rng(i)
    env = make_envelope(robot.spec, cfg.safety, rotation=cfg.action.rotation)
    env.reset(ref)
    # BALANCED start: the TCP goes to aim + offset with a random sign per axis (30-90 mm), so every
    # relation (left/right, toward/away, up/down) is ~50/50 over samples. Small random moves from
    # the home pose left the target on the same side in nearly every sample (2026-10-07 probes).
    aim = aim_fn(robot) if aim_fn else _target(robot)
    off = rng.choice([-1.0, 1.0], 3) * rng.uniform(0.03, 0.09, 3)
    goal = aim + off
    tz = robot.spec.table_z if robot.spec.table_z is not None else 0.0
    goal[2] = max(goal[2], tz + 0.06)
    for _ in range(6):                                   # the envelope limits each step
        st = robot.state()
        delta = goal - st.tcp_pos
        if np.linalg.norm(delta) < 0.005:
            break
        acts, _ = env.filter([Action(ActionMode.EE_DELTA, tuple(delta), raw="probe start")], st)
        if not acts:
            break
        robot.execute(acts)
    obs = robot.observe()
    system = build_system_prompt(cfg, robot.spec, cameras=obs.cameras, state0=ref, obstacles=[])
    rendered = ObservationRenderer(cfg.observation, cfg.action, robot.spec).render(obs, None, 0)
    fb0 = format_feedback(0, None, None, None, obs, cfg, robot.spec)
    tr = Transcript(system)
    tr.add_user(_user_parts(turn0_text(robot.task_instruction, None, fb0), rendered))
    d = (_target(robot) - obs.state.tcp_pos) * 1000.0
    return tr.to_messages(), d, (obs.cameras, ref.tcp_pos, robot.task_instruction)


def score(d: np.ndarray, s: np.ndarray) -> dict:
    nd, ns = np.linalg.norm(d), np.linalg.norm(s)
    cos = float(s @ d / (nd * ns)) if ns > 0 else 0.0
    axes = [k for k in range(3) if abs(d[k]) >= 15]
    sign = [float(np.sign(s[k]) == np.sign(d[k])) for k in axes]
    return {"cos": cos, "sign_hits": sum(sign), "sign_n": len(axes), "still": float(ns == 0), "step": float(ns)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-n", type=int, default=24)
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    cfg = load_config(a.config, a.sets)
    key = os.environ.get(cfg.decisions.api_key_env, "")
    client = DecisionsClient(cfg.decisions.base_url, key, cfg, 60, 2)
    robot = make_robot(cfg)
    try:
        samples = [sample(cfg, i, robot) for i in range(a.n)]
    finally:
        getattr(robot, "close", lambda: None)()
    client.set_scene(*samples[0][2][:2])     # the cameras are fixed: one scene for all samples
    t0 = time.perf_counter()

    def ask(i):
        msgs, d, (_, _, task) = samples[i]
        c = copy.copy(client)                   # per-sample task text ({task} in decisions.target)
        c.task = task
        res = c.complete(cfg.llm.model, msgs)
        if res.error:
            return {"i": i, "error": res.error}
        steps = res.decisions["steps"]
        s = np.array([steps["dx"], steps["dy"], steps["dz"]], float)
        if cfg.action.pos_unit == "cm":
            s *= 10
        return {"i": i, "d": [round(x, 1) for x in d], "s": s.tolist(), "reply": res.text,
                "latency": res.timings.t_wall, "cost": (res.usage.raw or {}).get("cost", 0),
                "answers": res.decisions["answers"], **score(d, s)}
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(ask, range(a.n)))
    ok = [r for r in rows if "error" not in r]
    out = Path(cfg.log.root) / f"probe_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{a.tag or 'run'}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cos = [r["cos"] for r in ok]
    hits, n = sum(r["sign_hits"] for r in ok), sum(r["sign_n"] for r in ok)
    print(f"{a.tag or 'run'}: n={len(ok)}/{a.n} errors={len(rows) - len(ok)}  "
          f"cos mean {st.mean(cos):+.2f} (sd {st.pstdev(cos):.2f})  sign {hits}/{n}={hits / max(n, 1):.2f}  "
          f"still {st.mean(r['still'] for r in ok):.2f}  |step| {st.mean(r['step'] for r in ok):.1f} mm  "
          f"latency med {st.median(r['latency'] for r in ok):.2f}s  cost ${sum(r['cost'] for r in ok):.4f}  "
          f"wall {time.perf_counter() - t0:.0f}s  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
