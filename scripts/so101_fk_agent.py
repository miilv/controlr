"""Few-shot forward-kinematics agent for the SO-101 (docs/FK_AGENT.md).

    # offline (no arm): leave-one-out on the demo sessions, real LLM
    uv run python scripts/so101_fk_agent.py offline --dataset runs/fk_dataset --out runs/fk_offline
    # the request the model would get (no LLM call)
    uv run python scripts/so101_fk_agent.py prompt --dataset runs/fk_dataset --session gym_objects_demo_2#2
    # live / fake server: move to the start pose and save its photo only (no LLM)
    uv run python scripts/so101_fk_agent.py survey --port 7811
    # live episode
    uv run python scripts/so101_fk_agent.py live --dataset runs/fk_dataset --fewshot all \
        --task "pick up the plush carrot" --object-xyz 230,-40,30 --port 7811

LLM: OMNIROUTE_BASE_URL / OMNIROUTE_API_KEY from the environment (.env); ``--fake-llm`` replies
with a fixed safe target (plumbing tests). Arm: the SO-101 server (CONTROLR_SO101_AUTHKEY).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controlr.fkagent import geometry as G  # noqa: E402
from controlr.fkagent import offline as O  # noqa: E402
from controlr.fkagent import prompt as Pm  # noqa: E402
from controlr.fkagent.agent import AgentConfig, FkAgent, ServerArm  # noqa: E402
from controlr.fkagent.dataset import load_sessions  # noqa: E402

MODEL = "claude/claude-opus-5-5"


def make_llm(fake: bool):
    if fake:
        from controlr.llm import FakeLLM

        q = [0.0, -20.0, 45.0, 60.0, -90.0]
        return FakeLLM([json.dumps({"object": {"xyz_mm": [230, 0, 20], "uncertainty_mm": 50, "visible": False},
                                    "action": "move", "q_deg": q, "gripper_open_pct": 35,
                                    "expected_tcp_mm": G.tcp_mm(q).round(0).tolist(), "reason": "fake"})])
    from controlr.llm.client import LLMClient

    base, key = os.environ.get("OMNIROUTE_BASE_URL", ""), os.environ.get("OMNIROUTE_API_KEY", "")
    if not base.startswith("http") or not key:
        raise SystemExit("OMNIROUTE_BASE_URL / OMNIROUTE_API_KEY not set (load .env)")
    return LLMClient(base, key, timeout_s=400.0, max_retries=2, usage_grace_s=0.5)


def pick_fewshot(sessions, spec: str, exclude: str = "") -> list[dict]:
    if spec in ("none", "0", ""):
        return []
    pool = [s for s in sessions if s["session_id"] != exclude]
    if spec == "all":
        return pool
    ids = spec.split(",")
    return [s for s in pool if s["session_id"] in ids or s["recording"] in ids]


def _xyz(s):
    return None if not s else tuple(float(v) for v in s.split(","))


def cmd_offline(a) -> int:
    sessions, root = load_sessions(a.dataset)
    llm = make_llm(a.fake_llm)
    res = O.run(sessions, root, a.out, llm, model=a.model, conditions=a.conditions.split(","),
                views=tuple(a.views.split(",")), workers=a.workers, only=a.only.split(",") if a.only else None)
    rep = O.report(res)
    Path(a.out, "report.md").write_text(rep + "\n")
    calls = sum(len(r["llm"]) for r in res)
    tok = sum((c.get("prompt_tokens") or 0) for r in res for c in r["llm"])
    out_tok = sum((c.get("completion_tokens") or 0) for r in res for c in r["llm"])
    print("\n" + rep + f"\n\nLLM: {calls} calls, {tok} prompt tokens, {out_tok} completion tokens -> {a.out}")
    return 0


def cmd_prompt(a) -> int:
    sessions, root = load_sessions(a.dataset)
    q = next(x for x in O.queries(sessions, (a.view,)) if x["session"]["session_id"] == a.session)
    system, parts = O.build_query(q, a.condition, sessions, root, lim=G.Limits(**O.OFFLINE_LIMITS))
    print(system)
    print("=" * 80)
    for p in parts:
        print(p if isinstance(p, str) else f"[image {p.label} {p.width}x{p.height} {len(p.jpeg)} B]")
    return 0


def _arm(a) -> ServerArm:
    return ServerArm(a.host, a.port, tcp_speed_mm_s=a.tcp_speed)


def _limits(a, arm) -> G.Limits:
    kw = {}
    if a.box_x:
        x = _xyz(a.box_x)
        kw["box_lo_mm"], kw["box_hi_mm"] = (x[0], -150.0, 5.0), (x[1], 150.0, 260.0)
    return arm.limits(max_step_mm=a.max_step_mm, **kw)


def cmd_survey(a) -> int:
    arm = _arm(a)
    try:
        lim = _limits(a, arm)
        start = _xyz(a.start_q) or G.SURVEY_Q_DEG
        why = G.check_path([start], lim)
        if why:
            raise SystemExit(f"start pose breaks the limits: {why}")
        q0, _ = arm.state()
        why = G.check_path([q0, start], lim)          # e.g. rest -> survey: checked, never a blind joint move
        if why:
            raise SystemExit(f"the path to the start pose is not safe: {why}")
        r = arm.move([q0, start], a.gripper_open)
        jpeg, q, g = arm.observe()
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "survey.jpg").write_bytes(jpeg)
        print(f"stopped={r['stopped']} q_deg={q.round(1).tolist()} {G.pose_summary(q)} gripper={g:.0f} "
              f"-> {out / 'survey.jpg'}")
    finally:
        arm.close()
    return 0


def cmd_rest(a) -> int:
    """Start pose -> the server's rest pose (joint move, jaws nearly closed)."""
    import numpy as np

    arm = _arm(a)
    try:
        rest_meas = np.degrees(np.asarray(arm.info["rest_q"], float))
        q, _ = arm.state()
        print(f"from {q.round(1).tolist()} (model) to rest {(rest_meas + arm.offsets).round(1).tolist()} (model)")
        lim = _limits(a, arm)
        why = G.check_path([q, rest_meas + arm.offsets], lim)
        if why:
            raise SystemExit(f"the path to the rest pose is not safe: {why}")
        r = arm.move([q, rest_meas + arm.offsets], 2.0)
        q, g = arm.state()
        print(f"stopped={r['stopped']} q_deg={q.round(1).tolist()} gripper={g:.0f}")
    finally:
        arm.close()
    return 0


def cmd_live(a) -> int:
    sessions, root = load_sessions(a.dataset) if a.dataset else ([], Path("."))
    shots = pick_fewshot(sessions, a.fewshot)
    arm = _arm(a)
    try:
        lim = _limits(a, arm)
        cfg = AgentConfig(task=a.task, model=a.model, max_steps=a.max_steps,
                          start_q_deg=_xyz(a.start_q) or G.SURVEY_Q_DEG, object_xyz_mm=_xyz(a.object_xyz),
                          fewshot=shots, fewshot_root=str(root), release_at_end=a.release_at_end,
                          gripper_open_max=a.gripper_max, gripper_open_default=min(30.0, a.gripper_max),
                          place_back=not a.no_place_back, loop=a.loop, place_xyz_mm=_xyz(a.place_xyz),
                          carry_min_z_mm=a.carry_min_z, max_tokens=a.max_tokens,
                          extra_body={"reasoning_effort": a.effort} if a.effort else {})
        tag = time.strftime("%Y%m%d-%H%M%S") + f"_{a.loop[:4]}_{'fs' if shots else 'zero'}_{a.name}"
        agent = FkAgent(arm, make_llm(a.fake_llm), cfg, Path(a.log_root) / tag, limits=lim)
        print(f"log -> {agent.dir}; few-shot sessions: {[s['session_id'] for s in shots]}; fake arm: {arm.fake}")
        summary = agent.run()
        print(json.dumps(summary, indent=1))
    finally:
        arm.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    o = sub.add_parser("offline", help="leave-one-out evaluation on the demo sessions")
    o.add_argument("--dataset", required=True)
    o.add_argument("--out", default="runs/fk_offline")
    o.add_argument("--conditions", default=",".join(O.CONDITIONS))
    o.add_argument("--views", default="approach,pre_grasp")
    o.add_argument("--only", default="", help="comma-separated session ids")
    o.add_argument("--workers", type=int, default=4)
    o.add_argument("--model", default=MODEL)
    o.add_argument("--fake-llm", action="store_true")
    o.set_defaults(fn=cmd_offline)

    p = sub.add_parser("prompt", help="print one offline request (no LLM call)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--view", default="pre_grasp")
    p.add_argument("--condition", default="loo", choices=O.CONDITIONS)
    p.set_defaults(fn=cmd_prompt)

    for name, fn, help_ in (("survey", cmd_survey, "move to the start pose, save its photo (no LLM)"),
                            ("rest", cmd_rest, "move to the server's rest pose (no LLM)"),
                            ("live", cmd_live, "run one episode on the arm server")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--host", default="127.0.0.1")
        s.add_argument("--port", type=int, default=7811)
        s.add_argument("--start-q", default="", help="start pose, MODEL deg 'pan,lift,elbow,wflex,roll'")
        s.add_argument("--box-x", default="", help="TCP workspace x range 'lo,hi' mm (default 140,270)")
        s.add_argument("--max-step-mm", type=float, default=80.0)
        s.add_argument("--tcp-speed", type=float, default=50.0, help="mm/s along the path (safety max 50)")
        if name == "rest":
            pass
        elif name == "survey":
            s.add_argument("--out", default="runs/fk_agent/survey")
            s.add_argument("--gripper-open", type=float, default=30.0)
        else:
            s.add_argument("--dataset", default="")
            s.add_argument("--fewshot", default="all", help="all | none | comma-separated session ids/recordings")
            s.add_argument("--task", required=True)
            s.add_argument("--object-xyz", default="", help="operator-measured object centre 'x,y,z' mm (metrics)")
            s.add_argument("--max-steps", type=int, default=8)
            s.add_argument("--name", default="ep")
            s.add_argument("--log-root", default="runs/fk_agent")
            s.add_argument("--model", default=MODEL)
            s.add_argument("--release-at-end", action="store_true")
            s.add_argument("--gripper-max", type=float, default=35.0, help="widest jaw opening, %% (camera clearance)")
            s.add_argument("--no-place-back", action="store_true", help="return loop: keep holding after a grasp")
            s.add_argument("--loop", default="continuous", choices=("continuous", "plan_once", "return"),
                           help="continuous: re-plan from the current pose, start pose only when done; "
                                "return: back to the start pose after every action")
            s.add_argument("--effort", default="low", help="reasoning_effort for the router ('' = model default)")
            s.add_argument("--max-tokens", type=int, default=4000)
            s.add_argument("--carry-min-z", type=float, default=None,
                           help="while holding, keep the TCP at least this high along every path (mm)")
            s.add_argument("--place-xyz", default="", help="continuous: the place spot of the task 'x,y' mm (metrics)")
            s.add_argument("--fake-llm", action="store_true")
        s.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
