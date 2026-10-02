"""Episode runner: planner call + the turn loop + end conditions.

The loop is deliberately boring and sequential (see docs/ARCHITECTURE.md
"Turn loop"): the robot holds still while the model thinks, every reply is
parsed and safety-filtered before anything moves, and every turn is written to
the run log *before* the next request goes out — a crash or Ctrl-C always
leaves a readable run directory.

Caching: the stored transcript never contains cache markers; they are added to
a deep copy right before each request (``apply_cache_markers``), so the prefix
the provider sees is byte-identical from turn to turn and only the newest user
turn is uncached. The planner is a *separate* request (different model /
reasoning effort would invalidate the control cache anyway); only its text
enters the control transcript, in turn 0.

Dependencies (llm client, robot) are injectable so tests run with FakeLLM +
MockRobot and benchmarks with ReplayRobot.
"""

from __future__ import annotations

import dataclasses
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from controlr.config import Config
from controlr.runlog import RunLog, summarize_turns, to_jsonable
from controlr.types import (
    Action,
    EventLevel,
    ExecReport,
    GoalReport,
    ParsedReply,
    SafetyEvent,
    Status,
)

# Outcomes (summary.json "outcome"):
#   success          model said DONE and the goal check passed (or goal passed with trust_done)
#   done_unverified  model said DONE, trust_done=True, goal check failed
#   fail             model said FAIL (episode.end_on_fail)
#   max_turns        turn budget exhausted
#   safety_stop      a STOP-level safety event
#   parse_errors     episode.max_parse_errors consecutive unparsable replies
#   llm_error        the client gave up (retries exhausted / non-retryable 4xx)
#   interrupted      Ctrl-C
#   error            unexpected exception in the harness or backend (traceback in summary)


@dataclass
class EpisodeResult:
    outcome: str
    success: bool
    turns: int
    run_dir: str | None
    totals: dict = field(default_factory=dict)
    cache_read_share: float | None = None
    latency: dict = field(default_factory=dict)
    plan: str | None = None
    error: str | None = None
    final_goal: dict | None = None
    records: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        d = to_jsonable(dataclasses.asdict(self))
        d.pop("records", None)
        return d


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_llm_client(cfg: Config):
    """The real client for ``cfg.llm`` (one endpoint for planner + control)."""
    from controlr.llm.client import LLMClient

    key = os.environ.get(cfg.llm.api_key_env, "")
    if not key:
        raise RuntimeError(f"API key env var {cfg.llm.api_key_env} is not set (see .env.example)")
    return LLMClient(cfg.llm.base_url, key, cfg.llm.timeout_s, cfg.llm.max_retries,
                     usage_grace_s=cfg.llm.usage_grace_s)


def task_instruction(cfg: Config, robot: Any) -> str:
    """The instruction the model sees. Config wins; otherwise the backend's
    default for the task (``robot.task_instruction`` attribute or method), else
    the task name (better than nothing; flagged in the log)."""
    if cfg.task.instruction:
        return cfg.task.instruction
    ti = getattr(robot, "task_instruction", None)
    if callable(ti):
        ti = ti()
    if isinstance(ti, str) and ti.strip():
        return ti.strip()
    return f"Task: {cfg.task.name}"


def _usage_dict(usage: Any) -> dict | None:
    if usage is None:
        return None
    d = to_jsonable(usage)
    return d if isinstance(d, dict) else None


def _timings_dict(t: Any) -> dict:
    return {"ttft": getattr(t, "ttft", None), "t_complete": getattr(t, "t_complete", None),
            "t_end": getattr(t, "t_end", None)}


_HEADER_SKIP = ("date", "content-type", "transfer-encoding", "connection", "cache-control",
                "vary", "server", "content-encoding", "keep-alive", "strict-transport-security")


def _diag_headers(res: Any) -> dict:
    """Router/provider response headers worth keeping per turn (request ids, upstream /
    account hints, rate limits) — the evidence for cache misses caused by upstream
    rotation. Secrets never appear in response headers; cookies are dropped by the client."""
    h = getattr(res, "headers", None) or {}
    return {k: v for k, v in h.items() if k.lower() not in _HEADER_SKIP}


def _action_rec(a: Action) -> dict:
    """LLM units are the source line (``raw``); SI values alongside."""
    return {"raw": a.raw, "mode": a.mode.value if a.mode is not None else None,
            "values_si": list(a.values) if a.values is not None else None,
            "gripper_m": a.gripper}


def _event_rec(e: SafetyEvent) -> dict:
    return {"level": e.level.value, "kind": e.kind, "message": e.message}


def _state_rec(s: Any) -> dict | None:
    return to_jsonable(s) if s is not None else None


def is_parse_failure(parsed: ParsedReply) -> bool:
    """A reply counts toward the parse-error streak when it has no STATUS, or
    it had errors and nothing executable survived parsing."""
    return parsed.status is None or (bool(parsed.errors) and not parsed.actions)


def _empty_report(actions: list[Action], events: list[SafetyEvent], state) -> ExecReport:
    return ExecReport(requested=list(actions), executed=[], events=list(events),
                      state_before=state, state_after=state, duration_s=0.0,
                      stopped=any(e.level == EventLevel.STOP for e in events))


def _user_parts(text: str, rendered) -> list:
    """Text first, then images, then the renderer's text (labels / state).
    Order is fixed so the serialised prefix stays byte-stable."""
    parts: list = [text] if text else []
    parts.extend(rendered.images)
    if getattr(rendered, "text", ""):
        parts.append(rendered.text)
    return parts


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------

def run_planner(cfg: Config, llm, spec, instruction: str, rendered, obs_text: str,
                cameras: dict | None = None) -> tuple[str | None, dict]:
    """One separate, uncached, high-effort call. The planner reads the same
    operating manual as the controller (same frame, units, limits) plus the
    planning instructions; a different model/effort cannot share the control
    cache, so no markers. Returns (plan text | None, log record); a failed
    planner does not end the episode — the controller starts without a plan."""
    from controlr.llm.transcript import Transcript
    from controlr.prompts.builder import build_planner_prompt

    tr = Transcript(build_planner_prompt(cfg, spec, cameras=cameras))
    tr.add_user([f"TASK: {instruction}", *rendered.images] + ([obs_text] if obs_text else []))
    t0 = time.perf_counter()
    res = llm.complete(cfg.planner.model, tr.to_messages(), max_tokens=cfg.planner.max_tokens,
                       extra_body=dict(cfg.planner.extra_body) or None)
    rec = {
        "model": cfg.planner.model, "extra_body": cfg.planner.extra_body,
        "wall_s": time.perf_counter() - t0, "timings": _timings_dict(res.timings),
        "usage": _usage_dict(res.usage), "error": res.error, "http_status": res.http_status,
        "attempts": res.attempts, "finish_reason": res.finish_reason,
        "messages": tr.to_log_records(),
    }
    text = (res.text or "").strip()
    if res.error or not text:
        return None, rec
    return text, rec


def turn0_text(instruction: str, plan: str | None, feedback0: str) -> str:
    """Text of the first control turn (before the images)."""
    s = f"TASK: {instruction.strip()}"
    if plan:
        s += ("\n\nPLAN (from a separate planning step; a guide, not a script — trust the "
              "images when they disagree):\n" + plan.strip())
    return s + ("\n\n" + feedback0 if feedback0 else "")


# ---------------------------------------------------------------------------
# episode
# ---------------------------------------------------------------------------

def run_episode(cfg: Config, robot=None, llm=None, *,
                on_turn: Callable[[dict], None] | None = None) -> EpisodeResult:
    """Run one episode end to end and write its run directory.

    ``robot``/``llm`` default to ``make_robot(cfg)`` / the omniroute client.
    The robot is always held and closed at the end (also on Ctrl-C/errors).
    ``on_turn`` receives each turn record (CLI progress printing)."""
    from controlr.llm.caching import apply_cache_markers, cache_style_for
    from controlr.llm.transcript import Transcript
    from controlr.observation.renderers import ObservationRenderer
    from controlr.prompts.builder import build_system_prompt, cache_warning
    from controlr.protocol.feedback import format_feedback
    from controlr.protocol.grammar import is_complete, parse_reply
    from controlr.robot.safety import SafetyEnvelope

    log = RunLog(cfg.log.root, cfg.name, save_images=cfg.log.save_images)
    log.write_config(cfg)
    records: list[dict] = []
    result = EpisodeResult(outcome="error", success=False, turns=0, run_dir=str(log.path))
    planner_rec: dict | None = None
    last_goal: GoalReport | None = None

    try:
        if robot is None:
            from controlr.robot import make_robot
            robot = make_robot(cfg)
        if llm is None:
            llm = make_llm_client(cfg)
        spec = robot.spec
        renderer = ObservationRenderer(cfg.observation, cfg.action, spec)
        safety = SafetyEnvelope(spec, cfg.safety)
        style = cache_style_for(cfg.llm.model, cfg.llm.cache)

        t0 = time.perf_counter()
        obs = robot.reset(dataclasses.asdict(cfg.task), seed=cfg.seed)
        t_reset = time.perf_counter() - t0
        instruction = task_instruction(cfg, robot)

        # Task-free manual (the task goes in turn 0) -> identical across tasks/episodes;
        # cameras let it state how the base axes appear in each calibrated image.
        system_text = build_system_prompt(cfg, spec, cameras=obs.cameras)
        log.write_system_prompt(system_text)
        warn = cache_warning(cfg.llm.model, system_text) if style != "none" else None

        t0 = time.perf_counter()
        rendered = renderer.render(obs, None, 0)
        t_render0 = time.perf_counter() - t0
        log.save_images(rendered.images)
        fb0 = format_feedback(0, None, None, None, obs, cfg, spec)
        obs_text = "\n".join(x for x in (fb0, rendered.text) if x)

        plan: str | None = None
        if cfg.planner.enabled:
            plan, planner_rec = run_planner(cfg, llm, spec, instruction, rendered, obs_text, obs.cameras)
            log.write_json("planner.json", planner_rec)
            if plan is not None:
                log.write_plan(plan)
                result.plan = plan

        transcript = Transcript(system_text)
        head = turn0_text(instruction, plan if cfg.planner.include_in_context else None, fb0)
        transcript.add_user(_user_parts(head, rendered))
        log.sync_messages(transcript.to_log_records())
        log.write_json("setup.json", {"reset_s": t_reset, "render0_s": t_render0,
                                      "instruction": instruction, "cache_style": style,
                                      "cache_warning": warn, "state0": _state_rec(obs.state)})

        stop_when = is_complete if cfg.llm.early_stop else None
        parse_streak = 0
        prev_obs = obs
        outcome: str | None = None

        for turn in range(cfg.episode.max_turns):
            t_cycle = time.perf_counter()
            messages = apply_cache_markers(transcript.to_messages(), style, cfg.llm.cache_ttl)
            res = llm.complete(cfg.llm.model, messages, max_tokens=cfg.llm.max_tokens,
                               temperature=cfg.llm.temperature,
                               extra_body=dict(cfg.llm.extra_body) or None, stop_when=stop_when)
            t_llm_wall = time.perf_counter() - t_cycle
            rec: dict[str, Any] = {
                "turn": turn, "llm": _timings_dict(res.timings), "usage": _usage_dict(res.usage),
                "stopped_early": res.stopped_early, "finish_reason": res.finish_reason,
                "attempts": res.attempts, "n_images": transcript.n_images(),
                "headers": _diag_headers(res),
            }
            timings: dict[str, Any] = {"llm_wall": t_llm_wall}
            rec["timings"] = timings

            if res.error:
                rec.update(error=res.error, http_status=res.http_status, reply=res.text or "")
                timings["cycle"] = time.perf_counter() - t_cycle
                records.append(rec)
                log.append_turn(rec)
                outcome = "llm_error"
                result.error = f"{res.http_status}: {res.error}"
                break

            transcript.add_assistant(res.text)
            log.sync_messages(transcript.to_log_records())
            parsed = parse_reply(res.text, cfg.action, spec)
            parse_streak = parse_streak + 1 if is_parse_failure(parsed) else 0
            rec.update(reply=res.text, status=parsed.status.value if parsed.status else None,
                       note=parsed.note, parse_errors=list(parsed.errors),
                       actions=[_action_rec(a) for a in parsed.actions])

            # -- execute (safety first; the robot never sees unfiltered actions)
            report: ExecReport | None = None
            t0 = time.perf_counter()
            if parsed.actions:
                state = robot.state()
                filtered, sevents = safety.filter(parsed.actions, state)
                if any(e.level == EventLevel.STOP for e in sevents) or not filtered:
                    report = _empty_report(parsed.actions, sevents, state)
                else:
                    report = robot.execute(list(filtered))
                    report.requested = list(parsed.actions)
                    report.events = list(sevents) + list(report.events)
            timings["exec"] = time.perf_counter() - t0
            if report is not None:
                rec.update(executed=[_action_rec(a) for a in report.executed],
                           events=[_event_rec(e) for e in report.events],
                           exec_duration_s=report.duration_s, exec_stopped=report.stopped)

            t0 = time.perf_counter()
            goal = robot.check_goal()
            timings["goal"] = time.perf_counter() - t0
            last_goal = goal
            rec["goal"] = to_jsonable(goal)

            # -- end conditions
            stop_event = report is not None and (
                report.stopped or any(e.level == EventLevel.STOP for e in report.events))
            if stop_event:
                outcome = "safety_stop"
            elif parsed.status == Status.DONE and goal.success:
                outcome = "success"
            elif parsed.status == Status.DONE and cfg.episode.trust_done:
                outcome = "done_unverified"
            elif parsed.status == Status.FAIL and cfg.episode.end_on_fail:
                outcome = "fail"
            elif parse_streak >= cfg.episode.max_parse_errors:
                outcome = "parse_errors"
            elif turn + 1 >= cfg.episode.max_turns:
                outcome = "max_turns"

            if outcome is not None:
                rec["state"] = _state_rec(robot.state())
                timings["cycle"] = time.perf_counter() - t_cycle
                records.append(rec)
                log.append_turn(rec)
                if on_turn:
                    on_turn(rec)
                break

            # -- next user turn: feedback + observation
            t0 = time.perf_counter()
            new_obs = robot.observe()
            timings["observe"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            rendered = renderer.render(new_obs, prev_obs, turn + 1)
            fb = format_feedback(turn + 1, parsed, report, goal, new_obs, cfg, spec)
            transcript.add_user(_user_parts(fb, rendered))
            timings["render"] = time.perf_counter() - t0
            log.save_images(rendered.images)
            log.sync_messages(transcript.to_log_records())
            prev_obs = new_obs
            rec["state"] = _state_rec(new_obs.state)
            rec["feedback"] = fb
            rec["images"] = [p.sha for p in rendered.images]
            timings["cycle"] = time.perf_counter() - t_cycle
            records.append(rec)
            log.append_turn(rec)
            if on_turn:
                on_turn(rec)

        result.outcome = outcome or "max_turns"
    except KeyboardInterrupt:
        result.outcome = "interrupted"
    except Exception as e:  # noqa: BLE001 — the run dir must still get a summary
        result.outcome = "error"
        result.error = f"{type(e).__name__}: {e}"
        log.write_text("error.txt", traceback.format_exc())
    finally:
        if robot is not None:
            for fn in (getattr(robot, "hold", None), getattr(robot, "close", None)):
                try:
                    if fn:
                        fn()
                except Exception:  # noqa: BLE001
                    pass

    stats = summarize_turns(records)
    result.turns = len(records)
    result.records = records
    result.totals = stats["totals"]
    result.cache_read_share = stats["cache_read_share"]
    result.latency = stats["latency"]
    result.final_goal = to_jsonable(last_goal) if last_goal is not None else None
    # success = the objective goal check at the end, independent of what the model claimed
    result.success = bool(last_goal.success) if last_goal is not None else False
    summary = result.summary()
    summary.update(name=cfg.name, seed=cfg.seed, model=cfg.llm.model,
                   planner=planner_rec, task=cfg.task.name, backend=cfg.robot.backend)
    log.write_summary(summary)
    return result
