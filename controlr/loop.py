"""Episode runner: planner call + the turn loop + end conditions.

The loop is deliberately boring and sequential (see ARCHITECTURE.md
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
#   success          model said DONE and the goal check passed
#   goal_reached     the goal check passed without DONE (only with episode.end_on_goal)
#   done_unverified  model said DONE, trust_done=True, goal check failed
#   fail             model said FAIL (episode.end_on_fail)
#   max_turns        turn budget exhausted
#   safety_stop      episode.max_stops STOP events, or one STOP of kind "unstable"
#   parse_errors     episode.max_parse_errors consecutive unparsable replies
#   llm_error        the client gave up (retries exhausted / non-retryable 4xx)
#   interrupted      Ctrl-C
#   error            unexpected exception in the harness or backend (traceback in summary)
#
# Success fields (summary.json):
#   success_verified   outcome == "success": the model claimed DONE and the check agreed —
#                      the number to aggregate in comparisons
#   success            the objective goal check at the END of the episode, whatever the model
#                      claimed (so outcome=max_turns with success=true is possible: the goal
#                      was reached but never declared)


@dataclass
class EpisodeResult:
    outcome: str
    success: bool                     # goal check at the end (see "Success fields" above)
    turns: int
    run_dir: str | None
    totals: dict = field(default_factory=dict)
    cache_read_share: float | None = None
    latency: dict = field(default_factory=dict)
    plan: str | None = None
    error: str | None = None
    final_goal: dict | None = None
    records: list[dict] = field(default_factory=list)
    success_verified: bool = False    # outcome == "success" (DONE claimed and verified)
    cache_regressions: list = field(default_factory=list)
    llm_backend: str = "live"         # "fake" for --fake-llm dry runs

    def summary(self) -> dict:
        d = to_jsonable(dataclasses.asdict(self))
        d.pop("records", None)
        return d


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_llm_client(cfg: Config):
    """The real client for ``cfg.llm`` (one endpoint for planner + control). Fails
    fast on a missing key or an unexpanded ``${VAR}`` base URL — before the robot
    backend (Isaac: ~15 s) is started."""
    from controlr.llm.client import LLMClient

    key = os.environ.get(cfg.llm.api_key_env, "")
    if not key:
        raise RuntimeError(f"API key env var {cfg.llm.api_key_env} is not set (see .env.example)")
    if "${" in cfg.llm.base_url or not cfg.llm.base_url.startswith(("http://", "https://")):
        raise RuntimeError(f"llm.base_url is not a URL: {cfg.llm.base_url!r} (is OMNIROUTE_BASE_URL in .env?)")
    return LLMClient(cfg.llm.base_url, key, cfg.llm.timeout_s, cfg.llm.max_retries,
                     usage_grace_s=cfg.llm.usage_grace_s, headers=dict(cfg.llm.extra_headers))


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


_TIMING_FIELDS = ("ttft", "ttft_any", "t_headers", "t_complete", "t_end", "t_wall")


def _timings_dict(t: Any) -> dict:
    """Every Timings field: on thinking routes ``ttft`` (first content) includes the
    thinking, ``ttft_any`` does not; ``t_wall`` includes retries and backoff."""
    return {k: getattr(t, k, None) for k in _TIMING_FIELDS}


# Response headers worth keeping per turn (router/upstream ids, response-cache hits,
# rate limits): the evidence for cache misses caused by upstream rotation. Everything
# else (CSP, CORS allow-lists, ...) is ~2 KB of noise per turn.
_HEADER_KEEP_PREFIXES = ("x-omniroute", "x-request-id", "request-id", "x-ratelimit", "anthropic-ratelimit",
                         "retry-after", "openai-processing-ms", "x-cache")


def _diag_headers(res: Any) -> dict:
    """Allow-listed response headers (no secrets appear in response headers; cookies are
    dropped by the client)."""
    h = getattr(res, "headers", None) or {}
    return {k: v for k, v in h.items() if k.lower().startswith(_HEADER_KEEP_PREFIXES)}


def _action_rec(a: Action) -> dict:
    """LLM units are the source line (``raw``); SI values alongside, plus the resolved
    joint target (and waypoints) the envelope approved — what an action head learns."""
    rec = {"raw": a.raw, "mode": a.mode.value if a.mode is not None else None,
           "values_si": list(a.values) if a.values is not None else None,
           "gripper_m": a.gripper}
    if a.q_target is not None:
        rec["q_target"] = list(a.q_target)
    if a.q_path:
        rec["q_path"] = [list(q) for q in a.q_path]
    return rec


def _event_rec(e: SafetyEvent) -> dict:
    rec = {"level": e.level.value, "kind": e.kind, "message": e.message}
    if e.brief:
        rec["brief"] = e.brief
    return rec


def robot_obstacles(robot) -> list:
    """The backend's known obstacles at their current pose (optional ``Robot.obstacles()``;
    none for backends without a scene model)."""
    fn = getattr(robot, "obstacles", None)
    return list(fn() or []) if callable(fn) else []


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
                cameras: dict | None = None, nonce: str = "", state0=None,
                obstacles=None) -> tuple[str | None, dict]:
    """One separate, uncached, high-effort call. The planner reads the same
    operating manual as the controller (same frame, units, limits) plus the
    planning instructions. It runs on a different model, and its system text
    (manual + planner instructions) differs from the control system text, so it
    shares no cache entry with the control transcript (only with other planner
    calls of the same config within the cache TTL). Long read timeout and few
    retries (``planner.timeout_s`` / ``max_retries``): a retry re-pays the whole
    think. Returns (plan text | None, log record); a failed planner does not end
    the episode — the controller starts without a plan."""
    from controlr.llm.transcript import Transcript
    from controlr.prompts.builder import build_planner_prompt

    tr = Transcript(build_planner_prompt(cfg, spec, cameras=cameras, state0=state0, obstacles=obstacles))
    tr.add_user([(f"RUN {nonce}\n" if nonce else "") + f"TASK: {instruction}", *rendered.images]
                + ([obs_text] if obs_text else []))
    t0 = time.perf_counter()
    res = llm.complete(cfg.planner.model, tr.to_messages(), max_tokens=cfg.planner.max_tokens,
                       extra_body=dict(cfg.planner.extra_body) or None,
                       timeout_s=cfg.planner.timeout_s, max_retries=cfg.planner.max_retries)
    reasoning = getattr(res, "reasoning_text", "") or ""
    rec = {
        "model": cfg.planner.model, "extra_body": cfg.planner.extra_body,
        "wall_s": time.perf_counter() - t0, "timings": _timings_dict(res.timings),
        "usage": _usage_dict(res.usage), "error": res.error, "http_status": res.http_status,
        "attempts": res.attempts, "finish_reason": res.finish_reason,
        "request_bytes": getattr(res, "request_bytes", None), "headers": _diag_headers(res),
        "reasoning_chars": len(reasoning), "reasoning": reasoning,
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

def _raw_frames(log: RunLog, obs, turn: int) -> list[str]:
    """Native frames without overlays (log.save_raw_frames): ``raw/<turn>_<cam>.png``."""
    from PIL import Image

    out = []
    for cam, img in sorted(obs.images.items()):
        rel = f"raw/{turn:03d}_{cam}.png"
        (log.path / "raw").mkdir(exist_ok=True)
        Image.fromarray(img).save(log.path / rel)
        out.append(rel)
    return out


def _camera_rec(info) -> dict:
    return {"name": info.name, "width": info.width, "height": info.height,
            "K": to_jsonable(info.K), "T_cam_base": to_jsonable(info.T_cam_base)}


def run_episode(cfg: Config, robot=None, llm=None, *,
                on_turn: Callable[[dict], None] | None = None,
                on_status: Callable[[str], None] | None = None) -> EpisodeResult:
    """Run one episode end to end and write its run directory.

    ``robot``/``llm`` default to ``make_robot(cfg)`` / the omniroute client (the
    client — and its key/URL check — comes first, so a missing key fails before
    the backend starts). The robot is always held and closed at the end (also on
    Ctrl-C/errors). ``on_turn`` receives each turn record, ``on_status`` short
    progress messages (planner start/end) — the CLI prints both.

    Per-turn record (turns.jsonl), enough to rebuild (obs_t, action_t, obs_t+1):
    ``obs_images`` (shas of the images the model saw this turn), ``state_before``,
    ``actions``/``executed`` (with ``q_target``/``q_path``), ``next_obs_images`` +
    ``state`` (after the action; also captured after the terminal turn),
    ``reasoning`` (thinking text, if streamed), timings, usage, headers."""
    from controlr.config import validate
    from controlr.llm.caching import apply_cache_markers, cache_style_for
    from controlr.llm.transcript import Transcript
    from controlr.observation.renderers import ObservationRenderer
    from controlr.prompts.builder import build_system_prompt, cache_warning
    from controlr.protocol.feedback import format_feedback
    from controlr.protocol.grammar import is_complete, parse_reply, strip_partial_note
    from controlr.robot.safety import SafetyEnvelope

    validate(cfg)
    fake = bool(getattr(llm, "is_fake", False))
    log = RunLog(cfg.log.root, ("fake_" if fake else "") + cfg.name, save_images=cfg.log.save_images)
    log.write_config(cfg)
    records: list[dict] = []
    result = EpisodeResult(outcome="error", success=False, turns=0, run_dir=str(log.path),
                           llm_backend="fake" if fake else "live")
    planner_rec: dict | None = None
    last_goal: GoalReport | None = None
    say = on_status or (lambda msg: None)

    try:
        if llm is None:
            llm = make_llm_client(cfg)
        if robot is None:
            from controlr.robot import make_robot
            robot = make_robot(cfg)
        spec = robot.spec
        renderer = ObservationRenderer(cfg.observation, cfg.action, spec)
        safety = SafetyEnvelope(spec, cfg.safety, rotation=cfg.action.rotation)
        style = cache_style_for(cfg.llm.model, cfg.llm.cache)

        t0 = time.perf_counter()
        obs = robot.reset(dataclasses.asdict(cfg.task), seed=cfg.seed)
        t_reset = time.perf_counter() - t0
        spec = robot.spec                    # a backend may refine it at reset (e.g. home_q)
        instruction = task_instruction(cfg, robot)
        missing = [c for c in cfg.observation.cameras if c not in obs.images]
        if missing:
            raise ValueError(f"observation.cameras {missing} not provided by backend {cfg.robot.backend!r} "
                             f"(it has {sorted(obs.images)})")
        # nominal reset state (commanded start pose, free of settle jitter): fixes the
        # rotation=none orientation and keeps the manual identical across episodes
        ref = getattr(robot, "reference_state", lambda: None)() or obs.state
        safety.reset(ref)
        # known obstacles at their reset pose (the backend's current scene; refreshed every turn)
        obstacles = robot_obstacles(robot)

        # Task-free manual (the task goes in turn 0) -> identical across tasks/episodes;
        # cameras let it state how the base axes appear in each calibrated image.
        system_text = build_system_prompt(cfg, spec, cameras=obs.cameras, state0=ref, obstacles=obstacles)
        log.write_system_prompt(system_text)
        warn = cache_warning(cfg.llm.model, system_text) if style != "none" else None
        log.write_json("cameras.json", {n: _camera_rec(c) for n, c in obs.cameras.items()})
        log.write_json("spec.json", to_jsonable(spec))

        t0 = time.perf_counter()
        rendered = renderer.render(obs, None, 0)
        t_render0 = time.perf_counter() - t0
        log.save_images(rendered.images)
        raw0 = _raw_frames(log, obs, 0) if cfg.log.save_raw_frames else None
        fb0 = format_feedback(0, None, None, None, obs, cfg, spec)
        obs_text = "\n".join(x for x in (fb0, rendered.text) if x)

        nonce = log.path.name if cfg.llm.request_nonce else ""
        plan: str | None = None
        if cfg.planner.plan_file:
            from pathlib import Path
            plan = Path(cfg.planner.plan_file).read_text().strip() or None
            planner_rec = {"plan_file": cfg.planner.plan_file}
            log.write_json("planner.json", planner_rec)
        elif cfg.planner.enabled:
            say(f"planner {cfg.planner.model} ...")
            plan, planner_rec = run_planner(cfg, llm, spec, instruction, rendered, obs_text, obs.cameras,
                                            nonce=nonce, state0=ref, obstacles=obstacles)
            log.write_json("planner.json", planner_rec)
            say(f"planner done in {planner_rec['wall_s']:.1f} s"
                + (f" (error: {planner_rec['error']})" if planner_rec.get("error") else ""))
        if plan is not None:
            log.write_plan(plan)
            result.plan = plan

        transcript = Transcript(system_text)
        head = turn0_text(instruction, plan if cfg.planner.include_in_context else None, fb0)
        if nonce:  # defeats the router's response replay; see LLMConfig.request_nonce
            head = f"RUN {nonce}\n" + head
        transcript.add_user(_user_parts(head, rendered))
        log.sync_messages(transcript.to_log_records())
        log.write_json("setup.json", {"reset_s": t_reset, "render0_s": t_render0,
                                      "instruction": instruction, "cache_style": style,
                                      "cache_warning": warn, "lookback_warning": renderer.lookback_warning,
                                      "llm_backend": result.llm_backend,
                                      "state0": _state_rec(obs.state), "reference_state": _state_rec(ref),
                                      "obs0_images": [p.sha for p in rendered.images], "raw0": raw0,
                                      "scene": to_jsonable(getattr(robot, "scene_record", lambda: None)()),
                                      "obstacles": [o.to_dict() for o in obstacles]})

        stop_when = is_complete if cfg.llm.early_stop else None
        parse_streak = 0
        n_stops = 0
        prev_obs = obs
        obs_shas = [p.sha for p in rendered.images]
        outcome: str | None = None

        for turn in range(cfg.episode.max_turns):
            t_cycle = time.perf_counter()
            messages = apply_cache_markers(transcript.to_messages(), style, cfg.llm.cache_ttl)
            res = llm.complete(cfg.llm.model, messages, max_tokens=cfg.llm.max_tokens,
                               temperature=cfg.llm.temperature,
                               extra_body=dict(cfg.llm.extra_body) or None, stop_when=stop_when)
            t_llm_wall = time.perf_counter() - t_cycle
            reasoning = getattr(res, "reasoning_text", "") or ""
            rec: dict[str, Any] = {
                "turn": turn, "llm": _timings_dict(res.timings), "usage": _usage_dict(res.usage),
                "stopped_early": res.stopped_early, "finish_reason": res.finish_reason,
                "attempts": res.attempts, "n_images": transcript.n_images(),
                "request_bytes": getattr(res, "request_bytes", None),
                "headers": _diag_headers(res), "obs_images": obs_shas,
                "state_before": _state_rec(prev_obs.state),
                "reasoning_chars": len(reasoning), "reasoning": reasoning,
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

            # An early stop that cut the STATUS note mid-word: store the reply without the
            # partial note, so the cached transcript does not depend on router chunking.
            reply = strip_partial_note(res.text) if getattr(res, "truncated", False) else res.text
            transcript.add_assistant(reply)
            log.sync_messages(transcript.to_log_records())
            parsed = parse_reply(reply, cfg.action, spec)
            if res.finish_reason == "length" and parsed.status is None:
                # Seen live (smoke test): thinking routes spend all of max_tokens on reasoning
                # and return empty text; "missing STATUS" alone does not tell the model why.
                parsed.errors.insert(0, "your reply was cut off by the output limit (thinking counts "
                                        "toward it) before any STATUS line: decide faster and reply "
                                        "with one short command")
            parse_streak = parse_streak + 1 if is_parse_failure(parsed) else 0
            rec.update(reply=reply, status=parsed.status.value if parsed.status else None,
                       note=parsed.note, parse_errors=list(parsed.errors),
                       actions=[_action_rec(a) for a in parsed.actions])
            if reply != res.text:
                rec["reply_streamed"] = res.text

            # -- execute (safety first; the robot never sees unfiltered actions)
            report: ExecReport | None = None
            t0 = time.perf_counter()
            if parsed.actions:
                state = robot.state()
                safety.set_obstacles(robot_obstacles(robot))     # current pose (the box may have moved)
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
                if report.backend:          # e.g. Isaac: profile, contact peaks, pad forces (log only)
                    rec["backend"] = to_jsonable(report.backend)

            t0 = time.perf_counter()
            goal = robot.check_goal()
            timings["goal"] = time.perf_counter() - t0
            last_goal = goal
            rec["goal"] = to_jsonable(goal)

            # -- end conditions
            stops = [e for e in report.events if e.level == EventLevel.STOP] if report is not None else []
            if report is not None and report.stopped and not stops:
                stops = [SafetyEvent(EventLevel.STOP, "stopped", "execution stopped early")]
            n_stops += bool(stops)
            if stops and (n_stops >= cfg.episode.max_stops or any(e.kind == "unstable" for e in stops)):
                outcome = "safety_stop"
            elif parsed.status == Status.DONE and goal.success:
                outcome = "success"
            elif parsed.status == Status.DONE and cfg.episode.trust_done:
                outcome = "done_unverified"
            elif goal.success and cfg.episode.end_on_goal:
                outcome = "goal_reached"
            elif parsed.status == Status.FAIL and cfg.episode.end_on_fail:
                outcome = "fail"
            elif parse_streak >= cfg.episode.max_parse_errors:
                outcome = "parse_errors"
            elif turn + 1 >= cfg.episode.max_turns:
                outcome = "max_turns"

            # -- observation after the action (also after the terminal turn: the final frame)
            t0 = time.perf_counter()
            new_obs = robot.observe()
            timings["observe"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            rendered = renderer.render(new_obs, prev_obs, turn + 1)
            log.save_images(rendered.images)
            if cfg.log.save_raw_frames:
                rec["next_raw_frames"] = _raw_frames(log, new_obs, turn + 1)
            rec["state"] = _state_rec(new_obs.state)
            rec["next_obs_images"] = [p.sha for p in rendered.images]
            if outcome is None:
                fb = format_feedback(turn + 1, parsed, report, goal, new_obs, cfg, spec)
                transcript.add_user(_user_parts(fb, rendered))
                log.sync_messages(transcript.to_log_records())
                rec["feedback"] = fb
            rec["images"] = rec["next_obs_images"]          # backwards-compatible name
            timings["render"] = time.perf_counter() - t0
            prev_obs = new_obs
            obs_shas = rec["next_obs_images"]
            timings["cycle"] = time.perf_counter() - t_cycle
            records.append(rec)
            log.append_turn(rec)
            if on_turn:
                on_turn(rec)
            if outcome is not None:
                break

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
    result.cache_regressions = stats["cache_regressions"]
    result.latency = stats["latency"]
    result.final_goal = to_jsonable(last_goal) if last_goal is not None else None
    result.success = bool(last_goal.success) if last_goal is not None else False
    result.success_verified = result.outcome == "success"
    summary = result.summary()
    planner_ptr = None
    if planner_rec is not None:
        planner_ptr = {k: planner_rec.get(k) for k in ("model", "plan_file", "wall_s", "error", "attempts",
                                                        "finish_reason") if k in planner_rec}
        planner_ptr["file"] = "planner.json"
        planner_ptr["usage"] = planner_rec.get("usage")
    summary.update(name=cfg.name, seed=cfg.seed, model=cfg.llm.model, planner=planner_ptr,
                   task=cfg.task.name, backend=cfg.robot.backend,
                   cache_read_share_turns=stats["cache_read_share_turns"], sim_time=stats["sim_time"])
    log.write_summary(summary)
    return result
