# Integration notes (v0, 2026-10-02)

The five v0 modules were integrated, tested locally and on compute3, and run end to end in the
Isaac rig against the live endpoint. This file records what was fixed, what was measured and
what is still open.

## Test status

| where | command | result |
|---|---|---|
| local (no GPU) | `uv run pytest -q` | 246 passed, 12 skipped (live + isaac) |
| compute3 `~/controlr` | `CONTROLR_ISAAC=1 .venv/bin/python -m pytest -q tests/` | 254 passed, 4 skipped (live), 2 min 15 s |

Run with the mock backend and FakeLLM through the CLI (`controlr run -c configs/mock.yaml --fake-llm`,
with and without the planner): the run directory contains config.yaml, system_prompt.md, plan.md,
planner.json, setup.json, messages.jsonl, turns.jsonl, summary.json and images/. `scripts/remote_run.sh`
starts and stops the Isaac server correctly, which had not been tested before (the first Isaac
episode was a FakeLLM run).

## Fixes and contract changes

All changes are backwards compatible. The contract-level ones are reflected in ARCHITECTURE.md.

1. **Response replay by the router.** omniroute returns cached *responses* for byte-identical
   requests. New `llm.request_nonce` (default true) prepends `RUN <run id>` to turn 0 and to the
   planner request. The system prompt is left unchanged, so prompt caching across episodes still
   works.
2. **Episodes ended on the first STOP, but the manual tells the model to back off.** New
   `episode.max_stops` (default 3) ends the episode after that many STOPs. A STOP of kind
   `unstable` (the simulator diverged) ends it at once; the Isaac client now emits that kind. The
   manual states the rule (`{stop_rule}`).
3. **A stopped arm could never recover (Isaac server).** The force stop compared the running *peak*
   force with the threshold, so an arm resting against the box after a stop was stopped again on
   every later move. Seen live: Haiku's "back off" moves achieved 0 mm. The stop now uses the force
   of each sample and needs `> max(threshold, force before the motion + 20 N)`. Verified with a
   scripted episode: after the stop, the back-off move ran its full length.
4. **Thinking starved the reply.** `claude/claude-sonnet-5` thinks (14–695 reasoning tokens per
   turn) and `max_tokens` counts thinking. At 400, 6 of 15 turns ended with
   `finish_reason=length` and empty text. `llm.max_tokens` is now 2000 (config default and
   base.yaml); early stop still ends the stream at STATUS.
5. **The model did not know where the tool points.** With rotation=none the orientation is fixed
   and not shown in STATE. On this rig the tool is tilted, so the gripper housing sits 10–14 cm
   toward +x/+y of the TCP and hit the box wall in both Sonnet episodes. `build_system_prompt` and
   `build_planner_prompt` take a new optional `state0`. With rotation=none, the manual gives the
   tool axis, the side the gripper body is on and the jaw line, computed from the reset state.
6. **The waffle start orientation could not be changed.** The recorded `START_Q` holds the tool
   only 17° below horizontal, 30° away from PHANTOM's expert grasp orientation, which the scripted
   pick uses for every phase. `configs/sim_waffle.yaml` now starts at `task.params.start_q` =
   IK((-335, -250, 280) mm, `EXPERT_ROT_GRASP`). Reach and push are unchanged.
7. **Manual examples had the wrong sign.** The worked example and Appendix B used x=+300 mm while
   the workspace is at negative x. They are now built from the workspace centre and the table
   height of the spec.
8. **`--fake-llm` with the planner on.** The planner used up the first scripted control reply; a
   plan reply is now inserted first.

## Planner effort check (1 call)

`claude/claude-opus-5-5` with `extra_body.reasoning_effort: "low"` on a short maths prompt used
2237 reasoning tokens and took 25 s, so the effort setting is effectively ignored. The default
planner stays the suffixed id `claude/claude-opus-5-5-xhigh`, which exists in `GET /models`.

## Live runs (compute3, Isaac, seed 0)

Corrected after the reviews (see docs/FIXLOG.md): the "TTFT" column is the first *content* byte,
which on the thinking `claude/claude-sonnet-5` route includes the thinking (e.g. 9.4 s with 658
reasoning tokens); the Sonnet rows' 6 `finish_reason=length` turns of the first run have no TTFT
(median over 9 of 15 turns). Run 105745 was missing from this table. The reach runs all used seed 0,
whose marker is physically unreachable without driving the gripper body through the box wall
(fixed in the sampler) — their STOPs are not only depth misjudgement.

| model | task | outcome | turns | median LLM s (first content incl. thinking) | median turn cycle s | cache-read share: episode / per turn ≥1 (median) |
|---|---|---|---|---|---|---|
| no-think/claude/claude-haiku-4-5-20251001 | reach, no planner (run 105745, before fixes 2-3) | safety_stop (1297 N box contact) | 8 | 2.41 (1.35) | – | 0.47 / – (cache split: turn 4 read 0, rewrote 5250) |
| no-think/claude/claude-haiku-4-5-20251001 | reach, no planner (before fixes 2-3) | safety_stop | 3 | 2.13 (1.28) | 5.9 | 0.00 / 0.00 |
| no-think/claude/claude-haiku-4-5-20251001 | reach, no planner | max_turns (112 mm short) | 12 | 1.92 (1.25) | 5.2 | 0.74 / 0.90 |
| no-think/claude/claude-haiku-4-5-20251001 | reach + planner (opus-5-5-xhigh, 77 s) | safety_stop (3 STOPs at the box) | 6 | 1.48 (1.34) | 5.4 | 0.62 / 0.88 |
| claude/claude-sonnet-5 | waffle + planner (90 s), max_tokens 400 | safety_stop (old rule) | 15 | 5.22 (4.02) | 6.7 | 0.87 / 0.91 |
| claude/claude-sonnet-5 | waffle + planner (77 s), all fixes | safety_stop (physics diverged) | 8 | 6.27 (6.06) | 11.1 | 0.80 / 0.91 |

**Caching** (through omniroute; explanation corrected by the caching review):
- From the first cacheable turn on, a call normally reads everything up to the previous call's
  prefix and writes the previous user turn plus the previous reply.
- The newest user turn (~290–430 tokens: one 448×336 image plus feedback) stays uncached.
- Read share per turn is therefore 0.77–0.94 and rises as the transcript grows.
- The boundary is set by omniroute's `cc` provider, not by our markers: the cached prefix ends
  after the newest assistant reply, which we never mark, and the unmarked planner calls cached too.
  Whether `llm.cache`/`cache_ttl` do anything on these routes is unverified (needs a ~6-call probe).
- Not every call reads everything: run 105745 turn 4 read 0 and rewrote 5250 tokens while the
  previous entry was still alive (two upstream caches). `summary.json` now lists such turns as
  `cache_regressions`.
- Haiku 4.5: nothing caches until the prefix is about 4.5k reported prompt tokens. The 3.9–4.4k
  turns had no write; the first write was at 4.48k. The manual alone (~3.9k with image) is below
  Haiku's 4096 minimum, so turns 0–2 are uncached.
- The planner does NOT share a cache with the control transcript (other model, other system text):
  the 5358-token read was the second run's planner reading the first run's planner entry; control
  turn 0 read 0 in both Sonnet runs.

**Parsing**: Haiku valid 21/21 turns, but it adds 1–3 sentences of prose before the MOVE line. The
tolerant parser handles this; it costs about 0.5 s of output. Sonnet was valid 17/23; the 6
failures were all from fix 4.

**Actions and feedback**:
- EXEC, CLAMP, WARN, STOP and STATE lines matched the executed motion: commanded vs achieved within
  0.3 mm, clamps reported with the scaled result.
- The "not reachable with this orientation" rejection and the collision/contact events read
  correctly.

**Images**: 448×336 JPEG, sharp, no stale frames. The TCP marker of the `ee_marker` overlay
projects between the fingertips, which confirms the camera calibration is consistent.

**Latency**:
- Per turn: LLM ≈ 1.5–2 s on Haiku (no-think) and 2.3–9.6 s on thinking Sonnet, plus robot
  execution of 1.9–4.5 s.
- A 100 mm move takes ~4 s wall time because Isaac runs at 0.24× real time; this dominates the
  turn cycle.
- Reset ~4 s; Isaac server start ~15 s; Opus xhigh planner 77–90 s (68–83 s to first token).

**Model behaviour**:
- Both models misjudge depth (base y) from the raw tilted view by 80–100 mm. The planner placed
  the reach ball and the box wall 100–200 mm off.
- The models fixed this from STOP/contact feedback but ran out of turns.
- No episode succeeded. These are capability and observation problems rather than harness bugs.

**Spend**: 48 LLM calls in total:
- 44 control turns: Haiku 21, Sonnet 23.
- 3 planner calls (opus-5-5-xhigh).
- 1 Opus effort probe.

About 335k prompt tokens (control + planner), of which ~243k were cache reads.

## Open issues

- **Depth perception** is the main reason for failure. Try `observation.renderers: [raw, grid,
  ee_marker]` for the Isaac configs: the overlays exist and project correctly, but they have not
  been tested live. A second camera or a top-view tile could also help.
- **Isaac physics is 4× slower than real time** and dominates turn latency. The faster 4 ms
  sliding-pad scenes are not wired up.
- **Physics divergence** when the fingers press on the packet: there was a 170 N finger–packet
  contact one turn before the blow-up. The force stop ignores gripper/arm–object contact; a
  separate object-contact threshold is worth trying.
- **`claude/claude-sonnet-5` thinks every turn** (2–10 s). For ~1 s turns use
  `no-think/claude/claude-sonnet-5`, which was not measured in an episode here. The default
  control model was left unchanged.
- **Haiku 4.5 cannot cache the first turns** because the manual is below its 4096-token minimum.
- **No path collision checking**: the safety envelope knows only the box, table and joint limits.
  The manual now says so explicitly.
- **Mock vs Isaac table height**: the mock still uses table_z 0.053 m and the Isaac mat is at
  −0.0095 m. Mock-only, so harmless.
- **Old dev directories on compute3**: `~/controlr-dev-isaac` and `~/controlr-dev-loopcli`; the
  latter holds a copy of `.env`. Delete them when no longer needed.
