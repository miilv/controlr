# Experiments: how to design, run and report them

This project is an experimentation platform, so a result = a run that can be repeated and
compared. Reports live in [experiments/](experiments/) (index in its README).

## 1. Protocol

1. **Question and hypothesis** in one line: what we compare and what outcome we expect
   ("a second top-down camera reduces failures when carrying over the box rim").
2. **Config in git**: `configs/<experiment>.yaml` via `extends:` from the base; the varied part
   via `--set` or a sweep (`configs/sweeps/*.yaml`). Don't touch defaults.
3. **Control the noise**:
   - seeds (different scenes) **and** repeats per seed (model stochasticity);
   - when comparing control models — the same plan (`planner.plan_file`) or the planner off;
   - one variable at a time against a baseline run.
4. **Estimate calls** up front: control-model and planner calls (turns per episode × episodes).
   Soft cap ~5k calls a day overall; no per-run budget sign-off needed.
5. **Run**: `scripts/remote_run.sh run|sweep ...` on compute3 → `runs/` are synced back locally.
6. **Report** `docs/experiments/YYYY-MM-DD-<slug>.md` + a row in the index
   [experiments/README.md](experiments/README.md); videos of successes and a typical failure via
   `scripts/turn_video.py` into `docs/video/`; bugs/landmines found → [BACKLOG](BACKLOG.md).

Sample size: with 20 episodes per arm you can only distinguish a ≈40-point difference in success;
50 % vs 70 % needs ≈90 episodes per arm. "2/4" is a description, not a finding.

## 2. Experiment axes (all config fields)

**Models under test (until further notice, Ilia):** control models `claude/claude-sonnet-5-5`
(default, the baseline) and `cx/gpt-6-luna` (`configs/sim_waffle_yaw_luna.yaml`); the planner stays
`claude/claude-opus-5-5-xhigh`. Do not run Haiku (or other control models) without the owner's
go-ahead.

| Axis | Field | Values |
|---|---|---|
| control model | `llm.model` | `claude/claude-sonnet-5-5` (default), `cx/gpt-6-luna` (`cxa/` has no credentials), `claude/claude-opus-5-5`, `no-think/claude/...` (Sonnet 5.5 still thinks there) |
| control backend | `llm.backend` + `decisions.*` | `chat` (default) / `decisions` (decision head: `openai/gpt-6-luna-decisions` on OpenRouter; `decisions.levels` / `yaw_levels` = step resolution, `reduce` expected/argmax, `history`, `include_manual`, `image_mode`, `questions` = wording file; `head` signed/split, `magnitudes`, `target`; `state_layout` window/transcript, `history`, `history_image_every`) |
| cameras / fovea | `robot.params.extra_cameras` (Isaac) / `robot.params.cameras` (mock), `observation.cameras`, `observation.renderers: fovea`, `observation.fovea_px` | virtual `top` / `side` next to the D435; a zoomed crop around the TCP per camera |
| thinking effort | `llm.extra_body.reasoning_effort` | Sonnet 5.5: `low` / `medium` / `high`; unset ≈ `high`. Id suffixes `-low…-xhigh` are rejected for Sonnet 5.5 through the router; the Opus planner takes effort only via its id suffix. Luna: `none` (0 thinking) / `low` (thinks on ~1/3 of turns) / … `max`; `service_tier: priority` accepted, effect unverified |
| planner | `planner.enabled`, `planner.model`, `planner.plan_file` | on/off, model, pinned plan |
| robot manual | `prompt.system`, `prompt.extra_rules`, `prompt.fewshot` | `system_vN` versions, extra rules, demo in the cached prefix |
| action format | `action.mode` | `ee_delta` (default), `ee_abs`, `joint_delta`, `joint_abs` |
| tool rotation | `action.rotation` | `none`, `yaw`, `full` |
| chunking | `action.max_chunk` | 1 (default) … N MOVE lines per turn |
| units | `action.pos_unit`, `action.ang_unit` | mm/cm/m, deg/rad |
| cameras | `observation.cameras`, `observation.tile` | camera list, tiled into one image or separate |
| frame size | `observation.size`, `observation.first_turn_size` | 224 / 336 / 448 / 672 px |
| frame representation | `observation.renderers` | `raw`, `grid`, `axes`, `ee_marker`, `diff`, `heatmap` |
| proprioception | `observation.state_text` | STATE line on/off |
| feedback text | `feedback.level`, `feedback.repeat_task` | `short` (default: TASK / STATE / WARN / STOP only) / `full` (TURN, EXEC, CLAMP, WARN, EVENT, STOP, PARSE ERROR, GOAL, STATE — the legacy receipt) |
| tool orientation in STATE | `feedback.orientation` | `rpy` (default: roll/pitch/yaw) / `direction` (where the tool points + jaw line; needs `system_v2`) |
| tactile sensing | `observation.tactile` | fingertip pad forces, "touched the packet" events and pad-based `holding` in the full feedback (default off) |
| goal feedback | `episode.goal_feedback`, `episode.trust_done` | `never` / `on_done` / `always` (short feedback: a failed DONE is `WARN: task not complete yet`) |
| safety | `safety.*` | step, clearance, speed, force thresholds (`contact_force_stop_n`, `box_force_stop_n`, `object_force_stop_n`, `held_object_force_stop_n`) |
| box collision | `safety.box_collision`, `safety.box_clearance_m` | predictive wrist/housing-vs-box check: `block` / `warn` (default) / `off`; inflation 50 mm |
| task and scene | `task.name`, `task.params` | `waffle_pick_place`, `reach` (text target relative to the packet / box), `push` (text distance + direction); object pose/yaw spread, start yaw; `box_dynamic` (default true), `box_mass_kg` (0.4) |
| sim speed (Isaac) | `robot.params.physics`, `robot.params.{reader,substeps,contact_every,direct,predict_stop,tcp_speed_m_s,settle}` | physics step / solver iterations / USD write-back; contact reader, physics steps per drive-target update, contact sampling; see the contacts-and-speed report |
| cache | `llm.cache`, `llm.cache_ttl` | `auto` / `anthropic` / `none`, 5m/1h (effect through the router unconfirmed) |

## 3. Metrics (what `runs/<run>/` contains)

| File | Contents |
|---|---|
| `config.yaml`, `system_prompt.md`, `plan.md`, `planner.json` | exactly what was run |
| `setup.json`, `cameras.json`, `spec.json` | this seed's scene (object poses, start), camera calibration, robot |
| `turns.jsonl` | per turn: timings (render, LLM ttft/complete/end, execution), usage (prompt / cache read / write / reasoning), model reply, parsed and executed actions, safety events, state, goal |
| `messages.jsonl`, `images/<sha>.jpg` | the transcript exactly as sent (images by hash) |
| `summary.json` | outcome, turns, tokens, cache-read share, latency p50/p90, cache regressions |

Key metrics: success (the sim's goal check), turns to success, turn latency (LLM and full),
tokens and cache-read share, number of STOP/CLAMP events, outcome (`success` / `max_turns` /
`safety_stop` / `llm_error` / `unstable`), robot time vs wall time (`summary.json` `sim_time`;
per turn `turns.jsonl` `backend.profile`). Summary: `uv run controlr report runs/...`.

### Perception changes in the dojo stack: the adopt rule

A perception or belief change (a refine, a release read, a search, a re-detection) is an **adopt-candidate** only if
both of these hold:

1. **Write audit.** No excess of harmful belief writes against its A arm on the same episodes.
   - Run `scripts/perception/m95_score.py audit` on physics-only replays of both arms.
   - It compares every belief write the report logs (refines, post_place moves, any `before7` / `after7` event) with
     GT at that step.
   - New preregistrations must name the identity instrument version. `first-pick-v1` preserves #417;
     `spatial-assignment-v2` uses label-agnostic frame-zero Hungarian assignment (50 mm cap), then first-pick
     finalisation, and flags competing assignments within 20 mm as uncertain. Select it explicitly with
     `m95_score.py audit --instrument-version spatial-assignment-v2`; the CLI default remains v1 for reproduction.
     A version change after data collection is a supplement, never a registered verdict change.
   - GT identity is fixed at the first consumed pick/grasp-close; earlier unambiguous matches within 10 mm are provisional, and nearest-object fallback use is reported.
   - A write is **harmful** when it adds **more than 2 mm** of error to the tracked object (3-D, with z reported
     apart), and helpful when it removes as much. The tracked object is the label's persistently associated GT instance; a later retry never reassigns it.
   - Compare B's harmful and helpful counts per writer with A's, under these per-writer conditions:
     - **A writer that also writes in A** passes when B's harmful rate (harmful ÷ writes) is at most 5 percentage
       points above A's. Raw totals are reported but not gated. Once an arm takes a different action, the two
       trajectories diverge, so a downstream writer's raw count mostly measures how long the episode ran and what it
       looked at, not harm.
     - **A writer new in B** (zero writes in A) passes when its helpful writes are at least **2 ×** its harmful ones.
       Against A's zero, any harmful write would otherwise count as an excess, so every new read would fail by
       construction.
     - Judge both conditions on the cells where the lever acts, pooled across them. A parity guard (§ Camera A/B
       decision rule) is covered by its parity proof: its writes are reported but not gated.
     - Pre-registrations written before 2026-11-07 07:00 UTC keep the raw-count gate ("no excess per writer"). Each
       A/B keeps the rule it was registered with.
2. **Rendered A/B.** The paired rendered run passes (success / score, n and spread as in §1).

An offline "+N fixed / 0 broken" on grasp-time events (`m95_score.py table`) is evidence, not a qualifier on its own.
Grasp and place events, judged at the task's tolerance, do not see a sub-tolerance write that a servo consumes. An
open-loop replay keeps the recorded actions, so a harmful write cannot derail it.

Score grasp-time perception with `dojo.perception.grasp_metric` (R = 10 cm headline, required-only and R = 5 cm
alongside, gap table by root cause). Score consumers against their own precision with the skills' `consume` events.

### Camera A/B decision rule (dojo levers)

Every lever A/B pre-registers its rule before any data. Unless a stronger rule is justified in the pre-registration,
the default is:

1. **Gain cell(s):** at least 10 pairs.
   - The pooled score of B is at least **5 points** above A.
   - Pair wins and losses are counted on the per-pair SCORE: a win when B's score exceeds A's by at least 10 points
     (0.1), a loss when A's exceeds B's by as much. They must satisfy **wins ≥ 2 × losses**, with at least 2 net wins.
     Success flips are reported alongside. Pre-registrations written before 7 Nov 2026 count success flips instead;
     each A/B keeps the rule it was registered with.
2. **Guard cells:** on every guard cell, B's pooled score is no more than 5 points below A's, and B has no more flip
   losses than wins.
   - **Parity guards for scoped levers:** the pre-registration can prove a lever inactive on a guard cell. The proof
     needs an identical resolved config and identical skill parameters in both arms, plus zero lever actions and zero
     lever writes on every guard episode, checked from the logs after collection. That cell is then a **parity
     guard**: the parity proof is its gate, and its camera score is reported but not gated. On a high-variance cell
     at 5 pairs, a single success moves the pooled score by about 20 points, so a camera bound there would only gate
     noise. Pre-registrations written from 2026-11-07 07:00 UTC may use parity guards.
   - At least one **strong guard** (a cell the candidate solves at 90 % or more) still runs and is gated as usual. It
     catches crashes and infrastructure-level regressions that a parity proof cannot see.
3. **Re-tests:** a lever rejected with a positive pooled delta may be re-tested **once**, on layouts it has never
   run on, under a fresh pre-registration. Never re-run it on the same ids.

A stricter "zero success losers" rule decides on a single noisy flip at 5-15 pairs, so it rejects most real gains.
Counting on score rather than success fits the board, which averages scores. In low-success cells (one or two successes
in ten), success flips are too rare to separate a real gain from noise; the first post-D-C19 wave lost three such
levers, each with a +6 to +14 pooled gain, by one net flip.
The flip ratio and the guard bound keep the protection against regressions that the zero-loser rule was meant to give.

## 4. Report template

```markdown
# YYYY-MM-DD — <the question in one line>

**Hypothesis:** …
**Setup:** config (`configs/…`), models, seeds × repeats, plan pinned or not, commit.
**Budget and spend:** control/planner calls, tokens, cache share, GPU time.
**Results:** table (seed, scene, model, outcome, turns, LLM p50, cache share).
**Failure modes:** with quoted model replies and links to runs/videos.
**Conclusions (given n):** …
**Found along the way → BACKLOG:** …
**Next experiment:** …
```
