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
