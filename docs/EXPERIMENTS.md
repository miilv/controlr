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
4. **Budget** up front: control-model and planner calls (estimate: turns per episode × episodes).
   Live runs only with a budget from the owner.
5. **Run**: `scripts/remote_run.sh run|sweep ...` on compute3 → `runs/` are synced back locally.
6. **Report** `docs/experiments/YYYY-MM-DD-<slug>.md` + a row in the index
   [experiments/README.md](experiments/README.md); videos of successes and a typical failure via
   `scripts/turn_video.py` into `docs/video/`; bugs/landmines found → [BACKLOG](BACKLOG.md).

Sample size: with 20 episodes per arm you can only distinguish a ≈40-point difference in success;
50 % vs 70 % needs ≈90 episodes per arm. "2/4" is a description, not a finding.

## 2. Experiment axes (all config fields)

| Axis | Field | Values |
|---|---|---|
| control model | `llm.model` | `claude/claude-sonnet-5-5` (default), `claude/claude-opus-5-5`, `no-think/claude/...`, effort suffixes `-low…-xhigh`, other vision models on the router |
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
| goal feedback | `episode.goal_feedback`, `episode.trust_done` | `never` / `on_done` / `always` |
| safety | `safety.*` | step, clearance, speed, force thresholds |
| task and scene | `task.name`, `task.params` | `waffle_pick_place`, `reach`, `push`; object pose/yaw spread, start yaw |
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
`safety_stop` / `llm_error` / `unstable`). Summary: `uv run controlr report runs/...`.

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
