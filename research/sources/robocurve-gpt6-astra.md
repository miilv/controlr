# Robocurve: "GPT-6 Astra on robotic manipulation" (2026-09-04)

Deep-dive for an Opus-backbone robot harness. Sources: the report page; all 120 per-trial HTML transcripts (downloaded and parsed, about 259 MB); the public harness code (`robocurve/inspect-robots` at tag `v0.57.1`/`v0.58.0`, plus `robocurve/inspect-robots-yam`); sibling Robocurve reports; and third-party coverage. Anything I could not check against a primary artifact is marked **UNVERIFIED**.

## 1. Overview and who Robocurve is

- **Report.** "GPT-6 Astra on robotic manipulation", Robocurve, dated September 4, 2026. Authors: Achu Menon, Sravanthi Machcha, Sabrina Zou, Tzu Kit Chan, Jay Chooi. The byline was added after September 6: the Wayback snapshot `20260906020432` has no byline. The report is a follow-up to the Fable 5 vs 5.1 report at `anthropic.robocurve.org/fable-5.1/` (September 3, 2026).
- **Claim.** GPT-6 Astra (`gpt-6-astra`), Claude Fable 5.1 (`claude-fable-5-1`) and Claude Fable 5 (`claude-fable-5`) controlled bimanual I2RT YAM arms through the same Inspect Robots `agent` policy. Results:
  - Block into bowl: 19/20 vs 8/20 vs 1/20.
  - Puzzle into groove: 2/20 vs 2/20 vs 0/20.
  - Astra used about 80% fewer output tokens (bowl 2.1k vs 12.9k, which is −84%; puzzle 2.7k vs 10.5k, which is −74%) and ~~cost about half per run~~ [corrected: cost 0.44× Fable 5.1 on the bowl ($0.94 vs $2.12) but 0.62× on the puzzle ($1.36 vs $2.18); "about half" holds only for the bowl].
- **Wording correction.** The hint text and the original meta description said "19/20 vs 8/20 … in interleaved blinded pairs". That is false according to the report's own Limitations section:
  - Astra ran about 2 days after Fable and was not interleaved with it.
  - Grading was "operator-judged with the model known".
  - The September 6 Wayback snapshot shows the meta description with "interleaved blinded pairs" next to that same Limitations text. ~~The live page has since dropped the phrase.~~ [corrected: as of 2026-10-01 23:28 UTC (Cloudflare `cf-cache-status: DYNAMIC`, cache-busted fetch) the live page still carries "…19/20 vs 8/20 on block-into-bowl in interleaved blinded pairs…" in `meta name="description"`, `og:description` and `twitter:description`. The only diff between the 2026-09-06 snapshot and the live page is the added byline; the 120-row table is identical.] Only Fable 5 vs 5.1 were alternated ("20 per model per task, alternating models"), in the earlier report.
- **Robocurve.**
  - YC-backed Public Benefit Corporation, "independent evaluator for robotics".
  - $10M seed led by Initialized Capital, announced 2026-09-14 (`robocurve.org/blog/seed-raise/`).
  - Claims 97k+ PyPI installs of Inspect Robots, a $500k academic benchmark program, and free YAM arms for academic groups.
  - Co-founder Jay Chooi (X: @chooi_jeq, per RuntimeWire).
- **Other Robocurve reports on the same stack:**
  - "Test-time scaling of Opus 5 on robot arms" (2026-08-19): Opus 5 on raw joint targets ~~went from 46% (low effort) to 76% (high) on block stacking~~ [corrected: the 46 → 76 figures are mean *weighted milestone scores* on a 0–100 rubric (low 46, medium 64, high 76), not success rates; full stacks (score 100) were 0/5 low, 1/5 medium, 1/5 high; Messages wire with fast mode, 100-call budget, rig-facts document in the prompt, 10% speed cap], n=5 per condition.
  - StationeryBench, GPT-6 Astra vs MolmoAct2 (2026-09-10): Astra 7/100 completions vs 0/100.
  - RoboHarm, a refusal benchmark: Astra completed 60/100 harmful attempts.
  - "LLM inference speed over time" (`robocurve/llm-token-speed`).
  - ClapboardBench.
- **YAM arm (I2RT).**
  - 6-DoF, 750 mm workspace, 2 kg nominal payload, ~~5.02 kg with gripper~~ [corrected: 5.02 kg is the arm weight including gripper, not a payload figure], 95 mm gripper throw, CAN-USB, US$2,999 (`i2rt.com/products/yam-6-dof-arm`).
  - Driver: `github.com/i2rt-robotics/i2rt`. The rigs pin commit `ac096928…`, which has auto-recovery off.
  - Gripper type `LINEAR_4310`.

## 2. Is the harness public? Yes, MIT

- `github.com/robocurve/inspect-robots`: core plus first-party plugins. Created 2026-06-26, 631 stars at the time of reading.
  - The LLM policy is `plugins/inspect-robots-agent`, package `inspect-robots-agent` 0.26.0 at core `v0.57.1`. HEAD is agent 0.28.0.
- `github.com/robocurve/inspect-robots-yam`: the `yam_arms` embodiment, IK wrapper, cameras, guardrails, plus MolmoAct2/GR00T clients.
- The transcript headers record the version used:
  - Astra bowl (rig-1): `inspect-robots 0.57.1`.
  - Astra puzzle (rig-4) and all Fable runs (rig-3/rig-4): `0.58.0`.
  - The tech sheet says "0.58.0" for all of them, which is a minor discrepancy. The agent-loop code is identical between the two tags for everything discussed here.
- What is published per trial: an HTML transcript containing rendered turns, ~~the raw request bodies per LLM call~~ [corrected: per call attempt, the endpoint/status/latency line, model/effort/temperature/tool-count, the tool schemas once, and only the messages/input items that are new or changed since the previous call ("message N changed as sent"); top-level request fields such as `reasoning_effort`, `include`, `store` and all response bodies/usage are not shown (they exist only in the unpublished `wire/<run>/<trial>/calls.jsonl`)], latency per call, and the 224×224 frames the model saw (embedded as JPEG re-encodes; the wire carried PNG data URLs). There is also an MP4 and a Rerun `.rrd` per trial.

## 3. Architecture and dataflow

```
 I2RT YAM x2 (CAN, 10 Hz)      3 cams (top_cam, left_cam, right_cam) 640x480 -> cv2.resize -> 224x224 RGB
        |  joint_pos(14), joint_eff(14), FK -> eef_state(14)                 |
        v                                                                     v
  yam_arms embodiment (inspect-robots-yam) ---- Observation{state, images, extra{env_step, approvals}} ----+
        ^                                                                                                  |
        | joint targets (IK: 20 iters, <=0.2 rad/joint/tick, oscillation hold)                             v
  approver chain: ClampApprover -> DeltaLimitApprover (vs LAST APPROVED action) [-> collision, joint mode only]
        ^                                                                                                  |
        | ActionChunk: N absolute 14-D EEF waypoints, linear interp, played OPEN-LOOP at 10 Hz             |
  Toolset._move_absolute  <-- move_to{targets{left_x..right_gripper}, note} <-- LLM (1 tool call / turn)  |
        ^                                                                       ^                          |
        |                                    chat-format history (system + "Goal:" + per-step user msgs:  |
        |                                    state text + 3 PNG frames; frames older than 2 obs elided) <-+
        |                                    wire: Responses (Astra, encrypted reasoning replayed)
        |                                          OpenAI-compat /chat/completions (Fable, no thinking replay, no cache)
  done/give_up{summary|reason, hindsight} -> trial ends -> human operator grades stage 0..4
```

The trial loop works as follows:

1. `reset()` homes the arms and builds the prompt.
2. `act(obs)` makes one LLM call per observation. It only loops again if there is no tool call or a validation error.
3. The tool call becomes a chunk of absolute waypoints (typically 3–12 ticks, 0.3–1.2 s).
4. The rollout plays the chunk without any intermediate observation going to the LLM.
5. The next observation is taken after the last waypoint. Nothing waits for convergence unless `settle_tolerance` is set; the setting on these rigs is **UNVERIFIED**.
6. The trial ends on `done` or `give_up`, on a harness-forced `give_up` when the call budget is exhausted, on the thermal `overheat` trip, or at `max_steps` = 900.

## 4. Code walkthrough (paths at tag `v0.57.1` unless noted)

**Policy loop: `plugins/inspect-robots-agent/src/inspect_robots_agent/policy.py`**
- `:115-130`: `_SYSTEM_TEMPLATE`, quoted in §5.
- `:132-152`: alternative `images=on_demand` template with a `take_pic` tool. Not used here.
- `:258-292`: `AgentPolicyConfig`. Defaults are `max_llm_calls=100`, `max_speed_frac=0.1`, `image_horizon=2`, `images="always"`, `depth="render"`. The report overrode two of them: 20 calls and a 0.25 speed fraction.
- `:761-790`: `reset()` composes the prompt. It is the system template plus:
  - `"\n\nEmbodiment notes:\n" + embodiment_info.docs`;
  - optionally `"Notes from a previous attempt at tasks like this one. They may be wrong or stale; the current observation always wins:\n" + prior_learnings`;
  - then the user message `"Goal: {instruction}"`.
- `:850-1116`: `act()`.
  - Appends the observation message.
  - Enforces the budget: at `_calls_used >= max_llm_calls` it calls `_forced_give_up` (`:1118`).
  - Applies image eviction, `_evicted_view` (`:207-255`). Frames in user messages older than the newest `image_horizon` image-bearing messages are replaced by `"[N camera frame(s) elided]"`. On the Anthropic wire it also marks a `cache_anchor` for prompt caching.
  - If there is no tool call, it sends the nudge "Respond with exactly one tool call." After 3 consecutive failures it raises (`_MAX_CONSECUTIVE_FAILURES=3`, `:60`).
  - Only the first tool call is executed. Later calls in the same turn get `"ignored: one tool call per turn"` (`:1093`).
- `:1235-1278`: `_observation_content`. The observation text is "Current observation.", then "Instruction: …", then one `state[key]: …` line per state field, rounded to 4 decimals and labelled per dimension for the controlled field. After that come the `approver: N step(s) modified (delta_clamped ×k).` line, `operator feedback (step t): …` lines, and then per camera `camera 'top_cam' (step N):` plus a PNG data URL (`_png.py`, a stdlib PNG encoder).
  - Depth would add a grayscale rendering with a "bright 0.31 m -> dim 0.62 m" legend (`_depth.py`). **No depth appeared in any of the 120 transcripts.**

**Tool surface: `.../_tools.py`.** Tools are generated from the embodiment's action-space semantics (`eef_abs_pose`, `dim_labels`, bounds).
- `:136-274`: schemas.
  - `move_to(targets: object, note: string)`, both required.
  - `done(summary, hindsight)` and `give_up(reason, hindsight)`.
  - `_HINDSIGHT_DESCRIPTION` (`:42-48`): "What do you know now that you wish you had known at the start of this episode? Concrete, transferable facts about this rig, task, or embodiment (camera mounting and extrinsics, table and base geometry, gripper axis and offsets, controller behavior, metric scale)…".
- `:422-474`: `_move_absolute`.
  - `target = current.copy()`, where **`current` is the measured `eef_state`**, so unnamed dimensions hold their *measured* value.
  - Named values are bound-checked. An out-of-bounds value is returned as an error string the LLM can fix.
  - The number of steps is `ceil(max_i |Δ_i| / step_limit_i)`, then the move is a `linspace` interpolation, clipped. A move longer than 10 s is rejected (`_MAX_DURATION_S=10`, `:38`).
  - The tool result text is `"executing move_to over 8 steps (0.8s)"`. It is synthesized *before* playback, so it says nothing about the outcome.
- `:669-673`: per-tick limits. The tick fraction is `step_frac = min(max_speed_frac/hz, 0.05)`. At 0.25 and 10 Hz that is 2.5% of range per 0.1 s. With the rig bounds this gives x ≤13 mm/tick (0.13 m/s), y ≤11 mm, z ≤15.5 mm, yaw ≤0.157 rad, pitch ≤0.03 rad, roll ≤0.079 rad. The gripper uses its declared 0.1 per tick, so a full stroke takes 1 s.

**Wire clients**
- `_responses.py:62-72` (Astra): body `{model, input, tools(strict:false), store:false, include:["reasoning.encrypted_content"], reasoning:{effort:"medium"}}`.
  - It is stateless, but the raw output items, including encrypted reasoning, are cached by `call_id` and **replayed** on later turns (`:106-108`, `:151-156`).
  - ~~Up to 3 retries with exponential backoff.~~ [corrected: `max_retries=3` is 3 attempts in total (1 + 2 retries), backoff 1 s then 2 s, 120 s timeout; 4xx other than 429 are not retried (`_responses.py:27-29,75-115`). The chat client is the same.]
- `_llm.py:248-254` (Fable): `POST {base}/chat/completions` with `{model, messages, tools, reasoning_effort}`.
  - `_DIRECT_PROVIDERS["anthropic"]` points to `https://api.anthropic.com/v1` with wire `chat` (`_llm.py:40-55`), so Fable went through Anthropic's **OpenAI-compatibility layer**.
- `_anthropic.py:117-143`: the native Messages client, *not used for Fable here*. It sends:
  - `thinking:{type:"adaptive"}`;
  - `output_config:{effort}`;
  - `cache_control` on the system prompt and the newest elided frame;
  - verbatim replay of thinking blocks;
  - optional `speed:"fast"` with beta header `fast-mode-2026-02-01`;
  - default `max_tokens` 16000 (`:28`).

**Embodiment: `inspect-robots-yam/src/inspect_robots_yam/`**
- `config.py:145-156`: `control_hz=10.0`, `cam_height=cam_width=224`, `capture_width/height=640×480`. [corrected context: these inspect-robots-yam line numbers are at HEAD `178c930` (2026-09-19), which post-dates the trials. Trial-time code was most likely v0.36.0 (2026-09-02 09:57 PT). The `capture_width/height` config fields were only added on 2026-09-19 (#158); at v0.36.0, 640×480 was hard-coded (`embodiment.py:568-569`, `_capture_proc.py:28-29`) and the resize lines were 642/710/1020. Behaviour is the same.]
- `embodiment.py:645,717,1015`: plain `cv2.resize` to 224×224, which **squashes** 4:3 to 1:1. The StationeryBench tech sheet confirms "frames are 224×224 policy inputs". All transcript frames, including wire blobs, are 224×224.
- `config.py:713-729`: EEF interface semantics are `eef_abs_pose`, `rotation_repr="none"`, `frame="base"`, with yaw/pitch/roll relative to the orientation at reset.
- `embodiment.py:~~112-128~~ [corrected: 112-133]`: `_DOCS_EEF_POS`, quoted in §5. It is extended with joint-effort documentation when `report_joint_eff` is on (`:1325-1336`).
- `kinematics.py`: i2rt FK/IK wrapper. `ik_max_iters=20`, `ik_step_joint_limit=0.2` rad per tick, oscillation hold (~~`osc_reversals=2` within `osc_window=6`~~ [corrected: the hold fires when a joint's reversal count *exceeds* `osc_reversals=2`, i.e. at 3 or more sign reversals within the last `osc_window=6` ticks, `kinematics.py:216-230`; there is also `cmd_resync_threshold=0.35` rad, which resyncs the command to the measured joints], then hold for 10 ticks). It re-sends the previous pose on failure. That is what the models experience as "the arm stopped tracking".
- `embodiment.py:1694-1745`: thermal guardrail. `termination_reason="overheat"` parks the arms. This happened in 3 Fable bowl trials on rig-3.
- Core `src/inspect_robots/approver.py:140-167`: `DeltaLimitApprover` limits each step relative to the **last approved action**. The agent interpolates from the *measured* pose. Under sag these references differ, so ~~the first tick of most chunks gets clamped. That is the source of the ubiquitous `approver: 1 step(s) modified (delta_clamped)`.~~ [corrected: an approver line appears in only 30–52% of observations (Astra bowl 90/298 = 30%, Astra puzzle 162/370 = 44%, Fable 5 bowl 49%, Fable 5 puzzle 35%, Fable 5.1 bowl 52%, Fable 5.1 puzzle 52%). Of those, 48–81% are 1-step clamps. Multi-step clamps up to "10 step(s)" are common, e.g. gripper re-commands after contact, where the declared 0.1/tick gripper limit applies against the last approved value. The default limit is 5% of range per tick (`approver.py:147-148,220`), twice the agent's 2.5% step.]

Effective config for every trial, read from the transcript headers: `policy=agent`, `embodiment=yam_arms`, `effort=medium`, `max_llm_calls=20`, `max_speed_frac=0.25`, `image_horizon=2`, `images=always`, `depth=render`, `temperature=null`, `max_output_tokens=null`, `action_horizon=1`, `max steps=900`, `seed=0`, `wire_capture=true`. Wire and endpoint: Astra `responses` @ `api.openai.com/v1`; Fable `chat` @ `api.anthropic.com/v1`.

Rig bounds, identical on rig-1/3/4:
- `left_x/right_x [0.08,0.6]`.
- `left_y [-0.4,0.05]`, `right_y [-0.08,0.4]`.
- `z [-0.02,0.6]`, `yaw [-π,π]`, `pitch [-0.6,0.6]`, `roll [-π/2,π/2]`, `gripper [0,1]`.
- The bowl sat at left_y ≈ −0.38 to −0.40, which is at the workspace bound.

## 5. Prompts (verbatim)

System prompt, from `policy.py:115-130`, as seen in the transcripts:

> "You are controlling a real robot embodiment named 'yam_arms' through tool calls. Each observation message gives you the current proprioceptive state and camera images. Work toward the user's goal in small, deliberate motions; re-check the observation after every motion. Every move tool call must include a `note`: in one or two sentences, say what you observe in the current observation and why you chose this motion. The user is watching these notes to see what you see and what you decide, so write them for a human reader. Safety approvers clamp out-of-bounds and too-fast actions below you. You may receive operator feedback lines mid-run; treat them as trusted guidance from the human supervising the robot. Respond with exactly one tool call per turn. When the goal is achieved call done; if it cannot be achieved call give_up. Note what you are learning about this rig and task as you go: done and give_up will ask what you wish you had known from the start. You have a budget of 20 LLM calls for the whole trial."

Embodiment notes, from `embodiment.py:112-128`; key passages:

> "Each arm's targets are in that arm's own base frame: +x points forward out of the base, +y left, +z up; how the two bases are mounted relative to each other depends on the rig. … left_yaw / right_yaw: tool rotation in radians about vertical, relative to the trial's start orientation … Positive pitch tips the tool forward (+x at yaw 0), positive roll toward the arm's left … left_gripper / right_gripper: 0 is fully closed, 1 is fully open (about 9.5 cm between the jaws). Proportions: upper arm 0.26 m, forearm 0.25 m, wrist to grasp point 0.25 m when straight; reach from the shoulder about 0.76 m. An inverse-kinematics layer converts targets into joint motion; unreachable or awkward targets may be tracked slowly or held, so prefer modest steps and re-check the observation after each motion."
>
> "Joint effort: ``observation.state["joint_eff"]`` is per-joint estimated torque in N·m … Rising effort while position stops tracking indicates contact; flat effort with position error indicates controller sag."

What the prompt does **not** contain:
- camera intrinsics or extrinsics;
- the arm-base poses relative to each other;
- the table height;
- image overlays: no grid, no projected gripper point, no object boxes;
- any rig-facts document. The earlier Opus 5 report did include one.

Each trial therefore re-discovers the image↔frame mapping. ~~Almost every~~ [corrected: a majority of Astra hindsights: 24 of the 38 extractable ones contain an explicit axis-direction statement and 31 of 38 mention camera/image geometry] Astra hindsight says some version of "negative left_y moves toward the bowl; +x is up in the overhead image".

Task strings as sent:
- Bowl: "Pick up the red block from the table and place it inside the bowl."
- Puzzle: "…place it into the matching circular groove in the board. **Use your right arm only.**" The report omits that last sentence.

## 6. Results

**Reported table** (n=20 per cell; output tokens and cost are means per run; cost assumes $10/$50 per M tokens for all three models):

| Task | Model | Mean stage | Completions | Out tok/run | Est. $/run | Min/run |
|---|---|---|---|---|---|---|
| Bowl | Fable 5 | 1.30 | 1/20 | 19.2k | 2.69 | 8.2 |
| Bowl | Fable 5.1 | 2.40 | 8/20 | 12.9k | 2.12 | 6.8 |
| Bowl | GPT-6 Astra | 3.95 | 19/20 | 2.1k | 0.94 | 2.5 |
| Puzzle | Fable 5 | 1.50 | 0/20 | 16.3k | 2.63 | 7.9 |
| Puzzle | Fable 5.1 | 2.35 | 2/20 | 10.5k | 2.18 | 5.9 |
| Puzzle | GPT-6 Astra | 2.00 | 2/20 | 2.7k | 1.36 | 3.4 |

Rubric, with the same rubric in both reports: 0 no purposeful approach; 1 contact; 2 lifted clear; 3 above the deposit point; 4 placed.

**Stage distributions**, counts of stage 0–4, recomputed from the 120-row table:

| | Bowl | Puzzle |
|---|---|---|
| Fable 5 | 2/13/3/1/1 | 2/11/2/5/0 |
| Fable 5.1 | 2/6/2/2/8 | 1/4/4/9/2 |
| Astra | 0/0/0/1/19 | 3/6/1/8/2 |

**Derived from the transcripts (my analysis):**

| Condition | LLM calls/trial (mean) | Per-call latency median / p90 (s) | Share of wall-clock in LLM | Motion time (ticks/10) | Out tok/call |
|---|---|---|---|---|---|
| Astra bowl | 14.8 | 5.3 / 11.5 | 60% | 11.0 s of 153 s | ~140 |
| Fable 5.1 bowl | 17.6 | 12.4 / 28.2 | 71% | 11.7 s of 407 s | ~735 |
| Fable 5 bowl | 19.6 | 13.7 / 40.7 | 77% | 11.7 s of 492 s | ~990 |
| Astra puzzle | 18.4 | 6.1 / 10.1 | 66% | 11.8 s of 203 s | ~150 |
| Fable 5.1 puzzle | 18.6 | 12.2 / 22.4 | 76% | 11.4 s of 357 s | ~565 |

- Statistics:
  - Bowl, Astra vs Fable 5.1: Fisher exact two-sided p ≈ 4×10⁻⁴. Wilson 95% CIs are 76–99% vs 22–61%.
  - Bowl, Fable 5.1 vs Fable 5: p ≈ 0.02.
  - Puzzle: p = 1.0, so no evidence of any difference.
- The robot actually moves for only about 11 s per trial. Everything else is LLM latency plus about 50–120 s of non-LLM overhead (homing/grading/IO, composition **UNVERIFIED**).
- Implied input tokens, from cost minus output cost: about 84k per Astra bowl run (about 5.6k per call) vs about 148k per Fable 5.1 run (about 8.4k per call).
- How the token counts were measured is ambiguous. "Wire-level … not billed tokens" does not say whether hidden reasoning tokens are included. At about 140 output tokens per call, Astra's count is close to the size of the visible JSON tool call (**UNVERIFIED**).
- No operator feedback lines were injected in any trial. ~~API errors: 0 non-200 responses in 2,183 calls.~~ [corrected: the 2,183 logical calls took 2,188 HTTP attempts. 5 attempts, all Fable 5, hit the 120 s client timeout (`status null · 120.1 s`) and succeeded on retry: rig-3_6e8e0044, rig-3_c5d62cba ×2, rig-3_79082e74, rig-4_658c1744. There were 0 HTTP error statuses. The latency figures in the table above exclude these attempts; including them, Fable 5 bowl is 81% LLM share and p90 42.4 s.]

**How trials ended**, by my parse:

| Condition | Model `done` | Model `give_up` | Harness-forced `give_up` (budget) | `overheat` | `done` but stage <4 |
|---|---|---|---|---|---|
| Astra bowl | 19 | 0 | 1 | 0 | 0 |
| Fable 5.1 bowl | 9 | 9 | 1 | 1 | 1 |
| Fable 5 bowl | ~~1~~ [corrected: 0] | 6 | ~~11~~ [corrected: 12] | 2 | ~~1~~ [corrected: 0] |
| Astra puzzle | 9 | 10 | 1 | 0 | **7** |
| Fable 5.1 puzzle | 6 | 9 | 5 | 0 | 5 |
| Fable 5 puzzle | ~~3~~ [corrected: 1] | 3 | ~~14~~ [corrected: 16] | 0 | ~~3~~ [corrected: 1] |

[corrected, Fable 5 rows: the harness termination reason is the ground truth. A `done` that Fable 5 batched *behind* a `move_to` in its last turn was ignored ("ignored: one tool call per turn"). The trial then ended by the harness-forced budget `give_up`, so those trials are forced give_ups, not `done`. Affected trials: rig-3_29103b97 (bowl), rig-4_5ba76d19 and rig-4_2da7cfde (puzzle). Fable 5's only bowl completion (rig-3_e1117d44, stage 4) also ended as a forced `give_up` after 20 calls; it never called `done`. The only Fable 5 `done` is rig-4_cd296d55 (puzzle, stage 0, grader note "approached, no contact").]

- The Fable report itself says Fable 5 "exhausted its 20-call budget in 18 of 20 bowl trials". Several model `give_up`s are "out of budget" explanations given on the final call.
- Fable 5 sometimes batched a blind sequence of move/open/`done` calls into ~~its last turn: 3 trials per task, 10 ignored calls each~~ [corrected: one turn late in the trial, at the 17th–20th call and not always the last. It happened in 3 trials per task, with 10 ignored calls *per task in total*: bowl 1 + 2 + 7 (rig-3_45801dbe, rig-3_c16654a2, rig-3_29103b97), puzzle 1 + 3 + 6 (rig-4_5ba76d19, rig-4_2da7cfde, rig-4_658c1744)]. The one-call-per-turn rule discarded all but the first call.

## 7. Failure taxonomy and qualitative findings

These come from the `note`/`hindsight` fields and the move sequences.

1. **Grasp misalignment from wrist-camera parallax.** This is the dominant early failure for Fable. Its hindsight repeatedly says that when the block "appears between the fingertips" in the wrist view it is actually 3–5 cm ahead. Closing then pushes the block away or closes on air (gripper reads ≈0.01).
   - Astra learns the same thing within a trial: "The wrist-camera grasp region lies near the bottom of the image"; "grasp location is near image (110,180)".
   - With 224×224 frames and no projected grasp-point overlay, this is largely a *harness* problem.
2. **IK and tracking stalls at low or extended poses.**
   - Seen as "commanded z descents were not executed", "the arm is no longer tracking the small lowering commands", pitch sag of −0.2 to −0.35 rad despite a 0 target, and an elbow near its limit.
   - Models discover the workaround of commanding +0.2–0.4 pitch or lifting first.
   - The causes are the IK oscillation hold, the per-tick 0.2 rad clamp, and the measured-state target construction (item 4).
3. **Puzzle final insertion.** Both model families get the disc to the groove (stage 3: Astra 8/20, Fable 5.1 9/20). They then stall: the piece is left "on the near edge of the groove". Regrasp attempts are held by the controller.
   - Astra declared `done` in 7/20 puzzle trials that were not completed. It reasoned that the "gripper obscures confirmation that it is fully seated", which is a self-verification failure.
   - The report's phrase "stalls at the same final step" is accurate, but the mechanism is a mix of perception (224 px, occlusion) and control (sub-cm compliance and insertion through a position-controlled IK with a 2.5%-of-range step).
4. **Gripper slip from a harness quirk.** `_move_absolute` seeds unnamed dimensions from the *measured* state.
   - After closing on a block, the gripper reads about 0.44. The next `move_to({"left_z":0.19})` re-commands gripper = 0.44, which is the contact position. ~~That leaves almost no squeeze force on a position-controlled gripper.~~ [corrected: plausible in principle but not observed. In rig-1_8ca08ed9 the gripper-effort slot read 1.033 right after closing (obs 6, pos 0.421) and was still 1.033 after the next lift with the gripper unspecified (obs 7, pos 0.422). The approver also ramps the re-commanded gripper from the last approved value at 0.1/tick: "approver: 4 step(s) modified" at obs 7.]
   - In the only Astra bowl failure (`rig-1_8ca08ed9`) ~~the block slipped twice during transport. Astra then wrote "I'm lifting vertically while explicitly maintaining the closing force" and added `"left_gripper": 0` to every later move.~~ [corrected sequence: the first slip happened during move 8, with the gripper unspecified, i.e. re-commanded ≈0.42. Astra then added `"left_gripper":0` from move 13 on ("I'm lifting vertically while explicitly maintaining the closing force"). The block slipped *a second time* during move 14 anyway, with the gripper explicitly commanded to 0: afterwards the gripper read 0.006 and effort 0.58. So this trial does not show that measured-state seeding caused the slips.] It ran out of budget at stage 3.
   - The same mechanism compounds pose sag: a measured sagged pitch becomes the next target. Hindsight `rig-1_5f53a87d`: "Unspecified tilt drifted substantially during transport; explicitly resetting pitch and roll improved tracking."
5. **Scene and rig variance on rig-3.** Some Fable bowl resets put the block near or beyond reach. `rig-3_fe9d6f8f` (Fable 5.1): "The red block sits in a gap unreachable by both arms". Astra's rig-1 scenes were consistently near (0.39, −0.17) for the block and (0.39, −0.39) for the bowl.
6. **Thermal.** 3 rig-3 Fable trials ended in `overheat`. Long Fable trials keep motors loaded (IK holds) for 6–15 minutes.
7. **What is good.** Astra's behaviour is economical and closed-loop at the decision level:
   - about 12 moves to completion;
   - a 1–2 sentence note per move;
   - lift-to-verify after grasping;
   - explicit "do not claim success" give_ups;
   - hindsight that is metric and transferable, e.g. "grasp succeeded near actual (0.380, -0.135, 0.017) m … release … (0.381, -0.370, 0.076)".

## 8. Confounds the headline does not carry

1. **Different rigs for the bowl comparison.** Astra ran on rig-1 and Fable on rig-3. The report says rig-3 "was unavailable". Scene placement and controller sag differ between the two.
2. **Not interleaved, not blind.** Astra ran 2026-09-04 21:35 to 09-05 01:15 UTC. Fable ran 09-02 22:50 to 09-03 05:30 UTC. Grading was done with the model known.
3. **Wire asymmetry, which matters most for an Opus harness.**
   - Astra ran on the native Responses API with its encrypted reasoning replayed across turns.
   - Fable ran through Anthropic's OpenAI-compat `/chat/completions`. Anthropic's compat doc (fetched 2026-10-01) states:
     - "Prompt caching is not supported";
     - "the OpenAI SDK doesn't return Claude's thinking", so nothing can be replayed;
     - `reasoning_effort` is **"Ignored"**.
   - The harness comment (`policy.py:81-82`) assumes "Anthropic compat maps these to thinking effort".
   - So whether Fable actually ran at "medium" is **UNVERIFIED**, and plausibly it ran at the default (`high`). That would be consistent with its ~~5–7× output tokens and about 2.3× latency~~ [corrected: output tokens per run vs Astra of 6.1× (Fable 5.1 bowl), 3.9× (Fable 5.1 puzzle), 9.2× / 6.0× (Fable 5 bowl / puzzle), and median per-call latency of 2.3× (bowl) and 2.0× (puzzle)].
   - Fable also lost its reasoning between turns. The native Messages client in the same repo (adaptive thinking, `output_config.effort`, thinking replay, cache anchors, fast mode) was not used.
   - The comparison is therefore "Astra + native wire" vs "Fable + compat shim", not purely model vs model.
4. **Version skew.** Inspect Robots 0.57.1 for Astra bowl, 0.58.0 for everything else.
5. **n=20, a single scene per task, medium effort only, manual resets.** The puzzle result is a tie, and its CI ~~includes 0–30%~~ [corrected: Wilson 95% for 2/20 is 2.8–30.1%] for both models.

Third-party reactions:
- HN item 49582582 (242 points): almost no methodological critique; mostly general LLM/robotics discussion.
- A Substack summary ("The Numbers Are Weird", 2026-09-08) repeats the headline uncritically.
- Understanding Robots (Kai Williams, 2026-10-01) adds context:
  - Chooi called it a "huge jump";
  - academics note latency, edge deployment, and precision limits;
  - RoboDojo reportedly ranked Astra first briefly before real-world testing stopped "after it damaged RoboDojo hardware" (**UNVERIFIED** here; see the teammate note on RoboDojo);
  - EmbodiedSWE: Opus 5 trajectories were used to fine-tune π0.5, which unscrewed a lightbulb in 2/10 trials.

## 9. Assessment

**Strengths**
- Real hardware, a published per-stage rubric, and *every* trial released with full wire transcripts, video and `.rrd`. This is rare and allowed the forensic analysis above.
- The harness is open, typed, and well engineered:
  - action-space-derived tools;
  - an approver chain;
  - compatibility checks;
  - thermal and collision guardrails;
  - wire capture;
  - multi-provider clients;
  - a hindsight → `summarize` → `prior_learnings` loop.
- Honest Limitations section. ~~The meta-description slip was later fixed.~~ [corrected: not fixed. As of 2026-10-01 the meta, og and twitter descriptions still say "in interleaved blinded pairs".]

**Weaknesses**
- The bowl delta is real but inflated in an unknown direction by the rig swap and the wire asymmetry.
- The tasks are trivial by robotics standards: single-object pick-place and a coarse peg-in-hole.
- Grading is non-blind and done by a human.
- The headline "80% fewer output tokens" depends on an undefined token-accounting method.
- The cost estimate ignores caching. The report itself notes Astra's input cost is overstated.

**Novelty**
- Not novel: LLM-as-policy via Cartesian tool calls with IK has been around since Code-as-Policies / VoxPoser-style work.
- Repackaged but well executed: the eval methodology and tooling, i.e. "Inspect AI for robots".
- Genuinely informative:
  - frontier LLMs can now close the loop on simple pick-place *from raw 224 px images and proprioception alone*, with no perception stack;
  - the remaining wall is the last centimetre: insertion, grasp centering, and contact.

**Maturity.** Alpha, per the README. Research and eval tooling, not a deployment controller:
- the arm idles for 5–40 s per decision;
- there is no reactive layer;
- EEF mode has no collision checking.

## 10. What the Opus-backbone harness should borrow or avoid

**Borrow**
- Tools generated from a declared action-space contract: labels, units, bounds, per-tick limits. Validation errors go back to the model as strings it can correct. All safety lives *below* the LLM in an approver chain the LLM is told about.
- A required `note` per action. It is cheap and produces a reviewable rationale stream; the human-readable notes were the best debugging data here. Also `hindsight` on termination plus a `prior_learnings` file: cross-trial memory with a SHA recorded in the log.
- One action per turn, with a hard LLM-call budget and a forced `give_up`. Plus image eviction (`image_horizon=2`), which keeps context bounded. Astra used about 5.6k input tokens per call.
- Full wire capture and HTML/Rerun reports. Evaluate with an Inspect-Robots-style protocol, or simply plug Opus into Inspect Robots on a YAM/SO-101 rig: the `agent` plugin already supports `wire=messages`, `speed=fast` and `effort`.

**Avoid or fix**
- **Use the native Messages API, never the OpenAI-compat shim**, for Opus. You need thinking-block replay, prompt caching, and real `output_config.effort`. Opus 5.5 defaults to effort `medium`; fast mode costs $8/$40 per M.
- Watch "preserved thinking". On Fable 5.1 and Opus 5.5, editing earlier turns invalidates thinking, and accounts created from 2026-08-31 can get a 400 on edited history. Inspect Robots' `_evicted_view` *rewrites* older user turns. Design image eviction to be append-only, e.g. server-side context editing or frames carried in tool results, and test it.
- Hold *commanded*, not *measured*, values for unnamed dimensions, especially the gripper (§7.4). Measure the residual separately and report it as text. The `on_demand` mode already does this: "Largest remaining offset from the requested target is …".
- Do not report "executing over N steps" as the tool result. Return the post-motion outcome instead: reached or not, residual, contact detected, approver clamps.
- Do not send 224×224 squashed frames. Send at least 448–768 px with the correct aspect ratio. Overlay the projected grasp point, gripper axis and a metric grid on the wrist and top views, and give camera extrinsics and base poses in the prompt. Every Astra trial spent calls re-deriving the frame mapping.

**Where a light learned or servo action head should go.** The transcripts locate the gap exactly:
- final 1–3 cm alignment;
- grasp centering under parallax;
- descend-until-contact;
- insertion with compliance;
- grasp verification.

Give the LLM a few closed-loop skills that run at 10–30 Hz without it:
- `align_gripper_to(object|pixel, cam)`: visual servo on the wrist camera;
- `descend_until_contact(max_dz)`, using `joint_eff`;
- `grasp(verify=lift_test)`;
- `insert(target, compliance)`.

Then reserve Opus for coarse waypoints, sequencing, failure diagnosis and success verification. Validate success with an explicit check (lift test, unoccluded view) before `done`; 7/20 of Astra's puzzle `done`s were wrong.

**Latency budget.** Even Astra spends about 60% of wall-clock thinking, and Fable 5.1 about 71–76%. Effective motion is about 11 s per trial. Target fewer, larger decisions (multi-waypoint plans with on-robot verification) plus fast mode or low effort for routine steps. Keep high effort for replanning after failures; Robocurve's own Opus 5 sweep showed effort matters, ~~46% → 76%~~ [corrected: mean weighted milestone score 46 → 76 on a 0–100 rubric; full-task completions only 0/5 → 1/5, n=5 per condition].

## Sources

- https://openai.robocurve.org/gpt-6-astra/ (report; per-trial transcripts `runs/transcripts/rig-*_*.html`, videos, `.rrd`)
- http://web.archive.org/web/20260906020432/https://openai.robocurve.org/gpt-6-astra/ (snapshot with "interleaved blinded pairs" meta description)
- https://anthropic.robocurve.org/fable-5.1/
- https://openai.robocurve.org/stationerybench/
- https://anthropic.robocurve.org/stack-blocks/ (Opus 5 test-time scaling)
- https://robocurve.org/ and https://robocurve.org/blog/seed-raise/
- https://github.com/robocurve/inspect-robots (tags v0.57.1 `80b1915`, v0.58.0 `7e4d1b7`, HEAD `095172f`)
- https://github.com/robocurve/inspect-robots-yam (HEAD `178c930`)
- https://github.com/robocurve (org: clapboardbench, stationerybench, roboharm, llm-token-speed, ~~inspect-robots-capx~~ [corrected: inspect-robots-capx is the plugin directory `plugins/inspect-robots-capx` inside inspect-robots, not an org repo. The org's 15 public repos also include worldevals, worldpolicies, inspect-robots-{so101,franka,unitree-g1,widowx,agibot-a2}, dreamzero-yam and gr00t-n1.7-so-101])
- https://i2rt.com/products/yam-6-dof-arm ; https://github.com/i2rt-robotics/i2rt
- https://platform.claude.com/docs/en/api/openai-sdk (compat limitations: no caching, thinking not returned, `reasoning_effort` ignored)
- https://news.ycombinator.com/item?id=49582582
- https://nomanualbook.substack.com/p/gpt-6-astra-drove-a-robot-arm-the
- https://www.understandingrobots.org/p/openais-astra-model-is-shockingly
- https://ai-tldr.dev/releases/robocurve-astra-robot-arms/ ; https://www.datacamp.com/blog/gpt-6-astra and https://apidog.com/blog/gpt-6-astra-api/ (third-party pricing, $10/$50 per M tokens)

## Verification (fact-check pass)

Fact-check run on 2026-10-01 by an independent pass. I did not reuse the author's cache.

What I re-fetched and re-checked:
- **Pages:** the live report page, the Wayback captures `20260906020432` and `20260913161847`, and all sibling Robocurve pages.
- **Transcripts:** I re-downloaded all 120 per-trial HTML transcripts (271,322,807 bytes = 258.8 MiB). They are byte-identical to the author's copies; md5 checked on 3 of them. I re-parsed them with my own parser, which reads the `raw-transcript` blocks and the `call N attempt M · endpoint · status · latency` lines.
- **Code:** core `robocurve/inspect-robots` at `v0.57.1` (80b1915), `v0.58.0` (7e4d1b7) and HEAD (095172f), and `robocurve/inspect-robots-yam` at `v0.36.0` and HEAD (178c930).
- **Docs:** the current Anthropic docs (OpenAI-SDK compatibility, effort, pricing, preserved thinking).

### Confirmed (seen in a primary source)

**Report metadata**
- Title, date and authors are confirmed (`citation_publication_date` 2026/09/04, five `citation_author` metas).
- Follow-up to `anthropic.robocurve.org/fable-5.1/`, dated 2026-09-03, which says "20 per model per task, alternating models; each task fixed to one rig".
- Technical-spec rows are verbatim: "Inspect Robots 0.58.0" and "Wire-level request and response tokens, not billed tokens; cost at list price, $10 / $50…".
- Limitations text is verbatim, including "operator-judged with the model known" and "OpenAI cached about a fifth of Astra's input automatically".

**Results table and statistics**
- All 6 rows match the live page: mean stage, completions, output tokens per run, cost per run and minutes per run.
- Recomputed from the 120-row table:
  - stage distributions match exactly;
  - mean output tokens are 2,073 / 12,913 / 19,234 / 2,728 / 10,529 / 16,315;
  - mean minutes are 2.54 / 6.77 / 8.21 / 3.38 / 5.95 / 7.94.
- Fisher exact two-sided p-values: 4.3×10⁻⁴ (Astra vs 5.1, bowl), 0.0197 (5.1 vs 5, bowl), 1.0 (puzzle).
- Wilson 95% intervals: 76.4–99.1% (19/20) and 21.9–61.3% (8/20).
- Implied input tokens: about 83.6k per Astra bowl run and about 147.4k per Fable 5.1 bowl run.

**Wayback snapshot `20260906020432`**
- It has the "interleaved blinded pairs" meta description and no byline.
- It contains the same Limitations text.

**Effective configuration (all 120 transcript headers)**
- Identical across trials except `model`, `wire` and `base_url`.
- `effort=medium`, `max_llm_calls=20`, `max_speed_frac=0.25`, `image_horizon=2`, `images=always`, `depth=render`.
- `temperature=null`, `max_output_tokens=null`, `action_horizon=1`, `seed=0`, `max steps=900`, `wire_capture=true`.
- `replan_interval`, `speed` and `prior_learnings` are all `null`.
- Astra used `responses` @ `https://api.openai.com/v1`; Fable used `/chat/completions` @ `api.anthropic.com/v1`.

**Versions, rigs and timing**
- Version per condition: Astra bowl ran on 0.57.1; everything else on 0.58.0.
- Rigs: Astra bowl on rig-1, Fable bowl on rig-3, all puzzle trials on rig-4.
- Start-time windows (trial header timestamps):
  - Fable: 2026-09-02 22:50 to 09-03 05:30 UTC;
  - Astra: 2026-09-04 21:35 to 09-05 01:15 UTC.

**Prompts and task strings**
- The system prompt is byte-identical in all 120 transcripts and matches `policy.py:115-130` (v0.57.1).
- The embodiment notes match `_DOCS_EEF_POS` plus the joint-effort docs.
- The puzzle goal really ends with "Use your right arm only." Both Robocurve reports omit that sentence.
- Per-dimension bounds are identical on rigs 1, 3 and 4.

**Derived-from-transcripts table** (reproduced)

| Condition | Calls per trial | Median / p90 latency (s) | LLM share | Out tokens per call |
|---|---|---|---|---|
| Astra bowl | 14.85 | 5.3 / 11.4 | 60% | 140 |
| Fable 5.1 bowl | 17.65 | 12.4 / 27.9 | 71% | 732 |
| Fable 5 bowl | 19.6 | 13.7 / 40.7 | 77% | 981 |
| Astra puzzle | 18.45 | 6.1 / 10.0 | 66% | 148 |
| Fable 5.1 puzzle | 18.65 | 12.2 / 22.4 | 76% | 565 |

- These numbers require counting successful attempts only and averaging the per-trial ratios. The method is not stated in the body.
- Motion seconds per trial (11.0 / 11.7 / 11.7 / 11.8 / 11.4) and the 50–120 s non-LLM remainder are confirmed. My figure for the non-LLM, non-motion time is 49–125 s.
- Total logical calls: 2,183.

**Other transcript checks**
- No depth parts and no `operator feedback` lines in any trial.
- 1,941 "[3 camera frame(s) elided]" stubs appear in the per-call wire views.
- Trial-ending rows for Astra bowl and puzzle, Fable 5.1 bowl and puzzle, and the 3 rig-3 Fable `overheat` trips are confirmed. The Fable 5 rows were wrong and are fixed inline.

**Transcript quotes verified, with the trial each appears in**

| Quote | Trial(s) |
|---|---|
| "Unspecified tilt drifted substantially during transport…" | rig-1_5f53a87d |
| "The wrist-camera grasp region lies near the bottom of the image" | rig-1_74369b9a |
| "grasp location is near image (110,180)" and "obscures confirmation that it is fully seated" | rig-4_f127e66b |
| "grasp succeeded near actual (0.380, -0.135, 0.017)…(0.381, -0.370, 0.076)" | rig-1_ce096d00 |
| "sits in a gap unreachable by both arms" | rig-3_fe9d6f8f, Fable 5.1 |
| "near edge of the groove" and "no longer tracking the small lowering commands" | rig-4_c3ae5367 |
| "commanded z descents were not executed" | rig-3_c5d62cba |
| "I'm lifting vertically while explicitly maintaining the closing force" | rig-1_8ca08ed9 |

- Fable's "3–5 cm ahead of the fingertips" parallax hindsight recurs in more than 10 trials, quoted as 2–3, 3–4 or 3–5 cm.

**Code: core `inspect-robots` at v0.57.1**
- Line references in `policy.py` are correct:
  - `_MAX_CONSECUTIVE_FAILURES=3` at `:60`;
  - the Anthropic-compat comment at `:81-82`;
  - `_evicted_view` at `:207-255`;
  - `AgentPolicyConfig` defaults at `:258-292`;
  - `reset` at `:761-790`;
  - `act` at `:850-1116`;
  - the nudge at `:957`;
  - `"ignored: one tool call per turn"` at `:1093`;
  - `_forced_give_up` at `:1118`;
  - `_observation_content` at `:1235-1256`.
- `_tools.py` references are correct:
  - `_MAX_DURATION_S=10` at `:38`;
  - `_HINDSIGHT_DESCRIPTION` at `:42-48`;
  - schemas at `:136-274`;
  - `_move_absolute` at `:422-474`, with `target = current.copy()` at `:429` and out-of-bounds returned as an error string at `:437-443`;
  - step limits at `:669-673`.
- Wire-client references are correct: `_responses.py:62-72,106-108,151-156`, `_llm.py:40-57,248-254` and `_anthropic.py:22-28,117-143`.
- `approver.py:140-167` is correct.
- Between v0.57.1 and v0.58.0 the agent diff is only a `ConfigError` type check for string parameters, and the `approver.py` diff is docstring-only. The plugin is `inspect-robots-agent` 0.26.0 at both tags and 0.28.0 at HEAD.

**Code: limits and kinematics**
- Per-tick arithmetic: `step_frac=min(0.25/10, 0.05)=0.025` of range, giving x 13 mm, left_y 11.25 mm, z 15.5 mm, yaw 0.157, pitch 0.03 and roll 0.079 per tick.
- The gripper's declared `max_step = 1/(gripper_stroke_s·hz) = 0.1` (`config.py:602-613`), with `declared_scale = min(0.25/0.1, 1) = 1`.
- IK defaults: `ik_max_iters=20`, `ik_step_joint_limit=0.2` and `osc_hold_steps=10` (`config.py:169-175`).

**Code: embodiment and safety**
- In EEF mode the collision guardrail is skipped: the message reads "collision guardrail skipped: absolute joints mode only" (`embodiment.py:~1424`).
- The thermal trip parks the arms and returns `termination_reason="overheat"` (`embodiment.py:1694-1745`).
- The `i2rt` pin `ac096928…` has `enable_auto_recovery=False` (`_i2rt.py:17-26`).
- Gripper type is `LINEAR_4310` (`config.py:144`).
- A decoded sample of 2,196 frames, from turn views and wire blobs, is all 224×224.
- README badge "Status: alpha"; `inspect-robots summarize` exists (`cli.py:152`); the `on_demand` residual text "Largest remaining offset from the requested target is" is at `policy.py:1160`.

**Repository metadata**
- `inspect-robots`: 631 stars, created 2026-06-26, MIT.
- `inspect-robots-yam`: 8 stars, created 2026-06-29, MIT.

**Robocurve and sibling reports**
- Robocurve:
  - YC-backed Public Benefit Corporation.
  - $10M seed led by Initialized Capital, 2026-09-14.
  - "97k+" PyPI installs.
  - "$500k in total funding and free YAM arms to academic groups".
  - Jay Chooi is co-founder, X handle @chooi_jeq per RuntimeWire.
- StationeryBench (2026-09-10): Astra 7/100 vs MolmoAct2 0/100, mean progress 46 vs 12.
- RoboHarm (2026-09-18): Astra completed 17 + 12 + 7 + 14 + 10 = 60/100 harmful instructions. Fable 5.1 completed 34/100 and refused 20/100, all on the stab task. Note the URL is `robocurve.org/roboharm/`; `openai.robocurve.org/roboharm/` returns 401.

**I2RT YAM product page**
- US$2,999 with gripper, 750 mm reach, 2 kg nominal payload, 95 mm gripper throw, CAN-USB cable in the box.

**Anthropic docs**
- The compat page has moved to `/docs/en/cli-sdks-libraries/libraries/openai-sdk`. It says:
  - "Prompt caching is not supported";
  - "the OpenAI SDK doesn't return Claude's thinking";
  - `reasoning_effort` is "Ignored";
  - on Claude 5 models thinking is on by default.
- Effort doc: Fable 5 and Fable 5.1 default to `high`; Opus 5.5 defaults to `medium`.
- Pricing: Fable 5 and 5.1 are $10/$50; Opus 5.5 is $4/$20 standard and $8/$40 in fast mode.
- Preserved thinking:
  - The prefix check runs on Fable 5.1, Opus 5.5 and Sonnet 5.5.
  - On accounts created on or after 2026-08-31 00:00 UTC it defaults to `"error"` (400).
  - Re-encoding or clearing earlier content invalidates every later thinking block.

**Third-party coverage**
- HN 49582582: 242 points, 190 comments, posted 2026-09-06 01:52 UTC. A keyword scan of all 190 comments found essentially no methodological critique.
- Substack "…The Numbers Are Weird", 2026-09-08: its only caveat is "Twenty trials per task. One arm."
- Understanding Robots (Kai Williams, 2026-10-01) contains the "huge jump" quote, the edge-deployment quote, the RoboDojo hardware-damage footnote and EmbodiedSWE 2/10.

### Corrections (all also fixed inline)

1. **Live page phrase.** Claim: "The live page has since dropped the phrase" (§1), and "The meta-description slip was later fixed" (§9). Correct: the live page still says "in interleaved blinded pairs" in its meta, og and twitter descriptions. I fetched it on 2026-10-01 23:28 UTC with a cache-busting query. The only change since 2026-09-06 is the byline.
2. **Fable 5 bowl trial endings.** Claim: done 1, give_up 6, forced 11, overheat 2, done-but-<4 1. Correct: done 0, model give_up 6, forced 12, overheat 2, done-but-<4 0.
   - In rig-3_29103b97 the `done` was the 8th call in a batch and was ignored, so the trial ended by forced give_up.
   - The single stage-4 Fable 5 bowl trial (rig-3_e1117d44) also ended by forced give_up after 20 calls.
3. **Fable 5 puzzle trial endings.** Claim: done 3, give_up 3, forced 14, done-but-<4 3. Correct: done 1 (rig-4_cd296d55, stage 0), model give_up 3, forced 16, done-but-<4 1. In rig-4_5ba76d19 and rig-4_2da7cfde the `done` was batched behind `move_to` calls and ignored.
4. **Batched calls.** Claim: "3 trials per task, 10 ignored calls each … into its last turn". Correct: 10 ignored calls per task *in total*; the batches occurred at calls 17–20.
5. **API errors.** Claim: "0 non-200 responses in 2,183 calls". Correct: 2,183 logical calls over 2,188 attempts. 5 Fable 5 attempts hit the 120 s client timeout (status null) and were retried.
6. **Approver clamps.** Claim: "the first tick of most chunks gets clamped … ubiquitous". Correct: an approver line appears in 30–52% of observations; 48–81% of those are 1-step, and multi-step clamps occur.
7. **Gripper slip in rig-1_8ca08ed9.** The claimed slip → fix sequence is reversed.
   - Astra added `left_gripper:0` after the *first* slip, and the block slipped again with gripper commanded 0.
   - "almost no squeeze force" is not supported: gripper effort read 1.033 both before and after the unspecified-gripper lift.
8. **Opus 5 effort sweep.** Claim: "46% (low) → 76% (high)" (§1 and §10). Correct: these are mean weighted milestone scores out of 100, not success rates. Full stacks were 0/5 at low, 1/5 at medium and 1/5 at high. That report used the Messages wire with fast mode, a 100-call budget, a rig-facts document and a 10% speed cap.
9. **Cost ratio.** Claim: "cost about half per run". Correct: 0.44× on the bowl and 0.62× on the puzzle.
10. **Fable/Astra ratios.** Claim: "5–7× output tokens and about 2.3× latency". Correct: per-run output ratios are 3.9–9.2×; median latency ratios are 2.0–2.3×.
11. **Puzzle confidence interval.** Claim: "CI includes 0–30%". Correct: Wilson 95% for 2/20 is 2.8–30.1%.
12. **Retries.** Claim: "Up to 3 retries". Correct: 3 attempts in total (2 retries), with backoff of 1 s then 2 s.
13. **Oscillation hold.** Claim: "osc_reversals=2 within osc_window=6". Correct: the hold fires when reversals *exceed* 2, i.e. at 3 or more in 6 ticks (`kinematics.py:226-230`). There is also `cmd_resync_threshold=0.35`.
14. **Published transcript content.** Claim: "the raw request bodies per LLM call" are published. Correct: only per-call message deltas plus metadata. No top-level request fields, no response bodies, no usage.
15. **Hindsight frequency.** Claim: "Almost every Astra hindsight". Correct: about 24 of 38 contain an explicit axis-direction statement.
16. **YAM weight.** Claim: "5.02 kg with gripper" placed next to payload. Correct: 5.02 kg is the arm weight including the gripper.
17. **Robocurve org repos.** Claim: `inspect-robots-capx` is an org repo. Correct: it is the plugin directory `plugins/inspect-robots-capx`.
18. **YAM citations.** The yam citations are at HEAD 178c930 (2026-09-19), after the trials. `_DOCS_EEF_POS` spans `:112-133`, not `:112-128`. The `capture_width/height` fields did not exist at trial time; 640×480 was hard-coded in v0.36.0.

Minor refinements, not counted above:
- The byline was absent through the 2026-09-13 16:18 UTC Wayback capture, which has the same digest as Sept 6. So it was added after **Sept 13**, not just after Sept 6.
- The Fable report says Fable 5 "exhausted its 20-call budget in 18 of 20 bowl trials". The note quotes this correctly, but the transcripts show 17 of 20 used all 20 calls; the others used 19 (model give_up), 17 and 16 (both overheat).

### Unverifiable / still UNVERIFIED

- **Fable's effective effort.** Whether Fable ran at `medium` cannot be verified. The current compat doc says `reasoning_effort` is ignored, and Fable 5 and 5.1 default to `high`. The doc's state on 2026-09-02/03 cannot be checked.
- **`settle_tolerance`.** Its value on the rigs is not in the headers. Env-step counters advance exactly by chunk length (0→7→17→21 in rig-1_8ca08ed9), which is consistent with no settle steps, but a settle could run inside `step()`.
- **Non-LLM overhead.** What makes up the 50–120 s.
- **Trial-time yam version.** The headers say "git unknown". v0.36.0 is inferred from tag dates.
- **RoboDojo damage.** The claim that Astra "damaged RoboDojo hardware" is only from the Understanding Robots footnote.
- **Eviction vs preserved thinking.** Whether Inspect Robots' `_evicted_view` would actually 400 on Opus 5.5 or Fable 5.1 for a new account is inferred from the docs and code, not tested. HEAD 095172f contains no `block_binding` or prefix handling (git grep).
- **Scene coordinates.** "Astra's rig-1 scenes consistently near (0.39, −0.17)/(0.39, −0.39)" is only partly checked. Left-arm minimum y in Astra's successful bowl trials was −0.32 to −0.40, with two readings of −0.47/−0.48.
- **Hidden reasoning tokens.** The exact count of Astra's hidden reasoning tokens. A strong inference is given below.

### Added missed details

**A. Tool schemas, verbatim from the wire**
- They are sent once per trial with `strict:false` on the Responses wire.
- `move_to` description:
  > "Move to absolute Cartesian end-effector targets (meters for positions, radians for rotations, per the dimension labels). The motion is a straight line interpolated at a fixed safe speed and the result reports its step count. Unnamed dimensions hold their current value. Coordinates are absolute in the embodiment's declared frame; on multi-arm embodiments each arm uses its own base frame and axes may differ between arms depending on mounting. Rotation dimensions are absolute targets measured relative to the trial's start orientation (0 means the start orientation) and interpolate linearly without wrapping, so prefer intermediate values for large rotations. Per-dimension bounds: left_x: [0.08, 0.6], left_y: [-0.4, 0.05], left_z: [-0.02, 0.6], left_yaw: [-3.142, 3.142], left_pitch: [-0.6, 0.6], left_roll: [-1.571, 1.571], left_gripper: [0, 1], right_x: [0.08, 0.6], right_y: [-0.08, 0.4], right_z: [-0.02, 0.6], right_yaw: [-3.142, 3.142], right_pitch: [-0.6, 0.6], right_roll: [-1.571, 1.571], right_gripper: [0, 1]."
- `targets` is a free-form `{"type":"object"}` whose only description is "Map of dimension name to value. Valid names: left_x, …, right_gripper". There are no per-key types or bounds in the JSON schema.
- `note` description: "What you observe right now in the observation (images, if any, and state), and why you chose this motion. The user reads these notes live and in the saved transcript to follow what you see and what you decide. Write for them, in one or two plain sentences."
- `done` description: "Declare the task finished. The trial ends; a scorer judges success."
- `give_up` description: "Stop trying; the task cannot be completed. The trial ends."
- `summary` and `reason` are untyped strings with no description.
- The model is explicitly told "Unnamed dimensions hold their current value", so the measured-state seeding is documented to it.

**B. Observation message, verbatim**
- Example from rig-1_8ca08ed9, call 1:
  > "Current observation.\nInstruction: Pick up the red block from the table and place it inside the bowl.\nstate[joint_pos]: [14 raw joint values]\nstate[eef_state]: left_x=0.3657 left_y=-0.0665 left_z=0.128 left_yaw=-0.0011 left_pitch=-0.0664 left_roll=0.0184 left_gripper=0.9963 right_x=… right_gripper=0.9993\nstate[joint_eff]: [14 values]"
- Then three `camera '<name>' (step N):` text parts, each followed by an `input_image` PNG data URL.
- `report_joint_eff` was on in every trial. Raw `joint_pos` is sent as well as the labelled `eef_state`.
- The instruction is repeated in every observation.
- The Responses wire shows `"summary": []` on reasoning items, so no reasoning summaries were requested. The `encrypted_content` blobs are replayed.
- The wire view literally marks rewritten history as "message N changed as sent". That is direct evidence that image eviction mutates earlier turns on every call.

**C. Eval protocol: scores were partly assigned or revised after the run**
- The in-harness scorer is `operator`, which is binary: "y" maps to `operator=1`, while "n", "partial" and "n/a" map to 0. Stage 0–4 lives only in free-text "Grader notes".
- In-harness "y" verdicts total **31**; published completions total **32**. The difference is rig-4_fd86a832 (Fable 5.1 puzzle):
  - the live operator judgement was "partial", with `operator=0` and grader note 4;
  - the model's own summary says it "may be slightly offset … and not fully seated";
  - the report counts it as a completion, so by the live verdict Fable 5.1 puzzle would be **1/20**.
- rig-4_fb8daf2d (Astra puzzle): the transcript grader note is "3", but the published score is 1. Using the note, Astra's puzzle mean stage would be 2.10, not 2.00.
- rig-4_3a441687 (Astra puzzle): live judgement "n/a" and no grader note, yet a score of 1 is published.

**D. False success claims and self-verification failures**
- rig-4_e409c0ea (Fable 5.1 puzzle) called `done` saying "The wrist camera confirms the disc seated in the groove". It was graded stage 0, judgement "n".
- rig-4_cd296d55 (Fable 5 puzzle) called `done` at stage 0.
- Of Astra's 7 wrong puzzle `done`s, 4 explicitly hedge: "partly obscures the final seating", "full insertion is not confirmed". One (rig-4_fb8daf2d) claims placement while published at stage 1.
- Astra's 3 puzzle give_ups explicitly refuse to claim success, e.g. "I cannot claim it was placed".

**E. Possible hardware fault or misdiagnosis**
- Two Fable 5.1 puzzle trials ended with the model diagnosing a frozen right gripper: rig-4_86114266 ("position reading has been frozen at 0.9306 with constant effort (~1.03)") and rig-4_b23f225d. Both were graded stage 2.
- An effort of about 1.03 is the same value the slot reads when a gripper is holding an object (rig-1_8ca08ed9). So this may be a misread grasp rather than a fault: **UNVERIFIED**.

**F. Thermal trips are confounded with the worst Fable trials**
- Fable 5's two stage-0 bowl trials are exactly its two `overheat` trials: rig-3_79082e74 (17 calls, 8.7 min) and rig-3_c2c68390 (16 calls, 9.7 min).
- Fable 5.1's overheat trial, rig-3_5bc8330b, is its 18.9-min, stage-2 trial.
- These trials were counted, not re-run, so the rig-3 thermal state penalises the Fable bowl numbers.

**G. Timeouts**
- 5 Fable 5 calls exceeded the agent's 120 s httpx timeout.
- The longest successful calls were 114.7 s for Fable 5, 92.0 s for Fable 5.1 and 26.1 s for Astra.

**H. Token accounting: strong inference**
- At v0.57.1, `ChatClient` and `ResponsesClient` never parse `usage`. Only `_anthropic.py:485-497` does, so `llm_usage` metadata is filled only on `wire=messages`.
- `_capture.py` stores full response JSON, usage included, in `wire/<run>/<trial>/calls.jsonl`, which is not published.
- Fable's visible output (text plus tool call) is only about 90–110 tokens per call (chars/3.5), yet the reported figure is 565–981. Since the compat shim returns no thinking, the reported counts must be provider-reported usage including hidden thinking.
- By the same method, Astra's 140–148 tokens per call almost certainly *includes* reasoning tokens. That implies only about 70 hidden reasoning tokens per call at `medium`.
- The later Robocurve RoboDojo-RC report states this method explicitly: "Output totals include reasoning tokens where reported by the provider."

**I. Directly relevant sibling report the note missed: "Opus 5.5 on RoboDojo-RC Tier 1" (robocurve.org/opus-5-5-robodojo-rc-tier-1/, 2026-09-23)**
- Setup:
  - 360 real-robot trials: 6 tasks × 20 trials × 3 models (Astra, Opus 5.5, Opus 5).
  - Same agent policy at medium effort, 40-call budget, 900-step cap, 25% speed cap, Inspect Robots 0.58.0.
  - Task selection is biased toward Astra: "We evaluated Astra on the 18 real-world RoboDojo tasks and selected the six easiest tasks with the highest scores".
- Results (my recomputation from `data/cells.csv` and the 360 `data/runs/*.json` files):

| Model | Mean progress | Completions | Mean output tokens per trial | Mean calls | Mean wall time | Cache-read share of input |
|---|---|---|---|---|---|---|
| Opus 5.5 | 36.0% | 1/120 | 14,496 | 34.2 | 269 s | 0.87 |
| Astra | 36.7% | 5/120 | 3,829 | 28.6 | 251 s | 0.33 |
| Opus 5 | 19.9% | 2/120 | 23,308 | 35.2 | 435 s | 0.86 |

- Reported cost per trial: $0.90 (Opus 5.5) vs $1.14 (Astra, cache-adjusted).
- Opus 5 had 10/120 `overheat` terminations.
- The cache-read shares show the Opus runs used the native Messages wire with caching, unlike the Astra/Fable report.
- For an Opus-backbone harness, this is the most relevant Robocurve datapoint: on a native wire, Opus 5.5 roughly matches Astra's progress and latency, but completes far fewer tasks.

**J. Robocurve later moved Anthropic runs off the compat shim**
- The RoboHarm README lists `anthropic/claude-fable-5-1` / `messages`.
- `roboharm/docs/running-on-robots.md:101` gives the exact CLI recipe:
  > `./run --policy agent --embodiment yam_arms --max-steps 900 -P model=openai/gpt-6-astra -P wire=responses -P effort=medium -P max_llm_calls=40 -P max_speed_frac=0.25 -P images=always -P depth=render -P image_horizon=2 -E control_interface=eef_pos -E report_joint_eff=true`

**K. Cost implication of the compat shim**
- Fable 5.1 cache hits cost $0.25/MTok (0.025× input), and Fable 5 cache hits cost $1/MTok.
- About 148k input tokens per Fable 5.1 bowl run were billed at full $10/MTok, so the cost comparison is also wire-confounded, not just the reasoning.

**L. Other report context the note omitted**
- The Fable report says "Fable 5 stalls at first contact in 13 of 20 bowl trials, never closing the gripper on the block".
- The embodiment docs define x/y/z as the *grasp point between the fingertips*, not the flange.
- The Opus 5 report names the cameras: an overhead RealSense D435 and a D405 on each wrist. The Astra report does not; assuming the same rigs is **UNVERIFIED**.
- The seed post claims frontier output-token speed is "improving by 2.1x/month".
- Understanding Robots adds:
  - Astra's RoboDojo scores were "barely correlated" with VLAs';
  - a Galbot Astra + π0.5 hybrid beat either alone;
  - Astra is now 7th on RoboDojo.
