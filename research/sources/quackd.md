# rokbenko/quackd: deep dive

Repo: https://github.com/rokbenko/quackd. I read it at `25635c6` (2026-09-30), release **v0.16.1**. The clone is at `research/repos/quackd`. Apache-2.0, on PyPI as `quackd` plus seven `quackd-<body>` adapter packages. At the time of reading it had 246 stars and 26 forks. It has 565 commits since 2026-08-28: 544 by Rok Benko, ~~the rest from four outside contributors and dependabot~~ [corrected: the other 21 are 7 by dependabot[bot], 5 by github-actions[bot], and 9 by three outside people: Bayway (6 commits, under the names "Massimiliano Fiori" and "Bayway" with the same email), Vallhalen (2, from two emails; GitHub counts 1) and r0jin (1). Source: `git shortlog -sne` and `gh api repos/rokbenko/quackd/contributors`]. Homepage: quackd.org.

## 1. TL;DR

**What it is.** An offboard, laptop-side harness. A frontier LLM acts as a **skill selector**: it picks exactly one tool call ("verb") per turn from a list built out of each robot's adapter manifest. An `Executor` enforces a contract on every call: allowlist, feasibility verdict, budgets, confirm gates, preconditions, heartbeat and kill switch.

**Bodies and providers.**
- Seven bodies: Microduck, Open Duck Mini v2, LeRobot SO-101, rosbridge base, XLeRobot, AlohaMini and ToddlerBot.
- Eleven cloud LLM vendors, plus local OpenAI-compatible servers.
- An MCP server (`serve-mcp`) so Claude Code/Desktop can be the pilot.

**What has actually run on hardware.** One body only: an SO-101 arm.
- 2026-09-15: 12 `--goal` runs with OpenAI `gpt-6-astra`. These were gestures (wave, reach, gripper open/close). There was no manipulation task.
- 2026-09-23: 26 runs with `gpt-6-sol`/`gpt-6-astra`. **19 of the 26 never moved the arm at a pilot's request.** ~~The failures were almost all calibration, torque and rest-pose integration bugs, not LLM reasoning (ADR-0045).~~ [corrected: CHANGELOG 0.14.0 (CHANGELOG.md:1137-1170) names **five faults**, and they were harness and prompt/gate design, not only calibration. (1) The rest pose lay past the calibrated travel. (2) The verdict gate refused honest answers: the arm's own datasheet rejected a `feasible` for a wave; 12 runs stopped at a y/N question about a pilot's `uncertain`; and a pen run ended on the prompt line "Decline any task that hinges on any of them" after a person had said go. (3) Three connects failed on one bad Feetech status packet ("Failed to write 'Lock' ..."). (4) `move_joints` had no motion-time parameter, so a pilot declined "raise the shoulder over ten seconds". (5) Nothing but the power switch could release torque. In addition, three pilots shown a joint reading past its travel refused to move the arm (ADR-0045). ADR-0047 frames it as "most of what went wrong there was not the pilots". Note that several of the stopped runs were the LLM's verdict interacting with gate and prompt design.]

**VLAs.**
- Supported: LeRobot ACT, SmolVLA and pi05 checkpoints, served by a separate `quackd policy serve` process over a token-authenticated 4-call HTTP protocol.
- The LLM calls `manipulate(instruction)`. That hands the arm to the policy for one bounded segment (10 s default). The instruction becomes LeRobot's `task` string.
- **No learned policy has driven the real arm.** A Hub ACT has driven only the MuJoCo twin. SmolVLA "took minutes a chunk" on a CPU. pi05 has never run.

**Jev / Laya / Kev.** These are *decision models* ("System One"). They are not LLM providers. They score a closed set of options and return probabilities.
- **Jev** is TypeSafe AI's hosted API (`POST /v1/systemone`, model `jev-1.13.0`).
- **Kev** (`jaredpalmer/kev`) is a self-hosted Qwen3.5 + LoRA pointer-head clone.
- **Laya** (`NandhaKishorM/laya`) is an in-process ModernBERT encoder.

quackd puts one in front of the LLM as an optional "stepper". The stepper answers only turns whose tool schema is fully discrete, gated by confidence floors and a mandatory `escalate` option. None of the decision models has ever been called against a real API from quackd: the client is tested against a stub only.

**Code character.**
- Huge and heavily defensive: ~78k LOC of code, ~73k LOC of tests, 2,876 test functions, 49 ADRs.
- Largely written with Claude: 508 commits carry `Co-Authored-By: Claude Opus 5/5.5/Fable 5/5.1/Sonnet 5` trailers.
- Very honest about what has and has not been tested.
- Real-world evidence is thin, and **there is no published Claude-piloted run at all.** All hardware runs are OpenAI. All published local transcripts are Qwen.

## 2. Provenance and context

**Author.** Rok Benko, a solo founder in Murska Sobota, Slovenia. His GitHub bio: "500 job rejections → 10+ failed startups → quackd".

**Origin.** The project began 2026-08-28 as "a brain for the Microduck", Pollen Robotics' $399 biped. His X announcement (x.com/rokbenko/status/2094747138454339637) was unreachable (HTTP 402); I know its text only from search snippets. The v0.9.0 notes then repositioned it as "One CLI for all your robots".

**Vision.** The README pitches it as "a ChatGPT like moment for robotics".

**Website oddity.** quackd.org's `llms.txt` links a "$QUACKD" pump.fun memecoin, labelled "No utility, not affiliated with the open-source project". This is a reputational flag worth knowing.

**Community.** I found no HN or Reddit threads. The only substantive third-party engagement is GitHub user **Vallhalen**, who ran controlled experiments on the feasibility gate (§6).

**Cross-references with Ilia's other sources.**
- **piper-astra-jev** uses the same pairing, GPT-6 Astra plus TypeSafe Jev, but on OpenRouter's `/api/alpha/decisions`. quackd explicitly says it cannot reach that endpoint, because the SDK hardcodes `/v1/systemone` (docs/decision-llms.md).
- **Same latency finding.** Both projects conclude that wall time is dominated by the frontier-LLM call.
- **llm-robotics-playground (dimentary)** has a "Jev as a real-time policy" demo. quackd forbids that use: its stepper can never author a number.

## 3. Architecture

```
 goal (CLI --goal | .duck file | MCP chat)
        │
        ▼
 ┌──────────── quackd process (laptop) ───────────────────────────────────────┐
 │ AgentLoop.run  (quackd/agent/loop.py:1588)                                  │
 │   observe: get_state + frames ─► detector (HSV blob | YOLO) ─► text line    │
 │   [optional] Stepper.advise ─► decision LLM (Jev/Kev/Laya)  ≤1.0 s timeout  │
 │   else provider.step(system, history, tools)  ← ONE tool call / turn        │
 │   meta tools: assess_task · declare_* · remember · tell                     │
 │   Executor.run_verb (quackd/safety.py:~~418~~ [corrected: run_verb is at :330; _run_verb, which holds the gate order, is at :418]): abort→allowlist→verdict→params  │
 │      →confirm→budget→abort_when→preconditions→dry-run→exec racing abort     │
 │   Heartbeat 0.5 s ─ miss ⇒ stop + abort                                     │
 │ verbs (from manifest): composite 10 Hz loops (go_to/search_scan)           │
 │   arm: move_joints (ramped @10 Hz, ≤5°/tick) · gripper · place · pick ·     │
 │        manipulate ──► PolicyLoop (policy rate 1–60 Hz, chunk queue)         │
 └──────────┬─────────────────────────────────────────────┬──────────────────┘
            │ intents (twist/skill/joint/gripper/look/sound)│ HTTP+token /v1/{policy,reset,step,end}
            ▼                                               ▼
   robot adapter (lerobot real/mujoco, zmq, rosbridge,  `quackd policy serve`
   daemons for Open Duck/ToddlerBot) → robot's own       (ACT/SmolVLA/pi05, torch,
   controllers (servo PID, RL gait @50 Hz)               separate process / GPU via ssh -L)
```

The design has three loops ("three rates, three owners"; docs/architecture.md, ADR-0003):

| Loop | Rate | What runs there |
|---|---|---|
| Deliberation | ~0.2–1 Hz | The LLM picks a verb. |
| Steering | 5–20 Hz, or the policy's own rate on the arm | Composite verbs and VLA segments. The LLM is never called here. |
| Reflexes | The body's own rate (50 Hz on the ducks) | The robot's controllers. |

The core principle (README): "**The LLM never generates motor commands.** Every verb is an *intent* the robot already understands." The one large exception is `move_joints` on the arm, where the LLM authors absolute joint angles (§4.5).

## 4. Code walkthrough

### 4.1 The loop (`quackd/agent/loop.py`)

**Connect first.** `run()` (1588) connects first. The adapter returns a `RobotManifest`, and the registry, tool schemas and prompt are all built from it.

- A `.duck` `datasheet:` override is applied.
- Verbs the task *requires* but the body lacks raise an error. Verbs it merely *allows* are dropped with a note.
- On an arm with a recorded rest pose, it drives to that pose (1852).

**Each turn:**
1. `_observe` (1115): state, every camera frame, and detections on the primary camera only.
2. Optional `stepper.advise` (1892). Under `--decision-mode on`, a confident answer becomes the call. Such turns **never enter the model's history**. The model is told afterwards in the next observation: "While you were not asked, the stepper chose these".
3. Otherwise `provider.step(system, history, tools)` (1932).
4. Zero tool calls trigger one re-prompt, `REPROMPT = "You must call exactly one tool. Choose now."` (103, 2006). A second miss ends the run as `failure`.
5. More than one call: only the first is executed (2021).
6. Meta tools: `assess_task` (2070) records the verdict; `declare_success` and `declare_failure` end the run (2115). For a `JudgedPilot`, a success is downgraded unless a person said yes.
7. Everything else goes to `executor.run_verb` (2136).

**Teardown.** The `finally` block always runs, in this order: heartbeat stop → `stop` intent → hand-back → rest move → optional torque-release offer → close. It then writes `summary.json` and records an episode in memory.

**History and frames.** `_history_for_provider` (1506) keeps images only on the last `keep_images_for_last_n=2` exchanges.

For Claude Opus 5.5 and Fable 5.1, which bind thinking blocks to their prefix, frames are trimmed only every `BINDING_TRIM_PERIOD = 8` exchanges (providers/anthropic.py:54). Invalidated thinking blocks are stripped, and the API is asked to `drop_block` the latest. So up to 9 exchanges of frames go out per request: with two cameras, 18 images.

### 4.2 The system prompt (`quackd/agent/prompts.py:656`)

The real composed prompt for `find-and-kick` is 5,304 characters (I extracted it from a local run). The fixed scaffold, verbatim:

```text
You are the brain of {blurb}. You are a high-level pilot:
you choose ONE verb per turn; the robot's own controllers handle balance and gait, and composite
verbs like `walk_to` close their own loops on the camera. Do not micro-manage.

## Rules (enforced by the executor — not optional)
- Call exactly one tool per turn. Never zero, never two.
- Only these verbs are allowed: {allow}. Anything else is refused.
- Budgets: {N} steps, {M} minutes, {K} LLM calls. The run stops when any is hit.
- Verbs marked confirm ({…}) ask a human before running.
- Before the first verb that moves the body, call `assess_task` with your verdict on whether this
  body can do this task at all, judged against its datasheet below: `feasible`, `infeasible`
  (the run ends, nothing moves) or `uncertain` (a human is asked). The verdict is about the body,
  not the view: not having found the target yet is not by itself a reason for `uncertain`. …
```

After the scaffold come these sections:
- `## Success criteria` and `## Abort conditions`.
- `## Verbs`: one line per verb.
- `## Your body: what it can and cannot do`. This is a datasheet in which every figure carries a confidence label ("official / estimate / measured"). It also lists per-joint calibrated travel ("the only goals quackd will send"), a "Whatever the task says, this body cannot:" list, and "Worth knowing" notes.
- Optional sections: `## Your executor` (VLA), `## Where this run starts` (by-hand), task pictures, `## What is a stand-in on this robot`, `## Persona`, `## What you remember from earlier runs` (memory), `## Your flock`, and a simulator note.
- Finally, the `.duck` Markdown body verbatim.

**The VLA section** (`executor_section`, prompts.py:262), verbatim:

> "A learned policy moves this arm when you call `manipulate`, one short segment at a time: you plan, and it executes. Give it one short subtask per call, of the kind a policy like it was trained on and in a few plain words ("pick up the red block", "put it in the bowl"), never the whole task at once. After every segment, judge from the fresh frame your next observation brings what the arm actually did before you choose the next subtask. `manipulate` coming back ok means only that the segment ran, not that the subtask was done, so never declare success on its word alone."

**Each observation** (`build_observation_text`, 696) is compact text, for example:

```
[step 3/40 · llm calls 3/40, …]
state: …
camera: ball at bearing 18° left ~0.58 m
last verb `x`: ok — …
Choose exactly one tool.
```

Images are attached separately as base64 PNG at native camera resolution: 640×480 on the bench webcam, 256 px in sim.

### 4.3 Tool schemas

Verb tools are JSON schemas generated from each verb's pydantic params. ~~They are combined with the meta tools in `META_TOOLS` (prompts.py:166):~~ [corrected: `META_TOOLS` (prompts.py:166) holds only `assess_task`, `declare_success` and `declare_failure`. `REMEMBER` (prompts.py:171) is appended only when memory is on, and `TELL` (prompts.py:197) only in a pilot flock: `tools = registry.tool_schemas(allow) + META_TOOLS`, then `+TELL` if `cfg.link`, then `+REMEMBER` if `cfg.memory` (loop.py ~1697-1703)]:

| Tool | Schema / behaviour |
|---|---|
| `assess_task` | `verdict` ∈ {feasible, infeasible, uncertain}; `reason`; `limits_consulted`; `estimates[]` (object, quantity enum, value, `basis` ∈ {image, detections, task_text, prior_knowledge}, confidence); `needs` (datasheet field names). A `feasible` verdict whose `needs` exceed the body's own datasheet is **refused** (`own_sheet_objection`, loop.py:1336). |
| `declare_success` / `declare_failure` | Each requires a `reason`. |
| `remember` | Saves a note for future runs. |
| `tell` | Flock message to peers. |

The arm's verb parameters:
- `move_joints{positions: {joint: deg}, duration_s ∈ [0.2, 12]}` (adapters/lerobot/src/quackd_lerobot/verbs.py:167).
- `gripper{open: bool}`.
- `manipulate{instruction: str ≤200 chars}` (229).
- `pick{target, max_s ≤60}`.

### 4.4 Providers and exact API parameters

**Anthropic** (`providers/anthropic.py`):
- Default model `claude-opus-5-5`, from the catalogue's first row.
- `max_tokens=16000` (`QUACKD_MAX_TOKENS`).
- `output_config.effort = QUACKD_EFFORT`, default **`medium`**.
- `thinking={"type":"adaptive","display":"summarized"}`, plus `block_binding.prefix_mismatch_behavior:"drop_block"` under beta `thinking-binding-controls-2026-08-01` for models that bind thinking.
- `tool_choice={"type":"any"|"auto","disable_parallel_tool_use":True}`. Opus 5.5 and Fable 5.1 get `auto`, because forced tool choice returns a 400 on them.
- Server-side refusal fallbacks are on by default (`server-side-fallback-2026-07-01`, `fallbacks:"default"`).
- **No `cache_control` anywhere**: "Nothing in quackd sets `cache_control` today" (anthropic.py:172).
- No sampling parameters (temperature, top_p, seed), in this provider or any other.

I checked the API claims (forced-tool-choice 400, block-binding beta, fallbacks beta, Opus 5.5 at $4/$20 with $0.20 cache reads, default effort `medium`) against the bundled Claude API reference. They are correct.

**OpenAI** (`providers/openai.py`):
- Chat Completions with `tool_choice="required"` and `parallel_tool_calls=False`.
- GPT-6 models are routed to the Responses API. Their catalogue prices are Sol $2/$10, Astra $10/$50 and Luna $0.10/$0.50.
- Optional `reasoning_effort`.

**Local models** (Ollama, vLLM, llama.cpp, LM Studio) get a JSON-text fallback and one retry.

### 4.5 Executor, safety and the robot abstraction

**Executor.** `Executor._run_verb` (safety.py:418) applies the checks in this order:
1. abort (`stop` is always allowed through)
2. allowlist
3. verdict gate: only `BEFORE_VERDICT` verbs, i.e. look/read/speak/stop, or verbs flagged `read_only`, may run before `assess_task`
4. pydantic parameter validation
5. confirm gate (y/N)
6. one-segment-at-a-time gate for policy verbs
7. budgets, including the policy-seconds budget

`_execute` (645) races the verb against the abort event and the verb timeout. On a cancel or abort it cancels the verb task and sends `stop`. If the same verb fails 3 times in a row, the run aborts. `Heartbeat` (759) pings every 0.5 s; one miss sends `stop` and sets the abort flag.

**Robot abstraction.** `DuckTransport` (transport/base.py:168) exposes connect/close/get_frame/get_state/send_intent/heartbeat/stop/now/sleep. The **only** commands are `Intent` kinds: ~~move (twist), stop, do (named skill), joint, gripper, look, sound, enable~~ [corrected: `IntentKind = Literal["move", "stop", "do", "look", "sound", "enable", "pose", "joint", "gripper"]` (transport/base.py:18), so `pose` is missing from the list]. "Intents, never motor writes."

Each adapter declares a manifest with embodiment, intents, sensors, verbs, limits, `safety_authority` and a datasheet. Adapters are discovered through the `quackd.adapters` entry-point group. Every upstream name is spelled in a per-adapter `upstream_api.py` tagged VERIFIED or UNVERIFIED.

**The SO-101 specifically** (`quackd_lerobot/verbs.py`, `real.py`):
- `move_joints` ramps each joint linearly across `duration_s` at `TICK_S=0.1`, capped at `MAX_STEP_DEG=5.0` per tick (50°/s).
- Arrival tolerance is `TOL_DEG=5`. A stall is <0.5° movement over 5 ticks.
- Goals outside the calibrated travel are refused.
- Joints at or above `HOT_C=60 °C` are refused.
- The SO-101 has **no IK or Cartesian verb**. The LLM authors raw joint angles, even for "lift two centimetres" (e162).
- "Holding" is inferred from the gripper stalling short of closed. That inference is unvalidated.

### 4.6 Perception

**Detector.** The default is an HSV colour-blob detector (~1 ms). It reports bearing from pixel x and the field of view, and distance from apparent size. YOLOv8n is optional (`--detector yolo`), or it can run on a Jetson ~~(`--host`)~~ [corrected: `--detector host` puts YOLO on the Jetson's GPU. `--host` names the board, and a real-body run with `--host` uses the board's detector by default (README "Perception"). The default model is `yolov8n.pt` (perception/yolo.py:85)].

**How the bench went** (README, "What happened in that run"):
- The detector reported a ball that was not there on every step.
- It called a blue figurine "a person about 8.5 m away" on 8 of 10 steps.
- It never saw the real people in the room.
- Frames were 5.6–12.5 s old when the model saw them.
- The camera cropped the raised arm, so the model verified the wave from joint readings instead.

There is no segmentation, no depth, and no open-vocabulary grounding.

### 4.7 The VLA path (`quackd_lerobot/policy/`, ADR-0048)

**Server.** `quackd policy serve --policy OWNER/NAME@REVISION`:
- Accepts types `act`, `smolvla` and `pi05` only (`SERVED_TYPES`, policy/upstream_api.py:256).
- Reads `config.json` and the processor JSONs before loading anything. It refuses processor steps it has not reviewed, `class` keys and remote code, and requires `--pin` for nested models.
- Loads weights strictly.
- Binds to loopback with a constant-time-compared token.
- Rejects NaN/Infinity in JSON.

**Protocol.** Four HTTP endpoints (protocol.py:87–90): `GET /v1/policy`, `POST /v1/reset` (carries the instruction, motor order and camera sizes), `POST /v1/step` (carries state, frames and the last command; returns a chunk) and `POST /v1/end`. The instruction becomes LeRobot's `build_inference_frame(..., task=instruction)` (pipeline.py:826–860), so it acts as a language prompt for SmolVLA and pi05. ACT ignores it.

**Fit check.** Before torque, the arm refuses a policy whose state or action dimensions, motor names, cameras or frame sizes differ from the arm. It also refuses one whose learned-state 1st–99th percentiles fall outside the calibrated travel. On the lab arm's calibration that refused **55 of 68** servable SO-100/101 ACT checkpoints on the Hub. `--accept-other-frame` overrides it.

**Client loop** (`PolicyLoop`, policy/loop.py):
- Paces ticks on deadlines, `start + k·period`. Overrun ticks are skipped, never sent late.
- At most one request is in flight.
- Refills when the queue drops to `max(2·latency_ticks, chunk/2)` (`refill_at`, 231).
- Incoming chunks drop actions for ticks already played and *replace* the queue tail.
- The server refuses to declare a latency greater than chunk/2, because that would starve every chunk (`starved_each_chunk`, 248).
- Per-send speed cap is `max_step_deg/TICK_S/rate_hz`, about 1.7° at 30 Hz (`speed_cap`, 184).
- Guards end the segment with the arm held: 1 s with nothing to play (`STARVE_S`; 5 s for the first chunk), a stall over 1 s, a hot joint, torque off, a missing camera frame, a non-finite action, 3 failed sends in a row, or a goal held past travel for 1 s.

**Task-level contract.** A `duck: 3` file can narrow `manipulate.instruction` to an enum of at most 12 instructions, set `segment_s` (≤60 s) and set `total_s`.

**`--controller vla`** (providers/vla.py) is a model-free pilot:
- It always answers `uncertain`, so a human decides whether to start.
- It plays each instruction in order.
- Only a person's "yes" to "Did the arm do it?" counts as success (`JudgedPilot`).

**Evidence for this path.**
- CI runs a tiny random ACT.
- On the simulator, `natsuki0000/act-so101-bluecap@82f75fe` ran on a laptop CPU. With `gpt-6-sol` as pilot it ran two segments, after which the model declared failure because there was no blue cap in the scene. This proves the plumbing only.

### 4.8 Decision LLMs: the stepper (`quackd/agent/decision/stepper.py`, ADR-0040/0043)

**Which turns it may answer.** A verb counts as a "choice" only if every parameter in its JSON schema is an enum, a constant, a boolean or optional-null. So on the arm the stepper can only pick from `report_state`, `stop`, `place`, `gripper(open=true|false)` and `observe`. `move_joints`, `pick` and every meta tool always go to the LLM. `manipulate` is offered but never taken ("shadow_only").

**What it sends.** One request per turn with four parallel questions (`build_questions`, 378):
- `next_verb` (Choice, always including `escalate`),
- `done` (Noul),
- `need_human` (Noul),
- `feasible` (Choice, recorded only).

The state is a short text JSON: goal, success_when, body, budget, now, camera, last, recent, tried, notes. No images are sent.

**Gating.** Floors (82): stop 0.50, read 0.60, motion 0.85, confirm 0.90. Only 0.5 and 0.9 come from TypeSafe's page, and I confirmed that page presents them as *illustrative*, not calibrated. Other rules:
- Timeout `TIMEOUT_S = 1.0` via `asyncio.wait_for` (849).
- No repeating the previous stepper call.
- At most ~~8~~ [corrected: `streak_limit` = min(`MAX_IN_A_ROW` = 8, half the run's step budget) (stepper.py:296, 666)] consecutive stepper turns.
- Every error escalates to the LLM.

**Measured gains: none.** No decision model has been measured. The README's 1.2× speedup on the wave run and 2.9× on `arm-grip-check` are arithmetic from TypeSafe's published 0.114 s/call figure. I verified the 0.114 s on typesafe.ai, and $42 per billion input tokens works out to $0.042/M.

### 4.9 Flocks, MCP, memory, record

- **Pilot flock.** One full loop per robot (2–8 robots). Peers' datasheets go into each pilot's prompt (`flock/talk.py:138`): "Nobody is in charge and nothing assigns the work: you divide it between you by saying what you will do." A `tell` tool sends messages. It has been exercised only with the scripted pilot on mock and simulated bodies.
- **Coordinator flock.** A deterministic Contract-Net auction with at most one planner LLM call, for simulated Microducks only.
- **MCP.** `serve-mcp` exposes 9 `robot_*` tools behind the same executor. No MCP session has driven real hardware.
- **Memory.** A per-robot JSONL of notes and episode outcomes, injected into the prompt.
- **Record.** Every run writes `transcript.jsonl` with every prompt, call, gate, intent, token count and cost; frames; `summary.json`; and `terminal.txt`.
- **Simulator rehearsal.** `quackd robot twin` and `quackd preflight` run task files on `lerobot:mujoco` (the real backend code over the SO-ARM100 MJCF). `.sim.yaml` sidecars define ground-truth checks.

## 5. Results (all from the repo; trial counts as stated)

| Setting | Model | n | Outcome |
|---|---|---|---|
| Real SO-101 wave run (README hero) | `gpt-6-astra` | 1 run, 10 calls | Success by self-declaration. 78.8 s wall, of which **62.1 s (79%) waiting on the model** (3.3–8.2 s/call, mean 6.21 s) and 12.2 s moving. 49,096 input / 491 output tokens, about $0.52 at catalogue rates (my arithmetic). 97 arm commands, all accepted. |
| Real SO-101, 2026-09-15 | `gpt-6-astra` | 12 runs | Gestures only. The arm fell at the end of every run (LeRobot `disconnect()` drops torque). ~~One run aborted on a heartbeat timeout~~ [corrected: one *dry* run aborted on a single heartbeat `TimeoutError` (README "Which robots work", `lerobot:real` row)], one on `uncertain` → human said no. `remember` was called in 7 of the 12 runs. |
| Real SO-101, 2026-09-23 (v0.12.0) | `gpt-6-sol`/`astra` | 26 runs | **19/26 never moved the arm** at a pilot's request. 6 aborted reaching the rest pose; 21/21 that reached close ended `TORQUE_LEFT_ON`. ~~Cause: the rest pose lay 20.5° past the calibrated travel, and the STS3215 clamps goals.~~ [corrected: that was one of five faults, the one behind the 6 rest-move aborts and the 21 `TORQUE_LEFT_ON` endings (`shoulder_lift` recorded at -104.7° against a travel of ±84.2°, 233 ticks or 20.5° below the floor; the STS3215 clamps `Goal_Position` to its EEPROM limits). The other four: the verdict gate (12 runs stopped at the y/N on `uncertain`; own-datasheet objections; the "Decline" prompt line), 3 connect failures on a bad status packet, `move_joints` lacking a duration, and no torque release (CHANGELOG 0.14.0). The 26 runs were 23 from ten task files plus 3 typed goals, on a build "still numbered 0.12.0" that became 0.13.0.] |
| Microduck sim2d `find-and-kick` | scripted | 10 seeds | 10/10 (ground truth). I reproduced seed 3: 4 steps, 0.2 s. |
| Microduck MuJoCo (upstream RL gait) | scripted | 10 seeds | 9–10/10; seed 4 is marginal. |
| `lerobot:mujoco` grasp via backend verbs | scripted | 10 seeds | 10/10 lifts (sim truth). |
| Preflight e001–e005 on twin | `gpt-6-sol` | 24 runs | All "passed" (the code survived; tasks were not done), $1.06. |
| Sim wave | `gpt-6-sol` | 1 | 7 steps, 9 calls, 69 s, $0.05. |
| Local models, find-and-kick sim | Qwen2.5-Coder-14B; Qwen3-32B-AWQ | 2 + 2 | Successes. Thinking on vs off: 1,290 vs 263 output tokens. |

## 6. Third-party evidence: issue #25

The most rigorous evaluation of quackd is Vallhalen's (issue #25, 2026-09-15→29), on Qwen3-32B-AWQ/vLLM in `microduck:sim2d`.

**The allowlist decides the verdict.**
- `--goal "Find the ball and kick it."` (15 allowed verbs): `feasible` in 1 of 6.
- The `.duck` file with 6 verbs: 6 of 6.
- At n=80: 12/80 vs 37/80 (**p=2.9e-5**).

With many verbs the model says "I cannot determine feasibility yet" instead of calling `observe`.

**Measurements are not reproducible.** No provider sets temperature or seed, so the same build and prompt gave 6/6 on 15 Sep and 2/6 on 18 Sep. Pinning sampling through `--extra-body '{"temperature":0}'` made the results deterministic.

**The maintainer's prompt rewording did not change verdicts:** 4/30 → 3/30. Blind-coded reasons moved only slightly (p=0.015), and 86% of `uncertain` verdicts still cited the forbidden reason.

**A one-sentence observation-time hint did.** Injected into the observation while no verdict exists and nothing is detected ("nothing detected yet. That is not a reason for `uncertain`…"):
- Feasible went from 5/30 to **27/30** (p=1.14e-8), and all 27 then succeeded.
- On an impossible "carry" task, correct `infeasible` went from 5/30 to 17/30, and `feasible` stayed at 0/30.

This fix was **not merged**: `grep` finds no such line on `main`. The maintainer changed the static prompt instead (0.14.0), and that change is unmeasured.

The lesson for any LLM gate: **context delivered at decision time beats rules in the system prompt**, and every gate must be measured at n≥30 with sampling pinned.

## 7. Assessment

**Strengths**
- **Execution contract.** A clean separation between the model's choice and a non-negotiable executor: a fixed enforcement order, verbs racing abort and timeout, `stop` always permitted, budgets that include LLM calls and policy seconds, and human confirm gates recorded with *who* answered (a person vs `--yes` vs a pipe).
- **Manifest-driven vocabulary.** A body only ever sees the verbs it has, and `quackd validate` checks a task's `requires` against the manifest before connecting.
- **Body self-knowledge in the prompt.** The datasheet carries a confidence label on every number, plus "cannot" lists and a structured feasibility verdict with `needs` checked against the sheet. This is a good pattern, though fragile (§6).
- **Well-engineered VLA integration.** The policy runs in its own process, has a defensive checkpoint loader and an auth'd protocol, and the async chunk queue has correct refill and stale-action logic. Speed caps and per-tick guards are applied *outside* the policy. And `manipulate`'s "ok" explicitly means only "the segment ran".
- **Instrumentation.** Per-call latency and cost, transcripts and replay, and a shadow mode for any cheaper component before it may act.
- **Simulator rehearsal.** The twin runs the *real backend code* over a simulated follower that reproduces firmware clamping.

**Weaknesses**
- **The evidence is gestures plus integration failures.** There is zero real manipulation, zero real VLA, zero real decision-LLM, and zero Claude runs on record. The scope (7 bodies, 11 vendors, flocks, MQTT, LAN, Jetson, browser demo) far outruns what has been validated.
- **The latency architecture is naive for manipulation.** One verb per turn at ~6 s, frames 6–12 s stale, and the LLM authoring absolute joint angles with no IK and no visual servoing.
- **Perception is a toy.** An HSV blob detector emitted confident false text that the model had to ignore.
- **Success is self-declared** on solo runs.
- **No sampling control** makes quackd's own evaluations unreproducible.
- **Cost and cache hygiene.** No prompt caching, and per-turn image trimming edits history, which breaks Opus 5.5's prefix-bound thinking. That forced the `drop_block` / 8-exchange workaround.
- **Code quality is mixed.** Every function is surrounded by multi-paragraph docstrings recounting incidents ("On 2026-09-15 …"). It is AI-generated and extremely defensive, with good test discipline, but hard to read and to fork.

**Genuinely novel vs repackaged**
- *Repackaged:* LLM-as-skill-selector (SayCan, Code-as-Policies lineage); MCP robot tools; LeRobot policies; Contract-Net auctions.
- *Novel-ish and worth studying:*
  - (a) A schema-derived split between "choice" turns and "number/sentence" turns, routing choice turns to a calibrated classifier with an `escalate` option and anti-loop rules.
  - (b) The datasheet-grounded feasibility verdict as a motion gate.
  - (c) A planner LLM → bounded VLA segments → re-observe → human-judged success protocol, with shadow-mode promotion.

**Maturity:** pre-alpha in the field, though polished as software. v0.16.1 shipped ~~33 days after the first commit, with 16 releases~~ [corrected: 32 days after the first commit (2026-08-28 12:05 +0200 to release 2026-09-29T20:26Z), as the **17th** GitHub release (v0.1.0 to v0.16.1; `gh api repos/rokbenko/quackd/releases`)].

### What Ilia's Opus-backbone harness should borrow

1. **An executor gate with this exact order, and `stop` always allowed.** Verbs race an abort event. A heartbeat or deadman exists independent of the LLM.
2. **The planner → VLA-segment contract.** `manipulate(instruction)` with a bounded duration, a step cap, per-tick guards, a policy process outside the bus process, and "ok ≠ done", followed by a mandatory fresh look. The `refill_at` / `starved_each_chunk` arithmetic is directly reusable. Note the 55/68 checkpoint-fit refusal statistic: a calibration-frame mismatch will gate any VLA-zoo plan.
3. **A typed discrete-action head with `escalate` plus shadow mode.** This is exactly the "light learned action head" niche. It can be a classifier over a closed action set, gated by per-class floors, never authoring numbers. Measure agreement in shadow first.
4. **A body datasheet and a feasibility gate,** with the decision-time hint from §6, sampling pinned for evaluation, and n≥30 per cell.
5. **A real-backend simulator twin plus preflight with ground-truth sidecars,** and transcripts that log cost and latency per call.

### What it should avoid

1. **Editing history to trim images on Opus 5.5.** Keep the transcript append-only, use `cache_control`, and remove old frames with server-side context editing or mid-conversation system messages. Do not mutate earlier turns.
2. **One serial LLM call per primitive** while the robot idles (79% of wall time). Batch plans, overlap thinking with motion, or delegate to a VLA or skills.
3. **Letting the LLM author joint angles open-loop.** Provide Cartesian/IK verbs and visual verification.
4. **Weak text-only perception.** Use open-vocabulary detection, segmentation and depth, as piper-astra-jev does with GroundingDINO, SAM3 and RealSense.
5. **Self-declared success.** Use an independent checker: sim truth, a VLM judge or a human.
6. **Breadth before one body works end-to-end on real hardware.**

## Sources

- https://github.com/rokbenko/quackd (README, CHANGELOG, PLAN.md; docs/architecture.md, docs/decision-llms.md and docs/decision-llms/{jev,kev,laya}.md, docs/policies.md; ADR-0003, 0040, 0045, 0048; docs/examples/lerobot/README.md; source files cited inline). Read at commit 25635c6.
- https://github.com/rokbenko/quackd/issues/25 (Vallhalen's gate measurements and the maintainer's reply)
- https://github.com/rokbenko/quackd/releases/tag/v0.15.0 , https://github.com/rokbenko/quackd/releases/tag/v0.12.0 , https://github.com/rokbenko/quackd/releases/tag/v0.9.0
- https://pypi.org/project/quackd/
- https://www.quackd.org/llms.txt (site mirror; the root page is JS-rendered)
- https://x.com/rokbenko/status/2094747138454339637 (UNREACHABLE: HTTP 402; text known only from search snippets)
- https://typesafe.ai/ (0.114 s vs 8.566 s example; 193.6×/444.6× claims; $42 per billion input tokens)
- https://docs.typesafe.ai/confidence (0.5/0.9 thresholds presented as illustrative; formula undisclosed)
- https://docs.typesafe.ai/concepts/system-one
- https://github.com/jaredpalmer/kev , https://github.com/NandhaKishorM/laya , https://github.com/razorback16/openjev (existence verified via the GitHub API)
- https://www.datacamp.com/blog/system-one-models-jev , https://www.truefoundry.com/blog/typesafe-ai-jev (third-party Jev context; vendor benchmark claims UNVERIFIED)
- https://gitdiagram.com/rokbenko/quackd
- Bundled Claude API reference (claude-api skill; shared/model-migration.md, error-codes.md): used to verify the Opus 5.5 / Fable 5.1 API behaviour quackd relies on.
- Cross-reference: research/sources/piper-astra-jev.md and research/sources/llm-robotics-playground.md (teammate notes)

## Verification (fact-check pass)

Checked on 2026-10-01 against the clone at `25635c6` (`git fetch` shows `origin/main` still at `25635c6`), the GitHub API (`gh`), PyPI JSON, issue #25 and its 4 comments, quackd.org/llms.txt, typesafe.ai, docs.typesafe.ai/confidence, and the bundled Claude API reference (claude-api skill: `shared/model-migration.md`, `shared/error-codes.md`). I also ran `uvx --from "quackd[microduck]==0.16.1" quackd run find-and-kick --llm fake --seed 3` locally. Paths below are relative to `research/repos/quackd/`.

### Confirmed (seen in a primary source)

- **Repo metadata.** 246 stars, 26 forks, Apache-2.0, homepage www.quackd.org (`gh api repos/rokbenko/quackd`). 565 commits, first one 2026-08-28 12:05 +0200; 544 by Rok Benko (535 + 9 under two emails). PyPI has `quackd` plus 7 adapter packages (`quackd-microduck`, `-lerobot`, `-rosbridge`, `-open-duck`, `-xlerobot`, `-alohamini`, `-toddlerbot`), all at 0.16.1 under Apache-2.0.
- **Claude trailers.** 508 commits carry a `Co-Authored-By: Claude` trailer. Counts by name: Opus 5 (1M) 333, Opus 5.5 (1M) 69, Fable 5.1 32, Fable 5 28, Opus 5 20, Sonnet 5 16, Opus 5.5 10.
- **Size.** 49 ADRs (`docs/adr/`). 2,876 `def test_` functions. Test code is 73,299 lines.
- **Author.** Bio and location (Murska Sobota) confirmed via the GitHub API. The bio reads "Solopreneur building in public • 500 job rejections → 10+ failed startups → quackd".
- **quackd.org/llms.txt** has a "Token" section linking "$QUACKD on pump.fun", described as "community token on Solana ... No utility, not affiliated with the open-source project, and it does not fund development", plus a DexScreener link.
- **Hero run** (README "What happened in that run"): 78.8 s run, plus 10.3 s to open the port. 10 calls taking 3.3–8.2 s each, 62.1 s total (79%). 12.2 s of arm motion (15%). 49,096 input / 491 output tokens, of which 22 were reasoning. 97 commands, all accepted. The cost arithmetic checks out: at Astra's catalogue $10/$50 per M, 0.491 + 0.025 ≈ $0.52.
- **Hardware days.** 2026-09-15: 12 `--goal` runs on gpt-6-astra, plus a scripted `lerobot-lookout`. 2026-09-23: 26 runs, 19/26 never moved the arm. 6 runs aborted on the rest move; 21/21 that reached their close ended `TORQUE_LEFT_ON`. The fold was 233 ticks (20.5°) below the floor (ADR-0045 §Context).
- **Loop code.** `REPROMPT` (loop.py:103) and its use at ~2000. Only the first of several tool calls runs. `keep_images_for_last_n: int = 2` (loop.py:169). `_history_for_provider` (1506). `own_sheet_objection` (1336). `stepper.advise` (1892). `provider.step` (1932). `executor.run_verb` (2136). Teardown order in the `finally`: heartbeat.stop, stop, `_hand_back`, `_rest`, `_offer_release`, transport.close, summary, `record_episode`.
- **Anthropic provider** (`quackd/agent/providers/anthropic.py`; the note's `providers/anthropic.py` is shorthand):
  - `BINDING_TRIM_PERIOD = 8` (:54).
  - `max_tokens` from `QUACKD_MAX_TOKENS` with default "16000" (:286).
  - `QUACKD_EFFORT` default "medium" (:287).
  - `thinking={"type":"adaptive","display":"summarized"}`, plus `block_binding.prefix_mismatch_behavior:"drop_block"` for binding models (:320-329).
  - Betas `server-side-fallback-2026-07-01` with `fallbacks:"default"` and `thinking-binding-controls-2026-08-01` (:349-355).
  - `{"type": "any"|"auto", "disable_parallel_tool_use": True}`.
  - The "Nothing in quackd sets `cache_control` today" comment at :172.
  - No temperature, top_p or seed in any provider.
- **Catalogue.** `claude-opus-5-5` is the first row, `Price(4.0, 20.0, 0.2, 5.0)`, with `forced_tools=False` and `binds_thinking=True`. GPT-6 Sol $2/$10, Astra $10/$50 and Luna $0.10/$0.50, all `api="responses"`.
- **Bundled Claude API reference agrees** on each of these: forced `tool_choice` any/tool returns 400 on Opus 5.5 and Fable 5.1; Opus 5.5 is $4/$20 with $0.20 cache reads; Opus 5.5's default effort is `medium`; the block-binding beta and its `drop_block` field exist; the fallback beta exists with `"default"`.
- **OpenAI provider.** `tool_choice="required"`, `parallel_tool_calls=False`, optional `reasoning_effort`, and `--extra-body`.
- **Prompts** (`quackd/agent/prompts.py`). The `build_system_prompt` return f-string starts at :656, and the scaffold quote is verbatim. `executor_section` (:262) quote is verbatim (the `frames=True` branch). `build_observation_text` (:696) format is confirmed, including "While you were not asked, the stepper chose these (newest last):".
  - **5,304-character `find-and-kick` prompt: reproduced exactly.** My run's prompt was 5,605 characters because memory held one earlier episode. With that replaced by the empty-memory placeholder it is 5,304.
- **Seed 3 reproduction:** 4 steps and 6 LLM calls. I measured 0.3 s wall time against the note's 0.2 s; this varies by machine.
- **Executor** (safety.py). `_run_verb` (:418) applies the gates in this order: abort (stop exempt), allowlist, unknown verb, verdict (`BEFORE_VERDICT` or `read_only`, not `MOVES_THE_BODY`), pydantic params, confirm, the one-segment gate, budgets (policy seconds first). `_run_admitted` (:559) then does get_state, abort_when, preconditions, dry-run, `_execute`. `_execute` (:645) races `asyncio.wait(FIRST_COMPLETED, timeout=verb.timeout_s)` against `abort.wait()`. `Heartbeat` (:759) has `period_s=0.5`; one failure sends stop and sets abort.
- **Arm verbs.**
  - `TICK_S=0.1`, `TOL_DEG=5.0`, `STALL_TICKS=5` and `STALL_DEG=0.5` (verbs.py:51-58).
  - `MAX_STEP_DEG=5.0` (real.py:151) and `HOT_C=60.0` (real.py:236). The holding band is `HOLD_MIN=8`..`HOLD_MAX=90`.
  - `MoveJointsParams` (verbs.py:167): `duration_s` default 5.0, range [0.2, 12].
  - `pick`: `max_s` default 20, range 1..60.
  - `ManipulateParams` (:229): instruction ≤200 characters.
  - `e162/first-grasp.duck` says "lift it two centimetres" with only `move_joints`/`gripper`.
- **VLA path.**
  - `SERVED_TYPES = ("act", "smolvla", "pi05")` (policy/upstream_api.py:256).
  - Endpoints `/v1/policy`, `/v1/reset`, `/v1/step` and `/v1/end` (protocol.py:87-90).
  - `build_inference_frame(..., task=self._task)` (pipeline.py:857-858).
  - `hmac.compare_digest` token check. Loopback-only bind (unless `--behind-tls`).
  - `parse_constant` refuses NaN/Infinity, and `allow_nan=False` on writes.
  - `strict=True` weight loading. `class` keys and `trust_remote_code` are refused.
  - `refill_at = max(2·latency, chunk//2)` (policy/loop.py:231). `starved_each_chunk = max(0, 2·latency − chunk)` (:248). `speed_cap` = 5/0.1/30 ≈ 1.67° at 30 Hz (:184).
  - `STARVE_S=1`, `FIRST_CHUNK_S=5`, `STALL_S=1`, `FAILED_SENDS=3`, `CLIP_SUSTAIN_S=1`. Rate bounds 1–60 Hz.
  - Task files: `MAX_INSTRUCTIONS=12`, `DEFAULT_SEGMENT_S=10`, `SEGMENT_MAX_S=60`, and a `policy:` section requires `duck: 3`.
  - `--controller vla`: always `uncertain` and gated by `JudgedPilot`.
  - The 55/68 checkpoint-fit refusal (CHANGELOG, ADR-0048, docs/adapters/lerobot.md:1070).
- **Stepper.**
  - `FLOORS` (stepper.py:82): brake 0.50, read 0.60, motion 0.85, confirm 0.90.
  - `TIMEOUT_S = 1.0` (decision/base.py:17), applied via `asyncio.wait_for` (stepper.py:849).
  - Four questions: `next_verb`/`done`/`need_human`/`feasible` (:378).
  - `manipulate` is `SHADOW_ONLY`.
  - There is a `repeat` gate.
  - The frozen classification for `lerobot` (tests/test_decision.py:78): gripper 2 calls, place 1, report_state 1, move_joints/pick/manipulate not a choice.
- **Decision catalogue.** Jev is pinned `jev-1.13.0` at $0.042/M input. Kev is "Qwen3.5 0.8B/4B/9B frozen + rank-16 LoRA + pointer head" (docs/decision-llms/kev.md:43). Laya is `convaiinnovations/laya-typed-decisions`, a 421M ModernBERT-large model. docs/decision-llms.md:205 says OpenRouter's `/api/alpha/decisions` is unreachable because the System One client's path is fixed.
- **TypeSafe pages.** typesafe.ai shows "Completed in 0.114s" against 8.566 s, "Cost $0.000081", "$42 Per Billion input tokens", "193.6x Faster, 444.6x Cheaper" and "238x" lower input price than Fable 5.1. docs.typesafe.ai/confidence presents 0.5/0.9 only inside a code example and says thresholds "depend on your domain".
- **1.2× / 2.9× estimates.** Both are in the README "Performance" section and are worked out in docs/decision-llms.md:463-535, labelled "estimates, not measurements".
- **Results table.**
  - Sim wave: gpt-6-sol, 7 steps, 9 calls, 69 s, $0.05 on 2026-09-28.
  - Preflight: 12 files from e001–e005, 24 runs, $1.06 of gpt-6-sol on 2026-09-29 (docs/examples/lerobot/README.md:53).
  - The Qwen transcripts (4 of them) all succeed. Thinking on vs off is 1,290 vs 263 output tokens on the five decisions both runs made; the totals are 2,049 vs 263.
- **Issue #25** (opened 2026-09-15, still OPEN). Every figure in §6 matches the thread:
  - The issue body: 1/6 vs 6/6.
  - 2026-09-18: 15-Sep 6/6 vs 18-Sep 2/6; temperature-0 determinism; 4/30 → 3/30; n=80 12/80 vs 37/80, p=2.9e-5; blind coding 42/42 → 44/51, p=0.015; 86%; carry task 5/30 → 17/30 with feasible 0/30.
  - 2026-09-20: 5/30 → 27/30, p=1.14e-08; 32/32 feasible runs succeeded.
  - The hint line is not on `main` (grep finds nothing).
- **MCP:** 9 tools. **Flocks:** pilot flocks have 2–8 members (flock/pilots.py:52). The talk prompt quote is at flock/talk.py:150; the note's :138 is the start of `prompt_section`.

### Corrections (also marked inline)

1. Contributors. "four outside contributors and dependabot" → 3 outside people (Bayway, also committing as Massimiliano Fiori; Vallhalen; r0jin) with 9 commits, plus dependabot[bot] (7) and github-actions[bot] (5).
2. Cause of the 19/26 (TL;DR and §5 table). "Almost all calibration/torque/rest-pose bugs, not LLM reasoning" and "Cause: the rest pose lay 20.5° past..." → CHANGELOG 0.14.0 lists **five faults**:
   - the rest pose past the travel;
   - the verdict gate refusing honest answers: an own-datasheet objection to a `feasible` wave, **12 runs stopped at a y/N on a pilot's `uncertain`**, and a pen run ended by the "Decline any task that hinges on..." prompt line;
   - 3 connect failures on a bad Feetech status packet;
   - `move_joints` with no duration;
   - no torque release short of the power switch.

   Three pilots also refused to move because a joint read past its travel. The rest pose is the cause of the 6 + 21, not of all 19.
3. Diagram. `Executor.run_verb (quackd/safety.py:418)` → `run_verb` is at :330; `_run_verb` is at :418.
4. Tool schemas. `META_TOOLS` contains only assess_task/declare_success/declare_failure. `remember` is added only with memory on, and `tell` only in a pilot flock.
5. Intent kinds. `pose` is also an `IntentKind` (transport/base.py:18).
6. Jetson detector flag. It is `--detector host`, not `--host`. `--host` names the board.
7. 2026-09-15 heartbeat abort. It was a **dry** run.
8. Stepper streak limit. The cap is min(8, half the step budget) (`streak_limit`, stepper.py:666), not a flat 8.
9. Release cadence. v0.16.1 is the **17th** GitHub release and shipped **32** days after the first commit, not 16 releases in 33 days.

### Unverifiable or approximate

- **"~78k LOC of code."** By my count, Python outside `tests/` is 76,006 lines (174 files under quackd/, adapters/, bridge/ and scripts/ give 74,429). The web/ JS/HTML/CSS adds 5,281. So ~78k is roughly right, but the counting method is not stated.
- **X announcement.** I did not re-fetch it (the note reports HTTP 402), so its text is still known only from search snippets: UNVERIFIED.
- **"No HN/Reddit threads."** A web search found only GitHub pages, forks (`r33drichards/quackd`, `Vallhalen/quackd`) and gitdiagram. This is consistent with the note but does not prove absence.
- **"No published Claude-piloted run at all."** This holds for transcripts and recordings: the README says only gpt-6-astra and gpt-6-sol recordings have a model in the loop, and "no Claude Desktop session on record". However, the 2026-09-28 run where a Hub ACT drove the twin "over `quackd serve-mcp`" does not name its MCP client. Whether that was a Claude Code session is UNVERIFIED.
- **Cross-references to piper-astra-jev and llm-robotics-playground.** These match the teammate notes. I did not re-verify them against their primary sources.

### Nuances and missed details (added)

**1. The 55/68 refusals are an artefact of one arm.** All 55 refusals were on `shoulder_lift`. The cause is that the lab arm's own calibration never recorded its fold, while the checkpoints' training arms did (CHANGELOG ~417; docs/adapters/lerobot.md:1070). It is not evidence that most Hub checkpoints are mis-framed for any SO-101. The general lesson is narrower: VLA fit checks depend on calibrating through the full range of motion.

**2. Measured VLA latencies** (docs/policies.md:80-130, 295-320; Intel i5-10210U laptop, no GPU, 2026-09-28):
- `natsuki0000/act-so101-bluecap@82f75fe`, first bench: round trip median 620 ms (p99 653 ms), server inference median 599 ms. With no `--latency-s` it achieved 18.7 Hz of 30, skipping 113 of 300 ticks.
- Re-served with `--latency-s 0.89`: 29.1 Hz of 30 (873 of 900 ticks), median round trip 634 ms (p99 880 ms), 27 starved ticks, all before the first chunk.
- `lerobot/smolvla_base@d9f33c9`: **169–188 s per chunk** in bf16 (474 of 500 tensors are bf16 and that CPU has no native bf16), about 18 s cast to fp32. Both exceed the client's 10 s step wait, so it never returned a chunk.
- pi05 needs the gated `google/paligemma-3b-pt-224` tokenizer.
- FLUX 3 Action is "not served at all".
- Remote serving: frames go raw over an `ssh -L` tunnel, or as JPEG (quality 90 by default) behind `--behind-tls`.

**3. Simulator hallucinations and limits** (docs/examples/lerobot/README.md:53-75). In the 24-run preflight:
- Both `e002` pilots (gpt-6-sol) "said a switch sat right below the gripper, where the table has none, and kept looking until their fourteen steps ran out".
- The `e001` drawing pilots declared failure because there was no marker or paper.
- Only `e162` and `e165` have `.sim.yaml` ground-truth sidecars, and **neither lays out on `arm-01-sim`**. Every connect is refused, e.g. "closing the gripper stops its moving finger 0.4 mm clear of cube". So no real-pilot grasp has been scored by sim truth.

**4. The 2026-09-23 runs.** They were 23 runs from ten task files (from nine experiments) plus 3 typed goals, on lerobot 0.6.1 under Windows. The build was "still numbered 0.12.0" but became 0.13.0. 223 task files now sit in docs/examples/lerobot/ (e001–e165; e068, e119 and e126 have none). Their budgets are estimates, and all but those ten files have never run on an arm. Servo temperature was 34–40 °C, with a single 67 °C reading that was not repeated 0.17 s later.

**5. Two admitted prompt bugs in the hero run** (README:320). The prompt said "Budgets: 40 steps" while 10 were enforced, and the `--goal` strategy paragraph named `observe`/`search_scan`/`go_to`, which the arm lacked. Both are now fixed by `goal_strategy()` (prompts.py:230). The model also asked for `elbow_flex` 25° and the arm settled at 30°. The frames reached the model 5.6–12.5 s old (about 8 s on average).

**6. Claude-specific implications not stated in the note** (bundled Claude API reference):
- **Sampling.** Opus 5 / Opus 5.5 / Fable 5.1 **reject `temperature`/`top_p` with a 400**. So the issue-#25 remedy of pinning `temperature: 0` is unavailable on a Claude backbone. Gate measurements there must rely on n≥30 statistics and on a fixed `effort` instead.
- **No forced tool choice.** Because forced tool choice 400s on Opus 5.5 / Fable 5.1, quackd sends `tool_choice: auto`. Zero-tool-call turns are therefore possible: they trigger `REPROMPT`, and a second miss ends the run as `failure`. How often this happens on Opus 5.5 is unmeasured.
- **Effort is lowered on every model.** quackd's `QUACKD_EFFORT=medium` default goes to every Claude model that takes effort, where the API default is `high` (e.g. Opus 5, Sonnet 5). Only on Opus 5.5 does it equal the API default.
- **`drop_block` scope.** It discards the first mismatched thinking block **and every thinking block after it**, for that request only (model-migration.md, block-binding section). quackd's 8-exchange trim therefore loses all reasoning continuity after each trim point.
- **The append-only alternatives the note recommends exist on Opus 5.5.** Mid-conversation `role:"system"` messages, server-side context editing (`clear_tool_uses_20250919`) and compaction are all listed in the Opus 5.5 feature set.

**7. Seven decision-LLM presets, not three** (decision/catalogue.py:86-190): `jev` (hosted), `kev` (Qwen3.5 + LoRA, :8009), `von` (395M encoder, "about 18 ms on a GPU", :8000), `openjev` (DiffusionGemma behind vLLM in docker `razorback16/openjev:0.3.0`, about 18 GB of weights, :8080), `opendecision` (zero-shot encoder, CPU), `local` (any `/v1/systemone` server) and `laya` (in-process).

**8. Stepper internals.**
- Request size: 527 tokens per decision on `lerobot:mock`, measured by hand (the estimator prints 421/460).
- `MAX_CALLS_PER_VERB=12`.
- State caps: soft 6,000 characters, hard 24,000. Trim order: flock, notes, recent, tried, camera. Never trimmed: goal, success_when, body, where, now, last.
- `DONE_THRESHOLD`/`HUMAN_THRESHOLD` are 0.5, on raw Noul probabilities.
- The escalate criterion text: "None of the other options is the right action now, or the right action needs a number, an angle, a distance, a target name or a sentence...".
- Two measured shares, on the mock with a fake decision LLM: `arm-grip-check` 4/6 turns, `lerobot-lookout` 3/6. The worked wave estimate is 78.8 s → about 65.9 s, roughly 16% shorter.

**9. Executor details.**
- The repeat-failure abort is not hard-coded. It is parsed from `.duck` `abort_when` text by `REPEAT_FAIL_ABORT_RE` (duckfile/schema.py:45, 656). The `--goal` contract supplies "Same verb fails 3 times in a row" plus "Battery below 15%" (duckfile/parser.py:138). Other `abort_when` lines are advisory and only shown to the model.
- Arm preconditions are `move_joints`: torque_on and not_hot; `place`: holding (quackd_lerobot/__init__.py:~182).
- The arm manifest gains `observe` only when a camera answers at connect.
- At connect, a "fall-blind" warning must be acknowledged for walking bodies that cannot detect a fall.

**10. MCP specifics.**
- Tools: `robot_list`, `robot_list_verbs`, `robot_observe`, `robot_assess_task`, `robot_run_verb`, `robot_say`, `robot_remember`, `robot_recall` and `robot_load_duckfile`.
- Transport is stdio only, so there is no remote connector yet.
- Starting the server drives the arm to its rest pose. There is no Ctrl-C; the brakes are `stop` and the power switch.
- The default budget is 40 steps / 5 minutes without a `.duck`.

**11. Model catalogue.** It holds 117 hand-curated model ids across the 11 vendors, as a snapshot read on 2026-09-12 and 2026-09-23. `--llm` refuses ids it does not list.

**12. Datasheet provenance.** Datasheet figures were read from makers' pages on 2026-09-13; nothing was measured. In 0.14.0 the "Decline any task that hinges on..." line became "say uncertain and name it rather than guessing; do not decline on it alone" (prompts.py:~410). The arm's reach is now published as a 0.4 m estimate summed from URDF link lengths.

**13. Issue #25 is still open.** Vallhalen (2026-09-29) says 0.14.0's changes (`mobility`/`manipulator` can be `none`; "uncertain and name it") are unmeasured. He offered to re-run on 0.16.1. The maintainer declined to narrow the `--goal` allowlist, arguing that it would shape every model's vocabulary from one model's behaviour.

### Net assessment of the note

The body is accurate in nearly every number and code reference I checked. All file:line anchors I tested land on the named symbol, give or take a few lines. The one substantive distortion is the attribution of the 2026-09-23 failures. The other corrections are minor counts and flags.
