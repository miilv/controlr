# GPT-Policy (cheng-haha): "In-Context Robot Learning with VLM Agents"

*Deep-dive note, written 2026-10-01.*

**What I read**
- The repo, cloned to `research/repos/GPT-Policy-cheng`, at `main` @ `ab970d8`. I read all core modules.
- The unmerged branches `fix/xingwu-robodojo-evaluation`, `chore/robodojo-integration` and `page`.
- The paper PDF in the repo (`docs/GPT-Policy.pdf`, 32 pp.), which is also arXiv 2609.19138.
- The live project page, plus the earlier pre-release page repo `cheng-haha/in-context-robot-learning`, cloned to `research/repos/`.
- The X thread and the GitHub issues and forks.
- A third-party machine review (Pith).

**Conventions**
- **[derived]** marks numbers I computed.
- **UNVERIFIED** marks claims I could not check.

## 1. TL;DR

**What it is.** A real-robot study of whether a frozen commercial VLM can do *in-context* robot learning. The main model is OpenAI `gpt-6-astra` at Codex `effort: "medium"`. The context can be:
- a human video;
- a robot video, with or without aligned state/action records;
- a goal image;
- the robot's own interaction history;
- live human interaction.

**How it controls the robot.** At each decision the VLM returns **one JSON tool selection**. The main tools are:
- `move_to` or `move_eef_chunk`, carrying absolute TCP poses `[x,y,z,qx,qy,qz,qw]`;
- `set_gripper`;
- `check_path`;
- `locate_point`;
- `done` / `give_up`.

A host-side "constrained controller" turns each request into motion: linear position + SLERP densification, per-sample IK with residual checks (2 mm / ~1°), and Ruckig time scaling. It then executes, waits for settling, and returns requested-vs-measured feedback plus fresh images.

**The harness is an agent CLI, not a raw API loop.**
- GPT runs through `codex app-server` (JSON-RPC, `outputSchema`, read-only sandbox).
- Claude (`claude-fable-5-1[1m]`) and Kimi K3 run through the `claude` Code CLI in `--print --json-schema` mode with `--tools ""`.

**Headline results.** 10 real tasks × 3 trials, scored by a human.
- Human video took towel and notebook pickup from 0/3 to 2/3.
- Robot video + action took bottle uncapping from 0/3 to 3/3 and plug reinsertion from 0/3 to 2/3.
- Goal-image, self-history and human-interaction tasks scored 3/3, but with **no no-context baseline**.
- Every run is slow: 24–96 decisions per episode at about **12–20 s/decision** [derived], so 5–25 min per task.
- One towel run used **12.0M tokens**.

**Caveats.**
- n=3 per condition, and the labels were not blinded.
- A null human-video result (glue cap) was moved out of the main table before release.
- ~~The code needed to reproduce the plug results is private: task prompt profiles, the Morphi Kino adapter, and the Claude Messages adapter with its completion reviewer.~~ [corrected: several pieces of code are private, but they map to different results. The plug results need the private plug prompt profile (`harness/task_prompts.py` ships `PLUG_PROFILE = None`, `PLUG_CONTROL = ""`) and the demo loader that produced the paper's F3a/F3b action format, which no public loader emits. The Morphi Kino adapter is needed for Movable Exploration. The ARX Claude Messages adapter with its completion reviewer is needed for the Claude Fable comparison (paper App. B).]
- No licence: "redistribution and commercial use are not granted".

**Relationship to GPT-as-Policy.** These are independent groups who released in the same week, with the same model, the same Codex app-server substrate and the same ARX X5 arms.
- GPT-as-Policy cites this work under its old name, "GPT-Policy-Eval".
- An unmerged GPT-Policy branch imports GPT-as-Policy's frozen RoboDojo 50-case panel to set up a head-to-head comparison.

## 2. Provenance

| Item | Fact |
|---|---|
| Authors | Dongzhou Cheng, Taoran Yi, Ye Fang, Xingwu Zhang, Fan Feng, Yixuan Li, Gengxiong Zhuang (co-first); Rongze Wang, Shuai Yang, Wei Song, Weizhi Xue, Minyan Wu, Jie Gui, Jiaqi Wang; Tong Wu (corresponding) |
| Affiliations | 1 **Morphi Robot**, 2 Shanghai Innovation Institute, plus HUST, Fudan, Hunan, CUHK, SJTU, Wuhan, Southeast and Beihang universities |
| Morphi Robot | Shanghai startup (墨奇智能, "Shanghai Moqi Wanxiang"). It showed the wheeled MORPHI KINO at WRC in August 2026. Morphi Kino appears in the paper's Table 3, but its adapter is not public. |
| Repo | Created 2026-09-10 as `GPT-Policy-Eval`, renamed to `GPT-Policy` on 2026-09-15. Public code dump `5840568` on 2026-09-16, version 0.1.0. 291★, 5 forks, 2 issues as of 2026-10-01. |
| Paper | arXiv **2609.19138** v1, submitted 2026-09-16 by Taoran Yi, cs.CV/cs.RO. The earlier project page's BibTeX read `anonymous2027incontext … ICLR submission`. |
| Announcement | X post by @z_code68632 ("Embodied AI @ Shanghai Innovation Institute"), 2026-09-11: "GPT-6 Astra might be the craziest robot policy I've seen… Give it ONE demo per task…" At the time of checking it had 31k views and 146 likes. Asked "How do you feed the video demo to Astra?" (by @YuXiang_IRVL), the author answered: "we use keyframes generated by GPT itself as the context." |
| Licence | `LICENSE`: "No license has been selected for redistribution or commercial use yet." The README says the public tree excludes "private prompts … site-specific calibration, and evaluation history". |

## 3. Method (from the paper)

The paper's formulation is a_t ~ π_θ(·|T, c_t, o_t, f_{t-1}) and (o_{t+1}, f_t) = E(a_t, o_t). Here a_t = (tool name, arguments) and f_t is the tool result. The model is θ-fixed, and "model-declared completion remains distinct from physical task success" (§3.3).

**Context compiler (App. C).**
1. A VLM picks keyframes from candidate frames in overlapping windows.
2. A global review pass deduplicates them, capped at **24 keyframes / 48 images**, each resized to ≤1280 px.
3. In Video + Action mode, each keyframe also carries:
   - the measured state nearest within 0.1 s;
   - the action segment up to the next keyframe, compressed to endpoints, about 1 Hz, and both sides of every gripper-command change.

| Task | Mode | Keyframes | Images | Action samples (raw → kept) |
|---|---|---|---|---|
| Unscrew Bottle Cap | Video+Action | 13 | 13 (top only) | 212 → 205 |
| Remove & Reinsert Plug | Video+Action | 14 | 42 (top + 2 wrists) | 1,405 → 131 |

Human videos used 8 keyframes (towel), 6 (notebook) and 7 (glue cap) (Table 4).

## 4. Architecture

```
 offline (before hardware opens)                     online loop (runtime/runner.py)
 ───────────────────────────────                     ─────────────────────────────────────────
 demo source: mp4 | recorded run dir | MCAP teleop      for step in range(max_decisions=100):
   │ ffmpeg PTS sampling 2 fps, ≤24 cand/30 s window       images = video.snapshot(after=t0)  (3×640×480 JPEG q85)
   │ (+ gripper/stop events ±0.3 s in video+action)        state  = robot.state()            (per-arm joints, TCP, gripper)
   ▼                                                       obs    = protocol.observation(...)  JSON, floats→1e-6
 CodexVideoSelector.select (gpt-6-astra, per window)       decision = agent.decide(obs, images, demo_content if step==0)
   ▼                                                        │  Codex: thread/turn + outputSchema   Claude/Kimi: CLI --json-schema
 CodexVideoSelector.review (global, ≤limit, keep 0/N)       ▼
   ▼                                                      done/give_up → return_home, human labels outcome
 prepare_demonstration: HISTORICAL… header, per-keyframe  else ToolExecutor → robot.execute:
 JSON + image blocks, compressed action rows, END…          FK(start) → linear+SLERP samples (5 mm/0.035 rad)
   ▼                                                        → per-sample IK (seeded) → residual ≤2 mm/1° else REJECT
 run_input.content (sent once, at step 0;                  → Ruckig scalar retime (v,a,j limits) → SDK traj / 100 Hz stream
 retained across Codex thread refreshes)                   → settle check → execution_feedback (req vs measured)
                                                          previous = compact result | IK rejection | tool_rejected
```

## 5. Code walkthrough

All paths below are relative to `src/gpt_policy/`.

### 5.1 Entry point and models

`gpt-policy`, `claude-policy` and `kimi-policy` all go to `main.py:_run` (lines 108-409). Agent profiles live in `configs/agents/*.json`:

| Profile | Model string | Transport | Params |
|---|---|---|---|
| codex (default) | `gpt-6-astra` | `codex app-server` | `effort: medium`, `live_image_window: 8`. Task naming uses `gpt-5.6-luna` at `effort: low`. |
| claude | `claude-fable-5-1[1m]` | `claude` CLI | `effort: medium`, `timeout_s: 180` |
| kimi | `kimi-k3[1m]` | `claude` CLI pointed at a Volcengine Anthropic-compatible endpoint (`~/.claude-volcengine/settings.json`) | `effort: medium` |

- **No temperature or max-tokens settings exist anywhere.** Everything is delegated to the CLIs.
- Codex thread start (`harness/codex.py:74-85`) uses:
  - `approvalPolicy: "never"`, `sandbox: "read-only"`, `ephemeral: true`;
  - `baseInstructions` set to the robot prompt plus `"Robot tool catalog:\n" + json.dumps(tools)`.
- Each decision is a `turn/start` with `outputSchema` and `effort` (`codex.py:127-135`). The final `agentMessage` is parsed as JSON.
- My inference: Codex's built-in shell tool is not explicitly disabled. The code relies on the read-only sandbox, and on rejecting any server→client request (`codex.py:288-296`).
- The Claude path (~~`harness/providers/claude_code/session.py:137-147`~~ [corrected: `session.py:48-58`; lines 137-157 are `close`/`_sum_usage`/`_max_usage`. The full command also includes `--verbose --safe-mode --model <m> --mcp-config '{"mcpServers":{}}' --disable-slash-commands --no-chrome --effort <effort>`]) runs `claude --print --input-format stream-json --output-format stream-json --system-prompt … --json-schema … --tools "" --strict-mcp-config --permission-mode dontAsk --no-session-persistence`. Each run gets its own throwaway `CLAUDE_CONFIG_DIR`.
- Host-side jsonschema validation is in `harness/validation.py:14-41`. An invalid selection raises `AgentProtocolError`, which the runner does not catch, so the run aborts.

### 5.2 System prompt

The system prompt is built by `harness/protocol.py:83-131`. Key passages, verbatim:

> "Return exactly one tool selection per turn: {"name": "...", "arguments": {...}}. No Markdown or text outside this object. The host executes the full motion segment, then supplies a fresh observation."

> "Every motion tool requires a note: 1-2 short **Chinese** sentences, usually 20-60 characters, stating current evidence and the next purpose." (`protocol.py:120`; the paper's App. D renders this language-neutrally)

> "Efficient motion: Prefer move_eef_chunk for clear, contact-free paths that need no intermediate observation… Stop to observe at contact, gripper changes, occlusion or tracking anomalies; efficiency never justifies skipping verification."

> "Release verification: … A fully-open command does not prove that a wide object detached… Do not lift a still-trapped object away and report completion."

> "Persistence: One failed action, tool rejection, missed grasp, occluded target or uncertain result does not establish impossibility… give_up requires evidence that the task cannot be completed or further attempts would violate safety constraints."

Other parts of the prompt:
- **Calibration notes** (`protocol.py:31-62`). Per-arm base frame +x forward / +y left / +z up. TCP = inner-fingertip midpoint on ARX, or `grasp_site` at `[0,0,-0.1347]` m in the flange frame on YAM. Tool +z points to the fingertips and +y is the jaw axis. "xyzw=[1,0,0,0] points tool +z along base -z".
- **Budget line** (`main.py:291-295`). "This task allows at most {max_decisions} decisions… Each tool selection counts, including rejected arguments or IK failures." The default is 100 (`settings.py:30`).
- **Private task profiles are stubbed out.** `harness/task_prompts.py` returns `None`, with `PLUG_CONTROL = ""`. The plug-specific "Additional control guidance" in paper App. D Case 4 is therefore not in the public code (`protocol.py:100-117`).

### 5.3 Tools

The tools are defined in `configs/tools.json` and compiled by `tools/catalog.py`.

- `output_schema` (catalog.py:138-155) is an object `{name: enum, arguments: anyOf[per-tool params]}`.
- For Codex, `_strict_parameters` (catalog.py:260-275) makes every field required and turns optional fields into nullable ones.
- In bimanual mode, a target is `{left: pose|null, right: pose|null}`, where `null` means "hold this arm".

| Tool | What it does |
|---|---|
| `move_to` | One absolute TCP target |
| `move_eef_chunk` | Ordered waypoints ("no schema-imposed waypoint limit") |
| `set_gripper` | Normalized gripper opening: 0 = closed, 1 = open |
| `check_path` | Dry-run IK/timing check |
| `locate_point` | Pixel → distortion-corrected ray. ~~Triangulates from two wrist views~~ [corrected: it triangulates the SAME wrist camera at two decision steps (motion parallax): current `pixel_xy` plus optional `reference_step` and `reference_pixel_xy` from the stored observation history. It does not use left and right wrist simultaneously (`vision/perception.py:64-113`). The top camera returns only base-frame rays for both arms and is never metric. Triangulation is also rejected if the intersection lies behind the camera.] if parallax ≥5°, residual ≤1 cm and condition ≤500 (`vision/perception.py:24-31`). |
| `done(summary, hindsight)` | Terminal; the model declares completion |
| `give_up(reason, hindsight)` | Terminal; the model abandons the task |

**There is no perception model.** No SAM, DINO, depth or pose estimator is used. The RealSense D405s are used as **RGB only**, and the VLM picks pixels itself.

### 5.4 Observation, feedback and context management

**Observation** (`protocol.py:134-160`). A compact JSON with `instruction`, camera metadata, per-arm state and `extra.env_step`, plus `previous_result`.
- Per-arm state: `joint_pos`, `joint_vel`, `joint_torque`, `tcp_pose_xyzrpy/xyzquat`, gripper measured/normalized/commanded, and torque.
- Floats are rounded to 1e-6 and temperature telemetry is stripped.
- Each turn then attaches three images: `"Camera image: left|right|top"`, at 640×480, JPEG q85.

**Feedback** (`runtime/runner.py:146-166`, `234-288`) has three cases:
- **IK failure** → `{"error":"motion_not_executed", requested_motion, segment, sample, residuals}`.
- **Bad arguments** → `tool_rejected: …`.
- **Success** → compacted `execution_feedback`:
  - target vs measured TCP, translation/rotation error, per-joint residual, settling report;
  - the planned TCP points;
  - per-segment duration and FK endpoint error.

**Context window (Codex).** Implemented in `harness/providers/codex.py:37-62`.
- After 8 image-bearing turns, the host unsubscribes the thread and opens a new one. [corrected/precision: this holds only for the first refresh. After a refresh `_live_groups` restarts at 1 for the one retained image group (`providers/codex.py:60`), so later refreshes happen after every 7 new image turns. A refresh is also forced after any `AgentOverloadedError` (`providers/codex.py:37-38, 67-71`).]
- It then replays all observation text and decisions, the first-turn demonstration content, and only the latest previous image group. The replay text:

> "HISTORICAL EXECUTION RECORD. Older live images were omitted; all observation text, host feedback and model decisions follow in order. Demonstration images remain. Historical decisions are reference data, never pending commands…"

- The Claude CLI path has no such window; it relies on the CLI.
- The paper (App. B) describes a separate "ARX Claude Messages adapter" that:
  - pins the initial input, limits live images and summarizes older exchanges;
  - adds TCP geometry and motion-rejection rules;
  - **reviews completion in a separate model session**.

  **None of this is in the public repo.**

**Model retries** (`runner.py:24-28, 93-115`). Up to 20 overload retries, with backoff capped at 8 s and a 300 s recovery deadline. The Codex thread is rebuilt without any partial output, and no robot action is replayed.

### 5.5 Demonstration encoding (the "context compiler")

**Video sampling** (`input/video.py:16-28`, defaults):
- `target_fps=2.0`, `max_candidates=24` per `window_s=30`, `max_keyframes=8` per window;
- candidates at 768 px, keyframes at 1280 px.
- In video+action mode, candidates are added at ±0.3 s around gripper transitions and moving→still events (`input/action_sampling.py:43-66`).

**Keyframe selection** (`harness/video_selector.py:52-116`). Two passes, both with a fresh Codex thread on the configured model, i.e. **gpt-6-astra itself**:
- a per-window `select_video_frames` call, which returns index, reason, stage, left/right role and result;
- a global `review` call that must keep the first and last frames.

~~Both prompts are quoted verbatim in the paper as P0a/P0b~~ [corrected: the paper's P0a/P0b are lightly edited and reordered versions, not verbatim. For example, code says "For the user's final task" and "Use short English stage names", while the paper has "For the current task" and "Use short stage names", with sections regrouped under headings. App. D says instructions are "grouped by function, with implementation-specific identifiers abstracted".] (`video_selector.py:122-135`). Two examples:

> "Similar start/end poses do not imply no motion." / "gripper closure alone does not prove a grasp."

**Demo framing** (`input/demonstration.py:22-56`). The demo is wrapped in `HISTORICAL DEMONSTRATION … END HISTORICAL DEMONSTRATION`, with mode-specific text. For example:

> "Use recorded positions, orientations and gripper events as numeric planning references after checking source robot, base frame, TCP site, quaternion order and current object alignment… Do not stream old absolute joint commands."

**Action rows** (`demonstration.py:84-101` and `174-199`):
- They use a shared column header (`action_sample_encoding.columns`).
- Joint, pose and gripper values are rounded to 3 decimals.
- `"="` means "same as the previous row".
- Compression (`action_sampling.py:69-105`) keeps endpoints, 1 Hz samples, gripper-command changes and channel-dropout boundaries.

**Demo sources.** There are three:
- plain video;
- the harness's **own previous run directory** (`input/recorded_demo.py`). Its demonstrator is `"policy_rollout_excerpt"`, so a successful rollout can become the next run's demo.
- YAM teleop MCAP episodes in `any_robo_episode_media_v1` format (`input/mcap_demo.py`), whose quaternions are converted from wxyz to xyzw.

### 5.6 Motion stack and safety

**Planner** (`motion/planner.py:37-131`). It reads joints, runs FK to get the TCP, samples each segment at 5 mm / 0.035 rad, and solves seeded IK at every sample.
- ARX IK is `multi_trial_ik` plus 12 iterations of damped-least-squares refinement (~~`motion/ik.py:268-379`~~ [corrected: `motion/ik.py` has only 153 lines. `solve` is at `ik.py:42-70` and calls `multi_trial_ik(target, seed, 5)`, i.e. 5 trials. The numeric-Jacobian DLS refinement `_refine` is at `ik.py:90-153`, using orientation weight 0.1 m, adaptive damping and a 0.05 rad step clip. The default of 12 iterations comes from `hardware/robot.py:169-170`]).
- YAM IK is I2RT/Mink (`hardware/yam_ik.py`).
- A residual over 2 mm / 0.01745 rad raises `TrajectoryIKError` before anything is submitted.

**Timing** (`motion/trajectory.py:121-240`). Ruckig on the scalar path coordinate, then time-stretching by max(1, r_v, √r_a, ∛r_j) × 1.001. Geometry is never altered.

**Limits.**

| Limit | ARX | YAM |
|---|---|---|
| TCP speed | 0.08 m/s | 0.08 m/s |
| TCP angular speed | 0.5 rad/s | 0.5 rad/s |
| Joint velocity | 0.25 × SDK max | 0.6 rad/s |
| Joint acceleration | 2 rad/s² | 2 rad/s² |
| Joint jerk | 12 rad/s³ | 12 rad/s³ |

**Bimanual.** Arms are synchronized by slowing the faster one, with a 0.1 s start delay.

**Settling.** 10 consecutive samples within 0.03 rad and ≤0.05 rad/s, with a 3 s timeout (`hardware/robot.py:627-671`).

**Safety.**
- The ARX feedback watchdog (`robot.py:538-566`) catches tracking error >0.12 rad, joint overspeed or overload. It latches a fault and switches the arm to damping mode.
- YAM has a motor-temperature limit of 80 °C.
- Ctrl+C triggers cancel and return-home.
- **There is no workspace box and no collision checking.** The paper reports "We repeatedly observed collisions between the two arms."

**Logging and outcome.**
- Each run writes an append-only directory: `events.jsonl`, 20 Hz states, 10 fps per-camera MP4, and the full prompt and schemas.
- Usage is priced from `harness/prices.json` ("checked_on 2026-09-12", UNVERIFIED against vendor pages):

| Model | Input ($/M) | Cached input ($/M) | Output ($/M) |
|---|---|---|---|
| gpt-6-astra | 10 | 1 | 50 |
| claude-fable-5-1 | 10 | 0.25 | 50 |
| kimi-k3 | 3 | 0.3 | 15 |

- **A human types the success label at the terminal after shutdown** (`main.py:390-403`).

## 6. Results

### 6.1 Main table

Paper Table 1. GPT-6 Astra, 3 trials per condition. Decisions and time are averaged over all trials.

| Task | Context | S/T | Decisions | Time (min) | s/decision [derived] |
|---|---|---|---|---|---|
| Pick Red Towel | None | 0/3 | 96.3 | 24.6 | 15.3 |
| | Human Video | 2/3 | 76.7 | 18.9 | 14.8 |
| Pick Up Notebook | None | 0/3 | 94.0 | 24.6 | 15.7 |
| | Human Video | 2/3 | 66.7 | 16.1 | 14.5 |
| Unscrew Bottle Cap | None | 0/3 | 71.0 | 16.1 | 13.6 |
| | Robot Video | 2/3 | 74.3 | 15.2 | 12.3 |
| | Video + Action | 3/3 | 54.7 | 17.9 | 19.6 |
| Remove & Reinsert Plug | None | 0/3 | 24.0 | 5.3 | 13.2 |
| | Robot Video | 0/3 | 33.7 | 7.9 | 14.1 |
| | Video + Action | 2/3 | 48.3 | 10.8 | 13.4 |
| Arrange T Shape | Target Image | 3/3 | 66.7 | 15.8 | 14.2 |
| Arrange Fruit | Target Image | 3/3 | 49.0 | 12.4 | 15.2 |
| Lemon → Pink Plate | Self History | 3/3 | 35.3 | 8.1 | 13.8 |
| Movable Exploration (mobile) | Self History | 3/3 | 40.33 | 25.53 | 38.0 |
| Tic-Tac-Toe | HRI | 3/3 | 69.7 | 13.6 | 11.7 |
| Pointed Fruit Pickup | HRI | 3/3 | 67.3 | 15.0 | 13.4 |

### 6.2 Model comparison

Paper Table 2: red towel, **single runs**, "task progress" rather than success.

| Model | Context | Progress | Time (min) | Tokens |
|---|---|---|---|---|
| GPT-6 Astra | None | 55% | 24.63 | 12.047M |
| GPT-6 Astra | Human Video | 100% | 15.85 | 4.956M |
| Claude Fable 5.1 | Human Video | 30% | 13.27 | 2.386M |
| Kimi K3 | Human Video | 20% | 11.73 | 1.735M |

The pre-release page also had single-run comparisons:
- Movable exploration: Astra succeeded in 22.5 min, Fable 5.1 failed in 17.5 min.
- Text-only "remove fruit from plate": Astra succeeded in 208.2 s; Kimi K3 was "Interrupted → failed". The page README admits a "known prompt/content mismatch".

### 6.3 Cost and latency [derived]

**Latency.** About 12–20 s per decision. This includes inference, motion at ≤8 cm/s and settling. The mobile task ran at about 38 s per decision.

**Tokens.** The 12.0M-token run works out to ~490k tokens/min, or ~120k tokens per decision if it used ~100 decisions. No per-decision token breakdown is published.

**Cost at prices.json rates.** The cached/uncached split is unknown, so only bounds are possible:

| Run | All input cached ($1/M) | All input uncached ($10/M) |
|---|---|---|
| No-context towel run (12.0M tokens) | ≈$12 | ≈$120 |
| Human-video towel run (4.96M tokens) | ≈$5 | ≈$50 |

Output tokens ($50/M) add to these figures.

### 6.4 Statistics and reporting caveats

**Significance [derived].** One-sided Fisher exact tests:

| Comparison | p |
|---|---|
| 0/3 vs 2/3 | 0.20 |
| 0/3 vs 3/3 | 0.05 |
| Pooled human-video, 0/6 vs 4/6 | 0.03 |
| Pooled robot None vs Video+Action, 0/6 vs 5/6 | 0.008 |

The Pith machine review (deepseek-v4-flash, not peer review) makes the same point: "the quantitative claims are not backed by the paper's own three-trial experiments." It also flags:
- no "None" baseline for the 3/3 families;
- **unblinded** human labels.

**A dropped null result.** The pre-release page (`cheng-haha/in-context-robot-learning`, 2026-09-14/15) listed **Remove Glue Cap** in the main table:

| Glue cap condition | S/T | Decisions | Time |
|---|---|---|---|
| Without human video | 3/3 | 49.3 | 12.2 min |
| With human video | 3/3 | 65.7 | 16.9 min |

So the demonstration made that task *slower*. The final paper keeps it only in App. Table 4 as "an additional reference task beyond the main comparison". Page commit `599e5f9` reads "Replace glue-stick demo with notebook pickup in the task gallery". T-shape numbers also changed between versions, from 59.3 decisions / 13.3 min to 66.7 / 15.8.

**Which robot ran which task?** The paper does not say.
- The repo supports bimanual ARX X5 and YAM, with top and two wrist cameras.
- Morphi Kino (head, chest and two wrist cameras, mobile base, private adapter) must have run Movable Exploration.
- The project page labels all videos "Head view", but that may be a stale default (UNVERIFIED).

## 7. Unmerged RoboDojo work and the GPT-as-Policy link

Branch `fix/xingwu-robodojo-evaluation` adds ~~7,864 lines~~ [corrected: 7,863 insertions and 12 deletions, per `git diff --shortstat main...origin/fix/xingwu-robodojo-evaluation`] across 57 files, 2026-09-21 to 2026-09-23. Several commits are authored by "Codex". It adds:
- `src/gpt_policy/robodojo/` (an XPolicyLab websocket policy and a DLS controller);
- `configs/robodojo_panel50.json`, whose `source_repository` is `anonymous-report-421/GPT-as-Policy` @ `8f3d362`, file `robodojo_panel50_scope_v2.json`;
- a `--control-mode dls` described as ~~"the same URDF numeric-Jacobian DLS as GPT-as-Policy"~~ [corrected: that wording is not verbatim; it translates the Chinese audit doc (`docs/robodojo-execution-audit.md:17`: "使用与 GPT-as-Policy 相同的 URDF 数值雅可比及阻尼最小二乘方法"). The English README calls it "GPT-as-Policy-style bounded DLS control" and "a matched controller method but not an identical policy action schedule", because GPT-Policy fixes up to 5 env steps per motion target while GPT-as-Policy Direct lets the model choose 1-5].

The audit doc (Chinese) lists its target as GPT-as-Policy's `robodojo_panel50_scope_v2`, 10 tasks × 5 cases.

**Two protocol differences make future numbers incomparable with GPT-as-Policy.**
1. A rejected `done` returns the simulator's failed reward checks to the model: "use previous_result.completion_feedback.checks to target only the failed conditions" (`robodojo/model.py:194-199, 246-269`).
2. Task-specific prompts state the success thresholds for push_T and fold_clothes.

`robodojo_plan.md` lists informal "pure gpt success rate" figures of unclear provenance (UNVERIFIED): organize table 30%, arrange largest number 57%, build tower 12%, fold clothes 40%.

**Citation direction.** GPT-as-Policy's report (Galbot group) cites "GPT-Policy-Eval … physical-robot execution from a single video demonstration, including insertion". The GPT-Policy paper cites neither GPT-as-Policy nor arXiv 2609.38537.

**Other links.**
- The README thanks RoboCurve's `inspect-robots` (the Robocurve GPT-6 Astra report harness), but I found no shared code.
- Awesome-Astra-Embodied-AI lists it as "Case 15: One-shot Plug Insertion".

## 8. Assessment

### Strengths

1. **The evaluation design is the most systematic among the user's sources.** It separates the *kind* of context (procedure, motion reference, goal state, memory, live cues) in one fixed harness. Video vs Video+Action holds the images fixed and adds only numbers, which is a clean ablation that isolates the value of numeric action references for contact tasks.
2. **The demo representation is careful.**
   - Keyframes are VLM-curated, cover multiple views, and are timestamp-aligned (≤0.1 s, explicitly "not hardware-synchronized").
   - Action rows are compressed sparsely but keep gripper events.
   - Commanded and measured values are separated, and missing means "unknown, never zero".
3. **The prompts are epistemically disciplined.** Requested ≠ submitted ≠ measured ≠ succeeded is repeated throughout the prompts and the feedback. This matters for LLM controllers, which tend to hallucinate success.
4. **The executor is clean.** It preserves model geometry and only changes timing, rejects on IK residual with precise diagnostics, offers a `check_path` dry run, and does synchronized bimanual retiming.
5. **Engineering hygiene is good.** Demo-input hashes, atomic preparation, prompt-keyed demo caches and a cost ledger.

### Weaknesses

1. **The evidence is thin.** n=3, human labels, no blinding, no baselines for 6 of 10 tasks, single-run model comparisons, and a null result moved out of the main table.
2. **It is extremely slow and costly.** 5–25 min and 24–96 decisions to pick up a towel. Every motion is one blocking LLM round-trip, and motion is capped at 8 cm/s.
3. **There is no geometric perception.** The VLM guesses metric TCP poses from 640×480 RGB plus its own pixel triangulation, while the D405 depth goes unused. There is no collision or workspace guard, and inter-arm collisions occurred.
4. **The CLI-as-transport design is brittle.** Codex app-server version pinning (the branch requires ≥0.155.1), no sampling parameters, and unclear control over built-in tools.
5. **Reproducibility is incomplete.**
   - The plug profile, Morphi Kino adapter, Claude Messages adapter and completion reviewer are private.
   - `pip install -e .` as documented **fails**. Hatchling rejects the git direct reference in the `yam` extra without `allow-direct-references`.
   - Tests pass 59/61 from source; two config tests fail on path assumptions [verified locally].
   - Error messages and the required motion-note language are Chinese.
   - There is no licence, so **the code cannot legally be copied**.

### What is novel and what is repackaged

**Repackaged.** The Codex/Claude-as-controller with a Cartesian tool API is the common 2026 "Astra harness" pattern. Text-encoded trajectories in context echo RoboPrompt and ICRT. VLM keyframe selection is standard.

**Genuinely new.**
- The five-family ICL taxonomy, evaluated on real hardware.
- The Video + Action context format: aligned keyframe state plus compressed commanded segments, with explicit frame/TCP/quaternion caveats.
- The empirical observation that numeric action references, not video alone, unlocked the ~~bimanual~~ [corrected: the bottle task is bimanual (left arm holds the body, right arm takes the cap). The plug task is not established as bimanual: paper App. D Case 4 says "Use one arm when sufficient, or both when support is needed".] bottle and plug tasks (0/3 → 3/3 and 0/3 → 2/3, versus 2/3 and 0/3 with video).

### Maturity

A research preview: v0.1.0, three weeks old, 23 commits on main, and an active RoboDojo WIP branch. The README is explicit: "IK acceptance and a model completion message do not establish collision-free motion or physical task success."

### For Ilia's Opus-backbone harness

**Borrow (as ideas, re-implemented):**
- **Absolute-pose tools with host-side execution.** Densify, per-sample IK with residual gate, Ruckig retime, preserve geometry. Return structured rejection feedback (segment, sample, residual). Expose `check_path` as a free dry-run tool.
- **The demo compiler.**
  - LLM-selected keyframes per window, then a global review; all views bound to an `image_id`.
  - A JSON header with frame conventions.
  - Run-length-coded action rows (endpoints, ~1 Hz, gripper edges).
  - Put this static block in the **prompt-cached prefix** on the Anthropic API, since it is the same on every turn.
- **Feedback schema.** Requested vs measured TCP error, settling, gripper command vs measured opening, and the planned path. Round to 1e-6 and strip telemetry the model does not need.
- **History hygiene.** A sliding live-image window, full text replay, and "historical decisions are not commands" framing. Re-use your own successful rollouts as demos.
- **Evaluation design.** The None / Video / Video+Action ladder is a good template for deciding whether a learned action head is needed. Run it with ≥10–20 trials, blinded labels, and every task reported.

**Avoid:**
- Driving Opus through a CLI wrapper. Use the Messages API directly with tool use or structured output, and explicit thinking/effort and token limits.
- One-LLM-call-per-motion as the only control rate. Add a fast inner loop (learned action head, VLA or servoing) and let the LLM choose subgoals and verify them. The paper itself concludes "Specialized VLA/WAM models may therefore have an edge in fast, low-level control".
- RGB-only metric guessing. Expose depth, segmentation and pose tools.
- Relying on prompt text for safety. Add an independent bimanual collision and workspace monitor that can veto commands.
- Feeding evaluator-privileged success checks back to the model in simulation, as the RoboDojo branch does. It inflates benchmark numbers.

## Sources

- Repo: https://github.com/cheng-haha/GPT-Policy (main @ ab970d8; branches `page`, `fix/xingwu-robodojo-evaluation`, `chore/robodojo-integration`)
- Paper: https://arxiv.org/abs/2609.19138 · PDF in repo `docs/GPT-Policy.pdf` · https://arxiv.org/html/2609.19138v1
- Project page: https://cheng-haha.github.io/GPT-Policy/
- Pre-release page repo: https://github.com/cheng-haha/in-context-robot-learning
- X thread: https://x.com/z_code68632/status/2098397364725895387 (read via api.fxtwitter.com)
- Issues: https://github.com/cheng-haha/GPT-Policy/issues/1 , /issues/2
- Fork with Copilot-backend design docs: https://github.com/Codegass/Copilot-Policy
- Pith machine review: https://pith.science/paper/2609.19138
- HF paper page: https://huggingface.co/papers/2609.19138 (fetch returned arXiv HTML only)
- GPT-as-Policy: https://github.com/anonymous-report-421/GPT-as-Policy (`hybrid_rollout/report_site/app/src/content/report/article-copy.en.json` cites GPT-Policy-Eval); teammate note `research/sources/gpt-as-policy.md`
- Awesome-Astra-Embodied-AI: https://github.com/zjwzcx/Awesome-Astra-Embodied-AI (Case 15)
- Show-Harness (cited competitor): https://arxiv.org/abs/2609.10522
- Morphi Robot / MORPHI KINO coverage: https://chinaminutes.com/2026/08/20/household-robots-shine-at-world-robot-conference , https://weibo.com/2/detail/5334389909357507

## Verification (fact-check pass)

*Adversarial re-check, 2026-10-01/02, against primary sources.*

**Sources re-checked**
- The repo `research/repos/GPT-Policy-cheng`: `main` @ `ab970d8` after `git fetch` (no new commits), plus branches `page`, `fix/xingwu-robodojo-evaluation` and `chore/robodojo-integration`.
- Repo PDF `docs/GPT-Policy.pdf` (32 pp.). I also downloaded arXiv `2609.19138v1`; its whitespace-normalized text is identical apart from the arXiv stamp, one figure caption and glyph encoding.
- arXiv abs page.
- `gh api` metadata for the repo, issues, forks and the fork `Codegass/Copilot-Policy`.
- Pre-release page repo `research/repos/in-context-robot-learning`, including git history.
- X thread via `api.fxtwitter.com/2/conversation/2098397364725895387`.
- Pith page, chinaminutes and Weibo for Morphi.
- GPT-as-Policy @ `8f3d362`.
- Awesome-Astra README.
- Vendor price pages for GPT-6 Astra and Kimi K3; Anthropic pricing via the bundled claude-api reference.
- Local install and test runs in a copy at `/tmp/gptp`. The research workspace was not modified.

### Confirmed claims (seen in a primary source)

**Repository and provenance**
- **Repo metadata** (`gh api repos/cheng-haha/GPT-Policy`):
  - `created_at 2026-09-10T23:34:52Z`; 291 stars; 5 forks; licence `NOASSERTION`.
  - 23 commits on `main`; `5840568` "Initial public release" dated 2026-09-16 03:14 +0800; `pyproject` version `0.1.0`.
  - Original name `GPT-Policy-Eval`, confirmed three ways: the author's 2026-09-11 X reply "GitHub: https://github.com/cheng-haha/GPT-Policy-Eval", the fork `keejkrej/GPT-Policy-Eval`, and Awesome-Astra Case 15.
  - Rename on 2026-09-15: commits `e3f69fa` and page commit `8bb9eb3`, both 2026-09-15 15:39 -0700.
  - 2 issues (#1, #2), both now closed.
- **arXiv 2609.19138**: v1 only, Wed 16 Sep 2026 17:58:35 UTC, submitted by Taoran Yi, cs.CV and cs.RO.
- **Authors, co-first marks, the single † (Tong Wu) and affiliations 1-10**: as in the note (paper p.1).
- **X post**:
  - 2026-09-11 13:04 UTC, 31,113 views and 146 likes at fetch time; quoted text matches.
  - @YuXiang_IRVL asked on 2026-09-12. The author replied on 2026-09-18: "In practice, feeding in the entire video introduces a lot of redundancy, so we use keyframes generated by GPT itself as the context."
- **LICENSE text and README "redistribution and commercial use are not granted"**: verbatim.
- **Morphi**:
  - chinaminutes: "Shanghai Moqi Wanxiang Intelligent Technology Co., Ltd. … showcasing its MORPHI KINO wheeled robot" at WRC 2026, Beijing, opened Aug 19.
  - Weibo: "通用具身智能机器人公司墨奇智能，首次公开亮相了轮式机器人 MORPHI KINO".
  - Linking the paper's "Morphi Robot" affiliation to this company is a reasonable inference; it is not stated in the paper.

**Agent harness and transport**
- **Agent profiles** (`configs/agents/*.json`): byte-exact. `gpt-6-astra` / `effort medium` / `live_image_window 8` / `gpt-5.6-luna` at `low`; `claude-fable-5-1[1m]` with `timeout_s 180`; `kimi-k3[1m]` with `~/.claude-volcengine/settings.json`.
- **Codex** `thread/start` (`codex.py:74-85`), `turn/start` (`127-135`) and reject-unexpected-request (`288-296`): exact.
- **Validation and retries**:
  - `validation.py:14-41`: confirmed. `AgentProtocolError` is not caught in `run_loop`.
  - Retry constants (`runner.py:24-28`): delays `min(2·2^k, 8)` for 20 attempts, plus 0-0.5 s jitter, and a 300 s recovery deadline.
- **No sampling or max-token settings**: `git grep` finds no `temperature` or `max_tokens` in `src`.

**Prompt, tools and feedback**
- **System prompt** `protocol.py:83-131`: all four quotes verbatim (lines 107, 120, 122, 128, 130). Calibration notes are at `protocol.py:31-62`.
- **Budget line**: `main.py:291-295`; default 100 at `settings.py:30`.
- **Plug stubs**: `task_prompts.py:8-15` in the original file (`PLUG_PROFILE = None`, `PLUG_CONTROL = ""`, `control_prompt_profile` returns `None`).
- **Tool catalog**: tool list, schemas, "no schema-imposed waypoint limit", and `_strict_parameters` at `catalog.py:260-275` / `output_schema` at `catalog.py:138-155`. Thresholds `perception.py:24-31` = 5°, 500, 0.01 m.
- **Observation and feedback**:
  - Observation payload `protocol.py:134-226`: 1e-6 rounding and temperature stripping.
  - Feedback branches `runner.py:146-166`; compaction `runner.py:234-288`.
  - Codex replay text `providers/codex.py:42-47`: verbatim.

**Demonstration pipeline**
- **Video defaults** (`video.py:16-28`): 2 fps, 24 candidates per 30 s window, 8 keyframes per window, 768 / 1280 px.
- **Events and compression**: ±0.3 s event candidates at `video.py:207-210` (deltas `(-.3, 0, .3)`); `action_sampling.py:43-66` and `69-105`.
- **Selector and demo text**:
  - Selector at `video_selector.py:52-116`. The two quoted phrases are verbatim at lines 132-133.
  - HISTORICAL / HISTORICAL_ACTION text at `demonstration.py:22-56`, verbatim.
  - Caps of 24 keyframes and 48 unique images at `demonstration.py:166, 238`.
  - 3-decimal motion columns and `"="` repeats at `demonstration.py:84-101`.
  - `policy_rollout_excerpt` at `recorded_demo.py:86`; wxyz→xyzw at `mcap_demo.py:175`; schema `any_robo_episode_media_v1` at `mcap_demo.py:33`.

**Motion stack and safety**
- **Planner and timing**:
  - Planner `planner.py:37-131`.
  - Ruckig scalar retime with `max(1, r_v, √r_a, ∛r_j)·1.001`: `trajectory.py:121-252` and paper Eq. 6.
  - IK execution tolerances 0.002 m / 0.0174532925 rad (`robot.py:180-186`, `yam_ik.py:40-41`).
- **Limits**:
  - ARX: 0.08 m/s, 0.5 rad/s, `joint_vel_max × 0.25`, 2 rad/s², 12 rad/s³ (`robot.py:124-149`).
  - YAM: 0.6 rad/s (`yam.py:84-86`).
  - Bimanual start delay 0.1 s (`bimanual.py:41-43`); paper Table 3 matches.
- **Settling and watchdog**:
  - Settling `robot.py:627-671`: 10 samples, 0.03 rad, 0.05 rad/s, 3 s.
  - Watchdog `robot.py:538-566`: tracking error >0.12 rad, SDK `joint_vel_max`, SDK `joint_torque_max`. A fault goes to `set_to_damping` (`robot.py:528-532`).
  - YAM motor limit 80 °C (`yam.py:66`).
- **No collision checking or workspace box on `main`**: `coordination.py:14` has `"collision_checked": False`, and `git grep workspace` finds only prompt text. Paper quote "We repeatedly observed collisions between the two arms during manipulation" is on p.13.
- **Human outcome label** at `main.py:390-403`.

**Prices** (`harness/prices.json`, `checked_on 2026-09-12`): now verified against vendor sources, so the note's "UNVERIFIED" can be dropped.
- **gpt-6-astra**: $10 input, $1 cached, $12.5 cache write, $50 output. Long-context rule from developers.openai.com: "Prompts with more than 272K input tokens are priced at 2x input and cache rates and 1.5x output for the full request." Context window 1.05M.
- **claude-fable-5-1**: $10 / $50, cache reads $0.25 (Anthropic pricing via the claude-api reference, cached 2026-06-24).
- **kimi-k3**: $3 / $0.30 / $15 (Moonshot forum post).

**Paper results**
- **Paper Tables 1, 2, 4 and 5**: every number in §6.1, §6.2 and §3 of the note matches the PDF.
- **Derived numbers**: all s/decision values recomputed and correct (e.g. 24.6·60/96.3 = 15.3; 25.53·60/40.33 = 38.0). Fisher one-sided p values also recomputed: 3/15 = 0.20, 1/20 = 0.05, 15/495 = 0.030, 6/792 = 0.0076.
- **Paper's own derived figures**: −35.6% time and −58.9% tokens (Astra None→HV).
- **Pre-release page** (`in-context-robot-learning/index.html:448-458, 508-547, 608-612`):
  - glue cap 3/3 at 49.3 decisions / 12.2 min vs 3/3 at 65.7 / 16.9;
  - T-shape 59.3 / 13.3;
  - Astra 22.5 min vs Fable 17.5 min (movable);
  - Astra 208.2 s vs Kimi "Interrupted → failed";
  - `anonymous2027incontext` / "ICLR submission".
- **Page commit** `599e5f9` message verbatim (branch `page`). Paper Table 4 caption: "Remove Glue Cap is an additional reference task beyond the main comparison."

**Third-party and related repos**
- **Pith**: "reviewed 2026-09-17 · deepseek/deepseek-v4-flash", quote verbatim. It flags missing None baselines for "three of five context families" and "no blinding".
- **RoboDojo branch**:
  - 57 files changed; 3 commits authored "Codex".
  - `configs/robodojo_panel50.json` source repo, commit `8f3d362` and path as stated.
  - `robodojo/model.py:185-200, 246-269`: completion-feedback replay and push_T / fold_clothes thresholds.
  - `robodojo_plan.md` percentages.
  - Codex "0.155.1 or newer in the tested setup" (branch `README.md:306`).
- **GPT-as-Policy** `article-copy.en.json` cites "[GPT-Policy-Eval] demonstrates physical-robot execution from a single video demonstration, including insertion…". The Awesome-Astra list identifies GPT-as-Policy as from Galbot (银河通用). The GPT-Policy reference list contains no GPT-as-Policy entry.

**Build and tests**
- **Install failure reproduced** (`uv pip install -e .` in a Python 3.11 venv): hatchling `ValueError: Dependency #1 of option 'yam' … cannot be a direct reference unless field 'tool.hatch.metadata.allow-direct-references' is set to 'true'`.
- **Tests reproduced**: 59 passed, 2 failed. The failures are `test_config.py::test_repository_main_and_agent_configs_resolve_together` and `::test_repository_native_profiles_resolve_without_legacy_credentials`, which cannot find `.../configs/agents/claude.json` (path resolution).

### Corrections (9, all also marked inline)
1. **Private code vs plug results.** The plug results need the private plug profile and the unpublished demo-format loader. The Morphi Kino adapter belongs to Movable Exploration, and the Claude Messages adapter to the Fable comparison.
2. **Claude CLI line reference.** `session.py:137-147` → `session.py:48-58`. The command also includes `--verbose --safe-mode --model --mcp-config '{"mcpServers":{}}' --disable-slash-commands --no-chrome --effort`.
3. **`locate_point` triangulation.** It is not "two wrist views" at once. It intersects rays from the same wrist camera at the current step and a stored `reference_step` (`perception.py:64-113`). The top camera gives rays only.
4. **Codex image-window refresh.** Only the first refresh comes after 8 image turns; later ones come after every 7 new image turns. A refresh is also forced after an overload error.
5. **P0a/P0b.** They are edited and reordered in the paper, not verbatim copies of `video_selector.py:122-135`.
6. **ARX IK line reference.** `motion/ik.py:268-379` does not exist (153-line file). Use `ik.py:42-70` (multi_trial_ik, 5 trials) and `90-153` (DLS); 12 iterations come from `robot.py:169-170`.
7. **RoboDojo diffstat.** "7,864 lines" → 7,863 insertions and 12 deletions.
8. **DLS "quote".** It translates the Chinese audit doc; the README wording differs (see inline).
9. **"Bimanual … plug".** Only the bottle task is shown bimanual; the plug prompt allows one arm.

### Unverifiable / unverified claims
- **Effort level in the paper runs.** That `effort: "medium"` and `live_image_window 8` were used is UNVERIFIED. These are public defaults, and Table 3 says "These are source-configuration defaults; individual runs may override them."
- **What the Table 2 token counts include.** UNVERIFIED (see missed detail 3 below).
- **Which wording the towel None trials used.** UNVERIFIED (see missed detail 7).
- **Plug guidance per condition.** Whether the private plug profile was active in the plug None and Video conditions is UNVERIFIED. It is keyed on the instruction text (`protocol.py:100`), and the instruction differs by condition.
- **Morphi Kino ran Movable Exploration.** Inference only. It is consistent with page text: "the robot lowers and leans forward, then scans unexplored sectors by rotating in place".
- **"Head view" labels.** Still UNVERIFIED. ARX and YAM have no head camera per Table 3, and the pre-release README says "All comparison videos use the head camera".
- **`robodojo_plan.md` percentages.** Provenance unknown; the file gives no trial counts.
- **HF paper page.** Not re-fetched.
- **"Galbot group" for GPT-as-Policy.** Supported only by the Awesome-Astra list and a `galbot-wordmark-source.md` asset, not by an author list.

### Missed details (added)

**1. Claude Code transport specifics** (`claude_code/session.py:22-126`, `config.py:13-22, 132-152`)
- The path strips every `ANTHROPIC_*` variable and `CLAUDE_CODE_USE_BEDROCK/VERTEX/FOUNDRY/OAUTH_TOKEN/SUBAGENT_MODEL` from the environment. It then injects only connection keys (`ANTHROPIC_BASE_URL`, exactly one of `ANTHROPIC_AUTH_TOKEN`/`ANTHROPIC_API_KEY`, proxies) from the settings file.
- One persistent CLI process carries the whole run, with no live-image window, so every image accumulates.
- Any `control_request`, or any `tool_use` other than `StructuredOutput`, raises `AgentProtocolError`.
- Any failed turn closes the process (`session.py:123-126`) and is not retried. Only Codex `serverOverloaded` is retryable.
- There is also a `--set-claude-key` path that stores a gateway key at `~/.config/gpt-policy/claude.json` (mode 0600).

**2. An unadvertised Gemini CLI provider exists.**
- `harness/providers/gemini_cli/` (ACP: `gemini --acp --extensions none --admin-policy …`) is registered in `factory.py:10-14`.
- It is unreachable from config, because `AGENT_NAMES = ("codex","claude","kimi")` (`config.py:13`).

**3. Cross-model confound in Table 2.** Keyframe selection and task naming are hard-wired to the Codex profile, i.e. gpt-6-astra, even when the controlling agent is Claude or Kimi.
- Code: `main.py:182-185, 420-422`; `CodexVideoSelector` raises unless `type == "codex"` (`video_selector.py:33-34`).
- Consequence: the "8-frame human video" context given to Fable 5.1 and Kimi K3 was curated and annotated by GPT-6 Astra.
- The usage ledger's scope is "model turns in this invocation, including naming and demonstration preparation" (`usage.py:139`). Its `total_tokens` counts cached re-reads as input (`usage.py:56-61`).
- So per-run "Tokens (M)" are not unique tokens and may include Astra calls.

**4. Schema and validation gaps.**
- The Codex `outputSchema` does not bind arguments to the chosen tool name: it is `name` enum plus `arguments: anyOf[...]` (`catalog.py:138-155`).
- Host jsonschema validation (`parse_decision`) runs only on the Claude and Gemini paths. On Codex, mismatched arguments surface as a planner `ValueError` → `tool_rejected`.
- Every selection consumes one of the 100 decisions, including `check_path`, `locate_point` and rejected calls (`main.py:291-295`). So the note's "free dry-run" recommendation does not describe GPT-Policy itself.
- An empty motion `note` is rejected (`planner.py:45-46`).

**5. The plug profile replaces the default guidance wholesale.**
- `if plug_profile: return common + PLUG_CONTROL` (`protocol.py:116-117`) drops the whole default tail: Orientation retention, Chinese-note rule, Efficient motion, Grasp sequence, Release verification and Persistence.
- The `done` prompt is overridden with `PLUG_DONE` (`protocol.py:100-101`).
- Paper App. D Case 4 prints that guidance (arm choice, inter-arm clearance, tool +z along the table's downward normal, seating check).
- The public `done` tool prompt already contains "For insertion tasks: confirm actual mating and seating, not merely resting on the socket" (`configs/tools.json`).
- There is also an optional `scene.safety_notes` → "Fixed scene and hidden obstacles" block (`protocol.py:12-28`).

**6. The paper's Video+Action format is not produced by public code.**
- The F3a/F3b fields do not appear in any branch (`git grep`): `control_sample_index`, `command_from_top_capture_s`, `state_from_top_capture_s`, flat `left_eef_x_m…`, `next_keyframe`.
- Public loaders differ:
  - `RecordedDemo` (own run directories) emits `kind: "measured_bimanual_segment"` whose rows are measured joints/TCP plus `gripper_command`. These are measured, not commanded, geometry (`recorded_demo.py:91-118, 159`).
  - `McapDemo` (YAM teleop) emits `"recorded_bimanual_segment"` with commanded `joint_target_rad`/`eef_target_xyz_xyzw` (`mcap_demo.py:141-176`).
- Teleop demos are described in App. C.1: "measured joint and end-effector states, and motion and gripper commands".

**7. Eval-protocol details the note missed** (paper §4.1, §4.6, App. D.4)
- **Success and decision counting**:
  - Success means "the final scene satisfies the task-specific geometric and semantic success criteria".
  - Decisions use "the task-specific counting convention, each generated action target or action block counts as one decision".
  - Episodes end at success, budget exhaustion or a safety termination; in code, the end is the model's `done`/`give_up` followed by a human label.
- **Tic-Tac-Toe**: "both wins and draws are counted as successful". The robot plays first as Green, and the prompt dictates move priorities: "Prioritize an immediate win, then block an immediate opponent win, then choose a move that creates a double threat or at least preserves a draw."
- **Pointed Fruit**: runs until the human makes an OK gesture.
- **Movable Exploration**: its instruction begins "Use the movable exploration history as context. Continue searching for the Sprite bottle…", so a prior history is injected.
- **Towel instruction variants.** The Human Video instruction is "Watch the historical human demonstration frames and imitate the demonstrated grasping method to pick up the red towel…". App. D.4 lists both a "prepared goal-only comparison" ("Pick up the red towel with the robot's right hand.") and a "recorded procedural variant without video" ("Block the towel with one hand, slide the other hand underneath it, and then grip the towel to lift it."). §4.2 says None "uses the same instruction", but the wording that was actually recorded for the 0/3 None trials is ambiguous.
- **Bottle wording**: "The recorded task wording mentions numerical references in both conditions" (Video and V+A).
- **The X post's "imitate a human dance" and "clear obstacles"** appear only as qualitative demos (Fig. 1), with no trial counts.

**8. Selection of showcased single runs** (pre-release page history)
- Commit `264c53e` "Keep one representative Claude towel run" removed a second Fable 5.1 towel run ("fixed": 3 decisions, 76.3 s). It kept the "paper" run: 35 decisions, 795.5 s = Table 2's 13.27 min.
- Commit `57a7203` "Keep one matched Kimi comparison" removed Kimi runs: "Run 1 · mismatch", 801.7 s, "Logged prompt mismatch", plus "Place fruit" and "Raise both arms" tasks. The README's "known prompt/content mismatch" sentence is stale and refers to the removed run.
- The Astra human-video towel run is 950.7 s (= Table 2's 15.85 min).
- Lemon decisions changed from 35.0 (pre-release) to 35.3 (paper) as well as the T-shape numbers.

**9. Demo-cache and review limits** (`video_cache.py:108-114, 142-164`)
- The global-review limit is `min(24, 48 // views)`, so at most 16 keyframes for 3-view demos (the plug demo used 14 × 3 = 42 images).
- The first and last candidates are forced into the window selection.
- The review prompt asks "Usually retain 12-16 frames".
- Selections are cached, so repeated trials reuse identical keyframes.

**10. Configuration values from paper Table 3 / App. A**
- **Common**: numerical IK stopping 1e-4 m and 5e-4 rad; final reference hold 0.12 s.
- **YAM settling** differs from ARX: ≥0.3 s and 10 samples, encoder span ≤0.002 rad, span/time ≤0.05 rad/s.
- **ARX**: gains `kp=[150,150,150,50,10,40]`, `kd=1` (`robot.py:105-109`). Gravity compensation is on because "Keeping it off caused a repeatable 17--28 mm downward TCP error" (`robot.py:97-99`).
- **Morphi Kino**:
  - hardware: 2×7-DoF arms; head, chest and 2 wrist RGB views at 1280×720;
  - trajectory and IK: 10 Hz reference; IK residual <0.003 m / <0.02 rad; inter-sample joint step <0.15 rad; joint velocity ≤ min(0.2, URDF) rad/s;
  - execution: 1 s hold; settle <0.015 rad; 30 s caps; one arm per call.
- **"The inspected YAM branch omits"** the Claude Messages adapter's extra execution and completion review.

**11. Limitations stated by the authors** (§5 "Scope and Limitations")
- "Incomplete ablations and unobserved pretraining limit causal and novel-skill claims."
- "Context quality, observation–action alignment, execution latency, and human intervention remain constraints."
- "The observed inter-arm collisions show that existing safeguards are insufficient by themselves for safe autonomous deployment."
- Six future directions are listed: safety layer, contact-aware harness, System 1/System 2 with a VLA, active-perception mobile manipulation, compositional context, and in-context dynamics adaptation.

**12. RoboDojo branch extras**
- `configs/agents/codex.json` sets `live_image_window: null`, disabling image-window refresh.
- `done` must be confirmed by the RoboDojo reward check; after 3 unconfirmed requests the episode ends as a policy failure.
- RoboDojo-mode workspace bounds are enforced (`z: [null, 0.45]`), unlike `main`.
- DLS limits: FK check <2 mm / <0.01 rad; target ≤0.05 m / 0.35 rad; per-step ≤0.02 m / 0.1 rad; ≤0.05 rad per joint.
- `robodojo_plan.md` also lists pack 50%, classify 100% and bottles 36%.

**13. Smaller details**
- Issue #1 reply (2026-09-15): "we support Ark AC One and YAM".
- The V4L2 camera path can fall back to a Z16 depth stream rendered as percentile-scaled grayscale (`camera.py:313-324`). It is never used as metric depth; the RealSense backend enables only the color stream (`realsense.py:25`).
- Live JPEG conversion happens only if `runtime.convert_camera_images_to_jpeg` is true. The code default is `False` (`settings.py:155`); the example profile sets `true` / q85.
- One positive reply in the X thread is from co-author Ye Fang (@YeFang0110).
