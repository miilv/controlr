# innate-os PR #817: "Imitate a recorded demonstration with GPT-6 Astra"

**PR:** https://github.com/innate-inc/innate-os/pull/817. It is **open and unmerged**, last updated 2026-09-11. Diff: +4162/−9 across 32 files, 7 commits.

**Reading basis.**
- PR head `ef8fd0d`, checked out at `research/repos/innate-os-pr817/`.
- Main at `1ed590a`, at `research/repos/innate-os/`.
- PR body, commit messages, Greptile review threads and author replies.
- Axel Peytavin's X thread, including both videos, which I downloaded and frame-sampled.
- docs.innate.bot.
- Sibling Innate PRs #737, #816 and #843.

Line references below are against `ef8fd0d`. Anything I could not check is marked **UNVERIFIED**.

---

## 1. TL;DR

- **What it does.** A single recorded teleop episode (HDF5, 30 Hz, two 640×480 cameras, joints, commands) becomes the whole context for `gpt-6-astra`. There is no training and no trajectory replay. The skill drives a fixed sequence of forced-tool calls:
  1. `note_observations` on a uniform survey of the episode;
  2. `record_phases`: the model writes a 2–6 phase map with frame intervals and `advance_when` conditions;
  3. `act`, repeated: one bounded action per observation.
- **Action space.** Each act chooses from `joint_step` (≤0.15 rad), `ee_absolute`/`ee_delta` (≤4 cm, ≤0.2 rad), `base_step` (≤3 cm), gripper open/close, `observe`, `retry`, `done` or `stop`.
- **Guards.** A host runtime re-reads telemetry after the model call, validates against caps, the workspace box, a "vertical descent only" rule, the shoulder joint floor and a grip latch. It then executes through IK or joint streaming and feeds the measured outcome back.
- **Evidence.** One public success: a 6-phase "knock the potted cactus off its platform with a held rod, drop the rod, pick up a small cube, place it on the platform" task.
  - The task was inferred without a text prompt in a different room and layout.
  - The run took ~~about 43–45 decisions at roughly 3.4–8.5 s per model call~~ [corrected: exactly 45 steps (#0–#44). `act` calls took ≈3.2–8.5 s, median ≈4.3 s. The two planning calls took 11.5 s (`note_observations`) and 8.1 s (`record_phases`), so step #0 took 23.2 s. Cumulative model wait was ≈3.7 min. All values read from the trace video's UI.] (read off the trace video).
  - It included one missed grasp, recovered via a gripper-aperture check.
- **Not reported.** Success rates, trial counts and cost.
- **Status.** The same author's later PR #843 (2026-09-18) says it **"replaces the `demonstration-skill` approach (≈4,000 lines, one model)"** with a ~360-line, model-agnostic, *demonstration-free* `do_task` skill. Treat #817 as a well-engineered, n=1 prototype that its authors have already moved past.

---

## 2. Context: Innate, MARS and innate-os

**Company.**
- Innate Inc., Palo Alto, YC F24. CEO Axel Peytavin (ex-Stanford). Per third-party trackers, co-founder/CTO is Vignesh Anand.
- PR author David Dobas (`david@innate.bot`); every commit is co-authored by Axel Peytavin.
- Repo: Apache-2.0, about 81 stars and 41 forks at reading time.

**Robot: MARS.**
- Price: $995 per innate.bot (earlier ~$2k).
- Differential-drive base.
- **5+1-DoF arm** built from Dynamixel XL430/XC430/XL330 servos: 40 cm reach, 2 mm repeatability, 250 g payload at full extension.
- Tilting head with stereo RGB (150° FOV). Wide-angle wrist camera.
- 2D LiDAR.
- Jetson Orin Nano Super 8 GB.
- Source: docs.innate.bot/robots/mars.

**Software stack ("agentic OS").**
- ROS 2 Humble with Zenoh.
- Key packages (`AGENTS.md`): `mars_arm` (servos, IK), `mars_cam`, `mars_nav` (Nav2/SLAM), `brain_client` (agent loop, skills action server), `manipulation` (recorder, ACT runner, LeRobot bridge).

```
            ┌───────────── cloud: LLM call only (Innate proxy or own key) ─────────────┐
            │  brain agent loop: look → think → act (brain_client/brain/agent.py)       │
            │  default model google:gemini-3.6-flash (innate_llm/configure.py:45);      │
            │  catalog also routes claude-*/gpt-* (models.py:30-31)                     │
            └───────────────┬───────────────────────────────────────────────────────────┘
                            │ tools = skills (execute() signature → schema, guidelines() → description)
   ┌────────────────────────┼──────────────────────────────────────────────────────────┐
   │ code skills (workspace/innate_skills/*.py)   policy skills (manipulation_server)   │
   │  e.g. pick_any_object: Gemini 3.5 Flash      ACT @25 Hz, TensorRT, 6 arm + 2 base   │
   │  detection + HSV wrist servo + IK            cmds, progress head auto-stop;         │
   │  imitate_demonstration (THIS PR):            trained in Innate cloud from 30+ demos │
   │  GPT-6 Astra stepwise policy                 replay skills: recorded motion         │
   └────────────────────────┬──────────────────────────────────────────────────────────┘
                            │ Manipulation/Mobility/Head facades (brain_client/robot/*.py)
             ROS 2: /mars/arm/*, /cmd_vel, /odom, /mars/main_camera/*, /mars/arm/image_raw
```

How the pieces relate:
- The main agent is a Gemini-backed "VLM picks a skill" loop. Peytavin likened it on HN to a small π0.5-style VLM→VLA split.
- Manipulation is normally either hand-coded perception-plus-IK skills or ACT policies trained from 30+ leader-arm demos (docs: training/overview).
- PR #817 adds a third kind: a skill that is itself an LLM closed-loop policy conditioned on **one** demo.
  - It does **not** go through the brain's `innate_llm` provider layer. It has its own ~~88-line~~ [corrected: 92-line at `ef8fd0d`. 88 is the figure in the PR-body table, which predates the last commit] Responses-API transport, "meant to be replaced by the brain's full provider support".
  - An agent can call it like any other skill. It is also launchable from the new `/icl` web page.

---

## 3. PR anatomy and timeline

| file (PR) | role | lines |
|---|---|---|
| `ros2_ws/src/brain/brain_client/innate/demonstration.py` | HDF5 loader, FK, grip-event detection, keyframe selection, base dead-reckoning, legacy converter | 325 |
| `.../innate/imitation_actions.py` | action vocabulary, caps, validators, grip latch, prompt-token filler | 216 |
| `.../innate/imitation_policy.py` | planning ladder, tool schemas, Responses request, per-turn context | 306 |
| `.../innate/imitation_prompt.py` | instruction assembly | 238 |
| `.../innate/openai_responses.py` | direct key or Innate proxy, `/v1/responses` | 92 |
| `.../innate/icl_trace.py` | `/brain/icl_trace` live mirror | 124 |
| `workspace/innate_skills/imitate_demonstration/imitate_demonstration.py` | run loop (the Skill) | 444 |
| `.../imitate_demonstration/imitation_runtime.py` | guarded execution, live telemetry | 425 |
| `manipulation/` (C++) | recorder writes `/observations/ee_pose` plus provenance | ~190 |
| `webapp/js/icl/*`, `css/icl.css`, proxy routes | "In Context Learning" operator page | ~1,600 |

**Timeline (UTC).**
- 2026-09-10 20:36: first commit "Add demonstration-conditioned skills and a prompt ablation". This was the two cactus-specific skills `SlashAndPickCactusNoPrompt/WithPrompt`.
- PR #816, the same content, was closed as "opened under the wrong user".
- Over the next ~4 hours:
  - generalised to one skill with a `task` input;
  - renamed (`gesture.py` → `demonstration.py`, etc.);
  - composed the runtime instead of using a 4-deep inheritance chain;
  - added the direct OpenAI route;
  - recorder now writes EE poses;
  - fixed dead-reckoning ~~sign/offset~~ [corrected: integration direction, not a sign error. Each command was applied to the interval *before* its own sample, which shifted the path one sample at every command change. Commit ffa575a: 0.5 m/s × 2 s now integrates to 1.000 m instead of 0.950];
  - URDF fingerprinting by kinematics instead of bytes;
  - clamped streamed j6;
  - tracked and joined the model worker;
  - 00:59 fix: "every run through the proxy failed on its first model call" (`ResponseNotRead`).
- Tweet at 01:06 on 2026-09-11.
- The demo video's UI still shows the skill as **`slash_and_pick_cactus_no_prompt`**. The showcased run therefore came from a pre-generalisation build, not the final PR code.
- Peytavin: "set it all up and running in less than a day".

**Review.**
- Only the Greptile bot reviewed. It raised 6 issues: recorder output not loadable, base path shifted one sample, byte-hash URDF matching, abandoned worker thread, docs describing a nonexistent skill, and unclamped explicit grip.
- All were fixed in follow-up commits. Greptile's final score was 5/5 "safe to merge".
- No human review. `reviewDecision: REVIEW_REQUIRED`.

---

## 4. How demonstration imitation works

### 4.1 Recording contract (C++ recorder + loader)

**Recorder side.**
- The recorder (30 Hz, 640×480 BGR per camera, `manipulation/config/recorder.yaml`) now loads `mars_sim/urdf/mars.urdf` at startup (`recorder_node.cpp:66-71`).
- Each tick it writes FK of the *same measured* `qpos` sample as `/observations/ee_pose` rows `[x,y,z,qx,qy,qz,qw]`, `ee_link` in `base_link` (`recorder_node.cpp:383-387`, `ee_kinematics.hpp`).
- It attaches the URDF, joint order and camera topics as HDF5 attributes (`episode_data.cpp:215-244`).
- Rationale (commit 67c447c): "the two drift apart while the arm moves, and a demonstration is read at exactly the moments it moves".

**Loader side.** The loader (`demonstration.py:125-259`):
- Validates shapes, finiteness, monotone timestamps and the 2..18000 row bound.
- Requires the exact camera mapping `/mars/main_camera/left/image_raw,/mars/arm/image_raw`.
- Older recordings use `legacy_urdf` with FK via PyKDL in memory.
- Robot compatibility is a SHA-256 over joint names, types, origins, axes and limits (`model_fingerprint`, `:97-118`). The skill refuses to run if this differs from the running robot (`imitate_demonstration.py:194-196`).

**Episode size.** Episodes are uncompressed. The showcased one is 1053 rows (35.1 s) and **1941 MB**. Hence "No recording ships".

### 4.2 What a "frame" is when sent to the model

Each selected frame becomes:
1. one `input_text`: `"DEMO " + json.dumps(frame minus images)`;
2. `"head"` followed by the head image;
3. `"wrist"` followed by the wrist image.

Images are base64 JPEG data URLs (`imitation_policy.py:109-123`). The frame dict (`demonstration.py:214-249`) holds:
- `index`, `time_s`;
- `ee_pose` (7 floats), `qpos` (6);
- `gripper_target_rad` (the *commanded* j6 from `/action[:,5]`);
- `head_degrees`;
- if the base moved: `base_command` `[v, ω]` and `base_dead_reckoned` `[x, y, yaw]`, integrated from commands, *not* odometry;
- `camera_observations` per camera: true image timestamp, `row_offset_s`, nearest arm sample index and that sample's pose/qpos.

Other details:
- Demo images are re-encoded at **JPEG Q75 at the native 640×480**, with no downscaling. Live images are the camera's compressed topics, also 640×480.
- Floats go out at full double precision. By contrast, sibling PR #737 rounds telemetry to 4 dp to save tokens.

### 4.3 Keyframes: grip events by midpoint crossing

`grip_events` (`demonstration.py:84-94`) finds indices where measured j6 crosses the midpoint of its own min/max range. The docstring explains why: "at 30 Hz a real half-second grasp moves only ~0.05 rad per step, so a 0.12 test silently never fires".

With `frame_selection="both"`, the default:
- **survey:** 24 uniformly spaced frames (`overview_frames`, 6–48);
- **close look:** each grip event ±8 rows (≈0.27 s), capped at 15 frames (`imitation_actions.py:94-102`).

`keyframes` mode uses events first, then an even fill. `uniform` mode uses even frames only. This is an ablation knob.

### 4.4 Planning ladder: forced tools, stateless calls

`ImitationPolicy.decide` (`imitation_policy.py:246-255`) runs on the first turn only:

| call | frames shown | only legal tool (`tool_choice` forced) | output |
|---|---|---|---|
| 1 | uniform survey (24 frames = 48 images) + live pair | `note_observations{observations:str}` | free-text notes, persisted |
| 2 | survey ∪ grip-event frames (≈33 frames = 66 images for 3 events) + live pair | `record_phases{phases:[{name,start_frame,end_frame,reference_frame,advance_when}]}` (2–6 items) | phase map |
| 3…N | acting window (≤7 frames = 14 images) + live pair | `act{phase, previous_phase_complete, evidence, actions[1..chunk_size]}` | validated decision(s) |

Every request has this shape (`:188-199`):

```python
{"model": "gpt-6-astra", "service_tier": "priority", "store": False,
 "reasoning": {"effort": "low"},
 "instructions": self.instructions + f"\nFor this run chunk_size={self.chunk_size}.",
 "tools": [ONE tool], "tool_choice": {"type": "function", "name": tool_name},
 "parallel_tool_calls": False, "max_output_tokens": 2500,
 "input": [{"role": "user", "content": content}]}   # HTTP timeout 40 s
```

**Each call is stateless.** It is one user message. No previous outputs and no encrypted reasoning are replayed. Cross-turn memory consists only of:
- `notes_so_far`, `phase_map`, `current_phase`;
- `history[-6:]`, sent as JSON entries with decision, measured execution outcome and the full observation dict minus images.

**Why a fixed ladder** (module docstring): "Given a free choice the model spent every call re-reading nearly identical frames and never committed a plan, so the ladder guarantees a turn always ends in a decision."

**Phase-map validation** (`:257-278`):
- 0 ≤ start ≤ ref ≤ end < N;
- phases ordered;
- `reference_frame` must be among frames the model was actually shown;
- name and `advance_when` 1–600 chars.

**Acting window** (`_acting_context`, `:91-107`):
- `linspace(start,end,4)` of the current phase, plus its reference frame and the neighbouring phases' reference frames;
- frames are lazily loaded from the HDF5.
- Rationale: "One instant per phase cannot show how a phase was performed, only that it happened."

**Phase pointer rules** (`_action`, `:280-302`):
- the model may keep the phase or advance by exactly one;
- advancing requires `previous_phase_complete=true` and a non-empty history;
- `done` is only legal in the last phase;
- only `joint_step`s may be batched (`chunk_size` 1–10, default 1).

### 4.5 Action vocabulary and caps

The vocabulary is defined once in `imitation_actions.py:15-72` and interpolated into the prompt via `with_limits` (`:205-216`).

| action | `pose` | cap / semantics |
|---|---|---|
| `joint_step` | `[joint 1-5, Δrad]` | 0 < \|Δ\| ≤ 0.15 rad |
| `ee_absolute` | `[x,y,z,roll,pitch,yaw]` in `base_link` | ≤4 cm and ≤0.2 rad/component from measured |
| `ee_delta` | `[dx,dy,dz,droll,dpitch,dyaw]` | same; rpy composed by addition (approximate for multi-axis) |
| `base_step` | `[m]` | ≤0.03 m straight; executed as 0.04 m/s `cmd_vel` pulses |
| `open_gripper` / `close_gripper` | `[]` | grip latch ordering |
| `observe`, `retry`, `done`, `stop` | `[]` | `retry` resets phase pointer to 0 (`restart_attempt`, `:304-306`) |

- **Motion toggles.** Default: `joint_step` on, `ee_absolute` on, `ee_delta` off, `base_step` on (`imitate_demonstration.py:430-444`).
- **Prompt rebuild.** The prompt is rebuilt so that disabled tools are never described (`imitation_prompt.py:206-238`).
- **Showcased run.** It used `joint_step` + `ee_delta` + `base_step`. The tweet's "it even chooses when to use end-effector or joint space" refers to this mixed vocabulary.

### 4.6 Execution and guards (host side)

**Run loop** (`imitate_demonstration.py:250-416`). For each step, up to `max_steps=110` and `time_budget_s=1200`:

1. **Observe.** `ArmRuntime.observe` waits ≤5 s for a snapshot ~~(`imitation_runtime.py:83-121`)~~ [corrected: the snapshot checks are `LiveObservation.snapshot`, `imitation_runtime.py:83-121`. The 5 s wait is `ArmRuntime.read`, `:197-205`, and the base-drift check is `ArmRuntime.observe`, `:207-216`. The demo-derived priors below are added by the skill (`imitate_demonstration.py:254-264`), not by `ArmRuntime.observe`]. Requirements:
   - head, wrist, joints, odom and health all present;
   - ~~each newer than the last action and ≤1.5 s old~~ [corrected: this applies to head, wrist, joints and odom. The health message only needs to be ≤3 s old (`:88-94`). The ROS header stamps of head, wrist and joints must also be ≤1.5 s behind ROS time];
   - camera/joint stamps within 0.25 s of each other;
   - arm healthy and torque on.

   It computes the live EE pose by FK and fails the run if the base drifted >2 cm or >0.05 rad outside a `base_step`. It then adds demo-derived priors:
   - `shoulder_demo_max` and `shoulder_limit_rad = min(demo max + 0.2, 0.5)`;
   - `joint2_min_rad`;
   - `demo_joint_range` per joint;
   - `gripper_closed_rad`/`gripper_open_rad` from the demo's own grip range;
   - `holding`, `finishing`.
2. **Decide.** `ArmRuntime.decide` runs the model on a worker thread so Stop stays responsive, with a 175 s overall timeout (`:218-246`).
3. **Stale check.** If the EE moved >1 cm during the model call, the action is discarded as `stale_observation` (`:280-295`). Inside a chunk, joint drift >0.03 rad discards the remainder.
4. **Semantic checks** (`:310-316`): the grip latch (`next_holding`: no close while holding, no open while empty, no `done` while holding) and `check_reachable_move` (`imitation_actions.py:165-185`):
   - workspace box x∈[−0.1, 0.45], |y|≤0.35, z∈[0.025, 0.5], r≤0.45;
   - per-step cap;
   - **"A descent must be vertical"**: refuse any target that drops >5 mm while moving >5 mm sideways.
5. **Execute.**
   - `try_move`: `manipulation.ik()` → refuse if IK fails or joint2 < `joint2_floor(j1)`, because the driver would silently clamp it (new `Manipulation.ik`/`joint2_floor`, `manipulation.py:360-380`) → `move_to(duration=1.5)` → reached if measured ≤1.5 cm.
   - `try_joint_step`: `stream_joints(max_speed=0.25 rad/s)` polled at 40 ms until error ≤ min(0.02, \|Δ\|/3) or 4 s. Sag on held joints >0.02 rad is reported separately.
   - `try_base_step`: odometry-closed, with lateral/yaw wander aborts and a 5 s timeout.
   - The EE speed cap is lowered to 0.03 m/s for the run. Motion never changes j6.
6. **Feed back.** The outcome goes into history with status `reached|not_reached|unreachable|rejected`, `measured_pose`, `measured_qpos` and errors.
   - An exact arm target that already failed is refused next time ("This target already failed; change the approach").
   - After 3 consecutive failures a `stuck_hint` is appended to the reason (`imitation_actions.py:113-121`).

**Termination.**
- `done` increments `verified`. Two `done`s (with only `observe` allowed in between) end the run successfully: "Task reported complete with an empty gripper and two confirming looks".
- Any motion after a first `done` fails the run.
- `stop` fails the run.
- The `finally` block halts the arm and base and **never reopens the gripper**.

**Grasp verification is the model's job, using a numeric aid.** The prompt says: "a live reading at or below the demonstrated narrow end means you are probably holding nothing". The demo's j6 range is the scale.

### 4.7 Observability

- Every run writes `workspace/custom_skills/.imitation_runs/slashpick-<uuid>/` containing:
  - `manifest.json`: model, route (`openai-direct` | `innate-proxy`), toggles;
  - per-step live JPEGs;
  - `phase_map.json`;
  - `trace.jsonl`: decision, observation, execution;
  - `inspection.jsonl`: raw tool calls.
- It publishes `/brain/icl_trace` JSON events (`run`, `tool`, `phases`, `step`, `execution`, `note`). Each repeats the header so a page opened mid-run can recover.
- The `/icl` page shows the demo frames, live cameras, phase ladder and per-decision cards (latency, frames seen, measured outcome).
- Proxy routes load demos "through the same loader the skill uses, so it cannot show a context different from the one the model receives".

### 4.8 Dataflow

```
 .h5 episode (1 demo, 30 Hz, qpos/action/ee_pose/2×640×480)
   │ Demonstration(): validate, FK provenance, grip midpoint events, select frames, JPEG q75
   ▼
 ImitationPolicy (stateless Responses calls, gpt-6-astra, effort=low, priority tier)
   call1 note_observations(survey 24f) → call2 record_phases(survey∪events) → callN act(phase window ≤7f)
   input each call: instructions(~~~3.7k tok~~ [corrected: ≈3.2k tok by o200k_base]) + DEMO frames + JSON{phase_map,notes,live state,history[-6:]} + LIVE head/wrist
   ▼  1..chunk_size decisions (JSON-schema strict, enum actions)
 check_decision / phase rules  ──(ValueError ⇒ run FAILS)
   ▼
 skill loop: fresh telemetry (stale >1 cm ⇒ discard) → latch + envelope + vertical-descent ⇒ "rejected" (recoverable)
   ▼
 ArmRuntime: IK + joint2 floor → move_to / stream_joints / cmd_vel pulses → measure → outcome → history
   ▼                                                           └→ /brain/icl_trace → /icl page; .imitation_runs/
 done ×2 with empty gripper ⇒ success
```

---

## 5. Prompt: key passages, verbatim

The prompt is assembled from blocks in `imitation_prompt.py`. With default toggles it renders to **14,728 chars, ~~about 3.7k tokens~~ [corrected: 3,175 tokens with tiktoken `o200k_base`, ≈3.2k. 3.7k is the chars/4 heuristic, and Astra's tokenizer is not published. The showcased toggle set (`ee_delta` instead of `ee_absolute`) renders to 14,754 chars / 3,183 tokens. Re-measured by building `build_instructions` offline]** (I measured this by building it offline).

- **Role and task inference** (`:79-82`): *"You are this MARS robot. A recorded demonstration of ONE task is your only description of what to do: infer the task from it and reproduce it on the live scene. Nobody will tell you what the task is. Do not guess from object names — there are none — infer from what the recorded robot actually did across the episode."*
- **Evidence, not script** (`:84-90`): *"Read the demonstration as evidence, not as a script to replay. … Watch gripper_target_rad across the episode: where it changes, the recorded robot grasped or released something. Those transitions are landmarks, not the task — most of the work happens between them…"*
- **Intent transfer** (`:98-99`): *"The live scene is not the recorded one: objects sit differently. Reproduce the recorded INTENT against what you see now, never recorded joint values as absolute targets."*
- **Grasp geometry** (`:125-131`): *"Grasp from above, or failing that from the front — never from behind. Centring and descending are two separate moves and must never be combined…"* The validator enforces this.
- **Phases** (`:164-170`): *"A phase is one intent — approach something, act on it, carry it, put it down, withdraw — not one gripper interval. … If the survey shows the arm doing something to an object and the scene changing as a result, that is its own phase even though the gripper never moved."*
- **Hardware lore** (`SHOULDER_LOAD`, `:34-68`): *"Joint 2 is the shoulder. … it simply trips a hardware overload, goes limp, and the run is over. What trips it is time under load … Every decision you make costs seconds of real time, so whatever pose you leave the arm in is a pose the shoulder holds while you think."* ~~The code comment beside it adds~~ [corrected: the comment is not beside `SHOULDER_LOAD`. It sits on `shoulder_hard_max = 0.5` in `imitate_demonstration.py:111` and reads]: "the run that tripped the shoulder reached 1.04".
- **Persistence** (`:44-51`): *"Setbacks are not reasons to stop. … Repetition is the only real failure mode: if the same approach fails twice, change the approach… Only call stop when the scene makes the task impossible…"*
- **Aperture-based grasp check** (`:180-185`): quoted in §4.6.
- **Optional task** (`:199-203`): *"THE TASK, stated explicitly: {task} … Use the description to resolve what the recording is ambiguous about, not to override what it shows you."* When `task` is set it is also injected into every observation.

---

## 6. Results and evidence

**There is no quantitative evaluation in the PR.**
- Verification is limited to lint, the existing test suites ("283 `brain_client` tests, 24 proxy tests"), a C++ recorder fixture, and "Loader, policy window, grip latch, dead reckoning and every validator exercised against a real 1053-frame episode". None of the Python imitation logic has committed unit tests.
- The claimed with-task vs no-task ablation has no published numbers.

**The one public run** (X post, 2026-09-11, 67k views; 20.7 s demo video plus 34.3 s sped-up trace video). These are my reads of the trace UI:

| item | value |
|---|---|
| demo | `replacecactus · episode_0`, 1053 rows, 35.1 s, 3 grasp/release events, base drove 0.00 m |
| setup | demo on a white table; live on a wooden coffee table in another room, different layout; no task text |
| run config | chunk_size 1, overview_frames 24, frame_selection both, motions joint_step + ee_delta + base_step |
| inferred phase map | 1 Approach and aim the held rod beside the upright pot (0–228, ref 228) · 2 Sweep the pot off the platform with the rod (229–294, ref 274) · 3 Release the rod onto the tabletop (295–320, ref 312) · 4 Approach and grasp the small clear cube (321–640, ref 630) · 5 Lift and transfer the cube onto the cleared platform (641–869, ref 869) · 6 Withdraw and verify the finished arrangement (870–1052, ref 1052) |
| decisions | ~~≈43–45 (`#42 observe` was followed by the closing `done`)~~ [corrected: exactly 45 steps, #0–#44. #42 `observe`, #43 `done` (first confirmation), #44 `done` (second confirmation, run ends "Completed")] |
| per-call latency | ~~3.4 s – 8.5 s for `act` (values visible: 3.4, 3.6, 3.7, 3.85, 3.9, 5.0, 5.05, 5.25, 5.6, 5.95, 6.0, 6.9, 8.4, 8.5)~~ [corrected: the UI prints seconds with an uppercase "S" ("3.2S", "5.2S"). Readings like "3.85", "5.05", "5.25", "5.95" are that S misread as a 5. Re-read at 1–4 fps: `act` latency 3.2 s (#3) to 8.5 s (#22), median ≈4.3 s, n≈44. Planning calls: `note_observations` 11.5 s (24 frames shown), `record_phases` 8.1 s (33 frames shown). Step #0 card shows 23.2 s, which includes both planning calls. Sum of model wait ≈224 s, so the 34.3 s trace video is sped up ≥6.5×] |
| recovery shown | `#26 close_gripper` → model: "The jaws closed to 0.006 rad, below the demonstrated empty stop…" → `#27 open_gripper` → `#28` rise → `#29` re-centre → successful regrasp |
| outcome | "Completed — Task reported complete with an empty gripper and two confirming looks" |

**Third-party reaction** (replies):
- Cost: "$20/min". Peytavin: "not if you use kv-caching". The code sets no explicit cache key.
- Speed: "you should not expect to have anything fast with this method, for a slow skill it can do wonders".
- A request for repeated-run statistics went unanswered.

**Cost (derived, UNVERIFIED).**
- Basis: the OpenAI model page lists `gpt-6-astra` at $10/$50 per M input/output, with cached input $1. Innate's PR #737 doc treats `service_tier:"priority"` as "Astra Fast" at 2×: $20 input, $2 cached, $25 cache write, $100 output per M.
- Per `act` call: ~~~3.7k~~ [corrected: ≈3.2k (o200k_base proxy)] instruction tokens, 16 images at 640×480, and ~3–5k tokens of unrounded JSON (history and frame metadata). That is plausibly 12–25k input tokens, depending on Astra's per-image token count, which I could not find.
- At about 45 calls plus two heavy planning calls (≈68 images), a run plausibly costs **$5–20 uncached**. Prompt caching of the stable prefix (instructions plus the current phase's demo frames) could cut input cost by roughly 2–4×.
- The trace shows no measured cost.

---

## 7. Issues I found in the code

These come from reading the code, not from running it.

1. **Task-specific assumptions remain in the "generic" skill.**
   - The run always starts with `holding = True`, commented "every run of this pair starts with the prop in the gripper", and re-closes the gripper at start (`imitate_demonstration.py:235, 246`).
   - The prompt hard-codes *"You begin with something already in the gripper."* (`imitation_prompt.py:147`).
   - A demo that starts empty-handed is therefore mis-specified. Stale "Slash-and-pick" docstrings, run-dir prefix and failure strings remain (`:47-76, 199, 252, 416`).
2. **Model-output validation errors are fatal, not re-asked.**
   - `check_decision` and the phase rules raise inside `policy.decide`. The worker converts that into `skill.fail("Demonstration model request failed (ValueError)")` (`imitation_runtime.py:227-242`).
   - So a `joint_step` of 0.16 rad, an `ee_delta` of 4.1 cm, an illegal phase advance, or a response that hits `max_output_tokens` all kill the run.
   - The JSON schema enforces only structure: `pose` has no numeric bounds.
   - Only host-side refusals (latch, envelope, vertical descent, IK) are recoverable.
3. **Phantom tool and stale names in the prompt.**
   - It tells the model to *"inspect_demo any source index to look closer"* (`imitation_prompt.py:93`), but no such tool exists in this version. This contradicts the PR's own rule that describing uncallable tools is noise.
   - `ee_delta` help says "the same rejections as move". The `act` schema's `pose` description says `move [x,y,z,r,p,y]`.
4. **No previous image pair is sent.** The prompt says *"Compare the newest live pair of images against the previous one"*, but each stateless call carries only the current live pair. History entries have their images stripped.
5. **`legacy_urdf` is not propagated to lazy frame loads.** `_load` constructs `Demonstration(path, frame_indices=…)` without `legacy_urdf` (`imitation_policy.py:86-89`). A legacy recording should therefore raise on the first acting turn.
6. **`retry` resets the phase pointer to 0.** After that the model must re-advance one phase per turn with evidence.
7. **Dead code:** `reload_failed_servos` (`imitation_runtime.py:130-151`) and `Demonstration.context()`. Doc drift: the 250 ms sync rule does not apply when `image_time_reference=True`, which the skill always uses (100 ms nearest-sample rule instead).

---

## 8. Related Innate work and cross-references

**Innate PR #737** (2026-08-29, open): `open_cabinet_with_gpt`, the same Astra pattern (one bounded action per observation, priority tier, low effort).
- Notably better context engineering than #817: an append-only text/action/reasoning history plus only the newest two image pairs; **explicit cache breakpoints** with a 30-min TTL; telemetry rounded to 4 dp.
- Measured on a six-observation replay: ~~input cost~~ [corrected: *total estimated API cost*, covering input, cache reads/writes and output at Astra Fast rates, for an offline replay of six saved observations with fresh model decisions and no motion (`docs/experiments/gpt6-cabinet.md` on branch `codex/pull-held-handle`). Input tokens went 21,828 → 19,090, cache writes 21,024 → 2,448, output 504 → 636] $0.5921 → $0.3362 (−43%), with 6,748 cached tokens.
- It is explicitly "inspired by Jay Chooi's tweet and Robocurve's public inspect-robots-agent", which links #817's lineage to the Robocurve harness (see `robocurve-gpt6-astra.md`).

**Innate PR #843** (2026-09-18, open), `do_task`: the authors' successor.
- Language task, any `innate_llm` model, ~360 lines, **no demonstration**.
- The head image carries a metric floor grid and a reach box; the wrist image a fingertip crosshair.
- Actions: relative `nudge` ≤5 cm/axis, `drive`, `grip`, `look`, `rest`, `done`/`fail`.
- Sim results on "pick up the pink cube", on an earlier revision: **Gemini 3.6 Flash 4/5, GPT-6 Astra 1/2, Claude Sonnet 5 1/8**.
- The final interface was "not re-benchmarked".

Other sibling PRs: #794 (Astra map scratchpad), #850 ("Demo Agent to Astra"), #810 (`learn_skill`).

Cross-references to other user sources:
- **GPT-as-Policy** cites #817 and the X post as a single-demo showcase, "not a multi-trial success-rate evaluation".
- **DexAgent** consumes the same input (one demo) offline: demo → sim data → trained VLA. #817 is the online, in-context counterpart.
- **cheng-haha/GPT-Policy** ("In-Context Robot Learning with VLM Agents") is the academic analogue of demo-as-context.
- **Robocurve, piper-astra-jev** and the others in the Astra cluster use different embodiments. None reports demo-conditioned success rates.

---

## 9. Assessment

**Strengths**
- **Context discipline for long episodes.**
  - Uniform survey, then sharpened looks at grip transitions, then a committed phase map, then per-phase windows of 4 samples plus neighbouring anchors.
  - This is a sensible, cheap way to fit a 35 s, 1053-row demo into a stateless call. The forced "write notes before the next look" step is a crude but effective chain-of-thought checkpoint.
- **Single source of truth for limits.** Caps live in one module and are interpolated into the prompt. The prompt is regenerated from the enabled tool set. This avoids the "model told X, enforced Y" failure mode.
- **Physics-aware guards that encode real failure modes:**
  - vertical-descent-only;
  - refusing targets the driver would silently clamp (joint2 floor);
  - "motion never changes the gripper";
  - stale-observation discard after multi-second latency;
  - sag reported separately from step error;
  - repeated-target memo and stuck hint.
- **The demonstration doubles as a safety and verification prior.** It supplies the shoulder limit (demo max + 0.2, ~~hard ≤0.5 rad~~ [corrected: the *value* is capped at 0.5 rad, but `shoulder_limit_rad` and `demo_joint_range` are advisory only. They are observation fields explained in the prompt, and no host-side check enforces them (grep: only `imitate_demonstration.py:185-193, 261-263` and `imitation_prompt.py:53,140`). Both are computed over the ≤39 *shown* frames, not "across the whole recording" as the prompt says]), the per-joint envelope, and the gripper empty-aperture scale used to detect missed grasps. This is a clean idea, and the trace shows it working.
- **Provenance and observability.** FK from the same joint sample, kinematic URDF fingerprinting, a per-run directory, a live trace topic, and a UI that renders exactly what the model saw.

**Weaknesses**
- **No measurement.** n=1 showcase from a pre-final build. No success rate, no ablation numbers (task vs no-task, frame selection), no cost.
- **Slow.** About ~~3.4–8.5 s per call and ~45 calls~~ [corrected: 3.2–8.5 s per `act` call (median ≈4.3 s) over 45 steps, plus 19.6 s of planning calls, ≈3.7 min of model wait in total] make it minutes per task. Arm speed is capped at 0.03 m/s. Holding poses while thinking is a real hardware hazard (the shoulder overload lore).
- **Brittle failure semantics.** Invalid model output aborts the run. There is no API retry. The latch assumes the first state.
- **The prompt is a pile of accumulated hardware- and task-specific patches** (~~~3.7k tokens~~ [corrected: ≈3.2k tokens, 14.7k chars]: wrist pitch, shoulder current, reach, grasp approach). It works for MARS and this task family but is clearly hill-climbed on a handful of runs.
- **Statelessness costs context.** There is no prior image pair, no reasoning carry-over, and the history is limited to 6 verbose entries.
- **Superseded.** The authors' own #843 abandons demo-conditioning for a simpler language-conditioned loop. Its sim numbers do not favour Astra (1/2) or Sonnet 5 (1/8) over Gemini 3.6 Flash (4/5).

**What is novel and what is repackaged**
- *Novel-ish:*
  - one-demo, in-context task inference with an explicit phase-map commitment;
  - per-phase demo windows;
  - grip-event midpoint keyframing;
  - demo-derived joint, shoulder and aperture priors.
- *Repackaged:* the "one bounded action per observation with host-side checks" loop. It follows Robocurve's inspect-robots-agent and Innate's own #737, and is close to Code-as-Policies/VoxPoser-era stepwise LLM control. Single-demo in-context imitation is itself an active research thread (see GPT-Policy). Nothing here is learned.
- *Maturity:* prototype (TRL ~3–4). Open PR, bot-reviewed only, a stated "experimental and supervised… with a hand on Stop", and abandoned in favour of #843.

**What Ilia's Opus-backbone harness should borrow**
1. **Staged, schema-constrained decisions:** survey → commit plan (phases with observable `advance_when`) → act, with a monotone phase pointer and evidence fields. The plan becomes an explicit, checkable artifact.
   - Claude API caveat: forced `tool_choice` (`{"type":"tool"}`/`"any"`) **returns 400 on `claude-opus-5-5` and Fable 5.1**. Port the "exactly one legal tool" ladder as:
     - offer only the stage's tool;
     - `tool_choice:auto`;
     - `strict:true` on the tool, or `output_config.format` structured outputs;
     - a prompt instruction naming the tool.
   - On `claude-opus-5` forced tool choice is still accepted on the Claude API.
2. **Validators defined once and rendered into the prompt.** Return refusals as recoverable tool results (`is_error:true` with measured state) instead of aborting. Add numeric bounds to the JSON schema too, so the API rejects out-of-range values before your validator sees them.
3. **Host-side physical guards:**
   - fresh-telemetry re-read after the LLM call;
   - drift-abort for chunks;
   - pre-checking IK solutions against driver clamps;
   - vertical-descent rule;
   - "motion never touches the gripper";
   - grip latch;
   - repeated-target memo with an escalating "stuck" hint.
4. **Use the demo (or any prior) as numeric scales:** envelope, aperture-empty threshold, load limits. Measured-aperture grasp verification is cheap and robust; pair it with images.
5. **Per-run trace directory plus live trace stream.** The UI should show exactly the frames and JSON the model saw, together with latency and the measured outcome.

**What to avoid or fix**
- Stateless one-message calls with unrounded floats and full-res images. Instead:
  - use an append-only multi-turn history (required by Opus 5.5 / Fable 5.1 preserved thinking anyway);
  - keep the newest 1–2 image pairs;
  - round telemetry;
  - downscale images (Claude bills about 1 token per 28×28 patch, so ~400 tokens per 640×480 image);
  - place `cache_control` breakpoints after the system prompt and the demo block. Max 4 breakpoints; 5-min TTL writes at 1.25×, reads at ~0.1×; `claude-opus-5-5` is $4/$20 per MTok with cache reads at $0.20.
- Hard-coded task priors (`holding=True`) and prompt references to tools that do not exist.
- Treating every model hiccup as fatal. Retry or re-ask with the validator message.
- Relying on per-decision LLM latency for contact-rich phases. Innate's own trajectory (#737 → #817 → #843) shows that the stepwise LLM policy works for slow, supervised tasks. The obvious upgrade for Ilia's "light learned action head" idea is to let the LLM commit phases and targets (as here) and hand each phase to a fast local controller or small policy: MARS already ships ACT at 25 Hz.
- Claims without trials. Build the eval harness first: fixed scenes, N≥10 per condition, and the task-vs-no-task ablation this PR parameterised but never ran.

---

## Sources

- https://github.com/innate-inc/innate-os/pull/817 (body, diff, 7 commits, Greptile review threads)
- https://github.com/innate-inc/innate-os (main @ `1ed590a`: README.md, AGENTS.md, CLAUDE.md, brain_client, manipulation, innate_llm)
- https://github.com/innate-inc/innate-os/pull/816 (same content, closed: wrong user)
- https://github.com/innate-inc/innate-os/pull/737 and `docs/experiments/gpt6-cabinet.md` on that branch
- https://github.com/innate-inc/innate-os/pull/843 (`do_task`, the successor)
- https://x.com/ax_pey/status/2098216469012283681 and its thread (via api.fxtwitter.com); videos `amplify_video/2098212559770013696` and `2098213452871593984`
- https://innate.bot ; https://docs.innate.bot/llms.txt ; https://docs.innate.bot/robots/mars ; https://docs.innate.bot/software/agent-loop ; https://docs.innate.bot/training/overview ; https://docs.innate.bot/software/skills/policy-defined-skills
- https://developers.openai.com/api/docs/models/gpt-6-astra (pricing, effort levels, limits)
- https://news.ycombinator.com/item?id=45504127 (Show HN: MARS); https://www.crunchbase.com/organization/innate-f5f3 ; https://tracxn.com/d/companies/innate/__vGfpZeb0DqEibidi206tzLx65-GaDRGP2KcG9iG3HFs
- https://anonymous-report-421.github.io/public-website/ (GPT-as-Policy, which cites this PR); teammate notes `sources/robocurve-gpt6-astra.md`, `sources/dexagent.md`, `sources/gpt-as-policy.md`

---

## Verification (fact-check pass)

**Pass date and basis.** Fact-checked 2026-10-01/02 against these primary sources:
- **Code** at PR head `ef8fd0d`, in `repos/innate-os-pr817`, and at main `1ed590a`, in `repos/innate-os`. Read file by file with line numbers.
- **GitHub API.** PR #817 metadata, per-file stats, all 7 commit messages, the 6 Greptile threads with author replies, and Greptile's summary comment. Also PRs #816, #843, #737 (plus `docs/experiments/gpt6-cabinet.md` on `codex/pull-held-handle`), #794, #850 and #810.
- **The X thread** via `api.fxtwitter.com/2/conversation/2098216469012283681`, which returned 40 replies.
- **Both videos.** Downloaded at 1080p. The trace video was frame-sampled at 1 fps and 4 fps, with the reasoning panel cropped and enlarged.
- **Web pages.** docs.innate.bot (`/robots/mars`, `/training/overview`, `/software/agent-loop`), innate.bot, the YC company page, HN item 45504127 (Algolia API), and the OpenAI `gpt-6-astra` model page.
- **Claude API reference.** The bundled `claude-api` skill reference, model table cached 2026-06-24.

Overall, the note is careful and mostly accurate. Code-level claims (file:line, constants, request body, guards, prompt quotes) check out almost everywhere. The errors cluster in numbers read off the video and in token estimates.

### A. Confirmed claims (seen in primary source)

**PR metadata**
- Title "Imitate a recorded demonstration with GPT-6 Astra". State OPEN, unmerged. `updatedAt` 2026-09-11T01:01:51Z. +4162/−9, 32 files, 7 commits, head `ef8fd0d`, branch `demonstration-skill`.
- Author `DavidDobas` (david@innate.bot). Every commit has `Co-authored-by: Axel Peytavin <peytavin@stanford.edu>`.
- `reviewDecision: REVIEW_REQUIRED`. Only `greptile-apps` reviewed.
- Greptile raised exactly the 6 issues listed, and each author reply cites the fixing commit (67c447c, ffa575a, 9cdd5ed). The final summary says "Confidence Score: 5/5 … appears safe to merge" (7 reviews, last on `ef8fd0d`).

**File line counts.** All match `wc -l` at `ef8fd0d`: 325 / 216 / 306 / 238 / 92 / 124 / 444 / 425. The C++ total is +192 lines and webapp+proxy is +1,639 lines, matching "~190" and "~1,600".

**Timeline**
- First commit `ae8fd71` at 2026-09-10T20:36:15Z. PR #816 was created 20:36:46Z and closed 21:14:11Z; it was opened from user `ugtthis`. The author's comment says the PR "was raised from a stale gh session", so "wrong user" is a paraphrase.
- PR #817 was created at 21:14:13Z. The last commit is 00:59:21Z, with the `ResponseNotRead` message quoted correctly.
- Tweet 2026-09-11T01:06:03Z: 67,399 views, 520 likes, 248 bookmarks. Demo video 20.67 s; trace video 34.302 s.
- "set it all up and running in less than a day" is in a 2026-09-12 01:20Z tweet.

**Request body** (`imitation_policy.py:188-200`): confirmed verbatim. Model `gpt-6-astra`, `service_tier:"priority"`, `store:false`, `reasoning.effort:"low"`, one tool, forced `tool_choice`, `parallel_tool_calls:false`, `max_output_tokens:2500`, HTTP `timeout=40`. All three tools are `strict: true`.

**Planning ladder**
- `decide` is at `:246-255`. `_record_phases` is at `:257-278`: 2–6 phases, 0≤start≤ref≤end<N, non-decreasing starts/ends, `ref` among held frames, name/advance_when 1–600 chars.
- `_action` (`:280-302`): keep or advance by one, advancing needs history plus `previous_phase_complete`, `done` only in the last phase, only `joint_step` batches.
- `restart_attempt` resets the phase to 0.
- The trace UI confirms `frames_shown` 24 for `note_observations` and 33 for `record_phases`.

**Statelessness.** Each call is one `user` message with `history[-6:]`, `notes_so_far`, `phase_map` and `current_phase`. Only the current live pair is sent.

**Frame encoding**
- Prefix `"DEMO "` (`imitation_policy.py:114`), whereas the dead `Demonstration.context()` uses "DEMONSTRATION ".
- JPEG Q75 with no resize (`demonstration.py:255`). Floats are unrounded; `icl_trace.jsonable` rounds to 3 dp only for the UI.
- `grip_events` midpoint crossing and the docstring quote: `:84-94`. `model_fingerprint` SHA-256 over joint name/type/origin/axis/limit: `:97-118`. The run refuses on mismatch: `imitate_demonstration.py:194-196`.

**Action caps** (`imitation_actions.py:15-19`): 0.15 rad, 0.03 m, 0.04 m, 0.2 rad.
- Workspace box and the "A descent must be vertical" rule with 5 mm tolerances: `:165-185`.
- Grip latch: `:188-202`. `stuck_hint` at ≥3 failures: `:113-121`. `event_frames` (spread 8, cap 15): `:94-102`. `with_limits`: `:205-216`.

**Runtime**
- 175 s decision timeout on a worker thread. Stale discard at >1 cm (`imitate_demonstration.py:280-295`) and chunk drift >0.03 rad.
- `try_move`: IK → joint2 floor → `move_to(duration=1.5)` → reached if ≤1.5 cm.
- `try_joint_step`: `stream_joints(max_speed=0.25)` until error ≤min(0.02, |Δ|/3) or 4 s; held-joint sag >0.02 reported separately.
- `try_base_step`: 0.04 m/s pulses of 0.15 s, wander aborts, 5 s timeout.
- EE speed is capped at 0.03 m/s for the run. `finally` halts without reopening the gripper. `max_steps=110`, `time_budget_s=1200`.
- Success needs two `done`s, with only `observe` allowed in between.

**Prompt quotes.** All the quoted passages and line ranges in §5 match `imitation_prompt.py`. Default prompt length is 14,728 chars, re-measured.

**§7 issues 1–7 confirmed in code**
- `holding = True` with the comment `:235`; re-close at `:246`; prompt `:147`.
- Fatal `ValueError` path at `imitation_runtime.py:227-242`.
- Phantom `inspect_demo` at prompt `:93`. It also appears in `icl_trace.py:63` and `webapp/js/icl/demoPanel.js:11,212`, and in the video's demo-panel text.
- No previous image pair is sent. `_load` drops `legacy_urdf` (`imitation_policy.py:88`).
- `reload_failed_servos` and `Demonstration.context()` are unused (grep).
- The 250 ms vs 100 ms doc drift is real: `demonstration.py:235-251` vs `docs/skills/imitate-demonstration.md`.

**Recorder**
- URDF load at `recorder_node.cpp:66-71`. FK of the same `qpos` plus `set_kinematics`: `:382-387`. HDF5 attributes: `episode_data.cpp:215-244`.
- No deflate filter, so episodes are uncompressed. `recorder.yaml` gives 30 Hz, 640×480 and `max_timesteps: 18000`.
- 1053 rows × 2 × 640×480×3 B = 1,940.9 MB, matching the UI's "1941 MB".

**Showcased run (trace UI)**
- `replacecactus · episode_0`, 1053 rows, 35.1 s, 3 grasp/release events, "drove 0.00 m · turned 0.00 rad", robot model `5c02b6ef889d`.
- Skill `slash_and_pick_cactus_no_prompt`, run `SLASHPICK-59…`. Toggles: chunk_size 1, overview 24, `both`, joint_step + ee_delta + base_step (ee_absolute off). The six-phase map is exactly as listed.
- The #26→#27 recovery quote is verbatim: "The jaws closed to 0.006 rad, below the demonstrated empty stop, and no cube is visible between them; acquisition failed despite the latch."
- Final card: "Completed — Task reported complete with an empty gripper and two confirming looks".

**Thread replies.** All verbatim: "$20/min ....", "not if you use kv-caching", "you should not expect to have anything fast with this method, for a slow skill it can do wonders". @hotmob's repeated-run question has no visible reply among the 40 fetched replies.

**Main repo**
- `DEFAULT_MODEL = "google:gemini-3.6-flash"` at `innate_llm/configure.py:45`. The `claude-`/`gpt-` catalog rows are at `models.py:30-31`.
- ROS 2 Humble + Zenoh (AGENTS.md:8). `pick_any_object` uses `google:gemini-3.5-flash` (`:295`) with an HSV/CamShift tracker.
- ACT runs at `inference_hz: 25.0` with `act_trt.py` and `auto_stop.py` (progress head `action[8]`). The docs say "Record 30+ demonstrations with the leader arm".
- Repo is Apache-2.0 with 81 stars and 41 forks (2026-10-01).

**Company and robot**
- MARS specs match docs.innate.bot/robots/mars exactly: XL430-W250-T (1,3), XC430-T240BB-T (2), XL330-M288-T (4–7), 40 cm / 2 mm / 250 g, head stereo 150° diagonal, Jetson Orin Nano Super 8GB, 2D LiDAR.
- $995 is on innate.bot. "< $2k" is in the HN title. The π0.5 comparison is in Peytavin's HN comment.
- YC F24 is confirmed on the YC page.

**Sibling PRs**
- #843 quotes, the 4/5 · 1/2 · 1/8 numbers and the "not re-benchmarked" caveat are all confirmed in its body.
- #737's 4-dp rounding, two newest image pairs, explicit breakpoints with 30-min TTL, Astra Fast $20/$2/$25/$100, and the Jay Chooi / Robocurve inspect-robots-agent credit are confirmed. The credit is in `gpt6-cabinet.md:86-89`, not in the PR body.
- Titles and dates of #794, #850 and #810 are confirmed.

**OpenAI pricing.** The `gpt-6-astra` page lists $10 input, $1 cached, $12.5 cache write and $50 output. It says "Fast mode … 2x the applicable rates", lists effort levels low/medium/high/xhigh/max, a 1.05M context and a knowledge cutoff of Apr 30 2026. It does not mention a "priority" tier by name. Innate's #737 doc maps `priority` onto "Astra Fast".

**Claude API claims (§9).** Consistent with the Claude API reference (skill, cached 2026-06-24):
- Forced `tool_choice` `any`/`tool` returns 400 on `claude-opus-5-5`, Fable 5.1 and Mythos 5.1. `claude-opus-5` is not in that list.
- `strict: true` and `output_config.format` exist.
- Images cost roughly one token per 28×28 patch, so 640×480 ≈ 392 tokens.
- Max 4 cache breakpoints. Writes cost 1.25× (5-min TTL) or 2× (1 h); reads cost ~0.1×.
- `claude-opus-5-5` is $4/$20 with $0.20 cache reads.
- Preserved thinking: the reference says to make every harness append-only. History edits invalidate thinking blocks, and accounts created on or after 2026-08-31 get a 400.

**Cross-reference.** The GPT-as-Policy site does cite the X post plus PR #817. The quote "not a multi-trial success-rate evaluation" is a translation of the Chinese "属于个例演示，不是多次成功率评测" (`public-website/data.json`).

### B. Corrections (also fixed inline)

1. **Decision count.** "≈43–45 decisions (#42 observe followed by the closing done)" → exactly **45 steps (#0–#44)**. #42 `observe`, #43 `done`, #44 `done` ends the run. Source: trace video, final frames.
2. **Latency.** "3.4–8.5 s; values 3.85, 5.05, 5.25, 5.95…" → the UI suffix is an uppercase "S", so those trailing 5s are the unit.
   - Re-read values: `act` **3.2–8.5 s, median ≈4.3 s** (n≈44, ±0.1 s).
   - The note omitted the planning calls: `note_observations` **11.5 s**, `record_phases` **8.1 s**. Step #0 = **23.2 s**. Total model wait ≈**224 s**, so the 34.3 s video is sped up ≥6.5×.
3. **Transport size.** "88-line Responses-API transport" (§2) → **92 lines** at `ef8fd0d`. The PR-body table says 88; the note's own §3 table already says 92.
4. **Dead-reckoning fix.** "fixed dead-reckoning sign/offset" → it was an **integration-direction / one-sample offset** bug, not a sign error. Commit `ffa575a`: "Driving 0.5 m/s for two seconds now integrates to 1.000 m where it used to give 0.950."
5. **Prompt token count.** "about 3.7k tokens" (§5, §4.8, §6, §9) → **3,175 tokens** (o200k_base) for 14,728 chars, ≈3.2k. 3.7k is chars/4, and the Astra tokenizer is unpublished.
6. **Shoulder comment location.** "The code comment beside it adds: 'the run that tripped the shoulder reached 1.04'" → the comment is at **`imitate_demonstration.py:111`** (`shoulder_hard_max = 0.5`), not beside `SHOULDER_LOAD` in `imitation_prompt.py`.
7. **Observe attribution** (§4.6 step 1).
   - The snapshot checks are `LiveObservation.snapshot` (`imitation_runtime.py:83-121`). The 5 s wait is `ArmRuntime.read` (`:197-205`). The base-drift check is `ArmRuntime.observe` (`:207-216`). The demo priors are added by the skill (`imitate_demonstration.py:254-264`).
   - The health message only needs to be **≤3 s** old, not 1.5 s.
8. **#737 cost figure.** "input cost $0.5921 → $0.3362" → **total estimated API cost** (input + cache + output) at Astra Fast rates, for an offline 6-observation replay with no motion.
9. **Shoulder limit.** "shoulder limit (demo max + 0.2, hard ≤0.5 rad)" read as a guard → `shoulder_limit_rad` and `demo_joint_range` are **advisory only**: observation fields plus prompt text, enforced nowhere (grep). Only the *value* is capped at 0.5. Both are computed over the ≤39 shown frames, although the prompt claims "across the whole recording".

### C. Unverifiable or soft claims (keep marked)

- **Run cost "$5–20 uncached"** and the "2–4×" caching saving remain UNVERIFIED. Astra's per-image token count is unpublished, and the trace shows no usage.
  - One calibration point (derived, UNVERIFIED): #737's replay used 21,828 input tokens over 6 decisions, each with ≤2 image pairs plus history. That is ≈3.6k tokens per decision, so Astra image tokens are probably in the hundreds, not thousands, per image.
  - OpenAI applies automatic prefix caching without a key. #817's per-phase prefix (instructions + tool + demo window) is byte-stable across `act` calls within a phase, so implicit cache hits are plausible. Peytavin's "kv-caching" reply may refer to this. UNVERIFIED.
- **"~3–5k tokens of unrounded JSON" is likely low.** Each DEMO frame carries ~49 full-precision floats including `camera_observations`. Every `act` call also re-sends `notes_so_far` (free-text survey notes, up to 2,500 output tokens) and 6 verbose history entries. UNVERIFIED either way.
- **CTO title for Vignesh Anand.** The YC page and HN confirm he is a co-founder; the CTO title is third-party only.
- **HQ.** YC lists Innate in **San Francisco**. "Palo Alto" is Peytavin's X profile location.
- **"Request for repeated-run statistics went unanswered".** True for the 40 replies fetched on 2026-10-01; later replies were not checked.
- **"TRL ~3–4" and "abandoned in favour of #843"** are interpretations. #817 is still OPEN, not closed. #843 says it "replaces" the approach.

### D. Important missed details (added)

1. **More failure modes in the one public run, beyond the missed grasp.**
   - #22 `joint_step j2 −0.15` was **NOT_REACHED**: "Joint2 stopped 0.076 rad short of its target".
   - #39 `ee_delta` was refused as **UNREACHABLE** by the new joint2-floor guard: "This pose needs the shoulder folded back to joint2=-0.70, past the -0.50 the body allows here; no movement issued." This is the only time that guard is seen firing live.
   - Phase 2 shows sweep oscillation: #7 j1 +0.15, then #8 "Reverse the sweep…".
   - Successful grasp: #31, then #32 evidence "aperture is 0.436 rad, clearly above the 0.112 empty stop", so the demo's `gripper_closed_rad` was 0.112.
   - Phase 3 (release the rod) took a single `act` (#13), which advanced the phase and opened the gripper together.
2. **Hallucinated base motion.** The model's own survey note reads "base advances ~0.123 m then arm raises…", although the recording's base channel is 0.00 m and no `base_*` fields were sent. Its first action (#0) was a `base_step 0.030 m`. Either the model misattributed arm/scene motion to the base, or the base moved without a recorded `/cmd_vel`. Unexplained, and a caution for demo-to-intent inference.
3. **Termination prior.** The latch refuses `done` while `holding` (`imitation_actions.py:200-201`) and the run starts with `holding=True`. A demonstration whose task *ends holding an object* therefore can never terminate successfully, which is a second hard-coded task-family prior.
4. **Fatal vs recoverable cap violations are asymmetric.**
   - Out-of-cap `joint_step`, `base_step` and `ee_delta` are rejected inside `check_decision`, in the policy thread, and are **fatal**.
   - An out-of-cap `ee_absolute` is only caught by host-side `check_reachable_move` and is **recoverable** ("rejected").
   - The `pose` schema has `maxItems: 6` but no `minItems` or numeric bounds, so a wrong-length pose is also fatal.
   - Any HTTP non-200, timeout or `status != "completed"` is fatal. `openai_responses.py` never retries.
5. **Single-look modes.** In `frame_selection` = `keyframes` or `uniform` there is no survey, so the ladder is just `record_phases` then `act`, with no `note_observations` call. The trace confirms that `both` makes 2 planning calls (24 then 33 frames).
6. **Close-look frame count.** "±8 rows" means **3 frames per event** (i−8, i, i+8), capped at 15. It is not a 17-frame window.
7. **Length limits.** `evidence` 1–1,200 chars, `reason` 1–240 chars. The act tool description is "Return a chunk of joint movements, or one decision of any other kind."
8. **Representation mismatch the prompt must patch.** Demo `ee_pose` is xyz + quaternion; live `pose` is xyz + roll/pitch/yaw. The prompt says "orientations never [compare] — never copy recorded orientation components". The prompt also states "The wrist camera looks 25 degrees DOWN relative to the level wrist" and that coordinates are base_link, +x forward, +y left.
9. **Images.**
   - `input_image` has no `detail` field, so Astra's default detail handling applies.
   - Live images are the raw bytes of the `/compressed` topics. Driver defaults are 640×480 for the wrist and the head-left publish size.
   - Note: the docs list the wrist camera as 1920×1080 capable, but the arm-camera launch default is 640×480.
10. **Transport details.**
    - `OPENAI_BASE_URL` lets the direct route target any Responses-compatible endpoint.
    - `INNATE_PUBLIC_DEMO` forces the proxy.
    - An `Accept-Encoding: identity` workaround handles the proxy stripping the gzip header.
    - Upstream error bodies are never surfaced, because they could echo the key.
11. **Dead constant.** `MAX_BATCH_TRAVEL` is defined but unused.
12. **Demo capture.** The video titles it "Collect 1 episode" and shows the operator teleoperating with a small leader arm plus a phone. Dobas replied "Look at the beautiful phone teleop interface".
13. **Author-admitted limitations** (thread and docs):
    - "We haven't completely cleaned it yet".
    - "you should not expect to have anything fast with this method".
    - "experimental and supervised … a hand on Stop".
    - "These are not a geometric collision planner, and model latency means the scene can change after an image."
14. **Caveats on #843's numbers.** The body says the Sonnet 5 1/8 result was partly "a since-fixed bug: the gripper state was assumed open at start". The task was a sim pink cube 0.5 m ahead with a ground-truth lift check; a push task moved the cube ~9 cm of the requested 20. With n=2–8 per model, these numbers do not rank the models. #843's model dropdown also lists Gemini 3.8, Claude Fable 5.1, Opus 5 and Haiku 4.5.
15. **Lineage.** The first-commit version (#816) already used the identical request parameters, caps and phantom `inspect_demo`. The showcased pre-generalisation build therefore differs from `ef8fd0d` mainly in naming and task-input plumbing, not in control mechanics. Inferred from the #816 diff; the exact build of the run is UNVERIFIED.
