# makermods-robotics/metal-arm-harness: deep dive

Repo: https://github.com/makermods-robotics/metal-arm-harness (Python, MIT per `pyproject.toml`, no GitHub description). State as of 2026-10-01: 5 commits, 2026-09-03 to 2026-09-08 (HEAD `1f3c6df`), 15 stars, 4 forks, 1 open external PR. All commits are by Isaac Sin (GitHub `IsaacSinn`), co-founder/CTO of MakerMods. Local clone: `research/repos/metal-arm-harness`. I ran it myself: 97/97 tests pass (`pytest tests`, 142 s) and I drove a sim session (outputs below).

## TL;DR

- It is a **safety and control layer for the MakerMods Metal arm (6 DoF + gripper, Damiao motors over CAN). A coding agent such as Claude Code or Codex acts as the policy by running shell commands.** The README says: *"No API keys, no model SDKs: the agent reading this repository* is *the policy."* The LLM issues `metal-arm-harness op tip X Y Z PITCH`, `op nudge forward=0.02`, `op gripper 45`, and so on, then opens the JPEGs that each reply points to.
- The **first commit (`50ec3bd`) was a direct Anthropic Messages-API tool-use loop** (`claude-opus-5`, 4 tools, raw joint targets). It **was removed about 8 hours later** (`e179e38`). After "two pick-and-place runs" the author moved to a persistent socket server plus a CLI, Cartesian IK primitives, and a hand-written operating manual and skill file. For Ilia this is the most important fact in the repo.
- The value is mostly **in the low-level work, not the LLM**:
  - a measured "table-touch" floor
  - a per-waypoint FK floor check that writes its rejection reasons for the LLM
  - a slow zone near the table
  - a 35° per-call excursion cap
  - a recovery-only-upward rule
  - Ruckig jerk-limited trajectories
  - gravity-sag compensation (commanded base, integral settle, "no-dip" stream origin)
  - grasp verification from proprioception (stall → "holding", torque threshold, "OBJECT LOST")
- **Perception is just the LLM looking at ~~2–3 uncalibrated 640×480 webcam JPEGs~~ [corrected: 2–3 uncalibrated camera JPEGs. 640×480 is set only on the direct OpenCV path (`camera.py:17-18`). The 2026-09-05 bench used two cameras (wrist + side) served by a local HTTP JPEG service (`HTTPSnapshotCamera`, `http://127.0.0.1:8765/camera-N.jpg`). That class does not resize. The same service's recordings were 1920×1080 (`repeat-blue-left-20260905.md`), so the agent probably saw 1920×1080 frames (inferred)].** There are no detectors, depth, extrinsics or pixel→3D tools.
- **Results are anecdotal.** Bench notes (2026-09-05) show three blocks placed into two cups, with several retries and human corrections. There is no success-rate evaluation.

## 1. Context: MakerMods and the arms

- **MakerMods** (GitHub org `makermods-robotics`, "Hardware for Physical AI", www.makermods.ai, dev@makermods.ai). The founders are Ryan Chan and Isaac Sin. Products:
  - **Metal Arm**, $2,499, shipping Sept 2026. Specs: "7-axis" (6 arm joints + gripper), 3 kg rated payload, 612.5 mm working radius, ±0.1 mm repeatability, 4.2 kg, CNC aluminium, 24 V DC, CAN through an XT30 2+2 connector, 180°/s (J1–J3) and 220°/s (J4–J6), 0–95 mm parallel gripper. It supports ROS1/ROS2, MoveIt, LeRobot and ACT/pi0 training.
  - **Maker Arm**, $999 kit / $1,199 assembled. 3D-printed, RobStride RS02 at shoulder and elbow, RS00 elsewhere, 48 V, 1.5 kg payload, ships October.
  - MakerMods Lab (a leLab fork), XLeRobot, OpenBooth SO101.
- Motor facts from code: Metal uses 7 Damiao motors, CAN IDs 0x01–0x07, classic CAN at 1 Mbps, MIT position mode. Joint limits are pan ±160, lift [-180,0], elbow [0,180], wrist_flex [-123,81], wrist_yaw ±85, roll ±145, gripper [0,137.5]. The drivers come from MakerMods' LeRobot fork, branch `arm/makermods-metal` (`DamiaoMotorsBus`, `MetalFollower`).
- **Lineage (same author, same day, 2026-09-03):**
  1. `makermods-inspect-robots`, branch `arm/metal-arm`: an `inspect-robots-metal` plugin for robocurve's **inspect-robots** eval framework. It runs `--policy metal_arm_agent -P model=claude-opus-5` and has a `TableProximityApprover` and an `ArmHealthApprover` (commits `c2c77d6`, `fdd9f31`).
  2. The plugin was extracted into this standalone repo, still with its own Anthropic loop.
  3. The API loop was replaced by a "coding agent drives the CLI" design.
  The diagnostics later cite inspect-robots' CaP-X plugin (Pyroki IK) as a comparison point. robocurve is also the author of the user-listed GPT-6 Astra report.
- **Other links:**
  - Isaac Sin posted on X on 2026-09-18 that he had Typesafe's **Jev** driving the Metal arm *in MuJoCo* with JSON state and typed bounded actions (hover/descend/grasp/lift/place), "9 decisions at ~150 ms each". awesome-jev lists this post.
  - MakerMods forked andlyu's **BluPe remote** (`blupe-remote-makerarm`, 2026-09-14). ~~It runs "Astra through your ChatGPT subscription" via Codex CLI or "Opus through your Claude subscription" via Claude Code, with no API key.~~ [corrected: the MakerMods fork has only `run-codex.sh`. Its README covers running `gpt-6-astra` "with your Codex subscription" and says "No OpenAI API key is needed"; its last commit is 2026-09-13. The quoted "Astra through your ChatGPT subscription … `./run-claude.sh` for Opus through your Claude subscription" text exists only in the upstream `andlyu/blupe-remote-yam` README, added after the fork.] This is the same subscription-coding-agent-as-policy pattern.
  - The team's other repos show heavy use of agents for development. In makermodslab's last 100 commits I counted 15 `Co-Authored-By: Claude Fable 5/5.1` trailers, ~~and 8 PR branches named `codex/…`~~ [corrected: the 15 trailers are confirmed (5× Fable 5, 9× Fable 5.1, 1× lower-case Fable 5). The 8 `codex/…` head branches are counted across all 167 PRs (#140–#161, 2026-09-09 to 09-15). Only one of them appears as a merge in the last 100 commits].
  - **UNVERIFIED:** which agent actually drove the September bench sessions. The repo ships both `CLAUDE.md` + `.claude/skills/…` and `AGENTS.md`, and the diagnostics are written in an agent's voice ("disclosed to the user", "at the user's direction").
- **Third-party discussion:** none found. There are no HN or Reddit threads, and search engines return only the org page. X is not indexable here; I read one tweet through the syndication endpoint.

## 2. Architecture

```
            Claude Code / Codex / human (the "policy")
   reads: CLAUDE.md|AGENTS.md -> docs/OPERATING.md, .claude/skills/metal-arm-pick-place
                 |  shell: metal-arm-harness op <cmd> ...      ^  text reply + JPEG paths
                 v                                             |  (agent Reads the JPEGs)
   cli.py main_op/main_ops --JSON line--> Unix socket ~/.metal-arm-harness/operator.sock (0600)
                                                   |
   operator.py serve() (single-threaded, listen(1)) -> OperatorSession.handle(cmd,args)
     tip/nudge -> ik.solve_tip (DLS, 4 joints) -> pre-check clearance
     goto/gripper/open/close/rest ----------------------------+
                                                              v
   control.py Controller.goto: commanded base, chunk into <=34deg legs, no-dip origin
          -> safety.plan_trajectory (Ruckig 0.15.3 + plan_move floor/limit/excursion/slow-zone checks)
          -> executor.play  @25 Hz: read -> check_runtime(temp/faults/NaN) -> arm.send
          -> executor.settle: integral trim (<=0.75 deg/s, |lead|<=2deg), stall detect
                                                              v
   arms/metal.py MetalArm -> LeRobot MetalFollower.send_action (soft-limit clamp +
          per-joint "lead cap" vs measured: shoulder_lift 5, elbow 3, gripper 8, others 2 deg)
          -> DamiaoMotorsBus (slcan/socketcan, MIT mode, Kp/Kd, v_des=0) -> 7 Damiao motors
   sleep 0.3 s -> _observe(): one JPEG per camera (OpenCV 640x480, q=85) + joints/clearance/tip
   EpisodeLog JSONL: move, observe, motion_sample (every tick), command_error, safety_abort
```

Module map (README §Architecture): `arms/base.py` holds the contracts, `arms/metal.py` and `arms/maker.py` are the adapters plus sim, `kinematics.py` is Pinocchio FK, `ik.py` is IK, `safety.py` is the envelope, `calibration.py` is the ritual, `executor.py` / `control.py` do motion, `operator.py` is the session, and `cli.py` is the CLI.

## 3. Code walkthrough

### 3.1 Entry points and session order

`cli.py` exposes `serve | op | ops | calibrate`. `main_serve` (`cli.py:230`) enforces a fixed order:

1. `connect()` with reads only; torque is left as found (`arms/metal.py:175`).
2. Floor setup in `_set_floor` (`cli.py:173`): an explicit `--table-z`, a saved ritual (`--reuse-table`), a synthetic sim floor 10 cm under the start pose, or a live ritual.
3. Optional resume (`--resume-log`).
4. The torque gate `_torque_gate` (`cli.py:217`). It prints *"About to energize N motors with firm gains. Clear the workspace and keep a hand on the power switch. Press ENTER to arm"*. `--confirm-armed` skips the ENTER prompt so a headless agent can start it after the operator has said "go" out of band.
5. `serve()`.

Without `--armed` the session is a real session with `send()` removed ("UNARMED: move computed and logged, not sent").

Table calibration files are keyed by `sha256(arm\0robot_id\0backend)[:16]` (`cli.py:208`). A sim run therefore never overwrites a real floor, and Maker never inherits Metal's floor.

`main_ops` (`cli.py:298`) runs several commands and stops at the first error. The docs explain why: *"Never chain with `;` … a rejected descent followed by `close` grips air (happened three times)."*

### 3.2 The LLM-facing action surface (current version)

There is no JSON schema. The "tools" are CLI verbs dispatched by `getattr(self, f"cmd_{command}")` (`operator.py:67`):

| command | maps to |
|---|---|
| `observe [tag]` | fresh frames, no motion |
| `tip X Y Z PITCH` | IK to the tool tip in base frame (m, deg); pitch 0 = horizontal, −80 ≈ straight down |
| `nudge forward= left= up= [pitch=]` | relative move in a heading-aligned horizontal frame (`ik.py:103` `offset_target`) |
| `goto joint=deg …` | raw joints; large moves are chunked into legs |
| `gripper DEG` / `open` (112) / `close` (0) | jaw angle |
| `rest`, `clear-faults`, `status`, `inspect`, `monitor S`, `trace-tip …`, `quit` | utility and diagnostics |

**Observation and reply.** Every motion reply is text containing:
- the move summary
- joints in degrees
- `clearance N mm` (the lowest monitored point above the measured floor)
- `tip (x,y,z) m pitch P deg`
- ARMED/UNARMED
- motor faults
- `frames: <paths>`

In the real run below I asked for `nudge forward=0.02 up=-0.03` and got:

```
Move played (1 leg, 45 steps, 2.2s; residual 0.0 deg on shoulder_pan).
joints (deg): shoulder_pan=-0.0 shoulder_lift=-22.4 elbow_flex=21.2 wrist_flex=11.3 ...
clearance 111 mm   tip (0.178, -0.000, 0.330) m pitch 9 deg   ARMED
frames: /tmp/mah-sim/frames/001_after_overhead.jpg
ik: tip [0.1777, -0.0003, 0.3301] pitch 9.1 (2 iterations)
```

Rejections come back as readable errors, for example `ERROR: that tip pose would leave only 1 mm above the table (margin 10 mm); aim higher`.

### 3.3 Perception and cameras

- `camera.py:17-18` sets 640×480 via `cv2.VideoCapture`, with `CAP_PROP_BUFFERSIZE=1` so frames reflect "where the arm is now". Reads are retried 3 times and the handle is re-opened if it dies (`camera.py:78`).
- `HTTPSnapshotCamera` (`camera.py:98`) can reuse an existing local JPEG server at `http://127.0.0.1:…/camera-N.jpg`. It rejects frames whose `/status` age is over 2 s.
- **Cameras are captured only after a move**, after `time.sleep(0.3)` (`operator.py:281`). The agent cannot see anything during motion. Each `op` call is open-loop.
- Bench setup (`OPERATING.md` §1, §4): `overhead=2` (which the doc says *"is NOT overhead … sits on the +y side"*), `front=0` and a `wrist=1` camera on the gripper. The September sessions used wrist + side only.
- Image-axis mappings are written in prose, for example *"Image right ≈ -y"* for the wrist camera and *"negative roll turns the jaws clockwise in the wrist image"*.
- **There are no perception models.** No detection, segmentation, depth or calibrated extrinsics. An Orbbec DaBai DCW2 depth viewer was added (`depth_viewer.py`, `docs/DEPTH_CAMERA.md`: 640×400 at 15 fps, 92.4% valid pixels, 321–671 mm). It is standalone and *"does not connect to the arm"*.
- `serve` refuses to start unless each camera index maps to a device named "KD-USB" (`camera.py:179`, macOS/ffmpeg only). The reason: *"when one USB camera drops off, every index after it shifts and index 2 becomes the laptop camera."*

### 3.4 From commands to motor torques

1. **IK** (`ik.py:45` `solve_tip`): damped least squares on a numeric Jacobian (0.5° probes) of `tool_pose`.
   - Only joints 0–3 are active (pan, lift, elbow, wrist_flex; `DEFAULT_IK_JOINTS`). Wrist yaw and roll are held.
   - Damping 0.02, pitch weight 0.3, tolerance 2 mm / 2°, max 200 iterations, step clipped to 10°.
   - It throws `IKError` rather than return a "nearby guess". Stated speed is "<1 ms".
   - Target orientation is pitch only. Jaw yaw relative to the object is set by hand through `goto wrist_roll=…`.
2. **FK** (`kinematics.py`): Pinocchio on the vendored 6-R URDF `metal_with_gripper.urdf` (sha256 pinned in `assets/urdf/PROVENANCE.md`). Motor degrees map to URDF radians by identity.
   - The tool tip is 0.12 m along the wrist-roll x-axis, a deliberate overestimate (`kinematics.py:27`).
   - `min_height_m` takes the minimum z over the elbow-to-wrist joint origins and the tool tip (`kinematics.py:67`).
3. **Controller.goto** (`control.py:201`):
   - Unmentioned joints hold the last *commanded* value, not the sagged measured one (`base()`, `control.py:182`). The reason given: *"otherwise gravity sag … is re-planned into the goal on every call and the arm creeps toward the table"*. If the arm was moved by hand by more than 6° (`STALE_BASE_DEG`), the measured value is used instead.
   - Requests are chunked into legs of at most `max_excursion − 1` = 34°, up to 64 legs (`control.py:257`). [corrected/nuance: the 34° is measured from the *measured* pose (`leg_target = current + clip(delta, ±34)`, `control.py:263`). The plan, however, starts from `_origin()`, which is the last command and can be up to `ORIGIN_MAX_LAG_DEG` = 4° away. A leg can therefore span up to 38°, and `plan_move` then rejects the first leg against the 35° cap. I reproduced this in sim: a 3.5° lead gave "shoulder_pan would move 37.5 degrees … over the 35 degree per-move limit". It matches the bench note "Direct rest and some long IK transitions were rejected before motion by the 35-degree step limit" (`repeat-blue-left-20260905.md`). If a *later* leg is rejected, the arm stops partway and the reply only notes "later leg rejected by the envelope; stopped early" (`control.py:273-278`).]
   - Each leg starts from the *last sent command* (`_origin`, `control.py:392`), which fixes a dip seen on the bench at the start of moves.
4. **Trajectory** (`safety.py:196` `plan_trajectory`): Ruckig, rest-to-rest, phase-synchronised.
   - Limits: 20 deg/s² acceleration and 80 deg/s³ jerk. Speed is 20 deg/s, or 5 deg/s when any sample enters the slow zone.
   - The trajectory is sampled at 1/25 s, and **every sample is re-checked** through `plan_move` and `_check_floor`.
5. **Executor** (`executor.py:45` `play`), at 25 Hz: read → `check_runtime` → `arm.send`.
   - `settle` (`executor.py:109`) then holds `goal + lead` with bounded integral action: gain 1.5/s, trim ≤0.75 deg/s, |lead| ≤ 2°, tolerance 0.5°, 0.4 s stable window, 4 s timeout.
   - Every tick's trim is again planned through the envelope.
   - The README reports the residual went from ~1.3° to ~0.2°.
6. **Driver** (`arms/metal.py`):
   - LeRobot's `max_relative_target`, the "lead cap", is bench-tuned per joint (`metal.py:39`). The code explains that *"torque under P control is kp × lead"*: shoulder_lift needed 5° to hold at long reach, ~~and a gripper lead of 30° or more stalls the motor into a rotor-overtemp fault.~~ [corrected: the code comment at `metal.py:36-38` says "the small gripper motor trips its rotor-overtemp fault above ~8 deg of sustained stall", and commit `e179e38` says the same. The "30° or more … within seconds (71-73 °C spike)" figure comes from `docs/OPERATING.md` §0. The repo is inconsistent here.]
   - `velocity_feedforward=False` and Kd is capped at 5 (`metal.py:137-147`). The reason: the MIT wire range silently clamps Kd at 5. `diagnostics/gain-encoding.json` shows that the configured Kd values of 11/6.5 encoded to exactly 5.
   - The Damiao status nibble is captured by monkey-patching `bus._process_response` (`metal.py:219`).

### 3.5 Safety envelope (`safety.py`)

`SafetyConfig` (`safety.py:68`) keeps every number in one place: 20 / 5 deg/s, slow zone 0.0762 m (3 in), floor margin 0.010 m, excursion 35°, 70 °C, 0.5° limit inset, 20 deg/s², 80 deg/s³.

- **Table-touch ritual** (`calibration.py:53`). The operator rests the gripper tip on the table. Two FK samples taken 0.3 s apart must agree within 3 mm, with up to 3 attempts, or the harness *"refuses to run"*. `watch_for_touch` (`calibration.py:107`) is a hands-free version: it waits for more than 3 cm of motion and then 5 s of stillness within 3 mm, so an agent without a keyboard can run it.
- **Hard floor.** Any waypoint whose lowest monitored point is within 10 mm of the floor rejects the *whole* move before anything is sent (`_check_floor`, `safety.py:301`).
- **Recovery rule** (`_floor_limit`, `safety.py:293`). If the arm is already inside the margin, every waypoint must stay at least at the current clearance (*"not one millimetre lower: no slack, or it ratchets"*). This rule was added because the envelope deadlocked on the bench after gravity sag.
- **Slow zone.** Both ends of each step are zone-tested.
- **Gripper is "not geometry".** It is exempt from the excursion cap and the slow zone.
- **Excursion cap.** The docstring gives the reason: *"Speed limits bound deg/s; this bounds distance travelled between looks at the camera, which open-loop execution otherwise leaves unbounded."*
- **Runtime vetoes** (`check_runtime`, `safety.py:320`): non-finite feedback, latched driver faults, and motors over 70 °C. A `SafetyAbort` from over-temperature calls `Controller.relieve()`, which zeroes P-torque on the hot joint (`operator.py:92`).
- **What is not modelled:** obstacles, containers, self-collision, workspace boxes, the jaw width on Metal, and any external force or E-stop. The server is **single-threaded** (`operator.py:393-437`, `listen(1)`), so **no "stop" command can be processed while a move is executing.** The only mid-move aborts are the runtime vetoes.

### 3.6 Grasp verification and error recovery

`settle` treats a gripper that stops more than 0.5° short of target for 0.6 s as *stalled*:
- while closing, that means "holding";
- while opening, it means "blocked".

`GotoReport.summary` (`control.py:93`) then adds one of three messages:
- `holding something (grip torque X N·m)`;
- `… grip torque is only X N·m: the jaws are probably resting ON the object … Do not lift; open, descend lower, close again` when torque is below 0.4 N·m;
- `OBJECT LOST`, when on a later move the jaws close more than 5° past the contact angle.

A stalled gripper keeps its closed goal as the "squeeze" through later moves. An earlier version relaxed it and *"dropped four grasps in a row"*.

Other recovery paths:
- `clear-faults` sends the Damiao clear-error (0xFB) and enable (0xFC) commands (`metal.py:233`).
- `resume.py` re-adopts a monitored, stationary, powered hold from the JSONL log without sending a command. The log must be under 5 minutes old, with matching floor and joints within 4°, velocity under 2 deg/s and grip torque at least 0.4 N·m.
- Rejections never end the session. Internal exceptions are returned with a traceback (`operator.py:426`).

**Known bug** (open PR #1 by `imwylin`): after a runtime abort mid-play or mid-settle, `relieve()` re-sends a stale `last_command` to unrelated joints. I reproduced it by running the PR's test file against `main`: **3 failed, 2 passed** (for example, a joint jumping 1.566° → 1.0°).

### 3.7 Logging

`EpisodeLog` writes append-only JSONL (`logs/episode-YYYYmmdd-HHMMSS.jsonl`). The event types are:
- `floor_set`
- `move` (targets, legs, steps, ~~residual~~ [corrected: no residual field; it logs `note`, `armed`, `duration_s`, `goal`, `measured`, `counter_dip_deg/joint`], stall, torque, `object_lost`) (`control.py:374-389`); other event types not listed here: `clear_faults`, `inspect`, `hold_summary`, `resume_hold`
- `observe` (frame paths and state)
- `command_error`, `safety_abort`, `relieve`
- `motion_sample` on every tick: measured/commanded/goal joints, efforts, velocities, FK tip, and the driver's `last_applied` with timestamps and gains

`scripts/plot_motion_trace.py` and `compare_motion.py` plot these, including 2–6 Hz tracking-error RMS. **The agent's reasoning and the images it actually looked at are not linked into this log.** They live in the Claude Code or Codex transcript.

## 4. The "prompt": operating docs as policy context

There is no system prompt in the current code. The context chain is:

1. `CLAUDE.md` / `AGENTS.md`: *"There is no model API in the loop: **you are the policy.**"*, *"never write to the CAN bus directly"*, and *"Do not commit, push, or open a PR without the user's explicit confirmation."*
2. `.claude/skills/metal-arm-pick-place/SKILL.md`:
   - The skill description triggers on *"connect to, calibrate, or move the metal arm"*.
   - Rules of thumb: *"Hover at 6 cm clearance, pitch -80, jaws at 112 before descending. Grasp at 20-25 mm clearance around the object's middle … Lift to ≥ 12 cm clearance before panning with a load … A rejection is information: read it, aim higher or lift first."*
   - "Never" list: bypass the envelope, disable torque on a held arm, or run more than one `serve`.
3. `docs/OPERATING.md` (about 240 lines) is the real policy prior.
   - The loop: *"Each step is `op <command>` → read the reply → **open the three JPEGs** → decide the next command."*
   - *"Say what you see and what you are about to do before each motion."*
   - *"Plan in those numbers, not in joint angles."*
   - A ~~7-step~~ [corrected: 8-step, numbered 0–7] pick-and-place recipe. Step 0 is *"Map the scene BEFORE grasping … hover at ~15 cm over the destination … read the wrist frame … Do not estimate it from the side cameras"*, because *"a plug at 6 cm looked 'inside' the case and landed 4 cm outside it"*.
   - §5, *"Things that already went wrong (do not repeat)"*: 10 entries. Examples: grid-searching poses took 40 s each; one process per command paid a 2 s CAN handshake; jaws at 84° shoved a 4 cm block 3 cm; ~~relative `nudge up` from a mis-remembered height put the jaws 5 cm too high twice;~~ [corrected: this one is in §3, after step 7, not in the §5 list. The text is "Relative `nudge up=-0.05` from a mis-remembered height put the jaws 5 cm too high twice".] swinging with a load at 5 cm clearance moved the case.

This is effectively a **hand-curated episodic memory** passed to the next agent as text.

## 5. The removed API loop (commit `50ec3bd`, `src/metal_arm_harness/agent.py`)

This is directly relevant to an Opus-backbone design. It ran `metal-arm-harness run "pick up the blue cube" [--armed]`.

**Parameters:**
- `DEFAULT_MODEL = "claude-opus-5"`
- `AgentConfig(max_llm_calls=40, max_tokens=1500)`
- no `thinking` setting, no `output_config.effort`, no temperature, no caching
- `client.messages.create(model, max_tokens, system, tools, messages)` once per decision
- one `tool_use` per turn; if there was none, it appended the user message "Respond with exactly one tool call."

**System prompt, verbatim core:**

> You are the control policy for a physical robot arm named "{name}". An operator has given you a task and is watching. You act ONLY by calling tools, exactly one per turn. After every call you receive the arm's joint positions in degrees, your clearance above the table when it is measured, and a fresh image from each camera. Safety rules, enforced by the harness (violations return an error you can correct, they never crash the run): {safety} Every move_joints call must include a `note`: one or two sentences on what you observe and why you chose this motion … You have a budget of {budget} model calls. When the task is complete call `done`; if you are stuck call `give_up`. Both take optional `hindsight`: what you wish you had known from the start, for the next attempt.

`{safety}` was filled by `SafetyEnvelope.describe()`. `{notes}` was filled with `METAL_NOTES` (`arms/metal.py:69`) [clarification: line 69 is the HEAD location. In `50ec3bd` it was `arms/metal.py:39`. The text is identical and still ships at HEAD as `ArmInfo.notes`, but nothing reads it any more], which told the model: *"Which Cartesian direction 'positive' maps to for each joint has not been verified against this build, so spend your first turns probing: one small motion on a single joint, look at the camera, note which way it moved."*

**Tools:**
- `move_joints{targets: {joint: deg}, note: str}`, both required
- `look{}`
- `done{summary, hindsight?}`
- `give_up{reason, hindsight?}`

Rejections came back as `tool_result` with `is_error: true` and the text `"move rejected: …"`.

**Observations:** a text block (`joints (deg): …`, `clearance_m above table: …`), then for each camera `"camera '<name>':"` followed by a **full-resolution PNG** in base64.

**Tests:** a scripted end-to-end test using `httpx.MockTransport` (floor rejection → recovery → done) ran with no network.

**My critique (inferred, not observed in logs):**
- Opus 5 runs **adaptive thinking by default**, so `max_tokens=1500` invites truncated turns with no `tool_use`.
- `stop_reason` (`max_tokens`, `refusal`) was never inspected.
- All past images stayed in history. At 640×480 that is about 400 image tokens per frame, roughly 1.2k per step for 3 cameras. Without pruning or caching, cumulative input grows quadratically: about 1M tokens over 40 calls, about $5 at $5/MTok. This is my estimate.
- Joint-space control plus "probe the joint directions" wastes calls.

The replacement design's commit message cites latency ("~0.1 s per command instead of a CAN handshake and camera warm-up per step"). Its real gains were Cartesian primitives, IK, and the curated operating manual. The commit gives no explicit reason for dropping the API, beyond "any agent that reads docs/OPERATING.md … can run it".

## 6. Results and bench evidence (all from repo notes, 2026-09-05; N = 1 bench, no trial statistics)

**Placements** (`diagnostics/placement-blue-left-20260905.md`, `repeat-blue-left-20260905.md`). Commanded successful grasps, all via `op tip`:

| Block | Pickup XYZ (m) | Pitch | Roll | Grip goal | Release XYZ | Notes |
|---|---|---|---|---|---|---|
| Blue | (0.295, 0.000, 0.024) | −80 | −20 | 10→16 | (0.445, 0.105, 0.150) | |
| Green | (0.174, −0.098, 0.026) | −80 | −35 | 15 | (0.415, −0.270, 0.150) | |
| Yellow | (0.309, −0.174, 0.024) | −75 | 40 | 14 | (0.425, −0.270, 0.150) | "required several retries"; earlier grasps slipped on verification lifts |

Repeat run: blue, green and yellow all succeeded at z = 0.028.

**Human interventions recorded:**
- *"Diagonal grasps pushed blue out. At the user's direction, wrist roll was changed to -55 degrees."*
- The user identified the true table height from frame 043 (z ≈ 0.011 m vs the saved floor of −0.0067 m), ~~so the saved ritual floor was wrong by about 18 mm for that session~~ [corrected: the 17.7 mm gap is real, but "wrong" is my inference. The source only says the old floor "is not a valid basis for descending below this user-specified limit" and that the running server's floor "was not changed". This bullet, the −55° roll bullet and the "right cup shifted" bullet all come from `placement-20260905.md` (episode `183234`). That was an earlier session with a different mapping (green-left; yellow and blue into the right cup). They are not from the blue-left session in the table above].
- *"The right cup shifted during an earlier failed approach."*
- Moves at z = 0.021/0.022 were rejected by the envelope, and the agent used z = 0.024.

**Wall clock:** the recorded part 2 ~~(green + yellow, including retries and pauses)~~ [corrected: this is the *repeat* run, `repeat-blue-left-20260905.md`. The source says only that part 2 "contains complete green and yellow placements"; it does not mention retries or pauses. Part 1 (~101.7 s) stopped during a pause and did NOT capture the blue placement] was **982.8 s** (4914 frames at 5 fps, 1920×1080).

**Motion quality.** Ruckig plus removing the velocity feedforward, on the same tool target, before vs after (`ruckig-hardware-results.md`, `ruckig-repeat-comparison.json`):

| Metric | Before | After |
|---|---|---|
| Shoulder 2–6 Hz tracking-error RMS | 0.225° | 0.030° |
| Elbow 2–6 Hz RMS | 0.712° | 0.040° |
| Shoulder overshoot | 2.441° | 0.952° |
| Final 5 s hold, max joint range | — | 0.011° |

The authors say plainly that this is not a controlled A/B test, and that 25 Hz encoder sampling cannot rule out higher-frequency vibration.

**Thermal event:** the gripper reached 72 °C and the harness vetoed further motion. The squeeze was then reduced from about 2.80 to about 1.16 N·m.

**Maker Arm:** not commissioned. On real hardware only read-only `inspect` works, and every `MECH_POS` query timed out (`docs/MAKER.md`).

**My sim measurements** (`--backend sim`, 20 deg/s):
- `op status` round trip: 0.23 s wall-clock, mostly Python start-up.
- 3 cm nudge: 2.2 s.
- Gripper close by 9.5°: 2.1 s.
- **Gripper open 0.5→112°: 7.4 s.** Jaw moves also run through Ruckig at 20 deg/s.
- Pan 0→90°: 3 legs, 9.7 s.

LLM think time comes on top. No API cost is incurred because the agent uses a subscription CLI.

## 7. Assessment

**Strengths**
- The safety layer is carefully reasoned, tested and auditable:
  - pure-logic envelope with 25 safety tests and 97 tests in total
  - unarmed mode with ~~an identical code path~~ [corrected: an *almost* identical code path. When unarmed, `settle` exits on its first tick (`executor.py:185`), `_origin` uses the measured pose (`control.py:397`), and chunked moves stop after the first leg (`control.py:328`). A dry run therefore never validates the later legs of a large move]
  - every Ruckig sample re-validated
  - defence in depth: envelope → driver lead cap → motor
  - opt-in torque
  - a holding arm keeps holding at session end
- The measured-floor ritual and the recovery-only-upward rule are simple, practical fixes for real failure modes (gravity sag deadlock, an unknown table offset).
- Error text is designed for an LLM reader. It explains the cause and the fix ("lift first", "aim higher", "open, descend lower, close again").
- Proprioceptive grasp semantics (stall, torque, slip) give the LLM a reliable signal that images alone do not.
- Low-level control hygiene for a cheap P-controlled arm: commanded base, no-dip origin, integral settle, zero velocity feedforward, honest Kd. The measurements are documented.
- An honest record of failures and measurement limits.

**Weaknesses**
- No perception stack or camera calibration. Metric targets come from the LLM eyeballing views, which the authors themselves call misleading (a 4 cm error). The depth camera is not integrated.
- The world model is a single plane. Containers get knocked over, and there is no keep-out geometry or self-collision checking.
- The single-threaded server cannot interrupt a move. There is no software E-stop or watchdog; the operator is expected to keep "a hand on the power switch".
- No evaluation: no trial counts, success rates or baselines. The best results needed human corrections.
- Orientation control is limited to tip pitch plus manual roll; IK is 4-DoF.
- Per-step latency is multiple seconds (Ruckig at 20 deg/s, 0.3 s camera wait, slow jaws).
- Reasoning traces are not joined with the robot log.
- Known stale-command bug after aborts (PR #1).
- Bench-specific hard-coding (camera names, ports) in the docs. Maturity is alpha: 5 days of commits, one bench, Maker unfinished.

**Genuinely new vs repackaged.**
- Repackaged: LeRobot drivers, Pinocchio FK, DLS IK, Ruckig, P-control trims.
- Also repackaged at the policy level: LLM-as-policy with bounded primitives is long established (Code-as-Policies etc.), and inspect-robots already had approvers. The repo's own predecessor was a plugin for it.
- Fresh or useful:
  1. the "**coding agent CLI + skill + operating manual is the policy**" packaging for a physical arm, with no model SDK in the robot code;
  2. the touch-calibrated floor plus the recovery rule plus slow zone, as a minimal LLM-safe envelope;
  3. the decision to tell the LLM about grasp state through text;
  4. evidence that the author abandoned a raw-joint Opus API loop within hours in favour of Cartesian primitives and curated procedural memory.

**What Ilia's Opus-backbone harness should borrow**
1. A pure, unit-testable envelope between the LLM and the bus that checks **every interpolated waypoint**. Return rejections as `tool_result` with `is_error: true`, phrased as an instruction.
2. A table-touch (or fiducial) measured floor, a recovery-only-upward rule, and a slow zone tested at both ends of each step.
3. A **per-call excursion cap** to bound open-loop travel between observations.
4. Cartesian tools such as `tip(x,y,z,pitch)` and `nudge(forward,left,up)` in a heading frame, IK that fails loudly, and replies in task-space numbers (clearance in mm, tip xyz).
5. Grasp state from proprioception (stall, torque, slip) inside every tool result.
6. A persistent single bus owner, a sim backend, an unarmed dry-run mode, and scripted-model end-to-end tests (MockTransport).
7. A "mistakes already made" operating document: version it and inject it (or have the model write `hindsight` into it automatically).

**What to avoid or fix**
1. Raw joint-space tools and "probe the joint directions".
2. Keeping every image in history. Downscale, keep the last K frames, use prompt caching or context editing, and set `max_tokens` with thinking in mind. Opus 5 thinks by default; Opus 5.5 cannot disable thinking and rejects forced `tool_choice`.
3. A blocking command server. Add an async stop/E-stop channel, a heartbeat watchdog and a deadman.
4. Relying on the LLM for metric 3D. Add calibrated extrinsics, depth, and a detector or segmenter that returns object poses or point clouds. Give the LLM pixel→3D or crop/zoom tools, and keep the LLM for task-level choices.
5. Plane-only collision. Add declared keep-out boxes or a depth occupancy map.
6. Anecdotal evaluation. Define tasks, trial counts, intervention counts and time-to-success, and log the model transcript and the exact images per decision, joined to the robot JSONL.
7. Slow jaw moves and post-move-only frames. Consider streaming frames during motion, or a light learned or servo action head for the last-centimetre alignment the LLM currently does by trial and error (roll, centring, depth).

## Sources

- https://github.com/makermods-robotics/metal-arm-harness (README, CLAUDE.md, AGENTS.md, docs/OPERATING.md, docs/MAKER.md, docs/DEPTH_CAMERA.md, diagnostics/*, src/*, tests/*; commits 50ec3bd, e179e38, d37e8e0, febaba2, 1f3c6df)
- https://github.com/makermods-robotics/metal-arm-harness/pull/1 , https://github.com/makermods-robotics/metal-arm-harness/pull/2
- https://github.com/makermods-robotics/makermods-inspect-robots/tree/arm/metal-arm/plugins/inspect-robots-metal
- https://github.com/robocurve/inspect-robots
- https://github.com/makermods-robotics/lerobot/tree/arm/makermods-metal
- https://github.com/makermods-robotics/metal-python-ros (URDF source per PROVENANCE)
- https://github.com/makermods-robotics/makermodslab
- https://github.com/andlyu/blupe-remote-yam , https://github.com/makermods-robotics/blupe-remote-makerarm
- https://www.makermods.ai/ , https://www.makermods.ai/metal-arm.html , https://www.makermods.ai/maker-arm.html
- https://x.com/IsaacSin12/status/2100833538224668699 (read via cdn.syndication.twimg.com; x.com itself returned HTTP 402)
- https://github.com/Frank-ZY-Dou/awesome-jev
- https://www.linkedin.com/in/isaac-sin-43389629a/ (search snippet only; not opened)
- https://github.com/pantor/ruckig

## Verification (fact-check pass)

Fact-checked on 2026-10-01 against these primary sources:
- the local clone at HEAD `1f3c6df`, fetched and identical to `origin/main`;
- the GitHub API (repo, PRs, forks, org, user);
- the removed code in `git show 50ec3bd:…`;
- the LeRobot fork pinned in `uv.lock` (`85a3000d…`);
- the `makermods-inspect-robots` branch `arm/metal-arm`;
- makermods.ai pages and robocurve.org;
- the tweet, read through `cdn.syndication.twimg.com`;
- `awesome-jev`;
- the bundled Anthropic `claude-api` reference (model table cached 2026-06-24).

I also re-ran the test suite and a sim session in a separate venv (`/tmp/mahv`, CPU torch, LeRobot fork installed `--no-deps`).

### Confirmed (seen in a primary source)

**Repo metadata**
- 15 stars, 4 forks, `description: null`, language Python, 1 open PR. PR #1 is by `imwylin`; PR #2 is merged.
- 5 commits, 2026-09-03 17:28 −0700 (`50ec3bd`) to 2026-09-08 (`1f3c6df`), all by Isaac Sin. His GitHub bio reads "CTO and Co-Founder of MakerMods".
- MIT appears only in `pyproject.toml`. There is no LICENSE file and the GitHub license field is null.
- `50ec3bd` → `e179e38` is 8 h 15 min ("about 8 hours" is correct).

**Tests and sim session**
- `pytest tests`: **97 passed** (125 s on my machine vs 142 s claimed; timing is machine-dependent). 25 of the tests are in `test_safety.py`.
- PR #1's `tests/test_abort_commands.py` (at `1dfa616`) run against main: **3 failed, 2 passed**. The failure shows 1.566 → 1.0 and 3.060 → 3.0. The PR body itself says "3 failed, 1 passed" for the 4 original regressions; the 5th test is a new one.
- The sim session reproduced the note's numbers exactly:
  - `op status` 0.20 s wall
  - `nudge forward=0.02 up=-0.03` → "1 leg, 45 steps, 2.2s", `tip (0.178, -0.000, 0.330)`, `ik … (2 iterations)`
  - gripper 10→0.5: 2.1 s
  - open 0.5→112: 7.4 s (171 steps)
  - `goto shoulder_pan=90`: 3 legs, 208 steps, 9.7 s

**Removed API loop (`git show 50ec3bd:src/metal_arm_harness/agent.py`)**
- `DEFAULT_MODEL = "claude-opus-5"` (line 33).
- `AgentConfig(max_llm_calls=40, max_tokens=1500)` (116-118).
- `messages.create(model, max_tokens, system, tools, messages)` (196-202). No thinking, effort, temperature or cache settings.
- 4 tools with the schemas as stated (58-111).
- "Respond with exactly one tool call." (210).
- `"move rejected: …"` with `is_error` (286, 316).
- Images are PNG base64 after a `"camera '<name>':"` text block (338-349).
- The system-prompt quote is verbatim (35-55).
- Test `tests/test_agent_e2e.py` uses `httpx.MockTransport`, falling back to `httpx2` for newer SDKs.

**Code claims (file:line as cited)**
- CLI: `cli.py:173/208/217/230/298`.
- Operator: `operator.py:67/92/281/393-437/410/426`.
- Control: `control.py:93/182/201/257/392`.
- Safety: `safety.py:68/196/293/301/320`, including every `SafetyConfig` number.
- Executor: `executor.py:45/109`, including every settle constant.
- IK: `ik.py:45/103`, including all IK parameters.
- Kinematics: `kinematics.py:27/67`, including the identity mapping.
- Camera: `camera.py:17-18/78/98/179`.
- Calibration: `calibration.py:53/107`.
- Resume: the `resume.py` thresholds.
- Metal driver: `metal.py:39/137-147/175/219/233`.

**LeRobot fork (`config_metal_follower.py` @ `85a3000`)**
- 7 Damiao motors; CAN IDs 0x01–0x07 (replies on 0x11–0x17); classic CAN at 1 Mbps; MIT position mode.
- Joint limits exactly as listed in §1.
- Gains (Kp, Kd):

  | Joint | Kp | Kd |
  |---|---|---|
  | shoulder_pan | 160 | 6.5 |
  | shoulder_lift | 390 | 11 |
  | elbow_flex | 320 | 11 |
  | wrist_flex | 160 | 5 |
  | wrist_yaw | 20 | 0.6 |
  | wrist_roll | 20 | 0.6 |
  | gripper | 20 | 0.6 |

- `gain-encoding.json` shows Kd 6.5/11/11 encoded to 5.

**Diagnostics**
- Placement poses table and the repeat-run z = 0.028.
- Ruckig before/after numbers (`ruckig-repeat-comparison.json`).
- 72 °C veto, then 2.80 → 1.16 N·m.
- Depth camera: 640×400 @ 15 fps, 92.4 % valid pixels, 321–671 mm.
- Maker: `MECH_POS` timeouts.

**Docs**
- `CLAUDE.md` / `AGENTS.md` / `SKILL.md` / `OPERATING.md` quotes are verbatim.
- §5 of `OPERATING.md` does have 10 entries.

**Lineage**
- `makermods-inspect-robots@arm/metal-arm` commits `c2c77d6` and `fdd9f31` (2026-09-03, Isaac Sin).
- The plugin README runs `--policy metal_arm_agent -P model=claude-opus-5`.
- `ArmHealthApprover` (`_safety.py:35`) and `TableProximityApprover` (`_safety.py:78`).
- `makermodslab` is a fork of `huggingface/leLab`.

**Website (makermods.ai)**
- Metal: $2,499, ships September 2026 (+$249 shipping), 7 DOF, 3 kg payload, 612.5 mm radius, ±0.1 mm, 4.2 kg, CNC aluminium alloy + resin, 24 V DC, CAN via XT30 2+2, 180 °/s (J1–J3) / 220 °/s (J4–J6), 0–95 mm gripper stroke, ROS1 Noetic / ROS2 Humble, MoveIt, ACT and pi0/pi0.5.
- Maker: $999 kit / $1,199 assembled, ships October, RS02 at shoulder pitch and elbow, RS00 elsewhere, 48 V, 1.5 kg payload.

**Other links**
- Tweet `2100833538224668699`: 2026-09-18, `@IsaacSin12`. The text confirms Jev, MuJoCo, JSON state, and bounded typed actions (hover, descend, grasp, lift, place) with confidence.
- makermodslab trailer count: 15.
- robocurve.org lists "GPT‑6 Astra on robotic manipulation".

**Claude API facts in the critique (per the `claude-api` reference)**
- `claude-opus-5` runs adaptive thinking when `thinking` is omitted.
- Opus 5 costs $5/MTok input.
- On `claude-opus-5-5`, `{type:"disabled"}` returns 400 at every effort level, and forced `tool_choice` `any`/`tool` returns 400.

### Corrections (also marked inline)
1. Perception "640×480": this is true only for the OpenCV path. The September bench used an HTTP JPEG camera service with 2 cameras, likely 1920×1080 (inferred from `scripts/record_camera_feeds.py` and the recording metadata).
2. BluPe: the Claude Code / "Opus through your Claude subscription" option is only in upstream `andlyu/blupe-remote-yam`. MakerMods' fork has only `run-codex.sh` (Codex / `gpt-6-astra`).
3. "8 PR branches named codex/…" is counted over all 167 makermodslab PRs, not over the last 100 commits.
4. Gripper "30° or more stalls" is attributed to `metal.py:39`. The code comment and commit `e179e38` say "~8 deg of sustained stall"; the 30° figure is from `OPERATING.md` §0.
5. The `move` log event has no `residual` field. The event list also omits `clear_faults`, `inspect`, `hold_summary` and `resume_hold`.
6. "7-step recipe" → 8 steps, numbered 0–7.
7. The "nudge … 5 cm too high twice" item is in `OPERATING.md` §3, not in the §5 list.
8. "Saved floor wrong by ~18 mm" is an interpretation. That bullet, the −55° roll bullet and the cup-shift bullet are from `placement-20260905.md` (an earlier session with a different task mapping), not from the blue-left table session.
9. The 982.8 s recording is the *repeat* run. The source does not say "including retries and pauses", and part 1 missed the blue placement.
10. "Unarmed … identical code path" → almost identical. Unarmed mode skips settle, uses the measured origin, and plays only the first leg of chunked moves.

Nuance, not counted: legs are capped at 34° from the *measured* pose but planned from the last command (up to 4° away), so the first leg can be rejected at the 35° cap. I reproduced this in sim; details are inline in §3.4.

### Unverifiable or only partly verified
- **"9 decisions at ~150 ms each"**: it is not in the tweet's visible `text` field, which is a truncated `note_tweet`. It appears in `awesome-jev` (README lines 56 and 122). I could not see the full note-tweet text.
- **"Founders are Ryan Chan and Isaac Sin"**: Isaac is confirmed by his bio. The site footer reads only "Ryan and Isaac". `MakerMods-XLeRobot/README.md` names "Ryan Chan" as co-maintainer. I found no primary page calling Ryan a co-founder.
- **Token and cost estimate (~400 image tokens per 640×480 frame; ~1M tokens / ~$5 per 40-call episode)**: it uses the (w·h)/750 rule (≈410). I did not check whether Opus 5's high-res vision tier changes the per-pixel rate (it is documented only as "up to 4784 tokens per image" at 2576 px). Still an estimate; check it with `count_tokens`.
- **Which agent drove the September sessions**: still UNVERIFIED. The diagnostics are written in an agent's voice, but no transcript is shipped. Raw logs, frames and videos are git-ignored and stay "on the original bench machine" (`diagnostics/README.md`).
- **"x.com returned HTTP 402"**: not re-checked.
- **"No HN/Reddit threads"**: HN Algolia returns 0 hits for `metal-arm-harness`. I did not check Reddit.

### Added missed details

**A. The API-era prompt described the arm wrongly.**
`METAL_NOTES` (`50ec3bd:arms/metal.py:39-61`; same text at HEAD `metal.py:69-91`) was injected as `{notes}`. It told Opus that:
- "The zero pose (all joints 0) is the arm standing fully upright";
- `elbow_flex` "0 is straight";
- the jaws are widest "near 116".

The harness's own URDF FK puts the zero-pose tip at (0.159, 0.000, 0.189) m with pitch −0.9°, which is folded with the forearm horizontal (I computed this with `MetalKinematics.tool_pose([0]*7)`). `OPERATING.md` §0 later says the same: "At the zero pose the arm is folded with the forearm horizontal … tool tip at about (0.16, 0, 0.19) m", and uses 112 as "safe fully open".

So the raw-joint Opus loop was primed with a contradictory kinematic description, on top of being told to "spend your first turns probing" joint signs. This probably contributed to the API loop being abandoned (inference).

**B. Hidden failure modes of the removed loop (inferred from code).**
- It handles only the first `tool_use` block (`agent.py:207`) and never sets `disable_parallel_tool_use`. Any parallel tool call would leave unanswered `tool_use` ids, and the next request would fail with a 400.
- It never checks `stop_reason`.
- With adaptive thinking on by default and `max_tokens=1500`, a turn cut off at `max_tokens` with no `tool_use` consumes budget, and the loop answers with a plain-text nudge.
- Every observation re-sends lossless PNGs for every camera, and the history is never pruned. Over 40 turns, base64 payload size (not only token cost) could approach the request-size limit (inference; not measured).
- Motion used the linear `plan_move` stream with no settle, IK, Ruckig or grasp detection. The model saw only joints, `clearance_m` and images.

**C. "clearance" is not tip height.**
`clearance_mm` = min(elbow…wrist joint origins, tool tip) − floor (`kinematics.py:67`). In my sim run, `nudge … up=-0.03` *raised* clearance from 100 to 111 mm because the elbow, not the tip, was the lowest point. An LLM told to "plan in clearance numbers" can misread this.

The tool tip is a fixed 0.12 m point. Neither the jaw geometry nor a held object (which the docs say hangs ~4.5–5 cm below the tips) is part of any floor or collision check. A carried object can therefore be driven into the table or a container without any rejection.

The floor was measured at pitch −80°. `OPERATING.md` says shallower pitches are "slightly conservative".

**D. Bench settings differ from the defaults.**
The 2026-09-05 Ruckig hardware tests ran at **10 deg/s** ("Speed 10 deg/s on this server"), half the 20 deg/s default, so real step latency was higher than the sim figures.

**E. Bench bugs found and fixed during the placements.**
`placement-blue-left-20260905.md` records a settle bug. It could send a wrist_flex target slightly above the soft limit when the planner clamped and returned no waypoints, "producing repeated limit corrections and blocking subsequent trajectories". The fix clamps `target + proposed_lead` before planning (regression test `test_settle_sag_at_soft_limit_never_sends_unclamped_empty_path`).

The same note records "Joint-step rejections were handled by shorter intermediate IK poses without widening limits". This matches the chunking/origin issue above.

**F. The original "two pick-and-place runs" (2026-09-03/04) were a phone charger/plug into a case, not blocks.**
`OPERATING.md` evidence: "8° lifts the charger fine"; "the plug drops"; "Releasing from 6 cm let the plug tumble off the rim twice"; "Grasp slipped four times in run 2".

The manual's grasp-height advice is inconsistent: "clearance ≈ 20-25 mm" (step 4 and `SKILL.md`) vs "grip the charger's middle at 12-18 mm tip clearance" (§5).

Grip force is documented as gripper Kp 20 × lead cap 8° ≈ 2.8 N·m. This matches LeRobot's gripper Kp = 20 (20 × 0.1396 rad = 2.79).

**G. Doc vs code mismatches.**
- `OPERATING.md` §4 says "the session discards a few frames per observe already". There is no frame-discarding code. `_observe` does one `camera.read()` per camera (`operator.py:293-297`); only `CAP_PROP_BUFFERSIZE=1` and retries exist.
- The URDF sha256 is documented in `PROVENANCE.md` (and matches the file: `faac0ba6…`), but no code or test checks it.
- `ruckig-hardware-results.md` says "Hardware was tested on Linux", while `OPERATING.md` and `MAKER.md` use macOS `/dev/cu.usbmodem*` ports. The camera-name guard (`check_device_names`) is a silent no-op off macOS or without ffmpeg (`camera.py:182-186`).

**H. Process facts.**
- The `op` client blocks for up to 600 s per command (`operator.py:445`).
- `clear-faults` refuses unless the session is armed.
- `--diagnostics-only` accepts only `inspect` and `quit`.
- Resume sets `_follower._synced = True` to skip LeRobot's 1°/step startup re-sync, so a held grip is not dropped (`resume.py:66-70`).
- LeRobot's default `velocity_feedforward=True` (α 0.08, cap 120 °/s) is what the harness disables.
- `ruckig-hardware-results.md` reported "92 full regression tests passed; an additional resume test passed separately" at that time.

**I. Sim camera.**
The `sim` backend's only camera is a 384×384 cartoon top view of a planar 2-link sketch and a blue cube (`metal.py:351-411`). Its "elbow" is drawn from `positions[4]` (wrist_yaw). Sim runs therefore say nothing about real visual grounding.
