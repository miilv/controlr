# dimentary/llm-robotics-playground: deep dive

Repo: https://github.com/dimentary/llm-robotics-playground (MIT; 163 stars and 8 forks as of 2026-10-01; created 2026-09-17; last push 2026-09-24). I cloned it with full history to `research/repos/llm-robotics-playground` (7 commits), read all code, ran the smoke checks and validators, reran one controller, and exercised the LLM-facing pilot interface.

## 1. TL;DR

- **This is not a runtime LLM-control harness.** The four MuJoCo demos were made by using **GPT-6 Astra inside Codex as a coding agent**. It wrote the environments and the scripted controllers, and the author iterated with it. At runtime the controllers are plain Python that read **privileged simulator state** (object poses, cable node positions, contact forces). They make **no model calls and use no images**. The README says so: "The first experiments used GPT-6 Astra in Codex to build the environments and write the robot-control code. I guided the task setup and gave feedback along the way… Replay and validation here run locally without calling a model API."
- **Model details were not preserved.** Each original `recording.json` (removed in commit `e4f5979`) has `"reported_label": "GPT-6 Astra", "exact_snapshot": null, "settings": null`. Reasoning effort, temperature, tokens and cost are all **unknown**. No prompts are published. The author's own publication plan says: "Use unknown for unavailable model settings; do not reconstruct an exact original prompt from memory."
- The fifth demo (Baoding balls, added 2026-09-24) is a **PPO policy trained in Isaac Lab/PhysX**. The README does not say whether Astra wrote that trainer (**UNVERIFIED**).
- **The most reusable artifact for an Opus harness** is a small design pattern: "trusted firmware vs. untrusted pilot." Its concrete form is `experiments/robo-spider/pilot.py`, a JSON-lines interface. It exposes body velocity, Cartesian hand poses and grip, plus optional head and wrist PNGs. Commands are validated and applied transactionally, and each accepted command advances physics by a fixed 0.1 s. That interface was **never used for live model control** in the recorded runs.
- The repo's strongest point is **honest, integrity-checked reporting**: asset hashes, compiled-model equivalence, no welds or mocap, ink only from loaded contacts, disclosed edits and speeds, and "best-of-archive" labels. Its weak point is that every result is a single tuned episode with ground-truth state, hard-coded coordinates and human-guided stitching.

## 2. Author and context

- **Dmytro (Dima) Hrybov**, GitHub/X `@dimentary` (commits from `dimentary@users.noreply.github.com` and `d.hrybov@modelroom.ai`). His homepage hrybov.com says: "ML research engineer in San Francisco, building evals for physical-world AI. Backed by EF." He also writes: "I'm benchmarking frontier VLMs on physical-world tasks, using real robot episodes rebuilt as replay-verified sim environments." He was previously co-founder and CTO of ModelRoom (Apr–Aug 2026), worked at Google Kaggle Research (2025–26) and Promaton, and has a CV/medical-imaging background.
- His related X posts, fetched via the fxtwitter API:
  - 2026-09-08 01:52Z, dove ([status/2097141042214797801](https://x.com/dimentary/status/2097141042214797801), 109k views): "asked it to build a MuJoCo setup and write a controller to draw Picasso's dove using a robot arm and a five-fingered hand".
  - 2026-09-08 22:43Z, Fibonacci ([status/2097455860541009958](https://x.com/dimentary/status/2097455860541009958), 70.5k views): "Astra, can you write a Python script for the Fibonacci sequence / i mean physically". Its video has the same 73.07 s duration as the release asset `fibonacci-writing.mp4`, and the dove post's video matches `dove-drawing.mp4` (45.27 s).
  - 2026-09-10, "rope-driven hand" attempt to recreate 1X's tendon hand ([status/2097857980150763900](https://x.com/dimentary/status/2097857980150763900)). The post is now **unreachable** (fxtwitter/vxtwitter return 404). It is known only from the Awesome-Astra list, which says Astra "uses simplified mechanics and illustrative cable deformation rather than full tendon-transmission physics".
  - 2026-09-18, **Jev as a real-time policy** ([status/2101018760371171420](https://x.com/dimentary/status/2101018760371171420)): "it struggled at first, so i split each update into two calls: decide what to do next, then decide how to move the arm and gripper. Jev doesn't accept images, it gets simplified geometry and contacts as text here." Frames I extracted from the 20 s video show the following:
    - Overlay: "real-time policy · 3.2 updates/s · no images", "MuJoCo state → text · Jev 1.13 · 100 ms target filter · 1× playback".
    - Input: cube-pinch, pad-cube and wall-pinch offsets in cm, plus contact flags.
    - Output: intent probabilities (Approach/Grasp/Lift/Carry/Lower/Release/Withdraw/Finished) and "Four motor distributions": X/Y/Z ∈ {+,−,stay} "up to 2 cm per axis", and fingers ∈ {open, close, stay}.
    - ~~Cycle times are 269–364 ms.~~ [corrected: the per-update "cycle" readout varies far more widely. Sampling the video at 2 fps shows values from 186 ms (t=4.46 s) to 578 ms (t=2.59 s), including 481 ms at 0.84 s and 401 ms at 5.92 s. The header gives the nominal rate as "3.2 updates/s" (about 312 ms average).] **This code is not in the repo.**
  - 2026-09-22, **"robot drawing on the board" bench, GPT-6 Sol vs Astra** ([status/2102539444834402314](https://x.com/dimentary/status/2102539444834402314)): "Sol used ~25k output tokens vs ~22k for Astra… Sol's controller drew twice as fast… way cheaper to run (almost 5x)". The video panels show a G1 drawing *The Creation of Adam*: Sol produced 13 strokes in 119.8 s of simulation and Astra 17 strokes in 259.7 s. This reuses the Fibonacci G1 + whiteboard setup as a model-comparison bench. **Not in the repo.**
- **Third-party uptake:**
  - Zjwzcx/Awesome-Astra-Embodied-AI lists the Fibonacci demo as "Zero-shot Control → Deploy in Simulation" (Case 11). That label contradicts the repo's own disclosure (offline code authoring with privileged state).
  - YuanKJing/Jev-as-Policy is a public **reconstruction** of the Jev demo (Franka Panda, `jev-1.13.0` via TypeSafe `/v1/systemone`, MuJoCo 3.3.7). It cites this repo as its "public mechanism reference" and says it "does not claim to contain the original author's private prompt, scene, controller, or source code."
  - Frank-ZY-Dou/awesome-jev archives the Jev video.
  - The headphone task is "Inspired by Qineng Wang's rope-threading experiment" ([x.com/qineng_wang/status/2099893504658866561](https://x.com/qineng_wang/status/2099893504658866561), Dual-ALOHA, 646k views).
  - I found no HN or Reddit discussion of the repo itself (HN Algolia: 0 hits).

## 3. Repository evolution (git log)

| Commit | Date (PDT) | Change |
|---|---|---|
| `1944286` | 09-17 10:44 | ~~Docs only:~~ [corrected: docs plus scaffolding (`README.md`, `LICENSE`, `.gitignore`), including] `docs/publication-plan.md` and `docs/experiment-template.md`, which define the disclosure fields: model, role, observations, tools and actions, task brief, human guidance. |
| `a8b22f1` | 09-17 11:10 | Publishes four experiments (+122k lines including `strokes.json` and `task.json`), `check.py`, `fetch.py`, `recordings.json`, per-experiment `recording.json`. Release v0.1.0 follows at 11:15. |
| `e4f5979` | 09-17 12:07 | Removes `recording.json` files (model metadata) and trims `task.json` (initial cable nodes; a path to an unpublished `../wiring-repair/` experiment). README detail is shortened. |
| `867fd52`, `0172fc2` | 09-17 15:01–15:13 | Switch to uv; README framing changes ("I used Astra…" → "The first experiments used GPT-6 Astra in Codex…"). |
| `2c89ceb`, `5ffdf58` | 09-24 11:33–11:46 | Adds Baoding: PPO checkpoint, trace, replay, and a "distilled" Isaac Lab trainer. |

The earlier READMEs (at `a8b22f1`) were more explicit about method, so it is worth reading them in history. For example: "It uses simulator state, not live image interpretation"; "The recorded controller uses known world coordinates and simulator state; its available camera interface was not used as a live VLM policy"; "The published video was assembled from continuous accepted stages with retries and human-guided revisions."

## 4. Experiment catalogue

| Experiment | Robot / sim | LLM role | Observation used by controller | Action interface | Result (as recorded) |
|---|---|---|---|---|---|
| robo-spider | Original hexapod (3 DoF/leg) + 2× Kinova Gen3 + Robotiq 2F-85; MuJoCo 3.12, dt 2 ms, nu=34 | Astra authored env + controller (Codex), human feedback | Ground-truth body/hand/object poses, contacts | Firmware: drive (fwd/turn/height) + Cartesian hand pose + grip; FSM in `mission.py` | Both objects placed upright: 97.602 s sim, errors 1.9 / 2.5 mm, 6.42 m walked, no robot–env contacts (500 Hz audit) |
| headphone-untangling | ALOHA 2 (2× vx300s, 14 actuators, unmodified) + 282-segment self-colliding cable; scene dt 0.2 ms (CG); recorded phases at 0.25–1 ms, PGS 150 it | Authored code; "substantial human feedback" on layout and strategy | Ground-truth cable nodes, geom positions, contacts | Scripted phases: precomputed Cartesian IK paths → stock position servos | Both tangled regions opened; **not untangled** (2 overlaps remain); 33.149 s sim, 45 phases, stitched from 19 source runs with 8 recoveries |
| fibonacci-writing | Unitree G1 (floating base, nu=43), free friction-held marker; dt 1 ms | Authored code | Marker-tip site pose, board contact forces | `least_squares` IK on 7 right-arm joints + force-regulated depth | 52/52 strokes, 272.21 s, 4,803 marks, median/p95 tip error 0.72/6.07 mm, max tip force 12.9 N |
| dove-drawing | Kinova Gen3 + Shadow Hand (27 actuators), free pencil pre-grasped at ~30° | Authored code | Pencil-tip site pose, tip/paper contact force | Damped least squares (DLS) null-space IK at 20 ms waypoints + 0.15 N force regulation | 9/9 strokes, 173 s, 10,064 marks, median/p95/max error 0.149/0.549/4.63 mm, ≥2 hand contacts throughout |
| baoding-balls | Sharpa Wave 22-DoF hand, 2 balls (r=19 mm, 35 g); Isaac Sim 5.1 / Isaac Lab 0.54.2 / PhysX | None stated (RL); Astra role **UNVERIFIED** | 107-D proprio + ball state | Δ joint-position targets (scale 0.025 rad) at 60 Hz | Best of 592 episodes: 4.935 shared net turns in 12 s, max penetration 0.321 mm, side-risk 70.65% (flagged as a warning) |

### 4.1 robo-spider: the only LLM-facing runtime interface

Architecture, as named in the docstrings:

```
                    (never used live in recorded run)
  LLM / VLM pilot ──JSON line──▶ pilot.py ──validated cmd──▶ control.Firmware (trusted)
        ▲                          │  rollback on error           │ gait: analytic 2-link leg IK, tripod
        │                          │  50 × mj_step (0.1 s)        │ arms: DLS IK every 5 ticks, quintic+slerp
        └──────JSON obs (+3 PNG)───┘                              │ joint vel ≤0.65 rad/s, acc ≤1.5 rad/s²
  mission.Mission (scripted pilot used in the demo) ── same Firmware API, 10 Hz (run.py:41-42)
```

~~same Firmware API~~ [corrected: Mission calls the same `command_drive`/`command_hand` methods, but it also goes around the validated API. RELEASE writes `h["grip"] = 0` directly (mission.py:189-191). `grippers_ready` reads raw `d.ctrl` and `d.contact` (mission.py:70-84), and `settled` reads base `qvel` (mission.py:39-47). pilot.py never exposes any of these signals. Mission also gets no rollback.]

- `pilot.py:1-3` says: "JSON-lines high-level pilot interface; 0.1 s of physics per accepted command. Run with --camera for head and wrist images. Models should receive only this process's I/O, not Python execution access to the trusted simulation process."
- **Observation** (`pilot.py:25-49`): `time`, `body_position`, `body_quaternion_wxyz`, `hands.{left,right}.{position, rotation(3×3)}`. With `--camera`, it adds base64 PNGs from cameras `head`, `left_arm_wrist` and `right_arm_wrist` at **320×180** (`mujoco.Renderer(m, 180, 320)`, line 23), with geom group 3 hidden.
  - I ran it: each observation line is about 70 KB (head PNG ≈34 KB, wrists ≈17–19 KB each).
- **Command schema** (`pilot.py:56-71`, `control.py:57-99`): `{"drive":{"forward","turn","height"}, "hands":{side:{"position":[3],"rotation":[[3x3]],"grip":0..1,"frame":"world"|"body"}}}`. Any other key returns `{"error":"Only drive and hands are accepted"}`.
  - Validation and budgets: `forward ∈[-0.35,0.35]`, `turn ∈[-0.4,0.4]`, `height ∈[0.36,0.51]` (otherwise "Drive outside budget"). The rotation must be orthonormal with det>0.99, and the grip must be in [0,1].
  - On error, the drive and hand state are deep-copied back (transactional), and **simulation time does not advance**. I verified both behaviors.
- **Firmware motion generation** (`control.py`):
  - Hand moves get duration `max(1.2 s, ‖Δp‖/0.065 m/s, angle/0.4 rad/s)` with a quintic minimum-jerk profile and quaternion slerp (lines 91-93, 191-217).
  - IK is DLS with 100 iterations, λ=1e-4, step clip 0.12 rad, rotation weight 0.35, and stopping tolerances of 1 mm / 0.008 (lines 107-133).
  - On gripper opening, the firmware removes saturated overtravel while preserving current force (lines 229-244).
- **The demo "pilot" is a ~~14-stage~~ [corrected: 16-named-stage. 15 stages are reachable and were executed; `ALIGN_DESTINATION` (mission.py:172-175) is never entered. `stages.json` logs 14 transitions after the initial `APPROACH`.] finite state machine (FSM) with hard-coded world coordinates** (`mission.py:120-198`). Examples: `REACH` to `[1.15,0.12,0.81]`/`[1.14,0.69,0.82]` and `PLACE_LOWER` to `[-0.82,2.48,0.643]`.
- Stage transitions are gated by verification predicates with dwell times:
  - `confirmed(cond, dwell=0.2)` (l.30-37);
  - `hands_ready`: the trajectory has finished and the hand is within 12 mm and 0.12 Frobenius rotation error (l.49-68);
  - `grippers_ready`: actuator saturated and finger-object contact present or absent for 0.3 s (l.70-84).
- Validation (`validate.py:42`) requires each object to be on the table, not touching the robot, within 3 cm, and upright with cosine > 0.99. It also checks force-limit peaks and the 500 Hz contact audit (`run.py:46+`).

### 4.2 headphone-untangling: largest controller ~~(2,806 Python lines)~~ [corrected: 2,806 is the line count of every .py file in the experiment, including render.py 257, validate.py 325, verify_release.py 109 and viewer.py 65. The control and planning modules alone are about 2,050 lines: control 790, run 475, strategy 249, robot 195, environment 113, cord_scan 90, stem_scan 85, grip 53.]

- **Scene** (`task.json`): main cable 114 segments at 1.8 mm diameter, left lead 143, right lead 25 (both 1.3 mm), bend modulus 2 MPa, twist 0.7 MPa. There are two clusters (an overhand knot plus a threaded bight on the main cable and on the left earbud lead). The scene "assumptions" include "Friction and cable elasticity are provisional." `task.json` also encodes human strategy guidance:
  - `"first_action": "Open the middle cluster, reassess the second region, then regrasp…"`
  - `"motion_preferences": ["Approach the cable directly from outside with the right wrist", "Release and withdraw diagonally", …]`
- **Planning on ground truth.** `cord_scan.scan` (l.10) sweeps grasp hypotheses at a cable segment. For each, it runs `least_squares` IK, then closes the finger opening in 0.05 mm steps until both pads touch the allowed segments ±2 with no rigid penetration. `strategy.endpoint_analysis` computes over/under crossing order from the true cable nodes.
- **Execution** (`control.py`):
  - `Rollout.move` (l.281) precomputes IK on a 15 ms grid. It rejects the motion if the joint speed exceeds 4 rad/s: "Discontinuous or too-fast joint path in {label}; replan the wrist posture" (l.367). It also collision-checks a scratch copy of the state.
  - `control()` (l.218) adds a bounded integral term (±0.025) plus bias, damping and friction feedforward on top of the stock position servos.
  - `monitor()` (l.244) aborts on: an unplanned arm collision deeper than 0.5 mm, MuJoCo warnings, cable self-penetration over 0.2 mm, or grip slip over 4 mm.
  - `follow_and_pinch` (l.482) servoes toward the *ground-truth* segment at 50 Hz. It closes once the tracking error is below 1 mm and requires 0.08 s of sustained opposing contact.
  - Every phase writes a state checkpoint plus the controller integral state, "so a later rejected motion need not repeat good actions" (l.770-775).
- **The recording was assembled, not generated in one run.** The downloaded report `two_clusters_current.json` lists `sources`: 19 checkpointed runs (`v5_middle_open` … `v5_trunk_released`). It also lists `recoveries`: 8 failures, for example "No opposing contact on main_cable_095" (×3), "Discontinuous or too-fast joint path…" (×2), "Free end moved outside its grasp", and "Blocked cable tracking pose: ['world','right/left_finger_link']". The report records `wall_s = 6862` for 33.149 s of simulation.
- **Many recorded phases have no counterpart in the published code**: "capture the settled loop", "close the deep cable grip", "pinch the resting trunk at the checked pad height" and the whole trunk-lift sequence from 28.06 to 33.15 s. The published `run.py` is the final revision of some stages only, so the demo cannot be regenerated end to end from the repo.
- `verify_release.py` defines 10 strict checks for full untangling (no free-span crossings, cable on the mat, settled, and so on). The selected endpoint does not pass them. The video cuts the 4.23–5.96 s grasp-retry interval.

### 4.3 fibonacci-writing and dove-drawing: tool use with force control

- **Fibonacci.**
  - `strokes.py` hand-codes a single-stroke monospace alphabet. The text is `CODE = "a, b = 0, 1\nfor _ in range(10):\n    a, b = b, a+b; print(a)\n"`, laid out at 15 mm per column, 85 mm line pitch and 35 mm glyph scale, giving 52 strokes (`script_paths`, l.169). The early README showed a 4-line variant, a small doc drift.
  - `Writer.solve` (`writer.py:82`) runs bounded `least_squares` IK on a *virtual tool site*. `sync_tool` re-estimates that site from the observed free marker each solve, so the in-hand pose is measured from simulator state rather than assumed rigid.
  - Depth is force-regulated toward a 0.5 N filtered contact force (`depth += clip((0.5 − f̄)·3.5e-5, …)`, l.223/236). Progress requires ≥3 of 10 substeps with loaded contact; after 160 waits the controller raises "Unable to maintain ink contact" (l.252).
  - Safety aborts: base height below 0.70 m or grip slip above 4 cm (l.194-198). The marker starts already grasped; "pickup is outside the task".
- **Dove.**
  - Strokes come from a user-supplied raster via "Threshold, Zhang-Suen centerline thinning, junction stitching, light smoothing" (`strokes.json`), giving 9 strokes with 458/401/319/94/35/19/10/25/25 points.
  - `firmware.py` ("Fixed low-level interface. No IK, grasp planner, drawing path, or model calls.") also exposes a **JSON-lines server** with ops `observe | reset | set_targets` (l.144+). `set_targets` takes named actuator targets plus `duration_s ∈ [0.002, 2]`, with limit checks and quintic interpolation (l.44).
  - `observe()` returns joint angles and velocities, actuator forces, the tip position, the pencil pose and **per-contact normal forces** (l.112).
  - `controller.py` ("GPT-authored scripted controller. Task intelligence is outside fixed firmware.") does null-space DLS IK and raises if the IK residual exceeds 3 mm.
  - `PhysicalWriter.draw` (`writer.py:59`) follows the *observed* free pencil tip (gain 0.15, clipped ±0.15 mm per step) and regulates tip force to 0.15 N (z step `(f−0.15)·2.5e-5`, clipped ±10 µm). The recorded median loaded force was 0.11 N.

### 4.4 baoding-balls (RL, not LLM)

- `train.py`: a `DirectRLEnv` with sim dt 1/240 and decimation 4, giving 60 Hz control. Actions are 22-D with `targets += 0.025·clip(a)` (l.181).
- The 107-D observation is: normalized joint position, 0.1·joint velocity, targets, actions, ball position ×10, ball velocity, goals ×10, and phase.
- Reward (l.218-227): `20·Δprogress + 0.3·exp(−d/0.015)·held − 50·dropped + 0.2·held − 0.5·¬geometry − 0.2·¬supported − 0.2·risk − motion costs − 0.2·middle-finger side risk − 0.03·posture`.
- PPO setup (RSL-RL 3.1.2): 1,024 envs, 32 steps per env, [256,256,128] ELU actor and critic, lr 1e-4 fixed, entropy 0, std frozen at 0.35, and `BoundsPPO` adding `1e-4·((|μ|−1.1)+)²` (`bounds_ppo.py:322`).
- The checkpoint is from iteration 3320. The selected trace is seed 982210, trial 12, out of 592 records ("best-of-archive… not the average success rate").
- Inference (mine): 4.935 turns in 11.97 s is about 2.6 rad/s, roughly 2× the trainer's `GOAL_SPEED=1.2`. This is consistent with the README's warning that the cleaned trainer "is not expected to recreate checkpoint 3320".

## 5. My verification runs (Linux x86-64, Python 3.14, MuJoCo 3.12.0 via `uv sync --locked`)

- `check.py`: all 4 scenes pass startup and a short physics step (8.2 s wall). The README only claims macOS, but Linux works with `MUJOCO_GL=egl`.
- `fetch.py` downloads have SHA-256 values that match `recordings.json`. All saved-run validators pass for robo-spider, fibonacci, dove and baoding.
- **robo-spider full rerun (`run.py`, 48 s wall for ~96 s of simulation) did *not* reproduce the recording.**
  - The trajectory differs from the first sample (>1e-12 at t=0.002 s; >1 cm by t=22.4 s). The mission finished at 96.402 s instead of 97.602 s, with placement errors of 1.90 and 2.46 mm.
  - `validate.py` **failed** with "Robot contacted a table or barrier": `left_grip_left_pad / source_table`, 2 samples, 0.087 mm penetration at t=33.564 s.
  - So the README's "A fresh run reproduced the saved trajectory exactly" holds only on the author's platform (macOS Apple Silicon). The strict zero-contact criterion flips on ~~numerically trivial differences~~ [corrected: a divergence that is not round-off-sized. After the first 2 ms step, the leg knee and hip joints already differ by about 2.7e-4 rad (`leg_-1_2_knee` 2.73e-4, `leg_1_2_knee` 2.66e-4), and the difference grows to about 1e-3 rad by t=0.042 s. The cause is unknown: possibly platform-dependent contact-solver behaviour in the first footfall solve, or some initialization difference. The resulting table contact (0.087 mm) is tiny, but the trajectory difference is not.]
- `pilot.py --camera`: the command validation, error handling and no-time-advance-on-error behavior work as described.
- Headphone `validate.py --seconds .4`: all 7 reach checks pass with 0 collisions. Cable settling (scene dt 0.2 ms, CG solver, 100 iterations) took **148 s of wall time per 0.08 s of simulation**, about 1,850× slower than real time on one core of an 8-core x86 box. [fact-check: this number depends on machine load and is the high end. On the same 8-core host, the fact-checker's `validate.py --seconds .08` reported `wall_s` 90.0, about 1,125×. Raw `mj_step` from the initial state took 8.35 s per 0.02 s at scene settings (CG, 100 iterations, 0.2 ms), about 418×. At run.py's settings (PGS, 150 iterations, 1 ms, dense) it took 6.76 s per 0.05 s, about 135×.] The report's top-level `wall_s` is 6,862 s for 33.1 s of simulation (about 207×, at the coarser 0.25–1 ms PGS steps). I stopped the stage-1 `run.py` rerun for time, so it is **not verified**.

## 6. Assessment

**Strengths**
- The repo clearly separates three modes: "model-authored controller", "model-guided iteration" and "live model control". It discloses which one each demo used, which is unusual in this space (see `docs/experiment-template.md` in history).
- Its anti-cheat integrity checks matter whenever an LLM writes both the environment and the controller:
  - asset SHA-256 checks and compiled-field equality against upstream Menagerie (`robot.py:robot_check`);
  - `m.neq==0`/`nmocap==0` (no hidden welds or teleporting) [corrected scope: `neq==0` is asserted only for fibonacci and dove (`fibonacci-writing/validate.py:24`, `dove-drawing/validate.py:17`, `check.py:41`). robo-spider has `neq=6`, the upstream Robotiq 2F-85 linkage `connect`/`joint` equalities (scene.xml:679-686). headphone asserts `neq==2`, the upstream ALOHA finger equalities (`validate.py:125`, `verify_release.py:81`). `nmocap==0` holds in all four scenes.];
  - free-jointed tools held only by friction;
  - "ink" deposited only at measured loaded contact (>0.015 N);
  - actuator force-limit audits and contact audits every physics step.
- The trusted-firmware API is safety-first: budgets, transactional rollback, firmware-side smoothing and IK, and "the model only gets process I/O".
- Its runtime monitors produce **actionable natural-language failure strings**, plus per-phase checkpoints that allow resuming. That is the right substrate for an LLM retry loop.
- It shows that a frontier coding agent can author long, contact-rich MuJoCo controllers: a deformable self-colliding cable, a friction-held pencil with 0.15 mm median tracking, and a humanoid writing 52 strokes.

**Weaknesses and limits**
- None of the demos is closed-loop LLM control, and none is perception-based. Every controller reads ground truth: cable nodes, `d.body(...).xpos`, contact forces.
- The coordinates are hard-coded, there is one layout and one seed, and nothing is randomized. Tools are pre-grasped and contact parameters are tuned. There are no success rates.
- The headphone result is stitched from 19 runs with 8 recoveries and human strategy edits, and it fails its own release verifier. The Baoding result is 1 of 592.
- Model version, settings, prompts, token counts and cost are unknown for the repo demos. The only cost data is the off-repo X post: ~22k output tokens (Astra) vs ~25k (Sol) per drawing controller, with Sol ~5× cheaper.
- Exact reproducibility is platform-specific (§5). The headphone simulation runs ~~200–1,850×~~ [corrected: about 135–1,850×, depending on solver settings and machine load (see §5 fact-check)] slower than real time, so it cannot support a closed-loop LLM iteration budget without parallelism.

**Novel vs. repackaged.** The control methods are classical: DLS and null-space IK, quintic minimum-jerk, a tripod gait with analytic leg IK, PI-style force regulation, an FSM, and PPO. The pilot API is a Code-as-Policies / VoxPoser-style action API. Two things are new:
1. The **evidence that Astra-class coding agents can produce this code** with human steering.
2. The **disclosure and integrity discipline**.

The Jev two-call decomposition (intent choice, then discrete per-axis motor choice) is a useful idea, but it lives outside the repo.

**Maturity:** demo-grade, though well packaged (checksummed releases, validators, uv lock). It is not a benchmark; the author says "does not yet define a standardized benchmark." [fact-check: that wording is from the initial README at `1944286`. The `a8b22f1` README says "These are exploratory examples, not a standardized benchmark or a general capability ceiling." The current README has neither sentence.]

**What Ilia's Opus-backbone harness should borrow**
1. **The trusted firmware / untrusted LLM split as Claude tools.** For example, `drive(forward, turn, height)` and `set_hand_pose(side, position[3], rotation[3x3] | quat, grip, frame)`. Enforce budgets server-side, roll back transactionally, and return errors as `tool_result` without advancing time. The firmware owns IK, minimum-jerk interpolation, velocity and acceleration limits, and gripper force logic. Opus emits sparse Cartesian goals at about 1–10 Hz.
2. **Fixed physics advance per accepted command** (0.1 s) in sim, so LLM latency does not corrupt the dynamics. On hardware, the firmware must servo and hold between calls.
3. **Postcondition predicates with dwell times,** like `hands_ready` (12 mm), `grippers_ready` (contact + saturation, 0.3 s) and `settled`, exposed to Opus as boolean facts. This matches the Jev observation format ~~(`cube_held_by_both_fingers`, …)~~ [corrected: the key `cube_held_by_both_fingers` comes from the third-party reconstruction YuanKJing/Jev-as-Policy (`jev_policy.py:52`), not from Hrybov's demo. The original demo's overlay shows plain text such as "Contact: both fingers", "Fingers 4.8 cm · close command" and "Cube bottom 11.3 cm · wall top 7 cm"], and Opus's verification turns become cheap.
4. **Monitors with human-readable abort reasons, plus phase checkpoints,** to support "retry from last good phase" loops. Examples: slip in mm, unplanned collision bodies, "replan the wrist posture".
5. **For author-time use** (Opus writing new skills offline), adopt the integrity suite: hashes, no welds or mocap, contact-only grasps, force audits. Add a held-out strict verifier like `verify_release.py` so that a reward-hacking agent cannot pass with a visually plausible result.
6. **The intent → motor two-stage call** (Jev demo) as a cheap discrete action head. A light learned head could replace stage 2.

**What to avoid**
- Do not cite these demos as evidence of LLM spatial or visual control competence. The vision path is untested and the observations are privileged.
- Do not report single selected episodes. Use randomized layouts, N≥20 seeds, and tolerance-based validators, and pin the platform if you claim bitwise replay.
- Do not let the model hold Python execution access inside the simulation process at runtime. The repo's own docstring warns against this.

## Sources
- https://github.com/dimentary/llm-robotics-playground (code, git history, release v0.1.0: https://github.com/dimentary/llm-robotics-playground/releases/tag/v0.1.0)
- https://hrybov.com (author bio)
- https://x.com/dimentary/status/2097141042214797801 (dove), https://x.com/dimentary/status/2097455860541009958 (Fibonacci), https://x.com/dimentary/status/2101018760371171420 (Jev), https://x.com/dimentary/status/2102539444834402314 (Sol vs Astra), https://x.com/dimentary/status/2097857980150763900 (rope hand, UNREACHABLE). All fetched via https://api.fxtwitter.com
- https://github.com/zjwzcx/Awesome-Astra-Embodied-AI (Cases 11, 27)
- https://github.com/YuanKJing/Jev-as-Policy (reconstruction; docs/provenance.md)
- https://github.com/Frank-ZY-Dou/awesome-jev
- https://x.com/qineng_wang/status/2099893504658866561 (inspiration for the headphone task)
- https://github.com/sharpa-robotics/sharpa-urdf-usd-xml (Baoding assets), https://github.com/google-deepmind/mujoco_menagerie (pinned 8161bba)
- https://news.ycombinator.com/item?id=49582582 (Astra-robotics HN context; no thread on this repo found via hn.algolia.com)
- https://openai.com/index/gpt-6-astra/ (UNREACHABLE: Cloudflare 403; Astra API params not verified here)

## Verification (fact-check pass)

Fact-checked on 2026-10-01. Every claim below was checked against primary sources:
- the repo at HEAD `5ffdf58` and its full history, in `research/repos/llm-robotics-playground`;
- the GitHub API and the release `v0.1.0` assets;
- `api.fxtwitter.com` JSON for each X post, plus frames I extracted from the Jev and Sol-vs-Astra videos with ffmpeg;
- hrybov.com (HTML and meta tags);
- the cloned third-party repos.

I also reran several things on this Linux x86-64 8-core host with Python 3.14 and MuJoCo 3.12.0, working in a scratch copy under `/tmp/lrp` so the repo outputs stay untouched:
- `check.py`;
- the robo-spider `run.py` and `validate.py`;
- `pilot.py --camera`;
- the headphone `validate.py --seconds .08`;
- the fibonacci, dove and Baoding validators.

### Confirmed (seen in a primary source)

**Repo metadata**
- 163 stars, 8 forks, MIT. Created 2026-09-17T17:44Z; last push 2026-09-24T18:47Z (`gh api repos/dimentary/llm-robotics-playground`). 7 commits.
- Commit times and authors match `git log`. `a8b22f1` adds 122,648 lines. Release v0.1.0 was published at 18:15Z (11:15 PDT).
- `e4f5979` deletes the four `recording.json` files. It removes `initial_nodes` and `simplification` from `task.json` and replaces `../wiring-repair/vendor/PROVENANCE.json` with `../../assets/manifest.json`.
- The README framing change "I used Astra to build four robot experiments…" → "The first experiments used GPT-6 Astra in Codex…" happens across `e4f5979` → `867fd52`.
- Every original `recording.json` has `"reported_label": "GPT-6 Astra", "exact_snapshot": null, "settings": null` (`git show a8b22f1:experiments/*/recording.json`).
- The publication-plan quote "Use unknown for unavailable model settings; do not reconstruct an exact original prompt from memory" is verbatim (`git show 1944286:docs/publication-plan.md`).
- The quotes attributed to the `a8b22f1` READMEs are verbatim:
  - "uses simulator state, not live image interpretation" is in the dove README.
  - "…its available camera interface was not used as a live VLM policy" is in the robo-spider README.
  - "assembled from continuous accepted stages with retries and human-guided revisions" is in the headphone README.

**robo-spider**
- `pilot.py:1-3` docstring, observation at 25-49, `Renderer(m,180,320)` at line 23, geomgroup[3]=0, commands at 56-71, deepcopy rollback at 58-71, and 50×`mj_step` per command at 75-77.
- `control.py` budgets at 57-61, rotation check at 69-76 (atol 1e-3, det ≥ 0.99), and duration `max(1.2, Δp/0.065, angle/0.4)` at line 92. DLS IK at 107-133: 100 iterations, λ=1e-4 added to JJᵀ, clip 0.12, rotation weight 0.35, tolerances 1 mm and 0.008. IK runs every 5 ticks; quintic plus slerp at 191-217; ±0.65 rad/s and ±1.5 rad/s² on the servo setpoint at 224-227; gripper overtravel logic at 229-244.
- `nu=34`, dt 2 ms, implicitfast. `run.py` calls `pilot.step` every 50 steps (10 Hz, lines 41-42) and audits contacts every 2 ms (46-93). `validate.py:42` thresholds are 3 cm and cos > 0.99.
- I ran `pilot.py --camera` myself:
  - The first observation is 69,741 characters (head 34,000 base64 chars; wrists 17,440 and 17,476).
  - Commands out of budget, with an unknown key, or with a bad rotation return `{"error": …}`, and time stays at 0.0. The next valid command advances time to 0.1 s.
- Original recorded result, from the re-downloaded release tarball (sha256 `2f8e0445…` matches `recordings.json`): t=97.602 s, errors 1.901/2.480 mm, path 6.4229 m, 500 Hz audit, no contacts.
- I reproduced the colleague's rerun result exactly:
  - The run took 51 s of wall time and finished at t=96.402 s, with errors 1.90/2.46 mm.
  - `environment_contacts = {'left_grip_left_pad / source_table': first_s 33.564, samples 2, max_penetration 8.68e-5 m}`.
  - `validate.py` raises "Robot contacted a table or barrier". The trajectory deviates by >1 cm at t=22.36 s.

**headphone-untangling**
- `task.json` values:
  - segments 114/143/25 (sum 282) and diameters 1.8/1.3 mm;
  - bend 2e6 Pa, twist 7e5 Pa;
  - the cluster kinds `overhand_knot` + `threaded_bight`;
  - the `first_action` and `motion_preferences` strings and the "Friction and cable elasticity are provisional" assumption.
- The scene option is `timestep=.0002 solver=CG iterations=100`, with nu=14.
- Line references are correct: `cord_scan.scan` at l.10 (opening 12→7 mm in 0.05 mm steps; allowed segments ±2), `Rollout.move` at l.281 (15 ms grid; 4 rad/s limit at l.367-368), `control()` at l.218 (integral clip ±0.025), `monitor()` at l.244 (0.5 mm / 0.2 mm / 4 mm thresholds), `follow_and_pinch` at l.482 (refresh 0.02 s; close when error < 1 mm and t ≥ 0.10 s; 0.08 s opposed and seated), and the checkpoint comment at l.770-775.
- Report `two_clusters_current.json`:
  - 19 `sources`, 8 `recoveries` (main_cable_095 ×3, "Discontinuous…" ×2, "Free end moved outside its grasp", "Blocked cable tracking pose…", "No opposing free-end contact; lift inhibited"), 45 phases;
  - `wall_s` 6862.4 for 33.149 s simulated;
  - integration PGS with 150 iterations at dt 0.25–1 ms;
  - `outcome_note` "Both entanglements opened; two overlaps remain beside the splitter";
  - `untangled: false`.
- `verify_release.py` has exactly 10 checks (l.69-82). The render edit is `omitted_source_intervals_s [[4.23, 5.96]]`.
- `validate.py --seconds .08` (my run): all 7 approach checks report error 0.0 and 0 collisions.

**fibonacci-writing**
- `CODE` string at `strokes.py:5`, with pitches 0.015/0.085/0.035. `script_paths` is at l.169 and gives 52 strokes (check.py asserts 52).
- `Writer.solve` at l.82 uses bounded `least_squares` on the `tool_target` site, re-synced by `sync_tool` (l.73-80).
- Depth update `(0.5−f̄)·3.5e-5` at l.223/235-236; ≥3/10 loaded substeps and >160 waits → "Unable to maintain ink contact" at l.243-255; abort if base < 0.70 m or slip > 0.04 m at l.194-198.
- nu=43, free base joint, dt 1 ms.
- `results.json`: 272.21 s, 4,803 marks, median/p95 error 0.7216/6.065 mm, `max_tip_force_N` 12.93.

**dove-drawing**
- `firmware.py` docstring and JSON-lines ops observe/reset/set_targets (l.145-172); `duration_s ∈ [0.002, 2]` with quintic interpolation (l.44-75); per-contact `normal_force_N` (l.112-142).
- The `controller.py` docstring "GPT-authored scripted controller…" is verbatim. It runs DLS null-space IK (λ=1e-5, null-space gain 0.005 toward nominal) and raises at a residual above 3 mm (l.69-70).
- `PhysicalWriter.draw` at l.59: gain 0.15, clip ±0.15 mm, force term `(f−0.15)·2.5e-5` clipped ±10 µm.
- `strokes.json`: method string verbatim; 9 strokes with 458/401/319/94/35/19/10/25/25 points.
- Results: 173 s, 10,064 stamps, error 0.149/0.549/4.63 mm, `median_loaded` 0.110 N, `min_hand_contacts` 2. nu=27.

**baoding-balls**
- `train.py`: `GOAL_SPEED=1.2`, `ACTION_SCALE=.025`, dt 1/240 with decimation 4, 22 actions, 107 observations (asserted at l.252), ball radius 0.019 m and mass 0.035 kg, reward at l.218-227.
- PPO config at l.286-300: 1024 envs, 32 steps, [256,256,128] ELU, lr 1e-4 fixed, entropy 0. Std is filled with 0.35 and frozen at l.352-354.
- `bounds_ppo.py:322` adds `1e-4·Σ((|μ|−1.1)+)²`.
- README: iteration 3320, seed 982210, trial 12, 592 records, Isaac Sim 5.1.0, isaaclab 0.54.2.
- `validate.py` (my run): 11.97 s, 4.935 turns, max penetration 0.321 mm, side-risk 70.65%.

**Author and X posts**
- The hrybov.com meta description is "ML research engineer in San Francisco, building evals for physical-world AI. Backed by EF. …". The page body confirms ModelRoom co-founder/CTO (Apr–Aug 2026), Google Kaggle Research (2025–26), Promaton and Surgalign.
- X posts (fxtwitter):
  - dove: 2026-09-08 01:52:42Z, 109,024 views, video 45.266 s;
  - Fibonacci: 2026-09-08 22:43:40Z, 70,529 views, video 73.066 s;
  - rope-hand 2097857980150763900: HTTP 404 (snowflake time 2026-09-10 01:21Z);
  - Jev: 2026-09-18 18:41Z, video 20.333 s;
  - Sol vs Astra: 2026-09-22 23:24Z. All quoted texts are verbatim.
- `ffprobe` on the release assets gives `fibonacci-writing.mp4` 73.067 s and `dove-drawing.mp4` 45.267 s, so the X videos and the release videos match.
- Jev frame overlays are as described: "real-time policy · 3.2 updates/s · no images", "MuJoCo state → text · Jev 1.13 · 100 ms target filter · 1× playback", the 8 intents, and "Four motor distributions · up to 2 cm per axis · physical fingers".
- Sol-vs-Astra frames show "13 strokes · 119.8 s simulation" vs "17 strokes · 259.7 s simulation" and "Michelangelo · The Creation of Adam", at 5× replay.
- Third-party repos:
  - Awesome-Astra Case 11 sits under "Zero-shot Control → Deploy in Simulation". Case 27 has the quoted tendon sentence and "Published: 2026-09-10".
  - Jev-as-Policy `docs/provenance.md:3,7,8` are verbatim: `jev-1.13.0`, TypeSafe `/v1/systemone`, and `mujoco==3.3.7` in requirements.
  - awesome-jev entry jev-005 archives the 20 s video.
  - The Qineng Wang post has 646,492 views. Its demo page says "Dual ALOHA · kinematic replay".
- HN Algolia returns 0 hits for "llm-robotics-playground". HN item 49582582 is "GPT-6 Astra on robot arms".

### Corrections (claim → correct value, with evidence)

1. **Jev cycle times "269–364 ms"** → observed values range from 186 to 578 ms. I sampled the readout at 2 fps from the fxtwitter video URL: 481 ms at 0.84 s, 578 ms at 2.59 s, 186 ms at 4.46 s and 401 ms at 5.92 s. The nominal rate is 3.2 updates/s. (Fixed inline.)
2. **"14-stage FSM"** → `mission.py` names 16 stages. 15 are reachable and were executed; `ALIGN_DESTINATION` (l.172-175) is dead code. The recorded `stages.json` has 14 entries because the initial `APPROACH` is not logged. (Fixed inline.)
3. **"mission.Mission … same Firmware API"** → partly wrong. RELEASE sets `h["grip"]=0` directly (mission.py:189-191). `grippers_ready` and `settled` read `d.ctrl`, `d.contact` and base `qvel` (l.39-47, 70-84). The scripted pilot therefore uses privileged signals that `pilot.py` never exposes, and none of them would be available to an LLM pilot. (Fixed inline.)
4. **`cube_held_by_both_fingers` as "the Jev observation format"** → that key belongs to the third-party reconstruction (`Jev-as-Policy/jev_policy.py:52`). Hrybov's original video shows prose fields instead: "Contact: both fingers", "Fingers 4.8 cm · close command", "Divider: ahead of cube". (Fixed inline.)
5. **"numerically trivial differences" (robo-spider rerun)** → the knee and hip joints already differ by about 2.7e-4 rad after the first 2 ms step, and by about 1e-3 rad at t=0.042 s. That is not round-off accumulation; the cause is unknown. (Fixed inline.)
6. **"largest controller (2,806 Python lines)"** → 2,806 counts all .py files in the experiment, including the renderer, viewer and validators. The control and planning code is about 2,050 lines. (Fixed inline.)
7. **"`m.neq==0`/`nmocap==0`" listed as a general integrity check** → it is enforced only for fibonacci and dove. robo-spider has `neq=6` (upstream Robotiq linkage equalities), and headphone asserts `neq==2` (upstream ALOHA finger equalities). These are legitimate upstream constraints, not object welds. (Fixed inline.)
8. **Headphone "200–1,850× slower than real time" and "148 s per 0.08 s ≈1,850×"** → the figure depends on settings and load. My measurements: `validate.py --seconds .08` gave `wall_s` 90.0 (about 1,125×). Raw stepping from the initial state gave about 418× at CG/0.2 ms and about 135× at PGS/1 ms/150 iterations. The recorded production run was about 207×. The corrected range is roughly 135–1,850×. (Fixed inline.)
9. **Commit `1944286` "Docs only"** → it also adds `README.md`, `LICENSE` and `.gitignore`. Minor. (Fixed inline.)

Clarification, not counted as a correction: the quote "does not yet define a standardized benchmark" is from the `1944286` README only; the current README no longer says it. Annotated inline.

### Unverifiable or partly verified claims
- **Sol vs Astra claims.** The "~25k vs ~22k output tokens" and "almost 5x cheaper" figures are the author's self-report on X. No logs, prices, model snapshots or code were published, so they cannot be checked.
- **Robot in the Sol-vs-Astra bench.** "This reuses the Fibonacci G1 + whiteboard setup" is an inference. The frames show a G1-like humanoid with a dexterous hand at an easel or whiteboard, but there is no code to confirm it.
- **Rope-hand post content.** The post itself is a 404. Its content is known only second-hand from Awesome-Astra Case 27.
- **Astra's role in the Baoding trainer.** Still unknown; the README is silent. Note also that the Baoding README says "The Isaac Lab trainer has not been executed on a GPU after this cleanup."
- **Headphone stage-1 rerun.** Not rerun by either checker.
- **macOS bitwise reproducibility.** Claimed by the author (`recording.json` `package_checks`: "full controller rerun: historical trajectory arrays reproduced exactly"). Not testable here.
- **Model labels.** "GPT-6 Astra", "GPT-6 Sol" and "Jev 1.13" appear only as labels in the repo, README, X posts and overlays. The vendor pages were not checked; openai.com returned a 403 to the colleague.

### Missed details (added)

**1. pilot.py is a thin observation channel, which matters for an LLM pilot.**
- The observation contains only time, base pose and the two pinch-site poses, plus optional 320×180 PNGs.
- It has **no object poses, no gripper aperture or grip state, no contacts or forces, no drive/odometry state, and no "trajectory finished" flag**. A live LLM would have to infer grasp success and object locations from 320×180 images alone.
- Hand goals must be **world-frame 3×3 rotation matrices orthonormal to 1e-3**, which is error-prone for LLM text output. Quaternions are not accepted.
- There is **no observe-only, reset, or "advance N s" op**. Even `{}` is accepted and advances 0.1 s.
- The firmware's internal dynamics make each step small:
  - drive acceleration is limited to 0.04–0.12 m/s² (`control.py:138-141`), so going from 0 to 0.3 m/s takes ≥2.5 s, i.e. ≥25 commands;
  - every hand move lasts ≥1.2 s, i.e. ≥12 commands;
  - the grip ramps at 160 ctrl-units/s toward 255, i.e. about 1.6 s to close (`control.py:244`).
- The scripted demo made 976 pilot decisions over 97.6 s. A live VLM pilot at this granularity would need about 1,000 calls per episode with about 70 KB of base64 images each, unless the harness adds a multi-step "advance until done" primitive. The dove firmware has one (`duration_s` up to 2 s); the robo-spider pilot does not.

**2. Tuned physics that make "friction-held" tools easier.**
- `fibonacci-writing/scene.xml:3` sets `noslip_iterations="10"`. This is MuJoCo's post-hoc slip-removal solver, and it strongly stabilizes friction grasps.
- `dove-drawing/scene.xml:3` uses `cone="elliptic" impratio="10"` with 80 iterations.
- The a8b22f1 READMEs admit "Finger gains and contact parameters were tuned". For the dove they say contact parameters "approximate a compliant fingertip patch rather than measured materials".

**3. Fibonacci setup-time teleport and stance.**
- `Writer.__init__` writes the free marker's qpos directly: it shifts the marker 5 cm along the barrel (`writer.py:37-39`) and then re-poses it relative to the palm after a robust IK solve (l.55-63). The grasp is therefore constructed, not physically achieved. The README says "pregrasp was prepared with temporary support".
- The floating-base G1 "stands" using joint position targets plus a **privileged inverse-dynamics feedforward** `qfrc_bias/gain` for actuators 0-35 (`writer.py:148-152`). There is no balance controller. The only stance safeguard is the abort when the base drops below 0.70 m.
- Peak tip force reached 12.9 N against the 0.5 N target, and grip slip reached 5.07 mm.

**4. Validators check self-reported metrics.**
- The fibonacci and dove `validate.py` scripts state that they check "the saved recording and report; it does not rerun the controller".
- Several checks only assert values the controller wrote into `results.json`, e.g. `non_tip_board_contact_samples == 0`.
- Tolerances are loose: Fibonacci allows grip displacement < 40 mm (`validate.py`), while the recorded value was 5.07 mm.

**5. Headphone edit and code gaps.**
- The released video **ends at source t=29.259 s** (`two_clusters_wide_render.json`: kept `[[0, 4.23], [5.96, 29.259]]`, 27.529 s of motion plus a 1 s hold). The final trunk-lift and lay-out phases 38–44 (29.26–33.15 s) are **not shown**.
- The omitted 4.23–5.96 s interval covers phases 5–9: secure, capture, pinch, seat, deep grip.
- 17 of the 45 recorded phase labels have no literal counterpart in `run.py` or `control.py`.
- The recorded run used five different integration profiles across phases (PGS/150 iterations; dt 1, 0.5 or 0.25 ms; dense or default Jacobian). The scene file itself specifies CG/100/0.2 ms.
- The `follow_and_pinch` docstring says "Only visual/material position feedback", but the code tracks ground-truth `d.geom(...).xpos` (control.py:531).

**6. Model-involvement details in history that the note omitted.**
- The a8b22f1 top-level README says the experiments "were developed with Astra through iterative code generation, simulation inspection, and human feedback". The headphone README says "The model inspected simulator state and rendered results". So **images were used offline during authoring, not at runtime**.
- Task-brief summaries exist even though prompts do not:
  - robo-spider: "make an unusual legged, dual-arm robot perform a useful transfer";
  - fibonacci: "physically write a short Fibonacci program, rather than compute numbers during the motion".
- The X posts give prompt-like text: "asked it to build a MuJoCo setup and write a controller to draw Picasso's dove using a robot arm and a five-fingered hand".
- Headphone human steering is spelled out: "The user helped select the layout, simplify entanglements, favor local lifting/relaxation, and choose the presentation."
- `physical_dove_results.json` `mode`: "GPT-authored scripted controller, free-pencil physical pre-grasp; no live model piloting".

**7. Admitted limitations missing from the note.**
- robo-spider: "This is a tuned route through one fixed layout. General navigation, perturbations, and physical hardware were not evaluated."
- fibonacci: "These checks do not establish robustness across writing styles or starting grasps."
- headphone: "The full guided solution was not rerun during packaging."
- Baoding:
  - "Intermediate physics substeps and the time-zero state were not recorded, so the checks are not full physics certification."
  - "Reused evaluation seeds and selection from the archive do not establish generalization."
  - The 592 records come from **37 distinct saved evaluation traces**.
  - The default trainer runs 200 updates (6,553,600 transitions), versus the checkpoint's multi-stage history to iteration 3320.
  - Training reward caps per-step progress at `GOAL_SPEED·dt` (`training_core.advance_progress`, `max_gain`) and penalizes angular overspeed. This supports the note's inference that the selected ~2.6 rad/s rollout came from a different, archived training configuration.

**8. Jev demo specifics.**
- The text observation also includes "sampled <t> s", "Cube bottom 11.3 cm · wall top 7 cm", finger width with the last command, and "Divider: ahead of cube".
- Outputs are numbered ("JEV MOTOR OUTPUT · #019" at t≈6 s).
- The scene is an arm with a parallel gripper, an orange cube, a blue divider wall and a pink target pad. The reconstruction repo uses a Franka Panda. The original robot model was not identified from frames.

**9. Author context relevant to a learned action head.**
- hrybov.com says he "fine-tuned MiniMax H3, a 33B video diffusion model, into a robot policy that predicts both video and actions, trained on DROID".
- At ModelRoom he built "a VLM harness that checks every shot against the prompt". This is the same verifier pattern the repo uses.
