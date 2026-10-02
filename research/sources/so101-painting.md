# HF Bucket `mishig/so101-painting`: GPT-6 Astra as an *offline engineer* for SO-101 brush painting (MuJoCo)

*Deep-dive note for the Opus-backbone robot-harness research. Read on 2026-10-01.*

- **Bucket:** https://huggingface.co/buckets/mishig/so101-painting
- **Local copy:** all 101 non-video, non-mesh files were downloaded to `research/repos/so101-painting/`. Every file matched `MANIFEST.sha256`; meshes and font were fetched later for the re-run.
- **Sibling bucket:** `mishig/so101-block-sorting` is mirrored in `research/repos/so101-block-sorting/` for cross-reference.
- **Line references:** all `path:line` references point at the bucket snapshot (bucket `updatedAt` 2026-09-14T21:22:01Z).

---

## TL;DR

- **What it is.** Mishig Davaadorj is a Hugging Face Hub engineer (`isHf: true` on the bucket page). He asked **GPT-6 Astra**, acting as a *coding agent*, to reproduce thijs's (@cdngdev) viral 2026-09-08 real-world demo, ~~"astra paints the Golden Gate Bridge with an SO-101"~~ [corrected: this is a paraphrase, not a quote. The tweet text is "i gave astra a robot, a paint brush, and a camera then asked it to paint the golden gate bridge in real life!" (syndication API, status 2097339677128982873). The SO-101 identification comes from the Roboflow blog of 2026-09-18, not from the tweet], in MuJoCo. The bucket is the agent's packaged output:
  - a MuJoCo scene of an SO-101 holding a spring-loaded brush;
  - a 2-D pigment model;
  - a scripted controller;
  - a frozen RGB evaluator;
  - four recorded "learning" attempts plus one exploratory landscape run.
- **The key architectural fact: Astra is not in the control loop.**
  - At runtime, the robot runs a **hard-coded 19-stroke plan** (`scripts/paris_design.py`) through **numerical IK** and timed joint trajectories.
  - ~~The only "learning" is that the coding agent looked at the camera result between runs and **hand-edited a 6–9-field `parameters.json`**: contact depth, stroke speed, easing, touch/lift times, reload interval.~~ [corrected: the between-run changes went beyond `parameters.json`. (1) They included controller code. `learning-report.md:3` says "I inspected the camera results between attempts and **changed the controller** in response", and attempt 02 introduced the `atomic_stroke` code path behind the new `atomic_strokes` flag. (2) The evaluator itself was "finalized after diagnosing the first two paintings" (`learning-report.md:30`). The `parameters.json` edits (contact depth, stroke speed, easing, settle frames, touch/lift times, reload interval, per-colour overrides) are the only *recorded* part.]
  - The bucket says so explicitly: *"This was agent-guided parameter refinement between runs. The robot follows a scripted stroke plan; no neural policy was trained."* (`learning-report.md:34`)
- **Results.** On a self-authored image metric, the score rose **83.37 → 92.97 → 95.62 → 96.26** over 4 attempts, with one run per setting in a deterministic simulator.
  - Caveat 1: the metric was "finalized after diagnosing the first two paintings" (`learning-report.md:30`).
  - Caveat 2: its ±3 px registration search saturates at the bound **[3, 3]** in all four runs.
- **What is missing.** There is no LLM prompt, transcript, model/API parameters, token count, cost or latency anywhere in the bucket. How Astra was invoked (Codex? ChatGPT? reasoning effort?) is **UNVERIFIED**.
- **Value for Ilia.** It is a clean, small, fully inspectable example of the **"frontier model as outer-loop robotics engineer"** pattern (same family as EmbodiedSWE), not of "LLM as policy". Its best parts are reproducibility hygiene and a validated HTTP motion API with timed atomic primitives, not robotics novelty.

---

## 1. Provenance and context

| Item | Value (source) |
|---|---|
| Bucket ID / created | `mishig/so101-painting`, created 2026-09-14T21:20:41Z, 126 files, 127,434,417 bytes (`/api/buckets/mishig/so101-painting`) |
| Author | Mishig ᠮᠢᠰᠾᠢᠭ (`mishig`, `isHf: true`, 402 followers). HF Hub engineer since 2021 (mishig25.github.io) |
| Announcement | X @mishig25, Sep 14: *"reproduced the experiment in MuJoCo: asked gpt6 astra to paint Eiffel Tower in simulation using so101 and a paint brush, see the entire experiment huggingface.co/buckets/mishig…"* (search-engine snippet; X itself returned HTTP 402 and would not render headless, so the full text and status ID are **UNVERIFIED**) |
| What it reproduces | thijs @cdngdev, 2026-09-08 (status 2097339677128982873, 20,724 likes via syndication API): *"i gave astra a robot, a paint brush, and a camera then asked it to paint the golden gate bridge in real life! it figured out how to control the robot, and progressively got better throughout its attempts."* |
| Sibling posts | Same day: *"no VLA - just gpt6 astra calibrates itself and solves the task of putting lego blocks into a box … the entire thing is done in MuJoCo simulation"*, which is bucket `mishig/so101-block-sorting` (created 14:48Z). Later: *"gpt6 astra paints mona lisa by controlling so101 in MuJoCo"*, with **no public bucket** (the `/api/buckets/mishig` listing shows only 8 buckets, none Mona-Lisa), so **UNVERIFIED**. Also: *"I keep hitting my astra token limits."* |
| Platform | macOS (`mjpython` path in `scripts/run_attempt.py:96-101`; "native macOS viewer exited once", `learning-report.md:42`). The sibling bucket's `REPRODUCIBILITY.json` records Darwin 25.4.0 arm64, Python 3.11.15 |
| Timeline | File mtimes cluster on 2026-09-14: the exploratory landscape at 16:37Z, the four Paris attempts 16:46–16:52Z, then release packaging at 21:20Z. These mtimes may be copy times. It is plausibly a one-day project |

**About HF Buckets.** A bucket is a non-versioned, mutable, S3-like repo type on the Xet backend.
- Files resolve at `/buckets/<id>/resolve/<path>`. There is no `main` revision; `/resolve/main/` returns 404.
- The tree is at `/api/buckets/<id>/tree?recursive=true`, and the CLI is ~~`hf buckets sync|cp|ls…`~~ [corrected: the documented subcommands are `hf buckets create|list|cp|sync|rm|settings` (https://huggingface.co/docs/hub/storage-buckets). The docs show `list`, not `ls`.].
- Mutability is why the bucket ships `MANIFEST.sha256` plus `verify_manifest.py`.

**The README's "agent-first" setup prompt** is worth quoting, since it is the actual interface Mishig offers (`README.md:7`):

> *"Set up and run the SO-101 painting simulation from https://huggingface.co/buckets/mishig/so101-painting on my computer. Read the README and learning report, download the files, verify their checksums, and use uv for all Python environment and package management. Start with attempt 04's settings to paint the Eiffel Tower and Seine. Record the new attempt, inspect its actual camera image, and explain any proposed changes before testing more attempts. Preserve every recording and use the unchanged camera evaluator to compare results."*

This is effectively the outer-loop "policy" in prose: it asks a coding agent to continue the parameter search.

---

## 2. Architecture

```
          OUTER LOOP (minutes, between episodes) — GPT-6 Astra as coding agent [interface UNVERIFIED]
 ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
 │ reads camera.png / camera-canvas.png / evaluation.json / diagnostic.png of attempt k         │
 │ writes results/attempt-(k+1)/parameters.json  {contact_depth_m, stroke_speed_m_s, easing,    │
 │        settle_frames, touch_seconds, lift_seconds, atomic_strokes, reload_every_strokes,     │
 │        color_parameters{brown:{…}}, "reason": "<hypothesis text>"}                           │
 │ (it also wrote ALL the code below, the evaluator, the video editor, and the reports)         │
 └───────────────▲──────────────────────────────────────────────────────────────┬───────────────┘
                 │ files                                                        │ uv run scripts/run_attempt.py
 ┌───────────────┴───────────────────────┐   HTTP JSON 127.0.0.1:8879   ┌───────▼──────────────────────────┐
 │ evaluate_painting.py (frozen, SHA     │                              │ paint_paris.py  (controller)     │
 │ feedc58a…) rectified RGB + plan →     │                              │  plan = paris_design.design(v1)  │
 │ score, metrics, rule-based            │                              │  localize_canvas(): 4 cyan dots  │
 │ "suggestions"                         │                              │    → cv2.findHomography          │
 └───────────────────────────────────────┘                              │  for stroke: load()/reload rule  │
                                                                        │    xyz (u,v→m) → IK per 2 mm     │
                                                                        │    POST /trajectory {targets,    │
                                                                        │      times, seconds}             │
                                                                        │    assert contact_frames grew    │
                                                                        │    POST /camera → HSV pixel log  │
                                                                        └───────┬──────────────────────────┘
                                                                                │
 ┌──────────────────────────────────────────────────────────────────────────────▼──────────────────────────┐
 │ painting_server.py: MuJoCo 3.13 loop, dt=16 ms (8×2 ms substeps); linear interp of 6 ctrl targets;      │
 │ position actuators kp=998.22 kv=2.731; brush = slide joint k=120 N/m, 0–9 mm; contacts → PaintSurface    │
 │ .stamp(x,y,F,angle,speed); palette contact → load_color(); front cam 960×720 fovy 30 → mp4 @30 fps       │
 │ + timeline.json; GET /state exposes qpos, ctrl, brush_color, contact_frames, paint.load …(privileged)    │
 └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

There is no LLM call anywhere in the Python code. `grep` for `openai`, `anthropic`, `api_key` or `model=` ~~returns nothing in `scripts/`~~ [corrected: returns one false positive, `build_scene.py:69` `model="SO-101 brush painting"`, which is an MJCF attribute. There is no LLM reference. In fact no file in the bucket mentions Astra, GPT, OpenAI or Codex at all.].

---

## 3. Code walkthrough

### 3.1 Runner: `scripts/run_attempt.py`
- **Preserves attempts.** It refuses a non-empty run dir and copies `parameters.json` into it (`:88-93`).
- **Owns its server.** It refuses an occupied port (`:32-39`) and spawns the server in its own process group. It accepts the server only after seeing the readiness line in *its own child's* log and getting `GET /state` with `scene=='painting'` (`:120-138`).
- **Shuts down in order.** SIGINT to the controller (so its `finally` block finalizes the video), then `/record/stop`, then `/shutdown` (`:153-167`).
- **Takeaway.** This lifecycle hygiene is reusable as-is.

### 3.2 Simulator and motion API: `scripts/painting_server.py` (256 lines)

**Endpoints** (`:103-171`):

| Endpoint | Purpose |
|---|---|
| `GET /state` | Current state snapshot |
| `POST /move` | Single joint target |
| `POST /trajectory` | N×6 joint targets, optional `times`, `seconds`, `easing`, `settle_frames`, `label` |
| `/camera` | Capture a frame |
| `/record/start`, `/record/stop` | Video recording |
| `/shutdown` | Stop the server |

**Input validation** (`:136-147`):
- targets must be an N×6 array and finite;
- they are **clipped to `actuator_ctrlrange`**;
- duration must satisfy `0 < s ≤ 120`;
- `times` must increase strictly and end at `seconds` (±1e-5);
- `settle_frames` is clamped to 0–20.

This is the closest thing to a "tool schema" in the project.

**Control loop** (`:125-244`):
- Each tick linearly interpolates `data.ctrl[:6]` along the path. It uses either a global smoothstep (`amount*amount*(3-2*amount)`) or explicit per-waypoint timestamps (`:174-188`).
- It then runs `mj_step(nstep=8)` at a 2 ms timestep, i.e. a **16 ms control tick ≈ 62.5 Hz**.
- When the motion plus settle frames end, it automatically captures a camera frame and replies (`:234-239`). **Every command returns an image path plus the full state.**

**Contact → paint** (`:190-224`):
- For each contact involving `brush_bristles`, normal force comes from `mj_contactForce`.
- Contacts with `paper_contact` above 0.002 N are summed; their XY is averaged and fed to `surface.stamp(x, y, force, dt, angle, speed)`.
- Contact with a `palette_<color>` geom calls `surface.load_color()` and recolors the bristle geom.
- ~~Contact is never debounced. Every dip logs **3** `Brush loaded …` events (`results/*/server.log`).~~ [corrected: a load fires on each *rising edge* of palette contact (`palette != current_palette`, `painting_server.py:204-214`), with no temporal debounce. Each event calls `load_color()`, which resets `load=1.0` and increments `reload_count` (`paint_surface.py:65-77`). Contact chatter gives about 3 events per dip, but attempt 01's orange dip logged **4** (`physics-report.json` `load_events`).]

**Privileged state** (`:79-84`): `/state` returns `brush_color`, `contact_frames`, `maximum_contact_force_n`, `paint.load`, `brush_tip_xyz` and warning counts. The controller *uses* several of these (below).

### 3.3 Controller: `scripts/paint_demo.py` (base `Painter`) and `scripts/paint_paris.py`

**Canvas localization, `Painter.localize_canvas`** (`paint_demo.py:66-89`):
1. HSV threshold for cyan: H 85–99, S > 130, V > 120.
2. Connected components with area 5–1500 px; keep the 4 largest.
3. Order the points by extreme x/y.
4. `cv2.findHomography(image_points, world_xy)`, composed with a world→512-px canvas matrix `W`.

This is done **once** per run. The four cyan cylinders lie exactly in the paper plane (`build_scene.py:112-118`). The run fails hard if it does not find exactly 4 markers or the ordering is ambiguous.

**Paint loading, `Painter.load`** (`paint_demo.py:109-121`): lift → approach the well at hover height → dip to `z - 2 mm` → dwell 0.35 s. It then **asserts `state['brush_color']==color and state['paint']['load'] >= .9`**, both privileged simulator values.

**Reload rule** (`paint_paris.py:100-103`): reload when the colour changes, **or when `paint.load < .35` (privileged)**, or every `reload_every_strokes` strokes.

**Atomic stroke, `ParisPainter.atomic_stroke`** (`paint_paris.py:47-76`), introduced in attempt 02:
- It precomputes 16 smoothstep samples for touch over `touch_seconds`.
- It adds a constant-speed sweep sampled every 2 mm, timed as `length/stroke_speed`.
- It adds 16 smoothstep samples for lift over `lift_seconds`.
- It solves IK for every point *while hovering* ("Solve everything while hovering: no paint puddle while planning", `:48`), then sends **one** `/trajectory` with explicit `times`.
- **Postcondition:** `if result['contact_frames']<=before: raise RuntimeError('Brush missed paper on '+label)` (`:75`).
- This is the single most important engineering change in the whole "learning" story. ~~Planning latency between touch and sweep was what caused the dark endpoint pools.~~ [corrected: the data do not support planning latency as the main cause. In attempt 01's `timeline.json`, the gap between the end of the `touch-*` move and the start of the sweep is **0.016 s (1 tick)** for the first strokes, with a maximum of 0.048 s. Client IK for a whole stroke takes 7–60 ms (fact-checker timing of `SO101Kinematics.solve` over the 2 mm-densified paths). The dominant stationary-contact time in attempt 01 had three sources: (1) `settle_frames=10`, i.e. 0.16 s of dwell after touch-down and again after the sweep (touch move = 0.65 s + 0.16 s = 0.816 s in the timeline); (2) global smoothstep easing, which brings velocity to ≈0 at both ends of the touch, sweep and lift; (3) the 0.65 s smooth lift. Attempt 02 removed all of these at once. The agent's code comment (`paint_paris.py:48`) states the planning hypothesis, but the timeline shows it was a minor term.]

**Per-stroke "camera check", `Painter.inspect`** (`paint_demo.py:91-103`):
- It warps the frame by H to 512×512 and counts vivid pixels in HSV bins for blue/green/orange/red.
- **There is no brown bin**, so the brown tower is logged as `orange` (see `controller.log`).
- The result goes only to `self.history`, i.e. into `run-report.json`. **No control decision uses it.** The README's phrase "It checks camera images after strokes" means logging, not feedback.

### 3.4 IK: `scripts/arm_kinematics.py` (146 lines)

The solver is a numeric, MuJoCo-model-based IK for a **vertical** tool. The objective is position plus 0.10-weighted alignment of the gripper x- and z-axes:
- **Jacobian:** finite differences (ε = 1e-5).
- **Step:** damped least squares, `lstsq([J; 0.001·I])`. Steps are capped at 0.18 rad, with a backtracking line search over fractions {1, .5, .25, .1, .025}. At most 160 iterations per seed (`:95-131`).
- **Seeds:** the previous solution plus 3 hand-picked postures. The analytic pan init is `-atan2(y, x-0.0388353)` and roll is `pan + 0.04868 + yaw` (`:81-93`).
- **Acceptance:** position error ≤ 0.8 mm and orientation error ≤ 0.006; otherwise it raises `ValueError("Unreachable vertical grasp …")` (`:139-145`).

The module docstring stresses that it "never reads object positions" (`:3-4`). It is shared almost verbatim with the block-sorting bucket; `diff` shows only docstrings and the default `grip` differ.

### 3.5 Stroke plan: `scripts/paris_design.py`

- **Plan.** A hand-written, deterministic `design(version=1)` returns **19 strokes in normalized canvas coordinates (u, v)**: 13 brown tower strokes (outlines, antenna, 3 decks, base arch as two quadratic Béziers, feet, 2 X-braces), 1 orange spiral sun (68 points), 2 green banks, and 3 blue sinusoidal Seine curves (`:77-116`).
- **Version table.** `VERSIONS = {1: {...}}` has the comment *"New versions should be added after inspecting an actual camera result. Version 1 is a credible initial design, not an intentionally weak baseline."* (`:20-24`). Only v1 was ever used; `stroke-plan.json` is byte-identical across all 4 attempts.
- **Conclusion.** The geometry was authored by the coding agent in code, not generated at runtime from an image or SVG. There is no SVG → trajectory pipeline and no image-to-stroke model.

### 3.6 Paint model: `scripts/paint_surface.py`

A 512×512 2-D model in which pigment accumulates as optical density:
- **Footprint.** Radius `1.8 mm + 2.3 mm·sqrt(F/1.2 N)`. The dose per contact tick is `dt·7·(0.12+0.88·sqrt(p))·load`, so stationary contact pools pigment (`:107-128`).
- **Brush depletion.** Load drops with speed and pressure (`:136-138`).
- **Spreading and drying.** Wetness diffusion and drying run at 10 Hz (`:188-208`).

The "endpoint pooling" failure that the agent learned to fix is therefore an explicit term of this hand-written model.

### 3.7 Evaluator: `scripts/evaluate_painting.py` (424 lines)

**Integrity and inputs.** `sha256` is ~~`feedc58a…f314f`~~ [corrected: `feedc58a…f1314f`. The full hash is `feedc58a77b40df49aceef666b859f8494512e05ebc0dc3fffe0c97629f1314f`]; I verified it matches the frozen hash in `learning-report.md:32`. It reads only the rectified camera PNG and the planned strokes:
1. White-balances from the brightest paper pixels.
2. Builds HSV+RGB masks for brown and blue.
3. Excludes other planned colours (dilated by 8 px) and boundary-connected blobs that never come near the design. The yellow SO-101 arm fragment in the corner is removed this way: 2,637 "brown" px are ignored in attempts 03/04.
4. Searches **one shared translation of up to ±3 px** (`:114-126`).

**Score** (`:238-239`):
`100 × (0.50·tolerant_Dice + 0.20·centerline_coverage + 0.10·width_agreement + 0.20·pigment_uniformity)`, averaged over brown and blue.

**Pigment uniformity** = 0.7 × endpoint/middle optical-density agreement + 0.3 × middle evenness (`:153-202`).

**Rule-based `_suggestions()`** (`:265-295`). These are human-readable hints for the agent:
- width ratio outside [0.75, 1.35] → ±0.25 mm depth;
- pooling → dwell × 0.6;
- faint lines → 10% slower;
- centerline offset > 5 px → "recheck camera calibration".

Every suggestion carries the caution *"Change one factor per attempt; RGB diagnoses are hypotheses, not guaranteed improvements."*

### 3.8 Scene: `scripts/build_scene.py`

- **Robot.** Upstream `TheRobotStudio/SO-ARM100` `so101_new_calib.xml` at commit `eecbe3e0…`. ~~Collision bits are separated so that only brush↔paper and brush↔palette collide.~~ [corrected: the bits isolate the *brush*. Bristles are contype 2/conaffinity 4 and paper/palette are 4/2, so the brush contacts only paper and palette. The robot's own collision meshes are still set to 1/1 (`build_scene.py:35-37`), so they keep colliding with the `floor`/`desk` (1/1, `scene.xml:17-18`) and with each other, subject to MuJoCo's default parent–child filter. Robot links do not touch paper, palette or brush.]
- **Brush.** A 0.0031 m capsule handle plus ferrule, and a `brush_compression` slide joint: range 0–9 mm, stiffness 120 N/m, damping 0.4. The bristles are an ellipsoid geom with friction 0.35 (`:39-58`).
- **Camera.** One fixed `front` camera at (0.65, −0.65, 0.65) looking at (0.18, 0, 0.04), fovy 30°.
- **Layout.** The canvas is 135 × 150 mm. The four paint wells sit at y = −0.12 m.
- **Actuators.** Taken from upstream: `kp=998.22, kv=2.731`, `forcerange ±3.35` N·m (`scene/so101.xml:31,160-167`).
- **No backlash.** A ±0.5° "backlash" default class exists but **no joint uses it**, so servo backlash and deadband are not simulated.

---

## 4. Calibration and perception (what "calibrates itself" means here)

**Painting bucket.** There is no camera intrinsics or extrinsics calibration.
- A single **planar homography** comes from 4 cyan fiducials (`camera-localization.json`). For attempt 04 the image points span x ≈ 361–651 px and y ≈ 323–512 px, ~~so the 135×150 mm canvas covers roughly **290×190 px** of the 960×720 frame~~ [corrected: 290×190 px is the axis-aligned bounding box of the *fiducials*. The fiducials sit 8 mm outside the canvas, and the canvas appears rotated about 45° in the image. Projecting the canvas corners through the attempt-04 homography gives a quadrilateral with edges of **≈152–167 px** and an area of **≈22.7k px²**, roughly 151×151 px equivalent, or ≈0.9 mm per camera pixel. The 17-px target line width (≈4.5 mm) is therefore only about 5 camera pixels wide.] and is then upsampled to 512×512. The learning report admits: *"enlarging and rectifying it cannot recover missing optical detail"* (`:20`).
- Robot→world uses the perfect simulated kinematic model and encoders. Contact height is a config constant: `paper_height − contact_depth`.

**Block-sorting bucket** (the "calibrates itself" tweet): `scripts/calibrate_camera.py` does a real camera-pose calibration.
- The arm moves through 13 poses. A **magenta marker** on the gripper is detected in RGB, and its 3-D position comes from forward kinematics.
- It then runs `cv2.solvePnPRansac` (EPnP, 3 px threshold) and `solvePnPRefineLM`, with intrinsics from fovy 42°, anchored by 4 board fiducials.
- `REPRODUCIBILITY.json` reports **0.139 px RMS, 16/16 inliers**.
- The unused `forward_marker()` in the painting code (`arm_kinematics.py:49-52`) shows the painting project was forked from this code.

---

## 5. Results

### 5.1 The four Paris attempts

Sources: `learning-history.json`, `results/attempt-0*/{parameters,evaluation,physics-report}.json`. All runs completed 19/19 strokes with 0 MuJoCo warnings.

| # | Change (agent's stated `reason`, abridged) | Key params | Score | Brown / Blue sub-score | Brown endpoint:middle OD | Blue endpoint:middle OD | Brown width px (target 17) | Video s | ~~Loads~~ [corrected: palette-contact load *events* (≈3 per dip). Actual dips were 4 / 4 / 7 / 7] | Max contact F |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| 01 | Reuse landscape controller, 1.5 mm depth | smooth easing, settle 10, touch/lift 0.65/0.65 s, 35 mm/s | 83.37 | 85.82 / 80.92 | 5.07 | 6.72 | 17.6 | 104.0 | 13 | 0.212 N |
| 02 | "Precompute contact, sweep, and lift as one timed trajectory, use constant sweep speed, and remove endpoint dwell" | atomic, linear, settle 0, 0.35/0.30 s | 92.97 | 91.96 / 93.98 | 1.66 | 1.11 | 18.4 | 75.5 | 12 | 0.207 N |
| 03 | "Halve contact speed to deposit more pigment … reload after at most four strokes" | 17.5 mm/s, reload ≤4 | 95.62 | 93.95 / 97.28 | 1.20 | 0.87 | 21.6 | 113.6 | 21 | 0.206 N |
| 04 | "Test lighter brown brush pressure for a narrower footprint and slightly slower brown strokes" | brown: 0.75 mm, 14 mm/s | **96.26** | 95.24 / 97.28 | 1.15 | 0.87 | 19.6 | 119.1 | 21 | 0.206 N |

**What actually moved the score:**
- Brown tolerant Dice is already **0.997 at attempt 01**, and ~~centerline coverage is **1.0** throughout~~ [corrected: brown centerline coverage is 1.0 in all four runs, and blue is 0.995 in all four], so geometry was never the problem.
- The gains come almost entirely from **pigment uniformity**: brown 0.31 → 0.83, blue 0.33 → 0.88. Blue width agreement also improved, 0.71 → 0.99.
- In other words, the loop mostly learned to stop pooling paint, which is an explicit mechanism of the hand-written pigment model (§3.6).

**Determinism.** Blue parameters were identical in 03 and 04, and the blue metrics are identical to 3 decimals. The simulator is deterministic, so each score is a single sample without noise, and no robustness to perturbation was tested.

### 5.2 Red flags in the numbers

1. **The registration shift saturates at the search bound in all 4 runs** (`alignment_shift_px: [3, 3]`, tolerance 3).
   - The evaluator docstring claims the bounded shift exists "so alignment cannot conceal a large trajectory offset". ~~A shift pinned at the corner of the search box suggests a systematic homography/IK/brush-tip offset of **≥3 px (≳0.8–0.9 mm)** that the metric partly absorbs.~~ [corrected: the offset is real but sits in the *camera/rectification path*, not in paint placement. Running the same frozen `analyze()` on the simulator's top-down pigment texture (`results/attempt-0{1,4}/painting.png`, same 512×512 canvas convention) gives a best shift of **[0, 0]** at both tolerance 3 and tolerance 6. The brush therefore deposits exactly where planned. Three rectified pixels ≈ 0.8–0.9 mm ≈ **one camera pixel**. Detected fiducial centroids deviate from the true projections of the fiducial tops by −0.4…−0.6 px in x and +0.1…+1.0 px in y; this is the fact-checker's projection through the MuJoCo `front` camera model, and is consistent with a pixel-centre convention plus the visible side of each 2 mm-tall cyan cylinder. That deviation is the likely source of the offset (hypothesis).]
   - Raw unaligned IoU is only **0.73–0.78 (brown)** and **0.54–0.60 (blue)**.
2. **The evaluator was co-designed with the data.** It was "finalized after diagnosing the first two paintings and frozen before assessing attempts 03 and 04" (`learning-report.md:30`). Attempts 01→02 are therefore not a blind comparison.
3. **n = 1 per configuration**, with no seed or pose perturbation.
4. **Privileged signals.** These are honestly disclosed: paint load (`paint.load<.35` reload), brush colour, and contact counters. Quoting `README.md:88`: *"This is not camera-only control. A physical implementation would need corresponding sensing or estimates for these quantities."*

### 5.3 My re-run (Linux, CPU EGL/llvmpipe, Python 3.11.14 instead of the pinned 3.11.15)

**Environment.** I ran `uv sync --locked --python 3.11`. The pinned 3.11.15 was not obtainable, so this used 3.11.14 with the locked deps: `mujoco==3.13.0`, `numpy==2.4.6`, `opencv-python-headless==5.0.0.93`, `pillow==12.3.0`.

**Full runner with video.** `MUJOCO_GL=egl .venv/bin/python scripts/run_attempt.py --run-dir repro/attempt-04-rerun --parameters results/attempt-04/parameters.json --headless --fast --port 8891` was far too slow with CPU rendering: 30-fps 960×720 frames with a 4096 shadow map gave ~~about 3 strokes per 10 min~~ [corrected: `repro/attempt-04-rerun/controller.log` shows 9 strokes completed, through `tower-right-foot`, plus 3 loads. The file times span 22:16:16 → 22:25:20, i.e. ≈1 stroke/min. The no-video run appears to have run concurrently, which may have slowed it]. I stopped it. Its first strokes' pixel counts matched the original within about 1%.

**No-video run.** A copy of `paint_paris.py` with only `/record/start`/`/record/stop` removed completed in **4 min 06 s wall**: 19/19 strokes, 0 warnings. The frozen evaluator (SHA verified) gave:

| | Published attempt 04 | My re-run |
|---|---:|---:|
| Score | 96.26 | **96.29** |
| Brown / blue | 95.24 / 97.28 | 95.32 / 97.27 |
| Brown width px | 19.60 | 19.58 |
| Registration shift | [3, 3] | [3, 3] |
| `contact_frames` / loads / max F | 3406 / 21 / 0.206050 N | 3406 / 21 / 0.206050 N (identical) |
| Fiducial pixels | — | identical to 4 decimals |

**Conclusion.** ~~The **physics is bit-reproducible across macOS→Linux**.~~ [corrected: the physics is *numerically near-identical*, not bitwise identical. `contact_frames` (3406), load count (21) and fiducial pixels are identical. `maximum_contact_force_n` differs in the 13th significant digit (0.20605036086917378 vs 0.20605036086904663), and `paint.deposited_amount` differs at ~3e-10 (128.28755760630 vs 128.28755763817). Final sim `time` differs a lot (121.32 s vs 164.14 s) because the server keeps stepping while idle between commands.] Only rendering differs slightly. The published result is genuine for this code.

**Probing the saturated registration.** I re-ran `analyze(..., tolerance_px=6)`, its maximum. The best shift moves to **[4, 5] px** for both attempts 01 and 04, which is about 1.1 mm × 1.5 mm on the canvas. ~~There is a systematic placement offset that the frozen metric clips at 3 px. It is probably the brush-tip/contact-centroid offset; I did not diagnose it further.~~ [corrected: there is a systematic *camera-rectification* offset that the frozen metric clips at 3 px. It is not a placement or brush-tip offset: the ground-truth pigment texture registers at [0, 0] (see §5.2 correction). At tolerance 6, attempt 02 gives [4, 4], not [4, 5].] (A wider tolerance also loosens Dice, so those scores, 84.78 and 97.08, are not comparable.)

Outputs are saved in `research/repos/so101-painting/repro/attempt-04-novideo/`.

---

## 6. Safety, error handling, logging

**Safety:**
- `ctrlrange` clipping and numeric validation in the server;
- an IK reachability and orientation tolerance check;
- the gripper `forcerange` is reduced to ±0.4;
- a missed-contact postcondition;
- a hard fail on any MuJoCo warning (`paint_demo.py:41-42`, `paint_paris.py:108`).

**What is missing:**
- No velocity, acceleration or jerk limits. Durations come from the plan; 0.9 s approaches are fixed.
- No workspace keep-out zones beyond IK failure.
- No force limit on the brush. Force is bounded only implicitly by the spring.

**Error recovery.** None: any exception aborts the attempt. The `finally` block still writes `run-report.json` and stops the video.

**Logging is excellent.** Each attempt keeps:
- `timeline.json`: 190 sim-timestamped events in attempt 04;
- ~~`work/actions.jsonl` with every command result;~~ [corrected: `actions.jsonl` is *not* per-attempt and is *not* in the bucket. The server appends to a single shared `ROOT/work/actions.jsonl` and writes frames to `ROOT/work/NNNN-camera.png` (`painting_server.py:30, 91, 236-237`), across all runs. The local `work/` directory was produced by the fact-check re-run; the bucket tree has no `work/` path.]
- `run-report.json` with per-stroke paint state;
- a 30-fps x264 video plus a ≤30 s 1080p edit (`edit_video.py`);
- `parameters.json` with a free-text **`reason`**, which serves as a hypothesis log.

---

## 7. Related work and competition (2026-09)

| Project | LLM role | Runtime | Notes |
|---|---|---|---|
| thijs @cdngdev, real SO-101 Golden Gate (2026-09-08) | Astra in the loop. Reportedly told to "plan out 1 minute of actions and monitor it in the background" (search snippet; **UNVERIFIED**), with human feedback between attempts | Real arm, one camera | 5 attempts (Understanding Robots). Roboflow: *"Thijs gave it starting reference points and feedback between attempts."* Code not public |
| **mishig/so101-painting** (this) | Astra as **offline coding agent and parameter tuner** | MuJoCo, scripted strokes | Fully public artifacts, no LLM transcript |
| riyer8/astra-paints (2026-09-11) | **Astra as runtime stroke planner.** One `chat.completions.create(model="gpt-6-astra", response_format={"type":"json_object"}, max_completion_tokens=2000)` per round, with the canvas PNG attached and a reply of `{"strokes":[{"color":[r,g,b],"points":[[x,y],…]}],"done":false}`, 3–8 strokes per reply (`agents/astra_agent.py:39-81`) | Custom sim arm | Prices hard-coded as $10/$50 per M tokens (**UNVERIFIED**). Bull → "beetle" → "donkey" → bull over 3 attempts, judged by an LLM |
| MrDee @sog_on_bird_app (2026-09-16, via Roboflow) | Real arm + pen + 2 cameras | — | *"Gave GPT 6 Astra access to my robot arm … but it couldn't do it"*. Grok 4.6 also failed. *"Brough out Fable 5.1 and it one-shotted it"* (single anecdote) |
| EmbodiedSWE (teammate note `embodiedswe.md`) | Coding agents write `solve(env)` programs | Isaac Lab | Same paradigm at benchmark scale: Astra/Codex 82% vs Fable 5.1/Claude Code 61% vs Opus 5/Claude Code 50% success |

---

## 8. Assessment

### Strengths
- **It is honest and inspectable.** Everything that produced the video is present and checksummed, and the claims are scoped carefully: no RL, a privileged state list, metric limits, the viewer crash. That is rare among the viral "Astra controls a robot" posts of September 2026.
- **The experimental discipline is good:**
  - immutable per-attempt directories;
  - one stated hypothesis per change in `parameters.json["reason"]`;
  - ~~a mostly one-factor-at-a-time search;~~ [corrected: *no* transition was single-factor. 01→02 changed 5 settings: easing smooth→linear, settle 10→0, touch 0.65→0.35 s, lift 0.65→0.30 s, `atomic_strokes` on. 02→03 changed 2: speed 35→17.5 mm/s and reload ≤4. 03→04 changed 2: brown depth 1.5→0.75 mm and brown speed 17.5→14 mm/s. This happened even though the evaluator's own caution says "Change one factor per attempt".]
  - a frozen evaluator pinned by hash;
  - every regression kept.
- **A clean motion API.** A validated `/trajectory` with explicit timestamps lets a high-latency planner emit an *atomic* approach–contact–sweep–lift primitive. ~~That primitive alone took the score from 83 to 93: it removed the dwell artifacts that came from planning pauses mid-contact.~~ [corrected: the 83→93 jump came from a *bundle* of 5 changes (see above), so it cannot be attributed to the primitive alone. The dwell it removed was mostly the 0.16 s settle frames plus smoothstep slow-downs at the stroke ends; client planning gaps were only 1–3 ticks (16–48 ms) in attempt 01's timeline.] This is precisely the failure an LLM-in-the-loop harness will hit on real contact tasks.
- **It runs end to end** on a laptop with `uv sync --locked`, with no robot, GPU or model download needed.

### Weaknesses and limits
- **Not LLM control.** The runtime is a 19-stroke hard-coded plan. Astra's contribution (authoring the code and choosing 4 parameter edits) is real but invisible: no prompts, turns, tokens, cost or wall-clock. For research purposes the *interesting* half is missing.
- **Simulation idealizations:**
  - perfect kinematics, no backlash;
  - deterministic contact;
  - a 2-D pigment model whose pooling term is exactly what got "learned";
  - privileged paint-load, brush-colour and contact signals.
- **Weak metric:**
  - co-designed after 2 of 4 runs;
  - geometry terms saturated at about 1.0;
  - registration pinned at the bound;
  - n = 1;
  - an evaluated camera view of only ~~about 290×190 px~~ [corrected: ≈152–167 px per canvas edge (≈22.7k px²), see §4].
- **Logging bug.** The per-stroke "camera check" has no brown class (brown is logged as orange) and is never used for control.

### What is novel vs repackaged
- **Not novel:**
  - homography-from-fiducials;
  - DLS IK;
  - scripted polylines;
  - spring-brush contact;
  - the optical-density pigment model ~~(a standard Kubelka-Munk-lite approach)~~ [corrected: it is a Beer–Lambert transparent-glaze model, `reflected = paper·exp(−density·(1+0.07·wetness))` with per-channel absorption `−log(rgb)` (`paint_surface.py:40-53, 165, 210-215`). There is no Kubelka-Munk scattering/absorption (K/S) term].
- **Mildly novel (as a 2026 artifact):**
  - an **agent-authored, agent-continuable experiment package**, with a README prompt addressed to the next coding agent plus a frozen evaluator for comparability;
  - publishing it as an HF *Bucket* rather than a git repo.
- **Maturity:** a one-day demo. It has not been tested on Linux/Windows upstream, and the README still requires `uv` with a Python patch version that my uv 0.10.4 could not fetch.

### What Ilia's Opus-backbone harness should borrow
1. **Two-timescale split.** Put Opus at the episode/stroke level as planner, critic and parameter tuner. Keep a deterministic executor below it. Express LLM outputs in a **task-frame representation** (normalized canvas (u, v) polylines plus per-primitive params), not joint angles.
2. **Atomic timed primitives with postconditions.** Example: `trajectory(targets, times)` plus `assert contact_frames increased`. Never let model latency fall inside a contact phase.
3. **Experiment ledger as a first-class harness feature.** Each episode gets `parameters.json` with a `reason`, the raw video, a timeline, a frozen evaluator hash, and keep-best-on-regression. This also generates clean (hypothesis, change, outcome) data for the later learned action head.
4. **A strict local tool API.** The server-side validation in §3.2 maps 1:1 onto tool `input_schema` constraints plus server checks.
5. **Self-calibration via PnP on an end-effector marker**, from the sibling bucket. This is cheap, and it gives a quantitative residual (RMS px) that the LLM can reason about.

### What to avoid
- **Do not trust single-run scores** on metrics designed after seeing the data. Freeze the evaluator *before* the first episode and run ≥5 seeds or perturbations, especially on real hardware.
- **Do not rely on privileged sim signals** (paint load, contact counters) unless there is a real-sensor substitute: a wrist camera, servo current, or a force proxy.
- **Do not judge fine-detail tasks from an oblique, low-resolution view.** Add a top-down or wrist view, or the rectified view will cap what the critic can see.

---

## Sources

- https://huggingface.co/buckets/mishig/so101-painting (page, `/api/buckets/mishig/so101-painting`, `/tree?recursive=true`, `/resolve/<path>`)
- https://huggingface.co/api/buckets/mishig (bucket listing) and https://huggingface.co/buckets/mishig/so101-block-sorting
- https://huggingface.co/docs/hub/storage-buckets ; https://huggingface.co/docs/huggingface_hub/guides/buckets ; https://huggingface.co/blog/storage-buckets
- https://x.com/mishig25 (HTTP 402 / not renderable headless; post text via search snippets: **UNVERIFIED** verbatim)
- https://x.com/cdngdev/status/2097339677128982873 (text and likes via `cdn.syndication.twimg.com/tweet-result`)
- https://blog.roboflow.com/gpt-6-astra-vision/
- https://www.understandingrobots.org/p/openais-astra-model-is-shockingly
- https://github.com/riyer8/astra-paints (`agents/astra_agent.py`)
- https://github.com/zjwzcx/Awesome-Astra-Embodied-AI (Case 20)
- https://github.com/TheRobotStudio/SO-ARM100 (upstream MJCF, commit `eecbe3e0a9ebb23e25ad7b2759b03884c6660903`)
- https://mishig25.github.io/
- Teammate note: `research/sources/embodiedswe.md`

---

## Verification (fact-check pass)

*Adversarial re-check done 2026-10-01 against these primary sources:*
- *the live bucket API (`/api/buckets/mishig/so101-painting`, `/tree?recursive=true`, `/api/buckets/mishig`, `/api/users/mishig/overview`);*
- *the local bucket copy `research/repos/so101-painting/` (115/115 non-video files re-hashed against `MANIFEST.sha256`; the 10 `.mp4` files are not local);*
- *the sibling `research/repos/so101-block-sorting/` and `research/repos/astra-paints/`;*
- *the X syndication API, the Roboflow blog and Understanding Robots;*
- *re-execution of the frozen evaluator in the bucket's `.venv` (mujoco 3.13.0, Python 3.11.14).*

### Confirmed (seen in primary source)

**Bucket metadata and author**
- Bucket metadata (bucket API): created 2026-09-14T21:20:41Z, updated 21:22:01Z, 126 files, 127,434,417 bytes.
- `/api/buckets/mishig` lists 8 buckets, none for Mona Lisa. `so101-block-sorting` was created 14:48:09Z.
- Author: `isHf: true` on the bucket page, 402 followers. mishig25.github.io says "Since 2021, building the Hugging Face Hub".
- `/resolve/<path>` returns 200 and `/resolve/main/<path>` returns 404. The storage-buckets docs confirm buckets are "non-versioned and mutable", S3-like and Xet-backed.
- The thijs tweet (syndication API): created 2026-09-08T15:02Z, 20,724 likes, text as quoted in §1.
- Sibling `REPRODUCIBILITY.json`: Darwin 25.4.0 arm64, Python 3.11.15, `rms_px` 0.13887, `inliers` 16 / `observations` 16. `calibrate_camera.py` has 13 poses (`:38-45`), uses `solvePnPRansac(... SOLVEPNP_EPNP, reprojectionError=3.)` then `solvePnPRefineLM`, and takes intrinsics from fovy 42° (`:60-69`).

**Quotes, scores and results**
- All quotes from `README.md:7`, `:88`, `learning-report.md:20, :30, :32, :34, :42` and `paris_design.py:20-24` are verbatim.
- Every score and sub-score in the §5.1 table matches `evaluation.json`, `learning-history.json` and `physics-report.json`: 83.37 / 92.97 / 95.62 / 96.26; brown/blue sub-scores; endpoint:middle OD ratios; widths 17.58 / 18.39 / 21.58 / 19.60; video 104.0 / 75.47 / 113.6 / 119.13 s; max F 0.2117 / 0.2075 / 0.2061 / 0.2061 N; 0 warnings.
- `alignment_shift_px` is [3, 3] in all four runs. Raw unaligned IoU is brown 0.730 / 0.745 / 0.775 / 0.763 and blue 0.541 / 0.538 / 0.604 / 0.604.
- Brown pigment uniformity 0.314 → 0.829, blue 0.329 → 0.883. Blue width agreement 0.706 → 0.988.
- Border-connected brown pixels ignored: 2615 / 2636 / 2637 / 2637.
- Re-running the frozen evaluator reproduces every published score exactly. With `tolerance_px=6`, attempt 01 gives 84.78 at [4, 5] and attempt 04 gives 97.08 at [4, 5].
- The evaluator SHA-256 equals `feedc58a77b40df49aceef666b859f8494512e05ebc0dc3fffe0c97629f1314f`. All 4 `stroke-plan.json` files are byte-identical (md5 `5e14e513…`).
- The plan has 19 strokes: 13 brown, 1 orange (68 points), 2 green, 3 blue.

**Code claims (all checked against the files)**
- `painting_server.py` (256 lines):
  - endpoints `:103-171`; validation `:136-147`;
  - settle frames clamped 0–20 (`:150`); `dt=.016` with `mj_step(nstep=8)` at `timestep=.002`;
  - auto-capture plus reply `:234-239`; state dict `:79-84`; 0.002 N contact threshold `:199`.
- `paint_demo.py`:
  - HSV 85–99 / S>130 / V>120 and area 5–1500 (`:66-89`);
  - `load()` asserts `brush_color` and `load>=.9` (`:109-121`);
  - `inspect()` has no brown bin, and brown is logged as `orange` in `controller.log`.
- `paint_paris.py`: reload rule `:100-103`; atomic stroke `:47-76`, with 16+16 smoothstep samples, 2 mm sweep sampling and the postcondition at `:75`.
- `arm_kinematics.py` (146 lines):
  - FD Jacobian with ε 1e-5; DLS `lstsq([J; 0.001·I])`; 0.18 rad step cap; backtracking {1, .5, .25, .10, .025}; 160 iterations; accept ≤0.8 mm / ≤0.006;
  - pan/roll seeds `:81-93`.
  - `diff` against the block-sorting copy shows only docstrings and the default `grip` (0.05 vs 0.6) differ.
- `paint_surface.py`: footprint radius, dose formula, depletion and the 10 Hz step all as stated.
- `evaluate_painting.py` (424 lines): score formula `:238-239`, uniformity `:153-202`, suggestions `:265-295`, registration `:114-126`.
- `build_scene.py` and `scene.xml`:
  - upstream commit `eecbe3e0…`; brush slide joint 0–9 mm, k=120, damping 0.4; bristle friction 0.35;
  - camera at (0.65, −0.65, 0.65) looking at (0.18, 0, 0.04), fovy 30; canvas 135×150 mm; wells at y=−0.12;
  - gripper `forcerange` set to ±0.4 (`build_scene.py:58`); `backlash` default class defined but used by no joint.
- `run_attempt.py`: claims at `:32-39`, `:88-93`, `:96-101`, `:120-138` and `:153-167` all as stated.

**Re-run and related work**
- The re-run is real: `repro/attempt-04-novideo/evaluation.json` gives 96.29 (95.32 / 97.27), width 19.575, shift [3, 3], 3406 contact frames, 21 loads, identical fiducial pixels. The venv is CPython 3.11.14 with uv 0.10.4.
- astra-paints `agents/astra_agent.py`:
  - `model="gpt-6-astra"`, `response_format={"type":"json_object"}`, `max_completion_tokens=2000` (`:76-81`);
  - prompt `:39-49`, "3-8 strokes per reply";
  - `PRICE_IN, PRICE_OUT = 10.0, 50.0` (`:84`);
  - bull → beetle → donkey → bull per the README, with the first result directory dated 20260911.
- The Roboflow quotes (MrDee, Grok 4.6, "Brough out Fable 5.1 and it one-shotted it", "Thijs gave it starting reference points and feedback between attempts") appear on the 2026-09-18 page.
- Understanding Robots: "After five iterations".

### Corrections (applied inline above; 19 distinct)

1. **TL;DR thijs quote.** "astra paints the Golden Gate Bridge with an SO-101" is a paraphrase, not the tweet text.
2. **TL;DR "only learning = parameters.json".** Controller code (`atomic_stroke`) and the evaluator also changed between runs (`learning-report.md:3`, `:30`).
3. **`hf buckets … ls`.** The docs document `list`.
4. **"grep `model=` returns nothing".** It hits `build_scene.py:69` (MJCF attribute). Also, no file in the bucket mentions Astra, GPT or OpenAI.
5. **"never debounced; every dip logs 3".** Loads fire on the rising edge of palette contact. Attempt 01's orange dip logged 4 events.
6. **"Planning latency caused the endpoint pools."** The timeline shows a 0.016–0.048 s touch→sweep gap. Most of the dwell came from `settle_frames=10` (0.16 s) plus smoothstep easing.
7. **Evaluator SHA suffix.** It is `…f1314f`, not `…f314f`.
8. **"Only brush↔paper/palette collide."** Robot collision meshes still collide with the floor, the desk and each other.
9. **"Canvas ≈ 290×190 px"** (§4 and §8). That is the fiducial bounding box. The canvas quadrilateral has ≈152–167 px edges and ≈22.7k px², i.e. ≈0.9 mm per camera pixel.
10. **§5.1 "Loads" column.** It counts palette-contact events. Dips were 4 / 4 / 7 / 7.
11. **"Centerline coverage 1.0 throughout."** Only brown is 1.0; blue is 0.995.
12. **§5.2 red flag 1.** The [3, 3] shift is a camera/rectification offset, not a placement offset. The ground-truth `painting.png` registers at [0, 0]. Three rectified px ≈ one camera px.
13. **"About 3 strokes per 10 min".** The stopped full re-run completed 9 strokes in ≈9 min.
14. **"Bit-reproducible".** The results are near-identical: max force differs at about 1e-13 relative and deposited amount at about 3e-10, and final sim time differs (121.3 vs 164.1 s).
15. **"Probably brush-tip/contact-centroid offset".** It is in the rectification path, not placement. Also, attempt 02 at tolerance 6 gives [4, 4].
16. **"`work/actions.jsonl` kept per attempt".** It is a single shared, appended scratch file at the repo root and is not in the bucket.
17. **"Mostly one-factor-at-a-time".** No transition was single-factor (5, 2 and 2 changes).
18. **"That primitive alone took 83→93".** Attempt 02 bundled 5 changes.
19. **"Kubelka-Munk-lite".** It is a Beer–Lambert transparent-glaze model with no K/S scattering.

### Unverifiable / still UNVERIFIED

- **All @mishig25 posts.** This covers the Eiffel Tower, the LEGO "calibrates itself", the Mona Lisa and "I keep hitting my astra token limits" posts. The syndication timeline returned only old popular posts, and web search shows snippet-level text only. **The bucket itself never names GPT-6 Astra or any model**: "agent-guided" (`learning-report.md:34`) and the first-person "I inspected" (`:3`) are the only agent references. The Astra attribution therefore rests on the X post alone.
- How the agent was invoked (Codex/ChatGPT, reasoning effort, tokens, cost, wall-clock, turns). Nothing is in the bucket.
- The thijs "plan out 1 minute of actions and monitor it in the background" quote. It does not appear in the Understanding Robots article (Kai Williams, dated 2026-10-01).
- The "4 min 06 s wall" time of the no-video re-run. All files in `repro/attempt-04-novideo/` share a single copy timestamp (22:25:42), so no timing evidence remains.
- Whether the four recorded runs are the original between-inspection runs or a batch re-execution (see provenance below).
- EmbodiedSWE numbers (82% / 61% / 50%). These were checked only against the teammate note `embodiedswe.md:21-23`, not the paper.

### Added missed details

**1. Provenance of the recorded runs.**
- The original attempts were launched with `uv run --no-project`, not via the published `run_attempt.py`:
  - Every published `results/attempt-0*/server.log` and `controller.log` starts with `WARN \`--no-project\` was provided, but no project was found`. That is a uv warning string, found in the uv binary.
  - `run_attempt.py` spawns its children with `sys.executable`, so it cannot produce that line. The fact-check re-run logs indeed lack it.
- Bucket `mtime`s (client file times kept by `hf buckets sync`):
  - `uv.lock` 16:46:42Z, i.e. *after* attempt 01's camera.png at 16:46:12;
  - `pyproject.toml` and **all scripts** 21:20:26Z (release packaging).
- The four runs finished within ≈6.5 min: camera.png at 16:46:12, 16:47:41, 16:49:42 and 16:52:40.
- `attempt-01/preview.mp4` (16:48:12) was rendered while attempt 02 was running. That leaves only 1.5–3 min per inspect → edit → run cycle, including finalizing the evaluator after run 02.
- `RELEASE-NOTES.md:5` promises unchanged "images, videos, measurements, controller parameters, and the frozen camera evaluator". It does **not** promise unchanged controller or server code.
- The published runner and README workflow are therefore post-hoc packaging. The near-exact attempt-04 re-run shows the packaged code is functionally equivalent for attempt 04 at least.

**2. Only brown and blue are scored.**
- The orange sun and green banks (3 of 19 strokes, including the 68-point spiral) are excluded from the metric entirely (`evaluate_painting.py:7, 24, 339-351`). Other planned-colour regions dilated by 8 px (16,160 px) are masked out.

**3. The simulator is free-running.**
- The server loop calls `mj_step` and paint stamping every tick, whether or not a command is active (`painting_server.py:125-244`).
- With `--fast` and recording on, it does not sleep at all while idle (`:240-244`).
- Any client latency while the brush is in contact therefore becomes paint dwell, and sim time depends on host speed: attempt 01's first dip happened at sim t=232.8 s versus 9.96 s in attempt 04, and the re-run's final time was 121.3 s versus 164.1 s.
- Painting outcomes are deterministic only because atomic strokes keep all client compute in the hover phase.

**4. Action interface details (relevant to an LLM tool schema).**
- **Joint space only:** 6 absolute joint targets in radians, clipped to `ctrlrange`. There is no Cartesian endpoint; IK is client-side.
- **Blocking:** a POST blocks until motion plus settle completes, with a 180 s server-side reply timeout (`:114`).
- **One command at a time:** a single queue serves all commands, and there is **no stop/abort/preempt endpoint**.
- **Image on every reply:** every motion reply carries a fresh 960×720 front-camera PNG path plus full privileged state, and is logged to `work/actions.jsonl`.
- **Easing ignored with `times`:** when `times` is supplied, `easing` is ignored (`:180-183`).

**5. Contact force is set by geometry.**
- The force follows from geometry, not force control. F ≈ k·depth = 120 N/m × 1.5 mm ≈ 0.18 N; the observed max is 0.206–0.212 N.
- With the stamp model, radius = 1.8 + 2.3·√(F/1.2) mm ≈ 2.6–2.8 mm. The brown 0.75 mm setting gives ≈0.09 N nominal.

**6. Evaluator advice was not followed literally.**
- Attempt 01's blue suggestion (+0.25 mm depth) was not applied.
- Even attempt 04 still triggers the brown "Wide or dark endpoint paint pools are visible; shorten stationary touch/lift dwell" suggestion.
- The published `suggestions` were recomputed with the frozen evaluator. What the agent actually saw before run 02 is unknown.

**7. Score sensitivity** (fact-checker runs of the frozen `analyze()`).

| Setting | 01 | 02 | 03 | 04 |
|---|---:|---:|---:|---:|
| `tolerance_px=0` | 72.50 | 82.80 | 86.34 | 86.49 |
| `tolerance_px=6` | 84.78 | 94.42 | 96.25 | 97.08 |
| Ground-truth `painting.png`, tol 3 | 85.94 | — | — | 96.97 |

- At `tolerance_px=6` the shifts are [4, 5] / [4, 4] / [4, 5] / [4, 5].
- The ground-truth texture registers at shift [0, 0].
- The ranking is preserved under every setting, so the improvement holds on ground truth and not just through the camera.

**8. The camera is the resolution bottleneck.**
- The canvas spans ≈0.9 mm per camera pixel, so a 4.5 mm stroke is ≈5 camera pixels wide.
- Detected fiducial centroids are biased by up to ≈1 px relative to the true projections of the fiducial tops. This is the likely source of the saturated shift.

**9. Sibling calibration caveat.**
- The reported "16/16 inliers" is tautological. After the board-fiducial step, `idx = np.arange(len(xyz))` (`calibrate_camera.py:85`), so "inliers" counts all observations, not RANSAC inliers.
- The intrinsics come from the known simulated fovy (privileged), and the render is noise-free, so 0.139 px RMS is expected rather than evidence of robustness.

**10. Unused capabilities.**
- The painting scene already carries the magenta `vision_marker` sphere on the gripper (`build_scene.py:41-42`), but it is unused for painting.
- `paris_design.design()` exposes validated knobs (`tower_width` .34–.50, `crossbrace`, `detail`, `river_spacing` .034–.057) and an `estimated_seconds()` planner. None were ever varied.

**11. The evaluator is not part of the runner.**
- `run_attempt.py` never calls `evaluate_painting.py`. Scoring is a separate CLI step: `evaluate_painting.py <camera-canvas.png> <stroke-plan.json> --output --diagnostic`.

**12. Packaging detail.**
- `uv.lock` records the HF-internal index `https://pypi.registries.huggingface.tech/`, which is not publicly reachable (curl returned status 000).
- The wheel URLs point to `files.pythonhosted.org`, so `uv sync --locked` still works. This is further evidence the package was built inside HF infrastructure.

**13. Context on the source demo.**
- Understanding Robots (2026-10-01, Kai Williams) describes thijs as "OpenAI robotics employee Thijs Simonian". The demo Mishig reproduced is therefore an OpenAI-staff showcase.
- The same article attributes "five iterations" to it and does not name the arm model.
