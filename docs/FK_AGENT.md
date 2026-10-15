# Few-shot forward-kinematics agent (SO-101)

An agent that picks an object ("take the carrot from the table") by choosing **joint angles**.
Each step a vision LLM sees the episode's history and proposes the next target:
- the history holds every step's joints, motion and wrist photo, plus the FK pose computed from them;
- the code checks the target and executes it;
- the arm photographs the result and returns to the start pose.

Optional few-shot examples are teleoperated demo sessions in the same format.

| Code | What |
|---|---|
| `controlr/fkagent/geometry.py` | joint offsets, FK in mm/deg, joint sensitivity table, `Limits`, `plan_target` (constraints + clamping), approach/lift paths |
| `controlr/fkagent/dataset.py` | recordings → sessions: grasp detection, key points, frame check, export |
| `controlr/fkagent/prompt.py` | system prompt, few-shot rendering, history rendering, reply parser |
| `controlr/fkagent/agent.py` | `ServerArm` (SO-101 server protocol in MODEL degrees), `FkAgent` loop, per-episode logs |
| `controlr/fkagent/offline.py` | offline leave-one-out evaluation and report |
| `scripts/so101_fk_dataset.py`, `scripts/so101_fk_agent.py` | CLIs |
| `tests/test_fkagent.py` | unit tests (fake LLM + in-process fake server) |

## Design

**Joint frames.** Recordings and the server use LeRobot DEGREES. The agent sees and proposes only
MODEL angles: `model = measured + [0, -5.07, -10.06, -1.36, 0]` (the fitted joint offsets,
`configs/so101.yaml`). `ServerArm` converts back the same way `So101Robot` does.

**Action space: joint targets.** The model outputs `q_deg` (5 joints) plus a jaw opening. That
is the spirit of the idea: the model does the kinematics itself, helped by FK numbers.
- Every observation shows the TCP computed by FK, the tool pitch, and a "d TCP per +1 deg"
  table: the local Jacobian, which tells the model the joint signs at that pose.
- The model also reports `expected_tcp_mm`, so the logs measure how well its own kinematics
  match FK (offline: median 8–17 mm).

A TCP-target mode via IK was not built. It would turn the experiment into "LLM as a 3D
localiser + IK", which is a different question.

**One step:**
1. Request = system prompt, then few-shot sessions (byte-stable, cacheable), then the history,
   then the reply instruction.
2. Reply = one JSON object: object `xyz_mm` with an uncertainty radius, `action`
   (`move` | `grip` | `give_up`), `q_deg`, `gripper_open_pct`, `expected_tcp_mm`, `grasp_q_deg`,
   `reason`. A bad reply is re-asked once with the parse error.
3. `plan_target` checks the target:
   - joint limits are clipped (the server's calibrated limits, shifted into the model frame);
   - a target outside the box, too far from the previous target, or with an unsafe path is
     pulled back along the joint line from the previous target (or the start pose) to the
     farthest safe fraction;
   - below 5 % of the way, the target is refused (no motion; the model is told).
4. Path: start pose → joint-linear to 40 mm above the target → straight down (IK every 5 mm,
   same tool orientation).
   - Every path is densified to 1° and checked: TCP in the box (x 140–270, y ±150,
     z 5–260 mm), elbow/wrist origins ≥ 20 mm above the table, joint limits.
   - Step limit: ≤ 80 mm of TCP motion and ≤ 35° per joint between consecutive targets.
5. Photo at the target.
6. `grip`: close to 0 %. Jaws stalled more than 2.5 % above the command (or the server's
   `gripper_blocked`) = grasp. A grasp is lifted 50 mm straight up and photographed again;
   still holding = success, and the episode ends. An empty close is reported, and the jaws reopen.
7. Return to the start pose along a checked path: the approach path reversed, or for a held
   object lift → start, falling back to lift → above → start. If no checked path exists, the
   arm stays where it is (outcome `stuck`).

A server STOP (contact) ends the episode with the arm frozen.

**Start pose.** The demos' start pose (folded rest, TCP ~150/5/17 mm) shows only the floor
next to the base, so no object is ever visible from it. The live start pose is therefore a
survey pose, `SURVEY_Q_DEG = [0, -50, 20, 90, -90]` (model deg):
- TCP ≈ (195, 8, 145) mm, pitch 60°;
- the camera is ~270 mm up, and its stock-mount footprint covers about x 100–380, y ±240 mm.

Check this view with `survey` before the first episode (see the protocol below).

**Logs**, one directory per episode under `runs/fk_agent/<time>_<fs|zero>_<name>/`:
- `config.json`: the config, the few-shot ids and the limits;
- `steps.jsonl`, one line per step: the decision, the plan (clamps and notes), the move, close
  and lift results, joints reached, FK pose, the LLM calls with usage and reply text, and
  `tcp_to_object_mm` / `estimate_error_mm` when `--object-xyz` is given;
- `images/step_NN.jpg` (plus `_lift`);
- `prompts/step_NN_k.json`: the exact request, with images redacted;
- `summary.json`: outcome, success, steps, motions, grips, LLM calls, final TCP and the
  final TCP–object distance.

## Dataset (few-shot source)

```bash
# recordings: read-only download from the rig (see the task notes), e.g. into /tmp/fkdata/<DIR>
uv run python scripts/so101_fk_dataset.py /tmp/fkdata/gym_objects_demo_2 /tmp/fkdata/ball_demo_4 \
    /tmp/fkdata/gym_objects_demo /tmp/fkdata/ball_demo_3 --labels runs/fk_dataset/labels.json --out runs/fk_dataset
```

**Grasp detection** (gripper only):
- A grasp is the follower's jaws standing still above the leader's command, `F − L > 1.2`,
  for ≥ 0.4 s.
- A stall below 1.5 is flagged `weak_stall`. The thin marker stalls at only ~1.4.
- Grasp moment = jaw contact (the start of the stall).
- An empty close (`miss`) = F dropped ≥ 8 within 1.5 s and ended within 1.2 of L, below 8 %.

**Frame check:** between the grasp and lift frames, the jaw region of the image should change
much less than the background, because the object moves with the camera. A ratio < 0.8 counts
as consistent: ball 0.72 / 0.37, carrot 0.47. The thin marker fills too little of the jaw
region and fails (1.00), but its contact sheet shows it held.

**Key points** (t = 0 when the arm leaves the start pose: the last sample with
shoulder_lift ≤ −90°):

| Key point | Where |
|---|---|
| `start` | leaving the start pose |
| `approach` | TCP ≥ 150 mm from the grasp |
| `pre_grasp` | TCP ≥ 60 mm from the grasp, before the first close |
| `miss` | each empty close |
| `grasp` | jaw contact |
| `lift` | +40 mm while holding |

Each key point carries MODEL joints, the joint delta from the previous key point, the gripper,
the FK TCP and pitch, and the nearest photo (downscaled to 640 px).

Outputs:
- `sessions.json`;
- `images/`;
- `sheets/<session>.jpg`: contact sheets to verify the key points by eye.

Grasps that start while already holding the object (set-down + regrip) are dropped unless
`--keep-regrips` is passed.

**Current data:** 4 sessions. `gym_objects_demo` and `ball_demo_3` contain no grasp;
`ball_demo_1` is a different format.

| session | object | grasp TCP (mm) | pitch | opened → stall |
|---|---|---|---|---|
| gym_objects_demo_2#1 | rope ball | (270, 26, 16) | 90 | 37 → 26 |
| gym_objects_demo_2#2 | plush carrot (1 miss before) | (280, −85, 24) | 83 | 53 → 5.5 |
| gym_objects_demo_2#3 | whiteboard marker | (296, 92, 2) | 74 | 32 → 9.6 |
| ball_demo_4#1 | rope ball | (237, 2, 14) | 70 | 36 → 26 |

Two things limit how far these examples transfer:
- The jaws sit at different image positions in `ball_demo_4` (centre) and
  `gym_objects_demo_2` (left), so the camera mount changed between recordings.
- The stock URDF camera model projects the TCP ~0.15–0.25 of the image width away from the
  real jaw midpoint.

So pixel → ray geometry is not usable without a hand-eye calibration, and the agent gets no
projection hints.

## Offline evaluation (no arm)

```bash
set -a; . ../controlr/.env; set +a
uv run python scripts/so101_fk_agent.py offline --dataset runs/fk_dataset --out runs/fk_offline/run1
uv run python scripts/so101_fk_agent.py prompt --dataset runs/fk_dataset --session gym_objects_demo_2#2   # no call
```

**Setup:**
- Query = one held-out session at one view (`approach` ~150 mm from the grasp, `pre_grasp`
  ~60 mm), given as "step 0" (joints + photo), with the task "pick up the <object>".
- Ground truth = the demo's grasp TCP (FK).
- Conditions:
  - `zero`: no examples;
  - `loo`: all other sessions;
  - `loo_rec`: only sessions from other recordings.
- The offline box is widened to x ≤ 320 mm because the demos grasped at x up to ~300.

**Metrics:**
- object-estimate error;
- error of the proposed grasp pose `FK(grasp_q_deg)`;
- error of the step target `FK(q_deg)`;
- `|expected_tcp − FK(q)|`;
- baseline "stay at the query pose".

Results with `claude/claude-opus-5-5` are below. There were 2 runs × 21 calls (234 k prompt
tokens, 59 k completion tokens). Run 1 had no `grasp_q_deg` field.

| median, mm | zero | loo | loo_rec |
|---|---|---|---|
| object estimate 3D, both runs (n = 14) | 31 | 35 | 37 |
| object estimate xy, both runs | 31 | 32 | 31 |
| object estimate, pre_grasp views / approach views | 29 / 44 | 37 / 31 | 31 / 52 |
| proposed grasp TCP 3D, run 2 (n = 7) | 30 | 35 | 33 |
| proposed grasp within 30 mm, run 2 | 5/7 | 2/7 | 3/7 |
| own kinematics: \|expected − FK(q)\| | 10–17 | 8–11 | 8–11 |
| baseline: stay at the query pose | 62 | 62 | 62 |

**Reading:**
- From a single wrist photo plus joints, the model localises the object to ~3 cm. That is
  about half the "stay" baseline at the 60 mm views, and the error is larger from the 150 mm
  views.
- Its own joint → TCP reasoning is within ~1 cm of FK.
- It never grips from a single view. It always takes a "move closer" step, which is sensible.
- **Few-shot examples gave no measurable benefit** (if anything a few mm worse) at this n. The
  run-to-run spread per query is 10–40 mm, larger than the differences between conditions.
- Likely causes:
  - only 1–3 examples per query;
  - the example camera mounts differ;
  - the model already reads the scale from the jaws and the object size.
- Few-shot may still help in the live loop: the examples show what a correct grasp looks like
  in the photo, which matters for the grip decision. Single-view offline queries do not test that.

## Live experiment protocol (needs arm time; a human at the power switch)

Prerequisites:
- the server runs on the rig and the tunnel is open (docs/SO101.md);
- `CONTROLR_SO101_AUTHKEY` is set;
- the workspace is cleared;
- one object lies on the table inside x 160–250, |y| ≤ 100 mm. The box stops the TCP at
  x = 270, and 3 of the 4 demo grasps were beyond that.

```bash
# 0. view check, no LLM: moves to the survey start pose, saves runs/fk_agent/survey/survey.jpg
uv run python scripts/so101_fk_agent.py survey --port 7811
# 1. episodes (alternate the conditions; same object placements for both)
uv run python scripts/so101_fk_agent.py live --port 7811 --dataset runs/fk_dataset --fewshot all \
    --task "pick up the plush carrot" --object-xyz 220,-40,25 --name carrot_p1 --max-steps 6
uv run python scripts/so101_fk_agent.py live --port 7811 --dataset runs/fk_dataset --fewshot none \
    --task "pick up the plush carrot" --object-xyz 220,-40,25 --name carrot_p1 --max-steps 6
```

**Limits in effect** (code, not prompt):
- TCP box x 140–270, y ±150, z 5–260 mm, at the target and along the whole path;
- elbow/wrist origins ≥ 20 mm above the table;
- the server's joint limits (calibrated range − 3°);
- ≤ 80 mm of TCP motion and ≤ 35° per joint between consecutive targets;
- approach from 40 mm above, then straight down;
- TCP speed 40 mm/s (the server also caps joint speed at 1 rad/s);
- server stop thresholds: current [400, 600, 600, 400, 300] mA, tracking 20°;
- jaws open at most 60 %; close command 0 %; lift 50 mm after a grasp;
- ≤ 6 steps;
- any STOP ends the episode with the arm frozen.

The initial move to the start pose is an unchecked `move_joints` from wherever the arm is (as
`So101Robot.reset` does): start from the rest pose.

**Design:**
- 2 conditions (`--fewshot all` / `none`) × 3 objects (rope ball, plush carrot, marker)
  × 3 placements = N = 9 episodes per condition.
- Alternate the order. Measure each object's centre with a ruler before the episode and pass
  it as `--object-xyz`.

**Metrics** (from `summary.json` / `steps.jsonl`):
- grasp success (`outcome == grasped`, confirmed from the lift photo);
- final TCP–object distance;
- steps and motions to success;
- the object-estimate error per step (does it shrink?);
- refused/clamped targets;
- LLM calls and tokens.

Report n and spread per condition (CLAUDE.md rule 8).

## Loop variants (`--loop`)

| loop | who computes poses | what the model returns | returns to start |
|---|---|---|---|
| `return` (first design) | model (joint angles) | object estimate + `q_deg` + action | after every action |
| `continuous` | model (joint angles) | same; re-plans from the current pose | only when done |
| `plan_once` | model (joint angles), once | grasp + place joints up front; code runs grasp → lift → carry → release → home with a check per stage, re-plan on a failed check (≤ 3) | only when done |
| `pixel` (owner's analytic design) | **code** | object / target pixel in the newest photo, `held`, `done`, `person`, ≤ 12-word note | only when done |

Common rules:
- Every episode starts and ends folded at the server's rest pose. Rest ↔ start pose is a
  path-checked move (`fold: true`).
- History: the last 25 states in order. Photos at full size only for the newest 3; older
  ones at 320 px.
- Few-shot examples: 2 photos per session at 320 px.
- Safety: any motor above 55 °C stops the episode.
- A move cut by the guard never closes or opens the jaws: the agent re-plans from where the
  arm stopped. In `plan_once`, cuts at the per-step limit are split into sub-moves instead.
- Targets keep 3 mm inside the box, so tracking error cannot leave the reached pose outside it.
- `wait`, or `person: true`, holds still and takes a new photo.

**`pixel` mode:**
- Code turns each pixel into a ray from the camera pose (FK of the measured joints + wrist
  camera model) and intersects it with the object's height (`object_z_mm`) or the target's
  (`target_z_mm`).
- It solves the steepest reachable tool pose by IK (`visual.tool_pose_q`).
- Sequence: look from 120 mm above → grasp at `grasp_z_mm` → lift 50 → place `release_dz_mm`
  above the target → open → lift.
- Completion is the model's `done`, judged from the photos. Code facts (gripper stall, 3D
  estimates) are shown only as text for orientation.

**Wrist camera correction** (`runs/fk_agent/camera_fit.json`, `visual.fit_rotation`):
- The stock URDF mount maps pixels to the table about 76 mm off (median).
- A 3-dof rotation correction was fitted on 8 live observations of the ball at positions known
  from later grasps. Rotation vector (−10.0°, −16.1°, −2.8°).
- Leave-one-episode-out error drops to 18 mm median, 49 mm at the image edge.
- With the model doing the pointing, the error is also 18 mm median. Each call takes ~2.7 s
  with ~60 output tokens.

## LLM latency

- Before: Opus at its default effort with 20+ images took 15–40 s per call, writing
  1.2–3.6k output tokens (mostly hidden thinking).
- JSON-first replies + closing the stream at the first complete JSON + `reasoning_effort: low`
  took the same query from 15.4 s to ~8–9 s. The router does accept `reasoning_effort` for
  Opus chat calls.
- Pixel mode (~60 output tokens) takes 2.4–4.7 s per call.
- Motion per move: 3–8 s at 50 mm/s, including the 40 mm approach from above. Fold/unfold
  ~3 s each; observe ~0.1 s.
