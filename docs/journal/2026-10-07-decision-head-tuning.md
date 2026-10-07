# 2026-10-07 — tuning the decision head until its answers move the gripper well

Goal (owner): find a setup in which GPT-6 Luna Decisions' answers produce good motion; turns are
fast, so steps may be small. Spend: ~2,500 OpenRouter calls, ~$1.9, until the key's credit ran out
(HTTP 402 mid-run; the last 6 seeds of the final run never started).

## Method
- `scripts/decisions_probe.py` — ONE call per scene (random target, start pose around it), the step
  scored against the true direction (cos, per-axis sign). `scripts/decisions_perception.py` — image
  relations (left/right, top/bottom) between the gripper and a VISIBLE object, per camera, against
  the calibrated projections. Then closed-loop mock episodes (`controlr run`).
- **Landmine (found mid-way): unbalanced probes.** Small random moves from home left the target on
  the same side in almost every sample (Isaac: packet left of the gripper 24/24), so a model that
  answers a constant scored "24/24". Starts are now placed around the aim point with a random sign
  per axis; read every perception number with its minority-class accuracy. Numbers below are the
  balanced ones unless marked.

## Findings (in order)
1. **v0 (one signed `score` per axis, -30..+30 mm):** cos +0.15, signs 35 %, mean step 4.6 mm — the
   model leans + on every axis. Axis hints in the question (v1) made it worse: z signs 0/23, mass on
   BOTH extremes — a signed ordinal scale is read as "how far", not "which way". → `head: split`:
   per axis a direction `choice` (neg/zero/pos) + an unsigned distance `score`.
2. **Base-axis wording is too abstract;** image relations work. The single-image red-square test
   was 1.00; "is the target toward +x" was ~0.5. → v3: the direction is asked as an image relation
   ("to the left of the fingertips in the `top` image") in the image where that axis reads
   unambiguously (`axis_views`).
3. **One angled camera cannot separate y from z** (both point "up", 40 vs 18 px per 100 mm). →
   virtual `top` and `side` cameras (mock + Isaac, `robot/isaac/cameras.py`).
4. **Unlabelled multi-image state:** the model cannot tell which image is `side`. → each image part
   is preceded by "image `<name>`:" (`build_state(labels=)`).
5. **Photoreal Isaac frames:** left/right of the gripper vs the packet was a constant "left" in the
   full scene and top frames (0/9 and 0/8 on the truly-right samples), also with one camera per
   request. The `ee_marker` overlay fixed top/bottom (scene 27/30) but not left/right. → **fovea**:
   a zoomed crop of the native frame around the TCP with a magenta cross (`observation.renderers:
   fovea`). Isaac, fovea, balanced: side left/right 28/29 (truly right 16/16), top top/bottom 27/27,
   scene top/bottom 30/30; scene/top left/right still lean "left" (5/9, 3/8 truly right).
6. **"higher/lower" is read physically in a top view** (the gripper is always above the object):
   0/23 there. Vertical image relations are now named by the image edges.
7. **The target must be named concretely.** Mock episodes with "the point the fingertips must reach
   next for the task" went 3/6 → 4/6 (fovea 320 px); the far seeds stalled with y "level" (the
   target was outside the crop, and the wide view could not find "the point"). Coarse-to-fine
   (`wide_<axis>`: the crop's "level" mass defers to the full frame) alone did not fix them; naming
   the target ("the red ball") did: **11/12 seeds within 15 mm, final distance median 9 mm, median
   10 turns to 15 mm**, 0.73 s per call, 0.93 s per turn, ~8.6k input tokens per turn
   (`decisions_v4` + target named; released as `decisions_v5` + `decisions.target`). The v5 rerun
   reached 4/4 within 15 mm (10, 9, 14, 12 mm) before the credit ran out.
8. **Isaac reach targets are invisible points** ("50 mm above the middle of the blue box's near
   rim"): with v4 + fovea the step probe stayed at cos +0.25, z signs 1/29. The decision head
   servos to things it can see; it does not imagine offsets.
9. **Refusals:** `502 OpenAI refused to answer question "dir_x"` killed two episodes (the same
   payload is refused again). The client now drops a refused question (that axis holds) and asks
   again, without backoff retries; the turn record lists `decisions.refused`.
10. **DONE is rarely claimed** (2/12 in the 11/12 run): the status question needs work, or the
    episode should end on a sensor-checkable condition.

## State of the setup
`configs/mock_dec_fovea.yaml`: `head: split`, `decisions_v5`, cameras scene+top+side, `fovea`
(320 px), `decisions.target: the red ball`. On Isaac the cameras exist (`robot.params.extra_cameras:
[top, side]`); what is missing is a visible, concretely named target per phase — for the pick-and-
place that is a job for the planner (named objects per phase), see BACKLOG.
