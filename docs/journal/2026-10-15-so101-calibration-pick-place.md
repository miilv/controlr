# 2026-10-15 — SO-101: joint zeros, ports, pick-and-place

## What went wrong first
- **FK was wrong, not the model.** LeRobot's joint zero (middle of the calibrated range) is not the
  URDF zero on this arm: FK read 25-50 mm low, ~55 mm short in reach, and "straight down" was ~20 deg
  forward. A touch probe trusting it drove the gripper into the rig laptop and broke the wrist camera.
  Lesson: no Cartesian motion on an unverified arm; replay recorded joints instead (that picked the ball
  first time, `scripts/so101_replay.py`).
- **ttyUSB0/1 swap on every re-plug** (both CH340s share one by-id name; socket paths change too), and
  LeRobot's connect() offers to write the calibration file into the motors on a mismatch: on swapped
  ports that put one arm's calibration into the other (three times today, once by me). Fix:
  `controlr/robot/so101/arm_id.py` identifies the arms from raw encoder + motor offset vs each file's
  ranges; the recorder and the server default to `auto`; nothing writes calibration unless the arm is
  identified (`--restore-calibration`).
- The recorder lost demos twice: lerobot-record kept joints in memory (serial glitch at 90 s), then my
  own writer died on `cv2.imwrite("x.jpg.tmp")` (codec chosen by extension). Now: lerobot-teleoperate's
  loop, writer thread, imencode, soft start, refuse a start with the arms far apart.

## Joint zero fit
`scripts/so101_fit_offsets.py` on `touch_3` (closed-jaw table touches, reach/side/tilt varied): 13 of
16 dwells kept (one in the air, one 8 mm off), offsets [0, -5.07, -10.06, -1.36, 0] deg, rms 1.6 mm
(34 mm before). Every touch then sits at z = 0 ± 4 mm from x 100 to 340 mm; the old laptop crash lands
at x 299, z +13 (the laptop's edge). Jog with the offsets: ±20 mm moves within 1.5 mm.

## Pick-and-place (Opus 5.5, `configs/so101_pick_place.yaml`)
| run | turns | result |
|---|---|---|
| 20261015T003500Z | 18 | picked on the 2nd close, placed at (199, 99): first full success |
| 20261015T004254Z | 13 (stopped) | 3 misses: the wrist camera had turned on its mount (same joints, jaws out of frame) |
| 20261015T012722Z | 10 | clean: 1st-close grasp at z ≈ 23 on the magenta cross, placed at (199, 100) |
| 20261015T012925Z | 22 | 1st-close grasp, but placed against a plush carrot and dragged it; ball rolled out of reach |
What made it work: offsets; tool tilted 20 deg like the operator's demos; aim mark midway between the
jaw tips at their height (re-measured after the camera moved); fixed grasp height z 22 mm with
centring done there; "ball below the cross = move -x". Open: check the place spot is clear and lift
straight up after the release (rule added).

## Parallel tracks
- `controlr/fkagent` (few-shot FK agent, docs/FK_AGENT.md): offline, Opus locates the object within
  ~3 cm from one photo; few-shot examples gave no measurable gain (31 vs 35 mm median, spread 10-40 mm).
  Live comparison pending.
- robocurve inspect-robots: works with our router; a draft embodiment for our server
  (~/skoltech/inspect-robots-so101); joint-space only, so it needs our workspace guard before real use.
