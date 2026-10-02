# Isaac Sim backend (`robot.backend: isaac`)

Drives PHANTOM's calibrated Isaac Sim 6.0 reconstruction of the real rig
(UR3 CB3 + Robotiq 2F-85 with DM-Tac W2L pads + D435, measured table / mat /
blue box / wafer packet) turn by turn. PHANTOM is imported, never modified.

```
controlr process (venv)                      Isaac python (scripts/isaac_server.sh, headless)
IsaacRobot (client.py)  --multiprocessing.connection, localhost, HMAC authkey-->  server.py: IsaacRig
  Action.q_target -> joint waypoints  ----execute---->  min-jerk joint targets through PHANTOM's drives,
                                                       gripper after the arm, settle, contacts, render
  Observation / ExecReport / GoalReport <---numpy dicts---  (protocol.py documents every op)
tasks.py (numpy only) is imported by both sides: placement randomisation + success rules.
```

* **Scene**: `phantom.sim.scene.import_robot` + `build_scene` with
  `configs/sim/waffles_w2l_adaptive_parallel_dt1ms_20260909.json` (native
  adaptive W2L gripper, 1 ms physics — the config PHANTOM's scripted-expert
  data campaign uses; factory D435 RGB K, measured extrinsics). The stage is
  built once; `reset` teleports the arm, moves the packet and toggles
  visual-only markers, so tasks switch without a rebuild.
* **Frames**: the stage world frame is the UR controller base frame.
  `CameraInfo.K` = PHANTOM's factory intrinsics (fx 609.28, cx 337.8, cy 249.7),
  `T_cam_base = inv(world_from_cv)`. TCP = sim `tool0` + 180 mm along tool z,
  identical to `controlr.robot.kinematics` (tested: |FK(q) − sim TCP| < 1 mm for
  random q). The gel-pad midpoint actually sits at 178.4 mm (1.6 mm short of the TCP).
* **Turn-based physics**: nothing steps between requests. `execute` takes the
  envelope's `q_target`s (unfiltered actions go through a default
  `SafetyEnvelope`), interpolates each segment over `max(dist/tcp_speed,
  dq/1 rad/s, 0.1 s)`, actuates the gripper after the segment, then settles
  (joint speed < 0.03 rad/s, TCP < 5 mm/s, joint error < 0.5 mrad; ≤ 1 s).
  A bounded integral term (≤ 0.01 rad) removes the ~2 mm gravity sag of
  PHANTOM's pure-PD drives — the real UR controller tracks to ~0.1 mm.
  Arm/gripper vs table/box contact above `safety.contact_force_stop_n` stops
  the motion (STOP event); > 5 kN or NaN contact data = solver blew up (reset).
* **Gripper**: opening = gap between the gel faces, calibrated at startup on
  the sim linkage (0.9 … 91.3 mm; the W2L pads sit 5.5 mm outboard of the
  Robotiq fingertips). "close" ramps the command and latches 0.12 closure past
  first bilateral contact (Robotiq-like object detection). Squeezing to full
  closure through the 35 mm packet makes PhysX diverge; PHANTOM's own expert
  commands 0.62 and the latch lands at ~0.58. `holding` = both pads > 0.1 N on
  the packet.
* **Events to the model**: STOP (force stop / instability), WARN collision
  (arm or gripper vs table/mat/box/packet, peak > 2 N), INFO finger–packet
  contact, WARN tracking (> 5 mm from the command), WARN not settled,
  WARN gripper linkage abnormal (PHANTOM `mechanical_diagnostics`).

## Base frame as seen by the D435 (verified)

`docs/img/isaac_axes_{0,x+,y+,z+}.png`: 80 mm TCP steps from the same pose
(achieved 80.1 / 80.0 / 79.9 mm, cross-axis < 0.3 mm).
+x → right in the image; +y → away from the camera (up the image, toward the
blue box); +z → up toward the camera (gripper grows and shifts slightly up).
The robot base is at the right edge, level with the box's near wall. This is
`IsaacRobot.spec.base_frame_doc`.

## Tasks (`task.name`)

| name | scene change | success (privileged, at check time) |
|---|---|---|
| `waffle_pick_place` (alias `pick_place`) | packet centre N(0,10 mm) clipped ±20 mm, yaw N(0,5°) clipped ±10° (PHANTOM `expert_campaign.py`); `params.nominal: true` = measured pose | PHANTOM `policy_metrics` full_task at one instant: all 8 packet corners inside the box interior (2 mm tol), pads < 0.1 N, robot–packet < 0.1 N, packet < 0.03 m/s and < 0.5 rad/s |
| `reach` | red ball on a pole, x∈[−0.50,−0.26], y∈[−0.33,−0.12], 5–16 cm above the table, ≥ 8 cm from the packet, never behind the start-pose gripper, reachable with the start orientation | TCP within 15 mm of the ball centre |
| `push` | green 5 cm square 6–10 cm from the packet along its long axis (−x side) | packet centre within 25 mm of the square centre, not held or lifted |

All tasks start from a real recorded start state (`tasks.START_Q`, PHANTOM
`initial_states/waffles_aug22_1787395928_000.json`), gripper open. The tool is
tilted like every real demo: a top-down UR3 tool cannot reach the packet on
this rig. Params in `task.params` override any key of `tasks.*_DEFAULTS`.
`tasks.scripted_pick_place_plan` is a privileged IK reference solution
(PHANTOM expert rotation means); it succeeds in the sim
(`docs/img/isaac_pick_{lift,lower,done}.png`).

## Running

```bash
# compute3, once per checkout (venv for controlr; Isaac uses its own python)
cd ~/controlr && uv sync --extra dev
# either let IsaacRobot launch the server (robot.params.launch: true, default) or run it yourself:
CONTROLR_ISAAC_AUTHKEY=<secret> scripts/isaac_server.sh --port 7801      # prints CONTROLR_ISAAC_READY
```

`robot.params` (all optional): `host`, `port` (7801), `launch`, `server_args`
(e.g. `["--render-updates", "4"]`), `startup_timeout_s`, `camera` ("scene"),
`tcp_speed_m_s` (default `safety.max_tcp_speed_m_s`), `joint_speed_rad_s`,
`force_stop_n`, `contact_report_n`, `tracking_warn_m`, `settle` (server
overrides: `max_s`, `joint_err_rad`, `grip_squeeze`, `grip_max_s`, …),
`home_q`. Env: `ISAAC_SIM_ROOT`, `PHANTOM_ROOT`, `CONTROLR_ISAAC_AUTHKEY` (a
random one is generated when the client launches the server itself).
The imported robot USD (~0.5 GB) is written to `runs/.isaac_cache/`.

Tests: `tests/test_isaac_tasks.py`, `tests/test_isaac_client.py` (CPU, fake
server) and `tests/test_isaac_sim.py` (`CONTROLR_ISAAC=1`, launches its own
server on port 7821; ~2 min).

## Measured on compute3 (RTX 5090, shared CPU)

| what | time |
|---|---|
| server startup (Kit 8 s + URDF import 1.5 s + stage/calibration 2 s) | 11.5–16 s |
| `reset` (teleport + 1 s settle in physics) | ~4.0 s |
| `observe` (4 RTX updates + readback, 640×480) | 0.04 s |
| `state` | 1 ms |
| 10 mm `ee_delta` incl. settle | 0.44 s sim, ~1.9 s wall (error 0.05 mm) |
| 80 mm `ee_delta` | 0.8 s sim, ~3.5 s wall |
| gripper close / open | ~0.5 s sim, ~2 s wall |
| scripted pick-place (9 phases) | ~11 s sim, ~50 s wall |

Physics runs at ~0.24× real time: 4.2 ms per 1 ms step, independent of solver
iterations (16–64), contact reporting, CCD or self-collision (all measured).
The RTX annotator lags one app update; 2 updates give the current frame,
further updates only refine the denoiser (4 used).

## Open issues

* Wall time per turn is dominated by physics (1 ms steps required by the W2L
  gripper, `phantom.sim.gripper_adaptive.validate_physics_timestep`). The legacy
  sliding-pad scenes run at 4 ms but need PHANTOM's separate gripper visual
  (not wired up here).
* Carton / egg scenes (`configs/sim/carton_*.json`, `egg_*.json`) would need a
  second server (different stage); `tasks.py` only registers the waffle rig.
* Contact forces are PhysX normal impulses / dt sampled every 10 ms: transient
  peaks (e.g. 137 N at gripper closure) are reported as-is.
* `controlr/robot/spec.py` defaults `table_z` to 0.053 m (PHANTOM's historical
  `waffles.json`); the current measured scene has the mat at −0.0095 m, which
  this backend passes explicitly.
