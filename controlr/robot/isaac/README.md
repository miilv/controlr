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
  built once; `reset` teleports the arm, moves the packet and puts the box back, so
  tasks switch without a rebuild. There are no visual markers: reach / push targets are
  text (see Tasks).
* **The blue box is a dynamic body** (applied after PHANTOM builds the stage; PHANTOM has
  it as five static colliders): `/World/Bin` gets `RigidBodyAPI` + `MassAPI` and its five
  collider cubes become one compound body, 0.4 kg (`task.params.box_mass_kg`), PHANTOM's
  friction (its `BinContact` material stays bound). `task.params.box_dynamic: false`
  makes it kinematic (immovable, the behaviour before). `reset` restores the authored pose;
  resting partly on the 3 mm mat the box tips ~0.7°. The server's `state()` carries the
  box's CURRENT geometry (`tasks.bin_info_at`): the goal check (exact containment in the
  moved / tilted box, `tasks.to_box_interior`), `IsaacRobot.obstacles()` (the envelope's
  wrist / housing check) and `tasks.body_box_clearance` use it.
* **Frames**: the stage world frame is the UR controller base frame.
  `CameraInfo.K` = PHANTOM's factory intrinsics (fx 609.28, cx 337.8, cy 249.7),
  `T_cam_base = inv(world_from_cv)`. TCP = sim `tool0` + 180 mm along tool z,
  identical to `controlr.robot.kinematics` (tested: |FK(q) − sim TCP| < 1 mm for
  random q). The gel-pad midpoint actually sits at 178.4 mm (1.6 mm short of the TCP).
* **Turn-based physics**: nothing steps between requests. `execute` takes, per
  action, the envelope's IK waypoints along the straight TCP line (`q_path`, every
  5 mm) and `q_target` (unfiltered actions go through the configured
  `SafetyEnvelope`) as one *group*: one min-jerk motion through the waypoints
  (`motion.py`), timed so the PEAK TCP speed ≤ `safety.max_tcp_speed_m_s` and the
  peak joint speed ≤ 1 rad/s (min-jerk peaks at 15/8 of the average). The gripper
  acts after the group, then the arm settles (joint speed < 0.03 rad/s, TCP < 5
  mm/s, joint error < 0.5 mrad; ≤ 2 s).
  A bounded integral term (≤ 0.004 rad, ~14 N·m) removes the ~2 mm gravity sag of
  PHANTOM's pure-PD drives — the real UR controller tracks to ~0.1 mm; it is frozen
  after a stop and while the robot presses on anything (> 5 N, anti-windup).
* **Force stop** (`contacts.py`, CPU-tested): contacts are sampled every control tick
  during motion and every ~10 ms while settling / gripping. Robot bodies: ONE PhysX
  contact-force matrix (all arm / gripper bodies × table, mat, box parts, packet;
  `reader: matrix`) or PHANTOM's per-body diagnostic views (`reader: phantom`, the legacy
  path, ~7× slower to read). The packet has its own view against table / mat / box parts.
  Rules: arm/gripper vs table above `safety.contact_force_stop_n`; vs the box above
  `safety.box_force_stop_n`; vs the packet above `safety.object_force_stop_n` only while
  the packet is NOT held (and not while the gripper itself closes); while it IS held (close
  command + both pads loaded) the packet-vs-environment force above
  `safety.held_object_force_stop_n` stops ("the held packet pushed against the box wall") —
  the grip's own pad forces never stop the arm (they reached 85–89 N in free air). Every
  pair also needs its force before the motion + 20 N (capped at 2× the threshold), so an
  arm resting against something can back away. `predict_stop` also stops when the force
  extrapolated one sample ahead exceeds the limit (sparse sampling). A stop at any point
  freezes the drive target at the measured pose and clears the integral term. > 5 kN or
  NaN contact data = the solver blew up: STOP of kind `unstable` even with the force stop
  disabled (reset). The server returns the stop structured (`kind`, `pair`, `force`,
  `threshold`); the client words it (no fingertip numbers without `observation.tactile`).
* **Gripper**: opening = gap between the gel faces, calibrated at startup on
  the sim linkage (0.9 … 91.3 mm; the W2L pads sit 5.5 mm outboard of the
  Robotiq fingertips). "close" ramps the command and latches 0.12 closure past
  first bilateral contact (Robotiq-like object detection). Squeezing to full
  closure through the 35 mm packet makes PhysX diverge; PHANTOM's own expert
  commands 0.62 and the latch lands at ~0.58. `holding` = a commanded closure and
  both pads > 0.1 N on the packet (tactile; shown to the model only with
  `feedback.level=full` + `observation.tactile`); `holding_grip` = the Robotiq-style object
  detection (closed fingers held > 0.03 closure short of the command and > 3 mm above full
  closure), logged only; `gripper_closed` = the last command was "close" (also when the
  latch stopped on a wide object).
* **Events** (what reaches the model depends on `feedback.level`): STOP (force stop /
  instability; `brief` = "the gripper pushed against the blue box", "the held packet pushed
  against the box wall", ...), WARN collision (arm or gripper vs table/mat/box/packet, peak
  > 2 N; full level only), `tactile` finger–packet contact (full + tactile only), WARN tracking
  (> 5 mm from the command), WARN not settled (full only), WARN gripper linkage abnormal
  (PHANTOM `mechanical_diagnostics`).
* **Run log**: every `execute` returns `profile` (seconds in drive targets / PhysX steps /
  contact reads / bookkeeping / settle checks, ticks, samples), `wall_s`, `sim_s`,
  contact peaks incl. pad forces, the box shift and the settings; the client puts them in
  `ExecReport.backend` → `turns.jsonl` `backend` (never shown to the model).

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
| `waffle_pick_place` (alias `pick_place`) | packet centre N(0,10 mm) clipped ±20 mm, yaw N(0,5°) clipped ±10° (PHANTOM `expert_campaign.py`); `params.nominal: true` = measured pose; `yaw_dist: uniform` = U(±`yaw_max_deg`); `yaw_offset_deg` = fixed packet yaw offset; `start_yaw_deg` = tool yaw offset at reset about the vertical through the start TCP (v → U(−v, v), [lo, hi] → U(lo, hi); drawn after the packet, so seeds keep their packet poses) | PHANTOM `policy_metrics` full_task at one instant: all 8 packet corners inside the box interior (2 mm tol), pads < 0.1 N, robot–packet < 0.1 N, packet < 0.03 m/s and < 0.5 rad/s |
| `reach` | a TEXT target relative to a visible object, drawn per seed from `tasks.REACH_TARGETS` (e.g. "60 mm above the centre of the top face of the wafer packet", "50 mm above the near-right corner of the blue box's rim"); only targets that are feasible from the start (IK with the start orientation, open fingertips above the table, wrist / housing ≥ 45 mm from the box walls — `tasks.reach_feasibility`); `targets` / `target_index` restrict the choice. No marker in the scene | TCP within 15 mm of the target computed from the object's CURRENT pose |
| `push` | TEXT: "push the wafer packet about N mm along its long side, toward the left of the image (−x)", N ∈ 60–100 mm (rounded to 10 mm in the text). No marker | packet centre within 25 mm of the target point, not held or lifted |

All tasks start from a real recorded start state (`tasks.START_Q`, PHANTOM
`initial_states/waffles_aug22_1787395928_000.json`), gripper open. The tool is
tilted like every real demo: a top-down UR3 tool cannot reach the packet on
this rig. Params in `task.params` override any key of `tasks.*_DEFAULTS`.
`tasks.scripted_pick_place_plan` is a privileged IK reference solution
(PHANTOM expert rotation means); it succeeds in the sim
(`docs/img/isaac_pick_{lift,lower,done}.png`).

### Rotation (`action.rotation: yaw`, `configs/sim_waffle_yaw.yaml`)

With the demo tool tilt fixed and only the heading commanded, the packet is graspable for yaw
offsets of about −40…+30° from its nominal pose (+35° only when the packet lies ≥ 20 mm toward
−y). Beyond +30° the gripper housing sits over the box's near wall (< 50 mm away; 47 mm already
touched it in the sim), and shifting the grasp away from the box runs out of reach; below −40° the
pregrasp is out of reach (elbow straight). At the high start pose a negative turn straightens the
elbow after ~27°, so `start_yaw_deg` up to 20 is safe and large negative turns must be made lower.
`tasks.run_scripted_yaw_pick_place` is a closed-loop scripted expert in the model's own action space
(ee_delta + dyaw + GRIP through the yaw envelope); `tasks.body_box_clearance` measures wrist/housing
clearance to the box. `reset` returns `episode["scene"]` (`tasks.scene_record`), logged to
`setup.json`.

## Running

```bash
# compute3, once per checkout (venv for controlr; Isaac uses its own python)
cd ~/controlr && uv sync --extra dev
# either let IsaacRobot launch the server (robot.params.launch: true, default) or run it yourself:
CONTROLR_ISAAC_AUTHKEY=<secret> scripts/isaac_server.sh --port 7801      # prints CONTROLR_ISAAC_READY
```

`robot.params` (all optional): `host`, `port` (7801), `launch`, `server_args`
(e.g. `["--render-updates", "4"]`), `physics` (server start args: `dt`,
`solver_position_iterations`, `solver_velocity_iterations`, `usd_writeback`; a running
server with other values is refused; `scripts/remote_run.sh` starts the server with the
config's values), `reader` (`matrix` | `phantom`), `substeps` (physics steps per drive-target
update / contact tick), `contact_every`, `direct` (PhysX stepped and drive targets set without
the isaacsim.core wrappers), `predict_stop`, `startup_timeout_s`, `camera` ("scene"),
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

* Speed: see the measurements in docs/experiments/2026-10-02-contacts-and-speed.md and the
  `robot.params` speed keys above. The legacy sliding-pad scenes (`waffles.json`, 4 ms) are a
  different, older reconstruction (table at +53 mm, estimated pad geometry) and need PHANTOM's
  separate gripper visual — not comparable with the calibrated W2L rig, not wired up.
* Carton / egg scenes (`configs/sim/carton_*.json`, `egg_*.json`) would need a
  second server (different stage); `tasks.py` only registers the waffle rig.
* Contact forces are PhysX normal impulses / dt (every step during motion):
  transient peaks (e.g. 137 N at gripper closure) are reported as-is.
* Robot self-collision is simulated but not observed by the force stop.
* One camera (`scene`); the mock backend has a synthetic `top` camera for the
  multi-camera axis. A second calibrated Isaac camera is not wired up.
