# Review: control and safety (v0, 2026-10-02)

Lens: control/safety correctness end to end: LLM mm/deg -> SI -> envelope -> IK -> Isaac joint
drives -> state -> feedback/overlays, plus goal checks and determinism.

Method: I read the code (grammar, feedback, safety, kinematics, Isaac client/server/tasks, loop,
builder, mock, renderers). I reprocessed every `runs/2026*_sim_*` log and looked at the images. I
ran small local numpy scripts against `controlr.robot.kinematics`, `controlr.robot.isaac.tasks`
and PHANTOM's scene config. No live LLM calls, no Isaac. `uv run pytest -q`: 246 passed,
12 skipped.

What checked out (no finding):
- The units chain mm/deg -> m/rad is consistent in the grammar, feedback, safety messages and
  overlays.
- FK matches the sim: controlr `fk(q)` vs the logged Isaac `tcp_pos`/`tcp_rotvec` gives
  ≤ 0.0002 mm and < 0.0001° over every logged state.
- `T_cam_base = inv(world_from_cv)` is correct.
- The manual's axis directions are correct. Projecting the scene config: +x goes right, +y up,
  +z up. The base origin lands at u = 539/640. The camera is tilted 25.4° from vertical.
- The jaw axis (tool x) matches the rendered fingers.
- IK restarts use a fixed seed (deterministic).
- Task sampling is seeded.

Severity: **blocker** = invalidates results or is unsafe as designed. **major** = wrong
safety/control behaviour that will bite in normal use. **minor** = a real defect with limited
impact.

---

## 1. [blocker] The reach task is physically infeasible at seed 0 (and at about half of all seeds): the gripper body must pass through the box wall

`controlr/robot/isaac/tasks.py:177-222` (REACH_DEFAULTS, `sample_reach`);
`controlr/robot/isaac/README.md` (claims "reachable with the start orientation").

**Problem**
- With `rotation=none` the tool keeps the START_Q orientation: fingers along (-0.76, -0.58, -0.29).
- The gripper body and the wrist therefore lie up to 180 mm behind the TCP, toward +x/+y/+z.
- Marker y goes up to -0.12 m and z up to about 0.15 m. The box's near wall is at
  y ∈ [-0.085, -0.065] with its top at z = 0.18 m.
- For many markers, the line from the TCP to the flange crosses that wall.
- `sample_reach` checks only the distance to the packet and image occlusion. It does not check
  IK, and it does not check whether the tool body is clear of the box. The README's
  "reachable" claim is not implemented anywhere.

**Evidence**
- Script: sample 200 seeds and test the TCP→flange centreline against the near-wall slab.
  - Margin 0: 96/200 seeds intersect.
  - ±30 mm margin for the housing/fingers: 130/200.
- **Seed 0, the only seed used in every live run, intersects.** Its marker is (-496, -159, 138) mm.
- All five live reach episodes ended in STOPs against the blue box: 1297, 83, 208, 112 and
  131 N; 125503 hit three box STOPs in a row.
- The image after turn 6 of `runs/20261002T125321Z_sim_reach` shows the gripper over the box
  interior with the fingers hiding the ball.
- INTEGRATION_NOTES puts these failures down to depth perception. A perfect controller cannot
  succeed at this seed either.

**Fix**
- In `sample_reach`, reject a marker unless both hold:
  - (a) `UR3Kinematics.ik(marker, R_start)` succeeds within the soft limits;
  - (b) sampled points on the segment TCP → flange → wrist (inflated by about 45 mm) stay
    outside the bin's collision boxes. `scene_info["bin"]` already has the geometry.
- Raise an error if no sample passes. The current loop silently keeps the last rejected sample.
- Alternatively, restrict `marker_y` to ≤ -0.20.
- Add a CPU test asserting feasibility for seeds 0..49.

## 2. [major] Table clearance applies only to the TCP point; with the tilted tool the open fingers go about 10 mm into the mat

`controlr/robot/safety.py:76-78, 170-187`; `controlr/prompts/system_v0.md:102` ("at least 5 mm
above the table").

**Problem**
- `z_floor` is `table_z + table_clearance_m` and applies to the TCP only. The TCP is the midpoint
  between the pads.
- In the waffle start orientation the jaw line is (-0.27, +0.91, -0.33).
- Fully open (91 mm), each pad centre sits ±45.6 mm along that line, so the lower pad centre is
  15 mm **below** the TCP.
- The pads are also about 48 mm tall along the tool axis, and the tool axis points 35° down. That
  puts the lowest pad corner about 30 mm below the TCP.
- So the envelope accepts targets that put the lower finger about 10–25 mm into the mat. The manual
  promises 5 mm clearance, and only the 80 N force stop (finding 5) catches it.

**Evidence**
- Jaw/tool axes from FK of `configs/sim_waffle.yaml` `start_q`. The manual text in
  `runs/20261002T131109Z_sim_waffle/system_prompt.md` quotes the same vectors.
- In that run the model descended to TCP z = 58, then commanded dz = -30. The physics diverged
  with gripper–mat contact one turn after a 170 N finger–packet contact.

**Fix**
- Apply the clearance to the lowest point of a small set of gripper keypoints rather than to the
  TCP. Keypoints: TCP ± (w/2)·jaw ± (pad_h/2)·tool_z ± (pad_w/2)·tool_y, using the current or
  commanded opening `w`.
- The server already reports `pad_midpoint_in_tool` and `closing_axis_in_tool`; put the pad
  extents into `RobotSpec` as an optional field.
- Report the CLAMP as "lowest finger at z=…".

## 3. [major] The envelope validates only the end point; execution is joint-linear, and near-singular IK solutions swing the wrist up to 78° for a 50 mm move

`controlr/robot/safety.py:64,79,188-194` (BRANCH_GUARD π/2 is the only check);
`controlr/robot/isaac/server.py:472-479` (min-jerk interpolation in joint space);
`controlr/robot/isaac/client.py:184-190`.

**Problem**
- The safety envelope validates only the start and end configurations: workspace, table and
  joint limits, all at the target.
- The server then interpolates linearly **in joint space**, so the TCP path is not the straight
  line the model asked for. Its orientation also wanders.
- IK accepts any solution within π/2 per joint of the seed. Near the wrist singularity
  (wrist_2 ≈ 0) a small Cartesian move then becomes a large wrist flip.
- There is no manipulability check, and no limit on dq per mm of TCP motion.

**Evidence**
- Script: 100 mm moves on the IK branch from START_Q and from the waffle start, with the fixed
  orientation and the envelope's IKOptions.

  | metric | START_Q | waffle start |
  |---|---|---|
  | median path deviation | 2.4 mm | 1.8 mm |
  | p90 path deviation | 8.5 mm | 7.5 mm |
  | moves deviating > 10 mm | 7.7 % | 7.2 % |
  | worst path deviation | **115 mm** (max dq 1.37 rad) | 33 mm |

  One waffle move dipped 12 mm below the table clearance mid-path.
- Live, `runs/20261002T105745Z_sim_reach` turn 5:
  - `MOVE ee_delta 0 50 0` at wrist_2 = 0.30 rad needed dq = [-0.58, 0.19, 0.28, 0.71, -0.27,
    **-1.36**] rad.
  - Achieved dx = -7.4, dy = 61.5, dz = 2.9 mm. TCP ended 14 mm off target with 5.3° of
    orientation error. "not settled" was reported.
  - The next move hit the box at **1297 N**.

**Fix**
- Execute ee-mode moves as Cartesian sub-waypoints. IK every ≤ 5 mm, seeded by the previous
  waypoint, and send the waypoints as a dense q list (the server already accepts several
  segments).
- Run the workspace, table and finger-keypoint checks on every waypoint.
- Replace the π/2 branch guard with a ratio check, e.g. reject when
  max|dq| > 0.02 rad per mm of TCP motion + 0.05 rad.
- Add a manipulability floor (σ_min(J) threshold) that shortens the move and warns:
  "near wrist singularity".

## 4. [major] An IK failure drops the whole move instead of shortening it, and the message gives the model nothing to act on

`controlr/robot/safety.py:188-194`.

**Problem**
- When the (step-limited) target has no IK solution with the fixed orientation, the action is
  skipped.
- Joint modes already bisect to the largest safe fraction (`_joint`, lines 258-271); ee modes do
  not.
- With `rotation=none` and a tilted tool the reachable set is thin, so this happens often.

**Evidence**
- `runs/20261002T105745Z_sim_reach` turns 1–2 and `20261002T125321Z_sim_reach` turn 2: three
  turns wasted on `ik_fail`.
- Reproduced from the logged q of 105745 turn 1: target (-290, -199, 199) has no solution, even
  with 200 restarts and no branch guard.
- But 75 % of the same motion is feasible. For turn 2's target, 90 % is feasible.

**Fix**
- On IK failure, bisect the fraction of the requested displacement (as in `_joint`) and execute
  the largest feasible fraction ≥ 10 %.
- Report it as CLAMP: "target not reachable with this orientation; moved 75 % (… mm), the
  reachable edge is toward …".

## 5. [major] The contact-force stop overshoots by 2–16× and ignores object and self contacts

`controlr/robot/isaac/server.py:413-432, 455-470`; `controlr/robot/isaac/client.py:224-230`.

**Problem**
- (a) Contacts are sampled only every `contact_every = 10` physics steps (10 ms), with no
  slow-down near contact. At min-jerk peak speed (finding 7) the arm penetrates several mm
  between samples.
- (b) Only `*-table` and `*-box` contacts count. Gripper/arm against the packet is reported as
  INFO "contact" and never stops anything. Pressing the packet into the mat or the box is
  invisible to the stop. Robot self-collision is not observed at all.

**Evidence**
- Every STOP in the logs is far above the 80 N threshold: 83, 112, 113, 131, 134, 187, 208, 209
  and **1297 N** (105745 turn 7).
- `131109` turn 4: a "gripper/fingers touched the packet (peak 170 N)" EVENT, no stop. Three turns
  later the solver diverged (TCP flung to (40, 144, -261) mm, contact forces up to 5e12 N).
- README: physics cost does not depend on contact reporting, so sampling every step is cheap.

**Fix**
- Sample every step during motion.
- Add an object-contact stop threshold (e.g. `safety.object_force_stop_n`, about 40 N; the packet
  weighs 35 g).
- Report robot–object contact above that threshold as WARN, not INFO.
- Optionally use PhysX contact-report thresholds.

## 6. [major] A STOP raised during settling keeps pushing into the obstacle, and the integral term alone can exceed the stop threshold

`controlr/robot/isaac/server.py:236-246, 489-511`.

**Problem**
- `q_cmd` is replaced by the measured pose (and `q_bias` zeroed) only when the stop happened
  during the arm segments, at lines 489-492.
- `step()` keeps checking the force inside the settle loop. If the force rises there, which is the
  typical case when the target lies inside an obstacle, `stopped` becomes True while the drives
  keep the penetrating target plus `q_bias`.
- The settle loop then exits after up to `max_s` still pressing. Yet the feedback says "the arm
  holds its current pose".
- The next turn's baseline `f0` is then high, so the following stop needs `f0 + 20 N`.
- The windup is also not small. `BIAS_LIMIT_RAD = 0.01` is applied to every joint. At PHANTOM's
  drive stiffness (3500 N·m/rad, consistent with the documented ~3 mrad elbow gravity sag),
  that is up to about 35 N·m per joint, roughly 90 N at the TCP through shoulder_lift.
- So the docstring's "can never push meaningfully into an obstacle" is false. Against a wall the
  integrator saturates within a few 10 ms iterations.

**Fix**
- Factor the stop handling into a function and call it whenever `stopped` flips, including in
  settle: freeze `q_cmd` to the measured pose and zero `q_bias`.
- Freeze integration while any robot–environment contact force is above a few N (anti-windup).
- Bound the bias by torque (stiffness·bias ≤ a few N·m), not by angle.

## 7. [major] Speed limits are average speeds; the min-jerk profile peaks at 1.875×, so the configured TCP and joint speed limits are exceeded

`controlr/robot/isaac/client.py:184-190`; `controlr/robot/isaac/server.py:474-479`;
`controlr/config.py:114`.

**Problem**
- The segment duration is `max(d/v_tcp, max|dq|/joint_speed, …)`, i.e. average speed.
- The profile is min-jerk, s = 10s³ - 15s⁴ + 6s⁵, whose peak speed is 1.875× the average
  (computed).
- So a 100 mm move peaks at 0.28 m/s, against `max_tcp_speed_m_s = 0.15`. Joints peak at
  1.875 rad/s against the 1.0 rad/s PHANTOM hardware limit the spec cites.
- This feeds the force-stop overshoot (finding 5). On a real UR it would violate the configured
  safety speeds.

**Fix**
- Multiply the duration by 1.875 (15/8), or use a trapezoidal profile with the limit as the peak.
- Unit-test that the peak speeds of the generated q(t) stay ≤ the limits.

## 8. [major] "Fixed orientation" (rotation=none) is re-anchored to the measured pose every turn, so contact-induced tilt is locked in and the manual's tool and jaw description goes stale

`controlr/robot/safety.py:133-143` (R0 = FK of the measured q); `controlr/prompts/builder.py:413`.

**Problem**
- With `rotation=none` the IK target orientation is the **current measured** orientation, not
  the reset orientation.
- After a collision, a STOP or a tracking error, the tilt persists for the rest of the episode.
- The system prompt, which is cached and generated from `state0`, still tells the model the old
  finger direction and jaw line. STATE omits orientation in this mode, so the model cannot notice.

**Evidence**
- Orientation error vs reset, from the logs:
  - `125503`: 15.3° after its first STOP, kept to the end;
  - `105745`: 5.3° from the singular move in finding 3;
  - `131109`: 0.5° after contacts.

**Fix**
- Store `R_ref` at reset: `SafetyEnvelope.reset(state0)`, or pass `state0` when it is built in
  `run_episode`.
- Use `R_ref` as the orientation target for rotation=none in both modes. The arm then
  re-straightens on the next move.
- Report a WARN when the measured tilt differs from `R_ref` by more than 3°.

## 9. [minor] Joint modes bypass the TCP step limit, and their workspace check assumes a monotone path

`controlr/robot/safety.py:216-271`.

**Problem**
- `max_step_m` ("max TCP translation per MOVE line", and the manual says the same in every mode)
  is never applied in joint modes. Only `max_step_rad` (30° per joint) is.
- From START_Q, 30° on shoulder_pan moves the TCP 233 mm, shoulder_lift 231 mm, elbow 199 mm.
- The bisection for "largest safe fraction" checks only end points and assumes that once a fraction
  is unsafe, every larger one is too. A path can leave the envelope and come back.

**Fix**
- Also limit the TCP displacement (FK) to `max_step_m`.
- Check FK along the interpolated path (see finding 3).

## 10. [minor] The elbow joint limit in the spec (±360°) is wider than the sim/URDF (±180°)

`controlr/robot/spec.py:264-265` vs PHANTOM `assets/sim/ur3/ur3_cb3.urdf:237`
(elbow ±π).

**Problem**
- The envelope and the manual allow elbow targets the Isaac drive cannot reach. Physically, the
  UR3 elbow self-collides past about ±170°.
- In `joint_abs`/`joint_delta` (or after a few IK steps) the arm stalls at the URDF limit.
  That shows up only as a "tracking" WARN.

**Fix**: use elbow ±π (minus the margin) in `ur3_cb3_spec`.

## 11. [minor] ee_abs examples and rotation=yaw are wrong for this rig

`controlr/protocol/grammar.py:135-137, 350-353`.

**Problem**
- The `ee_abs` grammar examples are `MOVE ee_abs 300 -50 150` and `… 300 -50 80`. Both are
  verified outside the workspace (x must be -550..-150).
- Integration fix 7 corrected only the builder's worked example; these remain.
- `ee_abs` + `rotation=yaw` always commands a top-down tool (roll 180°). `tasks.py:37-40` says a
  top-down tool cannot reach the packet on this rig. Every such line is rotation-step-limited
  toward an infeasible orientation, or fails IK.

**Fix**
- Build the examples from the workspace centre (as the builder does).
- Make ee_abs+yaw keep the current roll/pitch and set only yaw. Alternatively, reject that config
  for the Isaac backend.

## 12. [minor] The manual's workspace z lower bound is the table, not the enforced floor

`controlr/prompts/builder.py:423` (prints `spec.workspace_lo`) vs `safety.py:76-78`.

**Problem**: the manual says "z -10..480 mm", but the envelope floor is -4.5 mm (table + 5 mm),
and more once finding 2 is fixed.

**Fix**: print `max(ws_lo.z, table_z + clearance)`.

## 13. [minor] `gripper_closed` and `holding` semantics differ from the contract

`controlr/robot/isaac/server.py:332`; `controlr/types.py:58`.

**Problem**
- The types say `gripper_closed` is "last commanded state is closed". The server reports
  `closure_cmd >= 0.5`.
- A GRIP close that latches on an object wider than about 45 mm (latch closure < 0.38 + 0.12)
  therefore shows "open".
- A partial width command reads "open" or "closed" by threshold.
- `holding` is true whenever both pads touch the packet with > 0.1 N, even when the gripper was
  never closed (e.g. the packet is wedged).

**Fix**
- Track the last commanded intent (open/close/width) explicitly.
- Require `closure_cmd > 0` for `holding`.

## 14. [minor] The force-stop baseline is a global maximum across contact pairs

`controlr/robot/isaac/server.py:452-453`.

**Problem**
- `stop_floor = max(thr, f0 + 20)` uses the worst force of *any* table/box contact before the
  motion. An arm resting on the box with f0 = 300 N could then drive the gripper into the table
  up to 320 N without a stop.

**Fix**
- Keep the baseline per (robot group, env group) pair and cap it, e.g. ≤ 2·thr.
- With finding 6 fixed, f0 should be small anyway.

## 15. [minor] An unstable solver does not stop when the force stop is disabled

`controlr/robot/isaac/server.py:467-470`.

**Problem**: `force_stop_n = 0` (or None) disables `stopped`. The `> UNSTABLE_FORCE_N` and NaN
branches then set only `stop_reason`, and the motion continues on a diverged solver.

**Fix**: set `stopped = True` in the unstable branch regardless of `force_stop_n`.

## 16. [minor] The executed list on a stop omits the partially executed segment

`controlr/robot/isaac/client.py:212`.

**Problem**: on a stop in segment k > 0 (n_done = k), `executed[:k]` drops segment k, which did
partly move. For k = 0 it keeps segment 0.

**Fix**: use `executed[:n_done + 1]`, plus a flag "segment n_done interrupted".

## 17. [minor] An unsettled arm is frozen mid-motion between turns, and the settle timeout is short

`controlr/robot/isaac/server.py:440, 497-511`.

**Problem**
- `settle.max_s = 1.0` s. After it, physics pauses with non-zero joint velocities.
- The image and STATE show a transient. The next `execute` starts from that moving state, and a
  HOLD turn shows the same frozen pose (105745 turns 5–6).

**Fix**
- Settle while |qd| is large (e.g. max_s = 3 s for long segments, or keep stepping until the
  speed is below threshold).
- Zero the velocities only if the pose error is tiny.

## 18. [minor] Success bookkeeping and success definitions

**`loop.py:414`**: `result.success = last_goal.success` regardless of outcome. A run can be
`outcome=safety_stop` (or `error`/`max_turns`) with `success: true`, which double-counts in
sweeps. Fix: report both explicitly (`goal_success_at_end`, `success_claimed_and_verified`) and
use the latter in aggregates.

**`tasks.py:257-270` (push)**: success = centre within 25 mm of the zone, not held, not lifted *at
the check instant*. Picking the packet up, carrying it and releasing it on the square counts as
"pushed". Fix: record max lift / held during the episode on the server (a cheap running max in
`execute`) and fail if it exceeded 20 mm.

**`tasks.py:214-221` (reach sampler)**: after 500 rejected tries the last (rejected) sample is
used silently. Fix: raise an error instead.

## 19. [minor] The mock envelope ignores the configured safety, and a held cube can sink into the table

`controlr/robot/mock.py:105, 177-180`.

**Problem**
- The fallback envelope uses `SafetyConfig()` defaults, not `cfg.safety`.
- `sc.cube = TCP + hold_offset` with no floor. With a hold offset up to 15 mm the cube's bottom
  can sit below the table while the TCP respects the clearance.
- Mock only, but mock episodes are the regression baseline.

**Fix**
- Pass `cfg.safety` (as `IsaacRobot.from_config` does).
- Clamp the held cube to z ≥ table + size/2, or reject the move.

## 20. [minor] Stale docs

`controlr/robot/isaac/README.md:124` still says `spec.py` defaults `table_z` to 0.053 m. It is now
-0.0095 (`spec.py:235`).

---

### Not verified / out of scope
- Bitwise determinism of PhysX across episodes in one server process: contact warm-start caches
  are not reset. Needs an Isaac A/B run with the same seed twice.
- Whether `world.step` uses GPU PhysX: GPU PhysX is not deterministic across runs.
