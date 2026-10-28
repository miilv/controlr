# 2026-10-28: organize_table and store_tools from mode-gt rows; support-mesh rests and a push-order lookahead (stream H)

Same method as put_bottles (journal 2026-10-28-dojo-h95-put-bottles):
- diagnose on stream M's mode-gt rows (main 08b0dd7, 100000-04, `compute3:~/dojo-data/m95/gt/<task>_p1/`);
- screen fixes in G3 mode gt (physics-only twin, GT world, 0 LLM) on v1 draws of 100030-46 (D-C10c,
  `gen_layouts.generate(task, seed, variant=1)`);
- fixes behind flags whose default is today's behaviour.

## 1. organize_table (mode gt 0/5, mean 0.5)

Per item, from the checker: mouse 5/5, alarm 3/5, keyboard 2/5, garage 0/5 (`tasks/organize_table.py`).
- **The alarm:** the "dz -11 / -14 mm" misses are not placed too low.
  - The drawer's top is a tray 23 mm below its bbox top (the rims), with no front rim. The first ~16 mm of the bbox
    towards the robot has no surface at all.
  - `on_targets` plans on the bbox. It rests the alarm at rim height, so the alarm is let go 40 mm above the floor
    instead of 17. Its edge points walk towards the arm to the bbox's edge, so the alarm's front ends 15-20 mm over
    the open strip.
  - It lands tilted 15-18 deg (`is_axis_up` allows 15), and 2 of 5 later fall to the table. The flat ones read
    dz -22 mm: they sit on the tray floor.
- **The keyboard:** `push_to` pushes the longer axis first (+y, ~170 mm), which works. Afterwards the sideways stroke
  has no plan, "no push stroke along axis 0" in 5/5.
  - A CPU replay of the strokes (kinematic world, the poses before the push) finds what blocks it: the garage still
    standing by the keyboard's left end (100000, 100003), or the alarm fallen by the drawer's front (100002).
  - The checker needs the keyboard's box over x 0..0.2: 2/5 pass, those that started at x ~0.05.
- **The garage** (a 19 cm figurine, 0.25-0.45 kg, onto a 50 mm cushion beside the drawer; only the left arm
  reaches) fails 0/5: 2 slips of 25-40 deg tilted slice grasps, and no pair or plan in the others. Section 5.
- Episodes end on "2 failed attempts" with 500+ of 1000 steps left.

Prior work: BACKLOG "organize_table, three skill-side gaps" and journal 2026-10-27-dojo-longh §5 tried the same two
ideas on `feat/dojo-longh-organize` without a gain (mode A 0.444 -> 0.444): "the flat tray targets have pairs but
their full plans collide at the pick". Since then 989565a made arrange's support a mesh-height-map obstacle
(`container_obstacle`), not its box + margin. That is a reason the null could flip, so the screen was rerun.

## 2. store_tools (mode gt 2/5, mean 0.55)

Every miss happens in the hold. The tools that stay where the jaws put them reach their slots within 4 mm / 10 deg.
- **The hammer:** all 3 failures used "box close 2 approach +1 tilt +15 off -11mm", and it turned 34-87 deg / 55-79 mm
  in the pads before the slot. The 2 successes used "tilt +0 off +0mm".
- Pliers: 2 pick slips (rose 14-18 of 80 mm). Wrench: 1 slip, plus 1 hold that turned 194 mm. Tape: 1 hold shifted
  63 mm (the 113 mm miss).
- After a first miss the tools lie on or against the toolbox, and the retries find no reachable grasp.

A per-asset PhysX centre-of-mass table was the proposed fix, but main already has it. skills8 (b5a73e1) measured
`dojo/skills/data/physics_com.json`: the hammer's CoM is at x +1.6 mm, 12.4 mm from the mesh centroid, because the
colliders are inflated to 20 mm minThickness. `_level_grasps` already ranks by that lever plus tilt / 10 deg. So the
-11 mm, 15 deg grasp ranked worse and was taken because nothing better planned. BACKLOG "thin irregular tools" records
the reason: straight-down grasps do not reach the slots. Parked there with this evidence; no screen.

## 3. The two switches (default = today)

`RuntimeConfig.place_surface` / `Skills.place_surface`: "bbox" (default) | "mesh".
- `library.on_targets` reads the support's top from its mesh (`support_z_fn`, a `precision.height_map`).
- A target pose is kept only where the surface under the object's whole bbox footprint (`footprint_xy`, 9 x 9 points)
  exists and is flat (within `PLACE_FLAT_TOL` 4 mm). The object rests on that surface, not on the bbox top.
- The edge points walk out from the centre only while such poses exist.
- No such pose at all (a stand smaller than the object, the garage's cushion): the bbox targets, logged.
- `arrange`: when the flat poses give no plan (ik_fail / collision, 0 steps), it retries with today's bbox targets.
  The flat poses lie further in than the bbox's edge ones, so where only the edge is in reach, today's behaviour is
  kept.
- `precision.height_map(max_seg=)`: the default 24 samples per triangle edge left empty cells inside the large
  triangles of a decimated flat top. The support map uses 400 at a 3 mm cell; the existing callers are unchanged.

`RuntimeConfig.push_order` / `Skills.push_order`: "greedy" (default) | "lookahead".
- `library.push_to`: with both axes left to go, the other axis goes first when its stroke plans now but would not
  from the pose after the first axis is done.
- The stroke planner is factored out unchanged (`_push_stroke`; it takes a hypothetical centre).

G3: `--skill-attr place_surface=mesh --skill-attr push_order=lookahead` (`g3.SKILL_ATTRS`, recorded per row).

## 4. Screen: G3 mode gt, v1 draws of 100030-46

Both arms run the same code (`compute3:~/controlr-h95d`). Base = flags off. Fix = `place_surface=mesh` +
`push_order=lookahead`, with the bbox fallback. 100040 and 100046 are unstable layouts in both arms, which leaves
n = 14 paired.

The fix arm went in two steps:
- The first fix arm had no fallback.
- The fallback was then rerun on the 4 seeds whose flat plan had failed (100031, 35, 42, 44). Rows elsewhere are
  unchanged, since the fallback only acts after such a failure.
- Of 2 controls, 100032 replayed bit-identically. 100043 diverged at the alarm's relay search (a wall-clock cap,
  `RELAY_PLAN_S`, while another job loaded the box; DEVELOPMENT's determinism caveat) with the same score. Its
  first-run row is kept.

| | base | fix |
|---|---|---|
| official score, mean | 0.321 | 0.375 |
| paired | | 3 up (100032, 33, 34), 0 down, 11 equal |
| success | 0/14 | 0/14 |
| mouse (final state) | 10 | 11 |
| alarm (final state) | 1 | 5 |
| keyboard (final state) | 2 | 3 |
| garage (final state) | 0 | 0 |

Final state is an approximation of the checker from the rows' final poses. It reproduces the fix rows' scores. Base
scores more than its final state on 4 seeds: the task scores in "transition" mode, which credits a stage reached at any
step, so an alarm that stood briefly before falling off counts.

- Alarm plans:
  - base: 7 placements, 6 of them land tilted 10-19 deg and fall off;
  - first fix arm: 5 placements, all 5 stay flat (dz -0.2..-0.3 mm, tilt <= 0.3 deg), and 4 have no plan;
  - with the fallback: 100031 places at the bbox edge again ("ok" at 8.4 deg, it falls later, credited as in base);
    100042 is back to base; 100035 and 100044 have no plan either way.
- Keyboard: the lookahead fired in at least 4/14 (the rows keep only the last 60 log lines), and the final state gains
  100032. Pushes still overshoot and turn the keyboard (up to 34 deg), and `push_to` still ends "ik_fail" at skill
  level in 13/14.
- Still open: alarm pick slips ("top axis 0 tilt 40", 4-5 per arm) and the garage (section 5).

By rule 8 the official gain is within noise (3 up, 0 down of 14). The flat alarm placements (5/5 vs 1/7) are the real
effect. Adoption is the rendered A/B's call, as a guard in the h95 family.

## 5. The garage (parked)

Prior work: journals 2026-10-25-dojo-skills2 / skills3 (no full pick plan next to the drawer; 2/8 planned, then
slid out at the lift) and longh §5 (no pair within 12 mm of the stand with the drawer as an obstacle; stand offsets
ran out of the IK budget).

Now there are pairs: 168-194 in the G3 rows. They fail in the full plan instead. A CPU replay of 100030
(kinematic world, alarm already on the drawer) shows why:
- The alarm's step calls `container_obstacle(drawer)`, which stays in `sk.coll.include`, so the drawer is an
  obstacle for the garage.
- Without it, the garage's transfer plans. With it, every place approach (the lift's end to 4 cm over the cushion)
  sweeps the held figurine through the drawer or the alarm. The overhead via (z 1.0-1.04) is out of reach at 0.57 m
  from the left arm.
- Targets shifted 10-30 mm away from the drawer (inside every category's xy tolerance, 20-45 mm) raise the pairs from
  182 to 478, still with no approach.

The lever is motion-level: a lateral via around the drawer, or a lower pre-place. BACKLOG.

## 6. Landmine: `timeout` around g3.sh

`timeout 1900 scripts/skills/g3.sh` did not end the fix job. The python kept running its episodes (40 min, into the
10th layout) until killed by PID. The job script now uses `timeout -k 60 1800`. DEVELOPMENT pitfalls.

## 7. Spend

0 LLM calls. compute3 light2: 5 G3 jobs, ~128 min, each followed by a >= 60 s yield:
- base 100030-39: 31 min;
- fix 100030-39: 40 min, ~9 of them an overrun (section 6);
- base 100040-46: 14 min;
- fix 100040-46: 18 min;
- fallback: 25 min.

Disk under `~/dojo-data/h95/`: < 10 MB. No other process was touched: only captured PIDs were stopped.
