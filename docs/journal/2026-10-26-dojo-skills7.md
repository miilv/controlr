# 2026-10-26: dojo skills7: play_stacking_toy and hang_mugs

Stream: skills7. Branch `feat/dojo-skills7` from origin/main b6c70c4, rebased on cae6867 (skills-5's #40 and
integration-2's #51). Code: `dojo/skills/pegs.py` (new: `stack_on_pegs`), `hang_on_hooks` in `dojo/skills/library.py`,
small opt-in additions to the shared layer (section 5). Tests: `tests/dojo/test_skills7.py`.

Runs: `scripts/skills/g3_light.sh` from commit-stamped copies `~/dojo-skills7{,b,c,d,e,...}`
(`scripts/skills/deploy_skills7.sh [dir]`), at most two light slots, `SIMLAB_USD_SYNC_LINKS` on (the runner's default).
Data: `compute3:~/dojo-data/skills7/g3/<run>_<commit>/`. G3 means the TRUE state in the physics-only twin. Mode B means
`g3 run --mode-b` (no in-hand re-measure, a refine hook that sees nothing) with `--belief-offset` (an object's believed
pose is off until it is held). The entry point is always `planner.solve` (instruction words + objects). No task names,
no LLM calls.

## 1. Board context

- play_stacking_toy (Long-Horizon, weight 2.5 %): PyRUA 78.2, best on the board. We had 0.40-0.44 (skills-3/4).
- hang_mugs (Generalization std, 0.83 %): best VPP2 74.2, PyRUA 27. We had 0.36 (v2) / 0.26 (v4).
- hang_mugs_random (0.83 %): PyRUA 28.4. We had 0.20.

## 2. play_stacking_toy

### What the checker wants

From the task code:
- 10 pieces in 4 identical groups (4/3/2/1) go onto 4 pegs of heights 63/53/43/28 mm.
- Each group's first piece must be within 1 mm (xy) of its peg's support point.
- Consecutive pieces of a group must be within 1 mm of each other.
- Every piece's bottom must sit at least 5 mm below the base's bbox top.
- The tiers are 10/30/60/100 for any 1/2/3/4 complete pegs, with the grippers open.

### What failed before

1. **Tallest peg first.** Each later peg's pieces were let go level with the finished stack beside them, 7 mm away.
   The collision model's 5 mm held-box inflation plus a 6 mm object margin left 0 grasp-place pairs, for 3-7 of
   10 pieces per layout (skills-3/4's "far pegs").
2. **Yaw turns about the bbox centre.** `plan_transfer` turned the `yaw_sym=4` targets about the bbox centre. The star
   piece's hole is 2.9 mm off that centre, so 3 of 4 targets missed the peg by 4-6 mm. Every "in-hand drift 3 mm,
   rel tilt 20-31 deg" failure in skills-3/4 was a star piece.
3. **In-air handover fallback.** A ring handed over in the air dropped (drift 228-241 mm, 2/2).

### Built (v4 → v4.6)

| piece | what |
|---|---|
| hole axis | `pegs.through_hole`: the enclosed empty cells of a vertical depth map of the piece's mesh in its "up" pose. Radius 5.2-5.55 mm; the hole is at the origin for all four shapes (star, square, rectangle, disc). |
| peg geometry | `pegs.peg_geometry`: the axis from the support frame. The rounded top is 2 mm above the frame (mesh). The plate is at 10.3 mm. The shaft radius is 4.83 mm. |
| order | Shortest peg first: every piece goes in above (v4.1) or beside (v4.4+) shorter finished stacks. |
| targets | The hole on the peg's axis, 12 turns about the hole (`plan_transfer(yaw_about=)`), centred grasps only, at most 30 deg from vertical. A disc pinched 16 mm off centre slid out; a 40 deg pinch of a 13 mm piece did not lift. |
| place | Through the servo tail. v4.1 let go 4 mm down the peg; the piece slid down. v4.4+ seats it 0.5 mm over the piece below, then servos its measured axis to within 0.3 mm (`servo_xy_tol`) before letting go, against one side of the peg (`side_bias`: the hole's clearance less 0.15 mm). |
| neighbours | The base and every seated piece are `Collision.parts` (small boxes from the mesh). Seated pieces are checked against the held piece's own mesh points (`HeldBox.dense`), not its inflated box. |
| reach | 2 cm retreat: the far peg is at the edge of reach, and 3-5 cm failed the IK. Relay spots come 12 cm in front of the base, clear of it (`place_oriented(spot_avoid=)`). No in-air handovers. |
| mode B | Guarded tail (skills-5) with `guard_lag` 1.2 mm and `guard_seated` 1.5 mm. A piece left tilted on a peg is lifted off and tried once more (`stuck`). |

### Checker vs physics: the 1 mm pair check

The holes are 5.2-5.55 mm in radius and the pegs 4.83 mm, so a piece moves up to 0.65 mm off-axis.

- Let go at the peg top (v4.1/v4.2), the pieces of one peg ended 0.1-1.14 mm apart.
- 4 of 25 episodes stopped at 60 with all 10 pieces on: one pair 1.0-1.14 mm apart.
- v4.4+ seats each piece and servos it, all against the same side of the peg.

## 3. hang_mugs

### What the checker wants

From the task code, per mug:
- lifted;
- `handle` above z 0.9;
- handle z within 60 deg of the hook's z, handle x within 60 deg of the hook's ±x;
- handle within 45 mm of any hook point;
- stationary.

The tiers are 15/40/100 for 1/2/3 mugs. `is_axis_aligned` takes the FIRST frame of each side:
- Only the hooks oriented like hook 0 can score: the ±x arms at two heights. The ±y middle arms cannot.
- Mug 14 (rand set) has handle frame 0 = identity. Hung, its axis is horizontal, and the check wants it within 60 deg
  of up, so it can never score.
- 17 of 25 rand dev layouts hold a mug 14. The ceiling on our rand set is about 55.

### Built (v5 → v6.1)

| piece | what |
|---|---|
| rack model | The rack as `Collision.parts`: about 190 boxes from its mesh. Its bbox covers every hook, and before this it was not an obstacle at all, so the carry to a low hook ran unchecked under the upper arms. The target hook's own arm is left out of the boxes while planning that hook. |
| held mug | Checked against parts with its mesh points (`HeldBox.dense`). |
| hung mugs | Fine boxes (1.5 cm cells, 2 mm margin), registered right after the release. |
| hook choice | `hook_order`: the n highest hooks, filled lowest first. That is one low hook instead of two; the top hooks had taken 8/8 in v2, the low ones 4/10. |
| handle frame | `threads_along_x`: only frames whose x runs through the handle's hole. This skips mug 14's frame 0, which would stand the mug upright on the hook. |
| relay | `relay_spot`: a mug no hook plans for from where it stands is set down, as it is, on a spot 20 cm in front of the rack, if some hook plans from there. |
| lift | A hook plan is retried with the lift to the pre-hang height (rand: the low carry met a clutter lamp, and the via path over the top hook was out of reach). |
| exit | `_hang_exit`: after each release, back off 6 cm more and go home on a collision-checked path. The finish's unchecked straight joint move swept the open hand through a mug just hung on a low hook and flung it 40 cm (v5 100003). |
| speed | The parts check is one batch over the points inside the boxes' bounds: 30 → 3 ms per check. |

## 4. G3 numbers

GT state, physics-only twin, settled dev layouts, `SIMLAB_USD_SYNC_LINKS` on. Score = the checker's ladder tier. Steps are
median / max (limit). Runs before the rebase carry pre-rebase hashes (1f1928d = 844cdb3, b9a07b7 = 45c2822).

**play_stacking_toy** (1200 steps; board PyRUA 78.2):

| run (commit) | layouts | success | mean | steps | failures |
|---|---|---|---|---|---|
| skills-3 v2 / skills-4 v3 (before) | 100000-04 | 0/5 | 0.44 / 0.40 | 516-801 | far pegs: 0 grasp-place pairs; stars tilted on the peg |
| v4 (1f1928d: shortest peg first, hole axis, 12 turns) | 100000-02, 05-08 | 3/7 | 0.77 | 813 / 913 | far-peg retreat IK 2, pieces 1.1 mm apart 2 |
| v4.1/v4.2 (b9a07b7 / 82a1d28: one side of the peg, 2 cm retreat, relay spots, centred grasps) | all 25 | 21/25 | 0.93 | 833 / 930 | pieces of one peg 1.0-1.14 mm apart 3, a disc pinched off centre 1 |
| **v4.6 (37a223a: seated + servoed to 0.3 mm, base and seated pieces as mesh boxes)** | **all 25** | **24/25** | **0.98** | **938 / 1101** | 100002: a relay's last leg found no pre-place IK (0.6) |
| mode B, 1.5-2 mm belief offsets (b044443, let go 4 mm down the peg) | 100000-02 | 0/3 | 0.43 | 570 / 1036 | rings stuck tilted on the peg's top (no contact seen at 4 mm lag) |
| mode B, 1.5-2 mm (37a223a, seated) | 100000-04 | 0/5 | 0.30 | 896 / 1195 | a ring pushed 30-50 mm down the peg on a predicted pose binds, tilts, can't be re-grasped and blocks the next pegs; one ran out of steps |
| mode B, 1.5-2 mm (f24365a: seated only when measured, else threaded) | 100000-02 | TBD | | | |

v4.6 = the final code for mode A: v4.7/v4.8 change only the branch without a held-object measurement.
`stack_on_pegs` and the shared code it calls are unchanged from 37a223a to the PR head.

**hang_mugs** (800 steps; board VPP2 74.2, PyRUA 27):

| run (commit) | layouts | success | mean | steps | failures |
|---|---|---|---|---|---|
| skills-4 v2 (before) | 100000-09 | 2/10 | 0.36 | 324 / 392 | low hooks 4/10 (rack not modelled) |
| v5 (c2295b8: rack and hung mugs as mesh boxes, highest hooks lowest first, relay) | 100000-04 | 2/5 | 0.64 | 345 / 399 | a mug flung off a low hook by the unchecked move home, a mug out of the pads on the carry, no plan for the biggest mug |
| v6 (37a223a: + lifted retry) | 100005-07 | 1/3 | 0.52 | 317 / 560 | slide to the far hook short (mug fell), relays failing |
| **v6.2 (722a9db: + checked exit, slide reach, search caps)** | **all 25** | **5/25** | **0.44** | 365 / 641 | 1.0 ×5, 0.4 ×12, 0.15 ×8. place_miss 15, not_home 3 (2 mugs), collision 2 (relays); planning median 192 s, max 414 s |

v6.2 is below v5's first chunk: v5 scored 0.64 on 100000-04, v6.2 0.76 on the same seeds, but only 0.44 over 25. The
first five layouts were the easy ones.

v6.2 per mug, 100000-18: 34 hung of 57.
- Failures:
  - 7 fell out of the wall pinch on the carry (in-hand shift > 15 mm);
  - 7 not threaded (fell at the release: mostly the top hook on the far side, done by the other arm);
  - 4 loose, put down;
  - 3 relay places failed at the descent IK;
  - no plan 2.
- Per hook (hung / tried): low +x 16/19, top +x 7/11, top -x 8/20, low -x 3/9.

**hang_mugs_random** (rand ceiling about 55 on our dev set, mug 14): TBD.

**Regression on 722a9db** (shared code touched):
- plug_in_charger 100000-09: 8/10, 0.80. The reference is 8/10; the same two layouts fail (100002 no relay, 100008 the relay's last leg).
- make_kong 100000-04: 5/5, 379 steps.

## 5. Shared-code changes (all opt-in; defaults unchanged)

- `Skills.plan_transfer(yaw_about=None)`: the point the `yaw_sym` turns are about.
- `Skills.plan_place(guard_seated=, guard_lag=, servo_xy_tol=)`: the guarded tail's seated tolerance and contact lag,
  and the servo tail's xy correction once seated. All default to None, which means today's behaviour.
- `Collision.parts` (label → small boxes): obstacles from a mesh (`motion.mesh_parts`). `HeldBox.dense`: the held
  object's mesh points, used only against parts.
- `free_spots(avoid=)` / `place_oriented(spot_avoid=)`: fixtures the relay spots must clear.
- `KinematicEnv` (dry runs only): rows with `pegs` (a piece let go on a peg slides down it) and `hang_points` (an
  object let go around a hook point stays). `scripts/skills/dryrun.py` fills both from the assets.
- Registry entry `stack_on_pegs` now `dojo.skills.pegs:stack_on_pegs`. SKILL_DOCS v6.

## 6. Parked / next

- **stacking, mode B.**
  - With 1.5-2 mm belief offsets and no held-object measurement, 0/5 and 0.30.
  - The guarded tail now sees the contact (`guard_lag` 1.2 mm) and its probes find the hole. But a ring pushed down the
    peg on a predicted in-hand pose binds and tilts.
  - A tilted ring is not re-graspable with the centred top grasps, and its box blocks the neighbouring pegs.
  - v4.7/v4.8 thread blind and seat only with a measurement (`held_measured`: `Skills.remeasure` or perception-held's
    `held_pose`).
  - Next:
    - wire `held_pose` into the servo tail;
    - a per-arm sigma gate at the hover (`held_measured(sk, arm)`);
    - every piece left on the base as mesh boxes, not just the seated ones;
    - grasps for a tilted ring.
- **stacking, 100002.** A relay's last leg plans with the piece at the relay pose but finds no pre-place IK at run time.
  Homing both arms and re-planning did not help (dry run); not found.
- **hang, the third mug.**
  - It falls out of a wall pinch on the carry (7 of 23 failures).
  - Or it is not threaded and drops at the release (7): mostly the far-side top hook done by the arm across the rack.
    Prefer the hook-side arm, and check the slide with the measured grasp.
  - Relays fail at the descent IK (3).
- **hang, planning time.** 2-7 min per episode. The search is capped at 150 s per round plus 90 s for the relay.
- **hang_mugs_random.** The small rack's low hooks have no plan for big mugs (palm at the table, clutter). Mug 14 can never
  score (BACKLOG).
