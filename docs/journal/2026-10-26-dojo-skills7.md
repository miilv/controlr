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

TBD (filled at the end)

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

TBD
