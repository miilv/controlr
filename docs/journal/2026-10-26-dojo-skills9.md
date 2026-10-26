# 2026-10-26: dojo skills9: the refined pick keeps its paired grasp; the held-pose protocol at the pre-insert hover

Stream: skills9. Branch `feat/dojo-skills9` from main 331a551, rebased on 5b01218 (#59); part 1 as its own PR #64
(`feat/dojo-skills9-replan`), part 2 on top. Code: `dojo/skills/skills.py` (shared: `execute_pick`, `plan_pick`, `plan_transfer`,
`run_pick`, `run_transfer`; part 2: `run_place`, the servo / guarded tail), `dojo/skills/heldsim.py` (new, G3
only), `dojo/skills/g3.py` (flags), `pegs.py`, `library.py` (`insert_into` place kwargs), `data/held_noise.json` +
`scripts/skills/held_noise_table.py`. Tests: `tests/dojo/test_skills9.py`. Runs: commit-stamped copies `~/dojo-skills9{,b,...}`
(`scripts/skills/deploy_skills9.sh [dir]`) and main 331a551 in `~/dojo-skills9m`, light lanes through `g3_light.sh` with
`DOJO_STREAM=skills9`, at most two at once (a queue runner, `~/dojo-data/skills9/scratch/qrun.sh`). Data:
`compute3:~/dojo-data/skills9/g3/<run>_<commit>/`. The run commits (4db6a14, ...) predate the rebase; they stay
reachable on the branches `skills9-runs`, `skills9-runs2` and `skills9-runs3`. G3 = the TRUE state in the physics-only twin; mode B = `g3 run --mode-b` (no
in-hand re-measure, a refine hook that sees nothing) with `--belief-offset` (an object's believed pose is off until it
is held). The entry point is always `planner.solve`. No task names, no LLM calls.

## 1. The refined pick keeps the grasp its place was planned with

**The landmine** (skills-8, BACKLOG). With a refine hook (mode B), `Skills.execute_pick` hovers, refines the object
from the wrist, then re-planned the descent with `plan_pick(label, arm, "auto")`: the method's default grasps. A pick
that `plan_transfer` had paired with its place (a re-orienting transfer, a relay leg, a level carry, a rim grasp of a
coin) then ran some other grasp, and the place re-planned for it often had no IK; the nominal place (planned for the
other grasp) then put the object in at the wrong orientation.

**The fix** (`skills.py`):
- `PickPlan` keeps what it was planned with: `method`, `plan_kw` (lift, rates, squeeze, ...), and from
  `plan_transfer` a `repair(q_start)` closure (the same call, same arm, from the world as it is then).
- `execute_pick` with a refine: unmoved by the refine (0.2 mm / 0.2 deg) -> the original descent; else the SAME grasp
  in the object frame (`inv(T_tcp_obj)`) at the refined pose, planned with the pick's own arguments
  (`_same_grasp_pick`); only if that is infeasible, a paired pick re-pairs pick + place together (`repair`; the
  executed plan carries `paired_place`, which `run_transfer` then uses), and an unpaired pick falls back to the method's
  grasps as before. `_pick_rehover` (a re-hover over a hinted pose) tries the same grasp first too.
- `run_pick` records the executed plan (`Skills.last_pick`); `run_transfer` reviews / places with it.
- Without a refine hook (G3) nothing changes: `execute_pick` runs the plan as it did.

**G3 mode B** (`--mode-b`, 2.5 mm belief offsets, 100000-04; before = main 331a551, after = 4db6a14; score = the
checker's ladder):

| task | offsets | before | after | G3 (no offset), same seeds | what changed |
|---|---|---|---|---|---|
| store_tools_in_toolbox | hammer +2.5 mm x, wrench +2.5 mm y | 0.10 (0 / 0.5 / 0 / 0 / 0; main rerun, per seed as skills-8 0a59dac) | **0.40** (0.5 / 0.75 / 0.25 / 0.5 / 0) | 0.45 (0.5 / 1.0 / 0 / 0.75 / 0) | the tilted pinches the level carry paired run as planned |
| deposit_coin | coin +2.5 mm x | **0.00** (0/5: the rim grasp re-planned as a top grasp, 21 mm loose, put down twice) | **0.68** (3/5) | 0.48 (skills-5 final, same seeds) | the edge grasp runs; the handover and the slot follow |
| insert_tubes | 2.5 mm on each tube | 0.76 (3/5) | 0.60 (2/5) | 0.78 (25 layouts) | before: the default top grasp happened to fit the place; after: the paired box grasp (as in G3). Both lose the third tube to IK (no plan) in 100000 / 100002; the rest is one extra tube left tilted on the rack. Noise at n = 5 |

Regressions with no offset: G3 never enters this path (no refine hook); the set runs on part 2's code (below).

## 2. The held-pose protocol at the pre-insert hover (mode B)

Perception-held (#59, report `docs/experiments/2026-10-26-held-pose.md`) measures the in-hand transform from the
cameras: `held_pose(arm, ...)` with a calibrated sigma, and `show_pose(...)` proposes observer looks / show poses. The
skills side (`Skills._held_protocol`, called by `run_place` at the pre-insert hover when there is no in-hand re-measure
and the provider has `held_pose`):

1. **Read at the hover** (0 steps; `cfg_` = the place target + `view`). Good enough: ok, the leading point's lateral
   sigma <= `held_tol` and the axis sigma <= half the place's `tol_tilt`.
2. **Look** when it is not (failed, ambiguous, too uncertain), the place is a servo insertion, the provider has
   `show_pose` and the budget leaves the tail its steps + 25: the provider's candidates, planned with our own IK and
   collision model (`_plan_held_look`: the holding arm carrying the object to a show pose; the other arm's wrist to the
   observer look, its hand 1.5 cm off the held object's box; 4 cm between the arms on the way out and back), at most one
   observer look and one show pose; at each, 2 still steps, a read, back. A candidate without its observer is skipped
   (a show seen by the own wrist + head only is no better than the hover: plug 7 / 24 in the report).
   - A place that always looks (`held_tol` 0: `insert_into`) runs the observer's look during its approach instead
     (`_pre_look`, `show_pose` with `cfg_["q_hold"]` = the hover's joints): only the way back costs steps.
3. **Use** the last read within the gate (`held_gate`, default half the place's `tol_xy`; axis half its `tol_tilt`):
   the servo tail (as in G3) on that transform (`_servo_obj`: the measured TCP @ the adopted transform, never the
   belief's held pose) and the arm's measured sag; with `spiral` (insert_into 0.5 mm), a descent that stops on the
   receptacle is first dragged, pressed 1.5 mm, along a spiral (0.15 mm pitch) before the usual probes.
4. **No usable read**: the guarded / probing tail, aimed `blind_up` above the target. `stack_on_pegs` (seated in mode B
   only on a read with sigma <= 0.5 mm, `held_gate`) passes the threaded depth there: a ring is never pushed down to
   its seat blind (the BACKLOG gap: 0.30 when it was).

Every read is logged (`Skills.held_events`; G3 episode records `held`). G3 (mode A) never calls any of it.

**The G3 test provider** (`dojo/skills/heldsim.py`, `g3 run --mode-b --held-pose noise|gt`, `--held-belief`):
- the runtime belief's in-hand transform: the object is attached at the gripper's close with its BELIEVED pose (the
  belief offset included) and then follows the measured TCP; G3 mode B before had the true transform after the lift
  (plus the offset) in `verify_pick`;
- `held_pose` answers the TRUE transform moved by an error resampled from perception-held's dev evaluation, per
  category and view (`data/held_noise.json` from `eval_all_v5` / `eval_pens_v5` / `eval_coin_v6`, views "all": hover
  = the place hover + its grasp variants, look = the ranked observer look, show = the ranked show pose + look; sigma
  scaled and ok re-decided as the merged estimator does, `SIGMA_SCALE` rigid 1.21 / 1.31, spin 1.62 / 1.61, ok at
  1 mm / 2 deg), the error applied at the place's leading point; an ok read re-anchors the attachment;
- `show_pose`: generic candidates in `propose`'s format (looks from beside / across the pads at 14-24 cm, show poses in
  front of the head camera), our IK only, a fixed preference order (perception ranks by rendered silhouettes).

| category | hover: ok / n, lateral median / p90 (mm) | look | show |
|---|---|---|---|
| charger | 14 / 40, 0.49 / 1.18 | 37 / 40, 0.19 / 0.39 | 21 / 24, 0.14 / 0.28 |
| stackingblocks (rings) | 28 / 40, 0.13 / 0.28 | 34 / 40, 0.12 / 0.23 | 40 / 40, 0.06 / 0.11 |
| test_tube | 0 / 40, 0.30 / 0.95 | 1 / 40, 0.24 / 1.18 | 7 / 40, 0.17 / 0.40 |
| coin | 25 / 40, 1.02 / 2.87 | 11 / 22, 0.21 / 0.56 | 10 / 12, 0.16 / 0.25 |

(Lateral = the leading point's error in TCP y, z over all rows of the table.) The tubes' calibrated sigma is
conservative (report: about 3x the error), so a tube read is almost never ok: with this estimator the protocol leaves
the tubes on the fallback.

### G3 numbers (mode B, `--held-pose noise`)

Same seeds and offsets per row; "baseline" = today's mode B (no read: the guarded / probing tail on the in-hand
transform predicted with the offset). Score = the checker's ladder. Commits: 1e3acff / da8e9e6 on `skills9-runs2`
(before the rebase; da8e9e6's skills code equals this branch's 46608a1), 181cb81 on this branch.

| task | offsets | layouts | baseline | protocol | G3 (no offset), same seeds |
|---|---|---|---|---|---|
| plug_in_charger | charger +2.5 mm x | 100000-09 | 2/10, **0.20** (1e3acff) | **8/10, 0.80** (da8e9e6) | 8/10, 0.80 (skills-8; the same two fail) |
| play_stacking_toy | pieces 1.5-2 mm | 100000-04 | 4/5, 0.92 (threaded, da8e9e6) | 3/5, 0.78 (seated on the read, threaded first, da8e9e6); **4/5, 0.92** (threaded, the guarded tail on the read, 181cb81) | 0.98 (v4.6, 25 layouts) |
| play_stacking_toy | pieces + base 2 mm (skills-7's set) | 100000-04 | 2/5, 0.76 (threaded, 1e3acff) | 0/5, 0.36 (seated on the read, 1e3acff); 0/5, 0.36 (threaded first, da8e9e6); **3/5, 0.84** (threaded, the guarded tail on the read, 181cb81) | |

**plug.** The hover read was ambiguous or not ok in every episode (charger hover ok 14 / 40 in the table); no observer
look was feasible at the plug's hover (the other arm's palm or links at the table: the hover is 2 cm over the strip,
logged per candidate), so each episode took a show pose with its observer look: 26-30 steps (one 92: a second try).
The read was adopted in 8 of 9 places that reached the hover, and the servo seated the charger after its descent
(+0.9 to +1.5 mm, within `seat_tol`); the pressed spiral was never needed. The two failures are 100002 (no relay plan:
fails in every mode) and 100008 (a relay leg's collision, also in G3). Steps median 212 (limit 400).

**stacking.** The rings' hover reads are good (ok 28 / 40, 0.13 mm) and were adopted for most pieces, but seating on
them did not beat threading:
- With the base's belief 2 mm off, the read fixes the hand, not the peg. The pieces' beliefs (offset the other way)
  then read 2.8-4 mm off their peg in `pegs.on_peg` (3 mm), so pieces the checker counts were marked "not on", left out
  of `on_this`, and the next seat target was planned too low (0 grasp-place pairs). Threaded targets do not depend on
  `on_this` (BACKLOG: G3 mode B keeps a released object's offset).
- Pieces only: 0.78 vs 0.92. In mode B the servo's held pose follows the TCP, so a ring caught on a peg's top that
  slides in the pads is not seen; the guarded descent's arm-lag stop sees it.
- So `stack_on_pegs` in mode B stays threaded (`PEG_SEAT_MODE_B` off) and runs the guarded tail ON the adopted read
  (`guarded`: the read aims the threading, the lag stop senses the contact): 0.92 and 0.84 against the threaded
  baselines' 0.92 and 0.76. No gain beyond noise at n = 5: threading from a 2 mm prediction already finds the peg with
  its probes; the read saves a few probes. The remaining misses are rings tilted on a peg top (one per episode) and
  the G3 offset artifact above.

### Regressions (G3, no offset; da8e9e6 = part 2's G3 path: 181cb81 changes only mode-B branches)

| task | layouts | now | reference |
|---|---|---|---|
| plug_in_charger | 100000-09 | 8/10, 0.80 | 8/10 (skills-8 0a59dac; the same two fail) |
| make_kong | 100000-04 | 5/5, 1.00 | 5/5 |
| store_tools_in_toolbox | 100000-04 | 1/5, 0.45 (0.5 / 1.0 / 0 / 0.75 / 0) | 0.45, per seed the same (skills-8 final) |

## Spend and open

- No LLM calls. Light lanes only, at most two at once (a queue runner that counts our chunk runners, not lock headers:
  other tools take the light locks without rewriting them). About 6 h of lane time: 17 G3 runs of 2-10 layouts.
- Open:
  - runtime: `provider.show_pose` is not wired yet (BACKLOG); until it is, the protocol reads at the hover only.
  - tubes: with the merged estimator's calibrated sigma a tube read is almost never ok (sigma about 3x the error), so
    the protocol leaves them on the fallback; deposit_coin: the own-camera read under-covers and the protocol does not
    yet refuse a disc's hover read (BACKLOG).
  - stacking: the in-hand read does not help in G3 mode B; the remaining misses are rings tilted on a peg top and the
    G3 offset artifact (`on_peg` on offset beliefs). A real test needs the runtime's post-place refine.
  - plug: observer looks are infeasible at its low hover (the other arm at the table); the show pose carries it.

## 3. Non-grasp outcomes: topple and push confirmed through `provider.confirm_pose`

Integration-2's breadth triage (#67): make_kong 0/5 in mode B (G3 25/25). A topple or a push moves an object without a
grasp: no release, so no post-place re-measure, and no twin segment. The runtime belief keeps the tile standing, every
check reads "axis at 90 deg from up", and the skill retries and fails whatever the tile did.

- Skills side (`Skills.confirm_moved`): after the action, the predicted outcome goes to `provider.confirm_pose(label,
  pos, quat)` (the runtime: the head view of the current frame decides, the post-place second-look rule), then a
  refresh. `topple` predicts `Skills.toppled_pose` (a 90-deg turn about the far bottom edge, the far face on the
  support); `library.push_to` and `Skills.push` predict each stroke's step along the push. Without such a provider (G3)
  nothing changes. Every call is logged (`Skills.confirm_events`; G3 records `confirm`).
- G3 side (`heldsim`, `g3 run --freeze-touched` / `--confirm-pose`): an object the robot's fingertips came within 2 cm
  of and that then moved > 3 mm / 5 deg without a grasp keeps its believed pose (the runtime's blind spot); a grasp's
  release unfreezes it. `confirm_pose` adopts the expected pose when it is within 15 mm / 20 deg of the truth (+ the
  offset) and closer to it than the frozen belief. G3 mode B before re-read GT at every refresh, which hid this.

| make_kong 100000-04, mode B, 1 mm offsets on the 12 hand tiles | success | mean | steps |
|---|---|---|---|
| today's G3 mode B (the world re-read from GT) | 5/5 | 1.00 | 370-379 |
| `--freeze-touched` (the runtime's blind spot) | 0/5 | 0.00 | 143-146 (topple retries, "did not tip over") |
| `--confirm-pose` | **5/5** | **1.00** | 370-379; 3 topples per episode, all adopted (expected pose 0.8-1.9 mm / 0.1-2.3 deg off the truth; the frozen belief 50 mm / 90 deg) |

Regression (G3, no offset; run commit febf68e on `skills9-runs3`, the same code before the rebase): make_kong
100000-04 5/5 (no provider `confirm_pose`: nothing changes in G3).

Open: other non-grasp motions that change an object's pose (a lever press, a sweep) do not call it yet.
