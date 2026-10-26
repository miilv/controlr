# 2026-10-26: dojo skills6: a scripted opponent, a demonstration, a conveyor, an egg carton

Stream: skills6. Branch `feat/dojo-skills6` (from skills4 f349671, rebased on main after #28 and again after #33). Code:
`dojo/skills/interactive.py` (observation-driven skills), `dojo/skills/watch.py` (watch reads), `dojo/skills/holders.py`
(holes + hinged lid), planner rules in `dojo/skills/planner.py`. Tests: `tests/dojo/test_skills6.py`.

Runs: `~/dojo-skills6` (commit-stamped copy, `scripts/skills/deploy_skills6.sh`), light lanes through `g3_light.sh` with
`DOJO_STREAM=skills6`, at most two lanes at once. Data: `compute3:~/dojo-data/skills6/g3/<run>_<commit>/`. Catalogue for the
five tasks' 60 assets, built with perception's builder, `--no-glb`, 225 MB: `compute3:~/dojo-data/skills6/catalog`. It
includes `links.json` for the egg holder. Put it first in `DOJO_CATALOG`. Every G3 run has `SIMLAB_USD_SYNC_LINKS=1`, which
is the lead's rule and since #29 the runner's default. G3 = TRUE state in the physics-only twin. The entry point is always
`planner.solve`, chosen from instruction words plus objects. No task names, no LLM calls.

Runs made before the second rebase carry the pre-rebase commit hashes. The skill code is the same as on the branch: main
#29-#33 touched no file under `dojo/skills`.

## 1. Board context and order of work

Board scores (PyRUA-Lean / best entry; points to 100 in Average units):
- play_tic_tac_toe (LH): 97.2 / 100, 0.07;
- fill_egg_holder (LH): 21.2 / 21.2, 1.97;
- imitate_sorting_sequence (M): 92.4 / 92.4, 0.25;
- match_and_pick_from_conveyor (M): 86 / 92, 0.47;
- pick_from_conveyor_by_image (O): 22 / 52, 1.95.

The stack scored 0 on all five, so each one taken from 0 is worth its full score times the weight. Order by expected points
per hour:
1. tic-tac-toe: fast, and LH needed it;
2. imitate;
3. the conveyor match;
4. the egg carton;
5. the image conveyor, parked (section 5).

## 2. What was built

| piece | where | what |
|---|---|---|
| `play_board_game(board, pieces, opponent_pieces)` | `interactive.py` | The board is an object with support frames `cell/k`. "Ours" is the identical-piece kind nearer the robot, "theirs" the other kind. Our turn: tic-tac-toe rules (win, block, centre, the corner opposite theirs, corners, edges; ties by distance) choose the cell. The piece is the outermost free one on the cell's side, so the central pieces, which both arms reach, are left for the last cells. Pick and place are planned together (`plan_transfer`), one arm at a time, with a lower pre-place for far cells and centred grasps only. Then bank and send both arms home. Their turn: board watch reads every 2 steps until a new opponent piece is stable in a cell, then wait for the scripted arm to be at rest. |
| `copy_demonstration(container, demo_container, objects)` | `interactive.py` | Two same-kind containers; ours is the nearer one. Hold still at home and read the demonstrator's container every 5 steps; `ArrivalOrder`: an identity arrives after 4 inside reads in a row. Once all identities that appear twice have arrived and the scripted arm is at rest, put each one's twin into our container in that order (`put_in_open`). If it does not go in, retry. A failure stops the sequence, because a later object would be out of order and the checker fails the episode. |
| `put_in_open(label, container)` | `interactive.py` | For an open container on the table: carry the object level over a free floor spot and let go 12 mm above the rim. The spots are the ones farthest from what is already inside. Planned one arm and one pre-place at a time. Relay: the other arm puts the object on a table spot both arms reach, then the container's arm takes it. `put_into` only works with drop points off the table. |
| `match_and_pick(objects, conveyor)` | `interactive.py` | Belt velocity is the median displacement of the objects over 4 steps (+4.0 mm/step in x). "First" = the object leading along the motion; target = the next object of the same identity. Intercept: the earliest step at which an arm can do a top grasp (else a slice or antipodal grasp), at most 40 deg from vertical, at the predicted pose. Candidate grasps are computed once in the object frame. Collisions are checked against the other objects where they will be. The plan has one IK target per env step: a descent tracking the belt (8 steps), the close (5), a lift-off with the belt, then 16 cm straight up. Wait, approach, re-read the target and shift the track by the prediction error, then judge the lift after the arm settles. |
| `fill_holes(objects, holder, lid_tag)` | `holders.py` | Holes are the openings in a depth map of the body link's mesh (4 holes, r 24-25 mm, plate top 48 mm, floor 2 mm). Each egg is stood up (long axis 0/20/35 deg from vertical) over a free hole and released with the fingertips 3 mm above the body (`fingers_out` + `ref_top`), through the servo tail. Front holes first: the open lid stands behind the back ones. Grasps for round sections: closing through the long axis, rolled about it, tilted about the closing axis. End-to-end grasps take an egg out of a crowded basket onto the table. Objects inside a round container keep their fingers over its inner surface, a radial height field from the mesh. The lid (`swing_closed`): fingertips on the lid's trailing face, moved along the lid's arc about the hinge. Done only when every egg is in, because the lid counts only in the final stage. |
| watch reads | `watch.py`, `GTProvider.watch_read` | The shape perception-R&D proposed (`kind: cells / container` -> `cells` / `inside` + `outside`, `stable_vs_prior`, `new`, `occluded`). G3 serves it from GT, so moving to eval is a provider swap. `ArrivalOrder` has the same API as perception's helper. |
| scripted actors at rest | `GTProvider.actors_moving`, `RunnerEnv.actors_gt` (dev-only), `interactive.wait_actors_still` | Answers "did a scripted arm (one that is not ours) move since the last call?". G3 reads its GT joints, gripper included. "Still" = 12 consecutive still reads, at most 160 steps. A provider without the method makes the skill hold a fixed time: 60 steps for the game, 110 for the demonstration. |
| shared, additive | `skills.py`, `motion.py` | `plan_place(..., ref_top=)`: the receptacle top for `fingers_out` and the servo hover, used when the object's bbox top is wrong (an open lid). `Collision`: `"label#link"` in `ignore` drops one link; `slab_links` (opt-in) models an articulated link as 2 cm slabs. Defaults unchanged. |

Registry: four `j2-registry/1` entries with `obs_needs.notes` (section 6). SKILL_DOCS v4.

## 3. G3 numbers

GT state, physics-only twin, `SIMLAB_USD_SYNC_LINKS=1`. Score = the checker's ladder tier (board scale x 100). Steps are
median / max (limit).

| task | run (code) | layouts | success | mean | steps | failures | board (PyRUA / best) |
|---|---|---|---|---|---|---|---|
| play_tic_tac_toe | v1 (006c5a4) | 100000-01 | 0/2 | 0.30 | 444 (1100) | the right back corner did not plan | 97.2 / 100 |
| play_tic_tac_toe | v4 (8d21857, plan_transfer) | 100000-03 | 2/4 | 0.88 | 792-840 | the last back corner did not plan, 2x | |
| play_tic_tac_toe | v6 (d217619, one arm at a time, centred grasps) | 100000-04 | **5/5** | 1.00 | 826 / 901 | - | |
| play_tic_tac_toe | **v7 (watch reads + actors at rest)** | **all 25 settled** | **25/25** | **1.00** | **832 / 901** | - | |
| imitate_sorting_sequence | v1 (8cbc27c) | 100002, 100004 | 0/2 | 0.00 | 689-718 (1600) | order seen right 2/2; `put_into` has no drop point on a table | 92.4 / 92.4 |
| imitate_sorting_sequence | v2 (7a6e6e5, put_in_open) | 100002, 04, 10, 14 | 2/4 | 0.51 | 872-1328 | relay found no table spot (1), object turned in the hand (1) | |
| imitate_sorting_sequence | v3 (17acd42, watch reads) | first 2 of 13 | 0/2 | 0.10 | 1578-1591 | watched to the step budget: the container read never read as stable (identical twins keyed by asset); fixed, test | |
| imitate_sorting_sequence | **v4 (848f38e)** | **the 13 with validated demos** | **10/13** | **0.77** | **1276 / 1527** | the first object's grasp failed twice: a slip on the carry (toy car), a grasp that closed beside the object (phone), a relay collision | |
| match_and_pick_from_conveyor | v1b (369e7a0) | 100000-02 raw | 2/3 | 0.67 | 190 / 286 (700) | ice cream: no top grasp | 86 / 92 |
| match_and_pick_from_conveyor | v4 (7ad3f4c) | all 25 raw | 20/25 | 0.80 | 199 / 362 | ice cream 3 (no grasp), a standing bottle and a 24 cm figurine (16 cm lift out of reach) | |

| match_and_pick_from_conveyor | **v5a (e4235d6..5d9ba52: 12.5 cm lift, waits for a match to come into view)** | **all 25 raw** | **20/25** | **0.80** | **237 / 334** | ice cream 3, headset slipped 1, 24 cm figurine 1 (the standing bottle now lifts; the headset held in v4) | |
| fill_egg_holder | v1 (6b466c6) | 100000-01 settled | 0/2 | 0.00 | 59-60 (700) | no grasp (basket wall filter); lid read wrongly as closed | 21.2 / 21.2 |
| fill_egg_holder | v2 (62ed59b, wall model, end grasps, table relay) | 100000 | 0/1 | 0.10 | 363 | 1 egg in; end-to-end grasps slipped on the relay; planning 8 min | |
| fill_egg_holder | v3 (e4235d6) | 100000-01 | 0/2 | 0.00 | 174-333 | eggs fell out of the pads while the hand turned them up | |
| fill_egg_holder | v4 (dd05e1a, held 10 mm below the middle) | 100000-01 | 0/2 | 0.00 | 182-477 | same | |
| fill_egg_holder | v5 (5d9ba52, slow eased carry) | 100000-01 | 0/2 | 0.05 | 586-605 | rolled grasps 0/4 held; a level grasp 1/1 stood its egg in a hole | |
| fill_egg_holder | **v6 (29876c6, level pads only for the stand-up)** | **all 11 settled** | **0/11** | **0.03** | **160 / 452** | grasp_slip 6, place_miss 3, collision 2: one egg in 3/11 episodes; most eggs sit packed against the basket wall, where no level grasp is free, and the relay to the table loses them | |
| pick_from_conveyor_by_image | v1 (5d9ba52, layouts v13) | 100000-04 (ours, v13) | 0/5 | 0.00 | 7 (700) | no intercept for the basket: its handle is 1 m up, ~0.48 m out | 22 / 52 |

Notes:
- The ttt v7 run's chunks were deployed from 194e49b, 7ad3f4c, 848f38e and 17acd42. Between them `play_board_game` and
  its waits changed only in a provider fallback the G3 provider does not take.
- One episode takes about 2.5 min of wall time (planning 37 s median).
- In the watch-read runs the opponent's and the demonstrator's arm read "still" 45-62 steps after their last piece or
  object. The old fixed waits (48, 80) would have been just enough on our demos, with no margin for other demos.

## 4. Failure modes and what fixed them

play_tic_tac_toe:
1. Far back corners. Each arm reaches the back corner across the board (0.58 m) but not the one straight ahead of its own
   shoulder: the pre-place IK fails. `plan_transfer` gives its 24 full plans to the best-scored screened pairs, and one
   arm's pairs can take all 24, so the other arm never got a full plan.
   - Fix: plan one arm at a time, with a 1.5 cm pre-place as fallback.
   - The piece choice now keeps the central pieces for last. Before, the last piece (far right) had to reach the last
     cell (far left back) and slipped at that reach.
   - Before the fixes 2/4 episodes ended at 0.75 (ttt_v4); after, 5/5 (ttt_v6).
2. A round piece held 13 mm off its centre (box grasp `off +13mm`) is pinched on a chord. It turned out of the pads on
   the carry and landed 21 cm from its cell: 2 of ttt_v5's failures. Fix: centred grasps only.
3. corner indexing bug in `choose_cell` (rows' ends instead of the board's corners): KeyError, ttt_v2.

imitate_sorting_sequence:
1. `put_into` only drops off the table edge ("no drop point off the table"), and both containers here stand on the
   table. New `put_in_open` (imit_v1 -> v2).
2. Objects far right (x 0.3-0.4): the right arm cannot reach basket0 at x -0.4, and the relay to a table spot found no
   place either (the default pre-place too high). Now: more relay spots between the arms, pre-place fallbacks.

match_and_pick_from_conveyor:
1. The lift was judged 2 steps after the last command while the arm still lagged its command: "rose 35 mm", the arm
   sent home, the object dropped. Now judged after the arm settles, and the lift is slower.
2. Planning was 15 min on one episode: candidates for wide objects (slices / antipodal on 0.5 M-face meshes) were
   recomputed for every intercept time. Now once, in the object frame.
3. Objects wider than the hand at the top have no intercept within 40 deg of vertical (ice cream: 104 mm scoop over a
   69 mm cone, 3 of 25 targets). The tall ones (a standing 15 cm bottle, a 24 cm figurine) had their 16 cm lift out of
   reach: the lift is now 12.5 cm (the checker wants 10).

fill_egg_holder:
1. Eggs lying in the basket (about 31 of the 44 eggs in our settled layouts) were judged ungraspable: the finger's root
   block (38 mm thick, 56 mm wide, 40-70 mm above the TCP) meets the basket's rim or wall, or the neighbouring egg.
   - Grasp model: round-section grasps (rolled about the long axis), end-to-end grasps leaning up to 50 deg into the
     basket, narrowed opening, siblings' boxes without a margin, the basket's inner surface as a radial height field,
     and finger samples allowed 8 mm into it.
   - Kinematically graspable eggs went from 7 to 22 of 44 (dev layouts, both arms).
2. The open lid is a 4 cm shell standing behind the body with a lip on top. Its link box (with the lip) covered the
   back holes, so no back-hole place planned. Now: link slabs (opt-in) and the front holes first.
3. An egg's bbox top is the open lid (12 cm over the plate), so `fingers_out` raised the release far too high. Now
   `ref_top` = the body's top.

## 5. Parked and open

- **fill_egg_holder: first version only, no measured gain.**
  - Each part works on its own:
    - the holes come from the body mesh;
    - the stand-up place plans, kinematically for 22 of the 44 dev eggs;
    - an egg released over a hole drops in standing: twin v5 100001, 3.6 deg tilt, scored;
    - the lid swing plans (never run in the twin: it only runs once all four eggs are in).
  - The blocker is holding a 0.3 kg egg in the X5's pads while the hand turns it upright. In the twin it fell out
    between the pads on almost every carry: rolled, raised and end-to-end grasps, slow carries, grasps off the middle.
    Most dev eggs lie packed against the basket wall, so the level grasp that held is rarely free.
  - Next: a better hold (a regrasp from the table with the egg wedged against something, or pads at the egg's lower
    third measured in the twin).
  - PyRUA's 21 suggests 1-2 eggs per episode. Ours is near 0, so this task does not count in the projection yet.
- **pick_from_conveyor_by_image:** the generator fix (v13) and a first skill (`pick_into_held`); first G3 numbers in §3.
- **Wide objects on the belt** (ice cream: 3 of the 4-5 match_and_pick failures): a side grasp of the cone on the
  moving belt.

## 6. obs_needs (for integration-2 / perception-R&D)

Every skill reads through its provider, so moving from GT to perception swaps the provider, not the skill.

| skill | read | region (how the skill knows it) | when / how often | returns to the skill | tolerated latency |
|---|---|---|---|---|---|
| play_board_game | (b) `watch_read` kind `cells` | The board's pose and asset from the step-0 estimate; cells from its `cell/k` frames; own / opponent = the two piece assets | Every 2 steps during the opponent's turn | Per cell: occupant (own / opponent / None), pos; `stable_vs_prior` | A new piece must be seen within ~10 steps of its release (its arm needs 33-43 steps to go home) |
| play_board_game | (a) `actors_moving()` | The scripted arm (not ours) | Every step after its piece is stable, until 12 still reads (max 160) | bool (None = cannot tell -> hold 60) | 2-3 steps. Gripper motion must count as motion: arms pause ~10 steps while their gripper closes |
| play_board_game | object estimates (`refresh`) | Our pieces, the board | At the start of our turn and after each placement (our piece within 10 mm of its cell, else re-placed) | Poses | Normal refine |
| copy_demonstration | (b) `watch_read` kind `container` | The demonstrator's container (of two same-kind containers, the one farther from the robot, from the step-0 estimate); candidates = identities that appear twice | Every 5 steps from step 0 until the expected number has arrived and a read is stable | `inside` / `outside` [{asset, pos}]; an asset arrives after 4 inside reads in a row (`ArrivalOrder`) | Objects arrive >= 100 steps apart and only the order matters: any cadence up to ~20 steps |
| copy_demonstration | (a) `actors_moving()` | The demonstrator's arm | After the last arrival, until 12 still reads | bool (None -> hold 110) | As above |
| copy_demonstration | object estimates | Our twins (by identity), our container | Before each pick; inside-check after each drop | Poses, inside | Normal |
| match_and_pick | (c) object tracks on the belt | Every object on the belt | Velocity from two reads 4 steps apart, at the start and before a retry; one target re-read at the end of the approach (~8 steps before contact) | Positions by identity, belt velocity | Target error < ~8 mm at contact: velocity good to ~0.03 mm/step over the <= 300-step prediction, or a fresh re-read <= 3 steps old (a known latency is shifted by v * latency) |
| pick_into_held | (c) belt tracks + what the board's picture depicts | Every object on the belt; the picture (a Geometry board) | As match_and_pick; the picture once at the start | Identity of the depicted object (G3: the picture asset's public `map.json`) | As match_and_pick |
| fill_holes | object estimates (static scene) | Eggs, holder, basket | At the start, after each drop ("in a hole": root within r_hole - 4 mm of the hole centre, bottom >= 10 mm below the hole top), lid angle before / after the swing | Poses, lid joint angle | Normal |

The registry's `reads` tokens stay within the j2 vocabulary (`object_args`); the watch cadence is in each entry's `notes`.
