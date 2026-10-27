# 2026-10-27: dojo Long-Horizon stream (round 3): tic-tac-toe after #77, make_kong's topple confirm, organize / tower / store_tools / fill_pen triage

Stream: Long-Horizon. Box: compute8 (heavy lock 1 + light slots 1-6); before it was ready, compute3 light slot 4.
PRs: #107 (tic-tac-toe: the opponent's moves from the watch reads; merged), #141 (make_kong: `snap_row`). Branches not
merged (no evidence): `feat/dojo-longh-kong` (a second confirm look with the arm home), `feat/dojo-longh-ttt-actor`
(board read after the opponent's arm rests), `feat/dojo-longh-organize` (section 5). Data:
`compute8:~/dojo-data/longh/{g3,runs,replay,logs,scratch}`, `compute3:~/dojo-data/longh/`. G3 copies are commit-stamped
(`scripts/skills/deploy_skills2.sh dojo-longh-*`, run dirs `<run>_<commit>`).

## 1. Triage of breadth2 (proxy B, submit_v1, 100000-04)

**make_kong (2/5).** The brief said the skill side of `provider.confirm_pose` was never done; it was (#70, in integ7).
- 100001 / 100002: 8 of 8 topple confirms rejected. Head precision of the old standing pose 0.90-0.97, of the toppled
  pose 1.0; the rule needs +0.15. 100000 / 100003 passed on retries with the standing pose at 0.69-0.85.
- A synthetic check (head camera, the layout's meshes, the robot mask of the commanded joints at each confirm step):
  a tile lying flat leaves the standing hypothesis at 0.32-0.50. Something else fills it in the real frame; the
  topple ends with the TCP 4 cm above the tile's far side, the forearm over it.
- 100004: perception roles ("1 hand tiles match", the discard's identity).

**play_tic_tac_toe (0/5, 0.66).** Every place "dz -16.9 mm": #77's cause (pieces believed under the board).

**organize_table (0/5, 0.40).** alarm -> drawer "dz -227 mm" in 3/5: the drawer is a Geometry fixture, #77 covers it.
The rest is skill reach (G3 mode A is 0.44, section 5).

**build_tower (scan trigger 16/25, 3ac3597).** 7 perception_gross (to Perception-2). 100001 / 100020: a pick refine
"turned 90 deg" at iou 0.67-0.70 on pillar block5, a loose grasp (13-15 mm), the place at 90 deg rel tilt (to
Integration-3).

**store_tools (0.30).** Not the `execute_pick` landmine (fixed by skills-9 #64, in integ7: every executed grasp is the
planned one). The slot check reads "in: []" after places that reported ok, and the skill re-tries tools already in the
toolbox (0 grasps reachable). G3 mode A is 0.57 (2/10; the wrench fails 7/10).

## 2. tic-tac-toe

### G3 mode B with #77 + #79 (7d431e3, `--belief-drop fixtures`, 2 mm offsets on the 5 pieces)

| layouts | success | mean | steps |
|---|---|---|---|
| 100000-09 | **10/10** | 1.00 | 802-923 (1100) |

Reference (skills10, same offsets, old runtime rule, 100000-04): 0/5, 0.51.

### Genuine 5.1 B (compute8, `integ_eval.sh --renderer isaac51`, submit_v1)

Report: [experiments/2026-10-27-ttt-mode-b](../experiments/2026-10-27-ttt-mode-b.md). No fix (#77 + #79): 0.1 in 4/4;
#107 (merged): 100000-09 2/10, mean 0.66; + a board read after the opponent's arm rests: 2/10, 0.64 (not merged).

Found:
- #77 works rendered: every first piece "in cell/4, 0.1 mm off" (was dz -16.9).
- The opponent's move never reaches the belief in mode B (no grasp of ours, no release); `play_board_game` counted
  occupancy from the belief: "own 1 opp 0", a 200-step wait for a move that had happened (4/4 at 0.1).
  Fix (#107): occupancy = the belief + the cells a stable watch read saw the opponent land in; an opponent move = a new
  cell in the read (not a higher count, so a read that misses an older piece cannot hide it); a place believed off but
  seen in its cell by two reads counts as placed.
- The task's failure query: (either of our arms > 0.3 m / 30 deg from origin) AND (the opponent's arm away) ends the
  episode; the opponent starts only with an odd piece count on the board and our arms home. Any arm motion after our
  piece lands (a re-place) can trigger it. make_kong has the same query with the discard starting at step 0 (#102).
- The 5.1 cell reads drop pieces, flip own / opponent and see the opponent's piece in transit (spec sent to the lead
  for Perception-2; BACKLOG).
- The held_look in-hand refine moved a held piece's belief 42.0 mm at iou 0.66 (ok, adopted); the place was planned
  from it, the skill re-picked the piece from the believed spot (air), the motions overlapped the opponent's turn and
  the env ended the episode (100003). Same signature as breadth2's 41.7 / 42.1 mm place_miss rows. Runtime / perception
  (to the lead).

## 3. make_kong: the topple confirm

**Replay** (`scripts/dojo/replay_kits.py run --save-obs-at`, genuine 5.1 rendered kit, breadth2's 100001 action list):
- The topple worked: mahjong2_2 lies flat from step 135 at (-0.049, -0.103, 0.781), 2 mm from the pose the skill
  proposed. The belief kept it standing only because `confirm_pose` rejected it.
- At the confirm (step 141) the toppling arm's gripper hangs over the tile (flange (-0.049, -0.122, 1.03)) and covers it
  in the head view.
- Offline re-check on the replayed frames (perception service `check`, CPU silhouettes), IoU prior vs expected: head
  0.59 vs 0.57 (step 141), 0.62 vs 0.66 (170); the toppling arm's own wrist camera 0.075 vs 0.93, 0.08 vs 0.92.

**A skill-side fallback, measured and dropped.** `confirm_moved(arm=)`: a rejected confirm sends the empty arm home and
asks once more (`feat/dojo-longh-kong`, unit-tested; G3 unchanged: make_kong 100000-04 5/5, mode B `--confirm-pose`
100000-09 10/10 at 368-379 steps, plug 8/10, stack 5/5, store_tools 0.45, each equal to the reference). Genuine 5.1 B,
pre-#102 / #105 code:

| arm | layouts | success | what |
|---|---|---|---|
| main 48600bf | 100000-04 | 0/5 | "did not tip over" (confirms rejected) |
| + fallback | 100000-02 (stopped there) | 0/3 | the same; with the arm home the head check still reads standing 0.91-0.96 vs toppled 0.98-0.99 (8 of 8 rejected in 100000) |

So the head view cannot tell standing from toppled even unoccluded; the wrist view can. The runtime fix is #105 (a
wrist view with IoU >= 0.6 and >= prior + 0.15 also confirms, Integration-3), and #102 keeps scan / looks off the
discard (the task's failure query: either arm away from origin while the support arm is away ends the episode). The
fallback would only take the wrist away on the second look: not merged.

**Kong on main a3dc7e1 (#102 + #105)**, genuine 5.1 B, 100000-09: 6/10, 60.0 (report
[experiments/2026-10-27-kong-mode-b](../experiments/2026-10-27-kong-mode-b.md)); every topple that ran was confirmed (the
wrist view). The 4 losses are topple PLANNING: "push collision: waypoint 1/7: hits <neighbour>": perception puts a hand
tile up to 11.5 mm off its slot (5.1 p90), into the 13 mm gap the fingertips need. `snap_row`
(`feat/dojo-longh-kong-row`): the standing hand row onto its fitted slots in the planning world before each topple plan
(the belief untouched). G3 mode A 10/10, mode B `--confirm-pose` 10/10.

## 4. G3 checks on main after #93 (#77 + #79 merged)

| task | tier | layouts | result |
|---|---|---|---|
| build_tower | G3 mode B, fixtures, 2 mm offsets on 8 blocks | 100000-09 | 10/10 (steps 593-640) |

## 5. organize_table (G3; no PR)

Per item (approximating the checker from final poses), main 48600bf mode A, 9 layouts (100008's record is unstable):
alarm 2/9, figurine (garage) 0/9, keyboard 4/9, mouse 9/9; mean 0.44.
- Alarm: the drawer's top is a tray 23 mm under a rim (mesh height map). The place targets use the bbox top (the rim),
  the alarm lands across the rim (17-19 deg) or falls off the front onto the table (7/9).
- Figurine: with the drawer as an obstacle (arrange includes each place's support) no grasp-place pair within 12 mm of
  the 40 mm stand's centre (the held figurine meets the drawer's box); with the drawer ignored there is one.
- Keyboard: pushed away from the robot first, it ends beside the drawer, where no stroke fits behind its end for the
  sideways push; the pushes also turn it 15-26 deg and it slides on 20-30 % past a stroke.
Tried on `feat/dojo-longh-organize` (stacked on the kong branch), each G3 A/B on the same layouts: the push goal
(cover the frame, not its centre), a stroke-order lookahead on predicted poses (+ overshoot / 20-deg turn), off-centre
strokes, flat targets from the support's mesh height map, stand offsets to its half-size, push-confirm alternatives
(mode B). None moved the per-item counts (mode A 0.444 -> 0.444 in two versions; mode B 0.33 -> 0.32-0.33): the
flat tray targets have pairs but their full plans collide at the pick, the figurine's screen runs out of its 6000-IK
budget with 85 targets, the keyboard's sideways stroke still has no plan. Parked (BACKLOG).

## 6. Other G3 baselines (main 1f7e965, after #93; 100000-09)

| task | mode A | mode B (`--belief-drop fixtures`) | what fails |
|---|---|---|---|
| store_tools_in_toolbox | 2/10, 0.57 | 1/10, 0.55 (hammer / wrench 2.5 mm offsets) | the wrench 7/10: 0 reachable grasps against the toolbox (100003 / 04 / 07), or carried with a +71 mm offset grasp and missed; the hammer slips |
| fill_pen_holder | 1/10, 0.20 | 0/10, 0.11 (2 mm on the pens and the holder) | place_oriented ik_fail (0 grasp-place pairs), pens in the rim (collision), 1-2 pens in per episode |

store_tools mode B with the old runtime rule vs #77's (#79 code, 100000-04): 0.40 vs 0.40, identical per layout: G3's
emulation does not reach the "in: []" slot-check loop seen in breadth2 (proxy B), which needs a rendered check.

## 7. Spend and state

- LLM calls: ~350, all inside submit_v1's rendered episodes (the agentic perceiver), 70 rendered episodes on compute8.
  No direct calls.
- GPU: compute8 heavy lock 1 (rendered episodes, one replay), light slots 1-6 (G3); compute3 light slot 4 before
  compute8 was ready (one 10-layout G3 run).
- Open, ranked by points: store_tools planning (the wrench next to the toolbox; a 5.1 B run on main 1152d1d is the next
  measurement), organize_table (section 5), fill_pen_holder (G3 0.20), tic-tac-toe's cell reads (perception), the
  cross-reach pick of the far-left piece.

## 8. Later the same morning

**make_kong + `snap_row` (#141, merged).** Genuine 5.1 B, 100000-09: main a3dc7e1 (#102 + #105) 6/10, 60.0 (4 topple plan
collisions with a hand neighbour believed up to 11.5 mm into the 13 mm gap) -> main 13a28b6 + snap_row **10/10**. Report:
[experiments/2026-10-27-kong-mode-b](../experiments/2026-10-27-kong-mode-b.md).

**tic-tac-toe, a board read after the opponent's arm rests** (`feat/dojo-longh-ttt-actor`): 2/10, 0.64 vs #107's 2/10,
0.66 on the same layouts; not merged. Perception-2 rewrote the cell read (#151: gold chroma, arm-covered cells keep the
prior) from the replayed frames; the rendered pair main (#107 + #151) vs + actor v2 (no read-based landing while the
opponent's arm moves) runs after this entry.

**store_tools.** Genuine 5.1 B on main 1152d1d, 100000-09: 0/10, 42.5 (G3 mode A on the same layouts 0.57); losses:
belief_mismatch "stages_left 1" 6, plan 2, grasp_slip 2. A physics-only replay of 100008's action list reproduces its
"stages_left 1": the wrench (an off-centre grasp, 16 mm) slid out of the jaws ~15 steps after the lift; held_look had
said "held" just before; nothing re-checked before the release, so the skill placed an empty gripper. Under the strict
v1 contract the gripper observation is the command, so a finger-gap guard is not available: a vision check before the
release is the candidate (open).

**Why "0 grasp-place pairs"** (plan_transfer's screen instrumented on kinematic dry runs):
- store_tools' wrench against the toolbox (100003): 94 of 98 IK-feasible picks "hit toolbox" (its box + 6 mm);
- organize's figurine: the place hits the drawer's box;
- fill_pen's second pen "hits target2": an artefact of the kinematic env (no physics, the first pen not inside); in G3
  `fill_container` already ignores and grip_only's the pens inside.
PR #169: containers / supports as mesh-height-map obstacles (`container_obstacle`, `Collision.heightmaps`) in
`place_in_slots` and `arrange`. G3 mode A, paired: store_tools 100000-19 3/20, 0.588 -> 7/20, 0.675 (6 better, 0 worse);
organize 0.44 = 0.44; insert_tubes 10/10 = 10/10; regression plug 8/10, kong 5/5, stack 5/5.

**organize_table, genuine 5.1 B** on main 8c845c9 (#135), 100000-07: 0/8, 25.0, perception_gross 7 (the alarm not found
or 261-371 mm off, the figurine 42-90 mm off and turned 63-82 deg; p90 73.7 mm). To Perception-3. 100008 hung: Kit sat on
a GUI dialog (`zenity ... Failed to open display`) and held the heavy lock ~45 min; TERM did not stop it, `kill -9` of
the eval python did.
