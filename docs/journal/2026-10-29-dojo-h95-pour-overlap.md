# 2026-10-29: pour_by_language, the overlap pace: the other arm grasps the next bottle during each pour (stream H)

Method: G3 mode gt (physics-only twin, GT world, 0 LLM, traced) on the v1 draws of pour_by_language 100030-39, with
`--routes pour_order`, `ported_routes=on` and `ported_pour_aim=b10`. Kinematic-env unit tests for the mechanics.

## Why (the compute1 camera rows, journal 2026-10-29-dojo-h95-pour-lean)

- Camera mode spends 42-46 steps on two looks before the first pour, and that eats lean's 46-79-step G3 margin.
- The lean pace's normal-speed moves slipped 5 of 12 camera pours. v1's pace slipped none of 10.
- Three v1-speed pours cannot fit 800 steps even fully overlapped: 320 + 270 + 214 ≈ 804 in G3, ~848 with the looks.
  So the overlap needs a little speed too.
- The look budget stays as it is: a per-task look budget would break the single-fixed-system rule (the lead's
  ruling).

## The pace (`ported_pour_pace` "overlap"; default "v1"; lean unchanged)

- **Companion** (`pour_bottle_lean.companion_plan`, `run(companion=..., pre_held=...)`):
  - While a pour runs, the other arm grasps and lifts the next pair's bottle. Its approach, close and lift ride along
    with the pouring arm's moves, through the adapter's two-arm move. The close runs inside the drain.
  - Its carry to over its own bowl rides along with this pour's set-down, strictly after this pour's upright edge.
    The pours stay in order.
  - The next pour then starts with the bottle in hand and over its bowl. It skips its grasp and the first profile
    angle.
- **Side rule**:
  - A companion only for a side bottle (|x| > 5 cm), on its own arm's side.
  - Its approach stays ≥ 0.20 m from the bowl being poured.
  - The bugs behind this rule are below.
- **Arm choice** (`pour_order.alternate_arms`): a centre bottle takes the arm that makes consecutive pours alternate.
  This is what lets the companions happen.
- **Speeds**:
  - v1's stages (tilt-up 40 / 70 / 90, untilt 90 / 60 / 0), each re-solved from the measured in-hand pose; no
    single-move rush;
  - below 100° and in the untilt at 0.8 of the held joint limit (v1 "slow" = 0.6);
  - the flow angles at v1's speed and holds;
  - drain 12;
  - no homes between pours.
- **Adapter** (`harness_robot.PAIR_COMMANDS`, `move_pair`):
  - A synchronised two-arm move, built only when `ported` runs `pour_bottle_lean` under "overlap"
    (`HarnessRobot(pair_moves=True)`).
  - Every other caller gets "unknown command" as before (unit test).

## Bugs found on the way (G3)

| what | fix |
|---|---|
| the left arm reaching over to a bottle at x = +0.2 hit the pouring right arm (0.44 rad tracking error, grasp missed) | companions only on their own side |
| a companion approach 0.158 m from the poured bowl: the tilting bottle met it and turned 28-50° in the hand (both speeds) | clearance 0.20 m |
| companions on the centre bottle (between both bowls): 2 slips of 36-47°; their side-arm grasp (yaw ±30) slipped once more | no centre companions |
| the bottle righted inside the carry back spilled (lean journal) | the companion's carry starts only after the upright edge |

## Screen (final code; slip = the pour stopped on in-hand slip)

| speed | success | slip stops | successes end at | steps per full pour |
|---|---|---|---|---|
| v1 pace (pa_lang_b10) | 0/8 | 0 | - (all out of steps) | 355-358 |
| lean | 3/10 | 5/30 | 721-754 | ~264 |
| overlap 0.7 | 3/10 | 2/30 | 718-764 | median 281 |
| **overlap 0.8 (frozen)** | **5/10** | **1/30** | **682-747** | **median 268; pre-held 222-230** |

- **Speed: 0.8.** It has the lower slip rate (1/30 against 2/30) and more successes. Frozen as the "overlap" preset
  before any camera run (the lead's rule).
- Companions: 8 of 20 transitions held the next bottle.
  - All 5 successes had at least one held hand-over. In 100030/31 both were held (arm order l, r*, l*).
  - Without one the budget binds. 100035/38 order a right, a centre and a left bottle: no alternation is possible,
    no companion, out of steps at 798.
- The 5 failures:
  - 3 out of steps (one with no companion possible);
  - 1 slip (100039, the third pour, grasped by itself: 40°);
  - 1 neck grasp that did not lift (100036, second pour).
- Camera mode adds its 42-46 look steps. 4 of the 5 G3 successes would end by 791. The compute1 camera A/B against v1
  judges it.

Spend: 0 LLM calls. G3 on compute3: 14 pour_by_language jobs of 5 episodes (~15 min each), including the runs before
each fix; one at a time.
## Camera A/B and the park (lead, 2026-10-29)

Rendered A/B (Integration, compute4, measurement ids 100005-09 + 100020-24, 10 pairs, 53f10437): A = the candidate
(aim b10 + pour_order); B = A + `ported_pour_pace` "overlap".
- Score +4.0 on average, but 0 extra successes. The candidate is 0/10 and overlap is 0/10 too: every episode is bound
  by the step budget.
- Earlier, lean on the compute1 rows: 0/4 (2 out of steps, 2 "plan" endings with 5 of 12 pours slipped).

| pace | G3 (v1 draws 100030-39) | camera |
|---|---|---|
| v1 | 0/8 | 0/5 (out of steps at 776-799) |
| lean | 3/10, 5/30 slips | 0/4, 5/12 slips |
| overlap 0.8 | 5/10, 1/30 slips | 0/10 |

The arithmetic that binds:
- a v1-speed pour is ~320 steps from grasp to set-down, and ~214 from carry to the upright edge;
- three pours do not fit 800 even with every grasp overlapped (~804);
- camera mode adds 42-46 look steps before the first pour;
- in G3 the overlap's successes ended at 682-747, which camera turns into 726-791 plus whatever its pours lose to
  perception;
- the only motion fast enough to fit (lean's normal-speed rush and untilt) slips in camera.

**pour_by_language is parked** (the lead's ruling). Both switches stay default off. Unparking needs one of:
- a pour that is reliably faster without in-hand slip;
- a step-lean perception, which would be a generic all-task change with its own A/B, not a per-task look budget.

## The last pour's untilt may spend the move reserve (census, compute4 overlap rows)

- In 3 of 10 camera overlap rows (100006, 100020, 100023) the third pour ran its whole profile to 150 deg.
- Then its untilt's next move was refused for "budget" with 34-47 steps left, and the episode ended with the bottle at
  60 / 150 / 90 deg: no upright edge, no success.
- What refused it: the adapter's 20-step move reserve (TIER_A RESERVE, kept for a final home). The last pour has no
  use for that home, because its upright edge ends the episode. A 60 -> 0 move needs ~25 steps.
- In the candidate rows (v1 pace) the third pour never starts (15 steps, "step_budget" before the profile), so there
  is nothing to save there.

Fix:
- `pour_bottle_lean(untilt_reserve=0)`: the untilt moves and their "direct upright" fallback may spend the reserve.
- pour_order passes it only for the last pair, and only under "overlap".
- Every other move keeps its reserve (unit test: the same move sequence, and only the untilt moves change their
  reserve).

G3 check (v1 draws 100030-39, same code): 5/10 → 5/10, 0 up / 0 down.
- G3's own near misses (100033, 100035, 100038) are truly out of steps: 14-20 left at the refused move.
- So the fix can only show in camera mode, where the refused moves had 34-47 steps left. Integration's compute4
  A/B is the judge.
