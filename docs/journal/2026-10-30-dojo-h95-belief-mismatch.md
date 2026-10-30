# 2026-10-30: "belief_mismatch" across tasks: no false alarms; the toaster lever's margin; `press_repeat` (stream H)

**Scope.** Integration's camera rows on compute4 (`runs`, `runs2`), dev ids only (100000-04, 100030-49): 423 rows.
No measurement-id row was opened.

**What the label means.**
- `dojo.runtime.eval.classify` returns "belief_mismatch" when the skill finished OK on its belief, nothing in its
  history failed, and the checker does not count a success. That is 30 of the 423 rows.
- All 30 read `stages_left 1, final_left 0, success_flag True`.
- For these tasks `run_reward` is a single `rm.check([...])` stage, so "stages_left 1" means only "not a success".
  It does not mean "everything but going home".

| task | belief_mismatch rows |
|---|---|
| store_tools_in_toolbox | 15 |
| play_tic_tac_toe | 4 |
| make_toast | 3 |
| fasten_screws | 3 |
| hang_mugs | 2 |
| sort_nesting_dolls_by_size | 1 |
| arrange_largest_number | 1 |
| arrange_largest_number_random | 1 |

align_blocks, stack_bowls and fold_clothes_random have no dev belief_mismatch row on compute4.

## Which condition fails: replays

`scripts/dojo/belief_mismatch_replay.py` runs on compute3, in the physics-only twin, on each row's own dev layout.
1. It replays the row's executed actions open-loop.
2. It evaluates every condition of the remaining success stage (`check_once`).
3. Then it drives both arms home with open grippers over 60 steps and checks again.

Results:
- **The replays are faithful:** in 30/30 the replay's score equals the row's official score.
- **The world was wrong in 30/30.** Every row fails a task condition, never only "all_robot_back_to_origin".
  Going home converts 0/30. There is no false-alarm class for a tolerance, a home move or a re-check to recover.
- **What fails, by task:**
  - store_tools: tools not at their slot point and not covered by the toolbox (pliers / hammer / wrench).
  - hang_mugs: mugs not hung (`is_lift` false).
  - tic-tac-toe: a piece outside its cell's z range.
  - fasten_screws: a nut off its bolt (xy / axis).
  - arrange: one digit off its mat.
  - dolls: row alignment.
  - make_toast:
    - in 2 rows, ONLY the lever ("toast_botton" under 85 %);
    - in 1 row, slices tilted out of their slots.

## The runtime already sees most of them

The post-place reads (`post_place` events), compared per placed item with the replay's truth:

| post_place verdict | placed item truly failed | truly OK |
|---|---|---|
| not ok | 36 | 28 |
| ok | 6 | 40 |

- "Not ok" catches 36 of the 42 real failures, but 28 of its 64 flags are on good places. Precision is 56 %.
- So redoing every "not ok" place would re-pick 28 good items to fix 36. That is not cheap or safe in general, and it
  is not built here.
- By task:
  - store_tools: 25 of 41 "not ok" are real and 0 real failures read ok.
  - make_toast: the read is blind to bread in the toaster (6/6 "not ok", 4 of them fine).

## make_toast's lever: a 0.1 mm margin; `press_repeat` (new switch, default "off")

- **The margin.** `press_joint` presses the lever's end edge. In G3 mode gt the lever ends at a joint of 0.0405-0.042,
  and the 85 % check needs about 0.0404.
- **G3 mode B,** make_toast raw 100001-02, toaster believed off (4 / 8 mm in x or y): at 4 mm the lever stops at 0.039.
- **"cross":** four more strokes after the first, at the press point moved 4 mm each way along the object's horizontal
  axes. It is called from `slots_and_press` (make_toast only).

| toaster offset | off | cross |
|---|---|---|
| x 4 mm | 0.5, 1.0 | 1.0, 1.0 |
| x 8 mm | 0.25, 1.0 | 0.25, 1.0 |
| y 4 mm | 0.5, 0.5 | 1.0, 1.0 |
| y 8 mm | 0.0, 0.5 | 0.0, 0.5 |
| mode gt, 100000-04 | 5/5 | 5/5 (+~90 steps / episode) |

- In the 2 camera dev rows whose replay fails only the lever (make_toast rsk 100001, 100002, both 0.5), this is the
  class "cross" targets. The camera screen comes next.

0 LLM calls. compute3: 30 replays and 10 short G3 jobs, one at a time.
