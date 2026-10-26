# 2026-10-26: dojo skills9: the refined pick keeps its paired grasp; the held-pose protocol at the pre-insert hover

Stream: skills9. Branch `feat/dojo-skills9` from main 331a551, rebased on 5b01218 (#59); this part as its own PR
(`feat/dojo-skills9-replan`). Code: `dojo/skills/skills.py` (shared: `execute_pick`, `plan_pick`, `plan_transfer`,
`run_pick`, `run_transfer`). Tests: `tests/dojo/test_skills9.py`. Runs: commit-stamped copies `~/dojo-skills9{,b,...}`
(`scripts/skills/deploy_skills9.sh [dir]`) and main 331a551 in `~/dojo-skills9m`, light lanes through `g3_light.sh` with
`DOJO_STREAM=skills9`, at most two at once (a queue runner, `~/dojo-data/skills9/scratch/qrun.sh`). Data:
`compute3:~/dojo-data/skills9/g3/<run>_<commit>/`. The run commits (4db6a14, ...) predate the rebase; they stay
reachable on the branch `skills9-runs`. G3 = the TRUE state in the physics-only twin; mode B = `g3 run --mode-b` (no
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

Regressions with no offset: G3 never enters this path (no refine hook); the full set (plug 100000-09, kong 100000-04,
store_tools 100000-04) runs on the follow-up's head, which contains this commit, and is reported there.

Spend: no LLM calls. Light lanes, at most two at once: six 5-layout mode-B runs (about 1.5 h of lane time).

## 2. The held-pose protocol at the pre-insert hover

In a follow-up PR on the same branch line (`feat/dojo-skills9`): the skills side of perception-held's `held_pose` /
`show_pose` (#59), and a G3 test provider that answers them from GT with #59's calibrated noise.
