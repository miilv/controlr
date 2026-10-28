# 2026-10-28 — Stream M: the grasp-time perception metric, its baseline and the frame-0 ceiling

Brief: `/tmp/dojo-team/PERCEPTION_95.md` (stream M). Report: [../experiments/2026-10-28-grasp-time-metric.md](../experiments/2026-10-28-grasp-time-metric.md).
Branch `feat/dojo-m95-grasp-metric`. 0 LLM calls. GPU: compute3 light1-2, physics-only replays only, one job per
slot (never more than 4 of our GPU jobs on the box).

## What was built
- `dojo/perception/grasp_metric.py`:
  - `score_event` is pure. Any belief source (the runtime's, stream C's per-event belief, a frame-0 record) scores
    against GT rows.
  - `episode_events` gives the runtime's own belief at every grasp / place of a saved integ episode.
- `dojo/runtime/eval.py` `BeliefProbe` is dev-only. It snapshots the provider's belief at every gripper change into
  the report (`dev_belief_probe`). New runs no longer need the rebuild.
- `scripts/perception/m95_replay.py`: physics-only GT replays, many episodes per runner, stopped after the last
  gripper change.
- `scripts/perception/m95_score.py`:
  - `jobs` / `score`;
  - `table`, with `--taxonomy`, `--corr`, `--frame0` and `--fails`.
- `scripts/perception/m95_ceiling.py`: the perfect per-object selector over every frame-0 source.

## Landmines found on the way
- **Pre-shaped grippers** (stream C found it on hang_mugs).
  - A "first command below 0.5" rule takes the pre-shape on the approach for the close: the wall grasp goes 0.43 on
    the approach, then 0.0 about 30 steps later.
  - `grip_events` takes the last closing command of a run.
  - `GTProbe`'s `dev_gt_probe` closes still use the first-close rule (BACKLOG).
- **Origins vs bodies.** The charger's origin sits off its body centre, so B1's 180-degree flips read as 26-48 mm
  position errors. The metric compares body centres, and the flip shows as `rot`.
- **Empty jaws carried in the belief** (`not_held`).
  - Breadth-round store_tools: the hammer never left the table (GT constant over the carry, jaws at 0.0), while the
    belief carried it into the toolbox ("pick ok, in-hand drift 0.0 mm").
  - The episode category was `belief_mismatch`.
  - This is the largest class at place events.
- **Replays are exact.** Every close with a `dev_gt_probe` reproduced the episode's GT nearest-object distance to
  0.0 mm. That includes the genuine-5.1 rendered c4 runs, whose physics runs in the same engine.
- **compute3 is CPU-bound for physics-only replays** at load 13-30. 691 episodes took 2 h 40 min on two slots, with
  each replay stopped after the episode's last gripper change.
- **Stacked objects need 3-D matching.** A held block over its interchangeable base shares its xy. An xy-nearest match
  called it the base and an identity error: 34 of 38 stack_blocks episodes, 33 of them successes. Matching is now in
  3-D, as eval_g1 does for make_kong.
- **The rebuild must follow the runtime's own attach / release rules** (stream H): in 18 of 98 `not_held`, the
  runtime never attached the object at the close. **post_place moves are logged as from / to** (stream W).
  Stream W's #210 logs pose7.

## Results (report: experiments/2026-10-28-grasp-time-metric)
- **Latest round per task:** 900 events, 71 % clean at R10, 77 % required-only.
  - Blockers of the 264 unclean events: released 72, not_held 56, neighbour 40, in-hand 35, target 30, identity 27,
    reference 4.
  - 219 more clean events reach 95 %.
- **All rounds:** 4767 events, 75 % at R10.
- **The runtime vs the frame-0 record on the same events:** clean R10 333 -> 636 of 900.
- **R:** the AUC for success is 0.81-0.82 at R5 / R10 / R15 / required-only. Keep R10, with required-only and R5
  alongside.
- **The frame-0 ceiling** (a perfect per-object selector over B1, SAM, Identity and the agent arms): 70 % of scenes all
  right (75 % with make_kong's tile faces excused), against B1's 51 %.
- **Process:** the lead steers the streams by the gap table: W released, H not_held / in-hand, C ttt actors,
  S clutter.
