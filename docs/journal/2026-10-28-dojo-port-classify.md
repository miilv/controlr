# 2026-10-28 — Stream P: classify_objects / classify_objects_by_language through the ported classify_into_bins_v3b

**Done** (branch `feat/dojo-port-classify`, on top of the merged port #259):
- **Vendored code.** `dojo/skills/ported/skills/` gains robot-harness a21dd6f's `classify_into_bins_v3b`, copied unchanged.
  - It brings its whole dependency tree: classify_into_bins, put_into_bin(_rgb, _rgb_strict, _rgb_strict_v2, _rgb_strict_v3
    with fk.py), planar_drag_rgb(_strict), return_home_budget.
  - That is 26 files, all loaded by relative path.
  - TIER_A gains "classify_into_bins_v3b" (reserve 20, no args), as K4 ran it on both cells.
- **Route** (`planner.ported_plan`), behind `ported_routes` like the rest:
  - instruction: "baskets / bins" + "sort | classify | group ... category", or two or more "X objects into the left / middle /
    right basket" clauses (the skill parses the fixed map itself);
  - world: two or more basket / bin containers and two or more other rigid objects.
  - Layout 100001 has three objects for three baskets: the first condition, "more objects than baskets", printed Plan none
    there, and was fixed.
- **G3 record.** The episode record keeps the skill's own `details` log (`ported_runs[*].details`, cut to 20k characters):
  per object, the mode, the container, `why`, any fallback, and whether it ended in its container.
- **TierAView.** The skill wraps our adapter in Mikhail's `TierAView`. Its held_look asks `obs` for the objects near the
  TCP plus the finger state, i.e. our belief and finger sense outside G3 mode gt: no truth there.

**G3 mode gt** (compute7, settled v0 100000-09, a615b0a / 0aa98d5 (same code, plus the details log), `ported_routes=on`;
Mikhail's K4 numbers are on 100010-14):

| cell | ours | mean score | Mikhail K4 |
|---|---|---|---|
| classify_objects | 2/10 (+1 still running when written) | ~0.4 | 5/5 |
| classify_objects_by_language | 3/7 so far | ~0.55 | 3/5 (mean 70) |

**Failure classes** (from the details logs):
- **v3's clearance model refuses a carry ("carry: no clear path")**, most often for the last 1-2 objects of 7-9. An object
  then stays on the table; the score ladder gives 40 (or 10 / 15).
- **The old tier-A fallback pick reports "not held"** (`fallback_in=False`).
- **Step budget.** 9 objects at ~80-100 steps each, plus retries, run into 1100. Two by_language episodes stopped with
  "budget".

Our numbers sit below his on 100010-14. Two possible sources, both unmeasured:
- The slices may differ in difficulty (his own report: "generated layouts are not the same difficulty").
- Our IK stands in for cuRobo's. v3 plans every carry from the IK goals `reach` returns, then checks the joint line between
  them with its own FK capsules. A different IK branch sweeps differently.

Both cells score 0 today (Plan none), so even the partial scores count.

**Spend:** 0 LLM; ~25 G3 jobs (one layout each) on compute7 light1-4.
