# 2026-10-28: a pick refine's recentre from a partial view (stream H, put_bottles deep-dive)

**Context.** Stream M's arm-8 write audit found the pick refine's RECENTRE chain net-harmful on put_bottles: 7
harmful vs 4 helpful writes. A > 10 mm move is applied, the wrist re-hovers over the new spot, and the second view
"confirms" it.

**The mechanism is not a neighbour.** The audit text said the fit locked onto "the neighbouring lying bottle". In all
11 harmful refine_pick+confirm writes (bottles 7, organize 2, bowls 2), the nearest other believed object is 81-300 mm
from the recentred pose. In both named cases (gcoff_100004 @183 bottle1, gcon_100004 @56 bottle0), no other GT object
is within 150 mm. The fit slid 30-47 mm along the SAME bottle, at iou 0.85-0.90, and the second view moved it only 2-7
mm more. So:
- a footprint guard against believed neighbours would catch 0 of 11;
- "the confirm view's IoU >= the first's" would catch 4 harmful, but would reject ~12 of the 19 helpful recentres.

**What separates them** is the first refine's `visible_frac`, i.e. the object partly out of the wrist view:

| first refine visible_frac | harmful | helpful | neutral / uncertain |
|---|---|---|---|
| < 0.9 | 5 (0.53-0.63: both audit cases and the 191 mm one) | 1 (a gross 202 -> 177 mm) | 5 |
| >= 0.9 | 6 (5-31 mm) | 18 | 8 |

**Built.** `RuntimeConfig.refine_recentre_guard`: "centre" (default, today) | "vis".
- "vis": a pick refine whose move would trigger a recentre or confirm (> `rehover_min_shift_mm`) from a view showing
  less than `place_min_vis` (0.9) keeps the prior pose, and no recentre runs. Event: `applied: false` with the why;
  stat `recentre_partial_view`.
- 0.9 is the existing constant the place side uses for the same rule (a confident belief needs a full view), so
  nothing is tuned.
- Tests: `tests/dojo/test_recentre_guard.py`, built from the two audit cases' numbers.

**Evidence status.** The rule was chosen while looking at the arm-8 episodes (the X2 / audit set). Only a rendered A/B
on other layouts (v0 of 100005-09 + 100020-29) counts as its evidence. Variant left in BACKLOG: on a partial view,
re-hover to see the whole object and re-fit from the prior.
