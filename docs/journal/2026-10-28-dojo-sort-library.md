# 2026-10-28 — Stream P: classify with our own library (`ported_classify=library`, default v3b)

**Why.** The ported classify_into_bins_v3b (#266) is low in G3:
- classify_objects 2/10 (mean 0.40), classify_by_language 3/7 (mean 0.56);
- its clearance model refuses carries ("no clear path") for the last objects of 7-9.

**Done** (branch `feat/dojo-sort-library`):
- `dojo/skills/sorting.py`:
  - `sort_groups(instruction, world)`:
    - containers = basket / bin categories, left to right;
    - items = every other rigid object, clutter included, grouped by category;
    - the instruction's fixed map ("X objects into the left / middle / right basket"), else the least-distance one-to-one
      assignment;
  - `sort_into_bins(sk, groups)`: `library.arrange` "put" steps (our pick / place_oriented into each container), then one
    more pass over the objects not inside their container while ≥ 120 steps remain.
- `planner.ported_plan` routes the sort instructions to it with `RuntimeConfig.ported_classify="library"` / G3
  `--skill-attr ported_classify=library`. The default "v3b" is unchanged.
- The checker accepts any one-to-one category → basket map for the free sort (classify_objects.py: per basket, any
  category wholly inside; every category below the basket rim); by_language needs the named map.

**G3 mode gt** (compute3, one job at a time, settled v0):

| cell | layouts | v3b (#266) | library |
|---|---|---|---|
| classify_objects | 100000-04 | 2/5, mean 0.51 | 3/5, mean 0.76 (lib1 6e18859 and lib2 725221b, the same outcomes) |
| classify_objects_by_language | 100001-05 | see the PR | see the PR |

The library's misses are single objects:
- a grasp slip, or a loose grasp handed over and dropped;
- place_oriented finding no grasp-place pair for one object (100001 by_language).
arrange's place verdict reports "place_miss" for objects that are inside the basket (dz against the support-top target);
the episode score counts them.

**Spend:** 0 LLM; ~15 G3 jobs on compute3 (one at a time).
