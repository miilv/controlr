# 2026-10-29 — stream N: classify_into_bins_p1 (grounding that sorts what it can)

**Symptom.** With #299 (`series_labels` "numbered") the classify camera arms route to the ported skill. The first A/B rows
on compute4:
- c4b_classifysg_numon_100005 ran (720 steps, score 0.4).
- c4b_classifylangsg_numon_100005 ended at step 0, twice: "grounding: instruction map covers [...]".

That run had one phantom, a wooden_toy at confidence 0.08.

**Cause.** v3b's `fixed_map` raises unless the perceived categories are exactly the instruction's three: any unnamed
category (a misread item, a phantom) or a named one never seen. The free sort has the same weakness on its own side:
with more perceived groups than baskets, v2's `plan2` finds no permutation, raises, and nothing is sorted. Camera
rows had 4-5 perceived categories in 3 of 8 planning records.

**Change (default off).** `RuntimeConfig.ported_classify`: "v3b" (today) | "p1". p1 is the new vendored-copy variant
`dojo/skills/ported/skills/classify_into_bins_p1` (v3b plus "N:" changes):
- **Fixed map.** A named category with no single perceived group is skipped (logged in `skipped`). A perceived group the
  instruction does not name is left on the table (`left`, `left_groups`). It raises only when no named category is
  seen at all.
- **Free sort.** One group per container, the largest ones; the rest are left.
- **Left items.** They stay obstacles for v3's motion stages and are never assigned. They are not passed to the plan
  stage, because `plan2` gives every object it is passed a container.
- **Logging.** A refused carry logs v3's transit record (`refused`: the blocking obstacle, the via heights tried), for the
  true-state diagnosis.

When the categories match exactly, p1 does what v3b does.

**Tests.** `tests/dojo/test_classify_p1.py`:
- fixed_map skip and raise;
- `keep_groups`;
- the route switch and the default;
- kinematic end to end, an unnamed fourth category under a fixed map: v3b fails grounding at 0 steps, p1 sorts the
  three and leaves the fourth;
- kinematic end to end, free sort with 4 groups for 3 baskets: v3b does nothing, p1 sorts.
