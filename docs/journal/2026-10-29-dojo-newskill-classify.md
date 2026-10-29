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

## True state: refused carries (`ported_classify` "p2" = p1 + the retry)

**Cause.** Every item v3b lost at G3 went through a carry v3 refused ("carry: no clear path"), almost always into a basket
that already held an object. The refusal log (p1 records v3's transit for each refusal) shows two things:
- Most blocks were v3's own drop pose leaving the gripper over the basket rim ("wall:<target basket>" at frac 1.0 for
  every via height). Some were the held object against a wall or a placed item, or the other arm.
- After the refusal the object stayed in the hand. v3b moved on with that arm parked mid-air, which is what blocked the
  other arm ("other_arm").

**Change** (`classify_into_bins_p2`, a copy of p1 plus `high_retry`; p1 itself was adopted as it is). A refused carry
gets one more try, released `HIGH_CLEAR` 6 cm over the rim and `HIGH_MARGIN` 3.5 cm inside it
(v3: 2.5 / 2). It goes from the hand (`drop_held`) or picks again. If the object is still held after that, v3's
`put_back` sets it down where it was picked and homes the arm, and v3b's own fallback can then run. Tried and dropped:
parking the idle arm at home before a one-arm step. Home did not clear the "other_arm" blocks, and it cost 100031.

**G3 mode gt, v1 100030-39, compute3** (v3b ed44940f vs 6f2f6309, the p2 code under p1's name before the split; in
G3 p1 = v3b because the categories match, so this is p2 vs p1 too; `ported_routes=on`):

| cell | v3b (= p1) | p2 | changed rows |
|---|---|---|---|
| classify_objects | 6/10, 0.735 | 8/10, 0.855 | 100030 0.4→1, 100039 0.4→1 |
| classify_objects_by_language | 3/10, 0.52 | 4/10, 0.55 | 100038 0.4→1, 100037 0.4→0.1 (step budget) |

7 of 13 retries landed in the basket. Of the 6 put back, the fallback landed 4. What remains: pick refusals near a
basket wall, failed relays and nudges, and the step budget on 9-item scenes.
