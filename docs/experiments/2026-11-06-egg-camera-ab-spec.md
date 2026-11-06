# Egg grasp/drop feasibility: prospective camera A/B

This is a specification for the lead, not a camera result or permission to promote.
No camera episodes have been dispatched by EGG. The G3 journal records the skill screen.

**Question:** does a shell envelope with lateral grasps and axial drop spins improve fill_egg_holder with camera-only state?
**Hypothesis:** modeling the basket underside unlocks a physical entry, while axial spins make the resulting grasp compatible with a drop.

## Frozen arms and prerequisites

Use the same PR head in both arms, the official bridge, genuine Isaac 5.1 rendering on compute6,
and one frozen camera policy/config/calibration set. Freeze source `dd9ddca5d7ebd06b1beccb339cf34e5c3a9b537c` (or its documentation-only descendant).
A sets `egg_wall_model`, `egg_grasp_angles`, `egg_drop_spin` to `"off"`.
B sets them to `"shell"`, `"wide"`, `"axis"`, respectively. Both keep
`egg_grasp_height: "off"` and `egg_drop_approach: "off"`. Other skill, belief,
held-read, verification, model, effort and step-budget settings must be identical.
Freeze the complete resolved config, commit, catalogue hashes and layout manifest before dispatch.
Use the lead's approved non-Claude router configuration; do not inherit a Claude perceiver default.
No GT feedback or mode-A remeasurement enters either camera arm.

G3 is a prerequisite screen, not an adoption gate. Do not dispatch this test if its final paired
G3 report provides no useful gain; park the candidate instead. This is a combined feasibility candidate; it does not identify an individual lever's effect.
The individual screens in the G3 journal are negative/null. Any other combination needs a new
specification rather than substitution after viewing camera data.

## Sampling and execution

Gain cell: fill_egg_holder, at least 10 distinct settled **v6** dev layouts, one paired repeat each.
The lead must freeze 10 layouts never run through either candidate's camera arm. EGG's G3 tuning
ids are 100000, 100001, 100010, 100012, 100015, 100016, 100019, 100053-55. Do not use the held-out
100005-09 / 100020-29 or any id >=100100 for this tuning-stream test.
The frozen gain ids are **100067, 100071, 100072, 100075, 100080, 100084, 100091,
100096, 100097, 100098**: the ten stable fresh v6 draws beyond the tuning set.
Their [SHA-256 manifest](2026-11-06-egg-camera-layouts.json) is frozen before any camera outcome.
Records reside on compute6 under `~/dojo-data/egg/simlab/layouts/fill_egg_holder/`.
Do not use these ten for further G3 tuning.

Guard cells: fill_pen_holder and insert_tubes, each 10 paired settled dev layouts frozen by the lead.
These exercise neighboring precision paths; neither calls `egg_grasps`, so an effect there signals
configuration drift or simulator noise. They use their normal generator, identical within each pair.
Alternate A/B versus B/A execution order by pair index on the same box. Record technical failures,
rerun only infra-invalid pairs with both arms, and report all exclusions and repeats.

## Decision rule (the repository default)

For the gain cell, B must improve pooled score by at least **5 points**, and success flips must
satisfy **wins >= 2 × losses**, with **at least 2 net wins**, on at least 10 pairs.
For **each** guard cell, B's pooled score must be no more than 5 points below A, and flip losses
must not exceed wins. Report per-layout scores, success pairs, mean/spread, steps, failure classes,
logical calls, transport attempts and tokens. Report partial-score gains even if the success gate fails;
they do not authorize adoption. A rejected lever with positive pooled delta gets at most one fresh,
pre-registered retest on layouts never used by that lever; never replay the rejected camera ids.

This is a skill-candidate change, not a perception writer. If deployment also changes perception
or belief writes, the separate harmful-write audit in EXPERIMENTS.md becomes mandatory before
any adoption claim, and the extra lever must be separated or independently registered.

## Budget

10 gain + 20 guard pairs = 60 rendered episodes. The lead must estimate calls from the frozen
policy's per-episode cap before booking a camera lane; there are no model calls in this G3 screen.
The camera run belongs to the lead's queue and is not launched by this PR.

## Dispatch disposition

The frozen ten-layout G3 pair was null (23.5 = 23.5; 0/10 successes both). The prerequisite
for spending a camera lane is not met. **Leave this specification undispatched.** No camera
outcome was used to write or revise it. A materially new implementation requires a new
pre-registration; do not silently substitute it into this frozen candidate.
