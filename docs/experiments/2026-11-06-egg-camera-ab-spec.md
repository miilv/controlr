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

Gain cell: fill_egg_holder, at least **10 distinct settled v6 variant draws**, one paired
repeat each. Rendered/proxy ownership is restricted to our namespace by `dojo.simlab.paths`:
100010-19 and 100050-99 are the other team's rendered slices. They must not be used here.
The [unused G3-only draws](2026-11-06-egg-unused-g3-layouts.json) are **not** a camera manifest.

The pre-registered gain sampling procedure is:

1. Use only base ids **100000-04**, with fresh variants **v=1..50**; never the original records.
   Candidate G3 tuning used the original 100000/01, not these new variant draws. Do not use
   held-out 100005-09 / 100020-29 for tuning, training-block ids, or any id >=100100.
2. Preserve the archived **v6** placement algorithm (`0434230e:dojo/simlab/gen_layouts.py`).
   For each `(id, v, attempt)`, replace only its RNG sub-seed with the first four SHA-256 bytes,
   interpreted as an unsigned big-endian integer, of the UTF-8 string
   `EGG-camera-v6-2026-11-06-r1:{id}:{v}:{attempt}`. This replaces the original expression
   `seed * 1000 + attempt`; it does not replace task sampling/placement semantics with v15.
   Record `_simlab.variant=v`, the exact algorithm/source hash, and this protocol name.
3. Generate/presettle in ascending `(v, id)` order. Choose the **first ten stable records**,
   based only on settling, before any camera/skill outcome is read. Stop and report a sampling
   failure if fewer than ten survive 250 draws; do not lower n or choose by arm performance.
4. Stage each variant under its own private layout root with its actual base id. Freeze the
   ten `(variant, id, SHA-256)` tuples, settle exclusions, source and config in a manifest
   before dispatch. Pair by that tuple, so reused base ids are not counted as the same layout.
   Neither G3 tuning nor camera runs may use the selected fresh draws before the frozen A/B.

This is a frozen sampling **protocol**, not an already-generated camera manifest. Its generation
stage remains deferred with the failed G3 prerequisite; no camera inputs have been generated or
rendered under it. The lead must retain the manifest produced by the protocol if a new, separately
registered implementation later justifies dispatch.

Guard cells: fill_pen_holder and insert_tubes, each 10 paired settled **owned-namespace** dev
variant draws frozen by the lead using the same manifest-before-outcomes rule.
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
