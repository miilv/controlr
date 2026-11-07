# BYIMG round 3: released stability and target retention

Branch `feat/dojo-byimg-r3`, off the lead's merged round-2 source 3892232b.
[PR #416](https://github.com/miilv/controlr/pull/416).

The native checker and intended endpoint differ. Exact replay reproduces R2's
100012 closed-jaw latch, but the continued skill ends at only 14.3 mm target
rise. R2's skill-only 100017 stays released at 119.7 mm rise / 32.5 mm bottom gap. Keep
native task score and unlatched post-chain measurements separate.

Four carry misses include two target losses (385/456 mm relative drift,
empty closing gaps), one retained target with 89 mm drop error, and one basket
rolling 131 mm / 60° while the target stays in its pads. Fixed-wrist inset
carry, full squeeze, attachment checks and a second rim pinch were implemented
under new default-off `belt_first_recover` / `belt_first_phase` fields mirrored
in RuntimeConfig, Skills and G3. Thin riders reserve 24 approach-settle actions;
a 100 mm pull is tried only after a no-action planning miss. Defaults and the
round-2 baseline schedule remain unchanged.

The fresh prefix isolation is 5/10. Adaptive component screens reach 7/10 with
a geometry-specific timing gate and pull, but independent frozen-prefix
confirmation gives **6/10**, with three no-intercept plans and one -5 mm slip.
Successful targets stay 122.83–125.99 mm above reset after ten stationary
actions. This does not establish reliable target acquisition.

The basket attachment remains a physical limit. A dual-rim lift scores after
release on the spent glue pilot but continues to 50 mm bottom gap / 14.4° tilt.
A half-tilt correction ends at 42.6 mm gap, outside 40 mm. Raising the empty
basket before release reaches bottom 868 mm / 3.6°; after a 21 mm alignment and
release, load collapses the bottom to 817 mm. Long settling does not establish
a rigid rim attachment. Forty horizontal-hook profiles, long waits, staged
approaches, underside palm support and angled fork support were also probed.
Hook command/actual TCP residuals 43–103 mm persist; underneath approaches
hit native IK or physical tracking limits. Exact blocking contacts remain an
unresolved diagnostic; photo Geometry is not proven to collide.

The one registered full-task comparison uses sixteen fresh v15 variant-19
layouts 100060–75, new salt, native-phase settlement and identical hashed
inputs in A=B-r2 and B=B-r3. No outcome-driven tuning, retest, camera run or
board change. The result is **18.75±40.31 → 0±0**, 3/16→0/16 native successes,
**0 wins / 3 losses / 13 ties**. Reject B-r3. Nine layouts have no fixed-wrist
inset path; one completes the loaded command with its basket bottom still
762.2 mm. No B success qualifies for the post-termination audit. B-r3's
absolute success remains the candidate's zero; no Average gain is claimed.

The compute6 [perception specification](../experiments/2026-11-07-byimg-perception-spec.md)
is parked: classify missing buffers versus zero detections, capture native
5.1 calibration backgrounds for Dynamic/conveyor/00001, then use identical
measurement pixels to validate all six labels, photo identity and causal belt
tracking. Four GPU-hours / 20 GB are planning allowances, not observed
throughput. No agentic call is included.

All jobs use compute3 light3/light4, admission checks, 8G/600% user scopes,
physics only. The invalid component variant and a missing settlement module
were refused before episodes; corrected before their outcomes. No other
process, shared dataset, driver or renderer was changed. Model calls/tokens/
spend: **0/0/$0**.

A later panel-log audit found that shared G3 `planner.solve` unconditionally
calls `finish`: both grippers open and both arms home. Earlier direct-skill
continuations bypassed it and are explicitly labeled skill-only. Full-planner
continuation is a separate unscored diagnostic; the frozen official panel is
unchanged. A future held-basket bundle must scope its finish policy as well
as solve the physical grip. No post-outcome behavioral patch enters this panel.

Full local suite: **2084 passed, 191 skipped**; focused guards 20 passed.
All new behavior remains default-off. The PR stays open for lead review; the
negative bundle is not adopted. Exact paired inputs, source hashes and all 32
episodes are archived separately from 289 prefix/prototype records and one
spent-layout full-planner diagnostic.

The full-planner R2 100017 diagnostic reproduces latch 304/release 276, then
normal finish plus 100 hold actions ends at **-32.1 mm** target rise, **758.5 mm**
basket bottom and 9.7 mm bottom gap (current lift 0/inside 1). Thus even the earlier
skill-only retained 119.7 mm rise is not a stable full-controller success.
