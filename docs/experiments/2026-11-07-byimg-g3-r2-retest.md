# BYIMG lift-first: once-only independent fresh retest

Registration before new outcomes. The first registered panel (variant 11,
100010–19) is 10→20 mean score, 1/10→2/10 success, 2 score wins / 1 loss /
7 ties. Its +10 points passes the mean bound but fails the two-net-win bound.
The experiment protocol permits one fresh retest of a positive rejected lever.

Freeze the exact **same** implementation and A/B fields as round 2, source
`db52accf`. No tuning from the first panel enters this retest. The exploratory
rise-before-carry route is excluded (it fails or loses the target in pilots).

New inputs: v15 variant 13, salt `byimg-r2-once-retest-20261107`, fourteen
untouched BYIMG ids 100024–29 and 100042–49. The latter are fresh **variant**
draws permitted for G3, not the calibration records; no camera evaluation or
calibration read/write is performed. No input from either prior panel is used.
Native-phase settlement is identical to round 2; select by rest/stable reload
before skill outcomes. Freeze full input hashes and run all 14 pairs once.
Replace only infrastructure failures, never misses.

Common: `--routes basket_image`, `belt_container=on,belt_deposit=on,belt_lift=off`.
A: `belt_first=off,belt_target=off`; B: `belt_first=on,belt_target=match`.
G3 mode A, physics-only, same compute3, zero model calls.

The **independent 14 pairs**, not a retrospectively pooled total, decide the
same rule: >=5 mean score-point gain, >=2*losses score wins (10-point threshold)
and >=2 net wins. Report mean/sample SD, successes, pair flips, stage counts,
steps and infra/missing meshes. The combined 24-pair total may be reported only
as descriptive evidence. No third retest, no early stop, no outcome-driven
replacement or candidate selection.

If the independent retest passes, write a parked camera specification with
perception prerequisites; dispatch nothing and change no board candidate. Keep
PR #414 draft and unmerged for the lead. The two light lanes, admission gates,
8G/600% scopes and private BYIMG roots remain unchanged.

## Results

Pending registered independent retest.
