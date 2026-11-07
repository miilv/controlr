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

[Exact inputs/hashes and all 28 episode records](2026-11-07-byimg-g3-r2-retest-results.json).
Fourteen stable pairs, no exclusions or replacements, same frozen source.

A: **14.29 ±36.31**, **2/14 successes**. B: **21.43 ±42.58**, **3/14 successes**
(sample SD in score points). Score and success pair wins/losses/ties:
**2/1/11**. B rescues 100042/48, loses A's 100027, and both succeed on 100044.
The **+7.14-point** mean gain again passes the mean bound but fails the
**two-net-win** bound. The once-only independent retest therefore **does not
pass**. No third retest, camera A/B specification, camera dispatch or candidate
adoption follows.

A's failures: nine lift, three target-pick endings. B's failures: five target
picks, five carry/drop endings, one complete motion chain that still scores
zero. Three B episodes achieve full native checker score. Thus basket-first
has a physical scoring chain, but target acquisition/retention and release
stability remain unresolved; do not infer success from a skill's `ok` alone.

Median steps A 186 / B 211.5; maxima 311 / 384 of 700. Episode skill wall totals
355.4 / 390.3 s, planning 87.3 / 67.9 s; boot/reset time is additional. No
unstable episodes, missing meshes, infrastructure failures or budget exhaustion.
Shared-box wall times are descriptive, not a latency benchmark.

Across both registered blocks, **descriptive only**: A **12.50 ±33.78**,
3/24 successes; B **20.83 ±41.49**, 5/24; 4 wins / 2 losses / 18 ties.
The pooled total is not substituted for the pre-registered independent gate.
This is a small, consistently positive development signal, not camera adoption.

All new behavior remains default-off and PR #414 remains draft/unmerged for
the lead. Model calls/tokens/spend: **0/0/$0**.
