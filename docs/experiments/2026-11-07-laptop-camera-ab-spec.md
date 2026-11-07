# Prospective laptop headset camera A/B — not dispatched

## Scope and release prerequisites

This specifies a future camera experiment, not an adopted candidate or a job
request. No camera jobs, board changes or model calls are authorized by this
file. Compute3's current Isaac 5.1 renderer is incompatible; physics-only G3
cannot qualify it. A compatible rendering host must be assigned separately by
the lead, with the same box class for both arms. No compute4 access is implied.

The latest screened mechanical source is `18e70412`. A final qualifying source
and complete bundle must be pinned before dispatch; this candidate remains blocked.
Before any dispatch, retain exact source, baseline board-candidate config,
perception/model/asset/calibration pins and hashes in a manifest. Reuse the
lead-pinned board candidate in A unchanged; neither this spec nor the PR
changes its selection. No unresolved config pin may be filled from outcomes.
The G3 reliability/paired result in [round 2c](2026-11-07-laptop-g3-round2c.md)
must qualify first. If it does not, this registration stays blocked.

B changes only the runtime skill bundle:

```yaml
hang_cradle: straddle
headset_seat: 'on'
headset_carry: staged
headset_grasp: dense
headset_table: mesh
```

A retains all baseline values. No store_laptop route, perception refinement,
planner, model, identity routing, held-read policy, joint-limit or step-budget
change. Note that G3 isolated the bundle with straddle in both arms; this
camera experiment transfers the complete bundle against the actual baseline.
All new motion/grasp flags default off/legacy. Threaded mug hooks do not take
the cradle-specific code path.

## Paired sampling and budget

Gain cells: store_laptop_and_headphones and its random cell, ten fresh paired
settled dev layouts each (20 pairs). Guard cells: hang_mugs and hang_mugs_random,
five paired layouts each. Total 30 pairs / 60 episodes, separate from all G3
panels. No sealed records. For camera use only our measurement id ranges:
gain ids 100000–09; guards 100020–24. Generate v15 variant-14 salted dev draws with salt
`laptop-camera-prospective-2026-11-07-88c48b74`; freeze layout hashes before outcomes, settle once, share exact records between arms.
Never reuse a record for which this candidate has already produced a camera
outcome. Driver legality must be checked before generation/dispatch.

Alternate A/B order by id. Same settled layouts, host, model/perception pins,
step limit and deployment in each pair. Record all attempts, including retries;
predeclare infrastructure-only replacement rules before launch, never replace
a policy failure. Report actual n, exclusions, score mean/sample SD, headset
credits, full successes, score wins/losses/ties and success flips by cell.
Use the task checker's final partial score; internal `not_home` is not a score.

Actual calls/tokens/spend so far: **0/0/$0**. Future calls depend on the pinned
board perception route: inventory its per-episode requests and estimate 60
matched episodes before dispatch; set a lead-approved budget and retain actual
transport attempts, calls, tokens and spend in the report. This spec does not
permit using agent-session credentials or changing models to fit a budget.

## Decision rule and belief audit

Gain cells pooled B minus A score >=5 points; score wins (>=10-point increase)
>=2*score losses, with >=2 net wins. Report every cell separately; retain the
G3 7/10 per-cell headset-credit target as a separate reliability diagnostic,
not a substitute for the paired camera rule. Guards: B mean no more than five
points below A and no excess success-flip losses over wins.

Run the existing paired belief-write audit on physics-only replays of both
camera arms. Require no excess harmful writes by writer in B. These skill
flags add no perception requests or writes themselves, but changed paths and
occlusions can alter existing reads. Include those changes and unavailable
reads; do not count unknown held evidence as a negative. Do not feed audit GT
back to the policy. In particular, audit the known headset 00002/00004 identity
confusion and articulation-root plumbing without introducing a new fix here.

A gain is a camera adopt-candidate only after the score and write gates both
pass. Otherwise keep the flags default off and the board unchanged. Preserve
numeric traces and, on the compatible host, one successful and one typical
failed video using existing capture/replay tooling.
