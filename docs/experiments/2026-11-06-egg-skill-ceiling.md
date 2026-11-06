# Egg-holder grasp/drop feasibility in G3

**Question:** can improved geometry of a grasp corridor and drop pose substantially raise the
v6 fill_egg_holder skill ceiling with perfect state?

**Hypothesis:** a radial basket shell envelope, lateral grasps, and spins about the held egg's
long axis can retrieve eggs under the basket flare and make their drop poses reachable.
All changes are default-off RuntimeConfig fields, mirrored by Skills and G3.

**Setup:** compute6, Isaac 5.1 physics-only, `planner.solve`, GT mode A, no verifier, no model
calls. Source `dd9ddca5d7ebd06b1beccb339cf34e5c3a9b537c` (documentation-only PR descendants
are the same implementation). Same code, same ten settled v6 layouts in both final arms.
The private manifest is `compute6:~/dojo-data/egg/dev10/manifest.json`. Tuning ids:
100000,100001,100010,100012,100015,100016,100019,100053,100054,100055.
One run per layout and arm; the two-row screens are exploratory and do not establish a gain.

**Arms:** A = all five egg switches off. B = `egg_wall_model=shell`,
`egg_grasp_angles=wide`, `egg_drop_spin=axis`; `egg_grasp_height` and `egg_drop_approach` off.
The complete source and RuntimeConfig defaults are in the PR. The G3 commands are:

```bash
# Through the scoped, VRAM-checked light-slot launcher described in the journal:
scripts/simlab/rdpy.sh -u -m dojo.skills.g3 run --task fill_egg_holder \
  --seeds 100000,100001,100010,100012,100015,100016,100019,100053-100055 \
  --layouts ~/dojo-data/egg/dev10 --trace --out ~/dojo-data/egg/final-base10
# B: same command, --out final-combo10, plus:
# --skill-attr egg_wall_model=shell --skill-attr egg_grasp_angles=wide --skill-attr egg_drop_spin=axis
```

Do not execute the bare command without the light-slot lock, VRAM <=24 GB and memory scope.
No heavy GPU lock is used. The field `success` is full checker success; `success_flag` is not
a replacement for it. Scores below are in board points (100 × the raw G3 ladder).

The [journal](../journal/2026-11-06-dojo-egg-ceiling.md) contains the individual switches,
negative/null screens, original failure counts, sampling exclusions and deployment receipts.
The [camera specification](2026-11-06-egg-camera-ab-spec.md) is prospective and conditional:
no camera run is dispatched or adopted from a G3 score alone. Its ten fresh v6 survivors
are untouched by skill tuning. The board's task dimension labels differ between the brief
and the historical points table; both assign weight 1/40, and this report does not reclassify it.

G3 produces no camera frames, so no `turn_video.py` video can be made from these episodes.
Every failed rollout has the compressed per-step GT/joint trace and action stream in its run
directory; these are the evidence for blocked grasps versus carry and release failures.

## Frozen paired result

**No ceiling gain found.** A and B both score **23.5 ± 4.74** (mean ± sample SD), with
**0/10 full successes**. Every score pair ties: 0 score wins / 0 losses / 10 ties;
full-success flips 0/0. The final baseline reproduces the earlier baseline's ten scores
**and step counts exactly**. The machine-readable [results/config/manifest](2026-11-06-egg-g3-results.json)
contains both complete arms.

| Dev id | A score | B score | A steps | B steps |
|---|---:|---:|---:|---:|
| 100000 | 25 | 25 | 324 | 276 |
| 100001 | 25 | 25 | 209 | 413 |
| 100010 | 25 | 25 | 425 | 343 |
| 100012 | 25 | 25 | 272 | 322 |
| 100015 | 25 | 25 | 330 | 394 |
| 100016 | 25 | 25 | 201 | 314 |
| 100019 | 25 | 25 | 393 | 393 |
| 100053 | 25 | 25 | 420 | 420 |
| 100054 | 25 | 25 | 215 | 262 |
| 100055 | 10 | 10 | 373 | 468 |

Median steps: 327 -> 368; maximum: 425 -> 468 of 700. Recorded planning time totals
270.5 -> 945.6 s; episode wall totals 507.6 -> 1267.8 s. Load was not controlled for
this timing comparison, so the measured ratios are descriptive.

## Failure counts

Counts are logged attempts/events, not disjoint final-egg counts. Carry loss is diagnosed
from aligned GT traces at completed place approach (>60 mm object-origin to measured TCP),
then matched to the release outcome. Details and the separate raised-pinch failure table
are in the journal.

| Phase/event | A | B |
|---|---:|---:|
| No feasible pick after retries/nudge | 20 | 41 |
| Pick failed to lift | 3 | 19 |
| Carry loss ending in a missed placement | 5 | 4 |
| Successful pick, no drop plan | 9 | 10 |
| Placement/descent/release-tail miss | 2 | 2 |
| Episode step budget exhausted | 0 | 0 |
| Carry loss nevertheless landing in a hole | 2 | 1 |

Both arms have 19 successful release outcomes. B unlocks extra grasps, including a real lift
of an egg baseline cannot pick, but their post-pick routes still fail; it also executes more
bad grasps. Additional CPU pose/route probes and physics contact probes did not establish
a viable escape/drop sequence. This is evidence against these levers, not a proof that a
23.5 ceiling is intrinsic to the task.

## Disposition, spend and validation

All five switches stay **off**. The raised-pinch ten-layout arm also regresses (23.5 -> 19.0).
The prospective camera A/B is **deferred**: this G3 result supplies no useful gain to justify
spending a camera lane, and it is not adoption evidence under the default rule.

54 G3 episodes were recorded across screens and confirmations; 50 private v6 draws were
presettled, retaining 13. Recorded G3 episode wall/load plus presettle time totals 4978.8 s
(1.38 lane-hours), excluding Kit startup, diagnostic probes and lock waits. LLM/control/
planner calls **0**, tokens **0**, model spend **0**. Baseline replay waited about 35 minutes
for a light slot; only checked, still-unstarted EGG waiter PIDs were replaced, with no duplicate
rollouts. No extra locks or camera lanes were taken. All retained EGG GPU PID receipts have
exited at handoff.

CI on the implementation: **1791 passed, 187 skipped**, syntax and secret checks green.
Local full run: 1789 passed, 187 skipped; the latest focused skill/config/official-parity run:
27 passed. The local suite began before the last two tests were added; CI collected the final
source. Live/Isaac tests are intentionally skipped locally; the physics confirmations above
ran on compute6 from the frozen implementation. No robot/, safety or core loop changes.

The blocking evidence is a grasp and post-pick route problem, not lack of perception or robot
step budget. Further work needs a physically demonstrated grasp/escape sequence for eggs
under the flare before another skill rollout or camera batch is warranted.

A final geometry audit found why shell-only feasibility is especially weak evidence:
26/29 sampled radial bands are <=20 mm thick (median 2.28 mm), so the inherited -10 mm
band allowance can accept all of their material, and the six-sample descent can alias a
thin wall. A +1 mm-clearance, <=1 mm-step diagnostic in the recorded post-baseline state
left no target3 candidates; target2's two candidates per arm both failed IK. This is not
an impossibility proof, but no newly proposed grasp survived both strict clearance and IK.
