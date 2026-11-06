# EGG: egg-holder skill ceiling on v6 dev layouts

Workstream requested a large skill gain, not another slow-carry trial. No camera or model calls.
Compute6 only (`robotics`, RTX 5090), deploy `~/controlr-egg`, data `~/dojo-data/egg/`.
At most two EGG light jobs; shared light slots 1-4, no heavy lock. Job launchers check disk,
VRAM <=24576 MiB, use 8G / 600% scopes, and retain PID receipts. No compute4 or dc19 access.

## Prior evidence and protocol

Read CLAUDE, DEVELOPMENT, ARCHITECTURE, EXPERIMENTS (including the default camera rule),
fill_egg journal references and BACKLOG, and G3 mode A/B tooling. The old pair-planning and
slow-carry trials did not improve score; neither is repeated here. The basket wall-foot and
back-hole reach are the main prior hypotheses. Current perfect-state runs use mode A: no
`--mode-b`, no verifier, actual physics/checker, no LLM.

Initial ids 100000-04: only 100000/01 have stable settled records; 02-04 are exclusions,
not skill failures. Initial baseline: 0/2 successes, scores .25/.25, 324/209 of 700 steps.
Eight eggs: three no-pick exits, one picked egg with no drop plan; one carry loss was recovered
by a retry. The GT trace for 100000 shows target0 on the table x=.382 at the end of its
closed-jaw carry, before release (step 165). Budget failures: zero.

Private extra layouts: archived generator v6 (`0434230e:dojo/simlab/gen_layouts.py`, stored
on compute6 at `~/dojo-data/egg/v6_gen.py`), fresh ids 100050-99. Presettle retained 13/50:
53,54,55,67,71,72,75,80,84,91,96,97,98 (all with prefix 1000).
No shared layout records were modified. No tuning on 100005-09, 100020-29 or >=100100.

Frozen ten-layout tuning set: 100000,100001,100010,100012,100015,100016,100019,100053-55.
All ten manifest generator tags are `simlab/v6`; private settled records and SHA-256 manifest:
`~/dojo-data/egg/dev10/`. The other ten fresh survivors are reserved for the prospective
[camera A/B](../experiments/2026-11-06-egg-camera-ab-spec.md), not further G3 tuning.

## Switches (all default off)

- `egg_grasp_height=raised`: original candidates plus upper-shell pinches at +6/+10 mm;
  dense wall guard unchanged. The pad is 30 mm long; the hypothesis was rim clearance.
- `egg_grasp_angles=wide`: steeper rolls/tilts, including checked lateral corridors below
  the flare. Same clearance threshold; shallow side entries check their own 70 mm corridor.
- `egg_drop_approach=low`: 20 mm rather than 50 mm preapproach, release geometry unchanged.
- `egg_wall_model=shell`: a conservative lower/upper radial material band rather than
  treating all air below the highest mesh vertex as filled. Neighbor-bin inflation and
  clearance threshold unchanged. This is separate from candidate angles/heights.

Each is a RuntimeConfig/Skills/G3 switch, validated and covered by CPU tests. No default,
carry speed, object mass, simulator scene, safety threshold, score/checker or step limit changes.

## Exploratory screens

First raised screen (`raised-initial`, n=2): mean .325 vs .25; 100000 saved 144 steps at
unchanged score, 100001 gained .15. Its repeat in the larger sequence did not reproduce the
gain, so the small screen is not a ceiling claim.

Ten-layout baseline `base10`: 0/10 success, mean .235, sample SD .0474, scores nine .25
and one .10; median 327 steps, max 425. Episode budgets were never exhausted. Logged events
(not exclusive final-egg classes): 20 no-pick exits, 3 slipped picks, 9 no-drop-plan exits,
7 failed release outcomes, 19 successful release outcomes. Several failed carries were retried.

Ten-layout raised arm `raised10`: 0/10 success, mean .190, median 388 steps, max 603.
This lever regresses and must stay off; the positive two-row screen did not survive validation.

Wider-angle screen `wide2`: scores .25/.10, 0/2 successes vs .25/.25 baseline; parked.
An outward-contact diagnostic on 100000's target2/3 found neither a push plan nor a pick;
no outward-nudge switch was shipped. The probe did not alter the step limit (700).

The screen source was developed while jobs progressed; unchanged off paths were used by
baseline. Some early G3 `code.commit` receipts are absent/stale (deployment stamp lagged the
source transfer); treat these as exploratory. Freeze and replay the final PR head for any
candidate claim. The final paired measurement and test receipts follow below.

Additional screens (all n=2, 100000/01; baseline .25/.25):

- Lower preapproach `low2`: .25/.10. It can force a stand-up fallback and lose a previously scored egg.
- Shell envelope `shell2`: .25/.25. target3's formerly filtered plan rose 0 mm in both layouts;
  newly feasible geometry is not evidence that the physical hand gets under the rim.
- Shell + wide `shellwide2`: .25/.25. On 100000, target2 **did lift** (0.3 mm in-hand drift)
  but had no drop plan. This specifically motivates the axial-spin drop candidates.
- `egg_drop_spin=axis`: original drop candidates followed by spins of +/-25, +/-55, +/-90 deg
  around the egg's long axis. Unlike the old upright carry, the long axis stays lying.
  This is a fifth, separate default-off switch. Default-off poses are preserved bit-exactly.

Final source freeze: `dd9ddca5d7ebd06b1beccb339cf34e5c3a9b537c`. Ten fresh paired runs of
all-off versus shell + wide + axis are in `final-base10` / `final-combo10`; both use that
same source snapshot and the private `dev10` manifest. No further implementation edits
are made while those jobs run. The camera specification targets this combination, conditionally
on a useful G3 result; it does not authorize a run for the rejected raised-pinch lever.
