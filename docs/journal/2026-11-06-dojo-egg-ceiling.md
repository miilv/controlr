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
`~/dojo-data/egg/dev10/`. The other ten fresh survivors are [unused G3-only draws](../experiments/2026-11-06-egg-unused-g3-layouts.json).
IDs 100050-99 belong to the other team for rendered/proxy runs; they are not camera inputs.
The [camera A/B protocol](../experiments/2026-11-06-egg-camera-ab-spec.md) uses fresh variants
in our owned 100000-04 namespace, with settling-only selection and a manifest frozen before outcomes.

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

### Failure phases from aligned GT traces (the first ten-layout pair)

These are events/attempts, not mutually exclusive final-episode classes; retries can fail in
more than one phase. Every logged drop outcome aligns with a completed place-approach event
in its GT trace. At that event, object-origin to measured-TCP distance >60 mm flags an egg
already out of the pads (round grasps target its center). A failed release with that flag is
a carry loss; a failed release without it is a placement/descent/release-tail miss. This is
an analysis diagnostic, not a runtime threshold or a checker predicate.

| Phase/event | Baseline | Raised |
|---|---:|---:|
| No feasible pick after retries/nudge | 20 | 28 |
| Executed pick failed to lift | 3 | 23 |
| Carry loss ending in missed placement | 5 | 8 |
| Pick succeeded, no drop plan | 9 | 9 |
| Placement/descent/release-tail miss | 2 | 4 |
| Episode step budget exhausted | 0 | 0 |
| Carry loss nevertheless landing in a scored hole | 2 | 1 |

The last row is reported separately: a loss during approach can happen to fall into the
hole and score. Successful release outcomes total 19 -> 16. Raised versus baseline has
0 score wins / 3 losses / 7 ties; full-success flips are 0/0. Mean score 23.5 -> 19.0,
sample SD 4.74 -> 7.75 points. No ceiling gain is claimed from any small screen.

## Final confirmation and handoff

The clean `dd9ddca5` pair finished: **A 0/10, 23.5 ±4.74; B 0/10, 23.5 ±4.74** (sample SD).
All ten score pairs tie; A also reproduces the earlier baseline's ten step counts exactly.
B median steps 368 vs A 327, maximum 468 vs 425; recorded planning totals 945.6 vs 270.5 s
(timing load uncontrolled). The [final report](../experiments/2026-11-06-egg-skill-ceiling.md)
contains the per-id table, phase counts and [machine-readable config/results](../experiments/2026-11-06-egg-g3-results.json).
**The requested ceiling gain was not achieved. No camera dispatch or default change.**

The restart recovery checked the retained PIDs/logs before continuing. The completed candidate
was collected without duplication. Its baseline replay spent about 35 minutes waiting for a
shared light slot: the original polling waiter was starved, a single-lock waiter then waited,
and finally a checked unstarted waiter was moved to the candidate's released light slot.
Only the explicitly checked EGG waiter PIDs were stopped; other streams' processes were untouched.
The final baseline and all other retained GPU job PID receipts have now exited.

Extra diagnosis (not additional ceiling measurements):

- CPU reconstruction of 100000 target2's real held pose at step 258: no full drop route for
  standing-target tilts 90/80/70/60/45/25/0 deg from vertical. Some goal IKs exist, but the
  joint transit hits the basket and the fallback vertical lift fails IK near the reach edge.
- A direct Cartesian transit fallback, hypothetically empty access to all four holes, and
  short horizontal escape waypoints did not produce a complete drop. Short escapes themselves
  can plan; that is not evidence of a successful carry/place sequence.
- Physics shell-aware wall pushes (standalone source probes, not shipped switches): target3
  still had no nudge; target2 moved 21 mm inward or 28 mm outward, but no next pick planned.
  Steeper push approaches (35/55/70 deg lean) and two/three inward pushes did not fix it:
  a second push added about 7 mm, a third had no plan, and both arms still refused the pick.
  These contact probes reset the private settled 100000 each time and kept the 700-step limit.
  They did not bypass collision checks or modify physical properties.

Spend: 54 recorded G3 episodes, 50 presettle draws (13 kept), 1.38 recorded lane-hours from
G3 wall/load + presettle, plus startup/probes and queue wait; **0 LLM calls/tokens/spend**.
Local writes stayed below 1 GB (workspace about 147 MB plus small temporary data/archive).
CI implementation validation: **1791 passed, 187 skipped**, compile/shell/secret checks green;
local full 1789 passed/187 skipped, latest focused 27 passed. Final docs-only CI runs separately.
The first local suite's two parity failures were caused by changing RuntimeConfig while the
suite/subprocesses ran; a complete rerun and focused parity checks passed. No test was weakened.

Draft PR [#399](https://github.com/miilv/controlr/pull/399) is for lead review of a negative
feasibility investigation. All five knobs remain off; it is not an adopt candidate and is not merged.
The prospective camera spec defines ten fresh owned-namespace v6 variant draws and the repository's default gate;
its G3 prerequisite failed, so leave it undispatched. Next work needs a demonstrated physical
grasp/escape construction, not another slower carry or a larger episode budget.

### Shell-model landmine checked before final handoff

The sampled basket profile has 29 finite radial bands; 26 are <=20 mm thick, median
2.28 mm. With a signed band clearance and the inherited -10 mm allowance, every point
inside such a thin band can pass. Moreover, six descent samples can step over a thin
shell. The shell envelope is not a conservative swept-collision test under those settings;
its extra planned grasps must not be mistaken for physically viable entries.

A standalone strict probe used a +1 mm clearance and <=1 mm path spacing, with wide and
raised candidates, in the reconstructed recorded post-baseline 100000 state (two eggs
already placed). target3 had **zero** clear candidates in either arm. target2 had two
candidates per arm (the same pose at different opening widths); **both failed IK** in each
arm. No new grasp was executed or ceiling gain established. This stricter probe is not a
shipped switch. Its script/log is `~/dojo-data/egg/strict-probe.py` / `strict-probe.log`.
All EGG diagnostic CPU/GPU processes and retained waiters have now finished.

Camera-spec correction: the first draft mistakenly labeled unused 100050-99 G3 hashes as
camera inputs. No rendered/proxy run occurred. The final spec excludes that foreign rendered
slice, relabels the hash artifact as G3-only, and pre-registers an exact fresh-variant sampling
protocol in 100000-04. Since the G3 prerequisite failed, that generation stage is unarmed.
