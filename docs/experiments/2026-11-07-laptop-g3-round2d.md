# Laptop G3 round 2d — pilot-only wide-headset grasp diagnosis

The mesh-table panel's random B reaches only 4/8 completed credits (maximum
6/10), so it cannot satisfy the 7/10-per-cell rule. Finish/archive it without
changing source or using individual panel failures for tuning.

Return only to independent pilot random **100066**, variant-14 pilot bank.
It is headset asset **00008**, authored mass 0.22 kg / friction 0.32. Two 71 mm
pinches rise zero while the arm tracks its lift; a wider plan search and the
mesh table guard cannot create a hold. First measure actual finger joints
through close/lift, separate from policy inputs, to distinguish an empty gap
from contact/friction failure. This diagnostic introduces no action changes.

Then screen geometry-verified grasps on the public headband feature (centre
and offsets along the band), both thin/radial and depth closing axes, with
normal low pre-place height and the conservative mesh table guard. Earlier
bridge probes coupled the new grasp with a high pre-place; they do not rule
out a low, side approach. Retain all finger/arm/obstacle checks. No ignored
stand-hand collisions, smaller table tolerance, welded attachments, or forced
poses. Use physical lift/attachment and genuine support after release; do not
count a floor object merely near the cradle as a working hang.

If a candidate works, screen original and new independent pilots before a new
registration. Never tune on panel rows. The final source/bundle and fresh draw
recipe must be committed before any new panel. All box/path/resource/zero-LLM
rules continue, maximum two physics-only jobs on light2/light8.

## Band endpoint diagnosis and shallow insertion amendment

The actual jaws remain ~72 mm apart under close, supporting curved-surface
contact rather than an empty free-close. In the band bank, 64 generated grasps
produce zero accepted endpoints for either arm. Horizontal approaches put the
palm at z=0.718–0.728 m, below the 0.765 m table. The thin-axis downward bridge
site is z=0.7792 m (14.2 mm above table), not the hypothesised 64 mm clearance;
reachable offset sites fail fingertip clearance at z=0.765–0.767 m. Preserve
these refusals. Before any further episode, add shallow insertion proposals
at +6/+12 mm world z, recompute mesh opposition at each new TCP, and retain
native IK/finger/obstacle checks and original physical hold rejection. This
changes pad insertion depth, not table tolerance or collision exemptions.

Shallower insertion yields 24/124 reachable right-arm grasps but zero paired
hang endpoints. Next separate pickup feasibility from placement: test a
standalone checked band pick, and count exact place-endpoint IK/wrist/scene
rejections. This is pilot diagnosis, not a reason to disable cradle collision.

The standalone 12.9 mm centre-band pick rises 44/80 mm and is rejected for
33 mm root-relative drift. Instrument one repeat plus 30 stationary closed-
jaw steps: compare the original pinch site and object root in TCP coordinates
to distinguish rotation at a maintained pinch from continued sliding/falling.
Do not override verification or execute a rejected carry in this diagnostic.

## Individual physical grasp screen

The stationary thin-band hold falls: original pinch-site error grows ~10→82 mm,
and both jaws close to zero by step 51; the object returns to its table z.
The 24 reachable band hang endpoints split into 20 IK failures and four stand
collisions. Before a new mechanism, screen up to eight distinct checked pick
sites per arm on the same independent pilot, resetting the settled layout for
each. Candidates: existing box/slice bank, 800-face/13-approach antipodal bank
without the 35 mm cutoff, and mesh-verified shallow band sites. Rank by existing
contact/lever quality; require separation in site or closing axis to avoid
spending all probes on duplicates. Record native refusals, rise, attachment
and pinch-site drift, measured jaws and support clearance after 30 stationary
closed-jaw steps. No verifier override, release claim or panel credit.

## Cup fallback prototype registration

The independent right-arm screen finds two 25 mm ear-cup grasps suspended
through 30 stationary steps. Cleaner pick: rise ~76 mm, 6.2 mm root drift;
measured jaw gap ~37 mm (collider/contact width differs from mesh slice).
Their object-local centres are near (+/-46, -3, -60) mm, ~145 mm below the
bridge; the bridge-centre 13 mm grasp still falls. These cup grasps were
excluded by the 35 mm minimum. Test a targeted fallback: retain the existing
wide bank; add 18–35 mm antipodal proposals (800 faces/13 approaches) whose
centre is >=60 mm below the active bridge along its local up axis, with a
nonzero pad contact patch >=0.12 and contact depth >=6 mm. Native endpoint,
full-motion and original physical verification remain; no asset/id routing or
loose-hold exemption. First run the full hang on random pilot 100066, then
screen all independent pilots only if a physical hang works. The full-plan
budget remains 24 and source remains frozen during any panel.

## Centre rest/re-grasp prototype

Full cup hang has 364 right pickup endpoints but no right cradle pair; left
has 188 pickup endpoints/60 cradle pairs but tested lifts slip. All measured
pickup TCPs reach their target within 0.5 mm /0.1 degrees, excluding a wrist
stop as the cause in the per-arm screen. The direct prototype's high left
21 mm pinch rises only 23/186 mm; no credit. Do not relax verification.

Test a centre rest/re-grasp on this pilot: right cup pick → clear table space
near (0,-0.15) m with the original lying orientation → ordinary hang re-plan
from the measured rest. Use native free-space selection and checked transfers;
no forced pose or coupled carry. Restrict the new narrow cup proposals to
patch >=0.8: both suspended 25 mm contacts have patch 1.0; the fallen 27 mm
contact has patch 0.4. Preserve the wide bank and unchanged physical verifier.
This adds a staging leg only; all collision checks, limits and budgets remain.

The first staging wrapper did not execute: the skill registry retained the
original function. Its 0-credit row is a stricter-contact direct-hang probe,
not evidence for or against staging. Correct the registry binding and require
a `CENTRE_RESULT`/explicit no-spot record before interpreting the staged run.

## Upright centre-rest feasibility variant

In parallel with the corrected lying-rest probe, test the same checked right
cup pick to an **upright** centre rest. This changes reach geometry and makes
a later front/back bridge pinch possible above the table, with no large wrist
rotation during final carry. Require the existing `rests_stably` geometry
screen. Sample a fixed centre table grid near (0,-0.22), in addition to legacy
free spots, and let native full held/hand/scene collision checks validate each
actual pose; circle-footprint free spots alone can conservatively omit spaces
in front of an open laptop. Include fixtures as obstacles. No object/fixture
pose is written or forced. Record selected xy, all refusals, physical rest
stability and original pick/transfer verdict. No new panel is authorized yet.

## Production-candidate registration before pilot screen

The corrected lying-rest prototype earns the genuine 20-point tier on random
100066 in 412 steps. Right pick is clean; placement near (0.05,-0.33) has
2.2 mm xy error and 0.3 mm carry shift. Left rejects an 8.4 mm loose pick,
re-grasps, then finishes with 0.8 mm carry shift and releases on the cradle.
The independent upright-rest prototype is also physically stable (0.6 mm xy,
-0.5 mm z error, 1.6 deg tilt, upright through 30 steps) but is **not included**:
there is a complete lying-rest hang and no need for another mechanism.

Implement default-off `headset_stage=center`: stage only when the nearest
pickup arm differs from the nearest cradle arm and xy transfer exceeds 0.40 m.
Retain the successful lying orientation, native free spots (first eight),
checked plans (24 full max) and one actual transfer attempt, with original
verification. After a successful stage only, add the >=0.8-patch, >=6 mm-depth
18–35 mm cup bank alongside existing wide grasps for hanging; otherwise keep
the existing dense bank. Retain method-slip filtering, retry limits, slow carry
and seating. No asset/id dispatch. Screen all seven independent pilots/cell
(original 30/31 plus 65–69), with the existing mesh+staged+seat+dense profile.
Only after that screen freeze source/bundle and register new unused panel ids
100015–29, a new salt/variant, first ten stable ascending; no individual old
panel failure may be used for development. No panel dispatch under this section.

## Frozen fresh paired panel registration

Production action source **7faef88e** completes **7/7 standard and 7/7 random**
independent pilots, versus 7/7 and 6/7 with the preceding mesh-table bundle.
No pilot is a panel record. The hard random-100066 pilot is rescued in 362
steps; no verifier was relaxed. Source includes current main's unrelated
switches at their unchanged defaults. Focused unit suite: 27 passed; full
local suite is running. No action-source changes during generation/panel.

Generate v15 **variant 16**, salt
`laptop-round2d-centerstage-2026-11-07-88c48b74`, ids **100015–29** in both cells.
Preflight found no existing records at these ids in this stream. Settle once,
select first ten stable ascending before outcomes (expected 100015–24), hash
exact settled bytes, and share those records between A/B. Reserve 25–29 only
for prespecified stability/infrastructure replacement, never policy scores.
Never use protected/sealed ids >=100100 or rendered jobs.

A: `hang_cradle=straddle`, `headset_seat=off`, `headset_carry=legacy`,
`headset_grasp=legacy`, `headset_table=bbox`, `headset_stage=off`.
B: same straddle plus `headset_seat=on`, `headset_carry=staged`,
`headset_grasp=dense`, `headset_table=mesh`, `headset_stage=center`.
All other flags/source/step limits identical; no laptop route. Alternate which
arm runs first by id parity. Ten A/B pairs per cell; report actual n,
mean/sample SD on 0–100 points, headset credits, full successes, W/L/T,
success flips, steps and every exclusion. No selected pooling with prior panels.

Stopping criterion: **B headset credit >=7/10 in EACH cell**. Paired gain gate
for the prospective camera spec: pooled mean gain >=5 points, wins >=2*losses
and >=2 net wins. This remains a G3 finding, with camera qualification separate.
Two concurrent physics-only LAPTOP scopes max on light2/light8, who/GPU checks
and <=24 GiB used / >=10 GB free admission before each job, 8G /600% user
scopes and bounded timeout. Zero LLM calls/tokens/spend; no camera dispatch or
board candidate changes.
