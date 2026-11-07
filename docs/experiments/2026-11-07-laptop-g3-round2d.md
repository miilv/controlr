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
