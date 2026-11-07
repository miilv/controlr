# Laptop G3 round 2e — independent failure diagnosis

The frozen centre-stage panel has random B at 1/5 completed credits, so its
maximum is 6/10 and it cannot meet the 7/10-per-cell stopping rule. Finish and
archive that panel without changing source or inspecting individual panel
failures for development. Standard B is 5/8 at this decision point.

Generate a new independent pilot bank: ids **100045–49**, v15 variant 17,
salt `laptop-round2e-pilots-2026-11-07-88c48b74`, both cells. These ids have not
been generated/run by this stream. Settle once and retain all stable records.
Run frozen source **7faef88e**, seat/staged/dense/mesh/center bundle, on all five
pilots per cell before choosing another candidate. The seven earlier pilots
per cell all passed; they do not cover the current fresh-panel failure rate.

Use measured traces to distinguish staging refusal, physical pickup failure,
loose hold, carry contact and final support. Hypotheses, not selected fixes:
(1) narrow cup proposals should be a fallback when a reliable wide hold fits,
rather than displacing it after every staging move; (2) a failed physical
pinch should exclude a site rather than the entire antipodal method; (3) a
checked staging pose should be chosen for a demonstrated re-grasp geometry,
not solely table free-space. No relaxed physical verifier, joint-limit or
collision exemptions. Any change must have pilot evidence and unit tests.

No new panel is dispatched by this registration. A later candidate/source
and new salt must be registered before a new panel; ids **100000–14** are still
unused by this stream and reserved for that purpose (first ten stable,
reserve only for stability/infra, not policy scores). Keep all prior panels
separate, no selected pooling. Two concurrent physics-only LAPTOP jobs maximum
on light2/light8, unchanged who/GPU/disk admission and 8G/600% scopes. Zero
LLM calls/tokens/spend. No camera dispatch or board change.

## Coverage finding before candidate selection

The earlier seven random development records all contain headset asset 00008.
This new bank contains 00008 (45/46), 00010 (47/48) and 00009 (49). Standard
contains 00004 (45/46/49), 00005 (47), 00002 (48). This is a development coverage
gap, not evidence that any particular unopened panel failure has that cause.
Public metadata: 8/9/10 all mass 0.22 kg; friction 0.32/0.45/0.45; depth
72.28/80.64/78.30 mm; bridge local z 85/80/90 mm. Do not infer grip-force limits
from mass/friction alone. Complete the baseline and use actual per-shape lift,
carry and support measurements to select the next mechanism.

## Selected pilot prototype before new actions

Frozen baseline on all ten new pilots: **standard4/5, random2/5**. The failures
are standard48 (asset2), random47/48 (asset10), random49 (asset9).
Standard48: right26 mm cup pinch has patch1.0/depth and quality0.17 yet is
loose14.9 mm; subsequent left39 mm picks rise3/80 mm. Random47 stages with a
35 mm right pinch (4.9 mm lift drift), then left35 mm rises39/80 mm with40 mm
drift; the existing whole-method rejection discards every untried antipodal
site. A no-action-change repeat's stage lands17.6 mm off and remains a
place_miss under the original15 mm verifier. Random48 has only16 wide grasps,
one endpoint pair and no full approach. Random49's80 mm pinch drifts37 mm;
no alternative antipodal site is tried. These are pilot observations only.

Prototype two default-off switches: `headset_grasp=cups` adds the existing
geometry-verified18–35 mm broad/deep cup proposals to the wide bank even when
no stage succeeded; all IK/collision and physical verification remain. It
must not mark a failed stage as successful or infer a hold. `headset_retry=sites`
remembers failed **object-frame site + approach + closing axis per arm**, not
an entire method, and allows up to four physical staging attempts and four
hang attempts, each checked anew from the measured state. Candidate sites
within15 mm, closing axes within20 deg (finger flips equivalent) and approaches
within30 deg of a failed site are excluded. Existing800-step/reserve bounds
remain; no retry after budget/end or a placement miss, and no loosened hold,
place or table tolerance. Legacy modes keep their exact prior budgets/filters.

First screen the four failed pilot records, then all new pilots plus the
original seven/cell if supported. Any final bundle/source and fresh panel
must be registered separately before generation. No camera or model calls.

## Full-opening enclosure probe before new actions

Source d8387f61 first probes remain zero: standard48 repeats26 mm loose grips
then a bad third site knocks the object off the table; random47 tries18/19 mm
cup sites, all loose/slipped. Do not interpret more attempts as a solution.
The original standard48 diagnostic shows an earlier mechanism: while g=0.629
(~47.9 mm commanded open gap), descent moves the object from z0.8056 to0.8215
before closure. TCP target error is only0.03 mm/0.004 deg; this is approach
contact, not failed IK. During closure the relative root moves another11 mm.
The nominal26 mm visual-mesh slice is insufficient opening clearance for this
cup's physical envelope (asset depth88.78 mm; exact collider section not measured).

Probe **full88 mm opening** for narrow cup proposals, via existing Grasp.clear_w.
Keep the same TCP, closure, native finger/table/scene validators and original
physical verifier. A too-low/wide pose must be refused, never exempted. Test
standard48 and random47 first; screen the other failures only if supported.
This is an enclosure-clearance test, not a larger hold/slip tolerance. Source
remains d8387f61 plus a retained probe recipe until production proof.

## Enclosure and actual-support rest extension

The full-opening standard48 probe changes right26 mm pickup drift14.9→6.6 mm
(clean), followed by1.1 mm staging carry shift. Left26 mm remains loose8.4 mm;
a subsequent41 mm grasp opened by its nominal width is loose23.9 mm. Extend
full88 mm opening to the complete cup-mode bank, retaining all native checks.

Also probe an actual-mesh support target for the staging leg only. The current
bbox rest leaves sparse-headset mesh15–23 mm over the table; random47's release
falls15.5 mm and tilts11.5 deg (another repeat misses xy by17.6 mm). Set the
desired root z from the minimum of all transformed mesh vertices plus3 mm
clearance, rather than empty bbox corners; retain the original release gap,
full-mesh table tolerance and hand/scene validators. Missing/invalid geometry
retains the old target. This is a planned pose, never a forced scene pose.
Compare full-bank opening alone on standard48 and full-bank opening + mesh rest
on random47; then cross-check any physical improvement. No verifier relaxation.
