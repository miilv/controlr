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
