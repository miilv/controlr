# EGG2: reachable egg target search

Implementation follow-up to EGG #399 and PLAN #401. No model calls. Defaults stay
off; do not merge without lead review.

PLAN's twelve scene-free diagnostic requests are failed candidate goals, not a
proof of global egg-task unreachability. Eleven fail under its tested search.
EGG2 tests target selection under the X5 practical envelope rather than relaxing
IK bounds or collision checks.

`egg_reach=search` expands round grasp orientations, screens all wall-filtered
candidates before the existing 60-candidate planning cap, and includes axial-spin
drop poses. Grasp/pre/lift and release/hover endpoints must pass native IK under
practical limits: measured j5, |j4| <= 1.6, |j6| <= 3.0, planar reach <= 0.60 m.
The operation intersects existing bounds and restores them on return/exception,
invalidating the IK cache at each boundary. Existing normal collision/path checks
and both-arm order remain. It does not add GT reads, a handover or regrasp.

Compute6 deployment: `~/controlr-egg2`; data: `~/dojo-data/egg2`.
Private copies of EGG's frozen v6 dev10 records: 100000, 100001, 100010, 100012,
100015, 100016, 100019, 100053, 100054, 100055. Initial 100002-04 were excluded
by EGG's settling screen; no tuning uses forbidden ids. Two EGG2 workers maximum,
light locks only. Every start rechecks used VRAM <= 20480 MiB, MemAvailable >=
12288 MiB and load1 <= 26, with a resource receipt. No compute4 or dc19 access.

The scene-free bank reproduces 11/12 nominal failures. Every request has nearby
rotations with native IK inside practical bounds; the eleven failed nominal local
solves reach j4's bound. Raw initial-layout round-grasp endpoints are feasible for
all 40 eggs with some arm, but the actual wall-aware candidate set covers 24/40.
These CPU diagnostics do not certify collision-free paths or physical picks.

Validation: 15 focused tests passed. Full local suite: 1909 passed, 191 skipped,
one unrelated watchdog timing failure; that test passed alone on retry (12.6 s).
Implementation-head CI passed on `0453168b` (run 42298650994). Compilation,
new-file pyflakes and shell syntax passed. Existing `skills.py` has a pre-existing
unused `MoveResult` import; no unrelated cleanup. A second local pytest invocation
was stopped by its checked owned PIDs to avoid redundant concurrent work.

Paired comparison results follow in the experiment report. Initial G3 code stamps
precede deployment-stamp creation; controller-file hashes were checked against
`8be287c3` and retained. The diagnostic runner refactor in `0453168b` preserves
counting behavior; no controller changes were made during measurements.


Final ten-pair result: **23.5 → 17.5**, sample SD **4.74 → 7.91**, successes
**0/10 → 0/10**, score wins/losses **0/4**. Native IK failure fraction falls
54.7% → 29.7%, but queries triple and failed query count increases; planning
258.3 → 469.2 s. This is a regression, not a ceiling gain. Keep off, no camera
pre-registration, no merge. Full paired rows, query denominators and limitations:
[experiment report](../experiments/2026-11-06-egg2-reach-g3.md).

All twenty admissions pass: max VRAM 16,858 MiB, min available RAM 41,798 MiB,
max load1 17.08. Two EGG2 workers maximum; no worker remains after the panel.
No foreign processes signaled or system/vendor changes. Local footprint remains
under 200 MB. Raw artifacts stay on compute6 in the EGG2 data root.
