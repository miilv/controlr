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

Results and validation are appended after the frozen-source comparison.
