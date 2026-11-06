# PLAN cuRobo initialization poisoned its own Isaac context

## Event

An exploratory, uncommitted in-process cuRobo v2 world adapter on compute6
raised an illegal CUDA memory access during initialization. Its exception
handler returned legacy fallback, but that Isaac process subsequently emitted
PhysX CUDA errors: catching the Python error did not restore its CUDA context.

This was a standalone dev probe, not a scored episode. The separate paired
CPU comparison continued without infrastructure failures. No other stream's
process was signalled, no system changes or driver reset were attempted.

## Response and evidence

The checked own probe Python PID 246352 was stopped with TERM and then KILL
only after rechecking its command line. The receipt/script and logs stay under
~/dojo-data/plan (gpu-probe.log, gpu-probe-result.json, stop-probe.py).
The exploratory source evolved before the final freeze; the initial kernel
fault's exact cause is unproven. Do not attribute it conclusively to one of
the subsequent schema/warmup changes.

The final design isolates cuRobo in a spawned, owned GPU process. It passes
only numeric belief geometry and robot targets, enforces a request deadline,
and closes the worker on failure/reset/Skills replacement/episode completion.
A fatal worker cannot corrupt the simulator's CUDA context. There are no
global process-name signals or system-level recovery commands.

The next isolated probe exposed the installed MotionPlanner attachment
accessor's missing TrajOptSolver property; the adapter binds managers to the
actual models instead. A diagnostic showed recorded stack joint states valid,
while graph-goal handling refused plans. Excluding mesh-rendering links from
IK targets, using one trajectory seed and trying direct optimization before
graph seeding produced a working probe.

The CUDA-blocking debug probe hit the two-second request deadline; the parent
fell back in about 2.12 seconds and remained usable. The non-debug probe then
returned validated empty-hand and attached-box paths in about 0.59 seconds
each after 9.63 seconds cold startup (gpu-probe5wait.log).

## Prevention and limits

Keep GPU planning isolated. Never treat a caught fatal CUDA exception as
proof that an in-process physics context is usable. Log startup, deadlines,
fallback reasons and owned worker PIDs. Validate real G3 outcomes separately
from API smoke tests. The root cause of the first exploratory fault remains
open; successful later probes establish containment and API operation only.
