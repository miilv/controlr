# Compute6 tie launch: repeated rendered-driver watchdog failures

Round-3 update: a dev-only investigation completed one episode without an
infrastructure failure and resumed dispatch at 13:22:31Z. The
[recovery journal](../journal/2026-11-06-integ6-render-recovery.md) records the
offline extension overlay, active perception-fitting stacks, bounded watchdog
configuration and disclosed extra A/B infrastructure attempts. The timeline
below records the initial failure and hold; its stopped state was superseded
by that release. No comparative outcome is established here.

The camera deployment and all requested data parity gates passed before launch.
Both first tie arms started on compute6, with the accepted simulation Python
3.11.18 / msgpack 1.2.4 differences recorded. The first A attempt had reached
750 environment steps before the 108-row hand-over from compute4. Both first
attempts subsequently stalled and were classified as infrastructure failures.

Observed attempt history (UTC, launcher occupancy):

- A started 11:34:38, ended rc=4 at 11:47:38 after 781 seconds; last observed
  progress 922. Its one identical-input retry started immediately.
- B started 11:42:28, ended rc=4 at 11:55:14 after 766 seconds; last observed
  progress 963. Its one identical-input retry started 11:56:14.
- A retry ended rc=4 at 12:00:24 after 766 seconds, again after observed step
  922. The lane exhausted its permitted retry and retained both attempts.
- B retry ended rc=4 at 12:12:30 after 976 seconds, after observed step 1084.
  Both arms exhausted their one allowed retry: four infrastructure attempts,
  zero valid comparison rows.

At 12:00:45, new dispatch was stopped with `~/dojo-integ-c6-state/STOP`.
The 106 unstarted commands remain queued. All 54 pairs remain assigned to
compute6; nothing was silently restored to compute4. Its archive and
`lead-notes.log` record the original hand-over. No valid comparative result is
claimed from infrastructure zeros, and failed-attempt model spend remains in
the agentic transcripts even though synthetic watchdog rows report zero calls.

The earlier [watchdog audit](../journal/2026-11-05-dojo-imit-watchdog-audit.md)
describes the same broad rendered-driver/no-progress symptom on compute4.
Zenity warnings occurred, but neither those warnings nor these step counts
identify the blocked native call or establish a root cause. No evidence
attributes this to the accepted Python/msgpack patch differences. No CUDA
device error or HTTP 429 was observed; sampled power remained below 700 W.

No controller, perception, renderer, watchdog or timeout setting was changed,
no warning was suppressed, and no system/driver modification was attempted.
No scoring frames or action diagnosis was used to repair or tune the booking.
The wave's one infrastructure retry per arm was respected.

Operational evidence is under `compute6:~/dojo-integ-c6-state` and
`~/dojo-data/integ-c6/runs2`; exact deployment, hashes, provenance and hand-over
are in [the launch journal](../journal/2026-11-06-integ6-tie-preflight.md).
Completion ETA is withdrawn while dispatch is stopped. Further causal work
should use a separately released dev-only rendered reproducer with bounded
Python/native stack capture, rather than further measurement reruns or a
speculative timeout/display change.

Cleanup finished at 12:14:34: both lane managers had exited, all three owned
shared services were stopped using their recorded PID/unit ownership checks,
and the owned power logger was stopped after checking its command line.
Subsequent inspection found no PIDs in any owned job scope; failed scope
metadata remains available for audit. Other streams' processes and light locks
were untouched. The 106-row queue and STOP remain intact.

Recorded spend across all four attempts: 14 completed logged camera calls,
222,060 prompt / 15,993 completion tokens (11,318 reasoning, 167,992 cache reads).
The eight five-minute power samples ranged from 49.03 to 168.75 W, with no
HTTP 429 or power-threshold event. The final log sample was 12:09:37; the
post-cleanup GPU query at 12:14:34 was 15.61 W / 4,348 MiB, belonging to the
box's remaining workload rather than a claim that the entire GPU was idle.
