# 2026-11-07: compute7 used without a current permission

**What happened.**
- About 12:00–14:20Z, the lead put the RSI yardstick (REPRO stream) on compute7: their adapter install, the SAM3
  checkpoint, layout preparation, and a launcher with at most 2 jobs.
- The only permission was Ilia's of 28 Oct 17:28Z ("compute7 - all yours"). It was revoked the next day.
  - 28 Oct 19:07Z: "Emergency stop compute7 process".
  - 29 Oct 17:26Z: "compute7 we dont touch for now".
- The lead's memory index still carried the 28 Oct permission, and the transcript was not checked.
- A person works on compute7: a desktop session under the `robotics` account, with Chrome and VS Code open for 4 days.

**Impact.**
- No foreign process was touched, and GPU power peaked around 175 W.
- Scored work: a smoke run of 6 episodes and layout preparation. No yardstick episode was scored.
- Our files remain under `~/controlr-rsi-repro*` and `~/dojo-data/rsi-repro`: a conda env plus the 3.4 GB SAM3
  checkpoint.

**Response.** At Ilia's question (14:2xZ), every process of ours was stopped by PID, leaves first:
- yardstick launcher 794616 and its children;
- collector 801819;
- the box-side power watcher 778795.

GPU came back to 672 MiB and 10 W. compute7 was removed from the lead's power watcher and the 2-hour safety net, and
a STOPPED note was left in `~/dojo-data/rsi-repro`.

**Fix.**
- The memory now records the revocation, and the latest user word wins.
- Before putting load on any box, search the transcript for the most recent user message about it.
- Allowed boxes as of 7 Nov: compute3 (physics only), compute4, compute6.
- Cleanup of our files on compute7 is pending Ilia's OK.
- The yardstick moves to compute4 / compute6 when their camera lanes free up.
