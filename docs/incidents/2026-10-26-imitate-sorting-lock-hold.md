# 2026-10-26: one breadth episode held the GPU lane for 48 min; stopping it left the lock held

**What happened.** `B_imitate_sorting_sequence_breadth` (`~/dojo-integ4`, mode B, submit_v1 fields) started at 11:22
UTC. Its first episode ran at about 6 s per env step (perception service at ~140 % CPU), and was at step 496 of 1600
at 12:10. The team rule is ≤ 30 min per lock hold. The job budget (`--max-minutes 25`) only stops NEW layouts, so
nothing ended the episode.

At 12:10 integration-2 sent TERM to the job's `integ_eval.sh`.
- The script exited and the queue runner started the next job.
- The scope's processes stayed alive: the render, twin and perception services and the eval driver.
- Those processes inherited fd 9, the `~/dojo-gpu.lock` flock, so the next job sat waiting for the lock.

The four processes (ours) were killed by PID. The next job got the lock at 12:11.

**Fixes** (PR after this note):
- `dojo.runtime.eval --episode-max-minutes` (default 25; 0 = no cap). It ends an episode at the cap, and the row says
  `time_capped` / category `time_cap`.
- `integ_eval.sh` runs its systemd scope in the background under a named unit and traps TERM / INT to
  `systemctl --user stop <unit>.scope`.
  - Tested on compute3 with a stand-in scope: the scope stops and the lock is free.
  - The unit needs the `.scope` suffix: a bare name means `.service` and stops nothing.

**Not yet explained:** why imitate_sorting costs ~6 s per step in mode B. Watch reads during the demonstrator's
turns are the suspect. Breadth2 (`~/dojo-integ7`) runs the task again; it gets the cap only after a redeploy.
