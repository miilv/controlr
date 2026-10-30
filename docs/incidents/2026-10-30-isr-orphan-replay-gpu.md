# 2026-10-30: an orphaned Isaac replay held ~7.5 GB of isr-lab's shared GPU for about an hour

**Symptom:** stream S ran a diagnosis replay of a dev episode on isr-lab (`scripts/dojo/replay_kits.py run`, pour_balls
dev 100000), wrapped in `isr_job.sh` with a 30-min limit. The replay crashed after one env step. Isaac then hung on
shutdown, spawning a zenity crash dialog every 2 min with no display. isr_job's limit killed the wrapper and released
the coord GPU lock. The orphaned python, though, kept ~7.5 GB of GPU memory from ~07:22Z until S killed it by PID at
08:23Z. Mikhail's jobs ran beside it. isr-lab is shared with Mikhail's side.

**Root cause:**
1. The command line was wrong: `replay_kits.py run` without `--task` defaults to stack_blocks. The pour_balls layout
   loaded into a stack_blocks env, and RoboDojo's `is_stacked` reward check raised
   (`TypeError: 'NoneType' object is not subscriptable`).
2. Isaac's app did not exit after the exception: it waited on a GUI crash dialog (zenity, "cannot open display")
   forever.
3. `isr_job.sh`'s timeout killed only its direct child (`bash run.sh`). The python that `rdpy.sh` started survived as
   an orphan, outside the lock's lifetime.

**How it was fixed:** S killed the orphan by PID after checking its command line. The GPU went 7.5 → 1.4 GB, and the
lead told Mikhail. The re-run script (`bridge/pbreplay/run2.sh`) starts each replay in its own process group
(`setsid`), waits with its own deadline, and kills the whole group (`kill -- -PGID`, TERM then KILL), so a hung
Isaac cannot outlive the job.

**Lessons:**
- Any Isaac launch on a shared box runs in its own process group, killed whole at its deadline. A wrapper's
  timeout does not reach a grandchild.
- After any job ends on a shared GPU, check `nvidia-smi` for our leftover memory.
- Check a diagnosis tool's defaults (`--task`) before a GPU launch, not after.
