# PyRUA-RSI internal yardstick

Measurement glue only. Upstream Apache-2.0 programs stay in the external
installation and never enter controlr policy code. No benchmark submission
and no external language-model calls. Every child gets an environment
allowlist, a private HOME/cache/tmp and low CPU/I/O priority.

All launchers require absolute `--root`, `--state`, `--code`, `--rdhome`,
`--checkpoint`, `--lock` paths plus `--lane heavy1|heavy2` and
`--profile desktop|camera`. There is no hostname or account default.
`--dry-run` resolves commands on the CPU without writing or querying a GPU.
Shell wrappers call `runner.py` (`run`, `batch`, `smoke`, `collect`).

- `setup.sh --root ABS --models ABS [--uv ABS] [--python 3.11]`: pins the
  program bundle, evaluated runtime and SAM3 package separately. Managed
  CPython avoids 3.11.0rc1. Upstream's ModelScope downloader supplies SAM3;
  SHA-256 is checked before publishing `MODELS/sam3.pt`. Never run setup over
  an active installation. Shared RoboDojo installs are read-only; use a private
  farm (`scripts/dojo/rd_farm.sh`) for the policy registration.
- `manifest.py ROOT --tasks TASK... --ids ID...`: salted raw smoke inputs;
  `--verify` checks manifest, layouts, demos, ids and call order.
- `prepare.py COMMON --tasks TASK...`: freezes ten eligible inputs per cell,
  before that cell is scored. Existing prepared checkpoints are hash-checked
  and reused. A freeze lock serializes append operations. Previous cell bytes
  never change. Short cells stop; missing imitation demos never become stubs.
- `run_task.sh COMMON --task TASK --layouts ABS --episodes 5`: one official
  eval-client call. Native reset/scoring owns the episode. Call timeout is
  bounded; only owned, recorded child PIDs are signalled during cleanup.
- `yardstick.sh COMMON --episodes 5`: gap-priority cell order, first five of
  the frozen ten inputs, one desktop worker. `--shards 2 --shard 0|1` splits
  cells between two camera workers sharing state and the box's real locks.
  Existing `.started` without `.done` refuses automatic scored retries.
- After each closed cell, atomic `STATE/per_cell.json` includes success,
  official score and spread, steps, wall time and input hashes. Missing cells
  remain null. The publisher refuses a short scored cell or duplicate input.
  `collect.sh COMMON --episodes 5` rebuilds that incremental view on the CPU.
  The separate `summarize.py` CLI retains `--require-complete` for 10/cell.
- `publish.py --host HOST --remote-state ABS --checkout DOCS_WORKTREE --watch`
  runs on this agent machine, polling only our result file every five minutes.
  It verifies 54 cells, first-five positions and permitted ids, then pushes
  docs-only bar updates to main. Previously published cells cannot change.
  Use a clean dedicated docs worktree; any non-bar branch change refuses a
  direct main push. Record the local publisher PID before detaching it.
- The batch owns one `power_watch.py --state ABS` child (five-minute samples).
  It records its PID and stops it on exit; the logger has a singleton lock.

The `desktop` profile requires a systemd RSI scope limited to MemoryMax<=16G
and CPUQuota<=800%, and accepts only heavy1. Prep and scored calls share its
one explicit flock. Every admission checks and logs total used VRAM<=2048 MiB,
load1<=8 and power<=300 W. Counting all VRAM/load as foreign is deliberately
conservative: no guessed share of a desktop session is subtracted. Rejection
or failed telemetry waits 300 seconds; STOP files prevent new starts.

The `camera` profile requires `--lane-state ABS --stream NAME` and the actual
existing heavy lane lock. It shares lane2's `wave-start.lock`, STOP files,
`lane2.caps`, `running.*` and `side.*` counters, 16,384 MiB used / 16,223 MiB
free VRAM admission, 16 GiB available RAM and a 50 GiB disk floor. The startup lock is released only
after the 150-second window; the heavy lock lasts through cleanup. At most
two running lanes are admitted even if a legacy caps file allows more.
For a box sharing the boot protocol, pass only its agreed `--boot-stamp`;
no other coordination paths are read. See [install recipe](INSTALL.md).

The official client seeds by position in the ten-input call, including dynamic
choices and instruction wording. First five means positions 0–4. After separate authorization, camera workers can pass `--start-position 5`
with `--episodes 5`: the client excludes completed prefix positions without
renumbering the ten-input seed map. Results go to `yardstick-remaining/` and
`per_cell.remaining.json`, preserving the published first pass. Never reindex
a five-file directory as 0–4. The original ten inputs remain frozen for ports.

Operational evidence: `docs/journal/2026-11-07-rsi-repro.md`. Historical smoke
files under experiments/data are not yardstick episodes. A partial five/cell
bar must never be described as the completed 540-episode measurement.
