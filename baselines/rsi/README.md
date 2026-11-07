# PyRUA-RSI internal yardstick

Measurement glue only. Upstream programs stay in the external Apache-2.0
checkout; they are never imported into our policy. No benchmark submission.

On compute7, the install lives in `~/controlr-rsi-repro`; all data lives in
`~/dojo-data/rsi-repro`. `setup.sh` pins the program bundle, evaluated runtime
and SAM3 package separately, uses managed CPython and upstream's ModelScope
download script, then checks the checkpoint SHA-256. It installs XPolicyLab's
dependencies without writing into the shared installation. Create a private
RoboDojo farm with `scripts/dojo/rd_farm.sh`, then register the upstream
`PyRUA_RSI` policy only in that farm.

- `manifest.py ROOT --tasks TASK... --ids ID...`: fresh salted raw smoke
  inputs. `--verify` checks the manifest and each layout/demo hash.
- `power_watch.py`: foreground five-minute logger. Record its PID before
  detaching; stop only that owned PID at the end.
- `smoke.sh`: waits for the verified checkpoint, then two episodes of three
  tasks through the unchanged official eval client on our inputs.
- `prepare.py`: first ten eligible inputs per cell from our twenty permitted
  ids, preflighted before scoring. Short cells fail explicitly.
- `yardstick.sh`: requires the documented Phase 1 decision marker, prepares
  and freezes inputs, then runs ten episodes of all 54 cells serially.
  An unfinished scored call blocks automatic resume.
- `run_task.sh TASK LAYOUT_ROOT N [smoke|yardstick]`: credential-free runtime,
  one owned GPU slot, power admission gate, official-client scores, pid files
  and cleanup. There is no robot-harness dependency or sealed-data access.
- `collect.sh`: companion to a running batch; verifies the frozen inputs and
  atomically writes `per_cell.json` only after all 540 scored rows are present.
- `summarize.py LAYOUT_ROOT RUN...`: post-run official verdicts, per-episode
  input hash, success/score/steps/wall and per-cell spread. Missing random
  cells remain missing. Average uses `dojo.scoreboard.board.cells` weights.

The exact call order is part of the frozen input: the official client seeds
by layout position, including dynamic task choices and instruction wording.
Use the frozen records, demos and order for later ports. An external-LLM
request count remains zero; the upstream runtime's local HTTP requests are
fixed-program dispatch, not language-model calls.

Operational evidence and the Phase 1 decision belong in
`docs/journal/2026-11-07-rsi-repro.md`; final results belong in the experiment
report and its per-cell JSON. No missing cell may be claimed as a completed
540-episode yardstick.
