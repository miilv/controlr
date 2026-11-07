# SAMINT binding general-system scope and complete offline result

The lead superseded the per-task camera authorization with the board's one-system
rule. The five old specs are cancelled for board dispatch; the index, design JSONs
and spec headers identify them as diagnostic history. No SAMINT queue append occurred.
Existing prepared inputs/failures and CPU-only deployment/checkpoint staging remain
retained. The original egg salt failed 13/20 native settling checks; its bounded
lead-authorized refill remains prospective and was not executed.

The task-free API and evaluator are merged in [#428](https://github.com/miilv/controlr/pull/428),
merge `4d86ecb149a3a4883e0d49851581077b25ff0637`, with [green CI](https://github.com/miilv/controlr/actions/runs/42401558678).
Local full minimal suite: 2,291 passed, 206 skipped; four subsequent summary tests passed.
Defaults and existing episode behaviour remain unchanged. DEVELOPMENT prohibits the
historical task route/banks for board candidates and documents the new API's lack of
any task/registry/config/checker selector. The legacy fitter's task-prior dependency
remains an explicit architectural boundary; this component is not a board activation.

The [offline protocol](../experiments/2026-11-07-samint-general-offline-prereg.md)
was frozen on main at `ea08171a` before inference. Compute3 ran component `e695260c`
and evaluator `5324a662` unchanged through all 180 SAMBAKE views. It used one free
light lock, checked who/GPU admission first, capped its allocator to 30%, and ran in
10G RAM / 600% CPU user scope. One SAM3 checkpoint plus v2b; no rendering or LLM calls.
The owned run exited normally with status 0; its lock and CUDA process were released.
PID 3613 and other users' processes were untouched.

Complete [report](../experiments/2026-11-07-samint-general.md) and
[result receipt](../experiments/data/samint-general/result-receipt.json):
540 arm/view records, 60 recordings, 54 input-layout clusters, all 11 recorded cells.
All image/layout hashes and recovered instruction provenance checked. Catalogue snapshot
SHA equals the run metadata; the entire rerun v2b reference agrees with SAMBAKE.
Recall 68.5→93.5%, mean IoU 0.638→0.868, but phantoms 11→2378. Every one of 42
camera/phase groups fails the frozen phantom rule. No recall/IoU practical-harm
bound was crossed; pen IoU nevertheless falls slightly and has no recall gain.
Torch peak allocated/reserved 5.57/8.54 GB; joint proposer p50/p90 27.53/52.56s;
1.73 compute3 hours, zero calls/tokens/spend. Timings reflect changing shared load.

**No camera preregistration or manifest was created, and no row was queued.** The
final compute4 [locked no-dispatch receipt](../experiments/data/samint-general/general-no-dispatch-receipt.json)
at 19:29:10Z found 50 pre-existing rows, rather than the earlier empty queue; SAMINT did not
append, edit or reorder them. Queue SHA256
`cd17114cdbe8d4bb33111d5ca08f947ef32be48a55ddb8250c8e00d97a0989ac`.
STOP remained inode 49320090, unchanged mtime. No compute4 GPU process was started.
No coordinate gate paths or partner processes were accessed. The CPU-staged shared
SAM server was not started or health-loaded on the GPU. The cancelled directory
contains `CANCELLED-PER-TASK-DISPATCH.txt`; old task-wave recipes must not be used.
No compute6/7 work occurred. No experiment was mixed into IMIT2, POUR or MATCH2.
