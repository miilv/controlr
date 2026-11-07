# General SAM catalogue/instruction union: offline registration

Binding scope: one perception configuration for every episode. Historical task-keyed
SAM switches and phrase banks are diagnostic references only and cannot be adopted.
No per-task camera panel is queued. Compute4's intentional STOP remains the lead's.

Freeze [offline-design.json](data/samint-general/offline-design.json) before inference.
The task-free component is `e695260c0367f65a2aa83e6f5881d13e6988a9f3`.
SAM3 / bf16, API threshold 0.2, v2b confidence 0.25, base-first mask-IoU dedup >0.5,
and no area/region/competitive-category task filters. One checkpoint/configuration
is used for every image, including guard/risk cells. No tuning by cell or GT.

Prompts contain every category label and asset caption name from the one shared
catalogue index, plus generic grammatical/catalogue-noun phrases from the actual
logged instruction. All **60/60 instructions** were recovered from the first agentic
call transcript (literal string parsing; no calls, no reconstructed task templates).
The [instruction manifest](data/samint-general/instructions.json) freezes transcript
paths/hashes and instruction text. The predictor accepts only RGB, instruction and
the global catalogue. It reads no task key, task configs, checker values, role labels,
placement priors, per-task banks or GT. Catalogue geometry/placement fields are ignored.
The general API is separate from the legacy task-prior fitter: general masks alone
do not establish compliance of that downstream fitter.

Use exactly SAMBAKE's **60 recordings / 180 views / 11 tasks**: 54 frame-zero layouts
plus six pre-grasp survey observations. All cameras, all recorded cells, including
v2b-favoured coin/pens/bottles/organize/classify. Same permitted IDs, image/layout hashes,
precomputed GT silhouettes, >=30px visibility, greedy one-to-one matching and phantom
rule. Tiny-object observations are supplementary. The belt GT remains SAMBAKE's
approximate phase reconstruction; survey GT assumes survey motion did not move objects.
GT/task metadata enter **scoring only**, after proposals. Per-task-bank scores are a
historical reference, never a predictor input. Confirm the rerun v2b reference against
the published table rather than silently assuming equality.

Predeclared camera qualification: general union improves both micro object-view recall
and macro task recall by **>=5 percentage points**. Every task/camera/phase group must
avoid material harm: recall drop >=5pp, mean IoU drop >=0.02, or phantom increase >=0.5
per image with >=2 added phantoms. Report all risks even below those bounds. Cluster
uncertainty by layout; 10,000 paired bootstrap resamples, seed 0, descriptive 95% CIs.
Small n or a wide CI does not waive a practical-harm failure. These conservative
component thresholds define the lead's "helps overall / hurts nowhere significant"
condition; they are not claims of statistical population equivalence.

Only a passing configuration can proceed to a single fresh-salt camera prereg with
same A/B parameters across every cell, strong plus observed-risk guards, owned IDs,
Sonnet 5.5, current refined per-writer audit and #417 persistent identity. End-to-end
absence of task-config/checker inputs must also be demonstrated before dispatch.
If any offline gate fails, publish the table and **queue nothing**. Egg's bounded
fresh-salt/raw-check amendment remains available only if egg is a gain cell of that
future general experiment. No further egg refill is run for the cancelled task panel.

Compute3 inference only: check who/GPU first, one free light lock, allocator <=30%
(<10 GB on this card), 10G RAM / 600% CPU scope, no rendering or LLM calls. No process
on compute4/6/7 is started. Existing checkpoints are reused; no model is downloaded.
