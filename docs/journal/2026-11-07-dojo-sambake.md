# SAMBAKE — SAM3 task-prompt comparison

Branch `feat/sam3-task-prompts`. Lead requested a follow-up to the October
catalogue-caption null, using PyRUA-RSI's literal task vocabulary and the adopted
v2b proposer. [Report](../experiments/2026-11-07-sam3-task-prompts.md).

- Read the operating manual, development/architecture/experiment protocols,
  October SAM and v2b reports, and the §8 phrase inventory. No upstream task
  program code was copied; banks contain textual facts plus independent mappings.
- compute3 admission initially 3.4 GiB GPU used, 91 GB disk free. Existing torch
  2.9.1/CUDA reused from the identity environment; SAM source and small dependencies
  isolated under `~/controlr-sambake/vendor`. Initial uv dependency installs used
  uv's default shared download cache; experiment/model/kernel/Ultralytics caches
  were directed into `~/dojo-data/sambake`. No torch/CUDA installation, training,
  simulation, or LLM inference. No other stream's PID was signalled.
- Copied existing SAM code from compute4 read-only. The slow partial SAM3.1 `.pt`
  SSH transfer was stopped; a direct completed safetensors repack download was
  used. SAM3's single ModelScope download matched the lead's SHA-256. Weight hashes
  and the Meta code commit are in the report. No compute4/6/7 GPU work.
- Existing recorded dev captures: 54 frame-zero layouts, all eleven requested
  cells; six later pre-grasp look frames. Head and both wrists. IDs 100025–29
  were excluded to avoid the old v2b training range. A missing standard settled
  record was not grounds to substitute a perception estimate for GT.
- Provenance correction: loaded episode paths can point at salted blocks. The
  collector now follows the episode's exact path and records its hash. Standard
  same-numbered layouts were initially wrong on conveyor cells; those scores
  were discarded. Even the correct raw conveyor layout precedes reset pre-roll.
  The lead supplied MATCH2's authorized 4mm/native-step reconstruction. Anchors
  imply 49.37–53.06 steps; only untouched on-belt frame-zero riders are eligible.
- CPU Mesa device 0 segfaulted; device 1 worked. The first SAM3.1 architecture
  attempt correctly rejected incompatible standard-image keys; the final loader
  builds the multiplex detector. A camera/car substring mapping mistake was
  caught and fixed, covered by a regression test. The affected category results
  were rerun. A projection/inference race interrupted a SAM3.1 pass; its affected
  and remaining cells were rerun after all phase projections completed.
- Inference used sequential model processes in 8 GiB / 600% systemd scopes,
  allocator capped at 30% of the 32 GiB card. GT, exact input provenance, raw
  per-instance rows, hashes and timing/VRAM metadata remain in
  `compute3:~/dojo-data/sambake`; no model weights/frames/runs are committed.
- Main interpretation: SAM is a strong mask fallback for vocabulary coverage
  holes, not an asset-ID replacement. The frozen v2b classes contain **zero**
  egg/egg-holder/egg-basket and wine-bottle/wine-bowl keys; this explains much of
  the strongest gain. Labelled prompt masks still need candidate fitting before
  runtime belief writes. More masks alone can add many organize-table phantoms.
- Validation: full local suite 2,138 passed / 191 skipped; six new CPU-only tests
  pass independently; pyflakes, compile and whitespace checks. No runtime defaults
  or candidate configuration changed. Default-off integration is a written spec.
