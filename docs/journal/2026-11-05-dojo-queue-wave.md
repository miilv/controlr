# QUEUE — prepare the post-D-C19 camera wave

Prepared [the prospective wave](../experiments/2026-11-06-dojo-ab-wave.md) and exact Integration queue lines at pinned main df09500e723d21674a8c7825fc2da6deab8d79b8. The captured state CANDIDATE is tracked beside the queue, with its hash and explicit baseline differences from earlier lever registrations. No runtime or range-refusal code changed.

Lead clarification: new levers may use their first A/B measurement panel, for scoring only. Fresh re-tests use unique salted 100000–04 blocks; neither the 100030–49 training range nor Misha's 100050–99 range becomes eligible. Existing loader supports these records. All salted records must acquire individual raw/loaded md5 and validated settles/demos before dispatch; none was generated remotely during preparation.

The replacement booking is 316 camera jobs, 158 pairs, six mandatory physics head jobs (30 episodes), ~30.28 estimated camera lane-hours plus four preparation/physics lane-hours, excluding later replay audit occupation. Uses per-cell lock occupancy from permitted lane logs, with caveats for thin samples and routes. Coin is a prospectively amended 38-job first test retaining stronger gates; imitation retains its stronger score/CI gate. The 310 deferred jobs are not automatically replayed. No experiment results or adoption claimed.

Read-only SSH touched only Integration's named state CANDIDATE, lane2.sh, queue_after_dc19.txt and lane logs. Zero remote writes, GPU/process probes, sealed/coord/other-side reads, and LLM calls/tokens/spend. Local dependencies use the ordinary dev venv; no model/torch downloads. Local additions remain below 1 GB.

Static validation: 316 unique jobs, 158 pairs, exact baseline hash/config, only registered B deltas, valid RuntimeConfig fields, cx-only model pins, deployment SHA, matching salted block paths and allowed ids. Existing targeted tests: 21 passed, 1 skipped (RoboDojo assets unavailable locally), covering salts, presettle, demo-key routing and asset firewall. GPU validation is deferred to the explicit head jobs on compute4 after lead release. The plan PR is for lead review and remains unmerged.

Dispatch landmine documented in the plan: lane2 always calls integ_eval, so physics jobs must precede it through existing g3.sh; rendered mode-gt is not a substitute. g3.sh sets SIMLAB_GPU=0 and its child rdpy skips the disk guard, so the head commands require an explicit existing disk_guard preflight in the launching shell.
