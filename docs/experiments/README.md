# Experiments — index

Protocol and report template — [../EXPERIMENTS.md](../EXPERIMENTS.md). One experiment = one dated
file; never edited after the fact (new data = a new report linking the old one).
Runs — `runs/<run_dir>` (not in git), videos — [../video/](../video/).

| Date | Report | Question | Result (small n — see report) |
|---|---|---|---|
| 2026-10-02 | [smoke-v0](2026-10-02-smoke-v0.md) | does harness v0 work in Isaac; cache and latency per model | waffle 2/4 (Sonnet 5 and Opus 5.5 one each, both seed 1), reach 0/4 (ambiguous target); cache ≈ 95 %, LLM p50 1.4–2.3 s |
| 2026-10-02 | [rotation-yaw](2026-10-02-rotation-yaw.md) | can the model rotate the gripper (`rotation=yaw`, packet yaw ±40°) | 2/10 (Sonnet 5.5 1/4, Opus 5.5 1/4, Haiku 0/2); rotation works (8/10 within 12°), failures in the carry over the rim |
| 2026-10-02 | [contacts-and-speed](2026-10-02-contacts-and-speed.md) | dynamic box, packet-vs-environment stop, short feedback (TASK/STATE/WARN/STOP); Isaac speed | Sonnet 5.5 4/4 on the rotation seeds and plans (was 1/4); every wall touch recovered; sim 0.18× → ≈0.7× real time (≥1× only with settings that broke grasps) |
