# Experiments — index

Protocol and report template — [../EXPERIMENTS.md](../EXPERIMENTS.md). One experiment = one dated
file; never edited after the fact (new data = a new report linking the old one).
Runs — `runs/<run_dir>` (not in git), videos — [../video/](../video/).

| Date | Report | Question | Result (small n — see report) |
|---|---|---|---|
| 2026-10-02 | [smoke-v0](2026-10-02-smoke-v0.md) | does harness v0 work in Isaac; cache and latency per model | waffle 2/4 (Sonnet 5 and Opus 5.5 one each, both seed 1), reach 0/4 (ambiguous target); cache ≈ 95 %, LLM p50 1.4–2.3 s |
| 2026-10-02 | [rotation-yaw](2026-10-02-rotation-yaw.md) | can the model rotate the gripper (`rotation=yaw`, packet yaw ±40°) | 2/10 (Sonnet 5.5 1/4, Opus 5.5 1/4, Haiku 0/2); rotation works (8/10 within 12°), failures in the carry over the rim |
| 2026-10-02 | [contacts-and-speed](2026-10-02-contacts-and-speed.md) | dynamic box, packet-vs-environment stop, short feedback (TASK/STATE/WARN/STOP); Isaac speed | Sonnet 5.5 4/4 on the rotation seeds and plans (was 1/4); every wall touch recovered; sim 0.18× → ≈0.7× real time (≥1× only with settings that broke grasps) |
| 2026-10-04 | [luna-control](2026-10-04-luna-control.md) | GPT-6 Luna (`cx/`, effort none) as the control model on the rotation pick-to-box | Luna 3/8 (seeds 0–1 3/4, seeds 2–3 0/4) vs Sonnet 4/4; failures: grasp blow-ups, carry into the wall; LLM p50 2.4–2.9 s (floor on `cx/` ≈ 1.1–1.4 s to first token) |
| 2026-10-04 | [robodojo-pilot](2026-10-04-robodojo-pilot.md) | does controlr drive RoboDojo's dual ARX X5 through RoboDojo's own eval client (general_pickup) | 2/5 (Sonnet 5.5; RoboDojo's `_result.json` agrees); failures: the 200-env-step budget runs out during re-grasps; LLM p50 ≈ 7 s, cache ≈ 0.87 |
| 2026-10-04 | [robodojo-fast-ab](2026-10-04-robodojo-fast-ab.md) | effort low + 336 px / JPEG 75, and 0.12 rad per env step, on RoboDojo general_pickup × 10 | base 5/10 (8.2 s/turn), low 3/10 (3.7 s/turn, reasoning moves into prose), 0.12 rad/step 0/3 stopped (grasps slip); none significant at this n |
