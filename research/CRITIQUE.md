## Errors

*Senior review, 2026-10-02. Checked against the 22 notes in `sources/` and `landscape/`, their fact-check sections, local clones in `repos/`, the bundled Claude API reference, and live re-fetches. New primary data used here: all 360 `data/runs/*.json` files from robocurve.org/opus-5-5-robodojo-rc-tier-1/ (fetched 2026-10-02) and that report's results table. "§" means a HARNESS_DESIGN.md section unless noted.*

1. **`research/REPORT.md` (whole file)** | The file does not exist. At review time the directory holds only HARNESS_DESIGN.md (written 00:51). The root filesystem is at 100% with about 4 MB free, so a write probably failed. As a result there is no REPORT to review and no way to check the two documents against each other. | Free space first. Scratch directories are large: `tmp_planner` 498 MB, `tmp_fc_planner` 246 MB, `tmp_direct` 241 MB. Then regenerate REPORT.md with the corrections below, and diff its numbers against HARNESS_DESIGN (§0, §1.4, §4.1, §6) before calling either final.

2. **§0.6, §4.1 "Reading", §7.1 risk 1, §6.2 M4: "1/120 vs 5/120 completions … That is the 'last centimetre' failure this design targets"** | There are two problems. (a) The difference is not significant: Fisher exact p = 0.21. (b) The "last-centimetre" diagnosis is unsupported, because nobody analysed the Tier 1 trials. My parse of the 360 public run JSONs points to a different failure signature. Opus 5.5 ended **45/120 trials with `done`, and 44 of those scored below 1** (73 `give_up`, 2 `overheat`). Astra ended 15/120 with `done`, 12 of them incomplete, and gave up in 103/120. The per-task progress table shows Opus 5.5 ahead of Astra on Cap Pen (57.0 vs 41.0) and Pack & Pour (24.0 vs 17.0), and behind on Stack Bowls (42.5 vs 65.0). Every task ran 20/20/20 on the same rig for all three models. | Report p = 0.21 and drop "Opus 5.5 may simply be worse" as a Tier 1 conclusion. Replace the last-cm assertion with the termination data. Make "classify the Tier 1 failure stages" a prerequisite (Gap 4).

3. **§0.3, §1.3, §4.2 "It overclaims success: Opus 5 claimed success on 13/38 failed attempts vs 3/20 for Astra"** | This is weak evidence. p = 0.22. The comparison mixes harnesses (Opus reference harness vs Astra under Codex CLI) and uses Opus 5, not 5.5. The design's own §1.3 also cites Astra falsely calling `done` in 7/20 puzzle trials. The strong evidence was available and was not used. In Tier 1, among non-completed trials, Opus 5.5 declared `done` in 44/119 vs Astra 12/115 (p ≈ 1.5×10⁻⁶), and Opus 5 in 51/118 (p ≈ 1×10⁻⁸). | Cite the Tier 1 rates, with model names and n. This strengthens R5 and the "independent verifier" design. Keep CodeActionBench only as a secondary, non-significant datapoint.

4. **§0.3 "weaker at pixel pointing: RefSpatial 56.3 vs 78.0", §2C "about 20 points weaker … on RefSpatial and PixMo", §4.2 routing "Pointing, part grounding → ER 2 / GroundingPI"** | The comparison is cherry-picked and mislabelled. These are **Opus 5** numbers at adaptive-low effort (PhysBrain 1.5, Table 4), not Opus 5.5. On the same table Opus 5 trails Astra by only 3.3 on PointBench (68.6 vs 71.9) and 0.6 on VABench-Point (64.7 vs 65.3). On **Part-Affordance it leads every frontier model** (78.1 vs Astra 55.0 and Gemini 3.6 Flash 64.7). Routing `part_hint` (handles, parts) away from Claude contradicts the only part-level number in the evidence base. There is also no published RefSpatial or PointBench score for ER 2. | Restate as: "Opus 5 at low effort trails on RefSpatial (−22) and PixMo-Points (−18), is within about 3 points on PointBench and VABench-Point, and leads on Part-Affordance. There is no Opus 5.5 or ER 2 pointing data." Make the pointer choice an output of a measured eval (Gap 2), not a default.

5. **§5.4 "Run Inspect Robots unchanged first … `image_horizon=2` … reproduces the baseline arm on Opus 5.5 with native caching and thinking replay"; §6.2 M0/M1** | This contradicts the design's own §1.8 and §3.9. In `inspect_robots_agent/policy.py:210-258`, `_evicted_view` rewrites earlier user turns on every call. `_anthropic.py:~405-420` replays thinking blocks verbatim. HEAD `095172f` has no `block_binding` or prefix handling. The Anthropic reference (Opus 5.5 breaking change 3) says that for accounts created on or after 2026-08-31 the prefix check is enforced, and a replayed block after any earlier edit is a 400. Robocurve's runs most likely came from an older account. A new account will probably fail on call 4. | Before M0, run a 5-call smoke test on Ilia's account. To reproduce the baseline, either set `image_horizon` to keep all frames (224 px frames are about 64 tokens each), or send `thinking.block_binding.prefix_mismatch_behavior:"drop_block"` with beta `thinking-binding-controls-2026-08-01`. Disclose that either change departs from the published configuration.

6. **§1.4 cost derivation and §0.7 "≈ $0.81 … close to Robocurve's measured $0.90" / "This matches"** | The match is coincidental, not a validation. The Tier 1 page reports Opus 5.5 at **717,344 mean input tokens per trial** (not 1.295M) and 14,496 output tokens. It used three 224×224 frames per turn (about 64 tokens each, not one 1,242-token tile), mutating image eviction, and a 0.87 cache-read share (the derivation implies about 0.96). On the recorded mix, $0.90 ≈ $0.12 cache reads + about $0.45 writes/uncached + $0.29 output. | Drop the validation claim. Calibrate the cost model on per-trial `cache_read_input_tokens`, `cache_creation_input_tokens`, `input_tokens` and `output_tokens` from the 120 Opus 5.5 run JSONs, then re-project to the 1920×480 tiled design.

7. **§1.3 "Pigey's `is_grasped` gripper-width check overrides an optimistic backend; simple pick-place went from 80% to 100% (n=20 per arm)"** | This is misattributed. 80% is **TiPToP open-loop** and 100% is the full Pigey system, on 4 tasks × 5 trials. The paper attributes the gain to the closed verify-and-retry loop, not to `is_grasped` alone (llm-planner fact-check, "Minor nuances"). | Restate as a system-level comparison. Do not present it as the measured effect of a deterministic grasp sensor.

8. **§1.3 "Intermediate verification … 86.2% → 48.2%", §5.5 item 5** | The framing is misleading. The paper's abstract calls 86.20% the performance of "task-adapted low-level VLA policies". It is only +3.46 pp over a π0.5 reference taken from LingBot-VA's per-task numbers, not re-run. The planner is Qwen3-VL, not a frontier model. Without verification the loop scores 48.2%, **34 pp below the bare-VLA reference**. | Read it as: "an agent loop that decomposes and retries without verification destroys about 34 pp of VLA performance; verification only restores it." Cite arXiv 2609.01281 Table 5, not the README.

9. **§0.5 and §7.4.1: "a real, unfilled niche … no published system pairs a frontier LLM with a small learned head"** | This is overstated by the design's own evidence. Adjacent cells are occupied:
   - Pigey: Opus 4.7 + frozen π0.5 `VLARollout` + TAMP, 97.3% over 150 real trials.
   - Harness VLA / RPent: Claude Agent SDK planner + `pi0_pick` contact primitive.
   - VoLo: Opus 4.6 + π0.5 on a real Franka.
   - AWS Strands reference design: Claude Sonnet 4.5 + GR00T on SO-101.
   - RoboDual and RLT: small heads under a big model.

   What remains novel is narrow: a *task-trained 20–100M target-conditioned head* under a frontier API. No evidence shows that it beats a frozen VLA primitive or plain IK, and VoLo's real-robot result has No-VLA at 45.2% ≥ VLA at 42.9%. | State the niche precisely and name the closest prior work. Add a head-vs-frozen-VLA-primitive ablation (Gap 3).

10. **§6.2 M2/M3/M4 vs §5.2 protocol** | The milestones contradict the protocol. M3, "the differentiating result", is "R2 puzzle ≥ 10/20 vs the 2/20 published baseline". That is n=20 against an external baseline that was non-interleaved, graded with the model known, run on different rigs at 224 px, and partly run through the compat shim. Fable 5.1's puzzle score is even 1/20 by the live operator verdict. §5.2 itself requires "≥50 per arm, ABBA-interleaved, same rig, blinded". | Run the baseline arm on Ilia's own rig: Inspect Robots + Opus 5.5 and Astra, interleaved, blinded. Use ≥50 per arm for M3 and M4, and call n=20 results "development".

11. **§0.2 "Opus 5.5 went from 32% to 60%…"; §4.1/§4.2 "best URAI agent"; §7.2 "Opus 5.5 32% vs Astra 16%"** | (a) The URAI tools were **authored by a Fable 5.1 agent and frozen**, and a pre-written program scored only 24%. The gain is therefore tool-authoring plus an interface change, not "interface" alone. (b) The model comparisons are noise: 15/25 vs 13/25 gives p = 0.78, and native 8/25 vs 4/25 gives p = 0.32. | Say "no detectable model difference at n = 25 per agent". Credit the tool author.

12. **§3.4 motion contract "arrive within 1 mm / 5 mrad and at rest"** | This is copied from GRID's `moveToPose` on a **Flexiv Rizon**. The evidence on YAM contradicts it:
   - Robocurve transcripts show pitch sag of −0.2 to −0.35 rad against a 0 target, plus IK oscillation holds.
   - RoboProbe arrival checks report errors such as 7 mm / 2.1°.
   - The design's own receipt example (§3.6) shows `residual_mm: 8.0`.

   On YAM this contract would return `StoppedShort` constantly. | Measure YAM repeatability and sag in Phase 1. Set per-arm, per-region tolerances. Adopt metal-arm's commanded-base sag compensation.

13. **§3.5 "Fallback head … π0.5 via openpi … or MolmoAct2-BimanualYAM on YAM hardware"; §6.2 Phase 4** | Neither is a working zero-shot fallback on YAM:
   - MolmoAct2-BimanualYAM scored **0/100 completions** in StationeryBench (mean progress 12), and in 47/100 trials the arms never left the start pose.
   - Manda's simulated YAM rig gave it 0/4 on three unseen tasks.
   - π0.5 has no YAM checkpoint (DROID-Franka, ALOHA).

   | Treat both as fine-tuning bases that need YAM demonstrations, and budget that data. Otherwise the fallback for an untrained skill is "IK + visual servo".

14. **§3.5 training recipe "Training-time RTC with the measured delay distribution (LeRobot … π0.5 only)"; §3.8 "Use RTC/TT-RTC"** | This is a category error for a 20–100M local head. ACT runs in about 10 ms (2080 Ti), so there is no chunk-boundary delay to absorb. LeRobot's TT-RTC (PR #4056) is π0.5-only and not in the v0.6.x tags. The real staleness is a target spec that is seconds old, outside RTC's ≤300 ms regime (engineering-challenges §1.3). | For small heads, drop TT-RTC. Keep target re-binding every tick plus spec-age/staleness augmentation (§3.5 item 5). Use RTC only if the "head" is a remote GPU VLA.

15. **§5.1 RoboLab-120 row "via Manda's RoboLab-Verified fork … baselines FLUX 3 Action 42.9%, … Cosmos3 35.1%, π0.5 27.5%"** | The numbers are not comparable. The leaderboard and Manda's 6,000-episode re-runs used **upstream scoring** on the Isaac Sim 6.0 port. The Verified patches (at-rest success, carry-required grasps, rim-capped containment) are validated on about 25/120 tasks, and the full patched re-run is "planned, not yet scheduled". Scoring the harness on the Verified fork against upstream-scored baselines biases against the harness. | Report both scorers. Re-run π0.5 under Verified scoring; it is cheap at 49.7 episodes per GPU-hour.

16. **Appendix A (and §1.8, §3.6, §3.9) "UNVERIFIED" items that are resolved or wrong** |
   - (a) "Robocurve frame resolution … not stated": the robocurve note decoded 2,196 frames (all 224×224, squashed by `cv2.resize`), and the Tier 1 page states "recorded frames are 224 × 224 pixels".
   - (b) Strict tools: the Anthropic structured-outputs reference lists numerical constraints (`minimum`/`maximum`/`multipleOf`), string length, and complex array constraints as **not supported** (SDK helpers strip them and validate client-side). So `xyz_m` and `center_px` carry no length guarantee.
   - (c) "Compaction header conflict": `compact-2026-01-12` (threshold compaction) and `compact-2026-09-04` (the on-demand `compaction` parameter, designed to keep retained turns' thinking valid, not on Bedrock) are **two different features**.
   - (d) `clear_at:"next_user_message"` is listed for Opus 5.5 as a limited beta (`mid-conversation-system-clear-at-2026-08-21`).

   | Update Appendix A. Make server-side validation of array lengths and ranges a hard requirement. Pick `compact-2026-09-04` for preserved-thinking compatibility.

17. **§0.1 "5–15 s at low reasoning", §2 "Shared per-call numbers", §2A "5–14 s"** | Anthropic's 5–15 s figure is for **non-reasoning** Opus 4.x turns with 1–2 images. Opus 5.5 cannot disable thinking. Better data was available. From the Tier 1 run JSONs (wall_s / llm_calls, motion included), Opus 5.5 has a median of **7.6 s per call and p90 of 9.7 s**; Astra 8.8 s / 10.5 s; Opus 5 12.3 s / 16.3 s; 120 trials each on identical rigs. The "Astra 14.66 s" RoboDojo figure ran through ByteDance's internal model hub (`aidp.bytedance.net`) on a shared cluster, and its README says latency "compares runs, not models". | Replace §2's shared numbers with the Tier 1 distribution. Label the RoboDojo latency as a proxy-endpoint figure.

18. **§2C "ER 2 runs about 4.6–6.1 s per call (median, clapboardbench)" used as pointer latency** | This figure is ER 2 acting as a joint-motion controller (one motion per call, 3×224 images), not a pointing query. The ER docs also advise "Query multiple times and average", which multiplies latency. | Mark ER 2 pointing latency UNVERIFIED and measure it (Gap 1).

19. **§3.9 "quackd issue #25: one observation-time sentence moved correct `feasible` verdicts from 5/30 to 27/30"** | This is cited as support for Claude mid-conversation system messages, but it was measured on **Qwen3-32B-AWQ in `microduck:sim2d` with temperature pinned to 0**. Transfer to Opus 5.5 is unmeasured. | Label it as cross-model evidence. Cite Anthropic's own guidance instead: placement near the end of the request moves behaviour more than wording in the system prompt (Fable 5.1 migration notes). Plan an n≥30 test on Opus 5.5.

20. **§2D "GPT-as-Policy: Hybrid beat Direct on RoboDojo (24/50 vs 13/50)"; §7.2 row** | Material caveats are missing:
   - The π0.5 comparator is a reweighted leaderboard number, not a rerun.
   - The largest hybrid gains came with almost no LLM authorship: fold_clothes had 1.7% LLM-authored steps and put_bottles 0.2%. Executing 1–15-step prefixes of a 50-step chunk may itself explain them, as receding-horizon replanning.
   - The RoboLab arms used retries and were not paired.

   | Add the caveats. Include "π0.5 with a fixed 15-step prefix" as a required control before treating hybrid gains as LLM-supervision gains.

21. **§2A "Evidence for … negated xyz recovered 4/8, mirrored images 6/8"; §2E/§7.1 "GEN-1.5 59% one-shot", "Skild S1 66% vs 9%"; §7.1 Helix 2.5** | These break the design's own "n ≤ 10 is anecdotal" rule:
   - The RoboDojo perturbation layouts were "selected as those the unperturbed model solves", with 1 episode each.
   - GEN-1.5 and Helix 2.5 report no trial counts.
   - Skild's 66% comes from a filtered subset of its own pretraining pool.

   | Flag each as anecdotal or vendor-only where it is cited.

22. **§4.1 and §4.2 model attribution: "Claude … video success detection 81.0 … progress 37.1 … safety 95.9", "Claude's documented pointing gap"** | All of these are **Opus 5** (legacy) vendor-chart or low-effort numbers. Opus 5.5 "thinks more per turn than Opus 5 at the same effort" (docs), so transfer is an assumption. | Name the model in every row. Add a line that no ER-2-chart or pointing measurement exists for Opus 5.5.

23. **§6.1 hardware total "≈ $10.0k" and API budget "≈ $900"** | (a) Bimanual leader-follower teleop for Phase 3 needs **two** YAM leaders, which makes it about $13.0k, plus an unpriced GPU. (b) The campaign is 900 trials × the Tier 1 mean wall time (251–435 s) plus about 14 s of reset each, i.e. **≈ 70–110 robot-hours**. Robot time, not API dollars, is the binding budget, and YAM thermal trips (10/120 for Opus 5) add re-runs. | Correct the bill of materials and add a robot-hour and operator budget to §6.1.

24. **§1.5 and §0.8 "Text in the scene hijacks VLMs 27–29%"** | This was measured on GPT-4o and Gemini 2.5 Flash only. Claude and GPT-6 were untested. | Name the models. Add a scene-text injection test on Opus 5.5 to R7 (§5.1 R7 already lists it; state that the baseline is unknown).

## Gaps

### Gap 1: Measured per-call latency and cost of the candidate backbones under the harness's real request shape

Why: Every timing decision in the design depends on per-call latency. That includes the 0.03–0.2 Hz duty cycle, "park if a loaded arm waits more than 8 s", "2–4 calls per pick-place", speculative next calls, and routing pointing to ER 2. Yet there is no per-call measurement for Opus 5.5:
- the Artificial Analysis snapshot predates it;
- Tier 1 gives only wall time divided by calls (median 7.6 s, motion included);
- ER 2 pointing latency, Sonnet 5.5 `between_tools`, Opus 5.5 fast-mode time to first token (TTFT) and Astra `service_tier:"ultrafast"` are all unmeasured.

Research task:
1. **Mine the public Tier 1 trials.** From https://robocurve.org/opus-5-5-robodojo-rc-tier-1/ download the 360 `data/runs/*.json` files (fields `wall_s`, `llm_calls`, `cache_read_input_tokens`, `cache_creation_input_tokens`, `input_tokens`, `output_tokens`, `termination`). Download the per-trial "Log" transcripts as well; they contain per-call latency lines, as in the Astra report's `call N attempt M · endpoint · status · latency`. From these compute per-call latency p50/p90/p99 and output tokens per call for `claude-opus-5-5`, `gpt-6-astra` and `claude-opus-5`.
2. **Run a 50-call benchmark per configuration** with a fixed realistic request:
   - a 10k-token cached prefix with 8 `strict` tools;
   - one 1920×480 JPEG tile per turn;
   - history depths of 5, 20 and 35 turns;
   - a required tool call.

   Configurations:
   - Opus 5.5 at effort low, medium and high; also per-message effort under beta `mid-conversation-output-config-2026-07-01`, and `speed:"fast"` if access is granted;
   - `claude-sonnet-5-5` with `thinking:{type:"between_tools"}` at low;
   - `claude-haiku-4-5`;
   - `gpt-6-astra` on Responses at low and medium, and with `service_tier:"ultrafast"`;
   - `gemini-robotics-er-2-preview` answering a single-point query at `thinking_level` low and medium, with k = 1 and k = 3 averaged;
   - `gemini-3.8-flash`.
3. **Record per call:** TTFT, total latency, output and thinking tokens, cache read and write tokens, dollar cost, zero-tool-call rate under `tool_choice:auto`, and HTTP error and timeout rate.
4. **Deliver** a CSV plus a table of p50/p90/p99, and the implied maximum decision rate per configuration.

### Gap 2: Pointing and grounding accuracy, in millimetres, on robot camera images

Why: R3 and §3.3 route parts, grooves and handles to a "specialist pointer" (ER 2, GroundingPI-4B, or zoom plus marks) on the strength of benchmark scores. Those scores are Opus 5 at low effort, and they conflict: RefSpatial is −22, but Part-Affordance is +23 vs Astra. There are no Opus 5.5 numbers, no ER 2 RefSpatial or PointBench numbers, and no pixel-to-millimetre error for any model on a robot cell. The 1–8 mm task tolerances (§1.2) make this the deciding measurement for the contact-skill target specification.

Research task:
1. **Compile every published pointing number** for Opus 5/5.5, GPT-6 Astra, Gemini Robotics-ER 1.5/1.6/2, GroundingPI-4B, MolmoPoint/Molmo2, PhysBrain 1.5, RoboPoint and Embodied-R1.5. Sources: PhysBrain 1.5, arXiv 2609.14973, Table 4; GroundingPI, arXiv 2609.39601; Gemini Robotics 1.5, arXiv 2510.03342, Table 19; MolmoPoint, arXiv 2603.28069. Note effort settings and splits.
2. **Build a 300-target evaluation set** from overhead D435 and wrist D405 frames of a tabletop cell. If Ilia's cell is not ready, use Robocurve Tier 1 trial MP4s or DROID frames.
   - Target types: object centroids, graspable parts (handles, rims), slots and grooves, and place points.
   - Ground truth: click annotation plus depth deprojection to 3D in the base frame.
3. **Query each model in its native convention:**
   - Claude: absolute pixels on a locally pre-resized image with `"oversized_image":"error"`.
   - ER 2: `[y,x]` normalised to 0–1000, single query and 3-query average.
   - GroundingPI and PhysBrain: run locally.
4. **Run each query in 4 variants:** plain, zoom crop, Set-of-Mark ids, and a 5 cm grid overlay.
5. **Report:** pixel error, 3D error after deprojection at 0.5 m and 1 m, hit rate within 5/10/20 mm, failure and refusal rate, latency, and dollars per query.

### Gap 3: Which conditioning makes a light action head follow LLM targets, and does a trained head beat a frozen VLA primitive?

Why: This is the core of Ilia's "LLM backbone + light action head" idea. The design picks mask overlay + gripper-frame 3D keypoint + phase token for a 20–100M chunked head without direct evidence. Three findings argue for caution:
- Fast Plans, Faithful Actions (arXiv 2609.30833) shows executors ignore waypoint conditioning.
- Conditions here arrive every 5–40 s from noisy pointers.
- The closest systems (Pigey, RPent/Harness VLA, VoLo) use a *frozen* π0.5 primitive. VoLo's real-robot No-VLA arm matched the VLA arm.

Research task:
1. **Literature table.** For each of the following, record conditioning type, head size, demos, success, the conditioning-erasure ablation, and robustness to conditioning noise and latency: HAMSTER (2502.05485), PEEK (2509.18282), RT-Trajectory (2311.01977), TraceVLA (2412.10345), MOKA→Octo distillation (2403.03174), RoboDual (2410.08001), RL Token (2604.23073), LoHo-Manip (2604.21924), Fast Plans (2609.30833), MolmoAct trace editing (2508.07917), π0.7 subgoal images (2604.15483) and Embodied-R1/R1.5.
2. **Controlled simulation study.** Use the MuJoCo YAM model from `inspect-robots-yam`, or RoboTwin 2.0 with a YAM-like arm.
   - Tasks: 2 mm peg-in-hole, puzzle disc into groove, grasp centring.
   - Data: scripted privileged controllers multiplied with EmbodiedSWE-style clean-label noise (`data_engine/agent/prompts/noise.md`), 200 demos per task.
   - Heads: ACT-size (about 80M), trained under 5 conditionings: phase token only; 3D keypoint in the gripper frame; 2D mask overlay; 2D trace; keypoint + mask. Train each with 0% and 20% condition dropout.
   - Evaluation, ≥100 episodes per cell:
     - success rate;
     - sensitivity to erased or shifted conditions (success must drop);
     - robustness to keypoint noise σ = 2/5/10 mm and to specs 0/2/5/10 s stale after the object moves.
   - Baselines: a frozen generalist VLA primitive (π0.5 fine-tuned on the same data) and an analytic IK + wrist visual-servo controller.

### Gap 4: Forensic analysis of the 360 public Robocurve "Opus 5.5 on RoboDojo-RC Tier 1" trials

Why: This is the only real-robot evaluation of the chosen backbone, a user-adjacent Robocurve source, and it was used only as aggregates from a fact-check addendum. Several design claims rest on it: "last-centimetre failure", "Opus 5.5 vs Astra parity", the verifier design and the cost model. My quick parse already contradicts the last-cm reading: 45/120 Opus 5.5 trials ended with `done`, and 44 of those were incomplete. Per-task progress also flips between the models: Opus 5.5 leads on Cap Pen 57.0 vs 41.0 and trails on Stack Bowls 42.5 vs 65.0. Nobody has looked at the videos or transcripts.

Research task:
1. **Download** from https://robocurve.org/opus-5-5-robodojo-rc-tier-1/ all `data/runs/*.json` (score, rubric labels, termination, wall_s, llm_calls, tokens), the per-trial transcripts ("Log"), and the Left/Top/Right camera videos for the 120 Opus 5.5 and 120 Astra trials. Use a curl User-Agent; Python's default UA gets a 403.
2. **Classify each non-completed trial's terminal failure stage** from the rubric score plus video: no grasp, grasp slipped, transport collision or drop, final placement or alignment (the "last cm"), IK stall or sag, budget exhausted, overheat, or premature `done`.
3. **For each premature `done`,** extract the model's stated evidence (e.g. "wrist camera confirms") and check whether the decisive region was visible in the 224 px frames.
4. **Compute per-call latency, output tokens and cache share** per model and task.
5. **Deliver** a task × failure-class count table per model with Wilson CIs, and a list of which proposed contact skills (`center_grasp`, `descend_until_contact`, `insert`, `place_release`, `regrasp`) would have addressed which failures.

### Gap 5: Accuracy of the independent success verifier that gates `finish(success)`

Why: R5 and §3.10 make the harness fail closed on a "separate model" verifier plus sensors. The M2 target is an overclaim rate below 5%. But no failure-detection accuracy has been measured for Opus 5.x or GPT-6: FailBench (arXiv 2609.03611) tested neither, and its best detector reaches 0.77 balanced accuracy, below 0.60 on contact tasks. ER 2's 82.4% is vendor-reported. Robometer-4B is unmeasured. A fail-closed gate at those accuracies would reject many true successes, and Tier 1 shows Opus 5.5 self-reports are wrong in 44/45 `done`s.

Research task:
1. **Assemble a labelled corpus of real-robot episodes:**
   - Robocurve GPT-6 Astra report: 120 trials with MP4, `.rrd` and operator stage 0–4 grades (https://openai.robocurve.org/gpt-6-astra/);
   - Robocurve Tier 1: 360 trials with three camera videos and rubric scores;
   - GPT-as-Policy: HF dataset `YuMoool/astra-robodojo-rollouts`, 100 episodes with native scores and per-tick frames;
   - RoboDojo-Real published videos, scored by 3 blind raters;
   - any of Ilia's own YAM trials.
2. **Evaluate these verifiers:**
   - Opus 5.5 on Manda-style timestamped contact sheets (`manda-robot-episode-labeler/src/rel/video/contact_sheet.py`, prompt rule "Judge failure by what you can see. Do not infer success from the task description.");
   - Opus 5.5 on raw end frames;
   - `gemini-robotics-er-2-preview` video success detection;
   - LeRobot v0.6 Robometer-4B, run locally;
   - GPT-6 Astra;
   - deterministic gripper and effort predicates, where state logs exist.
3. **Report:** balanced accuracy, false-success rate at a 5% false-reject rate, accuracy on the contact/insertion subset, inter-verifier agreement, latency, and dollars per check. Also report the best 2-of-3 ensemble policy.

### Gap 6: Is ≤2 mm contact work feasible on a YAM-class, position-controlled, $3k arm, and with how much data?

Why: Milestone M3 ("puzzle ≥10/20 with 100–200 demos per skill") is the design's differentiating claim and its main hardware bet, but it has no evidence base. Four questions are open:
- What YAM repeatability and sag actually are; Robocurve shows −0.2 to −0.35 rad of pitch sag.
- Whether the Damiao motors expose usable impedance (MIT-mode Kp/Kd) and effort sensing for compliant insertion; Metal's Damiao bus silently clamps Kd to 5.
- What the thermal limits are under holds (10/120 overheats for Opus 5 in Tier 1).
- What small policies have achieved on comparable low-cost insertion tasks.

Research task:
1. **Characterise YAM from primary sources:**
   - the `i2rt-robotics/i2rt` driver and doc.i2rt.com specs;
   - `inspect-robots-yam` (`kinematics.py` IK oscillation hold and resync; `embodiment.py:1694-1745` thermal guardrail; `config.py` gripper `LINEAR_4310`);
   - the AGP YAM bridge (`agent-as-policy` repo: `SETTLE_MISS` semantics, 0.03 m/s cap);
   - ENPIRE's 8-station YAM fleet and its pin-insertion results (arXiv 2606.19980);
   - DexAgent's YAM Box and Manda's bimanual YAM sim.

   Extract repeatability, payload-dependent sag, available control modes (position, MIT impedance, current), effort-sensor resolution, gripper force control and thermal limits.
2. **Compile published ≤3 mm insertion, peg, plug and key results with small policies on low-cost arms:**
   - ACT / ALOHA slot insertion (2304.13705);
   - Diffusion Policy;
   - RL Token screw task;
   - ENPIRE pin-box 99%;
   - GPT-Policy plug 2/3;
   - RoboICL peg-in-hole 2/5;
   - piper-astra-jev SlideOver;
   - FAEA PegInsertion 0%.

   For each, record demos, sensing (wrist camera, force/torque, effort), compliance and trial counts.
3. **Write a one-day bench protocol** for the bought arm:
   - 50-return repeatability;
   - static sag map over the workspace;
   - effort-based contact-detection threshold;
   - achievable impedance stiffness;
   - a 20-trial scripted wrist-camera visual-servo insertion baseline at 2 mm clearance, which sets the bar the learned head must beat.
