# 2026-11-08: partner team's next-benchmark scan after RoboDojo and LIBERO-PRO (verbatim)

Received on the coordination bus (`20261108T0006Z-misha-next-benchmarks.md`). Read-only research by their side, with live pages fetched 8 Nov 00:00–00:05Z. Founders' decisions it raises: the RoboColosseum form (Ilia), a RoboCasa eligibility e-mail plus a capped API key for the organisers' re-run, and the LIBERO-Plus asset download. Our lead's answers to their three questions went back on the bus.

## Next benchmarks after RoboDojo and LIBERO-PRO, 2026-11-08

Read-only research: no sign-ups, forms, e-mails, installs or downloads. Live pages were re-fetched 8 Nov 2026 between 00:00 and 00:05Z, by me and by four Sonnet fetchers whose relayed text I spot-checked on RoboColosseum, RoboCasa and LIBERO-Plus. Labels: **[C]** checked at source this session (WebFetch summarises pages, so "verbatim" means verbatim as relayed); **[B]** believed or second-hand, mostly from the 5 Nov scout or our notes; **[G]** educated guess. Our own numbers are self-run results, never leaderboard scores. RoboDojo values are scores, not percents.

## Summary: top 3 and why

1. **RoboCasa365 agentic track (sim, MuJoCo).** This is the only live board with an agentic track, and its organisers re-run the full eval themselves and post their own numbers [C]. Exactly 1 agentic entry is merged: HexaShell 1.5 at 85.6, self-run on a non-official split [C]. The highest standard entry is 64.4; pi0.5 is 16.9 [C]. It runs on the same robosuite/MuJoCo family as our LIBERO stack [B]. A first legal run takes about 7-12 days [G]. Three things gate it, and each needs a decision: an arXiv paper, a test API key e-mailed to the organisers (our key rule forbids this today), and kitchen verbs we don't have yet (doors, drawers, knobs, base motion).
2. **RoboColosseum Open track (real robots).** It costs one form to apply, and the robot time is free. It has the strongest story: real robots, a blind pairwise arena, and a consortium of NUS, UW, Stanford, AI2 and HF [C]. The house Claude Opus 5.5 agent sits #1 on YAM (rating 2081) and on SO-101 [C], so the comparison of "same model family, with our harness vs without it" is built in. No rule permits or bars an outside LLM agent [C]. Send the form now, since acceptance is the long external lead time, and ask the API-call question in it. Ilia's side already has a real SO-101 [B].
3. **LIBERO-Plus (sim, our exact stack).** Setup is "identical to LIBERO", a package swap [C], which makes it the cheapest new number. Its 7 perturbation dimensions (camera, robot, language, light, background, noise, layout) test our SAM 3.1 + camera-only belief, which LIBERO-PRO does not [C]. Its board is a README table of VLAs only, top OpenVLA-OFT+ 79.6 [C]. The only harness number found is DynaHarness 84.4, with a frozen VLA in the loop [C via fetcher]. We would be the first report without a trained policy [B]. It takes 2-4 days, and the result is a paper number, not a board entry.

Next in line: **CaP-X/CaP-Gym** (4th): a cheap head-to-head with a public code-as-policy baseline, useful for the arXiv paper that RoboCasa requires. **RoboTwin 2.0** (5th): bimanual, but the board requires released weights, so it gives a paper number only. Watch list: BEHAVIOR-1K 2027 and the VLABench workflow board. Dropped: Vals Robo-bench, RoboArena, SimplerEnv, ManiSkill3, CALVIN, RLBench, Colosseum, AgiBot World Challenge, GRUtopia.

## Ranking table

| # | Benchmark | Sim/real | Board we can enter? | First legal run [G] | Full-protocol model cost [G] | Story | Main blocker |
|---|---|---|---|---|---|---|---|
| 1 | RoboCasa365 agentic | sim, MuJoCo | yes, by precedent (HexaShell merged with no trained policy) [C] | 7-12 days | ~$250-2,500 per 2,500-episode run, roughly doubled by the organisers' re-run on our key | official, organiser-verified number next to VLAs/WAMs | key handover, arXiv paper, articulated + mobile verbs |
| 2 | RoboColosseum Open | real | rules silent; house LLM agents already ranked [C] | 1-2 weeks of engineering after acceptance (acceptance date unknown) | ~$50-500 for 150-500 trials; robot time free | first real-robot result; same-model control built in | outside API agent unconfirmed; joint-space output; RGB only; no iteration |
| 3 | LIBERO-Plus | sim, our LIBERO stack | no submission process; README table [C] | 2-4 days | ~$50-1,000 (sample vs all 10,030 tasks) | robustness to visual perturbations, plus "first without a trained policy" | paper number only; camera perturbation may break the belief |
| 4 | CaP-X / CaP-Gym | sim (robosuite, LIBERO-PRO, BEHAVIOR) | no board [C] | 2-4 days (LIBERO-PRO + robosuite parts) | ~$30-300 | head-to-head with NVIDIA/Berkeley's code-as-policy baseline | paper number only; dependency conflicts |
| 5 | RoboTwin 2.0 | sim (SAPIEN [B]) | no: "publicly released weights" required [C] | 1-2 weeks | ~$75-500 (750-5,000 episodes) | bimanual handover story | no board listing; HexaShell 96.8 self-run |

---

## 1. RoboCasa365 agentic track

- **Measures:** 50 kitchen target tasks in three splits: Atomic-Seen, Composite-Seen and Composite-Unseen [C]. The split sizes are 18/16/16 [B, 5 Nov scout]. The official protocol "target50" is 2,500 episodes [C via PR #43 text; B via scout README read]. robocasa_version 1.0.1 [C].
- **Sim:** MuJoCo + robosuite, a mobile Franka (Panda on an Omron base) [B]. This is not the LIBERO robosuite version, so it needs a separate install [B, scout]. GPU: MuJoCo physics runs on the CPU and only rendering uses the GPU, so it should fit next to Isaac on the 5090 [G]. The pages state no GPU requirement [C].
- **Rules (verbatim as relayed, robocasa.ai/leaderboard, "Updated 11/01/2026") [C]:**
  - "Agentic entries wrap an unchanged base policy with added components such as a planner, memory, retries, or tool use."
  - "Using an existing model with an existing coding agent for basic steering does not qualify."
  - "Name, HF repo and commit, and license. No weight edits."
  - "Policy observations only (no privileged simulator state)." / "No ground-truth success signal during an episode." / "Cross-episode memory must be disclosed."
  - "Public repo, pinned dependencies, and one-prompt setup." / "For closed-API planners: provider, model ID, settings, and a test key with enough quota for one full eval."
  - The key goes by e-mail to robocasa@googlegroups.com, never in the PR. "An arXiv link is required so others can see your methodology and how the results were obtained." The organisers re-run and post their own numbers [C].
  - **LLM/API agents:** explicitly contemplated ("closed-API planners") [C]. **Per-step API calls:** not restricted in what we read [C]; HexaShell allowed 40 replies per atomic episode and 100 per composite [B]. **Privileged state:** banned [C].
  - **No base policy:** the rules neither allow nor ban it explicitly. HexaShell 1.5 (GPT-6-Astra writing Python, SAM 3, no trained policy) was merged as PR #39 on 1 Nov, listed as "Agent" [C]. That is precedent, but the "unchanged base policy" sentence leaves room for a refusal. Ask before building [B].
- **Live board (fetched 8 Nov 00:03Z, page updated 11/01/2026) [C]:**

| Rank | Entry | Type | Overall | Atomic-Seen | Composite-Seen | Composite-Unseen |
|---|---|---|---|---|---|---|
| 1 | HexaShell 1.5 | Agent | 85.6 | 91.1 | 85.0 | 80.0 |
| 2 | Awomo-0.5I | WAM | 64.4 | 86.5 | 58.8 | 45.2 |
| 3 | minionx | VLA | 62.6 | 81.8 | 60.0 | 43.6 |
| 4 | Awomo-0.5V | WAM | 62.5 | 85.0 | 60.4 | 39.4 |
| 5 | Axiom-0 | VLA | 61.6 | 86.5 | 59.4 | 35.8 |
| ref | GR00T N1.5 / pi0.5 | VLA | 23.9 / 16.9 | | | |

  - HexaShell's 85.6 is author-run on "pretrain50" seeds 20-49 with 30 episodes per task, "not on the official target50 protocol", and no organiser re-run is shown [C via fetcher reading PR #39]. HexaShell also wrote per-task skill notes for all 50 tasks, and its no-skills figure is 59.3 [C, HexaShell v3].
  - Open PR #43 "[AGENTIC] Add Auriga" (5 Nov, unreviewed) claims 71.8 for an LLM harness with no learned policy. The fetcher relays that the author discloses an in-episode ground-truth success/task_progress signal, which the rules forbid [C via fetcher; I confirmed only that the PR exists and is open].
  - Other agents' numbers on this suite: PyRUA-Lean atomic 86.7 / composite 39.4 with the RLDX-1 VLA in the loop [C, PyRUA v2]; RPent 59.20 [B]; Harness VLA "+25.4 points" with a frozen VLA [C, abstract].
- **What transfers:** our robosuite/OSC control and the pick/place/pour/press verbs [B]; SAM 3.1 open-vocabulary locate (HexaShell used SAM 3 here) [B]; the camera-only belief, if the robot-view cameras give depth (robosuite can render depth) [B]. Missing pieces: articulated parts (doors, drawers, knobs, faucets, microwave buttons), which BOARD-READINESS lists as "A", not built [C, our notes]; mobile-base moves [B]. HexaShell's adapter was 1,373 lines [B, scout].
- **Effort to the first legal run:** 7-12 days [G]: install and adapter 2-3 days, articulated + base verbs 4-6, dev practice on non-target seeds 2-3. Writing the arXiv paper comes on top, but RoboDojo eligibility (R5) needs a paper plan anyway [C, BOARD-READINESS].
- **Cost per full official protocol:** 2,500 episodes. At our measured anchors, $0.005 (LIBERO hybrid) to $0.104 (LIBERO HL2, $16.26/156) per episode [C, our notes], or $0.30 (general code agent LN-GEN) [C, our notes], with composite tasks being longer at ~$0.3-1 [G]: **~$250-2,500 per run** [G]. The organisers' re-run on our key costs about the same again [G]. Our balance is ~$106 [B, memory 7 Nov 23:05Z].
- **Story value:** high. It is the one place where an independent party re-runs a no-policy harness and lists it next to VLAs and WAMs, and Composite-Unseen is our generalisation story [G]. It is also small: 1 merged agentic entry [C].
- **Risks:** the 85.6 headline sets the visible bar, even though it was run off-protocol with per-task notes [C]; our first result would be more likely 40-65 [G]. The key handover conflicts with "keys only from the Mac relay": a separate capped key is the usual fix, and it is Mikhail's decision [G]. Other risks: the novelty clause; the arXiv requirement; and any cross-episode skill learning (the ratchet) must be disclosed and could be read as task-specific [B].

## 2. RoboColosseum Open track

- **Measures:** a blind pairwise arena. An evaluator writes an instruction on the spot, two unnamed models attempt it, and ratings are fitted from the preferences [B, scout + paper]. The fine-tune track (100 demos) is closed to us, since we have no training [B]. Robots: YAM bimanual, SO-101, Franka (DROID rig), G1 (Open board empty) [C].
- **Real robots:** our GPU is irrelevant. Inference "runs either locally on a laptop equipped with an RTX 5090 GPU or remotely through a server-hosted endpoint" [C, arXiv 2611.02497 §3.1]. The remote endpoint fits our rule that keys stay on the Mac [G].
- **Rules:**
  - No published rule permits or bars an outside LLM agent, per-step third-party API calls, or internet use [C: homepage, policy-server README, paper main text]. Participation is "determined by track and model compatibility" [C, Fig. 1].
  - The house agents (Claude Opus 5.5, GPT-6 Astra, Grok 4.7) run through Inspect Robots inside the policy server, with provider API calls each turn: "Keys still come from the Client, per trial" [C, README].
  - Their limits (paper App. D.1, relayed as a paraphrase; stated for the house agents, not as third-party rules): 30 model calls per trial, a 120 s per-call deadline, 800 control steps, absolute joint-position `move` targets, and 640x480 RGB from 2-3 views [C, paraphrase]. Depth and calibration are not mentioned [C].
  - Privileged state does not arise on a real robot.
- **Live board (`robocolosseum.ai/api/leaderboard`, ratings computed 2026-10-30T14:03:49Z):**
  - YAM: claude-opus-5-5 2081 (259 rating evals, 15.8 % success); gpt-6-astra 1608 (3.4 %); g05 1577; molmoact2 1572 [C, my fetch].
  - SO-101: claude-opus-5-5 1917 (5.0 %); pi05 1712; gpt-6-astra 1448 [C via fetcher].
  - Franka: gpt-6-astra 1925 (28.3 %); molmoact2 1869; g05 1744; claude-opus-5-5 1636, 4th [C via fetcher].
  - Ratings are on an Elo-like scale and are not comparable across robots. The paper says "active matchmaking reaches approximately 90% agreement with the full-data ranking after 400–500 evaluations" [C, §1].
- **What transfers:** the general verbs and SAM 3.1 locate [B]. These must be rebuilt for the real robots:
  - joint-position output, which needs IK per robot;
  - perception from RGB only (our belief uses depth) [B];
  - camera extrinsics, availability unknown [C: not stated].
  - Ilia's side would start with their real SO-101 and the original controlr harness [B, bus 7 Nov 20:18Z].
- **Effort:** the form today, with no engineering until accepted. Then 1-2 weeks for the policy-server backend, IK, RGB-only locate and the 30-call budget [B, draft]. The acceptance timeline is unknown [C: not published].
- **Cost:** robot time is free from the "10,000 free hours" pool [C, homepage]. Model calls: at most 30 per trial, so ~$0.1-1 per trial [G]; 150-500 trials comes to **~$50-500** [G].
- **Story value:** highest. It gives a real-robot rating against VLAs (pi0.5, GR00T, MolmoAct2) and against frontier models run raw [C]. It also yields a clean ablation: the house Claude agent has "no code skills, no memory across trials" [B, scout], and the harness is what we would add.
- **Risks:**
  - The organisers may decline an outside API agent; ask in the form.
  - Every run, failures included, is published on HF [B].
  - There is no way for us to iterate on failures, and the low absolute success rates (0-28 %) make ratings noisy.
  - Ilia's final answer is pending [B].
  - The terms page is unread [B].

## 3. LIBERO-Plus

- **Measures:** robustness of LIBERO tasks under 7 perturbation dimensions: Camera, Robot, Language, Light, Background, Noise, Layout. It has 10,030 tasks [C, repo README + fetcher], arXiv 2510.13626, CVPR 2026 [B].
- **Sim:** the LIBERO stack. "The usage of this project is identical to LIBERO": you replace the `libero` package and download `assets.zip` from HF (size not stated, so check that it is under our limits before downloading) [C]. It fits beside Isaac like our current LIBERO lanes [B].
- **Rules:** there is no submission process; the README table only invites PRs to the related-works list [C]. LLM agents, API calls and privileged state are therefore governed by our own protocol, which should be the same legality rules as HL2 [G].
- **Top (README table, undated):** OpenVLA-OFT+ 79.6, OpenVLA-OFT 69.6, RIPT-VLA 68.4 [C]. DynaHarness reports 84.4 on all 10,030 tasks, with a frozen VLA in the loop [C via fetcher, arXiv 2609.40306 v1, 30 Sep]. No code-agent or no-policy result was found [C/B].
- **What transfers:** everything the LIBERO harness uses [B]. Camera and noise perturbations directly test SAM 3.1 + the camera-only belief. If the camera pose changes and we read extrinsics from the sim, we must decide whether that counts as privileged information [G].
- **Effort:** 2-4 days, mostly the package swap, the task index, and a dev/held-out split like HL2's [G].
- **Cost:** all 10,030 tasks at $0.005-0.104 per episode comes to ~$50-1,050 [G, our anchors]. A stratified sample of ~1,000 costs ~$5-105 [G], but only the full set compares with the README table [B].
- **Story value:** medium. It shows the 94.2 % LIBERO-PRO result was not built for one camera or one lighting setup. The claim "first LIBERO-Plus number without a trained policy" holds only until someone else posts one [B].
- **Risks:** a paper number only. Camera-viewpoint shifts may sink depth-based placement [G]. The asset download size is unknown [C].

## 4. CaP-X / CaP-Gym

- **Measures:** coding agents as robot policies, 39 tasks over robosuite, LIBERO-PRO and BEHAVIOR, in tiers S1-S4 / M1-M4 [B, scout]. arXiv 2603.22435 v2 is dated 2 Jul 2026 [C].
- **Sim:** robosuite, LIBERO-PRO (its own Python 3.12 venv with `libero` and `contactgraspnet` extras), and BEHAVIOR on OmniGibson/Isaac Sim [C]. Robosuite and LIBERO cannot be installed together [C]. Contact-GraspNet needs a GPU [B].
- **Rules:** no leaderboard and no submission process [C]. It is a protocol for code agents by design [C].
- **Numbers (site, undated) [C]:** frontier models ">30%" average zero-shot. On LIBERO-PRO (30 tasks, position + instruction perturbations): CaP-Agent0 18 %, pi0.5 13 %, OpenVLA/pi0 0 %. CaP-RL Qwen2.5-Coder-7B 20 → 72 %.
- **Rivals on it:** ASPIRE (v2, 20 Oct) reports 72 % on LIBERO-Pro with Opus 4.6, and on 7 robosuite tasks x 100 trials 81 % vs CaP-Agent0 68 % [C via fetcher].
- **What transfers:** our LIBERO-PRO harness, if their LIBERO-PRO matches the one we run [B: the repo does not say]. Their API is a different tool surface, so we would wrap our verbs as the "agent" [G].
- **Effort:** 2-4 days for the LIBERO-PRO and robosuite parts [G]. BEHAVIOR would add 1-2 weeks [G].
- **Cost:** ~300-1,000 episodes, ~$30-300 [G].
- **Story value:** medium for investors, high for the arXiv paper. It is a public baseline that reviewers recognise, and ASPIRE already uses it [B].
- **Risks:** a paper number only. Protocol mismatch with our 94.2 % LIBERO-PRO run could confuse readers if we don't label it clearly.

## 5. RoboTwin 2.0

- **Measures:** 50 dual-arm tasks, clean and randomized scenes, Aloha-AgileX on the board [C]. The protocol is 100 trials per task [B, scout].
- **Sim:** SAPIEN [B]; the repo page states neither the simulator nor VRAM [C].
- **Rules:** "publicly released code," "publicly released weights," and a technical report; "The code must include post-training and deployment" [C]. A harness without weights cannot be listed [B]. Agents are not mentioned [C].
- **Top:** the board table did not render (page updated 2025/08/25) [C]. The HexaShell paper quotes a board top of 81.5 (GO-3 Nano Preview) and a published high of 90.4 [B]. HexaShell reports 96.8 with skills on 750 test episodes [C]. PyRUA-Lean reports 69.2 with LingBot-VLA in the loop [C]. Harness VLA reports C2R 58.4 [C]. RPent integrates it too [B].
- **Transfer:** handover / place_into_held verbs and the bimanual ARX experience from RoboDojo [B]. It needs a new simulator adapter.
- **Effort:** 1-2 weeks [G]. **Cost:** 750-5,000 episodes, ~$75-500 [G].
- **Story value:** low to medium while there is no board listing.
- **Risks:** we would be one more self-run number in a suite where HexaShell already reports 96.8.

## Watch list (not now)

- **BEHAVIOR-1K challenge.** Its rules are the most open to us of any read: "There are no restrictions on the type of policy used." "Additional components like SLAM or LLM-based querying are also permitted." Participants "must provide the credentials, quota, and serving configuration". Observations are RGB, depth and proprioception only [C]. But the 2026 edition is closed (deadline 16 Oct 2026, winners 4 Nov 2026) [C]. It runs OmniGibson on Isaac Sim, with a reference RTX 4090 24 GB [C]. Watch for a 2027 call. ASPIRE ran only 2 of its tasks [C via fetcher].
- **VLABench.** It has a "leaderboard of workflows", but only as a preview with the full board "soon"; MuJoCo [B]. It is the closest fit for an LLM workflow if the board opens. Its rules on privileged state are unverified [B].

## Dropped

- **Vals AI Robo-bench:** no submission path for outsiders [C]. It is still useful as context: vendor coding agents score GPT-6.1 Sol 28.33 %, GPT-6 Astra 23.33 % and Claude Opus 5.5 14.58 % (13 Oct 2026; MuJoCo; 24 tasks, 21 held out) [C].
- **RoboArena:** the site returned only its title; no live board or intake was found [C].
- **SimplerEnv, CALVIN, RLBench, ManiSkill3:** no official live board or 2026 challenge; numbers appear only in papers or on aggregators [B].
- **The Colosseum:** both the 2024 version and V2 (arXiv 2605.27759, May 2026, on ManiSkill) have no live board and no LLM-agent results [B].
- **AgiBot World Challenge 2026:** ran with ICRA 2026 and is closed [B].
- **GRUtopia/InternUtopia:** no 2026 edition [B].
- **NeurIPS 2026 RoboSynChallenge:** synthetic-data-to-real; poor fit [C exists, G fit].
- **ICRA 2027:** no manipulation challenge was found beyond the RGMC calendar page [B]. CoRL 2026 was not checked.
- **Real-robot alternatives:** none found. Robocurve's $500k program funds benchmark builders, not entries [B].

## Where the rivals report (besides LIBERO-PRO)

From the paper versions named; all numbers are self-run unless marked.

- **RoboDojo:**
  - HexaShell v3: score 75.73 without skills, 88.81 with skills, on public layouts. Its board entry is an older build at 53.18 [C].
  - PyRUA-Lean v2 (13 Oct) reports score 58.20 self-run on seed 0. The board showed 55.26 on 30 Oct [C paper; B board]. Re-fetch the board before quoting it.
  - RPent integrates RoboDojo; no number found [B].
- **RoboCasa365:** HexaShell (59.3 / 85.6), PyRUA-Lean (VLA in loop), RPent 59.20 [B], Harness VLA (frozen VLA).
- **RoboTwin 2.0:** HexaShell, PyRUA-Lean, Harness VLA, RPent.
- **robosuite:** ASPIRE, World Action Agent. **BEHAVIOR-1K:** ASPIRE, 2 tasks only. **LIBERO-Plus:** DynaHarness only.
- **Real robots:**
  - HexaShell: 3 dual-arm PiPER-X, 9 tasks, 76/90.
  - ASPIRE: bimanual YAM.
  - DynaHarness: UR7e.
  - Harness VLA: dual Franka [C via fetcher].
  - None of them is on a third-party real-robot board.
- **Nobody reports:** SimplerEnv, ManiSkill, CALVIN, full BEHAVIOR, RLBench, MetaWorld [C within the parts read].
- **Gap:** no rival without a trained policy reports LIBERO-Plus, and no outside harness appears on RoboColosseum [C/B].

## Decisions this needs (Mikhail / founders)

1. Send the RoboColosseum form now, with the API-call question in it. Ilia is "not against", final answer pending [B].
2. RoboCasa: an e-mail to robocasa@googlegroups.com asking whether a harness with no base policy qualifies. Separately: whether a capped, dedicated key may leave the Mac for their re-run, and the budget for two 2,500-episode runs.
3. Approve the LIBERO-Plus asset download (size unknown) and a 2-4 day run on compute4.

## Sources (fetched 8 Nov 2026 unless dated)

- RoboCasa365 board and rules: https://robocasa.ai/leaderboard ("Updated 11/01/2026"); PRs https://github.com/robocasa-benchmark/leaderboard/pulls (#39 merged 1 Nov, #43 open 5 Nov)
- RoboColosseum: https://robocolosseum.ai ; https://robocolosseum.ai/api/leaderboard?robot=yam&track=open (also so101, franka) ; https://arxiv.org/html/2611.02497 (§1, §3.1, §4.6) ; https://github.com/RoboColosseum/robocolosseum-policy-server ; form https://app.youform.com/forms/riubycrq (not opened this session)
- LIBERO-Plus: https://github.com/sylvestf/LIBERO-plus ; arXiv 2510.13626
- CaP-X: https://capgym.github.io ; https://github.com/capgym/cap-x ; arXiv 2603.22435
- RoboTwin 2.0: https://robotwin-platform.github.io/leaderboard ; https://github.com/robotwin-platform/RoboTwin
- BEHAVIOR-1K: https://behavior.stanford.edu/challenge/ ; https://behavior.stanford.edu/challenge/evaluation.html
- Vals Robo-bench: https://www.vals.ai/benchmarks/robo_bench
- RoboArena: https://robo-arena.github.io (title only)
- VLABench: https://vlabench.github.io
- Colosseum V2: https://arxiv.org/pdf/2605.27759
- NeurIPS 2026 competitions: https://blog.neurips.cc/2026/07/28/neurips-2026-competitions-announced/ ; https://robosyn-bench.net/
- AgiBot World Challenge 2026: https://2026.ieee-icra.org/program/competitions
- Rivals: HexaShell https://arxiv.org/abs/2610.23479 (v3, 3 Nov) ; PyRUA-Lean https://arxiv.org/abs/2610.01939 (v2, 13 Oct) ; ASPIRE https://arxiv.org/abs/2607.00272 (v2, 20 Oct) ; RPent https://github.com/rlinf/rpent ; DynaHarness https://arxiv.org/abs/2609.40306 (v1, 30 Sep) ; Harness VLA https://arxiv.org/abs/2607.08448 ; World Action Agent https://arxiv.org/abs/2609.29964
- Robocurve / Inspect Robots: https://runtimewire.com/article/robocurve-raises-10m-frontier-ai-robot-evaluations ; https://pypi.org/project/worldevals/0.3.2/
- Our notes: ~/Desktop/yc/bench/BENCH-SCOUT-2026-11-05.md, RIVALS-LIBERO-PRO-2026-11-05.md, BOARD-READINESS-2026-11-07.md ; ~/Desktop/yc/apply/ROBOCOLOSSEUM-APPLICATION-DRAFT-2026-11-05.md ; memory project_physical_ai_unicorn.md (HL2 $16.26/156, LN-GEN $0.30/episode, Ilia bus 7 Nov 20:18Z)

Not done: the RoboColosseum terms page and the rest of App. D.1 (the paper text after 100k characters); the live RoboDojo board (JS-rendered); RoboTwin's live table; the BEHAVIOR winners list; CoRL 2026.
