# Industry and competition: robot foundation models and LLM-driven robot software (2025–2026)

*Landscape sweep for Ilia Mikhalchuk's planned "frontier-LLM API backbone plus light action head" robot harness. Researched 2026-10-01.*

**Method.** Every 2025–2026 claim below links a page I opened on 2026-10-01: a company blog, docs, filing, arXiv abstract, GitHub API, or a news article when no primary page was reachable.

- **UNVERIFIED** marks numbers I saw only in search-engine summaries, or could not confirm on a primary page.
- **Unreachable pages:**
  - Bot walls: nvidianews.nvidia.com and investor.nvidia.com (403/Cloudflare), openai.com/index/gpt-6-astra (Cloudflare).
  - Other 403s: cnbc.com, scmp.com, the Business Wire Sanctuary release.
  - pi.website blocked plain fetches behind a Vercel challenge, so I rendered it with `agent-browser`.
- The shared web-search budget ran out near the end of this sweep. The last few checks used direct fetches only.
- GitHub star counts are from `gh api` on 2026-10-01.

Teammate deep-dives on the user's own sources are referenced, not repeated:
- `sources/manda-robotics.md`
- `sources/robocurve-gpt6-astra.md`
- `sources/general-robotics-auto-engineering.md`
- `sources/innate-os-pr817.md`
- `sources/metal-arm-harness.md`
- `sources/piper-astra-jev.md`
- `sources/robodojo.md`
- `sources/gpt-as-policy.md`
- the other `landscape/*.md` files

---

## 0. TL;DR

1. **The money and the talent sit in proprietary robot foundation models, not in LLM harnesses.** Reported valuations:
   - Figure: $39B (Sep 2025).
   - Skild: >$14B (Jan 2026).
   - Physical Intelligence (PI): $5.6B confirmed, ~$11B in talks.
   - Apptronik: ~$5B+.
   - Generalist: $3B.
   - Unitree: IPO at ~¥61B, then a ~$50B market cap on day one. [UNVERIFIED: "~¥61B" is not in the cited source. Crowdfund Insider gives ¥150.8/share and ~$900M raised for 10% of enlarged share capital, which implies roughly $9B at the IPO price, then "near $50 billion" at the day-one morning close.]

   ~~All of these train their own models~~ [corrected: all except Apptronik train their own models. Apptronik's humanoids are "powered by Gemini Robotics" through its Google DeepMind partnership (GlobeNewswire, Feb 11, 2026). Boston Dynamics' Atlas also adopted Gemini Robotics (CES, Jan 5, 2026; see Verification).] Their model types: VLAs, world-action models (WAMs), or in-context "physical prompting" models.
   - **None of them uses a third-party LLM API as the controller.**
   - The one large exception is **Google**. It ships the stack Ilia is contemplating as a product:
     - Gemini Robotics **ER 2**, a reasoning VLM, is sold by API at **$1/$5 per M tokens through 2026, with a free tier**.
     - It orchestrates Google's own **Gemini Robotics 2** VLA and **On-Device 2** VLA.
2. **Frontier LLM labs are entering robotics directly.**
   - **OpenAI** turned its world-simulation group into an **OpenAI Robotics** division (May 31, 2026, led by Aditya Ramesh). Altman says it will "definitely do a humanoid" ~~(UNVERIFIED quote)~~ [corrected: the quote is confirmed by a secondary source. Humanoids Daily (Sep 2, 2026) quotes Altman on the Sources podcast: "We will definitely do a humanoid... We will do other form factors as well."]. GPT-6 Astra ($10/$50 per M) is the strongest "LLM as policy" model on third-party benches.
   - **Anthropic's** physical-world work is research plus infrastructure:
     - Project Fetch phase 2 (Opus 4.7 programmed a quadruped about 19–38× faster than human teams).
     - "Claude plays robotics" (direct control is "bad, but improving").
     - Project Pilot (drones).
     - The **Model Hardware Standard (MHS)**: a model-agnostic, MCP-reachable driver standard with limits enforced in the driver. LeRobot and AWS Strands are already early adopters.

     Anthropic has no robot model and no robot product.
3. **The "API-LLM robot harness" layer already has many open-source competitors**, and its moat is low: dimOS (4.6k★, default `gpt-5.6-luna`), OpenMind OM1 (2.9k★), ROSA (1.6k★), ros-mcp-server (1.5k★), RAI, AWS Strands Robots (reference design pairs Claude with GR00T), innate-os (default Gemini 3.6 Flash), the closed GRID and Mbodi platforms, and NVIDIA's open-source Isaac ROS 5.0 agentic workflows (details in §5).

   NVIDIA is buying Hugging Face, the home of LeRobot, Pollen and Reachy Mini, for **$12.93B** (announced Sep 3, 2026).
4. **The sharpest threat to an "LLM plus light head" design is in-context learning inside robot foundation models.**
   - Generalist **GEN-1.5**: 59% one-shot success from a 3–12 s physical prompt.
   - Skild **S1**: 66% vs 9% for a language-prompted VLA on unseen tasks.
   - PI **π0.7**: language-coachable, with a learned high-level policy slot.

   These attack the same "teach a new task without training" value proposition that LLM in-context harnesses (e.g. Innate PR #817) are chasing.
5. **Where an Opus-backbone harness can still win:**
   - Embodiment-agnostic **System 2 / verifier / recovery** over cheap long-tail hardware where no good VLA exists.
   - **Safety and governance in the driver** (the MHS pattern).
   - Using the LLM as a **data engine** for a small fine-tuned action head.
   - **Provider-agnostic routing**: Claude, GPT-6, and Gemini ER 2 differ by up to 10× in price.

   Things that look unpromising:
   - LLM direct joint or torque control.
   - LLM humanoid locomotion.
   - Single-provider lock-in.
   - Yet another generic ROS↔MCP bridge.

---

## 1. Market map

Columns: what the company does, what is shipped or open, money, and whether it uses API LLMs or its own models.

| Company | Core approach | Shipped / open | Funding / valuation (cited) | API LLM vs own |
|---|---|---|---|---|
| Physical Intelligence | VLA family π0→π0.5→π*0.6 (RECAP RL)→**π0.7** (5B; Gemma3-4B + 860M flow action expert) | openpi: π0, π0-FAST, π0.5 weights, Apache-2.0, 14.1k★. π0.6/0.7 closed; partner program ("Physical Intelligence Layer") | $600M at ~$5.6B (Nov 2025); ~$1B at >$11B ~~in talks (Mar 2026)~~ [corrected: Bloomberg reported talks in Mar 2026. Corporate filings dated May 21, 2026 (tracked by Forge/Notice, per Revenue Memo) show a ~$1.05B Series C at ~$11–11.2B post-money. PI never announced it; secondary sources only.] | Own |
| Google DeepMind | **Gemini Robotics 2** VLA + **ER 2** reasoning VLM (Gemini 3.5 Flash) + **On-Device 2** | ER 2 on Gemini API/AI Studio; VLAs early-access only | Alphabet; Apptronik partner/investor; Intrinsic folded into Google (Feb 2026) | Own; **sells ER as API** |
| NVIDIA | GR00T N1.7 (3B; Cosmos-Reason2-2B backbone + DiT flow head); **GR00T N2** = DreamZero WAM, preview, "end of 2026"; Cosmos; Isaac Sim/Lab; Jetson Thor | ~~N1.7 weights Apache-2.0~~ [corrected: the code is Apache-2.0. The README's "License" section puts model weights under the NVIDIA Open Model License. The required backbone, `nvidia/Cosmos-Reason2-2B`, is gated and licensed `nvidia-open-model-license` (HF API). This contradicts the README intro line "fully commercially licensable under Apache 2.0".] (backbone gated on HF); Isaac ROS 5.0 agentic | Acquiring Hugging Face ($12.93B) | Own, open-weight |
| Figure | Helix 02: S2 (semantic) → S1 (200 Hz visuomotor) → S0 (1 kHz, 10M params); Helix 2.5 pretrained from scratch on "Index" | Closed; F.03 at BMW; Catalyst Brands | $1B+ Series C at **$39B** post (Sep 2025) | Own (no LLM named) |
| 1X | Redwood VLA; **1X World Model** (video model + inverse dynamics) as "cognitive core" | NEO $20k / $499 per month | SoftBank in talks for majority stake at ~**$6B** (Aug 2026) | Own; "off-board language" model unnamed |
| Tesla Optimus | In-house; production "preparations" | Not for sale | Public (TSLA) | Own |
| Skild AI | "Omni-bodied" Skild Brain; **S1** in-context learner (Aug 2026); bought Zebra's robotics unit (Apr 2026) | Closed; partners ABB, UR, NVIDIA, Foxconn | $1.4B Series C at **>$14B** (Jan 2026) | Own |
| Generalist AI | GEN-0 → GEN-1 (500k+ h wearable data) → **GEN-1.5** ("physical prompting", 100 Hz) | Early-access partners only | $400M at $2B (Jun 2026); +~$200M at **$3B** (Aug 2026) | Own |
| Dyna Robotics | DYNA-1 (Qwen3-VL-4B VLA, deployed) → **DYNA-2** WAM (1M h human video) | Robot cells only, no weights/API | $120M Series A (Sep 2025), >$600M ~~(UNVERIFIED)~~ [corrected: confirmed by a secondary source. Bloomberg (Sep 15, 2025) reported the valuation "tops $600 million", via Investing.com and Dealroom. No 2026 round found.] | Own |
| FieldAI | "Physics-first" Field Foundation Models, risk-aware navigation | Licensing/industrial deployments | $405M at ~$2B (Aug 2025) | Own; explicitly rejects "shoehorning" LLMs |
| Covariant → Amazon | RFM-1 (8B, 2024); license + founders to Amazon (2024) | — | Covariant $222M raised (2024, UNVERIFIED) | Own |
| Agility | Digit + **Arc** fleet cloud; "proprietary physical AI platform"; partners DeepMind, NVIDIA | 9 customer sites, 65k+ op-hours | SPAC with Churchill XI, $2.5B pre-money, ~~>$620M proceeds~~ [corrected: >$620M *expected* gross proceeds, assuming no redemptions: $420M from the trust plus ~$200M PIPE at $10/share. The deal had not closed as of the Jun 24, 2026 release.] (Jun 2026) | Own + partners |
| Apptronik | Apollo / Apollo 2; **runs Gemini Robotics** | ~~Pilots: Mercedes, GXO, Jabil~~ [corrected: the Feb 11, 2026 release calls Mercedes-Benz, GXO Logistics and Jabil "partnerships". It mentions "commercial and pilot deployments" in general but does not say which of these are pilots.] | $520M extension; Series A >$935M; "3×" prior valuation (~$5B per press) | **Google's models** |
| Sanctuary AI | Pivot (Jun 2026): Carbon AI on existing industrial arms | 99.5%+ wire-plug PoC, 2.54 s cycle | ~$149M total (UNVERIFIED) | Own |
| Unitree | Hardware leader; open UnifoLM VLA/WMA repos | Go2, G1, R1 | STAR IPO ¥150.8/share, ~$900M raised (Aug 2026) | Own + third-party |
| AgiBot | **GO-2** asynchronous dual-system (Apr 2026); AgiBot World dataset | LIBERO 98.5% (self-reported) | HK IPO process started (Jul 2026) | Own |
| Galbot | GraspVLA etc.; **authored the GPT-6 Astra-as-policy study** | Pharmacy/retail deployments | ¥2.5B (~$362M), Mar 2026; $3B post in Dec 2025 | Own + LLM-hybrid research |
| X Square Robot | WALL-OSS (open, Apache) → WALL-A/WALL-B | 58.com home trials | $276M Series B (Apr 2026); ">$2.8B" (headline only) | Own |
| OpenAI | Robotics division (May 2026); GPT-6 Astra | No robot product | — | Own LLM |
| Anthropic | Research (Fetch, Embody, Pilot) + **MHS** | MHS research preview | — | Own LLM |

---

## 2. Robot-foundation-model companies (the incumbents to beat or ride)

### 2.1 Physical Intelligence (π)

**π0.7** (arXiv 2604.15483, Apr 16, 2026):

- **Architecture:** a VLM backbone "initialized from the Gemma3 4B-parameter VLM", plus a "flow matching action expert with 860M parameters"; "about 5B total parameters".
- **Diverse context conditioning:** the prompt carries a subtask instruction, **subgoal images** generated by a BAGEL-based world model, and **episode metadata** (quality and speed).
- **Inference:** "language commands are produced by a high-level semantic policy based on the same architecture". Real-time chunking trains with simulated delays of up to 240 ms on a 50 Hz robot.
- **Verbal coaching:** π0.7 can be coached step by step in language on unseen appliance tasks (air fryer, toaster). The coaching transcripts are then distilled into the high-level policy.

**Implication for Ilia:** PI's own architecture has a **language-subtask slot that a frontier LLM could fill**. But PI fills it with a fine-tuned π0.7 head, not an API model. The 5% → 95% air-fryer improvement after coaching is press-reported only (UNVERIFIED).

**π\*0.6 / RECAP** (arXiv 2511.14759): advantage-conditioned offline RL plus interventions. It "more than doubles task throughput and roughly halves the task failure rate".

**Business model.** "The Physical Intelligence Layer" post (Feb 24, 2026) explicitly pitches π models as the robotics equivalent of an LLM API layer, through partners:
- **Weave:** laundry folding. Pretraining on Weave's data cut missed grasps 42% and interventions 50%.
- **Ultra:** e-commerce packing at 96.4% autonomy over a shift.

**Openness.** `Physical-Intelligence/openpi` (Apache-2.0, 14.1k★, pushed 2026-08-24) ships π0, π0-FAST and π0.5 base and expert checkpoints. It is the most realistic **"light action head"** for Ilia's design.

**Funding.**
- $600M led by CapitalG at ~$5.6B (Nov 2025; per Wikipedia's citations of The Robot Report and Bloomberg).
- ~$1B at >$11B in talks (Bloomberg via The AI Insider, Mar 30, 2026).

### 2.2 Google DeepMind: Gemini Robotics 2 / ER 2 / On-Device 2

DeepMind blog, Jul 30, 2026.

**Three models:**
- **Gemini Robotics 2 (VLA):** "controlling full humanoids, from feet to fingertips".
- **ER 2:** a VLM "that acts as our agent… plan multi-step tasks lasting several minutes… robots… work together as a team".
- **On-Device 2:** "fast adaptation to completely new robot embodiments with a few hours of data".

**Demonstrated embodiments:** one checkpoint ran on Apptronik Apollo 2 (SharpaWave 22-DoF hands and Inspire hands) and on a Franka Duo. The blog itself says "multi-finger dexterous manipulation remains challenging."

**Access:**
- ER 2 is on Google AI Studio, and in private preview on Gemini Enterprise Agent Platform.
- The VLAs are "available to early-access partners".

**API facts** (ai.google.dev robotics overview and pricing page):
- **Model IDs:** `gemini-robotics-er-2-preview`, and `gemini-robotics-er-2-streaming-preview` (Live API, bidirectional, function calling, no code execution).
- **Base model and limits:** ER 2 is built on Gemini 3.5 Flash; 131,072 input / 65,536 output tokens.
- **Point output:** `[y, x]` normalized to 0–1000.
- **Thinking:** `thinking_level` is recommended at "medium" as a latency/performance balance.
- **ER 1.6 retirement:** ER 1.6 "will be shut down at the end of August".
- **Pricing:** a **free tier**, plus paid **$1.00 input / $5.00 output per M tokens through Dec 31, 2026**, rising to $2 / $10 from Jan 1, 2027.

**Ecosystem:**
- **Apptronik:** strategic partner and investor.
- **Intrinsic:** joined Google (Feb 25, 2026, TechCrunch) and "will tap into Google's Gemini AI models".

**Implication for Ilia:** Google is the only incumbent that sells **exactly the System-2-over-API product category**, vertically integrated with its own VLA, at roughly 1/10 of Claude Fable 5.1 or GPT-6 Astra prices.

### 2.3 NVIDIA: GR00T, Cosmos, Isaac, and now Hugging Face

**GR00T N1.7.** `NVIDIA/Isaac-GR00T` README, 8.2k★:
- General Availability release, "fully commercially licensable under Apache 2.0".
- **Architecture:** VLM backbone `nvidia/Cosmos-Reason2-2B` (Qwen3-VL; **gated** on HF), plus a flow-matching DiT action head (16 layers, down from 32).
- **Action space:** relative end-effector actions.
- **Pretraining data:** 20k h of EgoScale human video.
- **Deployment:** ONNX/TensorRT export. Base checkpoint `nvidia/GR00T-N1.7-3B`, also available via the LeRobot `groot` policy type.

**GR00T N2.** NVIDIA's GTC (Mar 16, 2026) release says it is a WAM based on DreamZero research, "more than twice as often" successful on new tasks, and "slated to be available by the end of the year". I could not open the NVIDIA newsroom; TrendForce (Mar 19) repeats the "end 2026" timeline.

**DreamZero** (arXiv 2602.15922) is a 14B autoregressive video-diffusion WAM:
- closed-loop control at 7 Hz;
- adaptation to a new embodiment from 30 min of play data;
- **1750 Elo** on RoboArena (Apr 2026), vs 1622 for π0.5 (NVIDIA dev blog, Jun 15, 2026).

**Isaac ROS 5.0** (ROSCon, Sep 22, 2026) adds "agentic workflows": agent-ready docs and skills that "help humans and AI agents build robots together". These tools also go through the open-source AgenticROS project.

**Hugging Face acquisition.** NVIDIA agreed to buy Hugging Face for **$12,930,300,000**. Jensen Huang's blog post (Sep 3, 2026) promises HF "will remain an open platform… NVIDIA compute will not be required".

**Implication for Ilia:** NVIDIA now spans weights (GR00T), the robot-learning library (LeRobot), the low-cost hardware brand (Pollen / Reachy Mini / Microduck), simulators, and edge compute. A harness built on LeRobot + SO-101 + GR00T lives inside NVIDIA's stack.

### 2.4 Figure, 1X, Tesla: closed humanoid stacks

**Figure Helix 02** (Jan 27, 2026):
- **S0:** a "10M‑parameter" whole-body controller at **1 kHz**, trained on ">1,000 hours" of retargeted human motion in ">200,000 parallel environments". It "replaces 109,504 lines of hand‑engineered C++".
- **S1:** joint targets at **200 Hz**.
- **S2:** semantic latent goals.
- **Demo:** a 4-minute dishwasher sequence of "61 loco-manipulation actions".

**Helix 2.5** (Sep 17, 2026):
- Pretrained "from random initialization entirely on Index", Figure's human-experience dataset.
- Zero-shot in **30 Bay Area homes**: 56% vs 9% from scratch. Trial counts are not stated.
- Figure says it has committed "$3.5B of compute".
- No external LLM is used.

**Funding:** $1B+ Series C at a $39B post-money valuation (Sep 16, 2025).

**1X:**
- Redwood is a VLA that controls "locomotion jointly with manipulation".
- The 1X World Model became NEO's "cognitive core" on Jan 12, 2026.
- An "off-board language" model infers intent; it is not named.
- NEO costs $20,000 early-access or $499/month (Robot Report).
- SoftBank was reported negotiating a majority stake at ~$6B (Reuters via Euronext, Aug 26–27, 2026), below 1X's 2025 target.

**Tesla.** The Q2-2026 10-Q only says Tesla is "capitalizing on our strengths in real-world AI data to advance the development of Optimus… as we make preparations and investments in large-scale production." Press production figures (hundreds per week) are UNVERIFIED.

### 2.5 Skild, Generalist, Dyna, FieldAI, Covariant: model-first startups

**Skild AI.**
- **Series C:** $1.4B led by SoftBank at **>$14B** (TechCrunch, Jan 14, 2026; CEO: >$2B raised in total).
- **2026 blog posts:** Zebra robotics acquisition (Apr 15), ABB/UR/NVIDIA partnership (Mar 19).
- **S1** (Aug 18, 2026) is an in-context learner. Its results:
  - On unseen tasks at 100k h of pretraining: **66% vs 9%** for a language-prompted VLA.
  - "One in-context demo was worth roughly 380 post-training demos."
  - Plant-potting went from one human egocentric video to autonomous execution in 11 minutes.
- No public release.

**Generalist AI.**
- **GEN-1** (Apr 2, 2026): trained on more than 500k hours of wearable data with no robot data in pretraining.
  - 99% vs 64% (GEN-0) average success, about 1 h of robot data per task.
  - Consecutive-success counts: e.g. 200 box folds, and more than 1,800 block packs.
- **GEN-1.5** (Aug 19, 2026): "physical prompting" with a single 3–12 s demonstration in a 30 s context window, outputting **100 Hz** actions.
  - **59% (±10%)** one-shot over 10 tasks.
  - **83%** after 10 gradient steps on 5 min of data.
  - Trial counts are not reported.
- **Funding:** $3B valuation after an ~$200M extension (TechCrunch, Aug 25, 2026).
- Early-access partners only.

**Dyna Robotics.**
- **DYNA-1:** a production VLA "initialized from Qwen3-VL-4B" (MarkTechPost). The press release claims "99+% success rate in 24 hours of non-stop operation".
- **DYNA-2** (Aug 10, 2026): a WAM pretrained on 1M h of egocentric human video, with reported power-law fits.
  - At unseen customer sites it met production criteria **87% vs 46%** for DYNA-1.
  - No weights or API; customers buy robot cells.
- **Funding:** $120M Series A (Sep 15, 2025).

**FieldAI.**
- **Funding:** $405M (Aug 2025; ~$2B per CNBC, UNVERIFIED).
- CEO Ali Agha said they avoided trying to "shoehorn large language and vision models into robotics" and built "intrinsically risk-aware architectures".

**Covariant → Amazon.**
- Amazon took a "non-exclusive license to Covariant's robotic foundation models" and hired Abbeel, Chen, Duan and ~25% of staff (2024).
- Covariant's 2024 RFM-1 (8B, next-token over text/image/video/action) was the early "LLM-for-robots" commercial bet.

### 2.6 Agility, Apptronik, Sanctuary: hardware-first, model-partnering

**Agility** (Jun 24, 2026):
- SPAC with Churchill Capital Corp XI at a $2.5B pre-money valuation, >$620M gross proceeds, ticker AGLT.
- **Deployments:** 9 customer facilities, >65,000 operating hours, >$300M "multi-year contracted Digit v5 orders".
- **Partners:** Google DeepMind and NVIDIA. Arc is the fleet cloud.

**Apptronik** (Feb 11, 2026):
- $520M extension bringing the Series A above $935M.
- "Industry-leading strategic partnership with Google DeepMind… powered by Gemini Robotics".
- This is the clearest example of a humanoid OEM **outsourcing the brain** to a model provider.

**Sanctuary** (Robot Report, Jun 17, 2026):
- Pivoted to deploying its physical AI on "existing and next-generation industrial robots".
- PoC: >99.5% success at a 2.54 s cycle on a moving-conveyor wire-plugging task.

---

## 3. Chinese players

| Company | 2026 facts (cited) | Model / openness |
|---|---|---|
| **Unitree** | STAR Market IPO priced at ¥150.8/share; ~$900M raised; ~$50B valuation at the day-one morning close (Crowdfund Insider, Aug 19, 2026). The CNBC/SCMP "460% close" figure is UNVERIFIED (403). | Open repos: `unifolm-world-model-action` (1.2k★), `unifolm-vla` (0.6k★). Go2/G1 are the default hacker platforms (Anthropic Fetch/Embody, dimOS, OM1, HomeBody). |
| **AgiBot** | GO-2 released Apr 9, 2026 (Robot Report). LIBERO 98.5% average and LIBERO-Plus 86.6% zero-shot, both self-reported. HK IPO process started (TechNode, Jul 27, 2026). The HK$40–50B target is UNVERIFIED. | Asynchronous dual-system: a slow planner plus a fast action follower. AgiBot World dataset repo 3.2k★. |
| **Galbot** | ¥2.5B (~$362M) round (Caixin, Mar 2, 2026); a $3B post-money round in Dec 2025 per the same page. **Galbot Team authored "Systematically Exploring the Capabilities of GPT-6 Astra as Embodied Policies"** (arXiv 2609.38537). | GraspVLA family. Hybrid Astra+π0.5 reached 48% on a RoboDojo subset. Astra-assisted vs direct control on 50 RoboDojo instances: **624.8M vs 1.132B tokens**. A 30 s locomotion run needs "250 model calls averaging 39.86 seconds each". |
| **X Square Robot** | ~$276M Series B led by Xiaomi (The AI Insider, Apr 22, 2026). Series C ">$2.8B" (Pandaily headline only). | `X-Square-Robot/wall-x` (WALL-OSS, Apache-2.0, 1.3k★). WALL-B runs in home trials with 58.com. |

The Chinese ecosystem is model-first but cheap-hardware-first too. Galbot's Astra paper shows the large Chinese labs are actively testing **LLM-as-policy hybrids**. Their conclusion favours LLM + VLA, not LLM alone.

---

## 4. Frontier-LLM labs and their robot-relevant API surface

### 4.1 Anthropic

Anthropic's robotics output in 2026 consists of the following. None of it is a product.

- **Project Fetch phase 2** (Jun 18, 2026):
  - **Setup:** Claude Opus 4.7 in **Claude Code**, max-effort adaptive thinking, 3 runs.
  - **Speed:** it finished the four tasks both human teams completed in **9 min 35 s**, vs 181 min (humans + Opus 4.1) and 361 min (humans alone).
  - **Code size:** 1,045 lines vs 10,309.
  - **Failure:** autonomous fetching still failed for lack of "closed-loop error correction".
  - **Quote:** "This doesn't mean that LLMs have now solved robotics. Far from it."
- **"Claude plays robotics" / Embody** (Jul 9, 2026):
  - **Scope:** 12 models across 5 providers. Four interfaces: direct, programmatic, policy-supervision (Go2 gait policy; MolmoAct VLA on Franka/LIBERO), and RL-supervision.
  - **Direct LIBERO manipulation:** 0–**5.5%** success (Mythos Preview best).
  - **VLA supervision:** helped every model but stayed "substantially worse than MolmoAct does on its own".
  - **Vision aid:** a gripper-camera cursor raised Mythos from 6% to 32%.
  - **Latency:** locomotion needs ~83 Hz, while inference runs at 0.2–0.4 Hz. Turns take 2–8 s text-only, 5–15 s with images, and 15–60 s at high reasoning.
  - **Code:** promised at `github.com/safety-research/embody`. Not yet public; the repo 404s as of 2026-10-01.
- **Project Pilot** (Jul 24, 2026, with Andon Labs):
  - **Setup:** 15 models wrote code for a $129 DJI Tello. Fable 5 was best.
  - **Result:** Fable 5 passes the human-AI baseline on 4 of 5 subtasks; reconstruction lags (~47%).
  - **Failure:** "confidently flies a drone into what it thinks is a doorway but is actually a wall".
- **Model Hardware Standard** (Aug 27, 2026, research preview):
  - **Design:**
    - `read`/`write` primitives;
    - generated device manifests "what it can measure, what can be adjusted, and what safety limits will be enforced";
    - device-level limits;
    - control via "MCP, the command line interface, and code files".
  - **Partners:** HHMI Janelia, Genentech, CMU, QuEra, Tetsuwan, UW. Vendors include Universal Robots, Doosan and AWS Strands Robots, plus **Hugging Face LeRobot** as an early adopter.
  - **Fault tests:** CMU induced 6 faults, and MHS "correctly blocked all six before any device moved".
  - **QuEra laser relock:** 58% → 99.3% (695/700).
  - **Status:** not open-source yet (waitlist at modelhardwarestandard.com).

**Current Claude API** (platform docs, 2026-10-01):

| Model | ID | Price in/out per M | Notes |
|---|---|---|---|
| Fable 5.1 | `claude-fable-5-1` | $10 / $50 (cache read $0.25) | Released Sep 1, 2026; "Slower"; adaptive thinking always on; default effort `high` |
| Opus 5.5 | `claude-opus-5-5` | $4 / $20 (cache read $0.20) | Released Sep 22, 2026; "Moderate" latency; default effort `medium`; Fast mode (research preview) |
| Sonnet 5.5 | `claude-sonnet-5-5` | $2 / $10 | "Fast" |
| Haiku 4.5 | `claude-haiku-4-5-20251001` | $1 / $5 | "Fastest", 200K context |

Breaking changes on Opus 5.5 that matter for a harness:
- "thinking can't be disabled";
- "forced tool use returns an error" ~~(also on Fable 5.1)~~ [corrected: the docs say the first three breaking changes "also apply on Claude Fable 5.1". Those three are: thinking can't be disabled, forced tool use errors, and thinking blocks are tied to the model and the conversation.];
- `computer_20251124` is not accepted [clarified: only "on the Claude API and Google Cloud"];
- ~~text between tool calls returns inside `thinking` blocks unless `display` is set~~ [corrected: the docs class this as a non-breaking "further change" in response shape. Text between tool calls comes back in thinking blocks whose text is empty at the default display setting. The fourth *breaking* change is "thinking blocks are tied to the model and the conversation".].

A harness that relies on `tool_choice: any` to force a single action per turn must change.

### 4.2 OpenAI

- **OpenAI Robotics division** (Humanoids Daily, May 31, 2026):
  - "Our world simulation research program… has evolved over the past year into OpenAI Robotics."
  - 11 SF roles, including actuator design, sim realism and data-acquisition ops.
  - Near-term goal: robots for skilled infrastructure workers.
- **GPT-6 Astra:**
  - Announced Sep 3, 2026 (OpenAI developer-community post): $10 in / $1 cached / $50 out per M, 1,050,000 context window, 128k max output. API-live tweet Sep 4.
  - OpenAI's launch never mentions robots.
- **Third-party robot evidence for Astra:**
  - RoboDojo: 22.48% success over 2,100 trials, above every public entry at the time (arXiv 2609.24170).
  - Galbot study (above).
  - Robocurve: 19/20 on YAM block-into-bowl (see `sources/robocurve-gpt6-astra.md`).
  - HomeBody (Stanford/Caltech): Astra calls modular skills on a Unitree G1; code is scheduled for Oct 4–18, 2026.
  - Understanding Robots (Oct 1, 2026) notes Astra "now ranks seventh" on RoboDojo, behind an LLM-agent-plus-π0.5 entry. Real-world RoboDojo testing "stopped after Astra damaged hardware".

OpenAI is now both the best API "policy" model and a future vertically-integrated robot competitor.

---

## 5. The LLM-agent / harness software layer (direct competitors)

| Project / company | What it is | LLM wiring | Openness / traction |
|---|---|---|---|
| **Dimensional (dimOS)** | Python, ROS-free "agentic OS for physical space". Modules over LCM/ROS2/DDS, "blueprints", `@skill` methods exposed as LangChain tools **and** MCP (`dimos mcp call …`). Go2 (stable), G1/xArm/**PiPER** (beta), drones (alpha). | LangGraph agent; **default `gpt-5.6-luna`**; Ollama option | 4.6k★, pushed 2026-10-01, custom license |
| **Innate (innate-os, MARS)** | ROS 2 agentic OS for the MARS robot ($995 per teammate note). VLM picks skills; ACT policies at 25 Hz on-device. | Default `google:gemini-3.6-flash`; PR #817 GPT-6 Astra imitation | Apache-2.0, 81★ (see `sources/innate-os-pr817.md`) |
| **OpenMind OM1** | Hardware-agnostic robot "OS"; Unitree G1/Go2, TurtleBot, UBTech agents | "Plug-and-play" OpenAI, Gemini, DeepSeek, xAI (Robot Report, Sep 18, 2025) | MIT, 2.9k★; raised $20M (Aug 2025) |
| **AWS Strands Robots** | `Robot("so100")` as an agent tool, sim→`mode="real"`. Zenoh / AWS IoT / **MHS** mesh backends. LeRobot, GR00T, MolmoAct2, π0/π0.5 policies. | Dec 2025 reference design: **Claude Sonnet 4.5 on Bedrock (cloud) + Qwen3-VL-2B (edge) + GR00T on SO-101**; the VLA is wrapped as `@tool execute_manipulation` | Apache-2.0, 171★; HF blog (Jun 17, 2026) |
| **General Robotics (GRID)** | "Auto-Engineering": LLM as robotics *engineer* (ingest URDF → sim → skill method → preflight) | LLM undisclosed (GPT-family hints) | Closed; see `sources/general-robotics-auto-engineering.md` |
| **Mbodi AI** | YC X25; multi-agent orchestration of small task models for industrial picking; ABB partnership | Models not named | Closed |
| **Intrinsic (Google)** | Flowstate skills IDE; now inside Google, using Gemini | Gemini | Closed platform plus open tools |
| **ROSA (NASA JPL)** / **ros-mcp-server** / **RAI (Robotec)** | ROS↔LLM agents / MCP bridges | Any (Claude, GPT, Gemini) | 1.6k★ / 1.5k★ / 0.6k★, Apache-2.0 |
| **NVIDIA Isaac ROS 5.0** | Agent-ready docs/skills; AgenticROS; Nemotron/NemoClaw | NVIDIA models | Open source |
| **Show-Harness** (NUS Show Lab, arXiv 2609.10522) | "Discrete semantic action units" grounded by deterministic per-embodiment interpreters; frontier VLMs zero-shot; small VLMs fine-tuned in "a few GPU-hours"; PiPER/Franka | Any VLM | Open (academic) |
| **Hilstart** (YC S26) | "Physical agent harnesses to enable AI to debug, validate, and ship functional hardware" | — | Closed, early |

**Evaluators and hacker-harness projects** (teammate deep-dives):

- **Robocurve.** A PBC that calls itself "METR/Epoch for robots"; $10M seed led by Initialized (Sep 14, 2026). Its `inspect-robots` harness (MIT, 631★) has been downloaded "more than 97k times". It claims output-token speed improves "about 2.1x per month".
- **Manda Robotics.** Evaluation infrastructure, about two people, no funding found.
- **RobotKitAI.** `piper-astra-jev` (11★): "LLM-driven demo runs on a real AgileX PiPER arm: Astra, Jev+DINO, Jev+SAM3".
- **MakerMods.** `metal-arm-harness` (15★), and a fork of inspect-robots.
- **RoboDojo** (HKU MMLab). Its #1 entry is "PhysicalRSI", tagged AGENT+VLA (`sources/robodojo.md`).

**Reading:** the harness layer already has:
- multiple well-starred open-source frameworks;
- two hyperscaler-backed ones (AWS Strands; Google ER 2 + Intrinsic);
- an emerging device-driver standard from Anthropic itself (MHS).

A new harness will not win on "LLM can call robot skills via MCP". That is now table stakes.

---

## 6. Low-cost hardware used by LLM-harness hackers (prices as opened on 2026-10-01)

| Platform | Price (source) | DoF / notes | Used by |
|---|---|---|---|
| **SO-101** (TheRobotStudio / Seeed) | Seeed Pro motor kit (leader + follower servos, no prints) **$249.90**; printed parts $29.90. DIY BOM total **$229.88** (SO-ARM100 README). Search snippets showed $277.99, so the price fluctuates. | 6-DoF incl. gripper; STS3215 serial servos; 7.6k★ repo | LeRobot default; Strands; so101-painting; Show-Harness-style demos |
| **Koch v1.1** | BOM **$199 leader + $278 follower** (jess-moss/koch-v1-1 README) | Dynamixel XL330/XL430 | Early LeRobot |
| **AgileX PiPER** | **$1,999** (AgileX global store); US resellers $2.8–4k (UNVERIFIED) | 6-DoF, 1.5 kg, 626 mm, 0.1 mm, 4.2 kg; CAN; `piper_sdk` MIT | RobotKitAI, RoboDojo real, dimOS, Show-Harness |
| **I2RT YAM** | Standard **$2,999**, Pro $3,499, Ultra $4,299, BIG YAM $4,999, Leader $2,999 (doc.i2rt.com) | 6-DoF, DM4340/DM4310 on 1 Mbit CAN, 400 ms motor timeout, MIT SDK, MuJoCo | Robocurve (Astra/Fable/RoboHarm), Dyna-2 eval, Manda |
| **MakerMods Metal Arm** | **$2,499 + $249 shipping**, ships Sep 2026 | 7-DoF, 3 kg, ±0.1 mm; ROS1/2, ACT/π0/π0.5 pipelines; no LLM mention on product page | MakerMods harness |
| **XLeRobot** | **~$660** basic BOM (README; excludes printing/tax); 5.6k★ | Two SO-101 arms + LeKiwi base + IKEA cart | Household mobile manipulation |
| **AlohaMini2** | Self-build BOM **< $1,000** (README, 2026-06-06) | Dual-arm, motorized lift, 6+1 DoF, 52 cm reach | Mobile ALOHA-style |
| **Open Duck Mini** | BOM **< $400** (README) | BDX-style biped, RL gait | Hobby RL |
| **Microduck** (Pollen/HF) | **$399**, pre-orders opened Aug 27, 2026, ships before Christmas (MarkTechPost) | 25 cm, 15 motors, RK3566. Software Apache-2.0; **hardware not open** | — |
| **Reachy Mini** | **$399 Lite / $499 Wireless** (Pollen page; ~90-day lead time) | Expressive head; "realtime voice model" conversation; 60+ apps | LLM voice agents |
| **ToddlerBot** (Stanford) | **< $6,000** (arXiv 2502.00893) | 30 DoF, Jetson Orin NX; GPT-4o realtime speech demo | Humanoid research |
| **Innate MARS** | $995 (per teammate note citing innate.bot) | Jetson Orin Nano 8GB, 5+1 DoF arm, lidar | innate-os |

**Pattern.** The hacker LLM-harness scene has converged on three classes:
- serial-servo SO-101 (≈$250–300 per pair);
- CAN-bus 6-DoF arms around $2–3k (PiPER, YAM, Metal Arm);
- Unitree Go2/G1 for mobility.

Backlash, compliance and calibration differ a lot across these. An embodiment-agnostic harness must normalise them; this is where MHS-style manifests and limits matter.

---

## 7. Where an "API-LLM robot harness" fits, and the main threats

### 7.1 Structural facts the design must respect

1. **Latency and cost.**
   - Anthropic measured 0.2–0.4 Hz model turns against an ~83 Hz locomotion need.
   - Galbot measured ~~39.86 s per Astra call~~ [corrected: 39.86 s is the mean per call in a 30-second locomotion run (250 calls, physics paused during inference). It is not a general per-call latency.], and 624.8M–1.132B tokens per 50 RoboDojo instances (≈12–23M tokens per episode). At Astra's $10/M input that is roughly $125–225 per episode before caching.
   - Every incumbent runs **10–200 Hz learned control on-device**: Helix S1 200 Hz, GEN-1.5 100 Hz, π0.7 50 Hz robots with real-time chunking.
   - So the LLM can only be a slow System 2, critic or programmer. **The "light action head" is not optional.**
2. **The System-2 slot is being filled by the model vendors themselves**: Gemini ER 2 → GR2 VLA; π0.7's own high-level policy; Figure S2; AgiBot GO-2's planner.
3. **In-context task acquisition is moving into the action model**: GEN-1.5 physical prompting, Skild S1, π0.7 coaching. LLM-in-context imitation from demonstrations is the harness's most distinctive trick, and it now competes with purpose-built models that run at 100 Hz.

### 7.2 Where it fits (promising)

- **Long-tail embodiments and low-volume customers.** Many arms and grippers will never get a π/GR00T/Gemini fine-tune: SO-101, PiPER, YAM, Metal Arm, lab automation.
  - A harness that wraps any open VLA (openpi π0.5, GR00T N1.7, SmolVLA, MolmoAct2 via LeRobot) and puts a frontier LLM on top as planner, verifier and recovery agent is the AWS Strands pattern.
  - Third-party evidence that LLM+VLA beats either alone: RoboDojo #1 is an AGENT+VLA entry; Galbot hybrid 48% vs direct; Anthropic's VLA-supervision gains on novel tasks.
- **Safety and governance as a product.**
  - Robocurve's RoboHarm shows frontier models execute harmful physical requests: Astra "completed 60/100" (teammate note).
  - Anthropic's MHS puts limits in the driver, not the prompt.
  - A harness with an MHS-compatible device layer, an action governor, an audit log and stale-observation checks is differentiated. Innate PR #817 and RobotKitAI's "governor" already hint at this.
- **LLM as data/skill engine for the light head.** Use Opus/Astra offline to:
  - write controllers (Project Fetch: 9.5 min vs hours);
  - generate sim environments and trajectories, then fine-tune a small action head;
  - auto-engineer deployments (the GRID pattern).

  This sidesteps the latency problem and gets cheaper with every model generation.
- **Provider-agnostic routing.**
  - Prices span 10×: Gemini ER 2 at $1/$5 with a free tier; Opus 5.5 at $4/$20; Fable 5.1 / Astra at $10/$50.
  - Capabilities are polarised: Astra is strong at semantic tasks but weak at precision; ER 2 has native pointing/trajectories in a 0–1000 frame.
  - A harness that benchmarks and routes per step (e.g. ER 2 for pointing, Opus 5.5 for planning and recovery, Haiku/Sonnet for monitoring) has a cost moat that single-vendor stacks lack.

### 7.3 Unpromising directions

- LLM direct joint or torque control for contact-rich or dynamic tasks. Anthropic: LIBERO direct 0–5.5%; "no model successfully stood the [G1] up". Galbot: dense locomotion "remains unreliable".
- Betting on one closed model's tool semantics. For example, Opus 5.5 and Fable 5.1 reject forced `tool_choice`, and thinking cannot be disabled on Opus 5.5.
- A generic ROS/MCP bridge or "robot OS" with no data flywheel. dimOS, OM1, ROSA, ros-mcp-server, RAI, Strands and Isaac ROS 5.0 already exist, several with hyperscaler backing.
- Humanoid-first products. This is the most capital-intensive arena: Figure $39B, Tesla, 1X, Apptronik + Google, Agility SPAC, Unitree, AgiBot.

### 7.4 Main threats (ranked)

1. **Google.** ER 2 API (cheap, free tier, multi-robot orchestration, Live API streaming) + GR2/On-Device 2 VLAs + Intrinsic/Flowstate + Apptronik. It is the integrated version of Ilia's idea.
2. **NVIDIA + Hugging Face.** It controls the open stack a small harness would build on: LeRobot, SO-101/Reachy/Microduck, GR00T N1.7 ~~Apache~~ [corrected: code Apache-2.0, weights under the NVIDIA Open Model License; see Verification], Cosmos-Reason2 (gated), Isaac ROS 5.0 agentic tooling.
3. **Robot-FM in-context learning** (GEN-1.5, Skild S1, π0.7 coaching). It erodes the "teach by demonstration through the LLM" story at 100 Hz.
4. **Frontier labs going vertical.** OpenAI Robotics (hardware plus Astra). Anthropic MHS, which makes the device layer a commodity standard that any harness, and Anthropic's own agents, can use.
5. **Commoditised open-source harnesses**, plus AWS Strands as a free reference architecture already pairing Claude with GR00T on SO-101.
6. **Liability and regulation.** Harmful-instruction compliance (RoboHarm) and hardware damage (RoboDojo stopped real testing). The EU Machinery Regulation (2023/1230) applies from 20 Jan 2027; this is background knowledge, not re-verified here.

**Bottom line.** The defensible position is not "Claude drives the robot". It is:
- an **embodiment-agnostic reliability layer**: verifier, recovery, safety envelope, eval harness;
- a **data engine** that turns frontier-LLM intelligence into cheap fine-tuned action heads for long-tail hardware;
- **model-agnostic routing**, with MHS/MCP compatibility from day one.

---

## Sources

All accessed 2026-10-01.

**Physical Intelligence**
- π0.7 paper: https://arxiv.org/abs/2604.15483 (HTML https://arxiv.org/html/2604.15483v2)
- π0.7 blog: https://www.pi.website/blog/pi07 (rendered)
- Partner post: https://www.pi.website/blog/partner (rendered)
- π\*0.6: https://arxiv.org/abs/2511.14759
- openpi: https://github.com/Physical-Intelligence/openpi
- Funding: https://theaiinsider.tech/2026/03/30/report-physical-intelligence-to-raise-1b-with-valuation-north-of-11b/ and https://en.wikipedia.org/wiki/Physical_Intelligence_Inc.

**Google / DeepMind**
- Gemini Robotics 2 blog: https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/
- Gemini Robotics 2 coverage: https://siliconangle.com/2026/07/30/google-deepmind-debuts-gemini-robotics-2-model-series-humanoid-robots/
- Robotics API overview: https://ai.google.dev/gemini-api/docs/robotics-overview
- Pricing: https://ai.google.dev/gemini-api/docs/pricing
- Intrinsic joins Google: https://techcrunch.com/2026/02/25/alphabet-owned-robotics-software-company-intrinsic-joins-google/

**NVIDIA / Hugging Face**
- GR00T repo: https://github.com/NVIDIA/Isaac-GR00T
- DreamZero: https://arxiv.org/abs/2602.15922
- WAM dev blog: https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/
- GTC coverage: https://www.trendforce.com/news/2026/03/19/insights-nvidia-expands-robotics-ecosystem-at-gtc-as-physical-ai-moves-toward-large-scale-deployment/
- Isaac ROS 5.0: https://www.roboticstomorrow.com/news/2026/09/22/nvidia-isaac-ros-50-advances-agentic-open-source-robotics-development/27137
- HF acquisition: https://blogs.nvidia.com/blog/nvidia-to-acquire-hugging-face/
- LeRobot v0.6.0: https://huggingface.co/blog/lerobot-release-v060

**Figure**
- News index: https://www.figure.ai/news
- Helix 02: https://www.figure.ai/news/helix-02
- Helix 2.5: https://www.figure.ai/news/helix-2-5-zero-shot-30-home-generalization
- Series C: https://www.figure.ai/news/series-c

**1X**
- AI page: https://www.1x.tech/ai
- World model: https://www.therobotreport.com/1x-launches-world-model-enabling-neo-robot-to-learn-tasks-by-watching-videos/
- SoftBank talks: https://live.euronext.com/en/financial-news/softbank-talks-buy-stake-1x-6-billion-valuation-information-reports

**Tesla**
- Q2 2026 10-Q: https://www.sec.gov/Archives/edgar/data/0001318605/000162828026049270/tsla-20260630.htm

**Skild AI**
- Series C: https://techcrunch.com/2026/01/14/robotic-software-maker-skild-ai-hits-14b-valuation/
- Blog index: https://www.skild.ai/blogs
- S1: https://www.skild.ai/blogs/s1

**Generalist AI**
- GEN-1: https://generalistai.com/blog/gen-1
- GEN-1.5: https://generalistai.com/blog/gen-1.5
- $3B valuation: https://techcrunch.com/2026/08/25/robotics-startup-generalist-reaches-3b-valuation-sources-say/

**Dyna Robotics**
- DYNA-2: https://www.marktechpost.com/2026/08/13/dyna-robotics-introduces-dyna-2-a-world-action-model-pre-trained-on-1-million-hours-of-human-video/
- Series A: https://www.prnewswire.com/news-releases/dyna-robotics-raises-120-million-to-advance-robotic-foundation-models-on-the-path-to-physical-artificial-general-intelligence-302556817.html

**FieldAI**
- https://www.therobotreport.com/fieldai-raises-405m-scales-physics-first-foundation-models-robots/

**Covariant / Amazon**
- https://www.aboutamazon.com/news/company-news/amazon-covariant-ai-robots

**Agility**
- https://www.agilityrobotics.com/content/agility-robotics-to-go-public-through-merger-with-churchill-capital-corp-xi

**Apptronik**
- https://www.globenewswire.com/news-release/2026/02/11/3236352/0/en/Apptronik-Closes-Over-935-Million-Series-A-with-New-520-Million-Extension-Round.html

**Sanctuary AI**
- https://www.therobotreport.com/sanctuary-ai-validates-physical-ai-performance-tier-1-automotive-supplier/

**Unitree**
- https://www.crowdfundinsider.com/2026/08/298797-unitree-robotics-delivers-solid-shanghai-market-ipo-debut/

**AgiBot**
- GO-2: https://www.therobotreport.com/agibot-releases-go-2-foundation-model-embodied-ai/
- IPO process: https://technode.com/2026/07/27/agibot-starts-hong-kong-ipo-process/

**Galbot**
- Funding: https://www.caixinglobal.com/2026-03-03/galbot-raises-362-million-in-fresh-funding-eyes-hong-kong-ipo-102418742.html
- Astra-as-policy paper: https://arxiv.org/abs/2609.38537

**X Square Robot**
- Series B: https://theaiinsider.tech/2026/04/22/x-square-robot-raises-276m-in-series-b-funding-for-household-robots/
- Series C headline: https://pandaily.com/x-square-robot-2-8b-valuation-series-c-jun2026
- WALL-OSS: https://github.com/X-Square-Robot/wall-x

**Anthropic**
- Claude plays robotics: https://www.anthropic.com/research/claude-plays-robotics
- Project Fetch phase two: https://www.anthropic.com/research/project-fetch-phase-two
- Project Pilot: https://www.anthropic.com/research/project-pilot
- Frontier Red Team index: https://www.anthropic.com/research/team/frontier-red-team
- Model Hardware Standard: https://www.anthropic.com/news/model-hardware-standard-research-preview
- Models overview: https://platform.claude.com/docs/en/about-claude/models/overview
- Fable 5.1: https://platform.claude.com/docs/en/models/fable-5-1/overview
- Opus 5.5: https://platform.claude.com/docs/en/models/opus-5-5/overview

**OpenAI and GPT-6 Astra evidence**
- OpenAI Robotics division: https://www.humanoidsdaily.com/news/openai-pivots-directly-into-hardware-launching-internal-robotics-division
- Astra announcement: https://community.openai.com/t/introducing-gpt-6-astra-the-most-intelligent-and-aligned-model-in-the-world/1394703
- Understanding Robots on Astra: https://www.understandingrobots.org/p/openais-astra-model-is-shockingly
- RoboDojo Astra eval: https://arxiv.org/abs/2609.24170
- HomeBody coverage: https://the-decoder.com/researchers-plug-gpt-6-astra-directly-into-a-robot-and-let-it-clean-up-an-unfamiliar-kitchen/
- HomeBody code: https://github.com/Stanford-TML/homebody

**Harness layer**
- dimOS: https://github.com/dimensionalOS/dimos
- innate-os: https://github.com/innate-inc/innate-os
- OM1 launch: https://www.therobotreport.com/openmind-launches-om1-open-source-robot-agnostic-operating-system/
- OM1 repo: https://github.com/OpenMind/OM1
- Strands Robots: https://github.com/strands-labs/robots
- Strands + MHS: https://strandsagents.com/blog/robots-working-together-model-hardware-standard-strands-robots/
- AWS physical-AI reference design: https://aws.amazon.com/blogs/opensource/building-intelligent-physical-ai-from-edge-to-cloud-with-strands-agents-bedrock-agentcore-claude-4-5-nvidia-gr00t-and-hugging-face-lerobot
- Strands + LeRobot on HF: https://huggingface.co/blog/amazon/strands-lerobot-hub-to-hardware
- ROSA: https://github.com/nasa-jpl/rosa
- ros-mcp-server: https://github.com/robotmcp/ros-mcp-server
- RAI: https://github.com/RobotecAI/rai
- Show-Harness: https://arxiv.org/abs/2609.10522
- Mbodi (Robot Report): https://www.therobotreport.com/mbodi-ai-launches-y-combinator-developing-embodied-ai-industrial-robots/
- Mbodi (TechCrunch): https://techcrunch.com/2025/10/27/mbodi-will-show-how-it-can-train-a-robot-using-ai-agents-at-techcrunch-disrupt-2025/
- Hilstart: https://www.ycombinator.com/companies/hilstart
- Robocurve seed: https://robocurve.org/blog/seed-raise/
- inspect-robots: https://github.com/robocurve/inspect-robots
- piper-astra-jev: https://github.com/RobotKitAI/piper-astra-jev
- metal-arm-harness: https://github.com/makermods-robotics/metal-arm-harness

**Hardware**
- SO-101 (Seeed): https://www.seeedstudio.com/SO-101-Low-Cost-AI-Arm-Kit-Pro-p-6427.html
- SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100
- Koch v1.1: https://github.com/jess-moss/koch-v1-1
- AgileX PiPER: https://global.agilex.ai/products/piper
- I2RT YAM: https://doc.i2rt.com/products/yam
- MakerMods Metal Arm: https://www.makermods.ai/metal-arm
- XLeRobot: https://github.com/Vector-Wangel/XLeRobot
- AlohaMini: https://github.com/liyiteng/AlohaMini
- Open Duck Mini: https://github.com/apirrone/Open_Duck_Mini
- Microduck: https://www.marktechpost.com/2026/08/28/pollen-robotics-hugging-face-microduck-399-open-source-rl-biped-robot/
- Reachy Mini: https://pollen-robotics.com/reachy-mini/
- ToddlerBot paper: https://arxiv.org/abs/2502.00893
- ToddlerBot site: https://toddlerbot.github.io/
- Unitree UnifoLM: https://github.com/unitreerobotics/unifolm-world-model-action
- AgiBot World: https://github.com/OpenDriveLab/AgiBot-World

---

## Verification (fact-check pass)

*An adversarial fact-check run on 2026-10-01/02 by a second agent. It re-opened primary pages with curl or WebFetch, and used `gh api` plus the local clones in `repos/` for code. A claim counts as **confirmed** only if the text below was seen on the source page or in the code. Disk was nearly full during this pass, so HTML was converted to text on the fly and not kept.*

**Overall:** the note is accurate. No entry looks fabricated or mis-attributed. Every 2025–2026 page it cites resolved, and the papers and repos exist with the stated authors and dates. The errors are in licensing, deal status and scoping: the GR00T weights licence, PI's round status, the Opus 5.5 breaking-change list, the scope of Galbot's latency number, Agility's proceeds, and the claim that "all train their own models". The note also misses the Boston Dynamics–DeepMind deal, Astra Ultrafast, Skild's ARR, and Haiku 4.5's near-term retirement date.

### A. Confirmed claims (with evidence)

1. **NVIDIA → Hugging Face.**
   - "NVIDIA has agreed to acquire Hugging Face for $12,930,300,000" (Jensen Huang, blogs.nvidia.com, Sep 3, 2026).
   - Also: "NVIDIA compute will not be required to build on or deploy through Hugging Face."
   - Status is *agreed*. The page states no closing date or regulatory status.
2. **Gemini Robotics 2 / ER 2 / On-Device 2** (deepmind.google blog, Jul 30, 2026, Carolina Parada).
   - All quoted phrases match, including "from feet to fingertips", "acts as our agent… plan multi-step tasks lasting several minutes", and "a few hours of data".
   - Apollo 2 (SharpaWave 22-DoF and Inspire hands) and the Franka Duo run from one checkpoint.
   - ER 2 is on AI Studio and in private preview on Gemini Enterprise Agent Platform. The VLAs are limited to early-access partners.
3. **Gemini ER 2 API** (ai.google.dev robotics overview and pricing, via WebFetch; curl failed on TLS).
   - Model IDs: `gemini-robotics-er-2-preview` and `gemini-robotics-er-2-streaming-preview`.
   - "Builds on Gemini 3.5 Flash"; 131,072 input / 65,536 output tokens; `[y, x]` normalized 0–1000.
   - "For thinking level use medium…"
   - "Gemini Robotics ER 1.6 model will be shut down at the end of August".
   - Free tier; $1.00/$5.00 per M "through December 31, 2026", then $2.00/$10.00 "starting January 1, 2027".
4. **Figure.**
   - Series C: ">$1 billion… $39 billion" post-money, Sep 16, 2025 (figure.ai/news/series-c).
   - Helix 02 (Jan 27, 2026): S0 "10M-parameter" at 1 kHz, ">1,000 hours", ">200,000 parallel environments", "replaces 109,504 lines of hand-engineered C++". S1 runs at 200 Hz. The demo is 4 minutes with 61 actions.
   - Helix 2.5 (Sep 17, 2026): "pretrained from random initialization entirely on Index"; 30 homes; 9% vs 56%; "$3.5B of compute". No trial counts are given anywhere on the page, including the appendix rubric.
5. **Skild.**
   - TechCrunch (Jan 14, 2026): $1.4B Series C led by SoftBank at ">$14 billion"; CEO says ">$2 billion" raised to date.
   - S1 (skild.ai/blogs/s1, Aug 18, 2026):
     - At 100k h of pretraining, ICL reaches **66%** vs **9%** for a language-prompted VLA. This comes from a controlled study on a *filtered subset* of the pretraining pool.
     - "worth roughly 380 post-training examples".
     - Plant potting took "11 minutes" from demonstration to autonomous execution.
   - Blog index confirms Zebra (Apr 15, 2026) and ABB/UR/NVIDIA (Mar 19, 2026).
6. **Generalist.**
   - GEN-1 (Apr 2, 2026): 99% vs 64% (GEN-0) average success; "approximately one hour of robot data" per task; "over half a million hours"; "data from low-cost wearable devices"; ">1,800" block packs, ">200" box folds.
   - GEN-1.5 (Aug 19, 2026): "100 Hz action trajectories", "30 seconds of memory", a 3–12 s demo, "59% (±10% std. dev.)", and "83% (±9%)" after 10 gradient steps on 5 min (~50 demos). No trial counts.
   - TechCrunch (Aug 25, 2026): $3B valuation; "nearly $200 million" extension of a $400M Series B at $2B (June).
7. **Physical Intelligence.**
   - π0.7 is arXiv 2604.15483 (v1 Apr 16, 2026). Its HTML says:
     - "VLM backbone initialized from the Gemma3 4B-parameter VLM… action expert with 860M parameters… about 5B total";
     - "language commands are produced by a high-level semantic policy based on the same architecture";
     - BAGEL-based subgoal model;
     - delays of 0–12 timesteps = "240ms on a 50Hz robot".
   - π\*0.6 (arXiv 2511.14759): the quote is correct, but the abstract qualifies it with "On some of the hardest tasks".
   - openpi: Apache-2.0, 14,070★, pushed 2026-08-24 (`gh api`).
   - $600M/~$5.6B led by CapitalG (Wikipedia, citing Bloomberg).
8. **Anthropic Project Fetch phase two** (Jun 18, 2026).
   - Opus 4.7, "adaptive thinking with effort set to maximum in Claude Code", three trials.
   - The chart alt-text gives 361 min (Claude-less) / 181 min (Team Claude) / 9-min-plus (Opus 4.7). Lines of code: 10,309 / 1,136 / 1,045.
   - The text says ">37 times faster than Team Claude-less and more than 18 times faster than Team Claude", which is consistent with the note's 19–38×.
   - Quote: "This doesn't mean that LLMs have now solved robotics. Far from it."
9. **"Claude plays robotics"** (Jul 9, 2026).
   - 12 models across 5 providers; four interfaces (direct, programmatic, policy, RL supervision).
   - Direct LIBERO is "from 0 to 5.5%". Figure alt-text: "Mythos Preview has the highest success rate, at 5.5".
   - VLA supervision: "substantially worse than MolmoAct does on its own".
   - Cursor aid: 6%→32% for Mythos.
   - Latency: ~83 Hz needed vs ~0.2–0.4 Hz; turns of 2–8 s, 5–15 s, and 15–60 s.
   - "no model successfully stood the robot up".
   - `github.com/safety-research/embody` returns 404 (`gh api`, 2026-10-02).
10. **Project Pilot** (Jul 24, 2026).
    - Andon Labs ran Drone-Bench: "15 models from three developers".
    - Fable 5 got past the baseline "on all tasks except reconstruction". Chart alt-text: "Reconstruct lags at about 47%".
    - The doorway/wall quote is verbatim.
    - The drone was a DJI Tello EDU, "$129".
11. **Model Hardware Standard** (Aug 27, 2026).
    - Design: "read"/"write" primitives; the reference file covers "what it can measure, what can be adjusted, and what safety limits will be enforced"; control via "MCP, the command line interface, and code files (APIs)".
    - CMU: six induced faults, "correctly blocked all six before any device moved".
    - QuEra: the bespoke script started at 150 s and 58%; the overnight dev run reached 96%; the blind test reached 695/700 = 99.3%.
    - Vendors: Universal Robots, Doosan and AWS Strands. Hugging Face is adding MHS to LeRobot. There is a waitlist, and it will be made open source later.
12. **Claude API table** (platform.claude.com, 2026-10-01).
    - IDs, $10/$50, $4/$20, $2/$10, $1/$5, latency labels and default efforts all match.
    - Fable 5.1 released Sep 1, 2026; cache read $0.25.
    - Opus 5.5 released Sep 22, 2026; cache read $0.20; "Fast mode… research preview".
    - Haiku 4.5 has a 200K context window.
13. **GPT-6 Astra** (community.openai.com topic 1394703, Discourse JSON, created 2026-09-03T19:51Z).
    - "1,050,000 context window, 128,000 max output tokens, Apr 30, 2026 knowledge cutoff".
    - Input $10.00, cached input $1.00, cache writes $12.50, output $50.00.
    - The embedded @OpenAI tweet dated 4 Sep 2026 says it is "also live in the API".
14. **OpenAI Robotics.** Humanoids Daily (May 31, 2026): Ramesh-led world-simulation program "evolved… into OpenAI Robotics"; 11 SF roles; near-term focus on skilled infrastructure workers.
15. **RoboDojo / Astra** (arXiv 2609.24170, Sep 21, 2026).
    - "22.48% average success rate and 28.97 Score over 2,100 trials, ranking above every public entry".
    - GPT-5.5 scored 0.88% and DeepSeek-Flash 1.92% (10 episodes/task).
16. **Galbot Astra paper** (arXiv 2609.38537, Sep 29, 2026, authors "Galbot Team").
    - Hybrid with π0.5: 48%.
    - Tokens: 624.8M (policy-assisted) vs 1.132B (direct) "across 50 RoboDojo instances per condition".
    - "250 model calls averaging 39.86 seconds each, with physics paused".
17. **Understanding Robots** (Kai Williams, Oct 1, 2026): "currently seventh… the top model is an LLM agent with access to π0.5"; "real-world testing stopped after it damaged some of RoboDojo's hardware".
18. **HomeBody.** The `Stanford-TML/homebody` README schedules SIM on Oct 4, REAL2SIM on Oct 11, and REAL on Oct 18, 2026. As of 2026-10-02 the repo holds only the README and docs (113★).
19. **GR00T N1.7** (repos/NVIDIA_Isaac-GR00T/README.md @ 51d4c89, 2026-08-20).
    - Backbone: Cosmos-Reason2-2B (Qwen3-VL), "a **gated** model".
    - DiT goes "from `32` to `16` diffusion layers".
    - Relative EEF actions; "20K hours of EgoScale"; ONNX/TensorRT export.
    - 8,150★.
20. **DreamZero / GR00T N2.**
    - DreamZero (arXiv 2602.15922): 14B, 7 Hz, "30 minutes of play data".
    - NVIDIA dev blog (Jun 15, 2026): "1750 Elo on the April 2026 RoboArena leaderboard, outperforming Pi-0.5 at 1622".
    - TrendForce (Mar 19, 2026): N2 "by end 2026", DreamZero-based, "more than double the success rate".
21. **Isaac ROS 5.0** (RoboticsTomorrow, Sep 22, 2026, ROSCon Toronto): "helps humans and AI agents build robots together"; AgenticROS.
22. **1X** (1x.tech/ai):
    - Redwood is "among the first VLAs to control locomotion jointly with manipulation", with an "off-board language" model.
    - "The 1X World Model now serves as NEO's cognitive core" (1.12.26).
    - NEO: $20,000 or $499/month (Robot Report).
    - SoftBank in talks for a "majority stake… about $6 billion" (Reuters via Euronext, Aug 26–27, 2026).
23. **Tesla.** The Q2-2026 10-Q quote is verbatim (sec.gov).
24. **Dyna.** MarkTechPost (Aug 13, 2026):
    - Dyna-1 is "initialized from Qwen3-VL-4B".
    - At unseen customer sites: 87% vs 46%.
    - Pretraining on "1M+ hours".
    - "no public checkpoint, API, or license".

    The PR Newswire release (Sep 15, 2025) confirms "99+% success rate in 24 hours of non-stop operation".
25. **FieldAI.** $405M (Robot Report, Aug 20, 2025); the "shoehorn large language and vision models" quote is verbatim.
26. **Covariant → Amazon.** "non-exclusive license"; "around a quarter of Covariant's current employees".
27. **Agility.** SPAC (Jun 24, 2026): $2.5B pre-money; ticker AGLT; "nine customer facilities"; ">65,000 hours"; ">$300 million" in Digit v5 orders; DeepMind and NVIDIA named as partners.
28. **Apptronik.** $520M Series A-X; Series A ">$935 million"; "3x multiple of the Series A valuation"; "powered by Gemini Robotics".
29. **Sanctuary.** "99.5%+ task success rate at a cycle time of 2.54 seconds"; deployment "on existing and next-generation industrial robots".
30. **Chinese players.**
    - Unitree IPO: ¥150.8; ~$900M raised; near $50B (Crowdfund Insider, Aug 19, 2026).
    - AgiBot GO-2: LIBERO 98.5%, LIBERO-Plus 86.6% zero-shot; asynchronous dual-system (Robot Report).
    - AgiBot HK IPO process (TechNode, Jul 27, 2026).
    - Galbot: ¥2.5B ($362M) (Caixin, Mar 3, 2026).
    - X Square: ~$276M Series B "led by Xiaomi's strategic investment arm", 58.com partnership.
31. **Harness layer** (`gh api`, 2026-10-02).
    - Stars: dimos 4,601; OM1 2,939 (MIT); ROSA 1,647; ros-mcp-server 1,479; RAI 597; strands-labs/robots 171; innate-os 82; inspect-robots 631; piper-astra-jev 11; metal-arm-harness 15; wall-x 1,281; unifolm-WMA 1,161; unifolm-vla 634; AgiBot-World 3,199; SO-ARM100 7,637; XLeRobot 5,564.
    - dimOS default model `gpt-5.6-luna`: `repos/dimensionalOS_dimos/dimos/agents/mcp/mcp_client.py:84` and `dimos/evals/agents/base.py:85`.
    - innate-os default `google:gemini-3.6-flash`: `ros2_ws/src/cloud/clients/innate-llm/innate_llm/configure.py:45` and `sim/launcher/config.py:107`.
    - AWS reference design (Dec 12, 2025): Claude Sonnet 4.5 in the cloud, `qwen3-vl:2b` via Ollama at the edge, a `@tool def execute_manipulation` VLA wrapper, and GR00T + SO-101 demo code.
    - OM1: "plug-and-play support for OpenAI, Gemini, DeepSeek, and xAI"; "$20 million".
    - Show-Harness: arXiv 2609.10522, Sep 9, 2026.
    - Robocurve: "$10M seed round led by Initialized Capital" (Sep 14, 2026); "downloaded 97k+ times"; "output token speed… improving by 2.1x/month".
    - Hilstart: YC Summer 2026; quote is verbatim.
32. **Hardware prices** (pages opened 2026-10-02).
    - SO-101 Pro motor kit: $249.90.
    - PiPER: $1,999.00.
    - YAM: $2,999 / $3,499 / $4,299 / $4,999.
    - Metal Arm: $2,499 + $249 shipping; 7-DoF, "3 kg payload", ±0.1 mm; ships September 2026.
    - Reachy Mini: $399 Lite / $499 Wireless.
    - Microduck: $399; pre-orders opened Aug 27, 2026; "Software is Apache-2.0; the mechanical and electronic design files are not open".
    - XLeRobot: ~$660 basic.
    - AlohaMini2: "under $1,000 self-build BOM".
    - Open Duck Mini: "under $400".
33. **Figure news index.** "F.03 Arrives at BMW" (May 26, 2026); Catalyst Brands agreement (May 8, 2026).

### B. Corrections (also fixed inline with ~~strike~~ + [corrected: …])

1. **"All of these train their own models"** (TL;DR 1) → wrong for Apptronik, which appears in the same list.
   - Apptronik runs Google DeepMind's Gemini Robotics ("powered by Gemini Robotics", GlobeNewswire, Feb 11, 2026).
   - Boston Dynamics also adopted Gemini Robotics for Atlas (The Robot Report, Jan 5, 2026).
2. **PI "~$1B at >$11B in talks (Mar 2026)"** → outdated.
   - Filings dated May 21, 2026 show a ~$1.05B Series C at ~$11–11.2B post-money (Forge/Notice, per revenuememo.com/p/physical-intelligence-funding).
   - PI has not announced it. Secondary sources only.
3. **GR00T N1.7 "weights Apache-2.0"** → the code is Apache-2.0, but the weights are not.
   - The README's own License section says "Model weights: NVIDIA Open Model License" (`repos/NVIDIA_Isaac-GR00T/README.md`, License section).
   - `nvidia/Cosmos-Reason2-2B` is `license_name: nvidia-open-model-license` and gated (huggingface.co/api/models).
   - The intro line "fully commercially licensable under Apache 2.0" is inconsistent with the License section. Treat the weights as NVIDIA Open Model License plus a gated backbone.
   - Fixed inline in §1 and §7.4.
4. **Opus 5.5 breaking changes.** The note lists "text between tool calls returns inside thinking blocks unless display is set" as breaking. Per the docs:
   - It is a non-breaking response-shape change: thinking blocks "whose text is empty at the default display setting".
   - The real fourth breaking change is "thinking blocks are tied to the model and the conversation".
   - The first three (no disabling thinking, forced tool use error, thinking-block binding) "also apply on Claude Fable 5.1", not only the tool-use one.
   - `computer_20251124` is rejected "on the Claude API and Google Cloud" only.
5. **"Galbot measured 39.86 s per Astra call"** (§7.1) → the figure applies only to the 30-s locomotion run: "250 model calls averaging 39.86 seconds each, with physics paused" (arXiv 2609.38537 abstract).
6. **Agility ">$620M proceeds"** → ">$620 million of *expected* gross transaction proceeds… (assuming no redemptions)": $420M trust plus ~$200M PIPE at $10/share. Not closed as of Jun 24, 2026.
7. **Apptronik "Pilots: Mercedes, GXO, Jabil"** → the release calls them "partnerships" and does not say which are pilots.
8. **Altman "definitely do a humanoid" (UNVERIFIED)** → upgraded to verified via a secondary source: Humanoids Daily, Sep 2, 2026, Sources podcast, "We will definitely do a humanoid… We will do other form factors as well."
9. **Dyna ">$600M (UNVERIFIED)"** → upgraded to secondary-confirmed: Bloomberg (Sep 15, 2025), "valuation tops $600 million" (Investing.com and Dealroom headlines). No 2026 round was found.

### C. Unverifiable or weakly sourced claims (left in place)

- **Unitree "IPO at ~¥61B"** and the "460% close": not in the cited source. The implied IPO value is roughly $9B, since ~$900M bought 10% of enlarged capital.
- **Galbot "$3B post-money in Dec 2025"**: appears only in Caixin's *AI-generated* digest ("AI generated, for reference only"), as ">$300 million… led by a China Mobile-backed fund". The article body is paywalled.
- **Apptronik "~$5B"**: press only. The release gives only "3x" the Series A valuation.
- **Sanctuary ~$149M; Covariant $222M; AgiBot HK$40–50B target**: not re-checked.
- **PI partner-post numbers** (Weave −42% missed grasps / −50% interventions; Ultra 96.4%): pi.website was not re-rendered in this pass. The π0.7 air-fryer 5%→95% figure stays press-only.
- **Robocurve "METR/Epoch for robots"**: not in the seed-raise post text. It may be on another Robocurve page.
- **Robocurve 19/20 YAM and RoboHarm 60/100**: taken from the teammate note `sources/robocurve-gpt6-astra.md:24,487`. robocurve.org/roboharm was not re-opened. The local `repos/roboharm` README confirms the Astra/Fable/MolmoAct2 harness setup (`openai/gpt-6-astra`, `anthropic/claude-fable-5-1`; 900 steps / 40 LLM calls budget).
- **Hardware prices not re-checked**: Innate MARS $995; Koch v1.1 totals ($199/$278; the README lists only per-part prices); ToddlerBot <$6,000; PiPER US reseller prices.
- **Other items not re-opened**:
  - LeRobot `groot` policy type;
  - Mbodi–ABB;
  - Strands HF blog (Jun 17, 2026) and the Strands + MHS blog;
  - LeRobot v0.6.0 blog;
  - HomeBody using Astra on a G1 (the-decoder);
  - the EU Machinery Regulation date, which is background knowledge.

### D. Important details the note missed (verified)

1. **Boston Dynamics × Google DeepMind** (The Robot Report, Jan 5, 2026, CES).
   - "DeepMind will help make the Atlas humanoid smarter with its Gemini Robotics foundation models".
   - Comes with a productized Atlas and a roadmap into Hyundai factories.
   - This is a second major humanoid OEM, after Apptronik, that outsources its brain to Google. It strengthens Threat #1 (§7.4).
2. **Google's Safari SDK** (`google-deepmind/gemini-robotics-sdk`, Apache-2.0, 613★, pushed 2026-09-18; local clone `repos/google-deepmind_gemini-robotics-sdk/README.md`).
   - Covers the full lifecycle: "access checkpoint, serving a model, evaluate… upload data, finetuning".
   - "Most of the functionality requires you to join Gemini Robotics Trusted Tester Program."
   - This is how the "early-access" VLAs are actually consumed.
3. **More ER 2 API facts** (pricing and robotics pages):
   - Batch tier: $0.50 in / $2.50 out per M through 2026 ($1/$5 from 2027).
   - Context caching: $0.10/M (2026).
   - The streaming model lacks caching, code execution, structured outputs and Batch.
   - The doc's own code samples use `"thinking_level": "high"` although the text recommends "medium".
4. **GPT-6 Astra Ultrafast** (NVIDIA blog, Oct 1, 2026).
   - "available now in the OpenAI API"; "up to 8x faster token generation than the Astra Standard mode" on Blackwell.
   - This bears directly on §7.1's latency argument. The pricing ("Ultrafast guide") was not opened.
5. **Claude lineup risks for routing** (§7.2):
   - **Haiku 4.5 "Retirement: Not sooner than October 15, 2026"**, two weeks after this note. The cheapest current-generation Claude is then Sonnet 5.5 at $2/$10.
   - **Claude Mythos 5.1** exists, invitation-only via Project Glasswing, with the "same capabilities" and the specs and pricing of Fable 5.1.
   - Opus 5.5 extras: Batch output up to 300K tokens with beta header `output-300k-2026-03-24`; minimum cacheable prompt 512 tokens; 5-min cache write $5/M.
6. **Skild commercial traction.**
   - "Skild AI crossed $100 million in annual recurring revenue, ten months after our first commercial deployment" (blog, Sep 10, 2026).
   - "Physical Self-Play" post-training on S1 (Sep 23, 2026).
   - This is the strongest revenue signal among robot-FM startups.
7. **Genesis AI** was "in talks… to raise capital at a $3 billion valuation" (TechCrunch, Aug 25, 2026, Generalist article). It is a robot-FM competitor missing from the map.
8. **MHS details.**
   - Raspberry Pi is an early adopter: "enabling MHS integration across a number of their products following successful tests using their Camera MHS Driver".
   - Anthropic calls MHS "model-agnostic, and any agent harness can access it".
   - The CMU faults were missing plate, rotated plate, reader busy, disconnected camera, unreachable device and active e-stop. Integration to a completed curve took "eight hours".
   - QuEra's loop used four fresh Claude instances: hypothesis, code, run and log-reader.
9. **Fetch phase 2 nuance.**
   - Team Claude-less wrote **1,136** lines, so "~10× less code" holds only vs Team Claude (10,309).
   - Opus 4.7 was used as "our most advanced non-Mythos-class model".
   - A researcher approved commands throughout.
10. **Project Pilot nuance.** Over 10 simulations, "even Fable 5… reaches the human baseline on average for only three of the five tasks". "The frontier of what models can do is about six months ahead of what they do consistently."
11. **Embody nuances.**
    - GPT-5.4 was "the only model that consistently learned a competent policy" on TwinFlipper RL.
    - Mythos Preview under-performs Opus 4.5/4.6 as a VLA supervisor because "it overrides the VLA more often than is warranted". This is a design warning for "LLM supervises VLA" harnesses.
    - Reasoning budget barely matters for high-level locomotion: Opus 4.6 stayed within a 2.6-point band.
12. **ByteDance Seed GR-3** (arXiv 2507.15493, Jul 21, 2025) is a large-scale VLA co-trained on web vision-language data and VR human trajectories. It is a Chinese big-tech VLA player missing from §3.
13. **AgenticROS** is "sponsored by… RealSense". It "connects Isaac ROS with NVIDIA Nemotron open models and NVIDIA NemoClaw blueprints".
14. **LeRobot scale.** `huggingface/lerobot` has 27,897★ (2026-10-01). This is the asset moving under NVIDIA, and it is adding MHS support.
15. **Astra robotics virality.** OpenAI robotics employee Thijs Simonian's Golden Gate painting demo with a cheap arm triggered the wave (Understanding Robots). Related local clones: `repos/astra-paints`, `repos/so101-painting`.
16. **phosphobot** (`phospho-app/phosphobot`, MIT, 394★) is an open SO-101/LeRobot control and VLA-training stack from a hardware-kit vendor. It is an adjacent harness-layer competitor.
17. **Helix 2.5 vs Helix 02.** Helix 2.5 "matched [Helix 02's] success rate while using half as much adaptation data", zero-shot, where Helix 02 had been trained in the evaluation environment.
18. **GEN-1 per-task numbers** (no trial counts given):

    | Task | GEN-1 | GEN-0 | Scratch |
    |---|---|---|---|
    | Vacuum servicing | 99% | 50% | 2% |
    | Box folding | 99% | 81% | 13% |
    | Phone packing | 99% | 62% | 42% |

19. **Agility nuance.** The Digit v5 orders are "subject to the realization of certain contractual milestones". Agility is the launch partner for NVIDIA Halos.
