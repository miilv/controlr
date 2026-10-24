# Deep research: RoboDojo agent harnesses and hybrids (ChatGPT deep research, 2026-10-24)

Prompt: [../prompts/deep-research-robodojo-agent-harnesses.md](../prompts/deep-research-robodojo-agent-harnesses.md) (the first, narrower version).
Full report and evidence: [deep-research-robodojo-2026-10-24.html](deep-research-robodojo-2026-10-24.html), [.json](deep-research-robodojo-2026-10-24.json) (63 project cards, 17-mechanism matrix, 62-row leaderboard with per-dimension scores).
Spot-checked 2026-10-24: XPolicyLab PRs #175/#195/#198/#201 exist as described; the repos CodeActionBench, URAI, RACaP, Noematrix RoboRSI, RoboSkill, R2SI and OpenETA exist.

---

## What I would change in controlr

**Build a persistent Python controller that executes bounded feedback programs, give it calibrated geometry and explicit effect checks, then add a separate agent that repairs and promotes reusable tools between episodes.**

The evidence supports that sequence. It does **not** support simply replacing one `MOVE` per call with longer open-loop action sequences. PyRUA supports code-cell control; URAI shows the value of decisions made during execution; Guava and Show-Harness show that indiscriminate batching can reduce success. [arxiv.org](https://arxiv.org/html/2610.01939v2?utm_source=chatgpt.com)

I found **30 additional direct-controller or compiled-program approaches**, **14 substantial hybrid systems**, and **10 pending or adjacent records** worth tracking. The complete deliverables are:

- **[controlr_robodojo_research_2026-10-24.html](https://chatgpt.com/library?file_id=libfile_fe92e940438c81918c2bcd216ff92ede&account_id=personal)** — searchable report with **63 audit cards**, a **17 × 63 mechanism matrix**, **30 quantitative comparisons**, all **62 leaderboard entries**, adapter findings, repository metadata, and implementation estimates.
- **[controlr_robodojo_evidence_2026-10-24.json](https://chatgpt.com/library?file_id=libfile_56a24b99ae508191ad3651f88da9d1f1&account_id=personal)** — the structured evidence, sources, matrix, leaderboard snapshot, and experiment proposals.

Each project card includes the requested group, release date, stars, last push, control interface, observation and privilege boundary, robot API, memory and verification, task specialization, models, costs, results, red flags, and transferable ideas. **Paper claims and inspected code are distinguished.** No experiments were independently rerun.

### Four corrections that materially affect the comparison

1. **28.97 is GPT-6-Astra’s RoboDojo Score, not its success rate.** Its official SR is **22.48%**. SenseNova’s **54.17% SR** is an author-run **ICL-plus-VLA** condition, with substantially more tokens; it is not a listed official VLA-free result. [robodojo-benchmark.com](https://robodojo-benchmark.com/leaderboard?utm_source=chatgpt.com)
2. **PyRUA-Lean’s official RGB configuration has no cross-episode learning.** Its **50.35% official SR** is evidence that control representation and tools can produce a strong result before adding RSI. [GitHub](https://github.com/DAGroup-PKU/PyRUA-Lean?utm_source=chatgpt.com)
3. **Compiled controllers are a separate category.** COAP’s **70.24%**, and pending REAL_RSI’s **61.92%**, execute generated task programs without a runtime LLM. Those are relevant software-generation results, but not measurements of a general online agent making every decision. [arxiv.org](https://arxiv.org/html/2610.12369v1?utm_source=chatgpt.com)
4. **Your 9/20 on `general_pickup` is not comparable to a 42-task aggregate.** It is one task’s 45% sample result. The board combines task categories, while several papers additionally use different weightings, selected tasks, repeated resets, or cumulative “solved at least once” outcomes.

---

## 1. Ranked discoveries: direct LLM control and compiled robot programs

This is a **reading and implementation priority for controlr**, not a ranking by incomparable headline SR. Seeds are excluded.

### Highest-priority additions

| Rank | Project | Mechanism to inspect | Result and qualification |
|---:|---|---|---|
| **1** | **[CodeActionBench](https://github.com/lyhkk/CodeActionBench)** — NTU, Sept. 27 | RGB-only **physical wrist stereo**, uncertainty-aware triangulation, gripper ruler, planned-pose overlays, and composable code execution. | Astra **55/75 = 73.33%** across 25 tasks. **One fixed scene per task**, three attempts; no held-out-layout or triangulation ablation. This is the closest new tool reference for your observation constraints. [arxiv.org](https://arxiv.org/html/2609.33807v1?utm_source=chatgpt.com) |
| **2** | **[PhysEvo](https://arxiv.org/abs/2610.08995)** — UCAS/CAS/Tsinghua, Oct. 6 | Frozen task agent plus a meta-agent that edits shared tools and task skills; calibrated RGB geometry, temporal wrist triangulation, history review, paired development checks. | **62.00% category-weighted SR**, but **58.57% equal-task SR**, on **210 held-out author-run episodes**. No public implementation found. Completion verification queries native termination; its full evaluator-feedback boundary remains unverified. [arxiv.org](https://arxiv.org/html/2610.08995v1?utm_source=chatgpt.com) |
| **3** | **[URAI](https://github.com/HexaFuture/URAI)** — HexaFuture, Sept. 30 | A compact robot-use interface with runtime measurements and decisions. Particularly useful evidence about when the model should regain control. | **56% versus 24%**, 50 paired episodes, when runtime decisions replace an advance-written program using the **same tools**. Uses depth and task semantics. The released hardware tools differ from the unpublished simulation implementation. [arxiv.org](https://arxiv.org/html/2609.39018v1?utm_source=chatgpt.com) |
| **4** | **[RACaP](https://github.com/Robo-Harness/racap)** — CUHK/HKUST/Knowin, Sept. 24 | Failure critic → minimal patch → isolated checks → paired physical evaluation → promotion. Keeps a record of rejected and accepted changes. | LIBERO-PRO **32.8→45.0%**, **180 episodes/condition**, 33 rescued and 11 lost. Uses RGB-D and typed primitives. This is among the clearest engineering references for your repair loop. [arxiv.org](https://arxiv.org/html/2609.29394v1?utm_source=chatgpt.com) |
| **5** | **[Noematrix RoboRSI](https://github.com/nssmd/RoboRSI)** — Noematrix, Aug. repository / Oct. paper | Hierarchical executable skills, responsible-node repair, code consolidation, explicit frozen/evolving modes, and a simulator-state access firewall. | Consolidation **21.5→29.0%**, **600 paired episodes**, +7.5 pp. Depth and learned grasp perception remain inside the tools. **Distinct from SenseNova-RoboRSI.** [arxiv.org](https://arxiv.org/html/2610.12424v1?utm_source=chatgpt.com) |
| **6** | **[RoboSkill](https://github.com/SII-dannyXSC/RoboSkill)** — Fudan/SII, Sept. 29 | Retain **actual executed code**, text, keyframes, metadata and failure evidence; retrieve a small number of relevant experience packages. | Astra first-attempt SR **72.5→97.5%**, n=40. Code shows **GT-derived initial 2D object boxes**, depth, and a lengthy resettable evaluation budget. Generated skills/results are not released. [arxiv.org](https://arxiv.org/html/2609.37810v1?utm_source=chatgpt.com) |
| **7** | **[OpenETA, RoboDojo branch](https://github.com/OpenMOSS/OpenETA/tree/robodojo-v112)** — OpenMOSS, Oct. 16 branch | Persistent coding-agent session, strict observation freshness, reusable repair tools and versioned evidence. | **23/42 first-pass = 54.8%**; **28/42 = 66.7%** is cumulative across changing versions and human-assisted repairs. Uses aligned depth and native completion feedback; historical experience is private. [GitHub](https://github.com/OpenMOSS/OpenETA/blob/main/docs/robodojo-rsi-results.md?utm_source=chatgpt.com) |
| **8** | **[Encore](https://github.com/YIFANK/encore)** — Sept. 29 | Event-indexed demonstration packs; develop a complete feedback program, freeze its hash, and evaluate it with sealed feedback. | LIBERO-PRO position **96.93%, n=1,500**. Selected RoboDojo results include **36/50** on both `build_tower` and `make_kong`; no 42-task aggregate. RGB-D and separately developed task programs. [arxiv.org](https://arxiv.org/html/2609.37359v1?utm_source=chatgpt.com) |
| **9** | **[RATs](https://github.com/Playful-RATs/RATs)** — Berkeley/Impossible, June 17 | Curious play, verifier/curator roles and retained executable skills; useful exploration-budget controls. | Same executor: **36.3% without play →44.3% with curious play**; random play 38.3%. Depth and substantial perception scaffolding; exploration costs must be counted. [arxiv.org](https://arxiv.org/html/2606.19419v1?utm_source=chatgpt.com) |
| **10** | **[COAP / Embodied Turing Machines](https://arxiv.org/abs/2610.12369)** — NTU/Ropedia, Oct. 8 | Offline coding agents compile shared perception/control software, with explicit world-state confidence and age. | **70.24% SR**, **6,300 author-run RoboDojo episodes**. Zero runtime LLM/VLA calls; known meshes/shapes/table geometry, privileged development, task programs and no public runtime. [arxiv.org](https://arxiv.org/html/2610.12369v1?utm_source=chatgpt.com) |
| **11** | **[OpenRUA](https://github.com/terminalworld/OpenRUA)** — UCL/CMU, Sept. release / Oct. paper | Coding agent directly uses native ROS 2, treating perception as files and constructing its own control clients and feedback programs. | **87% LIBERO-PRO, n=800**. RGB-D; up to **500 turns or 240 minutes**, simulator paused during reasoning, and no usual native-step horizon. Fresh workspace per episode, not RSI. [arxiv.org](https://arxiv.org/html/2610.02459v1?utm_source=chatgpt.com) |
| **12** | **[VIA](https://github.com/hengyuan-hu/via)** — Stanford, July / Oct. revision | Stage a virtual gripper in a visual interface, inspect it, then execute waypoints. | Matched-interface results include Astra **93.3→100%** and Sol **25→73.3%**, n=60/model. Uses an RGB-D point cloud. Rotating its virtual camera does **not** acquire another physical observation. [Paper](https://arxiv.org/html/2607.11119v2). |
| **13** | **[Scale and Selection](https://arxiv.org/abs/2609.39304)** — NovaXBot/QuantGroup, Sept. / Oct. 22 v4 | Separate task and optimizer sessions; bounded coherent commits; rejected-change history; paired champion–challenger selection. | No-VLA held-out SR **51→67%**, n=100, task-disjoint. Larger batches use more compute; unconditional evolution stops earlier. Later **71.2% RoboCasa** is a separate hybrid result. [arxiv.org](https://arxiv.org/html/2609.39304v4?utm_source=chatgpt.com) |
| **14** | **[RPG: Reconstruct, Practice, Go Real](https://rpg-robot.github.io/)** — Berkeley/Amazon/MIT/Chicago, Oct. 1 | Diagnose failure, evolve shared skills, and retest **both individual changes and the merged library**. | Same scripted caller **65→92.5%, n=40**. Privileged development diagnosis, reconstructed tasks, and no released runtime. Held-out initializations do not imply held-out tasks. [arxiv.org](https://arxiv.org/html/2610.02204v1?utm_source=chatgpt.com) |
| **15** | **[SimEX](https://robo-simex.github.io/)** — Amazon/UT Austin/CMU, Sept. 30 | Inherit only `toolbox.py` and `skill.md`; discard task scripts; replay a capability bank during repairs. | Barcode tasks **69.14→85.14%** with capability-bank replay, n=175. Custom task families, substantial optimization compute, no public runtime located. [arxiv.org](https://arxiv.org/html/2609.38982v1?utm_source=chatgpt.com) |
| **16** | **[Guava](https://github.com/hdacnw/guava-release)** — UMD and collaborators, June / Sept. v3 | Explicit tool-effect feedback, recovery-branch data and a released 4B tool-using student. | Student **87.1% simulation / 90% real**. Depth inside semantic tools. Crucially, multi-action programs **reduce** frontier-model SR in its controlled comparison. [arxiv.org](https://arxiv.org/html/2606.18363v3?utm_source=chatgpt.com) |
| **17** | **[Show-Harness](https://github.com/showlab/Show-Harness)** — NUS, Sept. 9 | Exact coordinate conventions, coarse/fine motion, adaptive chunking and gripper-effect recovery. Public Franka runtime is RGB-only. | In its ablation: **96% full**, **74% always chunk**, **72% no recovery**. No chunking also gets 96%, with more calls. Different published suites must remain separate. [arxiv.org](https://arxiv.org/html/2609.10522v1?utm_source=chatgpt.com) |
| **18** | **[RHO / HELIX](https://github.com/KE7/HELIX)** — June 15 | Multi-file controller evolution, isolated worktrees, evaluator boundaries and validation-selected deployment. | **70% Robosuite**, n=700; **45% LIBERO-PRO**, n=3,000. Canonical development: **$2,821, 79.4 hours, 6,814 evaluations**. Multi-file comparison is not compute-matched. [arxiv.org](https://arxiv.org/html/2606.16458v1?utm_source=chatgpt.com) |

### Additional eligible projects, ranked lower for controlr

| Rank | Project | Useful contribution and main limitation |
|---:|---|---|
| **19** | **[ORA](https://github.com/hao-ai-lab/ORA)** — UCSD | Compact direct-agent loop and geometric/grasp tools; **81.4% LIBERO-PRO, n=800**. Depth and an actor-accessible binary task-success oracle; only early selected RoboDojo results. [Hao AI Lab @ UCSD](https://haoailab.com/blogs/ora/?utm_source=chatgpt.com) |
| **20** | **[Skill-RSI / Compile Once](https://github.com/Orillusion/embodied-skill-rsi)** | Independently qualify a generated skill, freeze it and reuse it cheaply. Three reused skills achieve **148/150 code-only successes**, but acquisition succeeds on only **four of seven tasks**; failed search matters. [arxiv.org](https://arxiv.org/html/2610.25114v1?utm_source=chatgpt.com) |
| **21** | **[RAPID](https://yuyaoliu.me/projects/rapid)** — MIT/NUS/UPenn/NVIDIA | Generate effect predicates and test programs across synthetic geometric variants: **75.9% versus 53.2% without variants**. Demonstration/reconstruction overhead and no runtime release. [arxiv.org](https://arxiv.org/html/2609.30249v1?utm_source=chatgpt.com) |
| **22** | **[K1](https://github.com/Robo-Harness/k1)** — CUHK | Persistent visual anchors, typed tools and small-model training. Astra **11/18→16/18** in a bundled comparison; depth, native success feedback and development overlap. [arxiv.org](https://arxiv.org/html/2609.29389v1?utm_source=chatgpt.com) |
| **23** | **[Astranomy / RoboLab](https://github.com/NVlabs/RoboLab/tree/main/policies/ai_native)** — NVIDIA | Strong geometry/history studies and direct-versus-coding-agent comparison. Historical agentic results used native progress/criteria metadata; released code removes it, **without rerunning the published numbers**. [research.nvidia.com](https://research.nvidia.com/labs/srl/projects/astranomy/?utm_source=chatgpt.com) |
| **24** | **[KnowBody](https://github.com/Loule0-0/KnowBody)** — NJU/Ant/ZJU | Version body estimates and invalidate dependent rules. **12/16 versus 4/16** real successes. RGB-D, waypoint JSON rather than code-cell control, README-only implementation. [arxiv.org](https://arxiv.org/html/2609.28530v1?utm_source=chatgpt.com) |
| **25** | **[Local 27B coding agent / UR3e](https://rtalwar2.github.io/agentic-coding-for-robot-manipulation)** — Ghent/imec | Nine compact HTTP operations, local Qwen and force-guarded descent; **30/45 successes**. Extensive handwritten game skills, depth and no public SDK found. [arxiv.org](https://arxiv.org/html/2609.26499v1?utm_source=chatgpt.com) |
| **26** | **[Sakana: Learning and Transferring Closed-Loop Robot Software](https://arxiv.org/abs/2609.19906)** | Optimized program archives improve target-task success **45.2→57.0%**, n=270; initial code gives only 41.5%. Strong dependence on GT geometry labels; no runtime release. [arxiv.org](https://arxiv.org/html/2609.19906v1?utm_source=chatgpt.com) |
| **27** | **[Agentic Push-T](https://arxiv.org/abs/2608.18227)** — Berkeley | Agent-written perception and MPC-like control. **100% headline uses privileged state**; vision result 97.2%. Narrow geometry/renderer assumptions; **$1,500–2,000 and 221 hours** development. [arxiv.org](https://arxiv.org/html/2608.18227v1?utm_source=chatgpt.com) |
| **28** | **[VLCP](https://arxiv.org/abs/2608.16983)** | Periodically rewrite a local controller while numerical IK runs in generated code. **20/57 versus 2/57** successes, but GT object poses, host Jacobian, one trial/task and unequal call budgets. [arxiv.org](https://arxiv.org/html/2608.16978v1?utm_source=chatgpt.com) |
| **29** | **[LLM_LIBERO](https://github.com/IRVLUTD/LLM_LIBERO)** — UT Dallas | Useful direct-VLM baseline: Astra **44% on 100 episodes**, approximately **$7.09/episode**. No learned harness; primarily a model/reasoning-budget comparison. [arxiv.org](https://arxiv.org/html/2610.30051v1?utm_source=chatgpt.com) |
| **30** | **[SkillComposer](https://arxiv.org/abs/2608.14944)** — ASU | Reusable macro induction and natural-language programming. Macro addition helps one category but hurts spatial and multistep categories; low-level grasp privilege remains unclear. [arxiv.org](https://arxiv.org/abs/2608.14944?utm_source=chatgpt.com) |

### Hybrid discoveries worth mining

These retain a learned motor policy somewhere in the evaluated action path.

| Priority | Project | Transferable mechanism; qualification |
|---:|---|---|
| **H1** | **[WorldScape-R²SI](https://github.com/manifoldai-research/R2SI)** | Executed-action acknowledgments, episode-pinned versions, reflection and paired validation. **Official 39.35% SR / 46.58 Score**. Full source released **today**; WAM supplies motor actions. [GitHub](https://github.com/manifoldai-research/R2SI?utm_source=chatgpt.com) |
| **H2** | **[PhysicalRSI, official HKU adapter](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/physicalRSI)** | Typed capability contracts and frozen compositions of code policies and π0.5. **Official 31.38% SR**. Initial composition selection is not continuous coding or evaluation-time RSI. Distinct from the SAIL catalog. [GitHub](https://github.com/XPolicyLab/XPolicyLab/blob/main/policy/physicalRSI/README.md?utm_source=chatgpt.com) |
| **H3** | **[RoboHarn-Evo](https://github.com/RUCKBReasoning/RoboHarn-Evo)** | Hierarchical task/action memory: **82.5% versus 64.2% flat reflection**, same 120 cases. π0.5 executor; RoboDojo transfer covers only two tasks. [arxiv.org](https://arxiv.org/html/2609.37583v2?utm_source=chatgpt.com) |
| **H4** | **ChunkAct** | Compile subgoal memory into chunks with typed failure exits; **78.8→83.8%**, n=800, with **2.9× fewer planner calls**. Its LIBERO-Plus protocol is partial, not all 10,030 configurations. [arxiv.org](https://arxiv.org/html/2610.18874v1?utm_source=chatgpt.com) |
| **H5** | **SafeHarness v2** | Route selection and contact-aware verification. Astra SR **50.0→81.2%**, n=320/condition. Frozen π0.5; use October 2 v2 rather than superseded figures. [arxiv.org](https://arxiv.org/html/2609.20822v2?utm_source=chatgpt.com) |
| **H6** | **EmbodiedRSI, Columbia/Princeton** | Persistent code/skills and value-of-information memory management. Frozen VLA, but **PPO trains the memory manager**: the whole system is not fixed-weight. Distinct from EmbodiedEvo’s former PR name. [arxiv.org](https://arxiv.org/html/2610.10498v1?utm_source=chatgpt.com) |
| **H7** | **[RoboFoundry](https://github.com/robofoundry2026/RoboFoundry)** | Evolve the supporting system and promote recurring task changes into shared components. **72.7 versus 56.9% means +15.8 pp**, not +27.8 pp. Runtime release incomplete. [arxiv.org](https://arxiv.org/abs/2609.32862?utm_source=chatgpt.com) |
| **H8** | **[HarnessPAI](https://github.com/Darwin-Agent/HarnessPAI)** | Offline code evolution around VLA/WAM primitives. Evaluation includes evolution seeds in some suites; the large latency comparison is scoped to one task. [arxiv.org](https://arxiv.org/html/2609.29166v2?utm_source=chatgpt.com) |
| **H9** | **[HexaAnything / PhysicalCoding](https://github.com/HexaFuture/PhysicalCoding)** | Explicit world models in code, verification and software evolution. Main RoboCasa path uses XR1; cloth privilege changed between evaluation versions. [arxiv.org](https://arxiv.org/html/2609.35432v1?utm_source=chatgpt.com) |
| **H10** | **[Zetta](https://github.com/air-embodied-brain/Zetta-Embodiment)** | Critic-guided repairs and constrained interventions. Runtime critics receive object poses, contacts and goal semantics; **90.8% is a Goal subset**, not all LIBERO-PRO. [arxiv.org](https://arxiv.org/html/2608.16590v1?utm_source=chatgpt.com) |
| **H11** | **[RoboHarness, Huang et al.](https://github.com/markli1hoshipu/RoboHarness)** | Memory bridges between heterogeneous motor policies. Uses noisy simulator poses / real markers and adaptation during evaluation. Distinct from MiaoDX’s verification harness. [arxiv.org](https://arxiv.org/html/2607.18060v3?utm_source=chatgpt.com) |
| **H12** | **[AgenticRoboRL](https://github.com/xinxinliu-ai/AgenticRoboRL)** | Small agent post-training: **18.1→65.1→71.0%** through base/SFT/RL on the hybrid route. Separate VLA-free CaP-X results use privileged observations. [arxiv.org](https://arxiv.org/html/2610.28589v1?utm_source=chatgpt.com) |
| **H13** | **BATON** | Verify invocation, handoff and lookahead conditions between subtasks; retain transition-aware memory. Frozen VLA supplies contact actions. [arxiv.org](https://arxiv.org/html/2608.16889v1?utm_source=chatgpt.com) |
| **H14** | **2AM** | Agent-owned memory becomes language and 2D action hints. A VLA is the sole motor executor; **76.3% completion versus 11.8% strict success** are different metrics. [arxiv.org](https://arxiv.org/abs/2609.11308?utm_source=chatgpt.com) |

---

## 2. Mechanism matrix

The report contains the complete **17 mechanisms × 63 records**, with a source and scope note on every established cell. These excerpts cover the main seeds and the most relevant discoveries.

**U:** used. **A:** an ablation is available. **N:** explicitly absent in the named configuration. **?:** not established. A paper-only claim remains paper-only even when marked U.

### Seeds

| Mechanism | PyRUA | HexaShell | EmbodiedEvo | SenseNova | RAX | RPent | CaP-X |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Code cells / executable programs | A | U | U | ? | U | ? | U |
| Separate repair/critic agent | ? | U | N | ? | ? | ? | ? |
| Persistent skills / lessons | N | U | U | U | U | U | U |
| RGB triangulation | U | A | ? | ? | ? | ? | ? |
| Explicit wrist re-view | ? | U | ? | ? | U | ? | ? |
| Visual differencing | ? | ? | ? | ? | ? | ? | U |
| Exploration / practice phase | N | U | A | ? | U | U | ? |
| Multiple candidate reasoning | ? | ? | ? | ? | ? | ? | U |
| Prompt caching explicitly reported | ? | ? | U | U | ? | ? | ? |
| Depth in evaluated tools | N | N | N | N | U | ? | U |
| VLA/WAM motor actions | N | N | N | U | N | U | N |

Scope matters: PyRUA’s row describes its official RoboDojo configuration; SenseNova’s motor-action cell includes its headline hybrid; CaP-X’s observation tiers vary; RAX’s RoboDojo study adds depth. PyRUA’s small coder uses GRPO, so it is not labeled teacher-trace distillation. Primary sources: [PyRUA](https://github.com/DAGroup-PKU/PyRUA-Lean), [HexaShell](https://github.com/HexaFuture/HexaShell), [EmbodiedEvo](https://github.com/Einsia/EmbodiedEvo), [SenseNova](https://github.com/OpenSenseNova/SenseNova-RoboRSI), [RAX](https://github.com/robot-agent-exploration/rax), [RPent](https://github.com/RLinf/RPent), [CaP-X](https://github.com/capgym/cap-x).

MiaoDX’s `roboharness` and the Physical-RSI catalog are included as **not applicable** for controller mechanisms in the full matrix: one is verification infrastructure, the other a discovery source.

### New projects closest to the redesign

| Mechanism | CodeActionBench | PhysEvo | URAI | RACaP | Noematrix | RoboSkill | Guava |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Code cells / executable programs | U | ? | U | U | U | U | N |
| Repair/critic agent | ? | U | U | U | U | N | ? |
| Persistent skills / lessons | N | U | U | U | U | A | N |
| RGB triangulation | U | U | ? | ? | ? | ? | ? |
| Explicit wrist re-view | U | U | ? | ? | ? | ? | ? |
| Exploration / practice phase | N | U | U | U | U | U | ? |
| Distilled small tool agent | ? | ? | ? | U | ? | ? | U |
| Verification before retention | ? | U | U | U | U | U | ? |
| Depth inside actor/tools | N | N | U | U | U | U | U |
| Privileged runtime information | N | ? | N | ? | N | U | ? |
| VLA/WAM motor actions | N | N | N | N | N | N | N |

“Verification before retention” ranges from evidence/provenance checks to actual physical regression tests; those are distinguished in the full matrix. RoboSkill’s privilege is its initial GT-derived 2D boxes. PhysEvo’s uncertainty concerns its native-termination check, not hidden object coordinates. Primary sources are linked in the ranked table and each matrix cell in the report.

---

## 3. Mechanisms with useful ablation evidence

### The comparisons I would use to choose engineering work

| Mechanism | Evidence | What it supports |
|---|---|---|
| **Code-cell control** | PyRUA, VLA-free comparisons: LIBERO **73.0→80.5%**; RoboTwin **42.8→68.8%**; RoboCasa atomic **45.6→65.6%**, composite **7.5→20.0%**. | Control flow, persistent local state and observation delivery can help. These are not the official RoboDojo experiment, and guides affect the comparison. [arxiv.org](https://arxiv.org/html/2610.01939v2?utm_source=chatgpt.com) |
| **Decisions during execution** | URAI: **24→56%**, n=50 paired; 17 rescued, one lost. | Preserve useful runtime decision points even when executing programs locally. Same tools make this more informative than its bundled interface headline. [arxiv.org](https://arxiv.org/html/2609.39018v1?utm_source=chatgpt.com) |
| **RGB triangulation** | HexaShell: **29.6→38.3%**, **+8.7 pp**, 420 paired development episodes; 57 rescued, 23 lost. | Calibrated multi-view geometry beats the tested monocular-depth alternative. It is not an ablation of “all geometry” or the official hidden-seed result. [arxiv.org](https://arxiv.org/html/2610.23479v1?utm_source=chatgpt.com) |
| **Verified repair campaign** | RACaP: **32.8→45.0%**, **+12.2 pp**, n=180; 33 rescued, 11 lost. | Failure-driven tool repair can transfer beyond its development distribution. The entire repair phase is the intervention. [arxiv.org](https://arxiv.org/html/2609.29394v1?utm_source=chatgpt.com) |
| **Code consolidation** | Noematrix: **21.5→29.0%**, **+7.5 pp**, n=600; paired CI **[3.7, 11.5]**. | Compiling recurring skill sequences into code has direct support. Separate efficiency measurements report fewer tokens/calls/time. [arxiv.org](https://arxiv.org/html/2610.12424v1?utm_source=chatgpt.com) |
| **Curious play** | RATs: no play **36.3%**, random play **38.3%**, curious play **44.3%**. A separate compute-matched comparison gives retained skills **32.3% versus 26.0%** for extra turns. | Experience can help beyond merely spending more inference calls, within this scaffold. [arxiv.org](https://arxiv.org/html/2606.19419v1?utm_source=chatgpt.com) |
| **Inherited experience** | EmbodiedEvo: task-text baseline **8.50→30.17%**, n=210. | The learned package helps versus no inherited experience. It does not isolate code, lessons, recipes or exploration cost.  |
| **Structured memory versus raw logs** | EmbodiedEvo: **29.78→30.17%**, only **+0.39 pp**, CI **[−5.11, +5.94]**; estimated test cost **$12.203→$5.732**. | The stronger finding is efficiency, not an established SR improvement over raw history. Costs exclude Playground.  |
| **Executable code in experience** | RoboSkill, text-only→text+code: Sol **15→67.5%**, Astra **90→92.5%**, Opus **70→67.5%**. | Reusable code helps some models dramatically, but not uniformly. Same sensor/privilege caveats apply. [arxiv.org](https://arxiv.org/html/2609.37810v1?utm_source=chatgpt.com) |
| **Improved library with unchanged caller** | RPG: **65→92.5%**, n=40. | The library itself can carry the improvement. Its development process uses privileged diagnostics. [arxiv.org](https://arxiv.org/html/2610.02204v1?utm_source=chatgpt.com) |
| **Regression capability bank** | SimEX barcode: **69.14→85.14%**, **+16 pp**, n=175. | Preserve previously demonstrated capabilities during repair. Custom task families and expensive search limit transfer claims. [arxiv.org](https://arxiv.org/html/2609.38982v1?utm_source=chatgpt.com) |
| **Synthetic scene variants** | RAPID: **53.2±7.1→75.9±2.4%** across eight tasks, 50 scenes and three runs. | Qualification across variations is stronger than accepting a skill after one successful scene. [arxiv.org](https://arxiv.org/html/2609.30249v1?utm_source=chatgpt.com) |
| **Explicit physical feedback** | SenseNova local ablation: **15.56→24.72%**, **+9.16 pp**, 42 tasks × three layouts. | Effect feedback is high-priority. This is an independent variant versus the local baseline, not a sequential addition to every other improvement. [Technical Report](https://www.sensenova.ai/has-gpt-6-astra-achieved-embodied-intelligence?utm_source=chatgpt.com) |
| **History compression** | SenseNova: **15.56→21.11%**; **2.56M→0.86M tokens**, −66.4%. | Compact useful history can improve both SR and token use. Compression is not just dropping old frames indiscriminately. [Technical Report](https://www.sensenova.ai/has-gpt-6-astra-achieved-embodied-intelligence?utm_source=chatgpt.com) |
| **Hierarchical memory, hybrid** | RoboHarn-Evo: no memory **58.3%**, flat **64.2%**, full **82.5%**, n=120. | Strong memory-organization evidence, but with a π0.5 executor and no full RoboDojo evaluation. [arxiv.org](https://arxiv.org/html/2609.37583v2?utm_source=chatgpt.com) |

### Negative results that should shape the implementation

- **Guava:** per-turn feedback **88.6%**, multi-action program **82.9%**, n=105. Its matched-data study also loses **7.8 pp overall** when recovery/midtask branches are replaced with ordinary traces. [arxiv.org](https://arxiv.org/html/2606.18363v3?utm_source=chatgpt.com)
- **Show-Harness:** adaptive default **96%**, always chunk **74%**. Removing chunking keeps 96% but requires more calls. The useful target is adaptive duration, not maximum chunk length. [arxiv.org](https://arxiv.org/html/2609.10522v1?utm_source=chatgpt.com)
- **Sakana’s program archive:** initial source code is worse than no archive, **41.5% versus 45.2%**; optimized code reaches 57.0%. Retaining code is insufficient unless its quality has been established. [arxiv.org](https://arxiv.org/html/2609.19906v1?utm_source=chatgpt.com)
- **RHO, Skill-RSI and full SenseNova headline comparisons** do not isolate a clean “repair agent causes +X” effect: search budgets, inherited code, feedback, VLA access or inference budgets differ. The report retains those comparisons with their confounds instead of assigning a causal delta. [arxiv.org](https://arxiv.org/html/2606.16458v1?utm_source=chatgpt.com)

---

## 4. RoboDojo leaderboard, October 24, 2026

The table below includes **every entry above 25 Score**—which also includes every entry above 25% SR—plus the lower-scoring agent-labelled baselines. **All rows are officially listed.** The full 62-row snapshot is in the report.

I show Score and SR separately and sort across tracks for readability. The website’s track-specific ordering should not be interpreted as this combined ranking. Source: [official leaderboard](https://robodojo-benchmark.com/leaderboard), with values and protocol captured from its [public site data bundle](https://robodojo-benchmark.com/assets/index-DA5jHulR.js).

| Method | Score | Strict SR | Agentic interpretation |
|---|---:|---:|---|
| **PyRUA-Lean** | **55.26** | **50.35%** | VLA-free code-cell controller; official RGB configuration does not learn across episodes. |
| **HexaShell-Preview** | **53.18** | **46.85%** | VLA-free coding agent, RGB triangulation, retained tools and task notes. |
| **WorldScape-R²SI** | **46.58** | **39.35%** | **Agent + WAM hybrid**; full source released today. |
| **EmbodiedEvo** | **45.08** | **38.95%** | VLA-free coding agent; Playground artifacts frozen for fresh Test agents. |
| VPP2 | 39.26 | 32.26% | WAM-labelled; not a verified general coding-agent entry. |
| **PhysicalRSI** | **36.27** | **31.38%** | **Mixed code/VLA compositions**, selected through an agent. |
| Awomo-0.5 | 35.34 | 29.64% | WAM-labelled. |
| Simate-beta | 33.95 | 27.96% | VLA-labelled. |
| Rex-M1 Preview | 33.79 | 27.44% | VLA-labelled. |
| VPP2-Preview | 31.40 | 25.62% | WAM-labelled. |
| InternW0-Δ | 30.77 | 23.91% | WAM-labelled; included because Score exceeds 25. |
| **Liber-0 Preview** | **30.74** | **25.52%** | VLM-WAM-labelled; exact online agent/motor boundary unverified. |
| Liber-0 Lite | 29.24 | 24.23% | WAM-labelled. |
| **GPT-6-Astra** | **28.97** | **22.48%** | Direct VLM baseline. |
| **DeepSeek-Flash** | **2.99** | **1.92%** | Agent-labelled baseline; exact published adapter variant not fully resolved. |
| **GPT-5.5** | **1.13** | **0.88%** | Agent-labelled baseline; exact published adapter variant not fully resolved. |

The board’s protocol uses public released layouts for its main results and hidden verification as an auxiliary consistency check. It is therefore too strong to describe every listed result as exclusively hidden-test evaluation. Exact per-entry seed/episode rosters are not all exposed on the board.

### Important pending submissions and adapters

| Entry | What the primary submission establishes |
|---|---|
| **[REAL_RSI, PR #198](https://github.com/XPolicyLab/XPolicyLab/pull/198)** | Claims **61.92% SR / 68.31 Score**, 2,100 seed-0 episodes. Frozen generated programs, **no runtime LLM**, 42 base and 12 randomized task routes. Pending, not officially listed. |
| **[HarnessPolicy, PR #201](https://github.com/XPolicyLab/XPolicyLab/pull/201)** | Claims **33.83% SR / 40.12 Score**, 2,100 seed-0 episodes. DeepSeek-Flash + Simate VLA + SAM3. Opened today; advertised upstream unavailable when checked. |
| **[RunWu, PR #195](https://github.com/XPolicyLab/XPolicyLab/pull/195)** | Direct Astra/EEF tools, per-task bundles, public training keyframes and geometry priors. Verified submission figure is **generalization SR 43.83%**, not a complete aggregate. |
| **[SenseNova, PR #175](https://github.com/XPolicyLab/XPolicyLab/pull/175)** | π0.5 proposes action chunks and Astra reviews execution. Its report’s 54.17% is not yet a board entry. |
| **[EchoPolicy, PR #138](https://github.com/XPolicyLab/XPolicyLab/pull/138)** | Hybrid self-report **13.45% SR / 19.11 Score**, with replacements/runtime changes; no official listing. |
| **[RoboDojoEndpoint, PR #143](https://github.com/XPolicyLab/XPolicyLab/pull/143)** | Remote-agent integration and **2/2 stack-bowls smoke tests**, not a capability evaluation. |

**Open PR does not mean unofficial result:** PyRUA, HexaShell, EmbodiedEvo and WorldScape have official board listings independently of their adapter merge state.

The environment issue tracker also matters. [Issue #76](https://github.com/RoboDojo-Benchmark/RoboDojo/issues/76) requests optional wrist force/torque; do not assume contact sensing used by another benchmark is currently part of RoboDojo’s standard observation contract. [Issue #75](https://github.com/RoboDojo-Benchmark/RoboDojo/issues/75) questions Liber-0’s category but does not establish a resolved agent classification.

---

## 5. What to build: concrete mechanisms and effort

These are **my engineering estimates**, assuming familiarity with controlr and an existing functioning simulator connection. They exclude rollout costs and infrastructure delays.

### A. Persistent, bounded Python execution — **2–4 engineer-days**

Expose a robot object and let one model response execute a program containing perception queries, motion, checks, branches and bounded retries.

Keep your current safety envelope below every motion call. Add limits on native steps, wall time, retries and outstanding actions. The program should return control at meaningful checkpoints: uncertain localization, failed approach, grasp mismatch, release, settling, or stalled progress.

The design goal is **local closed-loop execution with selective model intervention**. URAI and Guava are the reason I would preserve those intervention points while adopting PyRUA-style code control. [arxiv.org](https://arxiv.org/html/2609.39018v1?utm_source=chatgpt.com)

### B. Geometry that returns evidence, not unexplained coordinates — **3–6 days**

A useful interface would include:

| Proposed operation | What it should return |
|---|---|
| `observe(...)` | Frames, robot state, timestamps, action epoch and calibration version. |
| `triangulate(...)` | Position, source views/pixels, reprojection residual and uncertainty. |
| `project_to_plane(...)` | Coordinate conditional on an explicit plane/height assumption. |
| `preview_pose(...)` | Projected gripper/target geometry and available IK/reachability checks. |
| `re_view(...)` | A physically acquired new wrist view with measured camera pose. |
| `verify_effect(...)` | Visual/proprioceptive evidence for the requested change and its uncertainty. |

Start with synchronized passive triangulation and pose overlays. Add physical wrist movement when visibility or triangulation baseline is inadequate. CodeActionBench and PhysEvo are particularly useful here; VIA contributes the pose-preview interface, while its virtual camera should not be mistaken for new sensing. [arxiv.org](https://arxiv.org/html/2609.33807v1?utm_source=chatgpt.com)

### C. Explicit effect checks and adaptive duration — **2–4 days**

Return compact, typed outcomes from each primitive:

- Reached target within tolerance.
- Executed only a prefix of the requested trajectory.
- Gripper closure differs from an empty grasp.
- Object moved with the gripper.
- Release occurred and the object settled.
- Evidence is stale or insufficient.
- Recovery exhausted its budget.

Use measured state and available images; keep the benchmark evaluator outside runtime decisions. The most useful upgrade is often turning an ambiguous “command completed” into an evidence-bearing physical result. SenseNova’s feedback ablation and Show-Harness’s recovery/chunking results support this priority. [Technical Report](https://www.sensenova.ai/has-gpt-6-astra-achieved-embodied-intelligence?utm_source=chatgpt.com)

### D. Repair agent with physical promotion gates — **3–5 days**

After a development episode, give a separate reviewer the trace, code, images and failure evidence. Require a proposal containing:

1. The earliest supported failure.
2. A minimal code or tool change.
3. Which contracts and skills it affects.
4. Paired development results.
5. Regression results on previously solved capabilities.

Run syntax/unit checks first, then physical cases. Retest the **merged** candidate, not just its constituent patches. Preserve rejected changes and reasons so the agent does not repeatedly rediscover them.

RACaP supplies the patch/promotion pattern; RPG makes merged-library regression explicit; SimEX motivates the capability bank. Freeze prompts, tools, skills, calibration and perception versions before held-out evaluation. [arxiv.org](https://arxiv.org/html/2609.29394v1?utm_source=chatgpt.com)

### E. Distill or compile after the interface stabilizes

The released assets worth inspecting first are:

- [Guava-4B](https://huggingface.co/AIcell/guava-v13b-qwen3.5-4b): teacher reasoning/tool traces plus recovery branches.
- [PyRUA-Lean 4B coder](https://huggingface.co/DAGroup-PKU/PyRUA-Lean-Coder-v0.1-4B): GRPO-trained coder, not simply a distilled Astra policy.
- [RAX robot session logs](https://huggingface.co/datasets/robot-agent-exploration/robot-session-logs): trace, code, skill and sensor-evidence formats.
- [SenseNova experience packages](https://huggingface.co/datasets/sensenova/SenseNova-RoboRSI-Packages): versioned experience artifacts.
- [Show-Harness models](https://huggingface.co/showlab/Show-Harness-VLMs) and [data](https://huggingface.co/datasets/showlab/Show-Harness-Data): compact direct-motion models.
- [AgenticRoboRL 9B checkpoints](https://huggingface.co/ciciciciliu/AgenticRoboRL-Qwen3.5-9B): useful post-training reference, with the hybrid/privileged-route distinctions preserved.

I would budget **1–2 weeks for a first distillation pipeline after enough validated traces exist**, rather than choosing a small model before defining the final control API.

Cost accounting needs both acquisition and deployment. RHO spends thousands of dollars developing a zero-runtime-codegen policy; Skill-RSI makes successful reuse extremely cheap but incurs failed acquisition campaigns; Astranomy’s published agentic dollar averages condition on successful episodes. Those figures answer different economic questions. [arxiv.org](https://arxiv.org/html/2606.16458v1?utm_source=chatgpt.com)

---

## 6. Cheap experiments that would resolve real gaps

I did **not** find verified matched RoboDojo ablations isolating the following questions. Related ingredients already exist, so these are evidence gaps rather than claims of untouched novelty.

| Experiment | Design | Why it would be useful |
|---|---|---|
| **Separate code execution from better tools** | **2 × 2:** per-step versus bounded code; current versus geometry/effect tools. Six tasks × 20 paired held-out layouts × four conditions = **480 episodes**, after separate development scenes. | Identifies whether your gain comes from representation, better physical information, or their interaction. |
| **Uncertainty-triggered wrist re-view** | Compare passive stereo, always re-view, and re-view only above an uncertainty/occlusion threshold, under equal sensing/action budgets. | CodeActionBench and PhysEvo use active views, but their evidence does not isolate this scheduling rule. |
| **Calibration-dependent skill invalidation** | Version camera, tool-tip and gripper calibration. Perturb one estimate in development; compare retaining everything, clearing everything, and selectively revalidating dependent skills. | KnowBody motivates dependency tracking, but its value for a RoboDojo code-skill library remains unmeasured. |
| **Executed-action acknowledgment** | Require an executed-prefix acknowledgment and observation epoch. Introduce interrupted chunks or stale frames in development. | WorldScape/OpenETA contain related lifecycle mechanisms; the separate success contribution is unquantified. |
| **Generic repair versus task recipe** | Give equal acquisition budgets to a shared-tool repair agent and a per-task recipe agent. Test seed holdout and task holdout separately. | Strong leaderboard systems often contain substantial task engineering; this measures how much of the gain transfers. |
| **Phase-triggered versus anomaly-triggered calls** | Compare fixed-step calls, phase boundaries and uncertainty/anomaly triggers. Count every attempt, retry, repair and abandoned generation. | Tests whether better handoff timing saves cost without sacrificing success. Related factorial plans already appear in [robot-agent-eval](https://github.com/gorgeouslife3/robot-agent-eval/blob/main/experiments/supervision-efficiency/plan.zh-CN.md). |

For every condition, retain **strict SR, progress, paired rescued/lost episodes, native steps, model calls, cached/uncached tokens, wall time and total dollars per success**. Keep failed and interrupted attempts in the denominator. Your first useful result should be a causal comparison on a frozen layout set, before adding several repair, memory and perception mechanisms simultaneously.

### Coverage limits

I enumerated **193 XPolicyLab PRs**, **28 RoboDojo environment PRs**, and their issue records; mined Physical-RSI; followed the requested seed references and fresh forward-search leads; queried all six requested GitHub topics; checked seven seed fork lists; and inspected relevant Hugging Face and primary lab/company releases.

The remaining gaps are explicit: **stargazer endpoints returned 401**, some fork/topic lists were capped at 100, authenticated code-import search was unavailable, Semantic Scholar returned 429, and inaccessible WeChat/unindexed X material prevents an exhaustive claim. The report therefore gives a broad, source-audited map with named unknowns, rather than claiming that every project on the internet has been found.