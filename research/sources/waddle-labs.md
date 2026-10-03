# Waddle Labs (waddlelabs.ai): deep dive

*Researched 2026-10-02. I read every page in the site's sitemap, both posts in full, the YC company/launch/job pages, the founders' sites, the YouTube description, Wayback snapshots and the open-source `waddlelabs/waddle-sdk` (shallow clone of `3b5cd67`, 2026-09-30, deleted after reading). I downloaded no videos; demos are described from their captions (`aria-label`s) only. The Fig. 4 numbers are my own pixel digitization of `/launch_alpha.png` (**[derived]**, roughly ±3 pp). UNVERIFIED means I could not confirm it at a primary source.*

## TL;DR

- **Who.** Two founders, Hanming Ye and Yiding "Vincent" Song, Harvard roommates who met at MIT RSI. Y Combinator **Summer 2026**, San Francisco, team size 2. The YC job post states **"$19M in seed funding"**. On positioning, the job post says: "Our thesis is that generalist models like LLMs will perform better than robot-specific models like VLAs. This means that we are counter-positioned against companies like Physical Intelligence."
- **What.** They pitch "Claude Code for robotics": a *hosted* LLM agent that **writes Python control programs**. This is code-as-policy. The programs are built from a growing shared **skill library**, sit on top of a fixed vocabulary of platform **primitives** (perception queries, IK, trajectory planning, `approach_until` contact) and can call VLAs as tools. A master agent spawns sub-agents, one per robot. The stack has three layers:
  - `waddle-sdk`: open source, Apache-2.0, hardware plus safety envelope;
  - `waddle-metal`: proprietary "IR" of tools;
  - `waddle`: the closed-source harness.
- **LLM role.** **Programmer and orchestrator, not controller.** Motion comes from IK and trajectory primitives that the SDK runs at 25 Hz by default (YAM), so the LLM is not in the servo loop. This is the design point controlr explicitly rejects (no skills, model emits numeric deltas).
- **Evidence is thin.** There are 12 short real-hardware demo clips, and one quantitative figure: success against cost per task for **Opus 4.8, Fable 5 and GPT 5.6 Sol** at high/xhigh thinking. The figure gives no task list, no trial counts, no seeds and no CIs.
  - Digitized xhigh plateaus: Fable 5 ≈87%, GPT 5.6 Sol ≈83%, Opus 4.8 ≈67%.
  - High plateaus: ≈70%, ≈73% and ≈53%.
  - The curves flatten at about **$50–85 per task**.
  - **No latency number appears anywhere**, even though the homepage claims "real time, on real robots".
- **For controlr.** They do not compete on the research question, since they sidestep direct numeric control. They are a well-funded competitor for the "LLM agents control robots" narrative. Worth borrowing:
  - their envelope recovery rule (which addresses controlr's straight-elbow trap);
  - per-action provenance and intervention logging;
  - success-vs-cost curves across effort levels;
  - a cross-episode "lessons" memory as a config axis;
  - perception *query* tools (not action skills);
  - their Apache-2.0 32-task MuJoCo catalogue.

---

## 1. Company, people, money, timeline

| Item | Finding | Source |
|---|---|---|
| Self-description | "We are an applied AI lab building LLMs that control robots." | homepage |
| Founders | **Hanming Ye**: Harvard math, TA for Math 55. Prior research: robot learning with Prof. Yilun Du, spiking nets with Haim Sompolinsky, predictive coding, and algebraic topology at RSI. **Yiding Song (Vincent)**: Harvard CS (Mind Brain Behaviour), Gershman Lab / Kempner Institute: "how Transformers meta-learn to perform program induction in-context". The job post says both "studied math/physics at Harvard before leaving to start Waddle Labs"; that contradicts Yiding's own "CS (MBB)" | dozenducc.github.io, yiding.rocks, YC job |
| Relevant papers | Ye & Song, *Model Capacity Determines Grokking…* (arXiv 2605.09724, 2026-05-10). Ye, Song, Du, *Few-shot Task Learning via Compositional Concept Inference*, "NeurIPS 2026" (listed on Hanming's site only; no arXiv found; UNVERIFIED). Ye, *Steering Diffusion Policies with Value-Guided Denoising*, NeurIPS 2025 workshop (code: `DozenDucc/VGD`). No robotics-agent paper from the company | founder sites, arXiv |
| YC | **Summer 2026**, "Founded: 2025", team size 2, primary partner Ankit Gupta. Tags: Developer Tools, Generative AI, Robotics, Automation. Launch post "Waddle Labs - Agents that control robots", 2026-07-27, 11 votes | ycombinator.com/companies/waddle-labs, launch JSON |
| Funding | "We've raised $19M in seed funding, with participation from researchers at OpenAI, Google Deepmind, and Cognition." Single source; no lead investor named (UNVERIFIED beyond the job post) | YC job post |
| Job | "Founding Research Engineer", $175K–$350K, 1–2%. Skills: PyTorch, distributed systems, fine-tuning, evals, AI agents, diffusion models. Duties: "agent harnesses, robot simulation and evaluation environments, trace-collection pipelines, and infrastructure for post-training models". The role will "deploy it on robots in factories" | YC job post |
| GitHub | Org `waddlelabs` was created 2025-06-10 (email developers@waddlelabs.ai, X `@thewaddlelabs`). It has one original repo, `waddle-sdk` (created 2026-07-14). Contributor counts: PerceptronV (Yiding) 421 commits, DozenDucc (Hanming) 4. The other repos are forks of LeRobot, joycon-robotics and dora | GitHub API |
| PyPI | `waddle-sdk` 0.0.0 (2026-08-05) → 0.1.11 (2026-08-31). Summary: "Python frontend for Waddle: supervision for real-world robot policy rollouts" | pypi.org JSON |
| Timeline | Wayback has `/contact` in January 2026. The post says "We have been using agents to control robots for the last six months" (cited as Jul 2026, so since about January). Introducing post archived 2026-09-08 (body unchanged since). The homepage on 2026-09-14 was product copy ("Connect our API to your robot. / Enter a prompt… / Our agents create a policy"); it now reads as a research lab with three research directions | Wayback CDX + diff |
| Other | LinkedIn: "We build AI agents that control robots", 2 employees. YouTube `@Waddlelabsrobotics`: 1 video (92 s, 1,756 views). No HN, Reddit, press or Hugging Face presence found. The X timeline was not readable (syndication 429) | — |

## 2. Site inventory

`sitemap.xml` lists exactly 5 URLs: `/`, `/writings`, `/early-access` (an embedded Tally form, `tally.so/r/xXE9oG`), `/research/introducing-waddle` and `/developers/waddle-stack`. `/research` and `/developers` redirect (307) to `/writings`. The layout JS also matches a `/tags/` prefix, but every `/tags/*` URL I probed returned 404. Both posts are fully server-rendered HTML, so no browser rendering was needed. `/writings` lists only these two posts.

## 3. "Introducing Waddle: agents that control robots" (Ye, Song; "Jul 2026")

**Thesis.** The post argues that VLAs/WAMs need "huge amounts of robot data", are "difficult to steer" and "do not yet generalize across embodiments". LLMs, in contrast, "are also uniquely capable of tool use, reasoning, and writing programs. Our hypothesis is that these capabilities can be transferred into robotics by using LLM agents to control robots."

**Architecture as stated.** "Our agents decompose goals into subtasks, and complete each one by viewing camera feeds, writing control code, and calling models like VLAs. The agent outputs a program that you can run and iterate on by talking to our agents." Fig. 1 caption: "an agent uses code as policy while retaining the ability to call action models as tools." The YouTube description adds that the agent "has access to a robot's cameras and control interface via MCP tools" and builds "VLM-based verifiability" into each stage.

**Three-level hierarchy (Fig. 2).**
- **Primitives** are a "fixed vocabulary provided by the platform":
  - `bounding_box(·)`: "perception: text query → box in frame";
  - `detect_in_base(·)`: "box → point in robot base frame";
  - `approach_until(·)`: "waypoints + stop criterion → trajectory";
  - `reset_home(·)`.
- **Skills** are "Created and refined by agents; shared in the library", for example `servo_align`, `orbit_view`, `top_grasp` and `fold_grasp`.
- **Programs** are written per task. The example `fold_tshirt.py` uses `parallel(left, right)`, `fold_grasp(anchor=f"{arm} sleeve")`, `robot[arm].execute(traj)`, `barrier()`, `fold_over(axis="midline", hold="left cuff")` and `verify("sleeves folded")`.
- The `fold_grasp` skill body is: `bounding_box(anchor)` → `detect_in_base(box)` → `preset("low sweep", at=point)` → `approach_until(wps, until="contact")`.

**Memory and continual learning.** "every solved task adds skills to a library shared by all agents. Programs compose, and nothing retrains between tasks." The only evidence offered is that `fold_grasp` was "first created … when attempting to flip over a package. Then, another agent adapted this skill to fold a t-shirt" (Fig. 3: flip package, fold towel, fold shirt). That is one anecdote with no ablation.

**Models and scaling (Fig. 4).** "We evaluated three LLMs (Opus 4.8, Fable 5, and GPT 5.6 Sol) across a suite of manipulation tasks and observe a consistent trend: larger models, given larger budgets, produce better policies. All models were capable of achieving easy tasks, such as 'pick up the lego', but only Fable 5 and GPT 5.6 with xhigh thinking accomplished harder tasks like 'fold the t-shirt'." No exact API model IDs are given. These models are a generation older than the ones controlr runs (Opus 5.5, Fable 5.1, GPT-6 Astra; see REPORT §1).

Fig. 4 is titled (read from the image) "Success scales with model size and thinking budget". Its axes are average success rate (0–100%) against cost per task ($0–$100); bold curves are xhigh and light curves are high. Digitized values:

| Model | xhigh plateau | ≈ cost to plateau | high plateau | ≈ cost to plateau |
|---|---|---|---|---|
| Fable 5 | ≈87% | ≈$70 | ≈70% | ≈$69 |
| GPT 5.6 Sol | ≈83% | ≈$52 | ≈73% | ≈$46 |
| Opus 4.8 | ≈67% | ≈$85 | ≈53% | ≈$52 |

At low budgets *high* beats *xhigh*: around $13, GPT 5.6 Sol high is ≈46% against xhigh ≈20% **[derived]**. The task suite, trial counts, seeds, success criterion and hardware or sim are **not stated**. The "cost" is the agent's authoring spend per task, not a per-episode runtime cost.

**Use cases (all demos, no numbers).**
- (a) "Create a working policy in 20 minutes", for example "place one microswitch inside each slot".
- (b) Data generation: "Pick and place lego bricks at random positions 1000 times". The agent "repeated this around a thousand times overnight and autonomously trained an ACT policy from scratch that could pick up LEGOs". No success rate is given for the collector or for ACT.
- (c) "robotics auto-research": scene resets between overnight policy-tuning trials.

**Videos (captions only).**
- An agent-written program folds a t-shirt.
- Inserting a USB connector.
- "handling a package on an xArm".
- A long-horizon task end to end.
- "Subagents coordinating multiple robots".
- Prompt to microswitch-insertion policy.
- Autonomous data collection, first and later trials.
- Scene reset between tuning trials.
- Flip package / fold towel / fold shirt.
- The embedded YouTube launch video `fGRYBtmzwoI`.

**Hardware.** The post names only the xArm. The Stack post and SDK add **i2rt YAM** (with bimanual site examples), **Synria Alicia** arms and **xArm 7**; the SO-101 appears in simulation only. Cameras: RealSense and Orbbec.

**Latency / control rate.** The post gives no number for either. The homepage says only: "We have shown that language models can solve robotics tasks in real time, on real robots, but latency remains a bottleneck. How can agents become faster with practice?" With code-as-policy, "real time" means the *program* runs in real time; the LLM spend happens beforehand (≈20 min per task as claimed).

**Next steps they list.**
- "Tools that agents prefer". They cite VIA (arXiv 2607.11119) and Claude Plays Robotics: "adding a movable cursor the model could query for position and depth raised success on a manipulation suite from 6% to 32%".
- A shared benchmark: "A shared benchmark for agent-controlled robots, with common tasks and success criteria, would make results comparable across systems."
- "Training more capable agents… We have been using agents to control robots for the last six months, generating large amounts of data and intervention traces. We plan to use this data to train LLMs that excel at performing physical tasks."

**Related work they position against.** Code as Policies, VoxPoser, L2R, SayCan, Inner Monologue, Voyager, and the 2026 works **CaP-X** (arXiv 2603.22435) and **ASPIRE** (arXiv 2607.00272). Their summary: "Waddle runs the loop as a deployed system: robots connect through our API, agents run against them continuously". There are no code links in the post.

## 4. "The Waddle Stack" (Song, Ye)

**Three layers.**
- **`waddle-sdk`** "handles all communication with hardware… Enforcing the hardware safety envelope, capping joint speed limits, sending joint angles to motors, and reading from cameras."
- **`waddle-metal`** "defines skills… Inverse kinematics solvers, trajectory planning, object segmentation, and moving a robot arm to a 3D pose." It is proprietary and described as an IR, by analogy to LLVM: "Agents write the same code and use the same tools regardless of hardware."
- **`waddle`** is the "closed-source server for hosting the agent harness. It routes models, builds a closed-loop environment for agentic reasoning, and handles context management. Examples: Sub-agent orchestration, MCP tools, skills library, compaction, reading / writing / running waddle-metal code."

**Figure labels (from `waddle.svg`).** "model routing" through GPT, Claude and Waddle adapters (so their own model is planned); "proprietary" above "open source".

**Graceful degradation.** "explicit capability matrices… if waddle-sdk does not declare depth information for a particular camera … The agent is restricted to tools available with RGB—such as visual masking and object bounding boxes—instead of failing repeatedly on depth-enabled tools such as point clouds."

**Porting contract for an arm.** `kind`, `estopped`, `read()` (joint position and velocity), `write(target)` (joint positions), `hold()`, `estop()`, `re_enable()`, `home(pose)` and `close()`. Optional additions unlock more: velocity feed-forward, a URDF for FK/IK and planning, collision spheres, and a gripper. A camera needs `capture() → CameraFrame` and `close()`, with depth optional.

## 5. What the open `waddle-sdk` code reveals

- **Scope.** It is the hardware-owning layer: "loads a strict site manifest, opens robot and camera drivers, enforces the owner envelope, and records timestamped raw evidence". Claims, leases, gating, clocks and recording live in a **Rust core**; Python is "a hollow frontend". It connects to `https://connect.waddlelabs.ai` with a per-customer API key and opens hardware only after the host accepts.
- **Action interface on the wire.** `waddle.v0` `Action` is a oneof of `joint_position`, `joint_velocity`, `ee_delta` (Twist), `ee_absolute`, `base_twist`, `composite` and `opaque`, plus an optional gripper command. Steps come in an `ActionChunk` ("the unit policies emit (10–50 steps typical)") carrying `t_obs_ns` "for staleness accounting".
- **Rates and limits (YAM defaults).** `DEFAULT_RATE_HZ = 25.0` ("deliberately far below the vendor's ~1 kHz servo"), `DEFAULT_MAX_JOINT_SPEED_RAD_S = 1.0`, and per-joint max position error of 0.04–0.2 rad. The YAM tabletop workspace preset is ±0.7 m in x and y, −0.015 to 1.0 m in z.
- **How the agent drives the robot.** Through **agent-invited episodes** (flag `waddle.v0.agent`): "The invited agent claims, engages, streams chunks, and finishes through the EXISTING intervention machinery". The hosted agent therefore streams action chunks from Waddle's cloud through the same gate a teleoperator uses.
  - Client API (`_core.pyi`): `agent(prompt, timeout_ns, task_metadata, pre_reset_prompt, post_reset_prompt, …) -> AgentResult`. Resets are themselves prompted agent tasks.
  - The agent's perception on the control plane is "Bounded-rate stills" (`FrameStill`, `still_fps`), not video; video goes over LiveKit for humans.
  - Durable "hosted-task conversations" support **MESSAGE, INTERJECT and INTERRUPT**. This is the HRI surface ("give robots tasks, see what they are doing, and correct mistakes").
  - A calibration flag resolves "one pixel against an exact retained frame" to a 3-D point (`detect_in_base`-style grounding) without images leaving the site.
- **Envelope semantics worth noting.**
  - "one refusal holds the addressed set and moves none of it". Commands are refused whole, not shortened.
  - "If measured feedback is already outside a workspace box, the SDK admits an intermediate command that reduces at least one TCP or robot-body violation without worsening or introducing any other violation."
  - Missing collision geometry "fail[s] closed".
- **Sim task suite.** MuJoCo, SAPIEN and Isaac/Gazebo (via ROS 2) backends run SO-101, YAM and xArm7. There are **"thirty-two interactive development task environments"** (21 single-arm, 11 two-arm), including `pick_lift`, `insert-usb`, `insert-peg`, `use-hook`, `stack-three-cubes`, `candy-bin-transfer`, `shampoo-packing`, `chocolate-packing`, `handover-block`, `two-arm-peg-insertion` and `loaded-tray-transport`. The docs say: "verify mechanics without adding task routes or success logic to the simulator". Most September 2026 commits are contact-stability work on these scenes. They are plausibly the "task suite" behind Fig. 4 (UNVERIFIED).
- **Earlier product framing.** The design-rationale doc (`waddle_api_design_doc.md`, "Draft v0.9", marked historical) describes a **"supervision layer for real-world robot policy rollouts"** — "Weights & Biases instrumented your training loop; the supervision service instruments your deployment loop" — with watch / intervene (teleop or "code-as-policy agents") / reset / VLM-judge / improve (filtered BC, HIL-SERL, RLPD). It also says: "Latency added to a customer's 50 Hz loop is a bug of the highest severity." Its self-critique flags judge Goodharting and adopts a "held-out, human-labeled audit slice" and pinned judge versions.

## 6. Comparison with controlr

| Axis | Waddle | controlr |
|---|---|---|
| LLM role | Writes and iterates **programs**; orchestrates sub-agents; calls VLAs and perception tools | **Is the controller**: one reply per turn = numeric `MOVE ee_delta dx dy dz [dyaw]` + `STATUS` |
| Action space at the LLM | Python over primitives/skills (IK, `approach_until`, presets, `verify`) | Bounded ee deltas in mm/deg, ≤ `max_chunk` lines; no skills (CLAUDE.md rule 1) |
| Action space at the robot | Joint-position / ee chunks at 25 Hz through the SDK gate | Each line → IK every 5 mm → one min-jerk motion; physics paused between turns |
| Real-time | Program runs in real time; LLM time spent at authoring (≈20 min/task claimed) | Turn-based; LLM p50 3.4–9.6 s (Opus 5.5) and turn cycle p50 ≈10.6–10.9 s in Isaac (smoke-v0) |
| Observation | Low-rate stills, perception primitives (text query → box → base point), depth when declared | Rendered frames with grid / axes / ee_marker / diff overlays + deterministic feedback text |
| Memory | Persistent cross-agent skill library, compaction, sub-agents | Append-only cached transcript per episode (cache-read share 0.81–0.93); `prompt.fewshot` appendix; separate planner |
| Safety | SDK owner envelope in Rust core: whole-command refusal, keep-outs, self-collision, speed and position-error caps, leases, hold-first handoff, e-stop | `SafetyEnvelope` in code: clamp/shorten with the reason reported to the model, fingertip-table clearance, stops on contact force |
| Eval rigor | One figure without n, tasks or seeds; demos | Configs in git, seeds, `success_verified`, n reported ("2/4 is noise"); still tiny n (2/4 waffle, 2/10 yaw) |
| Hardware | Real YAM (bimanual), xArm 7, Alicia; multi-robot | UR3 in Isaac Sim; real UR3 next |
| Data flywheel | Six months of agent traces + interventions → plan to post-train LLMs; LEGO → ACT | Per-turn (obs, action, obs′) logs; LeRobot export planned |

**Where they are ahead.**
- Real hardware breadth, including deformables (t-shirt), insertion and bimanual or multi-robot work.
- A productized connector with serious safety and supervision engineering.
- Autonomous data collection plus policy training, and resets as agent tasks.
- $19M and hiring.
- Their architecture matches what the field evidence favours. REPORT §1: direct low-level control by frontier models "fails"; the strongest systems put "a frontier LLM over a typed, verifiable tool surface".

**Where controlr differs or is stronger.**
- **It asks the question Waddle sidesteps:** can the API model *itself* be the closed-loop controller? Fig. 4 has no direct-control baseline, so Waddle cannot say how much of its success comes from the LLM rather than from IK/primitives and hand-written presets (`preset("low sweep")`).
- controlr's per-turn transcripts are fine-grained (obs, action, obs′) control data, rather than program text.
- Cache discipline and latency measurement are explicit and reported (TTFT, cache share, regressions).
- The experiment protocol (seeds, pinned plans, verified success) is stricter than anything Waddle has published.

**Competitive threat.** For the research thesis it is **low**: different design point, and Waddle publishes no numbers on direct control. For the narrative it is **medium**. "LLM agents that control robots", hardware-agnostic connectors and "Claude Code for robotics" are the framing a funded, visible company now owns, alongside harness-layer players already in REPORT §1.9 (dimOS, OM1, AWS Strands, RPent). Waddle's stated plan to post-train LLMs on intervention traces would put it into the model layer as well.

## 7. What controlr should borrow

1. **Envelope recovery rule (directly relevant).** controlr's straight-elbow trap (rotation-yaw report: the swing guard refused every move for 17 turns) is the kind of failure Waddle's rule targets: when the state is already in violation, admit a command that strictly reduces at least one violation without worsening others. controlr's 8° elbow fix is a special case of this. Generalize it in `SafetyEnvelope` and report "recovery move admitted" as feedback.
2. **Per-action provenance and intervention events.** For the real UR3 (a human at the e-stop), tag every executed action with its source (`llm` / `safety-clamp` / `human`) and log holds, e-stops and resumes as typed events in `turns.jsonl`. Waddle treats these as "DAgger-gold" data, and they will matter if controlr ever trains an action head.
3. **Success-vs-cost curves across effort levels.** Fig. 4 is the right *shape* for controlr's model/effort sweeps: verified success against cumulative $ per episode (controlr already logs tokens). Show high vs xhigh, because the crossover at low budget is a real decision variable. Unlike Waddle, publish n, seeds and Wilson CIs.
4. **Cross-episode memory as a config axis.** "Agents become markedly more reliable over time" is Waddle's core (unquantified) claim. The low-level analogue that fits controlr's rules is a curated "lessons" appendix (for example "turn the jaws low", "keep the elbow bent", from past run logs) in the cached prefix. That is prompt content, not skills. Ablate it against a pinned plan.
5. **Perception *query* tools, not action skills.** `bounding_box` / `detect_in_base` and the cursor result they cite (6 → 32%, already in REPORT) suggest an optional observation tool: text query → pixel box → base-frame point, returned as feedback text. The model still emits every motion, so rule 1 holds.
6. **Capability-matrix degradation.** Derive the set of offered renderers and feedback fields from what the backend declares (depth, calibration, a second camera) instead of from config alone. This prevents prompts that mention unavailable signals.
7. **Task catalogue.** Their 32 Apache-2.0 MuJoCo task definitions (geometry, masses, success intent) are a ready list of reach / push / insert / stack variants for controlr's next tasks. `waddle-sdk sim init <urdf>` could even compile a UR3 URDF into a GPU-free MuJoCo scene, though that needs a package install, so evaluate it first.

**Contact?** There are two plausible reasons, both for Ilia to decide (I sent nothing):
- (a) Their call for "a shared benchmark for agent-controlled robots, with common tasks and success criteria" fits controlr's evaluation discipline, and a direct-control baseline on their task suite would be a useful joint result.
- (b) Ask whether they measured direct numeric control as a baseline.

Research contact is founders@waddlelabs.ai; early access is wave@waddlelabs.ai (the `/early-access` Tally form asks for robot and task).

## 8. Unverified / unreachable

- **Funding.** "$19M seed" and the investors appear only in the YC job post; no lead or round date found.
- **Fig. 4.** Task suite, n, seeds, sim vs real and success criterion are all unstated. My plateau and cost values are a pixel digitization.
- **Unquantified claims.** "Real time" and "markedly more reliable over time" come with no numbers.
- **Data collection.** The LEGO "≈1000 times overnight" claim gives no success rate for the collector or for the ACT policy.
- **NeurIPS 2026 paper.** The Ye/Song/Du *Few-shot Task Learning via Compositional Concept Inference* is listed only on Hanming's site.
- **Unreachable.** The X account `@theWaddleLabs` (syndication rate-limited) and the LinkedIn posts. No HN, Reddit, press or Hugging Face artifacts exist.
- **Closed components.** Contents of the closed `waddle` harness and `waddle-metal` (prompting, model IDs, context management, MCP tool list).

## Sources

- https://www.waddlelabs.ai/ (homepage; Wayback 2026-09-14 snapshot for the earlier product copy)
- https://www.waddlelabs.ai/sitemap.xml
- https://www.waddlelabs.ai/writings
- https://www.waddlelabs.ai/early-access
- https://www.waddlelabs.ai/research/introducing-waddle (Wayback 2026-09-08 snapshot diffed: body unchanged)
- https://www.waddlelabs.ai/launch_alpha.png (Fig. 4; digitized)
- https://www.waddlelabs.ai/developers/waddle-stack, plus `/developers/waddle-stack/waddle.svg` and `/developers/waddle-stack/llvm.svg`
- https://www.youtube.com/watch?v=fGRYBtmzwoI (description and metadata only); channel RSS `UCa_mJUXyOP_x0PWkd50kh1Q`
- https://www.ycombinator.com/companies/waddle-labs
- https://www.ycombinator.com/launches/S33-waddle-labs-agents-that-control-robots
- https://www.ycombinator.com/companies/waddle-labs/jobs/RaRUAQK-founding-research-engineer
- https://www.linkedin.com/company/waddlelabs/ (public header only)
- https://github.com/waddlelabs (org API)
- https://github.com/waddlelabs/waddle-sdk @ `3b5cd67`: `README.md`, `sdk/README.md`, `AGENTS.md`, `docs/concepts/index.md`, `docs/python/site.md`, `docs/python/simulation.md`, `waddle-protocol/proto/waddle/v0/control.proto`, `services.proto`, `waddle-protocol/docs/VERSIONING.md`, `waddle-protocol/docs/rationale/waddle_api_design_doc.md`, `sdk/python/waddle_sdk/robots/yam.py`, `_core.pyi`, `docs/changelogs/CHANGELOG-0.1.0.md`
- https://pypi.org/project/waddle-sdk/
- https://dozenducc.github.io/ (repo `DozenDucc/dozenducc.github.io`); https://yiding.rocks/ (+ `/projects`)
- arXiv: 2605.09724 (grokking, Song & Ye), 2603.22435 (CaP-X), 2607.00272 (ASPIRE), 2607.11119 (VIA)
- controlr context: `ARCHITECTURE.md`, `research/REPORT.md` §1/§3, `docs/experiments/2026-10-02-smoke-v0.md`, `docs/experiments/2026-10-02-rotation-yaw.md`
