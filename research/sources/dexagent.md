# DexAgent: an agentic Human2Sim2Robot framework (Stanford/Columbia, 2026-09-28)

Deep-dive written for the design of an Opus-backbone robot harness.

**Sources read:**
- the project page `dexagent1.github.io`, plus its git history;
- the anonymized twin `dexagent123.github.io`;
- the full 15-page paper (arXiv 2609.35318v1; PDF hosted on the site);
- both project videos, frame by frame;
- the TL;DR thread on X;
- alphaXiv;
- the cited baseline papers (Do-as-I-Do, SPIDER) to cross-check the tables.

**No code is released**: the site says "Code (Coming Soon!)", and the only GitHub repos are the two Pages sites. So there is no code walkthrough. Section 4 covers the artifacts that *are* public (JSON schema, verifier pseudo-code, MJCF snippet, library taxonomy). Anything not checkable against a primary artifact is marked **UNVERIFIED**.

---

## 1. Overview

| | |
|---|---|
| Title | *DexAgent: An Agentic Human2Sim2Robot Framework for Dexterous Manipulation with Self-Evolving Tool Library* |
| arXiv | 2609.35318v1, cs.RO, submitted Mon 28 Sep 2026 14:49 UTC. 15 pages, 8 figures, 4 tables |
| Authors | Youhui (Jeffrey) Wang¹ (corresponding, `jeffreywang0303@cs.stanford.edu`; X bio: "Undergrad @Harvard & @Cornell, Researcher @StanfordAILab @StanfordSVL"), Yunzhu Li² (Columbia), Li Fei-Fei¹, Jiajun Wu¹†, Huang Huang¹†. ¹Stanford, ²Columbia, † equal advising |
| Acknowledged | "We thank Sharpa for equipment support" |
| Code | Not released ("Coming Soon!"). GitHub org `dexagent1` was created 2026-09-22 and holds only the Pages repo |
| Announcement | X thread, @jwang633, 2026-09-29 19:12 UTC. At the time of reading: about 9.6k views, 65 likes, 10 reposts, 3 replies, 47 bookmarks. alphaXiv: #16 trending, 513 views, no public comments |

**What it is, in one sentence.** An *offline* LLM/VLM coding agent turns **one egocentric RGB human video plus a task prompt** into a verified bimanual dexterous robot trajectory in simulation. It then augments that trajectory into **500 rendered episodes**, which are used to **fine-tune π0.5**. The LLM never controls the robot at runtime: the deployed controller is the VLA.

The paper describes the agent as a tool-making agent that:
- picks or writes **skills** (Python functions that produce an asset, pose or motion);
- picks or writes **verifiers** (Python functions that accept or reject a skill output using simulation);
- persists both in a **"self-evolving tool library"**, which grows to 103 skills and 188 verifiers.

**Headline claims.**
- Policy success averages **63.6%** over 11 real tasks, against **18.2%** for the best baseline (SPIDER), which is "3.5×".
- Open-loop replay succeeds on **11/11** tasks, against 6/11 at best for the baselines.
- Processing takes **2.1 h** per video on average, against 3.3 h for GPT-6 Astra and 3.7 h for V2D.

---

## 2. Architecture

```
 egocentric RGB video (RealSense D435, 640x480 @30fps)  +  NL task prompt
            │
            ▼
 ┌──────────────────────── VLM/LLM agent (backbone: UNVERIFIED, "any VLM switched between API") ────────────────────────┐
 │  per stage: select skill(s) from library ─or─ write new skill  →  run  →  run stage verifiers  →  feedback → refine    │
 │  loop until all verifiers pass OR budget exhausted (cap: "a given budget or 100 agent turns"); fail ⇒ recorded failure │
 │                                                                                                                         │
 │  S1 Semantic understanding ──► scene_semantics.json {objects[type,properties,parts], subgoals[acting_hand,hand_roles, │
 │     (frames + depth / gravity /                      goal], env}                                                       │
 │      hand-pose skills)                                                                                                 │
 │  S2 Property-based sim reconstruction ──► MJCF/URDF assets (+ verifiers per property), placement by sim↔video mask   │
 │     IoU, camera from depth + robot-hand/human-hand overlay ──► scene file (robot + objects, original camera)          │
 │  S3 Trajectory optimization ──► per-subgoal success condition → target hand pose (retarget / contact-opt / grasp     │
 │     synth) → pose verifiers (lift+shake, slide+spin) → IK for arm → agent-generated code connects key poses           │
 │     (optional human-motion prior) → whole-trajectory verifiers                                                         │
 │  S4 Data generation ──► 1,500 random object-placement seeds → feasibility check → 500 passing episodes;              │
 │     VOID inpaints human+objects out of source video → composite sim robot/objects → Blender retexture;               │
 │     render ego + left-wrist + right-wrist views                                                                        │
 └────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────┘
                     ▲  read/write           │
        Self-evolving tool library           ▼
        (typed Python skills + verifiers;    fine-tune π0.5 on 500 eps (3 views)  ──►  real bimanual YAM Box + 2 Sharpa
         forked, never overwritten)                                                     hands, closed loop, 10 trials/task
```

**Simulators.** MuJoCo for the data-quality experiments. For the real tasks, "Depending on the task, the framework uses MuJoCo or Isaac Sim" (p.5).

**Hardware (p.5).**
- Bimanual **YamBox** station: the I2RT YAM Box, ~~i.e. two 6-DoF YAM arms in an enclosed station.~~ [corrected: the paper says only "a bimanual YamBox station with two Sharpa hands" (p.5–6). The vendor page (i2rt.com/products/yam-box) describes the YAM Box as a portable, "self-contained deployment platform" with "Adjustable Arm Spacing" for bimanual tasks. It does not describe an enclosure; the arms are an optional add-on, and that page states no DoF. i2RT sells 6-DoF YAM/YAM Pro/YAM Ultra/BIG YAM and a 7-DoF "YAM 7" (i2rt.com/products.json handles). Which arm variant DexAgent used is **UNVERIFIED**. The black walls in the videos are the lab backdrop, not a product enclosure.]
- Two **Sharpa** hands. The model is not stated; Sharpa Wave (22 DoF) is likely but **UNVERIFIED**.
- An ego-view **RealSense D435** and two wrist-mounted **ZED Mini** cameras.

The action space, control rate and π0.5 fine-tuning hyperparameters are **not reported**.

---

## 3. The method, stage by stage (paper §III, pp.2–5)

**Global loop (p.2–3).**
- Each stage takes the previous stage's output and "selects existing skills or develops new ones based on task properties".
- Each stage is evaluated with its verifiers. Failure feedback is used "to refine the results or revise its choice of skills".
- Only passing outputs advance.
- Budgets, quoted: *"Each stage invocation is either capped by a given budget or 100 agent turns, and a stage that runs out of its budget without passing the verifiers is recorded as a failure rather than being allowed to deliver with an incomplete result. The agent backbone can be any VLM swicthed between API."* (sic, p.3)
- **No backbone model, reasoning effort, token counts or $ cost is reported anywhere** (paper, site, videos or thread). Which model ran DexAgent's experiments is therefore **UNVERIFIED**.
- The GPT-6 Astra baseline is described as "prompted zero-shot to generate robot trajectories **without our harness**". That wording hints, but does not state, that the harness ran on a comparable frontier model.

**Skills and verifiers (§III-A, p.3).**
- *"A skill produces an asset, a pose or a motion. A verifier accepts or rejects something a skill produced based on the interaction inside simulation. Both skills and verifiers are ordinary Python functions with typed arguments. When improving a skill or a verifier, it is forked from the original function rather than overwriting on it."*
- Asset caching is optional "due to its size variance".
- Tools must not "overfit to any sample specifically".

**Stage 1, semantic understanding (§III-B).**
- The VLM starts from the language prompt and hypothesizes the objects and their properties.
- It extracts frames from the video and "infer[s] the objects' movements to confirm its property assumptions".
- It may call skills for **depth estimation, gravity direction, and human pose**. The specific models are not named.
- It decomposes the task so that each subtask "only includes one motion or as few as possible". Pick-and-place becomes 3 subtasks: pick, move, place.
- Output: `scene_semantics.json`. Everything in it is natural language; there are no numeric parameters yet. Examples are in §4.

**Stage 2, property-based reconstruction (§III-C).**
- It "first selects or writes new verifiers to each reconstructed object asset based on the object properties". The paper gives three cases:
  - **rigid**: silhouette, scale, mass, resting stability;
  - **articulated**: joint existence, axis, travel range, self-locking;
  - **deformable**: topology.
- If no skill exists for the category, the agent writes one, and it must be "a parameterized function for a group of objects with similar properties rather than a specific object."
- **Construction protocol for new skills** (verifier-first, TDD-like):
  1. Start from an editable URDF with no joints.
  2. For each property: define a verifier first, then edit the asset until it passes.
  3. Run all verifiers jointly to check for conflicts.
  4. Run a second round that adds *new* verifiers, plus a **generalization test**: "the paramalized function will reproduce 3 more variants and ensure those 3 variants also pass the verifiers". Only then is the skill stored.
- **Placement:** "optimizing the IoU score between the object mask rendered in simulation and in human video from the same camera view."
- **Camera:** "estimated mainly by depth for the distance and the overlay of sharpa hand and human hand for the camera angle". It can also be set manually.
- Robot setup and initial pose are fixed across samples.

**Stage 3, trajectory optimization (§III-D, p.4).**
- Per subtask, the agent first defines a success condition, e.g. a target object position for transport or a target joint state for articulated objects.
- It then selects a skill for the target **hand pose**. Three options: "directly retarget the demonstrated human pose, optimize the robot's finger joints to reproduce demonstrated hand–object contacts, or generate a grasp based on ... contact, force closure, and clearance."
- Poses are checked by task verifiers: for transport, "tested through lifting and shaking"; for articulated objects, "through sliding and spinning to ensure all the degrees a joint can reach are possible".
- Then: "inverse kinematics determines the corresponding arm configurations. The agent then **generates code to connect these key poses**, optionally using the demonstrated trajectory as a motion prior."
- Subtasks are chained. "Earlier motions [are] revised when their resulting states prevent later subtasks from succeeding." Whole-trajectory verifiers check motion limits and physical consistency.

**Stage 4, data generation (§III-E, Appendix A, p.5 and p.10).**
- Randomize initial object and robot states. 1,500 seeds are sampled, and each is checked for reachability by the relevant hand and for the existence of a valid execution. 500 passing seeds are kept; more are sampled if needed.
- Each episode re-passes the pose and whole-trajectory checks.
- The human and objects are removed from the source video with **VOID** (arXiv 2604.02296). The sim robot and objects are composited in and **retextured in Blender**.
- Wrist views are rendered in simulation.

---

## 4. Public artifacts in lieu of code

**`scene_semantics.json` (Appendix B, p.10–11).** ~~Excerpts, verbatim.~~ [corrected: the excerpts are verbatim, but they are elided without "..." markers. The drawer object (p.13) also has an `"appearance"` dict (housing/front_panel/tray_interior/handle colours) and a third property, `"Relative rotation and translation perpendicular to the sliding axis are constrained."`, both dropped here. Appendix B runs p.10–13: the heading is on p.10, drawing is on p.11, bottle on p.12 and drawer on p.13.]

Drawer task, object entry:
```json
{"id": "drawer_unit", "type": "articulated", "parts": ["housing","sliding_tray","front_handle"],
 "properties": ["A single-DOF prismatic joint connects the tray to the housing.",
   "Bidirectional translation along the housing's longitudinal axis: outward to open and inward to close.",
   "Observed opening travel is approximately one housing depth; a rough visual estimate, not a measured mechanical travel limit."]}
```

Drawer task, subgoal entry:
```json
{"acting_hand": "left", "action": "Pull the drawer outward.", "object_refs": ["drawer_unit"],
 "goal": "Drawer interior and the free front placement region are accessible."}
```

Bottle task, bimanual subgoal with `hand_roles`:
```json
{"acting_hand": "both", "action": "Loosen the cap with repeated turns.",
 "hand_roles": {"left": "Resists rotation of the body.", "right": "Turns the cap and readjusts its grip as needed."},
 "goal": "Cap is loosened sufficiently for removal."}
```

It also records honest uncertainty: `"observed_outcome": "Pouring posture is visible; a liquid stream or level change is not clearly resolved."` The drawing task encodes *functional* properties that sim must reproduce, for example `"Sliding eraser contact removes contacted strokes while leaving unwiped strokes visible."`

**Skill and verifier examples (Fig. 2 / `static/images/method2.jpg`).**
- Skills: `construct_bottle`, `fit_pose`, `human_pose_prior`.
- Verifiers: `stability`, `stress_test_rotation`.

The screw cap is modelled in **MJCF**, even though the text says URDF. A thread is an equality coupling between lift and yaw:
```xml
<joint name="yaw" ... frictionloss="0.02"/>
<equality joint1="lift" joint2="yaw" polycoef="0 pitch/2π ..."/>
```

Verifier pseudo-code, as printed in the figure:
```
for yaw0 in 0..360 step
    for sgn in (+1, -1):
        reset(cap_yaw=yaw0); hold(pose, ...); wrench(...)
        assert sgn·Δyaw > ... and slip < ...
```

**Library taxonomy (Fig. 8, p.15; `static/images/library_breakdown.png`).**
- **103 skills:**
  - Trajectory & Execution 24;
  - Object Reconstruction 22 (screw caps, drawers, writable surfaces);
  - Scene Runtime & Rendering 21;
  - Perception & Grounding 18;
  - Grasp & Poses 10;
  - Measurements 8.
- **188 verifiers:**
  - Task & Rollout Gates 42;
  - Grasp Stress Tests 34 (lift, shake, sustained load);
  - Reconstruction Mechanism Tests 30 (threads, latches, writable surfaces);
  - Episode Audits 30;
  - Scene Assertions 28 (contact, IoU, joint range);
  - Scene Compile Checks 24 (penetration, unsupported objects, missing mechanisms).
- About 1.8 verifiers per skill. Verification is the bulk of the library.

**Site repo** (`research/repos/dexagent1.github.io`):
- Tables live in `index.html`: HOI4D at :261–292, OakInk at :309–330, replay at :355–371, policy chart SVG at :385–488, library/cost captions at :505–529.
- The early commit `37b9fdf` (2026-09-23) captioned Fig. 4 as: library "reaches 85 skills and 168 verifiers **and stops growing**, while the per-sample cost drops by **more than an order of magnitude**". ~~This was later replaced by the 2.1 h vs 3.3/3.7 h framing and the 103/188 counts.~~ [corrected: about 86 min later the same day, commit `88afb4e` (2026-09-23 17:34 -0400, "use the paper's library caption") swapped this caption for the paper's Fig. 4 caption. That caption still says 85 skills/168 verifiers and adds the 3.7 h/3.3 h/2.1 h framing. The 103/188 counts arrived separately on 2026-09-25 in commit `96d5b5a` as a new Fig. 8 figure; they did not replace the caption. Both changes predate the 2026-09-28 arXiv posting.]

---

## 5. Results

**Table I: HOI4D reconstruction** (F-5↑, F-10↑, CD↓)

| Method | Rigid F-5 | F-10 | CD | Artic. F-5 | F-10 | CD |
|---|---|---|---|---|---|---|
| FoundationPose | 0.71 | 0.91 | 0.49 | 0.40 | 0.60 | 1.26 |
| Do as I Do | 0.72 | 0.91 | 0.49 | 0.40 | 0.61 | 1.25 |
| **DexAgent** | **0.83** | **0.96** | **0.29** | **0.47** | **0.68** | **1.08** |

**Table II: OakInk retargeting** (success = mean E_pos < 0.1 m and E_rot < 0.5 rad)

| Method | Success % | E_pos (m) | E_rot (rad) |
|---|---|---|---|
| Dex-retargeting | 28.6 | 0.08 | 0.62 |
| SPIDER mjwp / mjwp_act | 71.4 / 77.1 | 0.04 / 0.04 | 0.57 / 0.42 |
| Do-as-I-Do (Sharpa hand) | 81.0 | 0.03 | 0.15 |
| ↳ w/o transition reward | 79.0 | 0.03 | 0.14 |
| ↳ annealed sampling only | 72.0 | 0.08 | 0.32 |
| **DexAgent** | **85.7** | 0.03 | **0.12** |

**Table III: real replay.** Each converted trajectory is replayed open loop 10×, and a task counts as a ✓ if **≥1 of 10** replays succeeds.

| Method | Tasks passed |
|---|---|
| **DexAgent** | **11/11** |
| SPIDER | 6/11 |
| V2D | 5/11 |
| GPT-6 Astra | 5/11 |
| TopoRetarget | 3/11 |
| Dex-retargeting, Do-as-I-Do, EgoInfinity | 1/11 each |

Astra "joined the benchmark after the first evaluation round".

**Table IV: π0.5 policy success.** % of 10 closed-loop trials per task, randomized initial states (Fig. 3).

| Task | DexRet | DoAsIDo | SPIDER | Topo | EgoInf | V2D | GPT-6 Astra | **DexAgent** |
|---|---|---|---|---|---|---|---|---|
| Cup in bowl | 0 | 50 | 70 | 60 | 60 | 70 | 60 | **90** |
| Giftbox (long-horizon) | 0 | 50 | 40 | 10 | 20 | 40 | 40 | **70** |
| Toy in drawer (artic.) | 0 | 10 | 10 | 0 | 0 | 30 | 50 | **80** |
| Rope knot (deform.) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **50** |
| Scissors (artic.) | 0 | 0 | 30 | 10 | 0 | 0 | 0 | **80** |
| Drawing+wiping (tool) | 0 | 0 | 20 | 10 | 0 | 10 | 30 | **40** |
| Computer install (long) | 0 | 10 | 20 | 0 | 0 | 10 | 0 | **50** |
| Open bottle (artic.) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **70** |
| Biology/pipetting (tool) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **40** |
| Battery insertion | 0 | 0 | 10 | 0 | 0 | 0 | 0 | **60** |
| Multi-cup grasping | 0 | 0 | 0 | 0 | 0 | 30 | 0 | **70** |
| **Average** | 0.0 | 10.9 | 18.2 | 8.2 | 7.3 | 17.3 | 16.4 | **63.6** |

DexAgent's 63.6% corresponds to 70 of 110 trials.

**Baseline protocol (p.6).**
- Retargeting-only methods (Dex-retargeting, SPIDER, TopoRetarget) receive DexAgent-provided scene assets and object poses.
- Do-as-I-Do, EgoInfinity and V2D keep their own reconstruction and trajectory components.
- "the same data-augmentation and visual-processing protocol" is applied to all methods, which train the same π0.5.

**Self-evolution (Fig. 4, 100 EgoDex videos, randomly ordered, with recurring-object cases placed last).**

Library growth by event (skills / verifiers added):

| Event | Skills | Verifiers |
|---|---|---|
| Growing period (rigid, drawer, bottle, box) | +60 | +98 |
| First deformable (string) | +4 | +13 |
| Clean-surface sample | +2 | +11 |
| Drawing | +2 | +19 |
| Batteries into a remote | +9 | +22 |
| Phone + USB-C | +8 | +5 |
| **Total** | **85** | **168** |

Processing time:
- 4.3 h for the first video;
- about 2.2 h with skill reuse;
- **13.1 min with asset reuse** (samples ~87–100);
- 210 h for 100 verified episodes, i.e. 2.1 h average.

The ~~headline~~ [corrected: paper-body claim, §V-B p.7, "approximately 17 times faster than V2D and 15.1 times faster than GPT-6 Astra"; it does not appear on the site or in the tweet, which use 2.1 h vs 3.3 h] "17× faster than V2D, 15.1× than Astra" divides the baselines' *averages* (3.7 h, 3.3 h) by DexAgent's *best case* (13.1 min, same objects recurring). The like-for-like average gives 1.76× and 1.57×.

---

## 6. Critical reading: what the numbers do and don't show

1. **Baseline rows are copied from another paper, with mismatched protocols.**
   - In Table I, all eight baseline "Rigid" rows match Do-as-I-Do's (arXiv 2606.19333) Table 2 **overall HOI4D** column exactly (e.g. HO 0.28/0.51/3.86, Do-as-I-Do 0.72/0.91/0.49).
   - Do-as-I-Do evaluated on 12 HOI4D videos with ground-truth hands supplied, and did not split rigid from articulated. It is unclear that DexAgent used the same protocol.
   - In Table II, the three Do-as-I-Do rows (81/79/72) are Do-as-I-Do's **OakInk2** results over **1,352 bimanual trajectories**. DexAgent cites OakInk v1 [31].
   - Meanwhile 28.6/71.4/77.1/85.7 are all multiples of 1/35 (10/25/27/30 of 35). ~~SPIDER reports ADD-AUC, not success.~~ [corrected: SPIDER's headline tables report ADD-AUC. Its Appendix A does define a thresholded task success, "Erot < 0.5 rad, and ... Epos < 0.1 m", for its own ablations, and says it "additionally report[s] ADD-AUC10". Its §4.1 explains it uses ADD-AUC "rather than a thresholded success rate because success is defined differently across baseline papers". No SPIDER success-rate table matching 71.4/77.1 exists (arXiv 2511.09484v3, 26 Sep 2026). Its Appendix B.1 OakInk ablation uses "7 distinct bimanual two-object manipulation tasks" with "Mean±std over 5 seeds" (Table 9). That gives exactly 35 runs, which supports the 1/35 hypothesis below.] These are consistent with DexAgent running its own small set, e.g. SPIDER's 7-task OakInk subset (×5); that is inference, **UNVERIFIED**.
   - So "5.8% relative improvement over the strongest baseline" plausibly compares numbers from different test sets.
   - The "two indented ablation rows" are **Do-as-I-Do's own ablations**, not DexAgent's.
2. **There are no ablations of DexAgent itself.** There is no "w/o verifiers", "w/o library", "w/o code-generated motion" or backbone swap. alphaXiv's summary makes the same point: the results do "not show that verification or library reuse alone caused the gain".
3. **The replay metric is lenient (best-of-10, open loop) and inconsistent with Table IV.** Several baselines score 0/10 on replay but above 0 on policy:
   - Do-as-I-Do: giftbox 50%, drawer 10%, computer 10%;
   - SPIDER: drawing 20%;
   - TopoRetarget: scissors 10%, drawing 10%;
   - EgoInfinity: giftbox 20%;
   - V2D: drawing 10%, computer 10%.

   This is ~~possible only if~~ [corrected: consistent with, but not proof that — a closed-loop π0.5 policy trained on 500 augmented episodes and evaluated from randomized initial states can also succeed where open-loop replay of one trajectory fails 10/10, so augmentation "repair" is one hypothesis, UNVERIFIED] the shared Stage-4 augmentation (which "adapts the trajectory to each configuration") repairs baseline data. In that case, part of DexAgent's machinery runs inside the baselines, and the attribution gets murky.
4. **The statistics are small.** n=10 trials per task (±~30 pp per-task 95% CI), one demo per task, one seed. The GPT-6 Astra baseline prompt is undisclosed, and Astra was evaluated in a later round.
5. **The library numbers drift.** The Fig. 4 caption says "over 100 samples… 85 skills/168 verifiers"; the text says "by sample 78"; the appendix, site and tweet say 103/188, "accumulated from the EgoDex experiment and our task samples". The early site claim "stops growing" was quietly dropped.
6. **Cost is reported only in wall-clock hours.** There are no tokens, $, GPU-hours or model name. "GPT 6: Astra takes 3.3 hours on average **at a flat rate**" is unexplained.
7. **Related work omits the agent/tool-making literature.** None of Voyager-style skill libraries, Eureka/RoboGen/GenSim-style LLM sim and reward generation, or Articulate-Anything/URDFormer-style LLM articulation is cited. The only agentic comparator is V2D (NVIDIA, `github.com/nvidia-isaac/video_to_data`), whose video ingestion is a LangGraph + vLLM VLM pipeline, also on Sharpa hands.
8. **Stated limitations (p.7):** *"semantic and geometric errors, incomplete verification, and simulation-to-reality gaps. Undetected errors propagate through the library, and force-sensitive interactions can fail despite passing simulation checks."* Future work: stronger verification, and RL as a fallback.

---

## 7. Relationship to the other user sources

- **GPT-6 Astra cluster** (Robocurve report, piper-astra-jev, innate-os PR #817). DexAgent supplies the only datapoint so far where Astra, **unharnessed and zero-shot**, generates *dexterous bimanual* trajectories:
  - 5/11 replay and 16.4% policy, roughly on par with specialized retargeting pipelines (SPIDER 18.2%, V2D 17.3%);
  - the DexAgent harness reaches 3.9× that.

  innate-os PR #817 ("Imitate a recorded demonstration with GPT-6 Astra") tackles the *same input*, one recorded demo, the opposite way:
  - **online**: Astra sees the recorded frames plus live cameras and emits one validated action per observation;
  - versus DexAgent's **offline** conversion of the demo into sim data for a trained VLA.
- **EmbodiedSWE** (teammate note). Both use agent-written code as a *teacher*, not as the deployed policy. EmbodiedSWE has the agent write controllers in privileged sim; DexAgent has it write sim assets, verifiers and glue motion code, then distils into π0.5. General Robotics' "Auto-Engineering" is likely in the same "agent engineers the robot offline" family (not read here; **UNVERIFIED**).
- **Live-control sources** (GPT-as-Policy, GPT-Policy, metal-arm-harness, quackd, llm-robotics-playground). DexAgent is orthogonal. It does not address latency, closed-loop visual servoing, or safety at runtime.

---

## 8. Assessment

**Strengths**
- A clean **agent architecture for physical problems**:
  - a staged pipeline with typed artifacts;
  - verifier gates between stages;
  - a hard budget (100 turns) with *fail-closed* semantics;
  - a persistent, versioned (fork, not overwrite) library of skills *and* verifiers.

  This is the most transferable idea.
- **Verifier-first asset construction**, plus a **generalization gate** (the function must produce 3 more variants that pass) before a tool enters the library. This is a cheap, concrete defence against overfit one-off code polluting the library.
- **Physics stress tests as semantic checks.** Grasps are lifted and shaken. Articulations are swept through their range with ± wrenches. Threads are modelled as joint couplings. These turn vague VLM beliefs ("cap unscrews") into falsifiable sim assertions.
- **Division of labour.**
  - The LLM does semantics, decomposition, success conditions, asset and verifier code, and glue motion code.
  - Numerical optimizers do retargeting and IK.
  - A learned VLA does closed-loop control.

  Each component works where it is strong.
- **Amortization is real.** Asset reuse cuts processing from hours to about 13 min, and new object classes add a bounded number of tools (+2 to +9 skills per new class after the initial period).
- An impressive task span for 1 demo and 0 robot teleop: knots, threads, scissors, pipetting, multi-step installs.

**Weaknesses**
- **Irreproducible as published:** no code, no backbone, no prompts, no cost, no policy hyperparameters, no action-space specification.
- **Evaluation hygiene:** copied baseline rows from different test sets, best-of-10 replay, no self-ablations, 10 trials per task, a later-round Astra baseline with an undisclosed prompt.
- **Throughput:** 2.1 h average per demo (4.3 h cold). This is a data factory, not an interactive system.
- Library growth is uncurated. The authors admit that undetected errors propagate through it, and retrieval and selection at ~~300+~~ [corrected: 291 (103 skills + 188 verifiers, Fig. 8)] tools is unaddressed.
- Sim verifiers cannot certify force-sensitive contact, which is the core sim-to-real failure mode for dexterous hands.

**What is novel vs repackaged.**
- Repackaged:
  - the Human2Sim2Robot pipeline (Do-as-I-Do, SPIDER, V2D, Lum et al.);
  - VOID inpainting and Blender retexturing;
  - π0.5 fine-tuning;
  - LLM skill libraries (Voyager lineage).
- Genuinely new: wrapping the *whole* video→sim→trajectory→data pipeline in a coding agent that **writes per-property reconstruction skills and simulation verifiers**, gates stages on them, and persists both, with evidence that this handles articulated and deformable objects that fixed pipelines miss (bottle, rope, biology: 0% for every baseline).

**Maturity.** Research preprint, one week old, code "coming soon". Treat the numbers as directional.

**What Ilia's Opus-backbone harness should borrow**
1. **A skill + verifier library as first-class, typed Python.** Fork on edit, and use a generalization gate (N parameter variants must pass) before admission. For an online harness, verifiers become **pre-execution checks**: simulate an LLM-proposed grasp or waypoint in MuJoCo (lift and shake, collision, reachability) before sending it to hardware. Also use **post-execution task gates** in place of "LLM says done".
2. **A structured plan contract** like `scene_semantics.json`:
   - objects with type, parts and functional properties;
   - subgoals with `acting_hand`, `hand_roles`, `object_refs`, and an *observable* `goal` string;
   - explicit `observed_outcome` uncertainty.

   This is an excellent schema for Opus's planning output and for per-subgoal success checking.
3. **Fail-closed stage budgets** (turn caps; a failed stage is a recorded failure, never a partial deliverable). Iterate within a stage until its gates pass.
4. **Key-pose plus code-generated interpolation.** The LLM picks verified key poses; IK and generated code connect them. This is a good action interface for a frontier LLM with a light learned head: the head handles fine closed-loop contact; the LLM handles poses, sequencing and recovery.
5. **The "LLM as offline teacher → VLA/action-head student" option.** If Ilia's learned head needs data, DexAgent shows one demo plus an agentic sim pipeline can produce 500 usable episodes per task.

**What to avoid**
- Hour-scale agent loops anywhere on the control path. Cache assets and skills aggressively, since reuse is where the 20× comes from.
- Reporting without model/version, tokens, $ and per-trial logs. Ilia's harness should log these from day one, as Robocurve's wire capture does.
- An unbounded, uncurated library. Add provenance, test coverage, deprecation, and retrieval evaluation, because errors otherwise compound.
- Lenient metrics (best-of-N replay) and borrowed baseline numbers in any internal benchmarking.

---

## Sources

- Project page: https://dexagent1.github.io/ (repo https://github.com/dexagent1/dexagent1.github.io, history read)
- Anonymized twin page: https://dexagent123.github.io/ (repo https://github.com/DexAgent123/DexAgent123.github.io)
- arXiv abstract: https://arxiv.org/abs/2609.35318 ; HTML: https://arxiv.org/html/2609.35318v1
- Paper PDF (site copy): https://dexagent1.github.io/static/paper/DexAgent_paper.pdf
- Videos: https://dexagent1.github.io/static/videos/dexagent_overview.mp4 , https://dexagent1.github.io/static/videos/dexagent_twitter.mp4
- TL;DR thread: https://x.com/jwang633/status/2105012942601367940 (read via https://api.fxtwitter.com/jwang633/status/2105012942601367940)
- alphaXiv: https://www.alphaxiv.org/abs/2609.35318
- Do as I Do (baseline tables cross-checked): https://arxiv.org/abs/2606.19333
- SPIDER (OakInk protocol cross-checked): https://arxiv.org/abs/2511.09484
- TopoRetarget: https://arxiv.org/abs/2606.16272 ; EgoInfinity: https://arxiv.org/abs/2606.17385 ; VOID: https://arxiv.org/abs/2604.02296
- NVIDIA V2D: https://github.com/nvidia-isaac/video_to_data
- I2RT YAM Box: https://i2rt.com/products/yam-box ; Sharpa Wave: https://www.sharpa.com/pages/wave
- innate-os PR #817 (cross-reference): https://github.com/innate-inc/innate-os/pull/817
- Teammate notes cross-referenced: `research/sources/robocurve-gpt6-astra.md`, `research/sources/embodiedswe.md`

---

## Verification (fact-check pass)

Fact-checked on 2026-10-01 against these primary sources:
- **Paper.** The site PDF (`repos/dexagent1.github.io/static/paper/DexAgent_paper.pdf`) and arXiv `2609.35318v1` were byte-identical (35,584,023 B, 15 pp). Text was extracted per page with `pdftotext -layout`.
- **Site.** `repos/dexagent1.github.io` @ `d6ec746`, with full `git log`, and `repos/DexAgent123.github.io` @ `e07adf7`.
- **Figures.** Inspected visually: `static/images/{method2.jpg,library_breakdown.png,lifelong_plot.png,success_rate_bar.png}`.
- **Videos.** Frames of both videos sampled every 3–4 s.
- **Metadata.** arXiv abs page; fxtwitter API for the tweet; alphaXiv HTML; `gh api` for the org/user/repo metadata.
- **Cross-checks.** Do-as-I-Do PDF (arXiv 2606.19333v1); SPIDER PDF (arXiv 2511.09484v3, 26 Sep 2026); NVIDIA V2D repo @ `33013fcc`. I shallow-cloned V2D, then deleted the clone because the shared disk hit 100%.
- **Vendor pages.** i2rt.com YAM Box and products.json; sharpa.com Wave.

**Overall:** the note is reliable. Every number in Tables I–IV, all Fig. 4/Fig. 8 counts, the quotes and the arithmetic check out. The errors are peripheral: hardware description, one claim about SPIDER's metrics, framing.

### Confirmed (seen in a primary source)

**Bibliographic and metadata**
- **arXiv record.** Title; 2609.35318v1; cs.RO; "Mon, 28 Sep 2026 14:49:06 UTC"; "15 pages, 8 figures, 4 tables" (arXiv abs page). Authors, affiliations, † equal advising, and corresponding e-mail (paper p.1). The arXiv abstract links the *anonymized* site `dexagent123.github.io`.
- **X thread.** Posted Tue Sep 29 19:12:49 2026 UTC by @jwang633. Counts: views 9,607, likes 65, RTs 10, replies 3, bookmarks 47 (plus 1 quote). Bio: "Undergrad @Harvard & @Cornell / Researcher @StanfordAILab @StanfordSVL" (fxtwitter). The tweet says "2.1 hours average inference time with the tool library, compared with 3.3 hours for the baseline" and "103 skills + 188 verifiers".
- **alphaXiv.** "#16 on Trending", 513 views. Exact quote: "It does not show that verification or library reuse alone caused the gain."
- **GitHub.** Org `dexagent1` was created 2026-09-22T22:11:08Z with public_repos=1 (`dexagent1.github.io`, 0 stars). The site says "Code *(Coming Soon!)*" (`index.html:129`).
- **Acknowledgement.** "We thank Sharpa for equipment support" (p.7).

**Method text (§III, pp.2–4)**
- The budget quote is verbatim, including "swicthed" (p.3).
- Skill/verifier definitions, fork-not-overwrite, and optional asset caching (§III-A, p.3).
- Stage 1: the 3-subtask pick-and-place example; JSON content in natural language only (p.3).
- Stage 2: the rigid/articulated/deformable verifier lists, the verifier-first protocol, the "3 more variants" generalization test, IoU placement, and depth plus Sharpa/human-hand overlay for the camera (pp.3–4).
- Stage 3: the three hand-pose options, lift+shake and slide+spin tests, IK followed by "generates code to connect these key poses", and whole-trajectory verifiers (p.4).

**Data generation and setup (§III-E, App. A, §V-A)**
- VOID [26]; Blender [27]; ego plus left/right-wrist views; 1,500 seeds → 500, resampled if fewer pass (§III-E p.4–5; App. A p.10).
- MuJoCo for §IV; "MuJoCo [28] or Isaac Sim [32]" for real tasks (p.5).
- D435 at 640x480 and 30 fps (p.5); two ZED Mini wrist cameras (p.6).
- 500 episodes, π0.5 fine-tune, 10 trials per task (p.6).
- The action space, control rate, π0.5 hyperparameters, backbone model, prompts, tokens and $ cost are indeed absent from the paper, site and tweet text.

**Results tables (pp.4–6)**
- Table I: all 54 cells match.
- Table II: all cells match.
- Table III: every ✓/✗ cell matches.
- Table IV: all 88 cells and the 8 averages match. I recomputed them: SPIDER 200/11 = 18.2; V2D 190/11 = 17.3; Astra 180/11 = 16.4; DexAgent 700/11 = 63.6, i.e. 70/110 trials.
- 63.6/18.2 = 3.49× and 63.6/16.4 = 3.88×.
- The replay-vs-policy inconsistencies listed in §6.3 are all real.

**Fig. 4 and processing cost**
- Event deltas sum to 85 skills and 168 verifiers.
- Times: 4.3 h first video, 2.2 h with skill reuse, 13.1 min with asset reuse, 210 h for 100 episodes (p.7).
- Caption figures: 36.4% = (3.3−2.1)/3.3 and 43.2% = (3.7−2.1)/3.7.
- The asset-reuse drop happens at sample ~87 (`lifelong_plot.png`).
- 17×/15.1× use 13.1 min: 222/13.1 = 16.9 and 198/13.1 = 15.1. The like-for-like ratios 1.76× and 1.57× are correct.

**Fig. 8 taxonomy**
- Skills: Trajectory & Execution 24, Object Reconstruction 22, Scene runtime & Rendering 21, Perception & Grounding 18, Grasp & Poses 10, Measurements 8.
- Verifiers: Task & Rollout Gates 42, Grasp Stress Tests 34, Reconstruction Mechanism Tests 30, Episode Audits 30, Scene Assertions 28, Scene Compile Checks 24.
- Source: `library_breakdown.png` and p.15.

**Fig. 2 artifacts**
- Skills are drawn blue: `construct_bottle`, `fit_pose`, `human_pose_prior`. Verifiers are drawn green: `stability`, `stress_test_rotation`.
- The XML snippet and the verifier pseudo-code match `method2.jpg`.
- The MJCF inference is sound: `frictionloss` and `polycoef` are MuJoCo attributes. The snippet is abbreviated pseudo-MJCF, though; real MJCF nests `<joint joint1= joint2= polycoef=>` inside `<equality>`.

**Site repo line references**
- Tables: HOI4D `index.html:261–292`, OakInk `:309–330`, replay `:355–371`.
- Policy chart: the figure starts at ~`:384`; the `<svg>` spans `:395–487`.
- Captions: `:505–531`.
- Early commit `37b9fdf` (2026-09-23 16:08 -0400) caption: "reaches 85 skills and 168 verifiers and stops growing, while the per-sample cost drops by more than an order of magnitude. A fixed pipeline stays flat."

**Baseline provenance (§6.1)**
- All 8 baseline "Rigid" rows of Table I equal Do-as-I-Do Table 2's single HOI4D column, e.g. IHOI 2.7 → 2.70.
- Do-as-I-Do used "12 annotated videos" and "supplying ground-truth hands" (Do-as-I-Do §4.1).
- DexAgent's Table II rows 81.0/0.03/0.15, 79.0/0.03/0.14 and 72.0/0.08/0.32 equal Do-as-I-Do's OakInk2 "+ Transition Reward", "+ Perturbation" and "Annealed Sampling" rows. OakInk2 has "1,352 clean bimanual human-object task trajectories" (Do-as-I-Do Table 3, §4.1).
- DexAgent's [31] is OakInk (Yang et al., CVPR 2022).
- 10/25/27/30 of 35 are arithmetically correct.

**Other confirmed items**
- No Voyager, Eureka, RoboGen, GenSim, Articulate-Anything or URDFormer in DexAgent's 35 references.
- Limitations quote (p.7) is verbatim.
- V2D's ingestion is LangGraph plus vLLM: README "a LangGraph-driven agentic workflow"; `video_ingestion_agent/pyproject.toml:37` `"langgraph>=0.2.0"` and `:99` `"vllm>=0.24,<0.25"`. V2D targets Sharpa ("retargeted onto the target robot embodiment (Sharpa)") and uses RSL-RL PPO in Isaac Lab.
- innate-os PR #817 title "Imitate a recorded demonstration with GPT-6 Astra" (`gh pr view`). Its default `chunk_size: int = 1` (`workspace/innate_skills/imitate_demonstration/imitate_demonstration.py:435` in `repos/innate-os-pr817`) supports "one validated action per observation".
- Sharpa Wave = "22 active DoF" (sharpa.com/pages/wave; also Do-as-I-Do §4.1: "the 22-DoF Sharpa Wave hand").

### Corrections (7; also fixed inline)

1. **YAM Box.** "two 6-DoF YAM arms in an enclosed station" → the vendor calls it a portable, self-contained mounting and deployment platform with adjustable arm spacing. No enclosure is described, the arms are optional, and the page gives no DoF. The arm variant (6-DoF YAM/Pro/Ultra/BIG vs 7-DoF YAM 7) is UNVERIFIED.
2. **SPIDER metrics.** "SPIDER reports ADD-AUC, not success" → its headline tables use ADD-AUC. Its Appendix A defines the same Erot < 0.5 rad / Epos < 0.1 m success for its ablations, but it publishes no success-rate table that DexAgent could have copied. Its OakInk ablation is 7 tasks × 5 seeds = 35 runs, which strengthens the note's 1/35 inference.
3. **Replay vs policy.** "possible only if the shared Stage-4 augmentation repairs baseline data" → overstated. Closed-loop policies trained on 500 randomized episodes can succeed where a single open-loop replay fails, so augmentation repair is one hypothesis.
4. **Tool count.** "300+ tools" → 291 (103 + 188).
5. **JSON excerpts.** "Excerpts, verbatim" → verbatim but silently elided (the drawer's `appearance` dict and its third property are dropped). Appendix B spans p.10–13, not p.10–11.
6. **Site history.** "later replaced by … 2.1 h … and the 103/188 counts" → the "stops growing" caption was replaced 86 min later the same day by `88afb4e` (paper caption, still 85/168). 103/188 was added 2026-09-25 by `96d5b5a` as a separate Fig. 8. All of this predates arXiv.
7. **"Headline" speedup.** The "17×/15.1×" is a paper-body claim (§V-B, p.7). The site and tweet use 2.1 h vs 3.3 h.

### Unverifiable / inference-only (left as is, flagged)

- **Hand model.** "Sharpa Wave … likely" for DexAgent's physical hands is still UNVERIFIED. The only DexAgent mention is the Table II text: "Sharpa Wave denotes the robot hand used by Do-as-I-Do, which was a built-in asset option" (p.5). That supports the guess but refers to the baseline and sim asset.
- **Training seeds.** "one seed" (§6.4): the number of π0.5 training seeds is not reported either way.
- **Confidence interval.** "±~30 pp per-task 95% CI" is a reasonable approximation. At p = 0.5, n = 10: Wald ±31 pp, Wilson [24%, 76%].
- **Test-set inference.** The guess that DexAgent ran its own 35-clip set (SPIDER's subset) is plausible but unconfirmed. SPIDER's "Oakink (Zhan et al., 2024)" actually cites OakInk2, so DexAgent's OakInk-v1 citation [31] may itself be a citation slip.
- **"Only datapoint so far"** for unharnessed Astra on dexterous bimanual tasks (§7): a statement about the corpus, not checkable against this source.
- **alphaXiv comments.** "no public comments" — comments load dynamically and were not visible in the fetched HTML. alphaXiv, X and GitHub counts are time-varying snapshots.
- **Backbone in videos.** Frame sampling of both videos shows no model name, prompt, or cost. The 94 s teaser has an audio track (mean −10.3 dB, likely music) that I could not transcribe: faster-whisper install failed because the disk was full.
- **General Robotics "Auto-Engineering"** relation: not checked (the note already flags it).

### Missed details (added)

1. **Example task prompts.** These are the actual LLM inputs.
   - Fig. 2: *"Remove the bottle cap, then pour the water from the bottle into the mug."*
   - Teaser video: *"Install the hard disk into the computer"* and *"Insert the Battery."*
   - `task_summary` fields:
     - *"Write 1, 2, 3, and 4 with the right hand; erase only 4 with the left hand; return both tools."*
     - *"Remove the screw cap, direct a pouring motion into the mug, and return the uncapped bottle upright."*
   - The teaser's Stage-1 slide shows a per-hand state decomposition for the hard-disk task: "State 1 • Right hand: grap the disk • Left hand: stationary / State 2 • Right hand: Insert the disk • Left hand: stationary / State 3 • Right hand: Hold still • Left hand: finger pushes the disk". The teaser also states "All From 1 Human Demonstration, 0 Robot Data".
2. **More schema fields.** `scene_semantics.json` also contains `task_summary`, per-object `name`, and per-part `appearance` colours. It also carries **negative/role constraints**, e.g. mug *"Remains on the tabletop; neither hand grasps or stabilizes it."*; bottle part `attached_tube` with *"A long tube is carried with the cap; it must clear the mouth before the cap is moved aside."*; and *"Intended task outcome: …"* vs `observed_outcome`. These are directly reusable for an Opus plan schema.
3. **Viewpoint augmentation.** Stage 4 varies **viewpoints** as well as object states: Fig. 2 "View Augmentation"; caption "Stage 4 varies viewpoints and object states"; App. A "varied object placements and viewpoints". The note mentions only object and robot states.
4. **Astra missing from the site chart.** The project page's interactive policy chart **omits GPT-6 Astra entirely**: the legend (`index.html:386–394`) and the SVG `data-name` values cover only Dex-retargeting, Do as I Do, Spider, TopoRetarget, Egoinfinity, V2D and Ours. Astra's 16.4% appears only in paper Table IV, and on the site only in the replay table. The original chart PNG (`success_rate_bar.png`, commit `37b9fdf`) also lacks Astra and labels the biology task "Tube". Astra rows first appear on the site in `a7fb4fd`, together with the "joined after the first evaluation round" caveat.
5. **SPIDER counted twice.** Do-as-I-Do states "SPIDER serves as the Annealed Sampling baseline" (§4.1). DexAgent's "annealed sampling only" row (72.0) is therefore effectively *SPIDER on OakInk2 as run by Do-as-I-Do*. SPIDER thus appears twice in Table II, with numbers from two different test sets (71.4/77.1 vs 72.0).
6. **Articulated columns untraceable.** Table I's "Articulated" baseline numbers have no counterpart in Do-as-I-Do, which reports a single HOI4D column over 12 videos. Their provenance is undisclosed, and any rigid/articulated split of 12 videos leaves very few per split.
7. **Flat baseline costs.** In Fig. 4 the V2D and Astra costs are drawn as **flat constant lines** ("at a flat rate"; "their processing costs remain approximately 3.7 h and 3.3 h"). They are not per-sample measurements. A *zero-shot* Astra run taking 3.3 h per sample is unexplained; it implies a long agentic session, but the prompt and protocol are undisclosed. DexAgent's own curve is **non-monotonic**: it rises back to ~2.7 h around samples 20–35 and to ~2.3 h near sample 57 (`lifelong_plot.png`). The "base skill set with no task-specific tools" has unreported size; the plot counts only added tools.
8. **No failure accounting.** "The full run produces 100 verified episodes in 210 h" implies 100/100 EgoDex videos succeeded, or that failures went unreported. No count of fail-closed stage failures, turns used, or budget hits is given anywhere.
9. **Library scope.** The library covers Stages 1–3 only: "These skills and verifiers span from Stage 1 to Stage 3" (App. D). Stage 4 (augmentation and rendering) is fixed tooling. Fig. 1 and the videos use a *different* taxonomy (Scene Understanding / Object Reconstruction / Motion Analysis / Manipulation Skills; Physical & Geometric / Task Verifiers) from Fig. 8.
10. **Task taxonomy as the paper gives it (p.6).**
    - Rigid: cup placement, multi-cup grasping, battery insertion.
    - Articulated: bottle opening, drawer placement, scissor cutting.
    - Deformable: rope knotting.
    - Tool use: drawing and wiping, pipetting.
    - Long-horizon: gift-box packing, computer installation.

    Fig. 3 shows the randomized initial object-state distributions used for policy evaluation.
11. **Videos at 6×.** All robot rollout reels are shown at **6× speed** ("6x" overlay; commit `e731ce8` "Label both reels 6x"; clips retimed to 10 s in `52ae426`), so real execution is slow. No failure rollouts or per-trial logs are published.
12. **Double-blind twin.** The anonymized twin predates arXiv: user `DexAgent123` was created 2026-09-15T00:14Z, and its first commit (2026-09-14 22:03 -0700) reads "Project page: title, anonymous authors". Commit `79be54a` reads "Hide the arXiv, Paper and Code buttons for the anonymous version". This indicates a double-blind conference submission in mid-September; the venue is UNVERIFIED.
13. **Sharpa ecosystem.** Do-as-I-Do runs Sharpa Wave hands on UR3e arms "both commanded at 50 Hz" (§4.1), and V2D targets Sharpa. DexAgent's main comparators share the Sharpa ecosystem (Sharpa supplied DexAgent's equipment), which partly explains the baseline choice.
14. **Writing quality.** The paper has several typos: "swicthed", "paramalized", "ineraction", "grap the disk" in the video, and "stablility" in the PDF figure text. This fits a fast-turnaround preprint; it is not evidence of error in the numbers.
