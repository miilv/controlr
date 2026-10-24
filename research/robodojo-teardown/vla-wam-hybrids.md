# VLA/WAM checkpoints on RoboDojo, and how the hybrid agents call them

Question: which learned motor policies (VLA/WAM) we could actually call as a tool under our LLM loop on
RoboDojo-Sim, and what the existing hybrids teach about the interface between the LLM and the policy.

Sources, audited 2026-10-24 (nothing run; no weights, video or datasets downloaded):
- Code: XPolicyLab `main` @ `e08ba78` (sparse: `policy/*/README|deploy.*|model.py`, all of `policy/physicalRSI/`); R2SI @ `3edfe6a`
  (full source); PR heads #175 `fff098f` (SenseNova), #201 `145a526` (HarnessPolicy), #195 `c7afd5a` (RunWu, body only).
- Metadata only: HF model APIs (`?blobs=true` sizes, cards), ModelScope API (Rex-M1), HF tree of dataset `RoboDojo-Benchmark/RoboDojo`.
- **Per-task leaderboard numbers**: the site bundle `robodojo-benchmark.com/assets/index-DA5jHulR.js` embeds a `JSON.parse(...)` with
  per-task score/SR for all 62 entries (stamped `2026-10-24T00:00:00Z`). Per-dimension numbers: `official_leaderboard` in
  `research/sources/deep-research-robodojo-2026-10-24.json` (it has no per-task data).

## TL;DR

- **Downloadable RoboDojo fine-tunes that score well** (board Average score): VPP2 39.3 (~54 GB with encoders, 96 GB GPU);
  Awomo-0.5 35.3 (16 GB, CC-BY-NC); Simate-beta 34.0 (13.5 GB fp32, Apache); Rex-M1 33.8 (10.2 GB, Apache); InternW0-Δ 30.8
  (12.4 GB, Apache); Liber-0 Preview/Lite 30.7/29.2 (22–35 GB, no license); official π0.5 11.4; WorldScape-R²SI Flash/Memory WAMs
  (2 × 25 GB, Apache, no standalone score). DreamZero, LingBot-VLA and Cosmos have no usable RoboDojo checkpoint (§1.3).
- **All are multi-task** (one checkpoint), conditioned on the *task instruction*, trained only on the 34 tasks with demos, from the
  start pose. None was trained on sub-goal language.
- **Hybrids, most to least LLM involvement:** SenseNova (Codex/Astra reviews every π0.5 50-step proposal: accept/modify/reject a
  2–50 prefix) → WorldScape-R²SI (Astra picks native tools or a WAM per stage, reviews after *every acknowledged WAM action*) →
  HarnessPolicy (DeepSeek-Flash hands *whole tasks* to Simate from the start pose; code only for identify-the-object tasks) →
  PhysicalRSI (one LLM call at episode start picks π0.5 or one of 22 task-named code programs, frozen). RunWu has **no** VLA.
- **Most useful measured lesson** (HarnessPolicy design doc): slicing the VLA run into 10 s pieces with an LLM look between them
  scored **31.9 vs 42.4** for Simate alone. SenseNova's closed-loop review bought +2.2 pp SR at a huge token cost; R²SI is the best
  hybrid (46.6) but has Precision 23.5. **Hand over whole trained tasks; do not micro-review chunks.**
- **Per-task router bound** (best of agent/VLA per task, hindsight on board numbers): PyRUA + VPP2 → **67.1 score / 61.4% SR**
  (vs 55.3/50.4); Astra-direct + VPP2 → 50.7/42.4 (vs 29.0/22.5); Astra + Simate → 47.2/39.6. Gains: Gen-std, Long-H, Precision,
  part of Memory; Open gains nothing.
- **Recommendation:** pilot **Simate-beta** as a whole-task tool (fits a 32 GB card, Apache, strongest VLA on Gen-rand and
  Long-H); **Rex-M1** as a cheap second; VPP2 only on a rented 96 GB card if the pilot pays (§5).

## 1. Catalogue of released RoboDojo checkpoints

### 1.1 The calling contract (same for all of them)

Every adapter is `XPolicyLab/policy/<P>/`. The policy runs in its own env as a websocket+msgpack server
(`setup_eval_policy_server.sh … <port> 0.0.0.0`). Split-machine serving is supported.

- **Client** `XPolicyLab.client_server.ws.model_client.WsModelClient.call(func_name, obs)`; **model API** `reset()`,
  `update_obs(obs)`, `get_action() -> [action_dict, …]` (one chunk); batch variants; some add `reset_evaluation` / `reset_envs`.
- **Reference loop** (`demo_policy/deploy.py`): `reset → loop { update_obs(get_obs()); chunk=get_action(); for a in chunk:
  take_action(a); if end or last: break; update_obs(get_obs()) }`. **`update_obs` runs after every env step inside a chunk**;
  history models need it (Simate 25 frames × 20 steps, VPP2 8 × stride 25, Rex 20 × 25, Memory WAM 24 chunks).
- `obs`: `vision.{cam_head,cam_left_wrist,cam_right_wrist}.color` (640×480 RGB) + intrinsics/extrinsics, `state.*` (joints, EE
  poses), `instruction`. Transport: `request_timeout_s` 120 s; cold start retries 15 min.
- Actions: `joint` = 14-D `6+1+6+1`; `ee` = absolute pose + gripper per arm (VPP2 EE16 `[x,y,z,qw,qx,qy,qz,g]×2`). In EE mode
  RoboDojo's IK silently drops an arm on IK failure (docs/ROBODOJO.md).

For controlr: the shim's `execute` path gains a second client to a VLA server and runs this loop for N steps or until a stop
condition. **The actions bypass our cuRobo planning**, so `CartesianEnvelope` must check joint/EE targets per step (needs FK; rule 7).
**The observation crosses the wire every step** (~2.7 MB raw if the policy is on another host; compute3 is on Wi-Fi).

### 1.2 Checkpoints with RoboDojo weights (sizes from the HF/ModelScope APIs)

The size column covers the RoboDojo eval files only. VRAM figures are **my estimates**, from weight dtype × parameters plus encoders,
unless a doc states them. No adapter publishes latency. All of these models take the task instruction. "MT" means one checkpoint for
all tasks.

| Model (board type) | Weights (eval) | Size | License (weights / code) | Arch / action | Chunk / replan | VRAM (est.) |
|---|---|---|---|---|---|---|
| **VPP2** (WAM) | HF `Haodong082399/VPP2` → `checkpoints/joint2b_s100000/{video.pt 32.8 GB, action.pt 4.1 GB}` + Wan2.1 encoders (UMT5-xxl 11.4, CLIP 4.8, VAE 0.5) | ~54 GB (repo 252 GB; skip `checkpoints_video/`) | weights: no license field (Wan2.1 Apache); code MIT | Wan2.1-I2V-14B video + 2B action expert; **ee** absolute EE16 | 32 / 24, 10 Euler steps, bf16 | ≥ 50 GB; **reference GPU: one 96 GiB RTX PRO 6000; "smaller devices are unverified"** |
| **Simate-beta** (VLA) | HF `SIPAILab/sipai-robodojo-eval-ae` (`model.safetensors`, FP32); also the official bank `RoboDojo-Benchmark/RoboDojo:ckpt/RoboDojo/Simate-beta/…joint-0` | 13.5 GB | Apache-2.0 / MIT; needs the PaliGemma tokenizer (Gemma terms; init bundle is `license: gemma`) | PaliGemma/SigLIP-type ≈3.4B + AE history memory (25 frames, interval 20); **joint, relative arms, absolute grippers** | `exec_chunk_size` 10 (1–50), 10 flow steps, fp32 | ~16–18 GB fp32 (HarnessPolicy: Simate + SAM3 + kinematics ≈ 23 GB) |
| **Rex-M1 Preview** (VLA) | ModelScope `gzx2019/Rex-M1-preview` (`last.ckpt/.../mp_rank_00_model_states.pt`) | 10.2 GB | Apache-2.0 (base Xiaomi-Robotics-1-5B, Apache) | Xiaomi-Robotics-1 5B on Qwen3-VL-4B, 20 history frames; **ee only** | 30-step chunk, executes 15 | ~12–14 GB bf16 |
| **Awomo-0.5** (WAM) | HF `Auwomo/Awomo-0.5I-Robodojo` (`model.pt`: DiT + state encoder + Qwen LoRA) + Qwen3-VL-4B + FLUX.2 VAE | 16 GB + ~9 GB | **CC-BY-NC-4.0** / Apache | DiT WAM + Qwen3-VL-4B; joint absolute; input **BGR** | 16 / 8, 10 steps | ~26 GB |
| **InternW0-Δ** (WAM) | HF `InternRobotics/InternW0-Delta-RoboDojo` (`robodojo.pt`) + Wan2.2-TI2V-5B VAE/UMT5 + RynnBrain1.1-2B (ModelScope) | 12.4 GB + ~15 GB | Apache-2.0 / MIT | Wan2.2-5B video DiT + RynnBrain 2B; joint 14-D | 32 / 10, 10 steps | ~25–30 GB |
| **Liber-0 Preview** (VLM-WAM) | HF `LiberAI/Liber0-Preview-Robodojo` (`model.pt` 17.6 + `context/` 4.3) + HiDream-O1-Image base | 22 GB + base (Lite: "≈53 GB total download") | **none stated** (base MIT) | HiDream-based; joint; 288×384 four-grid | 32 / 16 | ≥ 30 GB |
| **Liber-0 Lite** (WAM) | HF `LiberAI/Liber0-Lite-Robodojo` | 35 GB | none stated | same, no context model | 32 / 16 | ≥ 25 GB |
| **π0.5 official** (VLA) | dataset `RoboDojo-Benchmark/RoboDojo:ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-{0,1,2}/59999` (orbax) | 12.4 GB each | dataset terms; openpi Apache | π0.5, config `pi05_base_aloha_full_sim_arx-x5_seed_0`; joint | 50-step chunk | ~8–14 GB |
| π0.5 (PhysicalRSI) | GitHub release `PhysicalRSI/XPolicyLab` `physicalrsi-skill-assets-v4`: `pi05-weights` 12.4 GB, `pi05-sparse-memory-weights` 12.4 GB, `code-skills` 7.0 GB | 32 GB | openpi Apache | π0.5 and π0.5 + sparse visual history | — | ~10–14 GB each, both resident |
| **WorldScape-R²SI WAMs** | HF `manifoldai-research/WorldScape-R2SI`: `wsp_2_flash_posttrain_robodojo` 25.3 GB, `wsp_2_memory_posttrain_robodojo` 25.3 GB | 50.5 GB | Apache-2.0 | WorldScape Policy 2 (DreamZero-compatible WAM), "interactive" mode (T5 instruction, no event memory); joint absolute | 24 / 24, obs interval 3, 8 history segments; Memory keeps 24 chunks | ≥ 26 GB **each**; the README puts them on two GPUs |

Other public RoboDojo-sim checkpoints, on the board but weak: the official bank (`ckpt/RoboDojo/*`: GalaxeaVLA, Xiaomi-1,
StarVLA variants, Pi-0, X-VLA, …), `Dexmal/DM05-MEM-Robodojo-Sim` (23 GB; 24.9), `OpenWAM/OpenWAM-Alpha-Sim-RoboDojo` (24.8 GB,
Apache; 17.2), `yinchenghust/SimpleMemVLA-RoboDojo-Sim` (12.6; Memory 33.7).

### 1.3 Requested but not usable

- **DreamZero:** the adapter is a train-it-yourself template; only `GEAR-Dreams/DreamZero-AgiBot|DROID` (45.8 GB, Apache) are
  public. No RoboDojo checkpoint (none in the official bank), no board row.
- **LingBot-VLA:** board 5.5; no RoboDojo VLA checkpoint (only RoboTwin post-trains of `robbyant/lingbot-vla-4b*`). The bank has
  `Lingbot_VA/robodojo_sim_arx_x5_v21/checkpoint_step_5000` (10.2 GB) with no board row.
- **Cosmos:** no adapter, no board row. Community fine-tunes with no license or score: `arunos728/cosmos3-robodojo-*-16k` (~91 GB
  distcp), `biyotteu/cosmos3-nano-robodojo` (551 GB). The brainstorm's Astra+Cosmos number (−8 pp) is from RoboLab, not RoboDojo.
- **Community π0.5 fine-tunes** (`Myungkyu/pi05_robodojo_*`, `zekaili/pi05_robodojo_step*`, `cjgogo/RoboDojo-pi05-checkpoints`):
  no license, no board score.

### 1.4 `gh api repos/XPolicyLab/XPolicyLab/contents/policy` (72 entries @ e08ba78)

A1 ACT AHA_WAM Abot_M0 Awomo05 Being_H05 CloudRobo_WM DP Dexbotic_DM0 Dexora_1B Discrete_Forcing **DreamZero** EgoVLA
EventVLA Evo_1 FastWAM G05 GO1 GPT_6_Astra_Direct_EEF GR00T_N17 GalaxeaVLA GigaWorldPolicy HWM_CogWAM H_RDT
Hy_Embodied_05_VLA InternVLA_A1 InternVLA_A1_5 **InternW0_delta** KinRT LDA_1B **Liber_0_lite** **Liber_0_preview**
LingBot_VA **LingBot_VLA** ME_Brain_1 ME_Dex_1_0 MM_ABC MachEmbodied-U0 Meituan_Robotics_0 Mem_0 MoPA MolmoAct2 OpenDM
OpenVLA_OFT OpenWAM PatchWAM Phoenix_MWA Pi_0 **Pi_05** Pi_0_Fast PrismWAM RDT_1B RISE **Rex_M1_preview** **Simate_beta**
SimpleMemVLA SmolVLA Spark_0 Spatial_Forcing Spirit_v15 StarVLA TinyVLA UniWAM **VPP2** ViGAR WorldLA
WorldScape_Policy_2 X_VLA X_WAM Xiaomi_Robotics_0 Xiaomi_Robotics_1 `__init__.py` demo_policy lib.sh **physicalRSI**

- `WorldScape_Policy_2` here is the **RoboTwin** adapter; the RoboDojo WAMs live in the R2SI repo as `policy/WorldScape_RSI/`.
- These are not in `main` yet: SenseNova-RoboRSI (PR #175), HarnessPolicy (#201), RunWu (#195), WorldScape_RSI (#162).

### 1.5 Per-dimension board scores (score / SR, from the evidence JSON)

| Entry | Gen-std | Gen-rand | Precision | Long-H | Memory | Open | Average |
|---|---|---|---|---|---|---|---|
| VPP2 | **52.1** / 46.4 | 20.6 / 12.6 | **43.0** / 35.2 | 56.3 / 37.0 | **56.0** / 55.6 | 4.6 / 4.1 | **39.3** / 32.3 |
| Awomo-0.5 | 37.7 / 30.9 | 22.9 / 16.8 | 39.7 / 33.5 | 40.4 / 26.8 | 42.3 / 41.2 | **24.1** / 22.9 | 35.3 / 29.6 |
| Simate-beta | 40.5 / 33.2 | **29.6** / 22.7 | 34.4 / 26.9 | **57.8** / 43.4 | 33.3 / 33.0 | 9.1 / 8.5 | 34.0 / 28.0 |
| Rex-M1 Preview | 42.6 / 34.8 | 25.5 / 17.3 | 32.8 / 23.8 | 46.9 / 33.0 | 48.5 / 48.2 | 6.7 / 6.1 | 33.8 / 27.4 |
| InternW0-Δ | 41.0 / 33.8 | 19.2 / 11.8 | 32.0 / 23.3 | 46.3 / 29.3 | 34.7 / 34.0 | 10.8 / 10.2 | 30.8 / 23.9 |
| Liber-0 Preview | 39.3 / 32.1 | 10.7 / 5.1 | 38.3 / 33.2 | 46.0 / 33.2 | 37.8 / 37.3 | 6.7 / 5.3 | 30.7 / 25.5 |
| Liber-0 Lite | 39.8 / 33.2 | 10.9 / 5.0 | 36.6 / 31.1 | 45.5 / 32.9 | 35.7 / 35.4 | 3.1 / 2.6 | 29.2 / 24.2 |
| OpenWAM-α | 33.2 / 25.6 | 8.3 / 4.1 | 18.5 / 9.3 | 34.9 / 25.3 | 10.4 / 9.1 | 1.4 / 1.1 | 17.2 / 11.9 |
| Pi-05 (official) | 20.9 / 14.9 | 5.8 / 1.4 | 12.4 / 5.5 | 23.5 / 14.7 | 5.9 / 4.7 | 2.0 / 1.7 | 11.4 / 6.9 |
| LingBot-VLA | 10.9 / 7.6 | 2.6 / 1.0 | 5.3 / 1.8 | 10.9 / 5.3 | 3.8 / 2.8 | 0.7 / 0.7 | 5.5 / 3.0 |
| *WorldScape-R²SI (agent+WAM)* | 46.6 / 40.3 | 39.3 / 31.3 | 23.5 / 15.5 | 51.5 / 34.3 | 65.8 / 63.7 | 49.2 / 47.5 | 46.6 / 39.4 |
| *PhysicalRSI (agent+VLA)* | 31.6 / 25.7 | 11.0 / 5.6 | 38.0 / 32.5 | 46.4 / 37.3 | 46.7 / 46.6 | 28.9 / 24.9 | 36.3 / 31.4 |
| *PyRUA-Lean (agent)* | 50.5 / 45.3 | 46.1 / 41.0 | 44.8 / 38.3 | 54.3 / 43.8 | 68.4 / 68.3 | 60.6 / 58.3 | 55.3 / 50.4 |
| *GPT-6-Astra (direct)* | 35.3 / 32.7 | 31.4 / 28.3 | 12.7 / 4.0 | 21.5 / 8.3 | 43.0 / 38.7 | 34.4 / 31.0 | 29.0 / 22.5 |

Notes on these rows:
- InternW0's own README reports one run over 54 tasks and 6,300 episodes: 22.92% SR, 30.35 score. It matches the board.
- VPP2's README says "the released results use `seed=1`".
- HarnessPolicy (pending, seed 0): Gen 38.1, Precision 37.5, Long-H 60.1, Memory 48.2, Open 16.7; Average 40.1 / 33.8 SR.

## 2. How the hybrids use the policy

### 2.1 WorldScape-R²SI (Manifold AI; official 46.6 / 39.4%)

**Inner agent** (`policy/WorldScape_RSI/harness`): GPT-6 Astra (`inner_agent.json`: effort `high`, `memory: full_episode`) on
top of RoboProbe's L3-Inspect-EEF harness (vendored `RoboRSI_RoboProbe`), with 34 generic Skills (SKILL.md), 4 tools, 2 primitives
and native cuRobo EEF motion. The two WAMs are code Skills `worldscape-wam-memory` and `worldscape-wam-nomemory` (Flash).

**WAM invocation:**
- `call_tool(id=worldscape-wam-*, arguments={})`. **Arguments must be empty**: the WAM gets the *original env instruction*; any
  agent-written instruction, subgoal, prompt, target or memory is forbidden and none is sent.
- Before each WAM chunk the agent must call `assess_control` with exactly `mode∈{native,flash,memory,return_home}`, `progress`,
  `evidence`, `expected_change` and booleans `instruction_sufficient`, `requires_precise_control`, `requires_agent_memory`,
  `history_available`.
- `ControlGuard` (code, not prompt) refuses the WAM when precise control or agent memory is required or the instruction is
  insufficient; after `wrong_direction`/`uncertain` or 2 stalls (until an acknowledged native correction); during homing; and on
  Flash↔Memory cycling to dodge the cooldown.
- The capability prior (`policy-applicability`) sends fine insertion, specified contact poses, attribute/identity/order rules, moving
  targets and complex discrete interaction to native tools: a WAM is "optional, not the default executor".

**Executed-action acknowledgements:** `model.py` validates the whole 24-action WAM chunk before exposing the first action, then
hands out **one action per `get_action`**. After each acknowledged action (`update_obs` with the post-step observation),
`review_pending()` calls Astra with fresh RGB/state ("compare with the previous frame and expected change") and one tool,
`review_policy_chunk(decision ∈ {continue,takeover}, progress, evidence)`. From code reading that is **up to 23 Astra calls per
24-step chunk**; a malformed review counts as takeover. Takeover discards the unexecuted suffix; only executed actions enter WAM
history; a fully executed chunk keeps the rolling state; switching policy resets the old context. `return_home` is native and
checked against the initial pose (5 mm / 5°, gripper within 0.02).

**Outer loop (`rsi/`):** collection → multimodal reflection → Skill/Memory/tool-code candidates → paired validation on the same
official layouts → immutable activation; each episode pins its release; WAM retraining is optional. **The default excludes the 8
Open tasks** (34 base + 12 random official layouts; the deep research flagged geometry-hash overlap).

**Board profile:** near VPP2 on Gen-std (46.6 vs 52.1), far better on Gen-rand (39.3 vs 20.6) and Open, but Precision 23.5 and
*below* most WAMs on `hang_mugs` (6), `insert_tubes` (12), `build_tower` (52): the agent mostly does not delegate, or delegates and
takes over. Its best cells (`push_T` 84, `stack_blocks_random` 72, `swap_blocks` 74) are agent-style wins.

### 2.2 PhysicalRSI (HKU MMLab, merged adapter; official 36.3 / 31.4%)

**One LLM call per episode** (`skill_model._start_episode` → `AgentPlanner.plan`, temperature 0). Inputs: the instruction;
observations whitelisted by `agent_observation` (RGB, intrinsics/extrinsics, joint/EE/TCP state; strips
`task_name/layout_id/seed_id/depth/segmentation/pointcloud`); `policy_memory.json` (generic guidance); a **capability catalogue**.
Output JSON `{composition, operations, rationale}`. The runtime checks that the composition is registered, that the operation
library gets exactly one operation, and that I/O contracts chain (`robodojo.observation-batch/v1 → robodojo.action-chunks/v1`).
The choice is then frozen until reset; an advisory `AgentLoop` "subgoal" (1 call) never reaches the policy.

**Typed capability contracts** (`configs/skill_capabilities.json`, 24 entries, schema `physicalrsi.skill-capability-memory/v1`):
- Each entry: `description`, `implementation.{kind,module,source sha256}`, `action_representation` (joint/ee), advisory
  `conditions`, `operation_memory` op-ids, `capability_limits`. The validator forbids `task_name`, `score_prior`, `success_rate`
  and `best_policy` keys, so the LLM sees no score table.
- **22 of 24 are task-named code programs** (align-blocks, arrange-largest-number-random, classify-objects(-by-language),
  cover-blocks, deposit-coin, general-pickup, insert-key, insert-tubes, make-kong, play-tic-tac-toe, plug-in-charger,
  pour-balls/-by-language/-liquid, press-by-number, push-t, solve-equation, sort-nesting-dolls, stack-blocks-by-language,
  store-tools, swap-t); the other 2 are `pi05` and `pi05-sparse-memory`, given the env instruction and both preloaded on the GPU.
- **So "code + π0.5 composition" is an instruction → task-program router with π0.5 as fallback.** The programs give its strong
  cells (`pour_liquid_into_cup` 93, `swap_T` 93, `make_kong` 82, `deposit_coin` 62, `insert_key` 48). The README admits they are
  task-specialised, use geometry priors and "do not claim invariance to arbitrary layouts", which fits Gen-rand at 11.0.

### 2.3 SenseNova-RoboRSI (PR #175, open; 54.17% SR author claim, not on the board)

- Planner: Codex CLI on a Responses endpoint, `reasoning_effort: max` (the report used Astra), with `vla_propose(subgoal)`,
  `vla_review(...)` and `vla_review_and_propose(...)` in a sandboxed Python tool.
- VLA: the **official RoboDojo π0.5** (seed-0 step 59999, 12.4 GB, pinned in `pi05/checkpoint.json`), **prompted with an
  agent-written English subgoal**, which is off-distribution for a checkpoint trained on task instructions.
- Each proposal is 50 joint actions, FK-converted to dual-arm EEF targets, reviewed against images, ICL task skills and geometry:
  `accept` releases the first `approved_steps` (2–50, default 5) unchanged; `modify` supplies that many full targets with ≥ 1 change;
  `reject` moves nothing or supplies replacement targets. Every target must be within **5 cm / 0.35 rad of the measured EE pose**;
  the span runs through the bounded EEF IK controller with no LLM call inside. Prompt prior: long spans in free transit, ≤ 5 near
  contact, 2–4 for precision.
- Also 27 frozen ICL packages (task skills, notes, helpers from public training data) and `review()` checkpoints (observed_state,
  demo_comparison, why_not_done, next_plan, ending_check).
- Report Table 13: ICL + VLA uses **19.33 M tokens per episode** (6.9× the chunked condition); closed-loop VLA 23.61 → 25.83% with
  4-step chunks; adaptive chunks 20.0% vs fixed-4 11.4%. The 54.17% headline mixes ICL, VLA and a large budget.

### 2.4 HarnessPolicy (PR #201, open; seed-0 self-report 40.12 / 33.83%)

- **Stack:** DeepSeek-Flash (`api.deepseek.com`), Simate (RoboDojo's released checkpoint), SAM 3 and a kinematics service, ~23 GB
  VRAM in total (models load on demand when they do not fit).
- **The model writes programs, not actions** ("move through free space, at most one contact, check, print"). It is re-asked only at
  decision points: `start`, an unhandled `escalation`, `error`, `program_end`.
- **The first motion is a strategy choice** `start(strategy, kind)`, from `list_policies()` descriptions and the first images:
  **A** a learned task of a kind the policy usually finishes: *the policy runs the whole task from the untouched start pose*, the
  model supervises; **C** another learned task: policy first, programs finish what it leaves; **B** picking one of several static
  objects by name/label/number/picture: native tools only. Moving targets always go to the policy.
- The handover runs **uninterrupted** until the episode ends or the policy stops (arms still 5 s after moving); then the model gets
  4 `post_policy_calls`, after which the policy runs on by itself.
- **Measured: watching and interrupting hurts.** v1.3.0 ran the handover in 10 s slices with a look between them: **31.9 vs
  Simate-alone 42.4** on those tasks; "the model's own tools after a run added cost, not score".
- Eval pack 0 (125 episodes): 113 handed over, **2.3 calls and $0.003 per episode**; 26.2 vs Simate-alone 22.2 on the same layouts.
  The full seed-0 run beats Simate where programs take over (`general_pickup` 58 vs 49, `solve_equation` 46 vs 0) and matches it
  elsewhere.

### 2.5 RunWu (PR #195, open)

- **No VLA or WAM.** It is GPT-6 Astra at `xhigh` calling deterministic EEF/code tools from per-task content-hashed "PolicyBundles".
  It uses 29 keyframe packs from the public training data and asset dimensions to ±1 mm.
- Verified number: Generalization only, 43.83% SR / 50.27 score (seed 0).
- If `get_action` takes more than 100 s it returns a hold and delivers the decision on the next call. That is a useful pattern for
  RoboDojo's 120 s client timeout.
- It is relevant here only as a counter-example: per-task priors, not a VLA, carry its numbers.

### 2.6 The LLM ↔ policy interface, compared

| | Text sent to the policy | Granularity | Chunk accept/edit | When control returns to the LLM | LLM calls per episode |
|---|---|---|---|---|---|
| PhysicalRSI | env instruction | whole episode | none | never (frozen) | 1 |
| HarnessPolicy | env instruction | whole task, from the start pose | none | the policy stops (arms still 5 s) or the episode ends; then 4 calls | ~2.3 |
| R²SI | env instruction (args `{}`) | stage = series of 24-step chunks | continue/takeover after **every** acknowledged action; takeover drops the suffix | each action; guard forces native after stall/wrong/uncertain | high (Astra per WAM step) |
| SenseNova | **agent subgoal** | one 50-step proposal | accept / modify / reject a 2–50 prefix; ±5 cm / 0.35 rad bound | after every approved span | very high (19 M tokens) |

Consensus and lessons:
- Nobody gets value from sub-goal prompting of a RoboDojo VLA. The checkpoints were trained on whole-task instructions from the start
  pose, and R²SI forbids sub-goals outright.
- Fine-grained supervision costs calls and, in the one measured A/B (HarnessPolicy), costs score.
- What works is a **routing decision at episode start**, a stall detector, and the agent finishing what the policy left. The LLM's
  value is in the tasks the policy cannot do: identification by language/label/image, Open, Gen-random.

## 3. Training data vs none, and where VLAs win per task

**Data** (`RoboDojo-Benchmark/RoboDojo`, `data/RoboDojo/`):
- 34 task folders with 100 HDF5 demos each, plus `dlc/arx_x5` (100 episodes, the domain-randomisation pack).
- In total 3,500 episodes, 1.86 M frames at 25 Hz. Exports: hdf5, LeRobot v2.1/v3.0, EE LeRobot v3.0, depth, real.

| Has demos (34) | No demos: Open (8) |
|---|---|
| all 12 Generalization (base layouts only; `_random` variants are unseen lighting, floor and distractors), all 8 Precision, all 8 Long-Horizon, all 6 Memory | `align_blocks`, `solve_equation`, `stack_blocks_by_language`, `general_pickup`, `classify_objects_by_language`, `pick_from_conveyor_by_image`, `store_tools_in_toolbox`, `pour_by_language` |

**Per-task scores** (bundle data; 50 episodes per task, 25+25 for Gen):
- Column key: Sim = Simate, Rex = Rex-M1, Awo = Awomo, IW0 = InternW0-Δ, LbP = Liber-0 Preview, pi05 = official π0.5,
  R2SI = WorldScape-R²SI, PRSI = PhysicalRSI, Hexa = HexaShell, Astra = Astra direct.
- **Bold** = best learned model. The last column is the best learned model minus the best of agents and hybrids (R2SI, PRSI, PyRUA,
  Hexa, Astra).

| task | VPP2 | Sim | Rex | Awo | IW0 | LbP | pi05 | R2SI | PRSI | PyRUA | Hexa | Astra | bestL−bestA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G:stack_bowls | **92** | 89 | 90 | 74 | 89 | 84 | 76 | 79 | 78 | 82 | 92 | 69 | -0 |
| G:stack_bowls_random | 19 | **67** | 48 | 48 | 29 | 19 | 14 | 78 | 10 | 65 | 82 | 64 | -15 |
| G:push_T | 3 | 0 | 1 | **5** | 0 | 0 | 0 | 84 | 65 | 80 | 100 | 68 | -95 |
| G:push_T_random | 0 | 0 | 0 | **16** | 0 | 0 | 0 | 80 | 0 | 92 | 92 | 52 | -76 |
| G:pack_objects_into_box | 38 | **50** | 40 | 31 | 27 | 31 | 23 | 27 | 26 | 53 | 38 | 11 | -3 |
| G:pack_objects_into_box_random | 23 | **43** | 29 | 16 | 27 | 17 | 15 | 36 | 15 | 39 | 28 | 12 | +4 |
| G:fold_clothes | 78 | 83 | 75 | 75 | **84** | 81 | 48 | 69 | 49 | 58 | 64 | 77 | +7 |
| G:fold_clothes_random | 64 | **78** | 65 | 42 | 38 | 9 | 17 | 62 | 43 | 60 | 31 | 76 | +2 |
| G:hang_mugs | **74** | 14 | 45 | 36 | 40 | 33 | 10 | 6 | 9 | 27 | 22 | 1 | +47 |
| G:hang_mugs_random | 14 | 11 | **20** | 10 | 6 | 4 | 5 | 3 | 2 | 28 | 16 | 1 | -8 |
| G:sweep_blocks | **20** | 5 | **20** | 1 | 9 | 7 | 0 | 0 | 0 | 4 | 8 | 8 | +12 |
| G:sweep_blocks_random | 1 | 0 | **8** | 0 | 4 | 0 | 0 | 0 | 0 | 8 | 8 | 0 | +0 |
| G:pour_liquid_into_cup | 68 | 57 | **73** | 64 | 68 | 57 | 28 | 8 | 93 | 4 | 4 | 8 | -20 |
| G:pour_liquid_into_cup_random | 32 | 53 | **61** | 52 | 52 | 12 | 1 | 0 | 9 | 4 | 0 | 0 | +52 |
| G:make_toast | **62** | 20 | 28 | 19 | 40 | 20 | 9 | 50 | 6 | 26 | 3 | 0 | +12 |
| G:make_toast_random | **38** | 8 | 14 | 7 | 23 | 16 | 3 | 25 | 5 | 17 | 0 | 1 | +13 |
| G:arrange_largest_number | 12 | 18 | 23 | 26 | 8 | **28** | 5 | 77 | 4 | 84 | 18 | 76 | -55 |
| G:arrange_largest_number_random | 5 | 4 | 2 | **10** | 4 | 7 | 0 | 65 | 32 | 76 | 43 | 61 | -67 |
| G:sort_nesting_dolls_by_size | **35** | 28 | 29 | 23 | 15 | 17 | 11 | 16 | 16 | 84 | 72 | 16 | -49 |
| G:sort_nesting_dolls_by_size_random | 3 | **5** | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 56 | 32 | 20 | -51 |
| G:store_laptop_and_headphones | **57** | 49 | 33 | 19 | 35 | 38 | 19 | 56 | 13 | 3 | 8 | 1 | +1 |
| G:store_laptop_and_headphones_random | 34 | **59** | 55 | 42 | 36 | 39 | 13 | 50 | 13 | 6 | 2 | 2 | +9 |
| G:stack_blocks | **88** | 73 | 55 | 78 | 78 | 76 | 23 | 86 | 19 | 100 | 90 | 90 | -12 |
| G:stack_blocks_random | 15 | 26 | 3 | **32** | 11 | 5 | 1 | 72 | 2 | 100 | 89 | 86 | -68 |
| P:fasten_screws | 64 | 41 | 54 | **69** | 43 | 62 | 15 | 35 | 8 | 81 | 30 | 36 | -12 |
| P:insert_tubes | **83** | 41 | 51 | 79 | 64 | 83 | 18 | 12 | 79 | 66 | 41 | 6 | +4 |
| P:plug_in_charger | 17 | **29** | 3 | 7 | 5 | 27 | 0 | 20 | 31 | 16 | 32 | 0 | -3 |
| P:pour_balls_into_vase | **61** | 45 | 50 | 39 | 54 | 31 | 13 | 42 | 39 | 66 | 80 | 4 | -19 |
| P:play_Xylophone | 0 | 3 | 1 | **31** | 0 | 0 | 0 | 10 | 1 | 62 | 58 | 8 | -31 |
| P:deposit_coin | **26** | 16 | 23 | 24 | 19 | 19 | 3 | 8 | 62 | 44 | 26 | 16 | -36 |
| P:insert_key | **15** | 11 | **15** | 12 | 10 | 1 | 12 | 10 | 48 | 0 | 15 | 14 | -33 |
| P:build_tower | 79 | **87** | 66 | 56 | 62 | 84 | 38 | 52 | 37 | 23 | 48 | 16 | +36 |
| LH:put_bottles_into_dustbin | 96 | **100** | 94 | 94 | 99 | 98 | 80 | 99 | 84 | 92 | 70 | 35 | +1 |
| LH:play_tic_tac_toe | 46 | 86 | 84 | 59 | 71 | **96** | 8 | 73 | 98 | 97 | 99 | 11 | -3 |
| LH:classify_objects | **62** | **62** | 52 | 31 | 17 | 20 | 25 | 92 | 56 | 97 | 85 | 69 | -35 |
| LH:fill_pen_holder | 50 | 50 | 39 | 48 | **62** | 50 | 23 | 32 | 22 | 24 | 6 | 5 | +30 |
| LH:fill_egg_holder | 10 | **17** | 10 | 7 | 11 | 13 | 2 | 17 | 2 | 21 | 9 | 6 | -5 |
| LH:organize_table | 63 | **72** | 62 | 50 | 60 | 48 | 23 | 62 | 27 | 25 | 14 | 40 | +10 |
| LH:play_stacking_toy | **44** | 0 | 1 | 4 | 4 | 27 | 0 | 13 | 0 | 78 | 58 | 7 | -34 |
| LH:make_kong | **79** | 76 | 33 | 30 | 47 | 18 | 27 | 24 | 82 | 0 | 2 | 0 | -3 |
| M:cover_blocks | **100** | **100** | **100** | **100** | 95 | **100** | 19 | 99 | 100 | 100 | 100 | 49 | +0 |
| M:match_and_pick_from_conveyor | 86 | 56 | 66 | 66 | **92** | 64 | 14 | 78 | 7 | 86 | 74 | 62 | +6 |
| M:swap_T | 25 | 0 | 19 | 0 | 1 | **53** | 1 | 20 | 93 | 92 | 96 | 18 | -43 |
| M:press_by_number | **96** | 1 | 90 | 67 | 11 | 0 | 0 | 80 | 79 | 0 | 96 | 70 | +0 |
| M:imitate_sorting_sequence | 3 | 5 | 2 | **16** | 3 | 9 | 2 | 44 | 1 | 92 | 63 | 59 | -77 |
| M:swap_blocks | 26 | **39** | 13 | 5 | 6 | 1 | 0 | 74 | 0 | 40 | 78 | 0 | -39 |
| O:align_blocks | 0 | 0 | 0 | **58** | 0 | 0 | 0 | 96 | 86 | 76 | 78 | 50 | -38 |
| O:solve_equation | 0 | 0 | 0 | **50** | 0 | 0 | 0 | 66 | 60 | 72 | 86 | 40 | -36 |
| O:stack_blocks_by_language | 14 | 17 | 3 | 15 | **21** | 16 | 2 | 78 | 0 | 100 | 96 | 48 | -79 |
| O:general_pickup | 18 | 49 | 45 | 61 | **63** | 34 | 12 | 86 | 47 | 86 | 82 | 84 | -23 |
| O:classify_objects_by_language | 5 | 6 | 6 | **9** | 3 | 4 | 1 | 67 | 21 | 95 | 79 | 46 | -86 |
| O:pick_from_conveyor_by_image | 0 | 0 | 0 | 0 | 0 | 0 | **1** | 0 | 0 | 22 | 28 | 4 | -27 |
| O:store_tools_in_toolbox | 0 | 0 | 0 | **1** | 0 | 0 | 0 | 0 | 7 | 16 | 14 | 2 | -15 |
| O:pour_by_language | 0 | 0 | 0 | 0 | **0** | 0 | 0 | 0 | 10 | 18 | 0 | 0 | -18 |

What the table says:
- **VLAs/WAMs beat every agent on:** `hang_mugs` (+47, VPP2), `build_tower` (+36, Simate), `pour_liquid_into_cup_random` (+52, Rex),
  `fill_pen_holder` (+30, InternW0), `make_toast` ± random (+12/+13, VPP2), `sweep_blocks` (+12), `organize_table` (+10, Simate).
- **They crush PyRUA specifically on:** `store_laptop_and_headphones` 57 vs 3, `make_kong` 79 vs 0, `press_by_number` 96 vs 0,
  `pour_liquid_into_cup` 73 vs 4, `insert_tubes` 83 vs 66.
- **They are useless on:** `push_T` (0–5); identification/ordering (`arrange_largest_number`, `sort_nesting_dolls`,
  `classify_objects_by_language`, `imitate_sorting_sequence`, `swap_T`); `play_Xylophone`; all of Open except `general_pickup`
  (InternW0 63, Awomo 61). Awomo's odd 58/50 on `align_blocks`/`solve_equation` is unexplained (Open has no demos).
- **Gen-random:** VPP2 collapses (`stack_bowls` 92 → 19); Simate degrades least (`stack_bowls_random` 67, `fold_clothes_random` 78,
  `store_laptop_random` 59).
- **Caveat:** each cell is 50 episodes (25+25 for Gen), so ±7–14 pp SR is noise.

## 4. What a VLA tool is worth: router bounds (computed from the bundle numbers)

Method: pick, per task (and per `_random` variant), whichever of {agent, policy} has the higher board score, then recompute RoboDojo's
Average. The Average is the mean of the 5 dimensions, with Gen = mean(std, rand).

This is an **upper bound**: it uses hindsight on the numbers it scores; a real LLM must route from the instruction and first
frames; and it assumes the policy runs from an untouched start pose, as on the board.

| Agent + policy (oracle per task) | Gen-std | Gen-rand | Prec | Long-H | Mem | Open | **Average score** | Average SR |
|---|---|---|---|---|---|---|---|---|
| PyRUA alone | 50.5 | 46.1 | 44.8 | 54.3 | 68.4 | 60.6 | 55.3 | 50.4 |
| PyRUA + VPP2 | 70.9 | 52.8 | 55.8 | 72.8 | 84.4 | 60.6 | **67.1** | 61.4 |
| PyRUA + Simate | 61.4 | 56.6 | 55.8 | 74.0 | 68.5 | 60.6 | 63.6 | 57.9 |
| PyRUA + Rex-M1 | 63.7 | 55.3 | 52.0 | 65.3 | 83.4 | 60.6 | 64.1 | 58.1 |
| PyRUA + VPP2 + Simate | 71.4 | 58.3 | 58.4 | 74.3 | 84.4 | 60.6 | 68.5 | 63.1 |
| Astra-direct alone | 35.3 | 31.4 | 12.7 | 21.4 | 43.0 | 34.4 | 29.0 | 22.5 |
| Astra + Simate | 52.6 | 45.0 | 35.4 | 59.6 | 57.9 | 34.4 | 47.2 | 39.6 |
| Astra + VPP2 | 63.1 | 41.8 | 44.0 | 57.2 | 65.4 | 34.4 | 50.7 | 42.4 |
| Astra + any released VLA/WAM | 65.1 | 49.6 | 50.1 | 67.2 | 73.1 | 36.6 | 56.9 | 49.7 |

Reality checks: HarnessPolicy (≈ "DeepSeek-Flash + Simate") got 40.1 vs the Astra + Simate oracle of 47.2 (imperfect routing,
weak LLM code); R²SI ("Astra + WAM, fine-grained") 46.6; PhysicalRSI 36.3. **Estimate: a realistic router recovers ~50–75% of the
oracle gap.**

## 5. Recommendation

**Yes, integrate one VLA as a tool, but as a whole-task executor chosen at episode start.** Neither a sub-goal actuator nor a
chunk-reviewed proposer:
- a released policy beats every agent and hybrid on 16 of 54 task cells (8 by ≥ 10 points) and beats PyRUA on 23; those are
  exactly where agents are weakest
  (contact-rich Precision/Long-H: `build_tower`, `insert_tubes`, `hang_mugs`, `make_kong`, `fill_pen_holder`, `organize_table`,
  `store_laptop…`, `pour_liquid…`, `press_by_number`);
- the one A/B on supervision granularity (HarnessPolicy 1.3.0) says interrupting the policy loses about 10 points.

### Which policy

1. **Simate-beta first:** 13.5 GB fp32, Apache weights, also in the official bank; joint actions, multi-task,
   instruction-conditioned, built-in visual memory; best released VLA on Long-H (57.8) and Gen-rand (29.6); strong on `build_tower`,
   `organize_table`, `fill_pen_holder`, `put_bottles`, `fold_clothes` (±rand), `store_laptop` (±rand), `stack_bowls_random`,
   `make_kong` 76; HarnessPolicy already uses it. ~16–18 GB VRAM fits compute3's 5090 (32 GB); a bf16 cast might fit compute2's
   4090 next to Isaac, but must be re-validated because the board ran fp32.
2. **Rex-M1 as the cheap second** (10.2 GB, Apache, ee actions). Strong on `pour_liquid_into_cup` (73/61), `press_by_number` 90 and
   `hang_mugs` 45. Its EE actions go through RoboDojo IK, which silently drops an arm when IK fails.
3. **VPP2 only if the pilot pays.** It is the best per-task executor and adds about +3.5 Average over Simate in the PyRUA bound. But it
   needs ~54 GB of weights and a 96 GB card ("smaller devices unverified"); that means renting an RTX PRO 6000 or H100 node and
   serving it over ws. The weights carry no explicit license.
4. **Not worth it:** official π0.5 (11.4); LingBot, DreamZero, Cosmos (no checkpoint or score); the R²SI WAMs (2 × 25 GB, no
   standalone score, and the hybrid using them is weak exactly where a policy should help). Awomo is CC-BY-NC (fine for research,
   but a listed entry must be "open"); keep it as an option.

### Dimensions and tasks it should cover

- **Gen-std, Long-Horizon, Precision, and Memory partly** (`press_by_number`, `match_and_pick`, `cover_blocks`).
- **Not Open:** no demos; the board shows Open ≤ 10 for every VLA.
- **Gen-rand only task by task:** `pour_liquid_into_cup_random`, `store_laptop…_random`, `fold_clothes_random`,
  `pack_objects…_random`, where Simate/Rex hold up (+10.5 Gen-rand for PyRUA + Simate in the bound).
- **Not identification or ordering tasks** (`push_T`, `arrange_largest_number`, `sort_nesting_dolls`, `swap_T`,
  `imitate_sorting_sequence`, `classify_*_by_language`).
- The LLM stays the controller for those and does the routing. It does not get a score table: route on generic kinds from the
  policy description, as PhysicalRSI and HarnessPolicy do.

### Interface for controlr (fits rule 1: the model is on top; the policy is a config-switched tool)

- **At turn 0, before any motion,** the model sees one extra tool, `POLICY run [until=stop|steps N]`, with a fixed description of
  what the policy can and cannot do. **The env instruction is passed verbatim; no sub-goals.** The policy is in distribution only
  from the start pose; if the model moves first, it is HarnessPolicy's strategy C.
- **The shim runs the XPolicyLab reference loop** against a separate policy server: `update_obs` every step, every step through
  the `SafetyEnvelope` (joint/FK bounds, table clearance). It stops at episode end, an arm stall (no joint motion for ≥ 5 s = 125
  steps after having moved), a step budget or an envelope stop. The model then gets the steps used, a frame diff and the current
  frames, and finishes with native turns or calls `POLICY` again.
- **No per-chunk review.** At most an optional watchdog frame every K steps as a safety stop, not for approval.
- Config axes: `policy.name`, `policy.host`, `policy.max_steps`, `policy.stall_steps`, `policy.allowed` (default off). These are
  new fields, so no default changes.

### Expected gain

- **Against our current loop:** the closest proxy is Astra-direct at about 29. Adding Simate as a whole-task route has an oracle bound of
  +18 score / +17 SR. **Realistically +9 to +13 score.**
- **Against a PyRUA-class agent** (after our code-cell work), the bound is +8 (Simate) to +12 (VPP2). **Realistically +4 to +8.**
- In the PyRUA bound the gains are Long-H +18–20 dimension points, Gen-std +11 (Simate) / +20 (VPP2), Precision +11, Gen-rand
  +7–11, Memory +0 (Simate) / +15–16 (VPP2, Rex).
- **LLM cost drops on handed-over tasks:** HarnessPolicy measured 2.3 calls per episode.
- **Validate with a pilot first:** 3 trained tasks where Simate ≫ agents (`build_tower`, `organize_table`, `fill_pen_holder`) plus 1
  where it should lose (`arrange_largest_number`). Run 10 layouts × 2 per task, checking the router choice and the stall handover.

### Cost on our side

- **compute2** (RTX 4090 24 GB, RoboDojo Isaac Sim 5.1 resident, its VRAM not yet measured): fp32 Simate (16–18 GB) plus Isaac
  almost certainly does not fit; a bf16 Simate (~8–9 GB) or Rex (~12 GB) might. Measure first (`nvidia-smi` during a run).
- **compute3** (RTX 5090 32 GB, shared; Isaac 6.0 for the UR3 when used) can host the Simate server alone with the sim on compute2
  over ws. But it is on Wi-Fi and `update_obs` ships ~2.7 MB raw per env step: 300–1,500 steps = 1–4 GB per episode, i.e. minutes
  of wall-clock on top of policy latency. Stop the server when done (rule 4).
- **Latency is unpublished.** My estimate is about 0.1–0.4 s per chunk for 3–5B flow VLAs on a 4090/5090, and seconds per chunk for
  14B-video WAMs. The sim is paused while the policy thinks, so this costs wall-clock only.
- **Disk:** Simate 13.5 GB, plus the PaliGemma tokenizer (gated, needs HF access acceptance), plus its conda env (torch 2.7/cu128,
  ~10 GB). Rex adds 10.2 GB plus Qwen3-VL-4B processor files. VPP2 needs about 54 GB.
- **Submission:** "module stacking" is said not to count as novelty. A policy route must be framed inside our method:
  a typed handover with a stall/receipt protocol, plus LLM recovery after the policy. It must also be disclosed in the paper plan.
  Everything used is public, so the open-artifact rule is satisfied.
