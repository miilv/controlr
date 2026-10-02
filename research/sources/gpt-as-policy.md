# GPT-as-Policy: "GPT 6 Astra as an Embodied Policy" (anonymous report, Galbot-affiliated)

*Deep-dive note, written 2026-10-01. Sources read: the rendered site (all `?view=` values and both languages), both GitHub repos (cloned to `research/repos/GPT-as-Policy` and `research/repos/public-website`), the HF dataset `YuMoool/astra-robodojo-rollouts` (metadata plus 11 of the 100 episode core bundles, in `research/repos/astra-robodojo-rollouts`), the repo's GitHub issues, the follow-up arXiv 2609.38537, and third-party coverage. Anything I computed myself is marked **[derived]**. Anything I could not check is marked **UNVERIFIED**.*

## 1. TL;DR

- **What it is.** A simulation study of OpenAI `gpt-6-astra` at reasoning effort `xhigh` used as a closed-loop robot policy. The robot is a bimanual ARX X5 in **RoboDojo** (Isaac Sim 5.1, 25 Hz control). The study compares two modes:
  - **Direct:** the LLM outputs bounded dual-arm end-effector (EEF) targets. Damped-least-squares (DLS) inverse kinematics turns them into joint commands.
  - **Hybrid:** π0.5 (the RoboDojo-finetuned OpenPI/JAX checkpoint) proposes a 50×14 joint chunk. The LLM sees that chunk as a forward-kinematics (FK) EEF trajectory, then either runs a 1–15-step prefix or replaces it with a 1–5-step bounded EEF correction.
- **The harness is OpenAI's Codex CLI**, not a custom API loop. `codex app-server` runs one persistent thread with 2–3 "dynamic tools" for the simulator and policy. Codex's own tools stay on: shell, image viewing, file editing, plus a `NOTES.md` scratchpad. All prompts are Markdown "skills".
- **Results.** These come from 10 RoboDojo tasks × 5 paired cases.

  | Method | Success | Mean Score |
  |---|---|---|
  | Hybrid | 24/50 (48%) | 62.60 |
  | Direct | 13/50 (26%) | 37.81 (48 scored) |
  | π0.5, reweighted public leaderboard (not rerun) | 15.67% | 24.43 |

  - In Hybrid, the LLM authored 14.4% of executed control steps.
  - On **RoboLab** (single-arm Franka, 10 tasks × 5), the order flips: Direct 49/50, Hybrid 46/50, π0.5 18/50. Those runs had retries and are not paired.
- **Cost and latency.** Hybrid used 624.8M tokens and Direct 1.13B, of which 97–98% was cached input.
  - Physics is paused while the model thinks.
  - In my sample of 11 released episodes, wall time was **18–106 s per decision**, or **46–318× slower than real time** [derived].
  - At list API prices that is about **$17 per Hybrid episode and $29 per Direct episode** [derived]. In practice the runs used a pool of 15 ChatGPT subscription accounts.
- **Who.** Jiayi Su, Yixin Zheng (equal contribution), Mi Yan, Li Yi, Zhizheng Zhang and He Wang (corresponding). The evidence for a **Galbot (银河通用)** affiliation is strong (§9). The work was later folded into "Galbot Team", arXiv 2609.38537.

## 2. Release artefacts

| Artefact | Contents |
|---|---|
| `github.com/anonymous-report-421/GPT-as-Policy` | One squashed commit, "Initial public release", 2026-09-15 02:21 UTC. MIT-licensed controller source (`hybrid_rollout/`), the prebuilt report (`report_web/`) and `public_results/`. The README says it is "history-free". About 565 stars on 2026-10-01. The old name `eval-of-gpt-6-astra-as-policy` redirects here (HTTP 301). |
| `github.com/anonymous-report-421/public-website` | The GitHub Pages site: 100 RoboDojo episode videos, 43 clips, and a RoboLab gallery (150 MP4s in release `robolab-gallery-20260914`). |
| Site `…/public-website/?lang=en&view=1` | A JS single-page app built with OpenAI Codex's "Data app" plugin (`report_site/app/AGENTS.md`, `.openai/hosting.json`). The `view` parameter does nothing: views 0–3 render byte-identical text. `lang=zh` is a translation of the same content. |
| HF dataset `YuMoool/astra-robodojo-rollouts` | 100 selected episodes, CC-BY-4.0. Each has a `core.zip` (decisions, π0.5 proposals, executed actions, the 480×360 previews actually sent to the model, prompts, a frozen source snapshot, public reasoning summaries) and an `observations.zip` (lossless 640×480 frames for every control step). About 7.9 GB of core bundles in total. HF user `YuMoool` has the display name "Jiayi Su". |
| arXiv 2609.38537 (2026-09-29) | "Systematically Exploring the Capabilities of GPT-6 Astra as Embodied Policies", 33 authors as "Galbot Team". It contains this study as §3, adds dexterous, mobile, navigation, locomotion and humanoid domains, and links "GPT-as-Policy" as its code. |

The runs themselves happened about 2026-09-12/13 UTC (agent-event timestamps), and the report snapshot is stamped `generatedAt: 2026-09-13`. Panel id: `robodojo_v3_pool15_20260912_01_fixtures`.

## 3. Architecture

```
               ┌──────────────────────── container (one frozen case) ─────────────────────────┐
 Isaac Sim 5.1 │ robodojo_server/server.py  ── RPC :19113 ──┐                                  │
 RoboDojo env  │  RoboDojoSession (no rollback, 25 Hz,       │                                  │
 dual ARX X5   │  3×RGB 640×480, 14-D proprio, link6 poses)  │                                  │
               │                                             ▼                                  │
 OpenPI/JAX    │ pi05_server/server.py ── ws :18830 ──► skill/run.py (Python host)              │
 π0.5 ckpt     │  50×14 abs. joint chunk, ~0.43 s/infer      │  RoboDojoTools / GPTOnlyTools     │
               │                                             │  validates, executes, records     │
               │                                             ▼  JSON-RPC over stdio              │
               │                              `codex app-server --stdio --strict-config`        │
               │                               model=gpt-6-astra, effort=xhigh, wire_api=responses
               │                               dynamicTools: robodojo_start / pi05_infer /       │
               │                                 robodojo_execute   (Direct: robodojo_act)       │
               │                               + native: shell, view_image, file edit, NOTES.md  │
               └────────────────────────────────────────────────────────────────────────────────┘
Hybrid decision loop (the model drives it; the host only enforces order):
  obs(3 RGB→480×360 PNG previews, proprio, EEF) → pi05_infer → {~~50×14 chunk + 50-row FK EEF traj
  (~37–42 KB JSON)~~ [corrected: 50-row FK EEF traj + action_diagnostics + prev obs/result,
  34.3–40.9 KB JSON in released packets; raw 50×14 chunk stays on disk in actions.npz]} → LLM gate assessment → robodojo_execute{student k≤15 | eef/edit ≤5 steps}
  → per-tick bounded DLS IK (eef) or raw chunk prefix (student) → new obs (3 images attached) ↺
```

## 4. Code walkthrough

All paths are relative to `research/repos/GPT-as-Policy/`.

**Model configuration**
- ~~`hybrid_rollout/robodojo/settings.py:6-10`~~ [corrected: `settings.py:8-12`] sets `MODEL='gpt-6-astra'`, `EFFORT='xhigh'`, `WIRE_API='responses'` and `DEFAULT_CODEX_IMAGE_MAX_EDGE=480`.
- `codex_backend/config.toml` adds `model_reasoning_summary="auto"`, `fast_mode=false`, and analytics off. There is no temperature or max-tokens setting; Codex defaults apply.
- `codex_backend/profiles.py:19-37` defines the auth profiles:
  - `galbot`: API key through a LiteLLM gateway. This is the default (`:47`).
  - `koozhan`: a relay.
  - 15 `codex_{a,b,c}[_2.._5]` profiles: managed **ChatGPT logins** at `chatgpt.com/backend-api/codex`.
- The released episodes record `teacher_model_provider: "openai"`, so they ran on the subscription pool. `subscription_migration.py` documents the move ("Drain the last Galbot container first"). Codex CLI version: `codex-cli 0.153.4`.

**Agent host: `skill/run.py`**
- `agent_config` (`:76-90`) turns on `features.shell_tool` and `features.view_image`. The agent workspace is writable; host artefacts are read-only.
- `tool_specs` (`:93-115`) defines three blocking function tools. Their descriptions are quoted in §5.
- `CodexPolicy.__init__` (`:133-202`) builds the developer prompt from SKILL.md, gate_prompt.md and teacher_context.md. It then calls `thread/start` with `developerInstructions`, `dynamicTools`, `approvalPolicy='never'` and `allowProviderModelFallback=False`. If the server reports a different model or effort, it aborts (`:183-185`).
- `run` (`:240-395`) is the event loop:
  - The whole episode is **one Codex turn**, opened by a fixed user message (`:241-248`). Up to two "unfinished, continue" turns are allowed (`:365-370`).
  - Tool calls go to Python handlers (`:282-322`).
  - `InputError` rejections return `no_execution=True, retryable=True` without touching physics (`rejected_input`, `:31-66`).
  - Network failures resume the **same thread**, up to 20 times with 10 s backoff (`network_recovery.py`, `_network_continue :211-238`).
  - Token usage is logged from `thread/tokenUsage/updated` (`:274-281`). `CODEX_MAX_TOTAL_TOKENS=0` turns the token cap off.
- `content_items` (`:118-130`) sends the JSON packet as text and attaches the three PNGs as base64 `inputImage`. Images are attached only for start/execute/act results, not for `pi05_infer`.
- `image_preview.py` downscales the images the model sees with Lanczos to a 480 px max edge (640×480 → 480×360). π0.5 gets full resolution. The model can open the originals with its native image viewer.
- `skill/schema.py:6-50` is the strict JSON Schema for `response`. `mode ∈ {student, edit, eef, stop}`, `steps ∈ [1,15]`, and per-arm `edit{delta_position, delta_rotation_vector, gripper∈keep/open/closed}` or `target{position, quaternion_wxyz, gripper_closed}`. It also requires an `assessment` object (§5). The Direct schema allows only `mode='eef'` with `steps ≤ 5`.

**Rollout services: `robodojo_server/client.py`**
- `next_call` (`:98-113`) hands the model the exact next tool name and fresh output path. `_check` (`:115-125`) rejects calls in the wrong order, with stale paths, or with paths that already exist.
- `infer` (`:204-237`):
  - Runs π0.5.
  - Calls `fk_preview` to turn all 50 joint rows into both arms' link6 poses plus gripper opening.
  - Adds `action_diagnostics` (shape, finiteness, opening range, maximum successive joint step).
  - Builds the request with a fresh `request_id`.
- `execute` (`:239-317`):
  1. Validates the response.
  2. Refuses `stop` on frozen-benchmark cases (`:246-254`).
  3. Records the decision on the simulator side.
  4. Runs the 1–15-step chunk prefix (`student`), or, for each correction step, asks the simulator for a fresh IK solution, executes it, and appends the step.
- `gpt_only_client.py:99-148` is the Direct equivalent (`act`).

**Kinematics and safety**
- `robodojo_server/kinematics.py:42-65` is the per-tick IK:
  - Numerical-Jacobian DLS: `dq = Jᵀ (JJᵀ + 0.05² I)⁻¹ e`.
  - Task error clipped to **2 cm / 0.1 rad** per tick (`:49`).
  - Joint step clipped to **±0.05 rad** within soft limits (`:58`).
  - Recomputed after every simulator ACK. There is no planner or collision checking (cuRobo is explicitly disabled, `server.py:87-89`).
- `check()` (`:82-94`) refuses to act if URDF FK disagrees with measured link6 by more than 2 mm or 0.01 rad.
- `validation.py:30-61` checks every decision. Targets must be within **5 cm / 0.35 rad** of the measured EEF, the quaternion norm must be within 1e-4 of 1, and the gripper flag must be a bool. It also rejects CJK characters in public prose (`:17-27`).
- `action_edit_kinematics.py:20-45` implements `edit`: offsets are ramped linearly by (i+1)/n over the student's FK targets.
- `gate_assessment.py:33-64` enforces the gate structurally. An edit or EEF takeover requires `execution_status=failed` or `intent_status=misaligned`, and ~~`not_started` is valid only at step 0~~ [corrected: `not_started` is *required* at step 0 and forbidden elsewhere — biconditional, `gate_assessment.py:56-57`]. It checks **consistency only**; it cannot verify that the model's claim is true.

**Simulator: `robodojo_server/session.py`**
- One reset per process, then no rollback.
- `chunk_step` (`:140-169`) takes 1–15 absolute 14-D actions, steps natively, and saves an NPZ plus a 3-camera frame for every tick.
- Score is 1.0 on native success, otherwise the native partial credit divided by 100 (`:171-196`).
- π0.5 client (`pi05_server/client.py:36-58`): the prompt is always the original instruction, never a subgoal from the LLM (confirmed in issue #4). The gripper channels are clipped to [0, 1].

**Evaluation plumbing**
- `paired_evaluation.py` keeps exactly one native-complete "v3-context" attempt per case. A campaign stack (`campaign.py`, `quota_guard.py`, `pool15_quota.py`, `idle_timeout_policy.py`, …) manages subscription quotas and adjudicates the 900 s simulator-RPC idle timeouts.
- ~~`prompt_context.py:164-182`~~ [corrected: `prompt_context.py:6-24` — the file is only 24 lines; `TASK_NOTES` :6-17, `task_context()` :20-24; `requires_arm_return` is true for every task except `make_kong`] gives each task a one-sentence "process" hint (`task_context`) and a `requires_arm_return` flag. The model therefore gets **task-specific hints**, but no geometry.

**RoboLab integration (`hybrid_rollout/robolab/`)**
- Hybrid only. Single Franka/Robotiq arm, 15 Hz, H15 π0.5-DROID proposals, two cameras.
- It is the "colleague baseline" from which the method was copied. `SOURCE.json` traces it to an earlier `rlinf_robolab` pipeline.
- **The RoboLab Direct ("pure Astra") controller is not in the release.**

## 5. Prompts (verbatim excerpts)

The gate, `skill/gate_prompt.md`, is described as the "unchanged colleague gate". ~~The prompt hash is enforced via `GATE_INSTRUCTION`~~ [corrected: the gate text is duplicated as the constant `GATE_INSTRUCTION` (`robodojo_server/gate_assessment.py:3-30`, byte-identical to `skill/gate_prompt.md` — checked) and its SHA-256 is only *recorded* in `run.json` (`client.py:193`); nothing compares hashes at runtime. Identity with the RoboLab gate is asserted only in the offline test `test_contract.py:26`]:

> At each chunk boundary, assess two separate questions:
> 1. What happened during the LAST executed chunk? Compare before/after RGB, measured robot state, executed gripper commands and same-episode history. A closed command alone proves neither grasp success nor failure. …
> 2. Given the current task phase, does the NEXT student chunk pursue an appropriate subgoal? Infer intent from its robot-only FK trajectory and gripper sequence … Realigning for a retry can be appropriate: allow student self-recovery.
> …An edit/eef takeover requires execution_status=failed or intent_status=misaligned. Uncertainty alone or an aesthetically imperfect pose is not a takeover reason. After recovery, hand back when the current state and student subgoal are suitable. Do not rewind.

From the Hybrid skill (`skill/SKILL.md`):

> Operate as the autonomous GPT + pi05 policy, not a JSON-only reviewer. … Three blocking services are additional tools, not your complete toolset.
>
> For container placement, clear the rim with both the object and fingers before moving laterally; align above the opening, lower and release, then withdraw upward before moving sideways.
>
> Correct only observed execution failure or misaligned next intent, never uncertainty alone. … No rollback, mid-episode reset, hidden planners, object truth, reward queries or hypothetical physics for planning.

The opening user turn (`run.py:241-248`):

> Act as the autonomous policy agent for this single simulation rollout. Use your normal file, image, shell/code and planning tools as useful, alongside the rollout service tools. … First call: {next_call JSON}

From the embodiment contract (`skill/context/eef_control.md`):

> EEF corrections use local damped least-squares IK, recomputed after every real control ACK. Task increments <=2 cm/0.1 rad and each joint increment <=0.05 rad. No cuRobo planning or teleportation. A requested target is not evidence of arrival/contact …

The Direct skill (`robodojo-gpt-only-rollout/SKILL.md`) is the same minus π0.5 and the gate:

> `eef` only: 1–5 steps toward explicit left/right poses, each within 5 cm and 0.35 rad … Direct joint commands are unavailable.

## 6. Results

### 6.1 RoboDojo, 10 tasks × 5 paired cases

Sources: `public_results/scores.csv` and `report_web/data.json`. Corrected-step percentages are recomputed from per-case rows [derived].

| Task (step limit) | Hybrid succ / Score | Direct succ / Score | π0.5 public SR% / Score | Hybrid LLM-authored steps |
|---|---|---|---|---|
| organize_table (1000) | 0/5 / 60.0 | 0/5 / 30.0 | 0.0 / 23.3 | 25.0% |
| classify_objects_by_language (1100) | 1/5 / 38.0 | 2/5 / 60.0 (n=4) | 0.0 / 0.6 | 13.7% |
| imitate_sorting_sequence (1600) | 2/5 / 53.0 | 0/5 / 0.0 | 0.0 / 1.6 | 16.3% |
| arrange_largest_number (1050) | 2/5 / 50.0 | 2/5 / 57.0 | 0.5 / 2.3 | 26.3% |
| pack_objects_into_box (1300) | 1/5 / 50.0 | 1/5 / 50.0 (n=4) | 2.4 / 18.4 | 12.5% |
| classify_objects (1100) | 3/5 / 71.0 | **5/5 / 100** | 12.7 / 24.7 | 7.6% |
| build_tower (1050) | 3/5 / 64.0 | 0/5 / 12.0 | 24.0 / 37.7 | 12.5% |
| make_kong (600) | 2/5 / 40.0 | 0/5 / 0.0 | 26.7 / 26.7 | 11.0% |
| fold_clothes (500) | 5/5 / 100 | 2/5 / 40.0 | 21.1 / 29.1 | 1.7% |
| put_bottles_into_dustbin (700) | 5/5 / 100 | 1/5 / 36.0 | 69.3 / 79.9 | 0.2% |
| **Overall** | **24/50 = 48%** / 62.60 | **13/50 = 26%** / 37.81 | 15.67 / 24.43 | 14.4% (6,174 / 42,750) |

- Hybrid correction steps break down as 6,001 `eef` steps and only 173 `edit` steps (HF `validation.json`). The relative-offset mode was almost never used.
- Best official model on the same reweighted subset: GalaxeaVLA G0.5 at 38.26 Score / 30.95% SR.
- **Statistics [derived].**
  - Wilson 95% CI: Hybrid [34.8, 61.5]%, Direct [15.9, 39.6]%.
  - Paired outcomes: 8 both succeed, 16 Hybrid-only, 5 Direct-only, 21 both fail. Exact McNemar p ≈ 0.027.
- The two missing Direct Scores are "simulator_rpc_idle_timeout_900s" episodes. They count as failures in the success rate and are excluded from the mean Score.

### 6.2 RoboLab, single-arm Franka (`report_web/robolab-scores.csv`; caveats in `report_web/robolab.json` / `data.json → reportData.robolab.notes`)

| Method | Success |
|---|---|
| GPT 6 Astra Direct | 49/50 (98%) |
| π0.5 + Astra | 46/50 (92%) |
| π0.5 (DROID, zero-shot) | 18/50 |
| Cosmos3-Nano-Policy | 18/50 |
| DreamZero | 17/50 |

~~The data file itself lists the caveats:~~ [corrected: the CSV has no caveats; they are listed in `report_web/robolab.json` and `data.json → reportData.robolab.notes`:]
- "Astra arms include retained historical results and explicitly authorized retries; initial states are not strictly paired".
- Two Direct BlocksInBin retries used a 500-decision budget instead of 180.
- The three baselines are historical June-2026 runs.

### 6.3 Usage (Table 3 of the report)

| | Hybrid | Direct |
|---|---|---|
| Control steps / action segments | 42,750 / 3,776 | 38,221 / 7,729 |
| Steps per segment [derived] | 11.3 | 4.95 |
| Mean simulated duration | 34.20 s | 30.58 s |
| Total tokens (incl. cache) | 624,762,828 | 1,132,343,772 |
| Cached / uncached input | 607.6M / 15.90M | 1,107.3M / 22.91M |
| Output (of which reasoning) | 1.305M (0.381M) | 2.087M (0.873M) |
| Per decision: input / uncached / output / reasoning [derived] | 165k / 4.2k / 346 / 101 | 146k / 3.0k / 270 / 113 |
| API-equivalent cost at $10 / $1 cached / $50 per M [derived] | ≈ $832 (≈ $16.6/ep, $34.7 per success) | ≈ $1,441 (≈ $28.8/ep, $111 per success) |

- The pricing used is from OpenAI's gpt-6-astra model page. The report gives no dollar figures.
- Authors' reply on issue #2: per episode, Direct uses 22.65M tokens over 154.6 decisions and Hybrid 12.50M over 75.5. "一个 episode 大概要用掉 20x 账号周额度的 1-2%" means one episode uses about 1–2% of a "20x" ChatGPT plan's weekly quota.
- Codex's context window was 258,400 tokens (`modelContextWindow`), and contexts are auto-compacted.
- **Reasoning is surprisingly short: about 100 reasoning tokens per decision even at `xhigh`.**

### 6.4 Wall-clock latency [derived]

This is not reported; issue #5, which asks for it, is unanswered. I measured the first-to-last agent-event span in 11 downloaded episodes:

- **Direct:** 17.7, 18.5, 31.0, 38.2 and 58.4 s per decision (89–318× real time).
- **Hybrid:** 27.2, 29.8, 32.2, 56.1, 66.4 and 106.4 s per decision (46–193× real time).
- Median across all 11: about 32 s per decision.
- π0.5 inference was about 0.43 s, after a ~~34 s~~ [corrected: 30.7–52.2 s first-call (JIT) time across the 6 sampled hybrid episodes; steady-state medians 0.415–0.454 s] JIT warm-up.
- Extrapolating to the mean decision counts gives roughly 0.6–2.5 h of wall time per episode. The sample is biased toward short tasks.
- The expanded paper adds one more number: its dense-locomotion run needed **39.86 s per model call**.

## 7. Failure analysis

**What the report itself shows (§5 and §6 of the report)**
- Repeated grasp failures use up the step budget.
- Collisions with container rims.
- Slips that happen inside a chunk are only seen at the next decision boundary.
- A Mahjong error that was not recognised.
- Direct produces "interesting" plans that fail in execution: sweeping bottles with the arm, grasping a board one-handed, single-arm instability.

**What I found in the released records**

1. **Interface friction.**
   - The model writes rounded quaternions such as `[0.707,0,0,0.707]` (norm 0.99985), which the 1e-4 check rejects as non-unit.
   - It exceeds the 5 cm bound.
   - It sends stale `request_id`s.
   - Each rejection costs a full model turn.
2. **Degenerate tool loop.** In `direct__classify_objects_by_language__standard__g0__l1` the model issued ~~**7,700 rejected `robodojo_act` calls**, all with a stale `observation_path`~~ [corrected: 7,750 rejected `robodojo_act` calls out of 7,808 service calls; 7,700 of them failed because `observation_path` was not the expected `observations/056/…` — the paths sent were *advanced, non-existent* ones (`observations/057` … `observations/2626`), not stale ones, and 5,135 carried the placeholder `request_id: "x"`. The whole episode used only 16.43M tokens, so these cannot have been 7,750 separate model requests at ~90k context each. Reasoning summaries ("Automating pose-based actions", "Waiting for script response", "Monitoring script execution") point to a model-authored automated loop; the exact Codex mechanism is UNVERIFIED]. The run ended only when the 900 s idle timeout fired. The cause is `recoverable_input_error_limit=None`.
3. **Tiny Direct steps.** Each Direct decision commits at most 5 ticks (0.2 s), with each tick capped at 2 cm by IK. An imitation episode spent 11 decisions and 7 minutes "continuing to approach" the first object. Then the native check terminated the episode as a failure at step 54. That task's demonstration phase seems to require waiting; the exact termination trigger is UNVERIFIED.
4. **The LLM overriding a correct "wait".** In `hybrid__imitate_sorting_sequence__…l4`, π0.5 held still for about 105 ticks. The LLM judged its intent "misaligned" and forced an approach. The episode terminated as a native failure at step 165. The causal link is plausible but UNVERIFIED.

## 8. Assessment

### Strengths

- A clean and well-engineered "LLM supervises a VLA" contract:
  - A fresh VLA inference before every commit.
  - The VLA's joint chunk rendered as a Cartesian FK trajectory, so the LLM can "read" its intent.
  - A schema-validated gate with separate *outcome* and *intent* questions and a rule that uncertainty alone does not justify takeover.
  - Mutually exclusive accept-a-prefix or replace-with-a-bounded-correction branches.
  - A safety envelope enforced outside the LLM, with an FK self-check.
- Unusually strong provenance: frozen panels and hashes, paired seeds, one-attempt-per-case selection, adjudication files, and a full per-tick dataset that includes the exact previews the model saw and its public reasoning summaries.
- A candid write-up: the "pausable video game" caveat, latency named as the open problem, and confounds acknowledged in §7 of the report.
- Shows that a frontier agent harness (Codex with shell and image tools plus a notes file) can be turned into a robot policy with very little code: the core loop is about 450 lines in `run.py`.

### Weaknesses and threats to validity

- **The π0.5 baseline is not a same-seed rerun.** It is the public leaderboard reweighted to the subset.
  - The hybrid's large gains on fold_clothes (100% vs 21%) and put_bottles (100% vs 69%) come with **only 1.7% and 0.2% LLM-authored steps**. There, the hybrid is essentially π0.5 executing ≤15 of 50 steps and re-inferring.
  - The current XPolicyLab `policy/Pi_05/deploy.py` executes the **entire returned chunk** before re-inferring. If the leaderboard used that protocol (UNVERIFIED), part of the "hybrid" gain is just receding-horizon replanning.
  - Missing ablations: π0.5 with a fixed 15-step prefix, and the LLM reviewing without correcting.
- Tasks were chosen from low-π0.5-success strata (6/2/1/1), with n = 5 per task. The CIs are wide, and RoboLab mixes in retries.
- Direct vs Hybrid confounds the action prior, the interface and chunk length. The authors say so. The Direct interface (≤5 ticks, ≤5 cm, 2 cm/tick) is restrictive, so "LLM-only control is weak" here partly measures this particular interface.
- Simulation only, with physics paused. Wall time is about 30 s per decision. Nothing here speaks to real-time control, dynamics or hardware.
- Model text includes task-specific hints (`task_context`, container-rim advice). The gate is self-reported and only checked for consistency.

### Novel vs repackaged

- **Repackaged:**
  - System-2 supervisor over a System-1 VLA.
  - Intervention-based correction in the DAgger style. The stated motivation for keeping π0.5's prompt unchanged is to collect DAgger data later (issue #4).
  - DLS IK and bounded EEF teleop-like corrections.
- **Genuinely useful and new in practice:**
  - The concrete *proposal → FK preview → gated prefix/correction* protocol.
  - Running it inside a production coding-agent harness.
  - The first fairly large, paired, fully released trace set: 100 episodes, about 81k control steps, 11.5k decisions.
- **Maturity:** a research evaluation harness. Robust operational code (quota and idle handling), but tied to Codex, Isaac and their own cluster. No reusable library API.

### What Ilia's Opus-backbone harness should borrow

1. **Proposal review over a learned head.**
   - Let the light action head (or a VLA) emit chunks.
   - Give Opus the FK-rendered EEF path, but **compress it**: here it was ~~37–42 KB~~ [corrected: 34.3–40.9 KB in the released packets, of which the FK trajectory is ~33 KB] of full-precision JSON per decision (about 10k tokens; UNVERIFIED estimate). Subsample to around 10 waypoints at mm/mrad precision, or draw it onto the image.
   - Let Opus choose the committed prefix length.
2. **Strict tool schemas with a `request_id` / `next_call` handshake** and pre-execution validation that returns errors without executing. But:
   - Normalise quaternions server-side, or accept rotation-vector deltas instead of rejecting them.
   - **Cap consecutive rejections** (for example, 5 → hold pose and escalate).
3. **A hard envelope outside the model:** per-tick task and joint caps, soft limits, FK-vs-measured self-check, no rollback.
4. **Prompt caching is the cost lever.** 97–98% of input was cached reads here; mirror that with Anthropic cache breakpoints on the stable prefix.
5. **Persistent scratch memory plus a code tool,** so the model computes geometry instead of eyeballing numbers.
6. **Native-termination discipline:** no model-initiated stop on benchmark episodes, and keep acting until the environment ends the episode.
7. **Log formats worth copying:** `decisions.jsonl`, `service_calls.jsonl`, the previews actually sent, and per-tick NPZ.

### What to avoid

- **Synchronous, paused-world control.** For real hardware the VLA or action head must run continuously while the LLM supervises asynchronously, using subgoals, waypoint edits and "hold" primitives. Budget about 20–60 s per Opus deliberation unless effort and context are cut hard.
- **Sending full episode history every turn.** That is about 150k tokens per decision here. Prefer state files, notes and summaries.
- **Takeovers driven by the model's self-assessed "misalignment" alone.** Add cheap checks (motion, contact, progress detectors) and allow "wait".
- **Tiny-step Cartesian interfaces for LLM-only control.** If the LLM acts directly, give it waypoint or trajectory primitives with a motion planner.
- **Leaving the VLA blind to the LLM.** Test passing the LLM's subgoal to the VLA as language. The authors call this an untested, promising direction (issue #4). RPent (RLinf) reportedly does something similar: agentic planner over frozen VLA primitives; listed in Awesome-Astra; UNVERIFIED.

## 9. Authorship and affiliation (public evidence only)

- **Authors as stated.** `README.md` and `report_site/app/src/content/report/authors.json` name Jiayi Su and Yixin Zheng (equal contribution), Mi Yan, and Li Yi, Zhizheng Zhang and He Wang (corresponding authors).
- **Pseudonyms.** The `LICENSE` copyright reads "Yu-Mool Shu and Lipxin Zheng". Papers-with-Code and an X post use those names. They look like pseudonymised forms of the author names.
- **Galbot evidence in the repo.**
  - The report shows the Galbot logo (`content/assets/galbot-wordmark-source.md`).
  - The default auth profile is `galbot` (`codex_backend/profiles.py:20,47`).
  - `subscription_migration.py` refers to "Galbot container[s]".
  - The cluster job owner is `sujiayi` (`paired_evaluation.py`).
- **Author identities.**
  - The co-first author's homepage is `steveouo.github.io`, and the issue #4 answer was posted by GitHub user `SteveOUO`.
  - The HF owner `YuMoool` is named "Jiayi Su".
- **Third-party attribution.**
  - arXiv 2609.38537 is signed "Galbot Team". Its contributor list names Su, Yan, Zheng and Xiaoqian Cheng for Manipulation, and Wang, Yi and Zhang for Supervision.
  - BigGo Finance (2026-09-16) attributes the report to the Galaxy General (Galbot) team. Galaxy General is Galbot's Chinese name; He Wang is described there as founder and CTO.
  - Awesome-Astra-Embodied-AI lists it as "from Galbot (银河通用)".
- **Infrastructure.** The cluster adapter `acp.py` targets SenseCore. No further affiliation inference is made.

## 10. Context and related work

- **~~RoboDojo team~~ [corrected: Wenbo Zhang et al., 12 authors with RoboProbe and RoboDojo affiliations; it cites the benchmark itself as "RoboDojo Team, 2026"], arXiv 2609.24170 (2026-09-21).** Astra alone on all 42 tasks × 50 episodes reached 22.48% SR / 28.97 Score, top of the public board. GPT-5.5 reached 0.88% and DeepSeek-Flash 1.92%. Real-robot tests were halted for safety. This is the full-benchmark counterpart to the Direct mode here.
- **Galbot expanded paper (2609.38537).** Adds RoboCasa365: Hybrid 38.7%, but Hybrid made *more* model requests than Direct (median 108 vs 85). Also navigation (RxR 92%, HM3D 82%), HumanoidBench (13 of 30 tasks beat RL baselines) and locomotion (0/5 courses).
- **Commentary.**
  - Understanding Robots (2026-10-01) cites the hybrid result as evidence that LLMs and VLAs are complementary. It also notes that Astra "pause[s] for seconds at a time".
  - An X post by @ClaraChengGo reported the Scores 24 / 37 / 62 as if they were success rates.
  - No HN or Reddit threads were found.
- **Cross-links to other user sources.** The report cites Robocurve (YAM 19/20 bowl, 2/20 puzzle insertion), innate-os PR #817, and cheng-haha's GPT-Policy-Eval as community explorations.

## Sources

- https://anonymous-report-421.github.io/public-website/?lang=en&view=1 (also `?lang=zh`, `view=0,2,3`)
- https://github.com/anonymous-report-421/GPT-as-Policy (issues #1, #2, #4, #5, #6)
- https://github.com/anonymous-report-421/public-website (release `robolab-gallery-20260914`)
- https://huggingface.co/datasets/YuMoool/astra-robodojo-rollouts (README, SCHEMA.md, REPRODUCE.md, manifest.json, validation.json, 11 core bundles)
- https://arxiv.org/abs/2609.38537 and https://arxiv.org/html/2609.38537v1
- https://arxiv.org/abs/2609.24170
- https://developers.openai.com/api/docs/models/gpt-6-astra
- https://github.com/XPolicyLab/XPolicyLab/blob/main/policy/Pi_05/deploy.py
- https://github.com/RoboDojo-Benchmark/RoboDojo
- https://github.com/zjwzcx/Awesome-Astra-Embodied-AI
- https://finance.biggo.com/news/2745ebeb-be73-4541-a460-f5d4e6b0b262
- https://www.understandingrobots.org/p/openais-astra-model-is-shockingly
- https://x.com/ClaraChengGo/status/2099422821911060669 (unreachable, HTTP 402; text seen only via search snippet)
- https://paperswithcode.co/paper/114186
- https://thedailycommit.in/story/2026-09-17/05-github-anonymous-report-421-gpt-as-policy (search result only, not read)

## Verification (fact-check pass)

*Fact-check run on 2026-10-01 against primary sources only. Sources used: the repo clone `repos/GPT-as-Policy` (commit `8f3d362`), `repos/public-website`, the HF dataset clone `repos/astra-robodojo-rollouts`, and all 11 downloaded core bundles (unzipped and re-analysed). I also used the rendered site via agent-browser (`view=0/1/2`), GitHub API (stars, issues, release assets, commits), arXiv API/HTML for 2609.38537 and 2609.24170, the OpenAI model page, BigGo, Understanding Robots, the Papers-with-Code page, and local clones of `RoboDojo`, `XPolicyLab`, `Awesome-Astra-Embodied-AI` and `RLinf_RPent`. Line numbers refer to the files at the commits named.*

### Confirmed (seen in a primary source)

**Headline results** (`report_web/data.json`, `public_results/scores.csv`, HF `validation.json`, report text):
- Hybrid: 24/50 successes and mean Score 62.60.
- Direct: 13/50 successes. Its mean Score is 37.8125 over 48 scored episodes.
- π0.5 reweighted: 15.67% SR and 24.43 Score.
- GalaxeaVLA G0.5: 38.26 / 30.95%.
- Corrected steps: 6,174 of 42,750 (14.4%), split into `eef` 6,001 and `edit` 173.
- All ten per-task rows, including the per-task corrected-step percentages (1252/5000 … 5/2262), recomputed and matched.
- Step limits 500–1600 match.
- Paired counts: 8 / 16 / 5 / 21.
- Exact two-sided McNemar test (16 vs 5): p = 0.0266.
- Wilson 95% CIs: [34.8, 61.5]% and [15.9, 39.6]%.

**Tokens** (`data.json → reportData.totals`):
- Hybrid total 624,762,828: input 623,457,803, of which cached 607,555,840. Output 1,305,025, of which reasoning 380,541.
- Direct total 1,132,343,772: input 1,130,257,208, of which cached 1,107,349,760. Output 2,086,564, of which reasoning 873,464.
- Segments: 3,776 (Hybrid) and 7,729 (Direct).
- All per-decision figures were recomputed and match: 165k / 4.2k / 346 / 101 and 146k / 3.0k / 270 / 113.
- Simulated durations match: 34.20 s and 30.58 s.
- List-price arithmetic matches: ≈ $832 and ≈ $1,441. The OpenAI page confirms $10 input, $1 cached input and $50 output per 1M tokens.
- `modelContextWindow` = 258,400 (`token_usage.json`).

**Issues.**
- Issue #2 reply: 22.65M tokens / 154.6 decisions (Direct) and 12.50M / 75.5 (Hybrid). The Chinese "20x 账号周额度的 1-2%" line is quoted correctly.
- Issue #4 was answered by `SteveOUO`. The DAgger motivation and "not tested" are confirmed.
- Issue #5 has 0 comments.

**Latency [derived].** Recomputed the agent-event spans for all 11 bundles. Values match exactly:
- Direct: 17.7, 18.5, 31.0, 38.2 and 58.4 s per decision (89–318× real time).
- Hybrid: 27.2, 29.8, 32.2, 56.1, 66.4 and 106.4 s per decision (46–193× real time).
- Median: 32.2 s.

**Code claims** (file:line verified):
- `profiles.py:19-37` (15 `codex_*` ChatGPT profiles; `galbot` default at `:47`).
- `config.toml` (`model_reasoning_summary="auto"`, `fast_mode=false`, analytics off).
- `run.py`:
  - `agent_config` :76-90, `tool_specs` :93-115, `content_items` :118-130.
  - `__init__` :133-202, with the model/effort abort at :183-185.
  - `run` :240-395: opening turn :241-248, tool dispatch :282-322, ≤2 continuations :365-370, token logging :274-281.
  - `rejected_input` :31-66.
- `network_recovery.py:4` (`(10,)*20`).
- `schema.py`; `client.py`:
  - `next_call` :98-113, `_check` :115-125.
  - `infer` :204-237, `execute` :239-317, stop refusal :246-254.
- `gpt_only_client.py:99-148`.
- `kinematics.py`: :42-65, :49 (2 cm / 0.1 rad), :58 (±0.05 rad), `check()` :82-94 (2 mm / 0.01 rad).
- `validation.py:30-61` (5 cm Euclidean, 0.35 rad, ‖q‖ within 1e-4, bool gripper; CJK check :17-27).
- `action_edit_kinematics.py:20-45` ((i+1)/n ramp); `gate_assessment.py:33-64`.
- `session.py`: `chunk_step` :140-169, score :171-196.
- `pi05_server/client.py:36-58` (prompt = `observation['instruction']`; gripper clipped to [0,1]).
- `server.py:87-89` (`need_planner=False`, "never object-aware cuRobo"); `rpc.py:16` (`settimeout(900)`).

**Prompt quotations.** All quotations in §5 match the files verbatim. The prompts shipped in the repo are byte-identical to `prompts/` inside the released episode bundles: `SKILL.md`, `eef_control.md` and `teacher_context.md` were diffed.

**Provenance and repos.**
- Codex CLI `codex-cli 0.153.4`.
- `teacher_model_provider: "openai"` and `max_decisions: 0` in all 11 sampled episodes.
- Panel id `robodojo_v3_pool15_20260912_01_fixtures`.
- Runs on 2026-09-12 11:15 → 2026-09-13 06:15 UTC (sample).
- `generatedAt` 2026-09-13T13:15:02Z.
- GitHub: 565 stars; old repo name returns HTTP 301.
- Release `robolab-gallery-20260914` has 150 MP4s plus 2 JSON manifests.
- The site has 100 rollout videos and 43 `clip__*` files plus 1 featured historical clip.
- Rendered text for `view=0/1/2` is identical (MD5 `6ded26cf…`).

**Dataset.**
- HF dataset is CC-BY-4.0. Core bundles total 7.86 GB.
- HF user `YuMoool` has the full name "Jiayi Su".
- Previews are 480×360 PNG; observations are uint8[480,640,3].

**Authorship and affiliation.**
- `authors.json`, the LICENSE pseudonyms "Yu-Mool Shu and Lipxin Zheng", and the Papers-with-Code page (same pseudonyms, dated 2026-09-13) all confirmed.
- Galbot logo rendered on the site (alt="Galbot").
- `subscription_migration.py:3` "Drain the last Galbot container first".
- `sujiayi` job owner; SenseCore adapter (README:55).
- Awesome-Astra README:421 "from Galbot (银河通用)".
- BigGo article: 2026-09-16, "Galaxy General team", Wang He "founder and CTO".
- Understanding Robots: 2026-10-01, "pause for seconds at a time".

**arXiv 2609.38537** (2026-09-29):
- Byline "Galbot Team" plus 33 names; `\code GPT-as-Policy`.
- §3 "Gripper Manipulation" contains §3.2 RoboDojo and §3.3 RoboLab.
- Manipulation contributors: Cheng, Su, Yan, Zheng. Supervision: Wang, Yi, Zhang.
- RoboCasa365 Hybrid 38.7%, with median 108 vs 85 requests (Hybrid vs Direct).
- RxR 92%, HM3D 82%, 13/30 HumanoidBench tasks, 0/5 locomotion attempts, 39.86 s per call.

**arXiv 2609.24170:**
- 22.48% SR / 28.97 Score over 2,100 trials; GPT-5.5 0.88%; DeepSeek-Flash 1.92%.
- Real-robot tests were "halted after the model repeatedly issued physically unreasonable or unsafe actions, including incidents that damaged hardware".

**XPolicyLab** `policy/Pi_05/deploy.py:8-13` executes the full returned chunk; `model.py` does no truncation.

### Corrections (also fixed inline)

1. **Settings lines.** `settings.py:6-10` → `settings.py:8-12`: MODEL :8, EFFORT :9, WIRE_API :11, IMAGE_MAX_EDGE :12.
2. **Prompt-context lines.** `prompt_context.py:164-182` → `prompt_context.py:6-24`. The file is 24 lines long, `CONTEXT_VERSION='v3'` is at :3, and `requires_arm_return` is false only for `make_kong`.
3. **"Prompt hash is enforced via GATE_INSTRUCTION"** → the hash is recorded, not enforced:
   - `GATE_INSTRUCTION` is a byte-identical copy of `gate_prompt.md` (`gate_assessment.py:3-30`, verified programmatically).
   - Its SHA-256 is written to `run.json` (`client.py:193`).
   - The text goes into the on-disk `request_NNN.json` but is stripped before the packet is sent to the model (`client.py:233`).
   - The only identity check is the offline test `test_contract.py:26`.
4. **Degenerate loop: "7,700 rejected calls, all with a stale `observation_path`"** → the paths were advanced, not stale. `service_calls.jsonl` shows:
   - 7,808 calls in total, 7,750 of them rejected `robodojo_act` calls.
   - 7,700 rejections came from `observation_path` mismatches. Another 28 had a stale `request_id`, 9 exceeded 5 cm, and the remainder had wrong output/observation paths or a non-unit quaternion.
   - The paths sent were advanced, non-existent ones (`observations/057` up to `observations/2626`), not stale ones.
   - 5,135 calls used `request_id: "x"`.
   - Total episode usage was 16,434,295 tokens, which rules out 7,750 model requests. Reasoning summaries suggest an automated script loop; the mechanism is UNVERIFIED.
   - Last simulator ACK at 04:33:29 UTC and simulator close at 04:48:30 UTC: `observed_idle_seconds` 900.69.
   - Note: the 58.4 s/decision and 318× figures for this episode include this ~15-minute runaway period.
5. **π0.5 warm-up "34 s"** → 30.7–52.2 s first-call time across the 6 sampled hybrid episodes. Steady state was 0.415–0.454 s (median per episode).
6. **Packet contents and size.**
   - The packet does not include the raw "50×14 chunk". `pi05_infer` returns the 50-row FK trajectory (per arm: position, wxyz quaternion, `gripper_closed`, `gripper_opening`, `rpy_deg`; about 33 KB), `action_diagnostics`, `fk_check`, and the previous observation and result.
   - The joint chunk stays in `actions.npz` (`pi05_server/client.py:1`: "Arrays stay in files, not in model tool arguments").
   - Measured size is 34.3–40.9 KB of compact JSON in the released, path-redacted packets, not 37–42 KB.
7. **`not_started`** is required at step 0 and only there; the check is biconditional (`gate_assessment.py:56-57`).
8. **RoboLab caveats** are not in `robolab-scores.csv`. They are in `report_web/robolab.json` and `data.json → reportData.robolab.notes`.
9. **arXiv 2609.24170** is not authored by "the RoboDojo team". It is Wenbo Zhang, Kaixuan Wang, … Tianxing Chen (12 authors), with mixed RoboProbe, RoboDojo, HKU, Tsinghua and UC Berkeley affiliations. It is also not an exact "Direct-mode counterpart": its interface (`move_eef` over world-frame grasp-point state with "simple non-learned post-processing") differs, and DeepSeek-Flash was run at only 10 episodes per task.

### Unverifiable or unverified

- **"Contexts are auto-compacted."** There is no compaction event or setting in the RoboDojo code or bundles. Auto-compaction is plausible as the Codex default but UNVERIFIED. The expanded paper mentions compaction only for RoboCasa (App. I.4).
- **"~10k tokens"** for the proposal packet. This is an estimate; no tokenizer count is available.
- **The @ClaraChengGo X post** (HTTP 402; snippet only) and "no HN/Reddit threads" could not be verified.
- **Whether the RoboDojo leaderboard π0.5 runs used XPolicyLab's full-chunk protocol** remains UNVERIFIED.
- **"Built with Codex's 'Data app' plugin."** `report_site/app/AGENTS.md` is a "Data App Authoring Guide" referencing a "plugin-owned reference snapshot" and a "build-report skill", and `.openai/hosting.json` exists. The exact plugin name is not stated.
- **Episode provider.** "All 100 episodes ran on the subscription pool" was checked only for the 11 sampled bundles (`provider: openai`).
- **Upgraded from UNVERIFIED to supported:**
  - The RPent claim: Awesome-Astra README:253-259 describes "GPT-6 Astra acts as the planner in RPent … compose frozen VLA calls", and `repos/RLinf_RPent` exists.
  - The causal link in failures #3 and #4 is now supported by a native rule; see missed detail 2.

### Missed details (added)

1. **No decision or token cap on RoboDojo, and failures are mostly step-budget timeouts.**
   - Benchmark campaigns set `max_decisions=0` (`campaign.py:174,259`; `run.json` in all sampled bundles) and `CODEX_MAX_TOTAL_TOKENS=0`. Combined with `require_native_termination=true`, an episode ends only at native success, native failure or the step limit. The code default of 180 decisions (`run.py:410`) was *not* used.
   - Of the 26 Hybrid failures, 25 ran to the native step limit. Of the 37 Direct failures, 29 hit the limit, 6 ended early through native failure, and 2 were idle-timeout adjudications. 5 of the 6 early Direct failures, and the only early Hybrid failure, are `imitate_sorting_sequence`.
   - RoboLab, by contrast, used a 180-decision budget (500 for two retries).
2. **Native early-failure rule in `imitate_sorting_sequence`.** Defined in `RoboDojo/task/RoboDojo/tasks/imitate_sorting_sequence.py`, `run_reward()` `reward_manager.query(..., 0)`; unchanged between the recorded revision `ee67a14` and HEAD `726e9aa`.
   - The episode is marked failed (`reward_manager.py:362-374`, `aim_num=0`) as soon as the left or right arm is >0.3 m / >30° from its origin while the demonstrator `support_arm0` is still away from its origin.
   - This is a concrete mechanism for the LLM's "misaligned → approach" override in `hybrid…imitate…l4`, where π0.5 held for 105 ticks, the `eef` correction came at ticks 105–110, and the episode failed at 165. It also explains the 54-step Direct failure.
   - Exact support-arm timing per episode was not checked.
3. **Pairing is not bitwise.** The 50 cases were selected from a 60-case candidate panel (`REPRODUCE.md`). Settings, robot state and text matched for 50/50 pairs, but `rgb_bitwise_equal_count = 0` (`pairing.json`): none of the initial RGB triplets were identical, e.g. render MAE of 0.25–0.36/255 per camera in the first pair inspected. `REPRODUCE.md` also warns that "Recovery code differs across some selected attempts".
4. **Codex harness details not in the note** (`run.py:76-90, 176-182`):
   - Thread parameters: `approvalsReviewer='auto_review'`, `ephemeral=False`, `permissions='rollout_agent'`.
   - The `rollout_agent` permission profile extends `:workspace`, with read access to the audit parent and write access only to `agent/`.
   - `shell_environment_policy.inherit='all'`.
   - `config.toml` also sets `codex_hooks=true` and `hooks=true`.
   - There is no tool allowlist. `NATIVE_WORK_ITEMS` includes `webSearch`, `mcpToolCall` and `collabAgentToolCall` (`run.py:27-28`); whether web search was actually enabled is UNVERIFIED.
   - `workspace.py:130-186` writes `agent/AGENTS.md` (forbidding a second simulator connection or reset via shell) and copies the skill to `agent/.agents/skills/robodojo-hybrid-rollout/`. It also creates `scratch/`, `workspace.json` (paths to `history.json`, observations, proposals, `execution_NNN.npz`, `edit_NNN.json`) and a seeded `NOTES.md`.
5. **Two separate 900 s timeouts.**
   - `CodexPolicy` raises if Codex makes "no completed service or native tool call within the configured timeout" (900 s, `run.py:134, 256-262`). The deadline resets on every tool call and every native work item.
   - Separately, the simulator RPC socket has a 900 s timeout (`rpc.py:16`). This one produced the two adjudicated Direct failures, because rejected calls never touch the simulator.
6. **Opening-turn consent and language enforcement.**
   - The first user turn includes: "The user authorizes sending this episode's three RGB images, proprio, robot-only FK, task text and same-episode history to OpenAI Codex for online decisions".
   - English-only prose is enforced by rejecting CJK characters in `reason`/`assessment` (`validation.py:12-27`).
7. **Direct-mode specifics.**
   - The Direct schema has no `assessment` and no `stop` (`schema.py:45-50`; `gpt_only_client.py:14-30`), so Direct decisions carry no structured outcome/intent self-report.
   - Each `eef` decision repeats one fixed absolute target for 1–5 ticks (`gpt_only_client.py:118-127`; Hybrid `client.py:61`). Each tick is IK-capped at 2 cm / 0.1 rad, so a 5 cm target needs at least 3 ticks even nominally.
   - The authors' RoboLab context documents tracking shortfall: "run06 decision4 requested a 4.85 cm displacement but moved only 2.58 cm in five controls" (`robolab/skill/context/eef_control.md:64-66`).
8. **π0.5 checkpoint identity** (`SOURCE.json`, `REPRODUCE.md`):
   - Config `pi05_base_aloha_full_sim_arx-x5_seed_0`, normalizer `arx_x5_sim`, checkpoint `RoboDojo-sim-arx_x5-joint-0/59999`, sha `d15fb8bd…`.
   - RoboDojo commit `ee67a146…`; OpenPI/XPolicyLab checkout `432f82b1…`.
   - The X5 contract warns against reusing DROID's +0.1311 m pad offset (`gripper_bias=0.145`, `skill/context/eef_control.md:42-44`).
9. **Per-request context is large.** "Decision" in all token tables means an executed segment, not a model request (authors' footnote in issue #2). In the 11 bundles, the last model request's input was 64–222k tokens, close to the 258.4k Codex window. Per-decision input ranged 112–410k, and reasoning tokens per decision ranged 13–347.
10. **API facts vs. this run** (OpenAI model page):
    - `gpt-6-astra` has a 1,050,000-token context (922k max input) and 128k max output.
    - Effort levels are `low|medium|high|xhigh|max`, so `xhigh` is *not* the highest.
    - Cache writes cost $12.5/M, but `cacheWriteInputTokens` = 0 in all records.
    - Prompts over 272K tokens are priced at 2× input and 1.5× output. This was not triggered, because Codex capped the window at 258.4k.
11. **Report claims the note omitted.**
    - Hybrid uses "44.8% fewer recorded tokens" than Direct, and Direct executes "approximately 2.05 times as many action segments".
    - The report was authored in Chinese first (`data.json reportData.language = "zh-CN"`).
    - `authors.json` lists `"affiliations": []`, so no affiliation is stated anywhere in the report.
    - Jiayi Su's homepage is `shuyumo2003.github.io`, consistent with the "Yu-Mool Shu" pseudonym and the HF handle `YuMoool`.
12. **The expanded paper tests the "LLM subgoal → VLA" idea.** The note says this is untested (§8 "What to avoid").
    - In the RoboCasa365 Hybrid of 2609.38537, Astra "can accept a 20-step prefix, construct actions, or rewrite the policy instruction as a subgoal". 66/75 episodes use rewritten instructions, which "condition 72.6% of proposals", and 44.8% of executed steps are Astra-authored.
    - Results: π0.5 17/75 (22.7%), Direct 25/75 (33.3%), Hybrid 29/75 (38.7%). The exact McNemar test gives p = 0.5235, i.e. no significant Hybrid > Direct gain there.
    - Hybrid beats Direct on seen tasks but trails it on unseen compositions (36% vs 56%).
    - Issue #4's "not tested" applies only to the RoboDojo study.
13. **Commit dates and issue timing.**
    - The single commit has author date 2026-09-15 02:21 UTC but committer date 2026-09-16 01:42 UTC, consistent with a force-pushed squash.
    - Issues #1 and #2 (2026-09-14) predate the visible commit.
    - In issue #1 the authors state that "GPT-6's context includes all previous observations" (full history h_t). That is how `imitate_sorting_sequence` is solvable from current views.
14. **Observation archive size.** The `observations.zip` files total about 126.75 GB, against 7.86 GB of core bundles (`manifest.json`).
15. **Failure clips.** The report's failure section (§5) shows four clips: container-rim clearance, an in-segment slip, repeated complex-grasp failure that consumes steps, and an unrecognised Mahjong error. In addition, §6.2 shows a Direct "proprioception not used effectively, control repeatedly blocked" clip (`video-selection.json`).
