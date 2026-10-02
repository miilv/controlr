# RobotKitAI/piper-astra-jev: deep dive

Repo: https://github.com/RobotKitAI/piper-astra-jev. There is one commit, `10d671e`, dated 2026-09-19 10:05 +0200 and authored by Michal Kubenka. It was pushed 2026-09-19 08:36 UTC under Apache-2.0. At the time of reading it had 11 stars and 1 fork (an unchanged mirror), and no issues or PRs. The clone is at `research/repos/piper-astra-jev`. The two released recordings are unzipped at `research/repos/piper-astra-jev-recordings/`.

## 1. TL;DR

- **What it is.** Eight single-trial demo runs on a real AgileX PiPER arm. A large model acts as a **skill selector**: it picks the name of the next hand-written skill from a menu. It never emits poses, joints or code. Under every choice sit a code "Governor" that enforces preconditions and limits, and a motion layer built on MuJoCo IK.
- **Astra.** OpenAI **GPT-6 Astra** (`gpt-6-astra`), called through the Responses API with `reasoning.effort="high"` and a strict JSON-schema output whose `skill` field is an enum.
- **Jev.** **TypeSafe AI's Jev**, a text-only "System One" *decision model*. It is not an LLM, a VLM or an action model. It takes a JSON state and a typed `choice` question and returns `{choice, confidence, probabilities}`. The repo calls `~typesafe/jev-latest` (which resolves to `typesafe/jev-1.13`) on OpenRouter's alpha endpoint `POST https://openrouter.ai/api/alpha/decisions`.
- **DINO.** **Grounding DINO tiny** (`IDEA-Research/grounding-dino-tiny`), not DINOv2/v3. It returns open-vocabulary boxes.
- **SAM 3.** Meta's **SAM 3** (`facebook/sam3`, a gated HF model, arXiv 2511.16719). It returns text-prompted instance masks.
- Both detectors run locally on an RTX 4090. Their output is fused with RealSense depth and turned into a boolean/metric JSON state.
- **The main finding is about where time goes, not about how smart the decider is.** In the author's thread:
  - Astra looking at images: cube-into-tray took **1 min 11 s**.
  - Jev with local perception: **27 s**.
  - Recorded run 3 (Jev+SAM3): 9 decisions, confidences 0.90–0.99, 0 refusals, a 16.6 s decision loop, cube in the tray.
- **Where the real engineering is.** It is not in the LLM part. It is in the grasp and geometry layer:
  - mask+depth grasp search;
  - CAD-silhouette pose fitting for chrome parts that defeat depth;
  - centre-of-mass-aware grasps;
  - relative visual alignment to 0.6 mm for a peg-over-spanner insertion with about 2 mm clearance per side.
- **The weakness in one sentence.** The decision is almost fully determined by hand-written booleans and preconditions. The decider adds little over a finite-state machine (FSM), task completion is self-reported rather than visually verified, and every result is n=1.

## 2. Provenance and org context

- **RobotKitAI.** GitHub org created 2026-05-13. Blog link: robotkit.dev. RobotKit is a platform for building, deploying and managing robots (MCP integration, visual agent canvas, VLA training), with the footer "Powered by Acheron Labs Inc." Acheron's footer reads "Wien - Prague - Bratislava". The object `meta.json` files link Hornbach **.sk** products, which fits a Slovak lab. Earlier RobotKit posts show TRON 1 plus a "Piper arm kit" for sanitary cleaning (Mar 2026) and a DOBOT Atom W integration (Aug 2026).
- **Author.** Michal Kubenka (@MichalKubenka, bio "Co-founder of @RobotkitAI"; GitHub `mkubenka`). He is a general software engineer: Ansible roles, a Bitbucket CLI, ~~a Camoufox MCP server~~ [corrected: `mkubenka/gomoufox` (Go library/CLI/MCP server for Camoufox) is a *fork* (`fork=true` in the GitHub API), not his own project], a fork of GR00T-WholeBodyControl. Nothing suggests a robotics research group.
- **Timeline.**

| Date (2026) | Event |
|---|---|
| Sep 3–4 | GPT-6 Astra released |
| Sep 15 | Jev launched |
| Sep 18, 00:01 UTC | Jev listed on OpenRouter |
| Sep 19, 05:26 UTC | @RobotkitAI posts "Weekend in our the labs starts: Jev vs. GPT-6 Astra" |
| Sep 19, 00:04–07:37 local | Runs recorded (folder timestamps in `docs/figures/make_figures.py:41`). Run order: 3, 5, 4, 7, 6, 8; runs 1–2 unknown |
| Sep 19, 08:36 UTC | Repo pushed |
| Sep 19, 08:38 UTC | Launch thread posted (x.com/MichalKubenka/status/2101229393079111800; 1.6k views, 5 likes) |

  ~~The only third-party reaction was a reply from AgileX ("impressive results, glad to see PiPER here").~~ [corrected: AgileX's reply (@AgilexRobotics, 2026-09-21 06:37 UTC, "impressive results, glad to see PiPER here") was not the only one: @SaroyaShehryar replied twice on 2026-09-19 (19:59 UTC "the link is broken?" and 20:03 UTC "I'd love to talk more"), and the root post has 1 quote and 1 repost (api.fxtwitter.com/2/conversation/2101229393079111800 and /2101231445876355113).] I found no HN or Reddit discussion of the repo.
- **README note.** It says the skill layer came "from simulation experiments", including bolt-hole approach skills. Leftovers are `APPROACH_SKILLS` (`piper_llm/skills.py:75-83`), `perception.cluster` and `Frame.to_base(ring=…)`, none used by the eight runs.

## 3. Exactly what the models are

| Component | Identity | Interface used in repo | Verified facts |
|---|---|---|---|
| **Astra** | OpenAI GPT-6 Astra, flagship reasoning model | `POST https://api.openai.com/v1/responses`. Body: `{"model":"gpt-6-astra","reasoning":{"effort":"high"},"input":[{"role":"user","content":[input_text…, input_image…]}],"text":{"format":{"type":"json_schema","name":"decision","schema":…,"strict":true}}}`. Timeout 600 s, no temperature or max tokens (`piper_llm/deciders.py:83-115`) | OpenAI model page: text+image in, 1.05M context, 128k output, effort `low/medium/high/xhigh/max`, **$10 / $50 per M tokens** (cached $1) |
| **Jev** | TypeSafe AI "System One" decision model; founder Diogo Almeida (ex-OpenAI/Google Brain), $40M seed | `POST https://openrouter.ai/api/alpha/decisions`. Body: `{"model":"~typesafe/jev-latest","state": json.dumps(state, indent=1),"questions":{"next_skill":{"type":"choice","instructions":…,"criteria": {skill: description}}}}`. Timeout 60 s. Reads `answers.next_skill.choice`, `.confidence` and `usage.cost` (`deciders.py:45-70`) | OpenRouter `/api/v1/models?output_modalities=decisions`: `typesafe/jev-1.13` (canonical `typesafe/jev-1.13-20260917`), modality `text->decisions`, **text-only**, 32k context, **$0.042/M input, $0 output**. TypeSafe claims 70–500 ms and training by "RLCD" (RL for Calibrated Decisions). The OpenRouter tutorial: ~~"`confidence` measures how concentrated the distribution is"~~ [corrected: the verbatim wording is "`team.confidence` summarizes how concentrated that distribution is" and "Confidence describes the distribution of the alternatives, not whether the workflow is safe to run"], and is not a safety signal |
| **DINO** | Grounding DINO, Swin-T "tiny" | `AutoModelForZeroShotObjectDetection`, fp16, box threshold 0.25, text threshold 0.2. Phrases are joined `"a. b."`, and a label is matched back to a phrase by the phrase's last word (`perception.py:81-98`) | Apache-2.0. README claims about 70 ms on a 4090 (measured in sim) |
| **SAM 3** | Meta "Segment Anything with Concepts" | `Sam3Model`/`Sam3Processor` (transformers ≥5), fp16, `post_process_instance_segmentation(threshold=0.4)`, one phrase per forward pass (`perception.py:100-117`) | Gated HF model created 2025-11-07. Notebooks say about 200 ms per phrase on a 4090 |

Jev has peers in the same "decisions" category on OpenRouter, ~~all sharing one contract~~ [corrected: most share the state+typed-questions contract, but not all. `respan/span-01` and its `-lite` variants in the same category are behaviour-*scoring* models. Every peer was listed after the repo: Kev-4B on 2026-09-25; Solar Decide, Mercury Decide, Tev1 and Liquid D1 between 2026-09-28 and 2026-10-01 (OpenRouter `created` timestamps)]:
- `jaredpalmer/kev-4b`: open-weight, LoRA plus a pointer head on Qwen3.5-4B-Base, 8k context.
- `togethercomputer/tev1-4b-experimental`.
- `inception/mercury-decide`, `upstage/solar-decide`, `liquid/d1`.

Because the contract is shared, the "fast decider" slot is now swappable and could even be self-hosted.

## 4. Hardware and stack

- **Arm.** AgileX PiPER (6-DoF plus a parallel gripper with 70 mm stroke) on CAN at 1 Mbit. It is driven through `piper_control` (Reimagine Robotics, pinned to commit `d82fc9f`) over `piper_sdk`.
- **Camera.** One **fixed** Intel RealSense, a D435i per the `make_figures.py:43` comment. Settings: 1280×720 RGB plus aligned z16 depth at 30 fps; intrinsics fx≈912.1, cx≈638.7, cy≈376.4. **There is no wrist camera.** From the calibration, the camera sits at base xyz (0.607, 0.031, 0.431) m, looking down about 19° off vertical towards the arm.
- **Compute.** An RTX 4090 workstation for DINO and SAM 3. The deciders are remote APIs.
- **Kinematics.** The MuJoCo Menagerie PiPER MJCF is vendored. A "tool" site is added 0.12 m from `link6`, between the pads (`kinematics.py:22,71`).
- **Python deps.** numpy, httpx, mujoco ≥3.3, trimesh, scipy, opencv, torch, transformers ≥5, pyrealsense2.

## 5. Architecture

```
            RealSense 1280x720 RGB + depth (fixed, 3rd-person)
                     |  (Recorder thread is sole camera reader, 15 fps)
                     v
   +---------------- Task.observe(rgb, depth, tool_pose, arm) ----------------+
   |  Detector.best(phrase)  --DINO boxes | SAM3 masks--> plan_grasp / geometry |
   |  depth + hand-eye calib -> base-frame grasp point, yaw, hang depth, tray   |
   |  memory: grasp kept while gripper occludes; tray kept while arm over it    |
   |  proprioception: holding()/closed()/jammed() from gripper opening          |
   +--------------------------> JSON state (~15 booleans + xy/z) -------------+
                     |                                         |
        Jev (text only): state + skill menu        Astra: subset of state + 1 image
        -> {choice, confidence}                    (512px PNG) + last 6 decisions
                     |                                         -> {skill, note}
                     v
   Governor: check_confidence (>=0.10, Jev only) -> check_precondition (rule table)
                     |  refused -> print/log, `continue` (re-ask, same state)
                     v
   Task.target(skill) -> metric xyz + yaw + gripper     (runs 1-7)
   Task.act(skill)    -> scripted multi-stage motion    (run 8, SlideOver)
                     v
   Runner.move_tool_to (15 Hz IK stepping, 1 cm / 4 mm steps)  or
   Runner.glide (pre-solved IK path every 1 mm, min-jerk, 50 Hz)
     + Governor: workspace box, table z, joint-step <= 8 deg, first-cmd <= 8 cm,
       lag/deviation stop, firmware error poll
                     v
   piper_control -> CAN -> PiPER  (firmware speed 10%)
```

### 5.1 The loop (notebooks, cell 8)

Every notebook has the same layout: dry checks, a read-only arm check, homing, the run loop, then shutdown. The loop runs `for step in range(25)`:
1. `observe`.
2. If `task.problem` is set, stop.
3. `choose`.
4. `check_confidence`, then `check_precondition`. On refusal it just `continue`s, with no feedback to the model.
5. On `done`, break.
6. `target`, then `move_tool_to`, then `after`.

Shutdown is `task.finish()`: if the gripper still holds the object, place it or put it back. Then `arm.close()` folds the arm onto its mechanical stops and cuts power.

### 5.2 The state the decider sees (`task.py:130-154`)

The keys are:
- `tool_xyz`, `target_xy`, `place_target_xy`
- `over_target` (within 3 cm), `over_place_target` (within 5 cm)
- `tool_z`, `at_grasp_height` (within 1 cm), `at_carry_height`, `at_release_height`, `at_clear_height`
- `gripper_closed`, `object_held`, `object_placed`, `fingers_jammed`, `task_complete`

Every flag is computed in code. `object_placed` becomes true when `open_gripper` reached its target while `over_place_target` held (`task.py:265-271`). That is a **kinematic assertion, not a perception check**.

### 5.3 Skills: the "tool schema" the decider gets (`skills.py:25-48`)

There are ten skills: `approach_target`, `descend_to_target`, `close_gripper`, `rotate_grasp`, `lift`, `move_over_place`, `lower_to_place`, `open_gripper`, `retreat`, `done`. Each description embeds its exact boolean precondition, for example:

> "descend_to_target": "Lower the open gripper onto the target, ready to grasp. Only when over_target is true, at_grasp_height is false, gripper_closed is false and object_placed is false."

> "open_gripper": "Open the gripper to release the object. Only when object_held is true, over_place_target is true and at_release_height is true; or when gripper_closed is true and object_held is false, after a missed grasp."

Run 8 swaps in `SLIDE_OVER_SKILLS` (`skills.py:51-73`), which adds `move_over_spanner`, `align_over_spanner`, `slide_down` and `let_go`. `Governor.check_precondition` (`safety.py:50-81`) ~~re-implements those rules as a Python dict, so the precondition contract is stated twice, once in the prompt and once in code~~ [corrected: holds a *looser* Python dict than the prompt, so the two copies disagree. `open_gripper` is allowed whenever `gripper_closed` (`safety.py:69`), with no `over_place_target`/`at_release_height` check. `lift` is allowed whenever `closed or held` (`:66`), so lifting an empty closed gripper passes. `approach_target` and `close_gripper` never check `over_target` (`:60,63-64`). `retreat` is always allowed (`:74`). A wrong decider choice such as `open_gripper` mid-carry would therefore **not** be refused].

`task.target` (`task.py:240-263`) maps each skill to a metric target, for example:
- `approach_target` → (grasp_xy, `approach_z`=0.16 m);
- `lift` → carry_z;
- `rotate_grasp` → yaw+45° with the gripper open.

### 5.4 Prompts (verbatim)

Jev instructions, run 2 (runs 3–8 vary only the object):

> "Pick the next skill for a PiPER arm that must put the red cube in the tray. Each skill says when it applies: pick the one whose condition matches the state. Order: approach, descend, close, lift, move over, lower, open, retreat, done."

Astra instructions, run 1:

> "A PiPER arm must pick up the red cube and put it in the tray. Pick the next skill from what you see and the state; each skill says when it applies. Approach from above, descend, close, lift, move over the tray, lower, release, retreat, done."

The Astra user message is assembled in `deciders.py:89-94` as instructions, then `"Skills:\n- name: desc…"`, then `"State:\n{json}"`, then `"Recent decisions:\n"` (the last 6 `skill: note` strings), and finally:

> "Pick one skill. In `note`, say in one or two sentences what you see and why you chose it."

After that comes `camera 'scene':` and the image, downscaled to fit 512 px (512×288 PNG). Astra deliberately receives only `ASTRA_SEES`: arm and gripper state plus heights, but **not** `target_xy`, `over_target` or `over_place_target`. It has to judge from the image whether the gripper is above the cube.

The governor still checks the full state, and **motion targets still come from Grounding DINO**: notebook 01 builds `Detector(backend="dino")  # object positions only; Astra picks the skill`. So the README's perception column for run 1 ("Astra sees the camera images directly") only describes the decider. The notebook text "Astra sees both camera views" is also wrong: one view is sent.

### 5.5 Perception → grasp, the part that differs between pipelines

- **Box (DINO), `plan_grasp` (`skills.py:326-344`).** The grasp goes to the box centre at depth minus `pad_depth`, with yaw 0 (fingers along base y). "Hang depth" is the farthest box corner projected onto the table (`task.py:78-87`).
- **Mask (SAM 3), `grasp_search` (`skills.py:154-303`).** The steps are:
  1. Build a 2 mm base-frame height map from depth.
  2. Consider every 3rd solid cell on the mask, 18 finger directions (10° steps) and tool heights in 5 mm steps.
  3. Require that only the part's own material lies between the fingers ("foreign" cells ≤ 4 mm) and that both finger landing zones are lower than the tips.
  4. Score each candidate as ~~`clear + 0.5·crest − 5·off_centre − 2·centred − 3·width − 0.002·off_axis − 0.5·Δz`~~ [corrected: `min(clear, 0.02) + 0.5·crest − 5·off_centre − 2·centred − 3·width − 0.002·off_axis − 0.5·Δz`, so clearance is capped at 20 mm (`skills.py:293-294`)].

  If nothing qualifies, it falls back to `grasp_from_mask`: distance-transform thickness and SVD axis (`skills.py:86-151`).
- **Rejection filters.** These handle gripper occlusion: a sighting higher than `target_max_z` or larger than `target_max_px` is treated as a sliver of the gripper. The plan made in clear view is kept while the tool is within 8 cm of it (`task.py:156-178`).
- **Geometry (runs 6–8, `geometry.py`).**
  - `locate` (`:103-145`): for each trimesh stable pose with probability ≥ 0.05, at 60 yaw steps of 6°, render the projected surface-sample outline, centre it on the SAM mask, then hill-climb yaw/x/y down to 0.5°/0.5 mm to maximise IoU. The table fixes z, so **no depth is used**.
  - `grasp` (`:189-304`): an exhaustive search over finger direction (2°), position and height. Constraints: pads meet the outermost material squarely (median normal cos ≥ 0.9), touch points are opposite each other, and the open fingers clear the part. Candidates are ranked first by the lever from the centre of mass to the pad line, which uses `meta.json` `centre_of_mass_m` when given.
  - `reach`: the hang-depth bound.
- **SlideOver (run 8, `task.py:317-707`).** Both chrome parts come from one SAM 3 phrase, "metal object". Each mask is fitted with both STLs and the better IoU names it; each must be ≥ 0.85. All six waypoints are pre-checked for IK with a 45° lean "about the fingers" and for fingertip clearance from the table, using the Menagerie collision boxes.
  - `_align` / `_look` (`:557-629`): re-detect, mask out the fingers (projected collision boxes, dilated) and the spanner, fit the held fitting's outline with `locate_held` (±12 mm grid, refined to 0.5 mm), and move by the difference until the error is ≤ 0.8 mm. Abort if the error exceeds 12 mm, after 8 looks, or if IoU < 0.35.
  - `_slide`: descend at 5 mm/s, stopping on lag > 3 mm or when depth shows the gripper within 10 mm of the spanner top on two consecutive readings.

### 5.6 Motion, IK and safety

- **IK** (`kinematics.py:96-138`). Damped least squares on position (λ=1e-4), with orientation solved in the position null space (λ=5e-2). Steps are clamped to 0.15 rad and run for up to 80 iterations.
- **`Runner.solve`** (`skills.py:564-597`). Keeps the tool vertical, or leans it up to 30°, and up to 60° for high carries, when joint 5 would exceed its limit. The lean changes by at most a degree or two per waypoint.
- **`move_tool_to`** (`skills.py:366-441`) streams joint commands at 15 Hz. The command leads the arm on a 15 mm leash, with 10 mm steps, or 4 mm when within 6 cm of the table.
- **`glide`** (`:443-538`) solves IK every 1 mm, or blends joints in joint space, then streams a minimum-jerk profile at 50 Hz with a `should_stop()` hook.
- **Hard limits** (`config.py:51-64`):
  - workspace x 0.10–0.50 (run 8: 0.58), y ±0.35, z 0.005–0.40 m;
  - joint step ≤ 8° per step;
  - first command within 8 cm of the measured pose;
  - command/measurement gap > 3 cm → stop;
  - J6 limited to ±1.695 rad ("commanding beyond it has dropped the arm into its protective damp");
  - firmware speed 10%.
- **Arm quirks encoded** (`arm.py`):
  - The arm forgets joint 3's zero on every power-up, reading about 8° off. `zero_joint3_at_fold` re-zeros it at the folded stop.
  - `emergency_stop` never sends RESUME (0x02), because "the arm drops".
  - Grasp state comes from gripper opening, read in metres under load with about 5 mm of play. Held means opening 0.15–0.85 of stroke; ~~jammed means > 0.6~~ [corrected: `jammed()` is `gripper() > 0.6 and not holding()` (`arm.py:130`), and `holding()` is true for 0.15–0.85 (`arm.py:114`), so jammed effectively fires only at ≥ 0.85. A cube corner reading 0.66, the case the 0.6 line was written for (`arm.py:125-127`, `docs/checks.md:76-77`), is reported as *held*, not jammed, so `rotate_grasp` is unreachable for it. This is a real bug in the repo] (`arm.py:107-130`).

### 5.7 Calibration (`calibrate.py`)

1. With motors off, the user clicks ~~≥ 6~~ [corrected: at least 3 (hard minimum, `calibrate.py:175`), 5 or more to enable the J3 grid search (`calibrate.py:63`), and 6 or more recommended in the docstring] table markers in one frame.
2. The user moves the arm by hand so the closed fingertips touch each marker.
3. A Kabsch rigid fit of camera points to forward-kinematics tip points runs inside a grid search over the **joint-3 zero offset** (±12° in 0.1° steps).
4. Markers with residual > 8 mm are re-touched. Below 5 mm counts as good (`docs/checks.md:54`); ~~the script accepts up to 10 mm~~ [corrected: the re-touch loop ends once max ≤ 10 mm (`GOOD_M`, `calibrate.py:41,210`), but pressing Esc accepts *any* residual. A > 10 mm fit is still written to `calibration.json` with only a printed warning (`calibrate.py:213-226`)].

The shipped `calibration.json` has a J3 offset of 0.044 rad (2.5°). Near the spanner top the README admits "the hand-eye error up there is a few millimetres", which is why run 8 measures *relatively*.

### 5.8 Logging (`record.py`)

A thread is the only reader of the camera. It writes:
- `colour.mp4` with the current decision overlaid;
- a colour-mapped `depth.mp4`;
- 16-bit raw depth every 5th frame;
- `joints.csv` at about 15 Hz (joints, gripper opening, holding);
- `events.csv` (decisions, refusals, stops).

`docs/figures/make_figures.py` regenerates every figure from these files.

## 6. The three pipelines compared (runs 1–3, same scene: red Duplo-like 2×2 brick into an orange crate)

| | Run 1: Astra | Run 2: Jev + Grounding DINO | Run 3: Jev + SAM 3 |
|---|---|---|---|
| Decider input | Text (instructions, skills, ~~11-key state subset, gripper opening~~ [corrected: 11-key `ASTRA_SEES` subset plus `gripper_open` plus a `task` string ("red cube into the tray"), 13 keys in all (nb 01 cell 8)], last 6 decisions) plus **1 RGB image at 512 px** | Full 15-key JSON state as a string, plus skills as `criteria` | Same as run 2 |
| Decider output | `{skill ∈ enum, note}` (strict schema) | `{choice, confidence, probabilities}` | Same as run 2 |
| Who locates objects for motion | Grounding DINO (local) | Grounding DINO | SAM 3 masks |
| Grasp point | Box centre, yaw 0 | Box centre, yaw 0 | `grasp_search` on mask + depth |
| Hang / carry height | Box-corner bound | Box-corner bound | Percentile of mask points |
| Confidence gate | None | ≥ 0.10 | ≥ 0.10 |
| History | Last 6 decisions with notes | None (Markov) | None (Markov) |
| Per-step perception | 2 DINO passes, about 70 ms each (sim number) | Same as run 1 | 2 SAM 3 passes, about 200 ms each |
| Decider latency | Not logged. Rough inference: about 4–6 s per call at effort=high (**UNVERIFIED**, derived from 71 s total minus motion) | About 350 ms (README, *sim*). Independent OpenRouter p50 527 ms (robokrunch, non-robot) | Same as run 2 |
| End-to-end | **1 min 11 s** at 10% speed (thread) | "27 s" for "Jev with the same local perception" (thread; does not say DINO or SAM 3) | Recording: home → done ≈ 22 s; first decision → `done` = 16.6 s |
| Cost per run | Not reported. Order of $0.1–1 at $10/$50 per M (**UNVERIFIED estimate**) | Request ≈ 2.3 kB ≈ 580 tokens ⇒ ≈ $0.000024 per call, ≈ $0.0002 per run (estimate; notebook outputs cleared) | Same as run 2 |
| Outcome | Not documented in repo (thread implies completion) | Not documented | Success (frames checked from the recording) |

Run 3's recording log (`events.csv`) reads: `0 approach_target 0.99 → 1 descend 0.98 → 2 close_gripper 0.98 → 3 lift 0.97 → 4 move_over_place 0.99 → 5 lower_to_place 0.98 → 6 open_gripper 0.99 → 7 retreat 0.96 → 8 done 0.90`. That is 9 calls with 0 refusals.

From `joints.csv`, the idle time between the end of one motion and the next decision is 0.45–1.17 s in run 3 (SAM 3) and 0.09–0.55 s in run 4 (DINO). This bounds perception plus the Jev HTTP round trip at ~~well under a second~~ [corrected: up to about 1.2 s. Re-derived with a 0.002 rad joint-change threshold: run 3 gave 0.63–1.17 s and run 4 gave 0.09–0.55 s. The gap also includes `grasp_search`'s pure-Python search and the arm settling, not only SAM 3 and the HTTP call]. The gripper reads 0.49 while holding the cube, which matches `docs/checks.md` ("34 mm, 0.49").

**Internal inconsistency in the author's own numbers.** The thread says Astra seeing images took 1 min 11 s, and that "Astra only decides" with local perception "drops to 70 seconds". That is a 1 s drop, yet the same post says "Perception was most of Astra's time." The "Astra + local perception" setup is not in the repo. Either a number is wrong or the claim is. Treat the Astra timing as **UNVERIFIED**. The robust conclusion is only that one Astra call at `effort=high` costs seconds, while Jev costs about 0.5 s.

## 7. Results, runs 4–8 (single runs; `docs/runs-4-to-8.md`)

| Run | Perception → grasp | Where the fingers closed | Lever from centre of mass | Carry / release z | Outcome |
|---|---|---|---|---|---|
| 4 | DINO box ("handle" boxed the whole trowel) | Neck, 48° across | 6 mm (by luck) | 32 / 21 cm | Trowel fell **across the crate rim**. Jev still declared `done 0.96` (recording) |
| 5 | SAM 3 handle mask + depth (`grasp_search`) | Across the handle | 55 mm (56 in README) | 32 / 21 cm | In the crate, but swung blade-down |
| 6 | SAM 3 + trowel STL (fit IoU 0.91) | Across the ferrule | 2 mm | 26 / 15 cm | In the crate, carried level, landed on its side |
| 7 | SAM 3 + fitting STL (IoU 0.97). On chrome, 17% of mask pixels had no depth and 11% were impossible; camera-only grasp points drifted up to 21 mm over 17 still frames | Across hex flats, 0.5 mm off the axis | <1 mm | 16 / 7 cm | In the crate. A camera-only variant also succeeded twice, but off the flats |
| 8 | SAM 3 + 2 STLs, SlideOver | Hex flats, 45° lean, 10 mm offset | n/a | 19 / 3 cm | Looks: 2.7 mm then 0.6 mm (tolerance 0.8). Slid 130 mm down at 5 mm/s; let go at t≈139 s. The spanner moved 2.8 mm (likely rubbing) |

Run 4's log: 8 Jev calls, confidence 0.96–0.99, `done` 18.2 s after the recording started. The author's own "what did not work" list: run 5's blade-down hang, run 8 nudging the spanner, and "depth is useless on chrome".

## 8. Context: how this relates to the other sources

- **RoboCurve (GPT-6 Astra on YAM arms).** That setup has the model emit `move_to` end-effector poses as tool calls, with 3 camera views every turn and medium effort. Results:
  - bowl task: Astra 19/20 at $0.94 and 2.5 min per run, against 8/20 for Fable 5.1;
  - puzzle task: 2/20 for both.

  This repo is the opposite design point. The model only chooses a skill name; code computes every pose. Kubenka's thread aims squarely at that style: "most send every camera frame to the big model. That is why a cube-into-tray takes minutes."
- **Same-week Jev robot repos.**
  - `FazalAAli/jev-robotics-demo` (sim): Jev against Claude Opus 5, both stacking. Jev took 19.1 s and $0.0006; Opus took 158.8 s and $0.75 (one retry). Its stated design rule is "code owns the math, Jev owns the judgment".
  - `YuanKJing/Jev-as-Policy` (MuJoCo): two chained Jev calls. The first picks an intent; the second picks per-axis `{positive, negative, stay}` and the fingers. This is a finer action granularity than here.
  - These are sim or n=1 demos. piper-astra-jev is one of the few on **real hardware**.

## 9. Assessment

### Strengths

1. **Clean separation of concerns.** The model picks names; code owns geometry, timing, IK and limits; failures surface as human-readable `problem` strings before anything moves. Examples: `_check_reach` at `task.py:212`, and `_check_path` at `task.py:462` with "the arm cannot reach …", "both parts must stand upright". This is the right default for a frontier-API harness.
2. **Real-world competence in the non-LLM layers.** These are the genuinely useful contributions:
   - occlusion-aware memory ("look, reach, touch");
   - proprioceptive grasp verification with load-play calibration;
   - pre-checked reach before any lift, accounting for how far the object hangs;
   - CAD silhouette pose fitting for specular parts;
   - centre-of-mass-aware grasps;
   - finger-masked, *relative* visual servoing that cancels hand-eye error.
3. **Honest, well-instrumented demos.** A single clock across video, joints and decisions; figures regenerated from data; a failure list in the thread; safety advice. The hardware-quirk notes (J3 zero, J6 limit, never RESUME, gripper play) are directly reusable for anyone on a PiPER.
4. **A useful systems datapoint.** With a fixed skill library, a per-step frontier VLM at high effort is the latency bottleneck. A sub-second typed decider plus local perception runs a pick-place in about 20–30 s at 10% arm speed.

### Weaknesses and gaps

1. **The decider is nearly redundant.** Each skill description states an exact boolean precondition ~~and the governor enforces the same table~~ [corrected: the governor enforces only a looser subset of those conditions (`safety.py:59-76`; see §5.3). The *prompt's* conditions, however, select exactly one skill at each step of the happy path, so the conclusion stands. An FSM written from the skill descriptions reproduces the released logs of runs 3 and 4 step for step; I hand-checked this against `events.csv`, including run 4 going straight from `open_gripper` to `done` because `at_clear_height` was already true at the 21 cm release], so a 20-line FSM would make every decision in runs 2–8 at 0 ms and $0. The demos measure **latency and cost of a selector, not intelligence**. There is no FSM baseline and no perturbation trials (moved object, failed grasp, distractors), which is where a decider would earn its keep. Jev's confidences of 0.90–0.99 just mean it reads the booleans reliably.
2. **No outcome verification.** `object_placed` and `task_complete` are asserted from kinematics. Run 4 logged `done 0.96` with the trowel hanging over the crate rim.
3. **Refusals are not fed back.** `safety.py` says reasons are fed back as observations, but the loop only prints and `continue`s with an unchanged state. Jev has no history at all, so a refusal can repeat until the 25-step budget runs out.
4. **The confidence gate is semantically weak.** `min_confidence=0.10` against a "concentration" metric that ~~TypeSafe~~ [corrected: OpenRouter's Jev tutorial, not the TypeSafe launch post,] says is not a safety signal.
5. **Brittle API handling.** No retries or fallback: `raise_for_status()` would raise mid-run with torque on.
6. **Task-specific code.** `SlideOver` is about 400 lines for one insertion. Generality comes from a human (or a coding agent) writing new task classes, not from the model.
7. **n=1 per condition, no randomisation, no blinding.** Run 1's and run 2's outcomes and logs are not in the repo; only runs 3 and 4 have released recordings. Timings come from a tweet thread with an internal contradiction. Notebook outputs are cleared.
8. **Known stale or contradictory text.** The README run-1 "perception" column; "both camera views" against one image sent; the 55 vs 56 mm lever.

### Genuinely novel vs repackaged

- **Repackaged.** LLM-as-skill-selector over parametrised primitives. This is the SayCan / Inner Monologue / Code-as-Policies lineage, with typed outputs.
- **New as of Sep 2026.**
  - One of the first real-hardware uses of a "System One" typed decision model as the per-step controller.
  - A head-to-head with a frontier VLM on the same skill stack.
  - A careful, reusable geometric toolkit (STL-silhouette pose under specularity, centre-of-mass grasp ranking, finger-masked relative alignment) for small, cheap arms.

### Maturity

Demo-grade research code. It is well structured and well commented, but it has no tests, a single commit and notebook-driven runs. The safety layer is thoughtful for a 10%-speed lab demo.

### What Ilia's Opus-backbone harness should take from it

**Borrow:**
- **The action contract.** The model chooses among named, parametrised skills, and metric targets come from perception and geometry, never from the model. If Opus does emit parameters, keep them semantic (object id, grasp-mode id), not raw poses. Make the skill menu a strict tool/enum schema. With Claude Opus 5.5 (`claude-opus-5-5`, $4/$20 per M, cache reads $0.20/M), forced `tool_choice` `any`/`tool` returns 400. Use `auto` with `strict: true` tools, or `output_config.format` structured outputs. Thinking cannot be disabled and effort defaults to `medium`, so set effort explicitly and cache the static skill catalogue with `cache_control`.
- **A governor with machine-readable refusals,** plus pre-flight feasibility checks (reach, path, joint margins, fingertip–table clearance via collision geometry) that return a `problem` string. Do feed those strings back, unlike this repo.
- **Occlusion memory, perception sanity filters** (height, size) and **proprioceptive grasp checks** with calibrated thresholds; a missed-grasp counter that stops at 2.
- **The geometric toolkit for known objects:** `ObjectGeometry.locate`/`grasp`/`locate_held`. Also relative measurement (held part against target in the same image) for anything with less than 5 mm tolerance; absolute hand-eye calibration will not get you there.
- **The recorder design:** one clock, decisions overlaid on video, raw depth, joints. Treat it as the eval substrate.

**Avoid / do differently:**
- **Don't put a frontier model in the per-step loop for phase selection that code can determine.** Use an FSM, or a cheap decision model (Jev, or the self-hostable Kev-4B / Tev1-4B), as the fast path. Invoke Opus at:
  - (a) task start, to choose and parametrise skills and perception phrases;
  - (b) anomalies, meaning refusals, stalls, missed grasps or a geometry-fit IoU below threshold;
  - (c) **visual outcome verification** at phase ends ("is the object inside the tray?"), which this repo lacks.

  This is where a "light learned action head" could also sit: replacing hand-tuned grasp scoring or the phase FSM, not replacing IK.
- **Always run an FSM baseline and perturbation trials,** n ≥ 10 per condition with randomised placements, and log the decider's latency and cost per call (the repo prints them but publishes nothing).
- **Handle API failure explicitly:** retries, timeouts and a deterministic fallback skill such as `retreat`/`finish`, so a 4xx/5xx never leaves the arm mid-air under torque.

## Sources

- Repo: https://github.com/RobotKitAI/piper-astra-jev (code read in full; `docs/runs-4-to-8.md`, `docs/checks.md`, notebooks 01–08)
- Recordings release: https://github.com/RobotKitAI/piper-astra-jev/releases/tag/recordings (runs 3 and 4: `events.csv`, `joints.csv`, `colour.mp4` analysed)
- Author thread: https://x.com/MichalKubenka/status/2101229393079111800 (and replies 2101229764497351126, 2101230393546457489, 2101230636509847740, 2101230964370280656, 2101231159032041903, 2101231357573714128, 2101231445876355113), fetched via api.fxtwitter.com
- RobotKit org post: https://x.com/RobotkitAI/status/2101181033106256021 ; https://robotkit.dev/ ; https://www.acheronlabs.com/
- Jev tutorial / API: https://openrouter.ai/docs/guides/community/jev-tutorial ; https://openrouter.ai/api/v1/models?output_modalities=decisions ; https://openrouter.ai/api/v1/models/typesafe/jev-1.13/endpoints
- TypeSafe launch post: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- GPT-6 Astra model page: https://developers.openai.com/api/docs/models/gpt-6-astra.md
- SAM 3: https://arxiv.org/abs/2511.16719 ; https://huggingface.co/facebook/sam3
- Grounding DINO tiny: https://huggingface.co/IDEA-Research/grounding-dino-tiny
- piper_control: https://github.com/Reimagine-Robotics/piper_control
- RoboCurve Astra report: https://openai.robocurve.org/gpt-6-astra/
- Related Jev robot repos: https://github.com/FazalAAli/jev-robotics-demo ; https://github.com/YuanKJing/Jev-as-Policy ; https://github.com/robokrunch/jev-physical-ai
- Claude Opus 5.5 API facts: Anthropic claude-api skill reference (model table and Opus 5.5 migration notes, cached 2026-06-24); OpenRouter listing `anthropic/claude-opus-5.5`

## Verification (fact-check pass)

Fact-checked 2026-10-01 against these primary sources:
- **Code.** The clone at `research/repos/piper-astra-jev` (single commit `10d671e01d46…`), every `piper_llm/*.py`, `calibrate.py`, all 8 notebooks (all code cells: 0 outputs, `execution_count=None`), `docs/runs-4-to-8.md`, `docs/checks.md`, `docs/figures/make_figures.py` and the figure captions (read as images).
- **Recordings.** Run 3 and run 4 `events.csv`, `joints.csv` and `frames.csv`, plus frames extracted from run 3's `colour.mp4` with ffmpeg.
- **GitHub API.** Repo, fork, release, org and user endpoints.
- **The X thread**, via `api.fxtwitter.com/2/conversation/…`.
- **Model and vendor pages.** OpenAI's GPT-6 Astra model page and changelog; OpenRouter `/api/v1/models?output_modalities=decisions` and the Jev tutorial; the TypeSafe launch post; HF API for `facebook/sam3` and `IDEA-Research/grounding-dino-tiny`; arXiv 2511.16719; robotkit.dev; acheronlabs.com.
- **Related work.** RoboCurve's Astra report, the `FazalAAli/jev-robotics-demo` README, the `robokrunch/jev-physical-ai` README and the `Jev-as-Policy` README.
- **Claude facts** from the bundled claude-api skill reference.

### Confirmed (seen in a primary source)

**Repo metadata.**
- Single commit `10d671e` by Michal Kubenka, dated 2026-09-19 10:05:48 +0200 (= 08:05:48 UTC).
- Repo created 08:05:50 UTC and pushed 08:36:18 UTC.
- Apache-2.0, 11 stars, 1 fork (`0xk1h0/piper-astra-jev`, created 2026-09-28, no new commits), 0 issues.
- Org `RobotKitAI` created 2026-05-13, blog robotkit.dev.
- The `recordings` release was published 2026-09-19 08:28 UTC with two zips (run3 19.1 MB, run4 18.5 MB).

**Thread and timeline.**
- The @RobotkitAI post went up 05:26:11 UTC with the exact text quoted.
- The thread root went up 08:38:20 UTC (1,597 views, 5 likes).
- Thread quotes are all verbatim: "1m11s at 10% arm speed"; "drops to 70 seconds… Jev with the same local perception: 27s… Perception was most of Astra's time"; "8 Jev decisions, confidence 0.96 to 0.99, done in 18 s"; the "What did not work" list.
- So the note's internal-inconsistency finding (71 s vs 70 s against "Perception was most of Astra's time") is **confirmed**.
- Run folder timestamps are at `make_figures.py:41-42`: run5 02:26:42, run4 02:36:12, run7 02:58:34, run6 04:20:20, run8 07:37:24, plus run3 00:04:45 from the zip name. The order 3, 5, 4, 7, 6, 8 is correct.
- GPT-6 Astra was released Sep 3 (OpenAI changelog).
- Jev launched Sep 15 (TypeSafe post by Diogo Almeida).
- `typesafe/jev-1.13` was listed on OpenRouter 2026-09-18 00:01:24 UTC (`created`=1789689684). Canonical slug `typesafe/jev-1.13-20260917`, `text->decisions`, 32,000 context, $0.042/M input, $0 output.
- `~typesafe/jev-latest` "always redirects to the latest model in the Jev family"; only 1.13 exists.

**API calls.**
- Astra: body and params exactly as stated (`deciders.py:83-115`): Responses API, `reasoning.effort="high"`, strict `json_schema` with a `skill` enum, timeout 600 s, PNG thumbnail ≤ 512 px (512×288 from 1280×720).
- Jev: body as stated (`deciders.py:45-70`): `POST https://openrouter.ai/api/alpha/decisions`, `model:"~typesafe/jev-latest"`, `state` = `json.dumps(state, indent=1)`, one `choice` question `next_skill` with `criteria` = skill descriptions, timeout 60 s.
- Astra's page: text+image input, 1,050,000 context, 128,000 output, effort `low/medium/high/xhigh/max`, $10 / $1 cached / $50.
- TypeSafe: 70–500 ms and RLCD (launch post).
- The $40M seed and the ex-OpenAI/Google Brain background are confirmed only by **secondary** sources (runtimewire, winzheng), not by the TypeSafe post.

**Perception.**
- Grounding DINO tiny: Apache-2.0, fp16, threshold 0.25, text 0.2, phrase join and last-word match (`perception.py:81-98`).
- SAM 3: `facebook/sam3`, gated (`manual`), HF `createdAt` 2025-11-07, one phrase per pass, threshold 0.4 (`perception.py:100-117`).
- arXiv 2511.16719 is "SAM 3: Segment Anything with Concepts".
- "~70 ms on a 4090 (sim)" is at `README.md:182-183`.
- "~200 ms per phrase" is in the markdown of notebooks 03, 05, 06 and 07.

**Hardware, kinematics and limits.**
- Camera: intrinsics fx 912.102, fy 912.117, cx 638.714, cy 376.352, "D435i colour, 1280 x 720" (`make_figures.py:43`).
- Calibration: translation (0.607, 0.031, 0.431). The optical axis is 18.7° off vertical, pointing toward −x (the arm). J3 offset 0.04398 rad = 2.52°.
- Tool site 0.12 m from `link6` (`kinematics.py:22,71`).
- IK: DLS λ=1e-4, null-space orientation λ=5e-2, 0.15 rad clamp, 80 iterations (`kinematics.py:96-138`).
- `move_tool_to`: 15 Hz, 15 mm leash, 10 mm / 4 mm steps, 6 cm caution distance.
- `glide`: 1 mm IK path, minimum-jerk, 50 Hz, `should_stop`.
- `Runner.solve` leans up to 30° / 60°.
- Hard limits are at `config.py:51-64`; J6 ±1.695 rad is at `config.py:12-16`.
- `piper_control` is pinned to `d82fc9f2c572…` (`requirements.txt`).

**State, skills and loop.**
- 15-key `PickPlace` state with 3 cm / 5 cm / 1 cm thresholds (`task.py:130-154`).
- `object_placed` is set only by `after()` (`task.py:265-271`).
- 10 skills, with the two quoted descriptions verbatim (`skills.py:25-48`). `SLIDE_OVER_SKILLS` is at `skills.py:51-73`; `APPROACH_SKILLS` is unused (`skills.py:75-83`).
- All prompts quoted in §5.4 are verbatim. The Astra message order is at `deciders.py:89-94`. `ASTRA_SEES` omits `target_xy`, `over_target`, `place_target_xy` and `over_place_target`.
- Notebook 01 uses `Detector(backend="dino")  # object positions only; Astra picks the skill`. Its markdown says "both camera views" but only `{"scene": rgb}` is sent.
- Refusals only `print` and `continue`, while the `safety.py` docstring says reasons are fed back. That contradiction is **confirmed**.

**Grasping and geometry.**
- `grasp_search`: 2 mm cells, every 3rd cell, 18 directions, 5 mm heights, foreign ≤ 4 mm (`skills.py:154-303`). The `grasp_from_mask` fallback is at `:86-151`. `plan_grasp` (box centre, yaw 0) is at `:326-344`.
- `locate`: stable poses with p ≥ 0.05, 6° × 60 yaw steps, hill-climb at 2°/2 mm then 0.5°/0.5 mm, no depth (`geometry.py:103-145`).
- `grasp`: 2° direction steps, cos ≥ 0.9, opposite contacts, lever-first ranking (`geometry.py:189-304`).
- `locate_held`: ±12 mm grid in 2 mm steps, refined to 1 mm then 0.5 mm (`geometry.py:147-186`).

**SlideOver (run 8).**
- Phrase "metal object"; both fits must reach IoU ≥ 0.85; six waypoints pre-checked with a 45° lean about the fingers; fingertip–table check via Menagerie collision boxes.
- `_align`/`_look`: 0.8 mm tolerance, abort at > 12 mm, after 8 looks, or at IoU < 0.35.
- `_slide`: 5 mm/s, 3 mm lag stop, depth gap < 10 mm on 2 consecutive readings.

**Recordings and run results.**
- Run 3 `events.csv` matches the note exactly: 9 decisions, confidence 0.90–0.99, first decision at 5.096 s, `done` at 21.704 s (16.6 s loop), 0 refusals, 0 stops.
- A frame at 28.5 s shows the brick in the crate.
- Median gripper reading while holding: 0.489.
- Run 4 `events.csv`: 8 decisions, 0.96–0.99, `done 0.96` at 18.162 s.
- Run 4–8 figures and numbers (48°, 6/55/2/<1 mm levers, 32/21, 26/15, 16/7 and 19/3 cm, IoU 0.91, 17%/11%, 21 mm over 17 frames, looks 2.7 → 0.6 mm, 130 mm slide, 2.8 mm spanner shift) match `docs/runs-4-to-8.md`.
- IoU 0.97 for run 7 comes from the `run7_plan.jpg` caption. Let-go at t 139 s comes from the `run8_sequence.jpg` caption.
- Jev cost estimate reproduced: the request body is 2,326 bytes, about 580 tokens, about $0.000024 per call.

**Related work.**
- RoboCurve (Sep 4, 2026): bowl task 19/20 for Astra vs 8/20 for Fable 5.1 (and 1/20 for Fable 5); $0.94 vs $2.12; 2.5 vs 6.8 min. Puzzle 2/20 for both. `move_to` absolute end-effector poses, three camera views per turn, medium effort, 20-call budget.
- `FazalAAli/jev-robotics-demo`: 158.8 s / $0.75 vs 19.1 s / $0.0006.
- `Jev-as-Policy`: two chained calls, with per-axis `{positive, negative, stay}` choices.
- robokrunch: p50 0.527 s over 300 OpenRouter calls on 2026-09-19.

**Claude Opus 5.5 facts** (claude-api skill, model table cached 2026-06-24, where the model is marked "launching"):
- `claude-opus-5-5`, $4/$20, cache reads $0.20.
- Thinking cannot be disabled; effort defaults to `medium`.
- Forced `tool_choice` `any`/`tool` returns 400; use `auto` + `strict: true` or `output_config.format`.

### Corrections (claim → correct value)

Each is also marked inline with ~~ ~~ and [corrected: …].

1. **Third-party reaction.** "The only third-party reaction was a reply from AgileX" → AgileX replied on 2026-09-21. @SaroyaShehryar also replied twice on 2026-09-19 ("the link is broken?", "I'd love to talk more"), and there is 1 quote tweet (fxtwitter conversation API).
2. **Author's projects.** "a Camoufox MCP server" (among his projects) → `mkubenka/gomoufox` is a fork, not his own work. His own repos include `bb` (a Bitbucket CLI) and Ansible roles.
3. **Governor rules.** "`Governor.check_precondition` re-implements those rules … stated twice" → the governor's dict is looser than the prompt's (`safety.py:59-76` vs `skills.py:25-48`):
   - `open_gripper` needs only `closed`;
   - `lift` needs only `closed or held`;
   - `approach_target` and `close_gripper` never check `over_target`;
   - `retreat` is always allowed.

   A decider error such as `open_gripper` mid-carry, or `lift` on an empty closed gripper, would pass the governor.
4. **Weakness 1.** "the governor enforces the same table" → same correction as #3. The FSM-redundancy conclusion still holds, because the prompt's own conditions select exactly one skill per step on the logged runs.
5. **Jammed threshold.** "jammed means > 0.6" → effectively ≥ 0.85. `jammed()` requires `not holding()`, and `holding()` is true for 0.15–0.85 (`arm.py:114,130`). The documented 0.6 line (`arm.py:125-127`, `docs/checks.md:77`) is dead code in practice: a corner grasp at 0.66 counts as *held*, so the lift is allowed and `rotate_grasp` never fires.
6. **Grasp score.** Score term `clear` → `min(clear, 0.02)` (`skills.py:293`).
7. **Marker count.** "user clicks ≥ 6 table markers" → at least 3 are required (`calibrate.py:175`) and at least 5 for the J3 grid search (`:63`); 6 or more is only recommended.
8. **Calibration acceptance.** "the script accepts up to 10 mm" → the loop exits when max ≤ 10 mm, but Esc accepts any residual and writes `calibration.json` with only a warning (`calibrate.py:213-226`).
9. **Idle time.** "bounds perception plus the Jev HTTP round trip at well under a second" → up to about 1.17 s in run 3. My re-derivation gives 0.63–1.17 s for run 3 and 0.09–0.55 s for run 4. The idle time also includes the pure-Python `grasp_search` and arm settling.
10. **Tutorial wording.** "`confidence` measures how concentrated the distribution is" (presented as a quote) → the verbatim text is "`team.confidence` summarizes how concentrated that distribution is"; the "not safe to run" sentence is also verbatim from the tutorial.
11. **Source attribution.** "a 'concentration' metric that TypeSafe says is not a safety signal" → the source is OpenRouter's Jev tutorial, not TypeSafe's post.
12. **Astra input.** "11-key state subset, gripper opening" → 13 keys: the 11 `ASTRA_SEES` keys plus `gripper_open` plus `task: "red cube into the tray"` (nb 01 cell 8).
13. **Decision-model peers.** "[peers] all sharing one contract" → not all. `respan/span-01` and its `-lite` variants, in the same OpenRouter category, are behaviour-scoring models. All listed peers post-date the repo: Kev-4B on Sep 25; Solar Decide, Mercury Decide, Tev1-4B and Liquid D1 on Sep 28 to Oct 1.

### Unverifiable / weakly sourced

- **Earlier RobotKit posts** (TRON 1 + "Piper arm kit" for sanitary cleaning in Mar 2026; DOBOT Atom W in Aug 2026): **UNVERIFIED**. The fxtwitter timeline endpoint returned nothing, and the account has only 6 tweets and 4 media items. The org-level facts (MCP, visual canvas, VLA training, "Powered by Acheron Labs Inc.", Acheron footer "Wien - Prague - Bratislava") are confirmed on the live pages.
- **"No HN or Reddit discussion"**: not re-checked, so **UNVERIFIED**.
- **Astra per-call latency of 4–6 s and Astra cost of $0.1–1 per run**: both remain **UNVERIFIED** estimates. Neither number is logged anywhere, and the notebook outputs are cleared. At effort=high, output and reasoning tokens at $50/M dominate. About 10 calls × 2k reasoning tokens alone is already about $1, so the upper bound is plausibly above $1.
- **What "27 s" in the thread measures**: unresolved. Released recording lengths are 29.1 s for run 3 and 27.06 s for run 4, and recordings include homing and folding. The figure captions' "after the run" timestamps are 26.8 s (run 4), 27.8 s (run 5), 29.8 s (run 6), 32.5 s (run 7) and 191 s (run 8). "27 s" could be run 2's full recording, but that cannot be checked.
- **Run 2's recording time**: `record.py:12`'s docstring example folder `out/recordings/20260918_234807_run2` *hints* that run 2 was recorded 2026-09-18 23:48 local. **UNVERIFIED** (it may only be an example).
- **Runs 1, 2, 5, 6, 7 and 8 outcomes**: known only from the author's docs and figures. No logs or videos were released for them.

### Missed details (added)

**1. Torque is on in the "read-only" cell.** Constructing `Arm()` enables torque. `Arm.__init__` calls `piper_init.reset_arm` ("cuts motor power for under a second, then enables torque and motion at full speed", `arm.py:86-91`), then sets speed 10%. It may also auto-run the J3 re-zero sequence, which enables and disables motors (`arm.py:76-85`). So the notebooks' "Read-only arm check. The arm does not move in this cell" has motors **enabled**. The README's "run without torque, then run for real" (`README.md:99-100`) is not what the notebooks do: there is no torque-off rehearsal of the decision loop, only sensor and perception reads.

**2. Protective stops do not end runs 1–7.** If `move_tool_to` returns a reason (deviation > 3 cm, joint-step fault, IK failure, firmware error, workspace refusal), the loop logs "stopped:" and asks the decider again. Only `task.problem` ends a run, and that is set only by a reach/plan failure or by 2 missed grasps (`task.py:113-118`). Run 8 is stricter: any non-slide `_glide` that stops sets `problem` and ends the run (`task.py:545-555`). A missed grasp ("the fingers closed on nothing", `task.py:518-523`) also ends run 8.

**3. Run 8's decider sees a different state** (`task.py:393-420`). It has 18 keys, including the metric values `alignment_error_mm`, `spanner_top_mm` and `fitting_bottom_mm`, plus `aligned` and `on_spanner`. `at_release_height` and `fingers_jammed` are dropped. `over_target` and `over_place_target` use 2 cm thresholds (vs 3 cm and 5 cm in `PickPlace`). `SLIDE_OVER_SKILLS` removes `rotate_grasp`, `move_over_place`, `lower_to_place` and `open_gripper` (still 10 skills). Run 8 also uses `gripper_effort_nm=2.0`, against the default 1.0 (nb 08 cell 4), and `Governor(limits=SafetyLimits(workspace_high=(0.58, 0.35, 0.40)))`.

**4. Logged telemetry is thinner than the code computes.**
- Jev's `probabilities` vector and the served snapshot id (`model` in the response) are discarded. Only `choice` and `confidence` go to `events.csv`.
- Astra's `note` is truncated to 80 characters in `events.csv`.
- Notebooks 01–07 print "median … ms" but compute a **mean** (`seconds / calls`); only nb 08 labels it "mean". The `median_ms` helpers are unused.
- The released zips omit `depth.mp4` (release notes say "left out for size") but include `frames.csv`.

**5. Detector call count per step.**
- `_plan` re-runs detection only while the arm is more than 8 cm from the planned grasp and nothing is held, closed or released.
- `_see_tray` runs only while the arm is more than 20 cm from the remembered tray.
- So a step uses 0–2 detector passes, plus extra `target_whole` passes in the handle scenes (`HANDLE_SCENE`: `target_whole=("tool","object")`).
- A geometry fit below IoU 0.6 silently keeps the last plan (`task.py:186-187`).

**6. Run 8 fit quality and caveats.**
- Plan-time fits: fitting IoU 0.96, spanner IoU 0.93 (`run8_plan.jpg`).
- The two in-hand "looks" fit only at IoU 0.78 and 0.79 (`run8_look.jpg`), against an abort threshold of 0.35.
- The 0.6 mm residual is measured by the same camera that drives the correction, so it is not independent ground truth. The 2.8 mm spanner shift during the 130 mm slide suggests the real error during descent was larger.
- Timeline: gripped at t 50 s, lined up at 84 s, at the floor at 138 s, let go at 139 s, run end at 191 s, so about 3.2 min end to end. Run 7's done confidence was 0.98, with `lift` and `lower_to_place` at 1.00.

**7. Objects and STLs are bespoke.** The STLs are not vendor CAD. Each `meta.json` says "generated from calliper measurements of the real part; no downloaded geometry", measured 2026-09-18.
- The trowel's centre of mass is *estimated* from assumed steel and beech densities ("Estimated mass 128 g, not weighed").
- Every new object needs a hand-built mesh and a centre-of-mass estimate, which is a major scalability cost of the geometry route.

**8. Unreported attempts exist; selection effects.**
- `arm.py:21-23` records a joint-6 sign-flip trial on 2026-09-19 that "mirrored every grasp angle (a plan of +50 deg came out at -32 deg, along the handle)".
- Run 7 also had two unreported camera-only trials (`runs-4-to-8.md:142-146`).
- "Each notebook was run once for the recording it describes" (README) therefore does not exclude failed or debugging attempts.

**9. Unmentioned sim claims in the README** (`README.md:179-184`): "SAM 3 locates bolt holes to about 0.3 mm when it finds them, and its recall is roughly 60%. Expect worse on real hardware."

**10. Astra constraints relevant to the design.** Per OpenAI's Sep 3 changelog, GPT-6 Astra supports no custom `temperature`/`top_p` and no `logprobs`. So Astra cannot supply a calibrated confidence, and the confidence gate necessarily applies to Jev only. OpenAI added `service_tier: "ultrafast"` for Astra on Sep 29, after these runs, which would change the latency comparison.

**11. Calibration of the related-work comparison.** The `jev-robotics-demo` README also says "That Opus run needed one retry; clean first-try Opus runs took about 55 s and $0.19". In that repo Opus drives low-level `move_hand_to(x,y,z)` tools, not a skill menu, so the 158.8 s figure is not a like-for-like comparison. Its "code owns the math, Jev owns the judgment" is credited there to `BrendanH18/jev_fsd`.

**12. SAM 3 licence.** The HF tag is `license:other` (the SAM License), not Apache/MIT. Check it before commercial use in a harness.

**13. API-shape nuance.** The repo sends `state` as a JSON *string* (`indent=1`), while the OpenRouter tutorial sends a JSON *object*. Both evidently work, but the string form inflates input tokens slightly.
