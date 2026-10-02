# Manda Robotics (mandarobotics.com): deep dive

*Researched 2026-10-01. Repos cloned to `/home/agent/skoltech/controlr/research/repos/manda-{RoboLab-Verified,robot-episode-labeler,real2sim-frontier-assets}`. Anything marked UNVERIFIED could not be confirmed from a primary source.*

## TL;DR

- **Manda Robotics is a robot-evaluation company, not a robot-control company.** Site tagline: "Manda Robotics builds evaluation infrastructure for frontier robotics models." Its about text: "We are building training infrastructure, simulation, and real-world evaluations for robot training. We measure how well robot policies perform, generalize, and transfer." It sells custom evaluation to labs: "We are working with select robotic labs on custom evaluation of frontier robotic policies in sim & real world environments (San Francisco)" (`/collaboration`). It has no hardware product and no LLM-driven controller product.
- **Team:** about two people. The public GitHub, Hugging Face and git-commit trail points to **Finn Metz** (GitHub `CptMgm`; commit emails finn@apartresearch.com; Apart Research board member, co-founder of the Seldon AI-safety accelerator) and **Yannick Metz** (GitHub `ymetz`; ETH Zürich postdoc working on RLHF and human-AI interaction). These are the only two members of the HF org `mandarobotics`. Their roles, titles and any relationship between them are not stated anywhere (UNVERIFIED). **No funding, investors or job postings were found.** Orgs were created recently: GitHub on 2026-07-31, X `@Mandarobotics` around 2026-08-28 (decoded from the snowflake ID).
- **Why it matters for an Opus-backbone harness.** Three of Manda's artifacts carry hard data for Ilia:
  1. A **6,000-episode matched comparison of five open VLAs on RoboLab-120**, with latency, failure-mode taxonomy and run-to-run noise. This gives baselines and a failure catalogue an LLM-driven harness must handle.
  2. A **VLM-as-policy connector** in their RoboLab fork. The same Gemini model scored **0/4 when emitting Cartesian deltas** and **6/6 when it output an image point plus a phase that a geometric controller converted to motion**. This is the clearest evidence in the set for the "LLM picks *what/where*, deterministic controller does metric motion" split.
  3. A **frontier coding-agent Real2Sim study** with exact model IDs and costs, including `claude-opus-5-5` vs `claude-fable-5-1` vs `gpt-6-astra`/`gpt-6-sol` vs `gemini-3.8-flash`.

  Manda also wrote a careful, measured **Gemini video-annotation pipeline** whose prompt and time-grounding tricks carry over directly to LLM success detection.

---

## 1. Company overview

| Item | Finding | Source |
|---|---|---|
| Product | Evaluation infrastructure: sim + real-world evals, benchmark tooling, data annotation, Real2Sim assets | homepage, `/collaboration`, social card ("Training infrastructure, simulation, and real-world evaluations for robot learning.") |
| Target customers | "select robotic labs"; collaboration form (name, company/lab, email, project link) | `/collaboration` |
| LLM/FM-driven? | Manda *evaluates* foundation models and uses Gemini as a tool (annotation, pointing policy). It does not ship a controller. | repos |
| Hardware? | None of its own. Sims: Isaac Sim 5.1/6.0 + Isaac Lab 2.3/3.0; MuJoCo, Newton and Genesis in the physics post. GPUs: RTX PRO 5000/6000 Blackwell, rented L40 on RunPod | blog `data.js`, `docs/verified/verification.md:107` |
| Location | San Francisco (collaboration page). Gemini credit was topped up in DKK ("the 200 DKK top-up", labeler `docs/research-log.md:288`), which suggests a Copenhagen link | |
| Funding | None disclosed or found (UNVERIFIED / likely bootstrapped). ~~A full labeler experiment program cost "≈ $18–19 total"~~ [corrected: "≈ $18–19 total" is the budget accounting for one batch, the 2026-08-30 second pass funded by "the 200 DKK top-up" (six runs at $15.57, judge calls of about $2, demo reruns). It is not the cost of the whole program (`docs/research-log.md:288-293`)] | |
| Jobs | No careers page; the JS bundle routes are `/`, `/collaboration`, `/where-are-all-the-robots`, `/blog/state-of-robot-policies`, `/blog/comparing-physics-engines`, `/asset/Klarpul` | `assets/index-Dc-n3B9M.js` |
| Public repos | `Manda-Robotics/RoboLab-Verified` (fork of NVlabs/RoboLab v0.3.1), `robot-episode-labeler` (Apache-2.0), `real2sim-frontier-assets`. Referenced but **private/404**: `Manda-Robotics/vlm-pinpoint`, plus a "`manda` product repo" mentioned in `docs/decisions.md:4-5` | GitHub API |
| Hosted | HF Space `mandarobotics/robot-episode-labeler` (public, Gradio, sleeping). Replicate `mandarobotics/robot-episode-labeler` (public page with 15 runs; a third-party catalogue lists it as "deprecated") | HF API, replicate.com |
| X | `@Mandarobotics` (display name "Manda"). Known post, 2026-09-23: "GPT-6 Astra generates the best zero-shot Real2Sim assets tested against Opus 5.5 and Fable 5.1. It's also the most expensive. We gave frontier models ~20 iPhone captures of real-world objects…" (360 likes per syndication API). The profile timeline could not be rendered (X login wall, syndication rate-limited) | cdn.syndication.twimg.com |
| Press / HN / Reddit | No press and no HN hits. Third-party mentions are limited to the Chinese wiki `ImChong/Robotics_Notebooks` (PR #2092, "Manda Robotics 开源通用策略 RoboLab-120 横评") and an X-post dataset (`Develata/AI-Barking`) | gh search |

Pages on the site:
- **"Understanding the Limits of Open-Source General Robotics Policies"** (2026-09-17). This is the main study.
- **"Where are all the Robots?"**: "Coming soon" (meta: "A post on the gap between robotics demos and real-world deployment").
- **Unlinked, `noindex`:** "Comparing physics engines for robotics simulation" (2026-09-30) and a sim asset page `/asset/Klarpul`. The latter is a "Klarpul KL003" stick vacuum in MJCF + USD for MuJoCo/Genesis/Isaac Sim with "Scripted suction and debris pickup". It looks like a Real2Sim asset showcase.

---

## 2. Study: 5 open policies × 120 tasks × 10 episodes on RoboLab-120

**Protocol.** NVIDIA RoboLab-120 has 120 tabletop DROID tasks in Isaac Lab. All five DROID-adapted checkpoints were run zero-shot through each model's own RoboLab adapter ("We retain each policy's native preprocessing and action execution"). Runs used Manda's Isaac Sim 6 / Isaac Lab 3 port (`github.com/ymetz/RoboLab`, Yannick Metz commits 2026-08-21…27). The protocol was 10 matched seeds per task. A SHA-256 audit over the non-camera `initial_state` datasets confirms all 1,200 physical start states are identical across policies (`sceneAlignment` in `data.js`). **The simulation is paused during inference**, so latency never eats into the task budget. On top of RoboLab's success and partial-credit Score, they analysed HDF5 control channels and event logs and did a **manual narrated review of 1,150 episodes** (50 tasks per policy, 42 shared). Aggregate SR whiskers are 95% exact binomial intervals.

**Aggregate results** (from `blog/state-of-robot-policies/data.js`, 1,200 episodes per policy):

| Policy (checkpoint) | Params | SR % [95% CI] | Score % | Zero-success tasks | Policy query avg | GPU | Episodes / GPU-h |
|---|---|---|---|---|---|---|---|
| Cosmos3-Nano-Policy-DROID | 16B | **35.1** [32.4, 37.9] | 50.7 | 42 | 829 ms | RTX PRO 6000 Max-Q | 19.0 |
| π0.5-DROID (`pi05_droid_jointpos`) | ≈3.3B | 27.5 [25.0, 30.1] | 42.8 | 54 | **128 ms** | RTX PRO 5000 | 49.7 |
| MolmoAct2-DROID | 5.4B | 13.8 [11.9, 15.9] | 27.1 | 87 | 473 ms | RTX PRO 5000 | 28.0 |
| G0.5 (Qwen3.5-2B + ActionCodec) | ≈2B+ | 10.5 [8.8, 12.4] | 21.5 | 86 | 584 ms | RTX PRO 5000 | 26.5 |
| GR00T-N1.7-DROID | 3B | 10.2 [8.5, 12.0] | 21.0 | 94 | 251 ms | RTX PRO 5000 | 36.2 |

- **Sanity check against the official leaderboard:** π0.5 27.5% vs 28.0% (336/1200); Cosmos 35.1% vs 36.8%.
- **Complementarity:** the per-episode oracle across all five policies reaches **49.4% SR** (593/1200). Cosmos has 169 unique successes and π0.5 has 93. Pairwise success Jaccard is 0.19–0.38. 30 tasks are solved by no policy.
- **Difficulty slice, "complex" tasks (170 episodes):** Cosmos 27.1%, π0.5 19.4%, MolmoAct2 1.8%, GR00T 0.0%, G0.5 0.0%.
- **Run-to-run noise:** π0.5 was re-run on identical seeds because of a rendering issue. "An episode that succeeded the first time succeeded again only 64% of the time, and per-task success rate moved by 20 points or more on 28 of the 120 tasks. One task fell from 8/10 to 2/10; another rose from 4/10 to 10/10." Overall, 80.0% of episodes had the same outcome (baseline 81.3%).
- **Metric validity:** RoboLab's SPARC smoothness metric ranks Cosmos the smoothest, but video shows it is the most jittery. Manda's jitter index (mean |Δv| / mean speed) gives Cosmos 12.07 vs 4.05–8.27 for the others.

**Policy-specific failure signatures** (human review counts):

| Policy | Signature | Count |
|---|---|---|
| π0.5 | bumps the carried object into the container rim | 48 episodes / 16 tasks |
| Cosmos 3 | "abandons the last centimetre", plus jitter | 28 / 19 |
| MolmoAct 2 | cannot re-angle the wrist; repeats one approach; acquires an object in 38% of episodes vs 57% for Cosmos/π0.5 | 32 / 20 |
| GR00T N1.7 | grabs whatever is centred in frame, then loses coherence (last-third activity 0.68 of first third) | 29 / 18 |
| G0.5 | "test grip" loops: touch, lift 1 cm, release; gripper closes at all in only 54% of episodes | 49 / 21 |

Wrong-object episode rates are 41–54% for every policy.

**Manda's stated thesis** (relevant to Ilia; quoted):

> "Our working hypothesis is that general-purpose VLMs will increasingly supply scene understanding, task decomposition, and planning. The useful research question is therefore where specialized robot policies add capability beyond an off-the-shelf foundation model equipped with a suitable control interface. Recent results show that for a fixed arm and parallel-jaw gripper such as DROID, general-purpose models already can propose a target position, orientation, and gripper command, with inverse kinematics (IK) translating the desired pose into joint configurations. That can simplify reaching, but a reachable pose alone does not solve collision-free transport, stable contact, controlled release, or recovery after a slip."

The "Recent results" link points to Jay Chooi's (Robocurve) X post 2026-09-05: "GPT-6 Astra scored 95% on a robot control task, up from Fable 5.1's 40%, with 6.2x fewer output tokens at 2.3x lower cost." **This links Manda directly to the user's "Robocurve GPT-6 Astra" source.** Manda also cites RoboDojo's leaderboard as a ranking reversal: G0.5 scores 20.23 / 14.88% SR there vs π0.5 11.41 / 6.91%, while on RoboLab π0.5 leads 27.5% to 10.5%. They argue rankings are "properties of a model–benchmark pairing". They also flag that the Cosmos authors benchmarked on RoboLab-120 during development.

---

## 3. RoboLab-Verified (fork of NVlabs/RoboLab v0.3.1)

**What they found in upstream RoboLab** (`README.md:17`, `docs/verified/findings.md`), from a 328-episode audit:
- "All 88 successes were declared while a target object was still moving."
- "64 % of logged 'object grabbed' events occurred with the hand open."
- One task's subtask ladder completed at 0.07 s, before the arm had moved.
- Success has no exclusivity: sweeping the whole table into the bin still scores 1.0.
- Friction defaults are unrealistically high: pads μ=2.0, some objects μ=10.
- Scenes are not settled: 62 (task, object) pairs sink at reset.

**What they changed** (`README.md:21-25`, `docs/verified/changes.md`):
- Success is confirmed only after targets are at rest.
- Episodes end early once success becomes impossible.
- Containment is capped at the rim.
- A grasp requires a carry: `OBJECT_GRIPPED → OBJECT_CARRIED → OBJECT_GRABBED_SUCCESS`.
- Release and drop are distinguished by the commanded gripper state.
- `WRONG_OBJECT_PLACED` added.
- Contact sensors on both finger pads, logged to HDF5.
- `--friction` is a run parameter. In a 32-episode-per-condition sweep, SR was insensitive to a 4× change in μ; the behaviour metrics were not.

Engineering details: 251 offline tests run in CI. Every change carries a RUNTIME/OFFLINE/NONE verification level ("About 25 of the 120 tasks have been run against the patched code, most with π0.5 only", `README.md:29`). Recording runs at 15 Hz control (`sim.dt` 1/120 × decimation 8; actions are 7 joint positions + gripper in [0, 1], `docs/verified/dense_annotations.md:24-35`).

New embodiments:
- **Bimanual YAM rig** (two I2RT YAM arms; one overhead and two wrist RealSense views) with Ai2's MolmoAct2-BimanualYAM client: 5/8 on a parity task vs 4/8 in Ai2's ManiSkill harness, and 0/4 on three unseen tasks.
- Dual-Franka rig.
- ALOHA config (π0.5 base 0/6).

The upstream repo also ships `/robolab-scenegen` and `/robolab-taskgen` **Claude Code skills** (added by NVIDIA's Xuning Yang, 2026-05-31).

### 3.1 The VLM-as-policy path (`policies/vlm_pinpoint/`)

This is the only Manda code that drives a robot with a foundation model. The controller lives in **`Manda-Robotics/vlm-pinpoint`, which is not public** (GitHub 404, not on PyPI). Only the ~124-line connector is visible.

```
RoboLab env (Isaac Lab, DROID Franka+Robotiq, 15 Hz)
  ~~obs["policy"] = {scene_rgb, scene_depth, scene_intrinsics, scene_cam_pos,
                   scene_cam_quat, ee_pos, ee_quat, gripper_pos}  (batched)~~
  [corrected: these are the keys the CONNECTOR looks for, under obs["policy"][k] or obs[k].
   Public RoboLab emits groups image_obs / proprio_obs / viewport_cam, e.g.
   image_obs["over_shoulder_left_camera"], image_obs["wrist_cam"].
   No public obs config emits scene_rgb/scene_depth/scene_intrinsics/scene_cam_*
   (grep: only connector.py and its test). The "policy" obs group feeding this connector
   is unpublished. Also, DROID ee_pos/ee_quat = Robotiq base_link, robot-root frame
   (robolab/robots/droid.py:279-309,478-480).]
        │  connector.to_harness_observation(obs, env_id)        connector.py:51-93
        │    - unbatch one env; missing channels -> None
        │    - tip_pos = ee_pos + R(ee_quat)·[0,0,0.1034]       connector.py:25,38-42
        ▼
  vlm_pinpoint.PointingController(GeminiBackend(**kw)).step(obs, instruction)   [PRIVATE]
        │  VLM returns (image point u,v ; phase)   <- "the model returns an image
        │  point and a phase; a geometric controller does the metric work"
        │  (deprojection with depth+K+camera pose presumably -> 3D target; UNVERIFIED)
        ▼
  action chunk  ──► returned "unchanged" to RoboLab (connector.py:122-124)
```

Key details:
- `connector.py:3-7` (docstring): "The model returns an image point and a phase; a geometric controller turns that into metric motion. The same Gemini scored 0/4 driven as Cartesian deltas and 6/6 through this path on BananaInBowl."
- `FLANGE_TO_FINGERTIP_M = 0.1034` (`connector.py:25`): "The controller aims the fingertip, not the flange; without this the arm stops a hand's width short." Separately, the recorded `ee_pose` is the Robotiq `base_link` and "the TCP is 15 cm ahead of the recorded point" (`dense_annotations.md`). TCP bookkeeping is a classic failure source for LLM pose-targeting harnesses.
- `to_harness_observation` passes missing keys as `None` "so the controller can say what it needs rather than failing on a KeyError deep inside". A unit test caught an `a or b` ValueError on numpy arrays (commit `16f32df`).
- Lazy import keeps the benchmark runnable without the package (`offline_tests/test_vlm_connector.py:21-23, 60-62`).

**Results** (`docs/verified/README.md:113-119`, `changes.md:130`):

| Task (RoboLab) | Gemini pointing + geometric controller |
|---|---|
| BananaInBowl | **6/6** (Cartesian-delta mode: **0/4**) |
| BananasInCrate | 5/6 |
| OneBottleInSquarePail | 2/6 |
| FruitsOnion | 1/6 |

Caveats stated by Manda: "Top-down, single-object, uncluttered picks only." These ran "on earlier revisions" of the harness (`verification.md:101`). The trial counts are tiny (4–6). **The exact Gemini model ID, prompt, phase vocabulary, and controller law are not public** (UNVERIFIED); ~~the dashboard groups them under the policy family "gemini_pointing"~~ [corrected: `gemini_pointing` is the per-run `policy` label that the runner stamps. The dashboard collapses it into the family "Gemini" (`dashboard/loaders/local.py:183-194`)].

---

## 4. robot-episode-labeler (Gemini video → timestamped subtasks + pass/fail)

**Contract:** `video + task description -> [{start, end, label, result, attributes, description}]`. Apache-2.0. Author: Finn Metz, 2026-08-27…31.

Pipeline (`src/rel/pipeline.py`; every setting is in `src/rel/config.py`):
1. **Sample** with ffmpeg on a 0.5 s grid at 224 px (`config.py:25-26`). The rationale (ADR-002): video APIs ingest ~1 fps, and "manipulation boundaries routinely fall inside one such frame".
2. **Contact sheets**: 20 frames per sheet, 5 columns, each tile stamped with its episode time *in the gutter* (`video/contact_sheet.py:71-78`). "Burned-in stamps are the only channel a model reliably reads a timestamp from."
3. **Segment**: one call per 60 s window (6 sheets, 1 sheet overlap).
4. **Label**: each segment gets ±1 s of context.
5. **Subdivide / refine**: `strict` mode only.
6. **Validate**: "The model proposes; the code decides" ~~(`validate.py:1`)~~ [corrected: this wording is from `README.md:125`. `validate.py:1` reads "Deterministic cleanup. The model proposes; this module decides."]. Ordering, clamping, minimum duration 0.15 s, and closed-vocabulary snapping are done in code.

LLM client (`annotation/llm.py:117-237`):
- `google-genai` `generate_content` with `response_mime_type="application/json"` and `response_schema=<pydantic>`.
- `temperature=0.0` by default; `thinking_level` / `media_resolution` / `media_processing` are optional.
- `REQUEST_TIMEOUT_S = 180` ("six workers wedged on network I/O at 0% CPU").
- ~~4 retries~~ [corrected: `max_retries=4` means at most 4 attempts, so at most 3 retries. Sleep is `min(2**attempt + random(), 20)` s (`llm.py:123,205-236`)] with exponential backoff, triggered by string markers (429/5xx/UNAVAILABLE…).
- Empty candidates are treated as transient.
- Prompts are cached once per process because edits mid-run "measured two different pipelines", and a prompt-set SHA fingerprint is stamped into every response.
- Default model `gemini-3.7-flash` (`config.py:16`). The judge is `gemini-3.1-pro-preview`.

Key prompt rules, quoted verbatim:
- `prompts/segment_v2.md:14-28`: "A subtask is one completed change in the state of the world… A pick followed by a place is TWO subtasks, not one… Never merge them into a single move-the-cup segment." and "Do NOT start a new segment for motion that leaves the world unchanged: approaching… adjusting or re-seating a grasp… hesitation…"
- `prompts/label_v2.md:13-27`: "label: the manipulation event that COMPLETES at the END of this segment… result: 'pass'… 'fail'… 'unknown' if the frames genuinely do not show the outcome… Judge failure by what you can see. Do not infer success from the task description. If the segment shows a grasp that slips, that is a failure even if a later segment recovers."

Measured on WGO-Bench (Macrodata; 100 episodes, 743 gold segments; DROID / Galaxea / HomER), from `docs/results.md`:

| Config | F1@IoU0.5 | ±0.5 s boundary recall | Median boundary error | $/video-hour |
|---|---|---|---|---|
| baseline sheets (3.7-flash) | 0.598 | 0.329 | 1.04 s | 1.08 |
| + pick/place decomposition prompt | 0.695 | 0.508 | 0.49 s | 1.24 |
| native video 2 fps + same prompt | 0.720 | 0.488 | 0.52 s | 0.81 |
| schema mode (gold vocabulary) | 0.771 seg / 0.699 end-to-end | | | |

Other measured results:
- Label accuracy on matched segments: 0.80 (model-judged). DROID improves 0.50 → 0.59 with `label_v2`.
- `thinking_level=low` is 40% cheaper but significantly worse at tight IoU (−0.050). Not adopted.
- Events shorter than 1 s are essentially undetectable (recall 0.038).
- Noise floor: "Temperature 0 is not determinism", with ±0.03–0.08 F1 swings between identical runs (`results.md:92-98`). All changes are gated by a 4,000-resample paired bootstrap.
- ADR-004: gripper-telemetry snapping recovers 56% of gold boundaries within ±0.5 s on Galaxea but only 13% on DROID, so it is optional.

---

## 5. real2sim-frontier-assets (frontier coding agents build Isaac Sim assets)

**Setup** (`README.md`, `prompts/`):
- Each agent CLI (`claude`, `codex`, `gemini`) got photos plus hand-held videos of an object. Dimensions and mass were withheld.
- Environment: OpenUSD 26.08 `pxr`, trimesh, MuJoCo 3.13, headless Blender, ffmpeg. "Isaac Sim itself is not installed on this machine" (`prompts/*/environment.md:15`).
- Task: build a functional UsdPhysics asset with joints and explicit masses. "Do not use animation or teleportation to fake physical interaction… Only claim what you tested."
- The paced launch prompt (`prompts/launch-paced.md:11-14`): "Save a first complete candidate under output/candidate_1/ at about 20 minutes, then keep improving output/ until about 60 minutes…"

Results (`results/runs.csv`, `results/private_scene_summary.csv`, `results/neutral_corkscrew.csv`):

| Object | Agent (served model) | Wall time | Cost (USD) | Mass vs ground truth | Physics result |
|---|---|---|---|---|---|
| Can (GT 0.367 kg) | gpt-6-astra | 57 min | 17.27 | 0.369 | travel 0.055 m, tilt 0° |
| | claude-fable-5-1 | 27.5 min | 9.35 | 0.369 | 0.056 m, 0° |
| | gemini-3.8-flash | 9 min | 1.61 | 0.370 | 0.055 m, 0° (visuals broken) |
| | **claude-opus-5-5** | 36.7 min | **5.98** | 0.368 | 0.137 m, **tilt 90° (fell over)** |
| | gpt-6-sol | 60 min | 13.06 | 0.370 | 0.083 m, tilt 90.7° |
| Corkscrew (GT 0.143 kg) | gpt-6-astra | 26 min | 9.08 | 0.24 | passive wing 0.06° (coupling rejected by PhysX) |
| | **claude-fable-5-1** | 16 min | 5.82 | 0.19 | **41.99°: the only working gear coupling** |
| | gemini-3.8-flash | 30 min | 1.69 | 0.15 | 2.17° |
| Chair (GT 7.44 kg) | gpt-6-astra | 25 min | 9.49 | 11.5 | rug travel 0.19 m; swivel 84.7° |
| | claude-fable-5-1 | 22 min | 7.38 | 12.75 | 0.16 m; 88.1° |
| | **claude-opus-5-5** | 40 min | **5.70** | 13.8 | **0.46 m**; 81.8° |

Token volumes are 2.4M–45M processed tokens per run, more than 90% of them cache reads. The neutral mechanism test (`eval/neutral_corkscrew.py`) holds the body kinematic and drives one wing toward 85% of its joint limit with a force drive: stiffness 50, damping 2, maxForce 5, 90 steps at 120 Hz (`:129-135`). Rendering copies measured PhysX poses onto a collision-free visual twin because "RTX on this pod does not consume the dynamic PhysX transforms reliably".

The X headline ("GPT-6 Astra generates the best zero-shot Real2Sim assets") rests mainly on visual quality; the qualitative overview is a claude.ai artifact that returned 403. The repo's physics data is mixed: Astra's corkscrew mechanism failed while Fable's worked. **Opus 5.5 was cheapest among the Claude/GPT agents, but its can tipped over and its chair slid ~~2.4× further than the others~~ [corrected: 0.463 m vs 0.195 m for Astra (2.4×) and 0.159 m for Fable (2.9×); `results/private_scene_summary.csv`].**

---

## 6. Unlinked post: "Comparing physics engines for robotics simulation" (2026-09-30, noindex)

PhysX (Isaac Sim 6.1 CPU TGS), Newton 1.5.2 / MuJoCo-Warp 3.11, MuJoCo CPU 3.11 and Genesis 1.4.1 were run on matched scenes:
- Sliding-block travel agrees to about 0.05 mm, but peak contact force differs ~10× (1.06 N PhysX vs 10.46 N MuJoCo). "Halving the timestep moves the cube more than switching engines."
- Single-scene instrumented real-time factor: MuJoCo CPU 15.6× vs PhysX 0.94× vs MJWarp 0.24× vs Genesis 0.11×.
- 131,072 Panda worlds: MJWarp 35.2M world-steps/s, Genesis 26.4M.

Take-home quote: "Before trusting a policy comparison, establish what your simulation comparison actually controls… report more than success."

---

## 7. Relationships to the other user sources

- **Robocurve / GPT-6 Astra report:** Manda cites Robocurve founder Jay Chooi's Astra-vs-Fable robot-control tweet as evidence that frontier LLMs plus IK can handle reaching. Robocurve is also a **direct competitor**: YC S26, "Evals for robots", SF, team of 3, Inspect Robots framework. Manda is more sim-centric and annotation-centric.
- **RoboDojo:** cited by Manda as an independent suite that ranks policies differently from RoboLab.
- **YAM arms:** Manda's fork adds a simulated bimanual I2RT YAM rig (`docs/bimanual_yam.md`). This is useful if Ilia targets YAM hardware and wants a sim twin.
- **GPT-as-Policy / GPT-Policy / EmbodiedSWE / DexAgent / innate-os:** these are LLM-as-controller approaches. Manda's `vlm_pinpoint` connector is the minimal benchmark-side adapter one would write to score such agents on RoboLab-120 against the five VLA baselines above.

---

## 8. Assessment

**Strengths**
- **Unusually rigorous measurement culture:**
  - matched seeds with a hash audit;
  - exact binomial CIs;
  - a reported rerun showing only 64% success reproducibility;
  - a paired bootstrap before adopting any prompt change;
  - human review checked back against logs;
  - every claim tagged with its verification level;
  - negative results retained.

  This is the best "how to evaluate" reference among the sources.
- Concrete, quantified failure taxonomies for the five leading open VLAs, with latency on fixed GPUs.
- One clean A/B on action interfaces for a VLM (deltas 0/4 vs pointing + controller 6/6).
- Open, readable code with documented ADRs.

**Weaknesses / caveats**
- Not a control product. The one LLM-control component, `vlm-pinpoint`, is **private**, and its numbers come from 4–6 trials on 4 easy tasks with an unnamed Gemini model.
- All policy evaluation is in simulation on one embodiment (DROID). The "real-world evaluations" claim has no public artifact yet.
- The RoboLab-Verified patches are validated on ~25/120 tasks, mostly with π0.5. The 6,000-episode study ran on the Isaac Sim 6 port; ~~it is unclear how many Verified scoring patches it used, since the blog says it uses "RoboLab's success and partial-credit scores" (UNVERIFIED)~~ [corrected: it used none. The `cli_*_robolab120` recordings are "Isaac Sim 6.0 + upstream scoring" (`docs/verified/dense_annotations.md:91,428-430,495-496,1151-1157`). verification.md:103 lists "The full 120-task re-run under the patched harness" as "Planned; not yet scheduled"].
- The Real2Sim study is n=1 per (model, object), uses private scenes and media, and its headline relies on qualitative judgement.
- Very young: orgs are about two months old, the team is about two people, and there is no disclosed funding.

**Novel vs repackaged**
- Novel: the eval-integrity audit of RoboLab, the matched cross-policy behavioural analysis, and the frontier-agent Real2Sim comparison.
- Repackaged / known ideas:
  - VLM pointing + depth deprojection + geometric controller (cf. MOKA, PIVOT, RoboPoint, MolmoAct, Gemini Robotics-ER pointing).
  - Contact-sheet video annotation: they explicitly reuse Macrodata's WGO-Bench sampling parameters (0.5 s, 224 px, 20 per sheet, 5 columns).

**Maturity:** alpha. The study and fork are solid research artifacts; the products are prototypes.

### What Ilia's Opus-backbone harness should borrow

1. **Action interface.** Have Opus output *semantic/geometric targets* (an image point or object ID plus a phase or primitive), not per-step Cartesian deltas. A deterministic layer (deprojection with depth, K and extrinsics, then IK or a Cartesian controller) should do the metric work. Manda's 0/4 → 6/6 is small-n but directionally strong and matches Manda's thesis. Put the optional learned action head *under* this interface for the parts Manda's taxonomy says IK does not solve: transport clearance, stable acquisition, release, slip recovery.
2. **TCP bookkeeping as a first-class, tested constant.** Examples are `FLANGE_TO_FINGERTIP_M = 0.1034` and the 15 cm `base_link`-to-TCP offset in recorded poses. Write pure-numpy unit tests for every frame transform, as `test_vlm_connector.py` does.
3. **Observation contract.** Use a flat per-step dict where missing channels are `None`, so the agent or controller can report what it lacks. Keep the backend import lazy and keep the policy as a server/client.
4. **Time-grounded visual verification.** For success/failure checks and recovery triggers, give Opus **timestamped contact sheets** (burned-in times, own sampling grid ≥2 Hz) rather than raw video. Reuse the prompt rules verbatim:
   - "pick and place are two subtasks";
   - "name the event that COMPLETES at the END";
   - "Judge failure by what you can see. Do not infer success from the task description."

   Enforce invariants in code ("the model proposes; the code decides"). Derive confidence from inter-stage disagreement, not self-report.
5. **Success semantics.** Declare success only when targets are at rest. A grasp requires a carry. Distinguish commanded release from drop. Flag wrong-object placements. These belong in the harness's verifier, not only the benchmark.
6. **Eval hygiene.**
   - Ten episodes per task is too few: task SR swung by 20 or more points on 28/120 tasks under identical seeds.
   - Use paired comparisons on matched seeds with bootstrap CIs.
   - Temperature 0 is not determinism, so replicate.
   - Log the full config plus a prompt hash with each result.
   - Verify proxy metrics (SPARC) against video.
7. **Benchmark target.** RoboLab-120 (DROID, 15 Hz, joint-position actions) through Manda's fork with a `vlm_pinpoint`-style connector is the most directly comparable public test bed. ~~Baselines to beat are Cosmos3 35.1% and π0.5 27.5%.~~ [corrected: those are Manda's re-run numbers for two policies. The official RoboLab-120 leaderboard, fetched 2026-10-01, is led by FLUX 3 Action at 42.9% (515/1200), then HiDream-O1-Embodied at 39.9%. It already lists non-VLA entries that are the closest comparators for an LLM harness: Phoenix (TAMP+FM, RGB+Depth) at 34.4% and VoLo (Agent, RGB+Depth) at 28.2%.] Because the sim pauses during inference, it **hides Opus latency**, so report wall-clock and per-decision latency separately (π0.5 128 ms/query vs multi-second LLM calls).
8. **Real2Sim with Claude.** Coding agents can produce loadable UsdPhysics assets for $2–17 in 10–60 min. In Manda's single trial Opus 5.5 produced physically wrong dynamics (tipping can, sliding chair), and Fable 5.1 was the only agent to get a gear coupling right. If digital twins are used for regression-testing the harness, add an automatic physics check (e.g., `neutral_corkscrew.py`) instead of trusting agent self-reports.

### What to avoid
- Per-step delta control by the LLM.
- Single-run, 10-episode headline numbers.
- Upstream RoboLab event logs as ground truth ("grabbed" means contact).
- Relying on a closed controller you cannot inspect, as with `vlm-pinpoint`: build your own.

---

## Unverified / unreachable
- Manda X timeline and threads (login wall, syndication 429); Manda LinkedIn; the claude.ai artifact overview (403).
- `Manda-Robotics/vlm-pinpoint` and the `manda` product repo are private: the controller law, prompts and Gemini model ID are unknown.
- Founder titles and roles, funding, headcount.
- ~~Whether the 6,000-episode study used Verified scoring patches.~~ [corrected: resolved. It used upstream v0.3.1 scoring on Isaac Sim 6.0; see Verification section.]

## Sources
- https://mandarobotics.com/ ; https://mandarobotics.com/collaboration ; https://mandarobotics.com/where-are-all-the-robots
- https://mandarobotics.com/blog/state-of-robot-policies/index.html (+ `data.js`, `video-config.js`, `app.js`)
- https://mandarobotics.com/blog/comparing-physics-engines/index.html (unlinked, noindex) ; https://mandarobotics.com/asset/Klarpul
- https://github.com/Manda-Robotics ; https://github.com/Manda-Robotics/RoboLab-Verified ; https://github.com/Manda-Robotics/robot-episode-labeler ; https://github.com/Manda-Robotics/real2sim-frontier-assets
- https://github.com/ymetz/RoboLab (Isaac Sim 6 port) ; https://github.com/NVlabs/RoboLab ; https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html
- https://huggingface.co/spaces/mandarobotics/robot-episode-labeler ; https://huggingface.co/api/organizations/mandarobotics/members ; https://replicate.com/mandarobotics/robot-episode-labeler
- https://huggingface.co/datasets/macrodata/WGO-Bench
- https://x.com/Mandarobotics ; https://x.com/i/status/2102811446052827455 (via cdn.syndication.twimg.com) ; https://x.com/chooi_jeq/status/2096064315115839904
- https://github.com/ImChong/Robotics_Notebooks/pull/2092 ; https://github.com/Develata/AI-Barking (x_coded.jsonl)
- https://scet.berkeley.edu/meet-finn-metz-the-founder-tackling-ai-safety/ ; https://www.linkedin.com/in/finn-metz/ (via search snippets only) ; https://yannickmetz.me
- https://www.ycombinator.com/companies/robocurve ; https://robodojo-benchmark.com/

---

## Verification (fact-check pass)

*Adversarial re-check on 2026-10-01 against primary sources. These were re-fetched live: mandarobotics.com (home, /collaboration, /where-are-all-the-robots, /asset/Klarpul, blog index.html + `data.js`, physics post, JS bundle `assets/index-Dc-n3B9M.js`); GitHub API (org, repos, `ymetz/RoboLab`); HF API; Replicate; X syndication API; the YC Robocurve page; the Apart Research /about page; SCET Berkeley; yannickmetz.me; the RoboLab-120 leaderboard. Local clones were read at: RoboLab-Verified `790c1a5` (2026-09-08), robot-episode-labeler `31f0b19`, real2sim-frontier-assets `9823603`. Inline fixes in the body are marked ~~old~~ [corrected: …].*

### Confirmed (seen in a primary source)

**Company and team**
- Homepage tagline, about text, and `team@mandarobotics.com`.
- `/collaboration` text: "select robotic labs … (San Francisco)". Form fields: name, company/lab, email, project link.
- Social-card text (image checked).
- "Where are all the Robots?" reads "Coming soon".
- `/asset/Klarpul` and the physics post are both `noindex,nofollow`. The bundle routes match the six listed.
- GitHub org created 2026-07-31T20:08:46Z, with 3 public repos (stars 2 / 2 / 8).
- `vlm-pinpoint` returns 404 on GitHub and on PyPI.
- HF org `mandarobotics` has 2 members: FinnLennard "Finn Metz" and ymetz "Y Metz". The Space is public, Gradio, SLEEPING, created 2026-08-30.
- Replicate shows "Public · 15 runs". modelpedia lists it as `"status": "deprecated"`.
- X user id 2093220998313041920 decodes to 2026-08-28 06:15 UTC.
- The 2026-09-23 Manda post text matches, with 360 likes.
- Jay Chooi's post (2026-09-05 02:34 UTC) text matches: "GPT-6 Astra scored 95% … Fable 5.1's 40% … 6.2x fewer output tokens at 2.3x lower cost". YC: Robocurve, Summer 2026, "Evals for robots", team 3, SF, Jay Chooi Founder/CEO, "Inspect Robots" v1.
- Finn Metz:
  - Board Member on apartresearch.com/about.
  - Co-founder of Seldon Lab (SCET).
  - SCET says he studied at Copenhagen Business School, which supports the DKK → Copenhagen inference. Whether Manda itself has a Copenhagen presence is still UNVERIFIED.
- Yannick Metz: postdoc at ETH Zürich (IVIA Lab), working on RLHF / human-AI communication.

**State-of-policies study** (blog text + `data.js`)
- Protocol: 5×120×10 = 6,000 episodes; native adapters; sim paused during inference; Isaac Sim 6 port linked to `github.com/ymetz/RoboLab`.
- Initial-state audit: `sceneAlignment.definition` = "SHA-256 over non-camera initial_state datasets…", with 1,200/1,200 equal for all 10 policy pairs.
- 95% exact binomial whiskers. 1,150 watched episodes (50 tasks per policy, 42 shared).
- Per-policy rows are exact from `policyMetrics`:

  | Policy | SR / CI | Score | Zero-success tasks | Query latency | Throughput | GPU |
  |---|---|---|---|---|---|---|
  | Cosmos | 421/1200 = 35.08% [32.38, 37.86] | 50.68 | 42 | 829.5 ms | 19.03 eps/GPU-h | RTX PRO 6000 Blackwell Max-Q |
  | π0.5 | 330 = 27.50% [24.99, 30.12] | 42.77 | 54 | 128.2 ms | 49.72 | RTX PRO 5000 Blackwell |
  | MolmoAct2 | 166 = 13.83% | 27.11 | 87 | 473 ms | 28.05 | RTX PRO 5000 Blackwell |
  | GR00T | 122 = 10.17% | 21.01 | 94 | 250.7 ms | 36.23 | RTX PRO 5000 Blackwell |
  | G0.5 | 126 = 10.50% | 21.48 | 86 | 584.5 ms | 26.47 | RTX PRO 5000 Blackwell |

- All runs used `parallel_environments: 10`.
- Leaderboard references: π0.5 336/1200 = 28.0%, Cosmos3-Nano 441/1200 = 36.8%. Both confirmed on the live leaderboard.
- Oracle 593/1200 = 49.42%. Unique successes: Cosmos 169, π0.5 93. Jaccard 0.194–0.383. 30 shared-zero tasks.
- Complex slice (n=170): 46, 33, 3, 0, 0 successes.
- Rerun numbers confirmed: 64%, 28/120 tasks, 8/10→2/10, 4/10→10/10, 80.0% vs 81.3%.
- Jitter index 12.07 vs 4.05–8.27.
- Failure-signature counts confirmed: 48/16, 28/19, 32/20, 29/18, 49/21. Also 38% vs 57%, 0.68, and 54% vs 82%.
- Wrong-object rates 41.2–53.9%.
- RoboDojo numbers: 20.23/14.88% vs 11.41/6.91%.
- The thesis quote is verbatim. The "Recent results" link is `x.com/chooi_jeq/status/2096064315115839904`.

**RoboLab-Verified**
- Confirmed against `README.md:17-29` and `docs/verified/findings.md` (A1, A2, B1, C1, C3):
  - 328-episode corpus;
  - 88/88 successes still moving;
  - 64% of grabs with the hand open;
  - 0.07 s ladder;
  - sweep-the-table;
  - μ 2.0 and 10.0;
  - 62 sinking pairs;
  - 251 offline tests;
  - RUNTIME/OFFLINE/NONE levels;
  - ~25/120 tasks.
- `dense_annotations.md:24-35`: 15 Hz (1/120 × 8); `actions` (T, 8) = 7 joint positions + gripper in [0, 1]; `ee_pose` = Robotiq `base_link`; "TCP is 15 cm ahead".
- `verification.md:90-91,101-102,107`:
  - YAM 5/8 vs 4/8, and 0/4 on three tasks;
  - ALOHA π0.5 base 0/6;
  - "Cosmos3 and Gemini pointing ran on earlier revisions";
  - L40 48 GB with Isaac Sim 5.1.
- The skills commit `84e8d1b` is by Xuning Yang, 2026-05-31.
- `connector.py` (124 lines): `FLANGE_TO_FINGERTIP_M = 0.1034` at :25, `to_harness_observation` at :51-93, and `PointingController(self._backend or GeminiBackend(**kw)).step(...)` at :120-124. The docstring says 0/4 vs 6/6 on BananaInBowl.
- `docs/verified/README.md:113-119`: 6/6, 5/6, 2/6, 1/6.
- Commit `16f32df` (2026-08-27): "8 tests … one of them caught a real bug".
- Lazy-import tests are at :21-23 and :60-62.

**robot-episode-labeler**
- Settings in `config.py`:
  - `DEFAULT_MODEL = "gemini-3.7-flash"` at :16;
  - 0.5 s / 224 px at :25-26;
  - 20 frames per sheet, 5 columns, 6 sheets per call, 1 sheet overlap;
  - `label_context` 1.0 s;
  - subdivide and refine only in `strict`.
- `JUDGE_MODEL = "gemini-3.1-pro-preview"` (`eval/judge.py:18`).
- LLM client: `REQUEST_TIMEOUT_S = 180`, JSON mime plus pydantic `response_schema`, empty candidates treated as `_Transient`, `lru_cache` on prompts, 12-hex SHA-256 prompt fingerprint stamped in `pipeline.py:238`.
- Prompts quoted verbatim are confirmed (`segment_v2.md:14-35`, `label_v2.md:13-27`).
- `results.md:35-40` table and the noise floor are confirmed. ADR-004 gives 0.561 vs 0.130.

**real2sim-frontier-assets**
- Every row of the §5 table matches `results/runs.csv` and `results/private_scene_summary.csv`. Wall times are `elapsed_seconds`/60: 56.95, 27.5, 9.15, 36.65, 60.1, 26.1, 16.3, 30.3, 25.4, 21.9 and 40.05 min.
- Token range 2,419,076–45,401,496, with every run having ≥91% cache reads.
- Drive parameters at `neutral_corkscrew.py:129-135` and "90 steps at 120 Hz" are confirmed.

**Physics post**
- Versions: Isaac Sim 6.1.0.0 CPU TGS, Newton 1.5.2 / MuJoCo-Warp 3.11.0, MuJoCo 3.11.0, Genesis 1.4.1.
- 126.422–126.472 mm travel; 1.06 N vs 10.46 N (Genesis 1.50 N).
- Real-time factors 15.60 / 0.94 / 0.24 / 0.11× (the "Multiple objects" scene).
- 131,072 worlds: 35.19 vs 26.38 M world-steps/s.
- Take-home quote is verbatim.

### Corrections (8, all also fixed inline)
1. **§3.1 diagram.** `obs["policy"] = {scene_rgb, scene_depth, scene_intrinsics, scene_cam_pos, scene_cam_quat, …}` is what the connector expects, not what RoboLab emits.
   - RoboLab's DROID env produces the groups `image_obs` / `proprio_obs` / `viewport_cam`. Examples: `policies/gr00t/client.py:221-224` reads `raw_obs["image_obs"]["over_shoulder_left_camera"]` and `raw_obs["proprio_obs"]`.
   - A grep finds no `scene_*` observation terms outside `connector.py` and its test. The obs group and runner that fed the pointing policy are unpublished.
2. **Labeler cost.** "Full labeler experiment program ≈ $18–19" is wrong. That figure is one batch, the 2026-08-30 "200 DKK top-up" (`research-log.md:288-293`).
3. **Retries.** "4 retries" is wrong. `max_retries=4` means 4 attempts (≤3 retries), with backoff `min(2**attempt + random(), 20)` s (`llm.py:123,205-236`).
4. **Quote attribution.** "The model proposes; the code decides" comes from `README.md:125`. `validate.py:1` says "this module decides".
5. **Dashboard label.** "policy family `gemini_pointing`" is wrong. `gemini_pointing` is the per-run policy label, and the family is "Gemini" (`dashboard/loaders/local.py:183-194`).
6. **Opus chair.** "Slid 2.4× further than the others" is imprecise: 2.4× Astra and 2.9× Fable.
7. **Scoring patches.** The UNVERIFIED "unclear how many Verified scoring patches" is now resolved: none.
   - The five `cli_*_robolab120` corpora are "Isaac Sim 6.0 + upstream scoring" (`dense_annotations.md:91,428-430,495-496,1151-1157`).
   - `verification.md:103`: the full 120-task patched re-run is "Planned; not yet scheduled".
8. **Baselines.** "Baselines to beat are Cosmos3 35.1% / π0.5 27.5%" is incomplete. The live leaderboard (2026-10-01) is led by FLUX 3 Action at 42.9%, then HiDream-O1-Embodied at 39.9%, Atomic-WAM at 39.6% and OASIS at 39.0%. It also lists two non-VLA comparators for an LLM harness: Phoenix (TAMP+FM, RGB-D) at 34.4% and VoLo (Agent, RGB-D) at 28.2%.

### Unverifiable / weakly supported
- **Gemini can "visuals broken".** Manda's text does not say this. It is consistent with `assets/can-gemini/preview.png`, which shows a photo-collage texture wrapped on the can (my visual check).
- **"Likely bootstrapped"** is speculation. No source covers funding either way.
- **Klarpul as a "Real2Sim asset showcase"** is inference. The page states only the specs (MJCF+USD; MuJoCo/Genesis/Isaac Sim; "Scripted suction and debris pickup").
- **The claude.ai artifact** still returns 403. The X timeline was not checked beyond the two syndicated posts.
- **Gemini pointing harness details** remain unknown: the model ID, prompt, phase vocabulary, controller law, and which Cartesian-delta interface produced the 0/4.
  - Upstream RoboLab ships `DroidRelIKActionCfg` (6-D relative, DLS IK, `scale=0.5`) and `DroidIKActionCfg` (7-D absolute pose), registered by `robolab/registrations/droid/auto_env_registrations_{rel,abs}_ik.py`.
  - The delta run plausibly used the relative config, but this is UNVERIFIED.

### Missed details (added)

**1. The 6,000-episode study used upstream scoring on a stack the fork itself flags.**
- **Towing artifact.** The fork found a pad-collider defect: a 6 mm Robotiq pad slab lets thin features hook behind the finger. It occurs in 563/7,186 (7.8%) tier-A episodes across the six `cli_*` corpora. Per policy: gr00t 10%, π0.5 8%, π0.5 rerun 10%, g05 7%, cosmos3 7%, molmoact2 4%. Tier-A episodes succeed 7.6% of the time vs 21.7% for the rest. Manda's own words: "Per-policy comparisons of grasp counts, drops, attempts and success on these corpora are contaminated unevenly by this artifact" (`docs/verified/towing.md:63-96`).
- **Isaac Sim 6.0 rendering.** It "renders the Robotiq gripper as detached fragments in some scenes (a policy input), observed in three of five sampled tasks". This is listed among findings.md's "Known defects", and is absent on 5.1. The blog's π0.5 rerun was forced by a "score-independent rendering issue". `cli_pi05_robolab120` has video for only 240/1,200 episodes (`dense_annotations.md:787-788`).
- **Wrong-object rate.** The blog's "wrong-object rate" equals the upstream `WRONG_OBJECT_GRABBED` event incidence exactly (`data.js` `event_episode_rates_pct`, e.g. π0.5 53.92 = 53.92). The fork's own findings say upstream grab events are contact-only (B1). They say `WRONG_OBJECT_GRABBED` read the keyword `"conditions"` as targets: 129 bogus events across ≥9 tasks (B4). They also say 19/120 tasks "could not name their own target, so every grasp of the correct object was flagged as a wrong object" (`docs/verified/README.md:36-37`). So the 41–54% figures are inflated by harness bugs. The blog itself calls the metric "interaction incidence".
- **H7 disagreement check.** Upstream's log contradicts Manda's state-based annotator in 3,652/6,000 episodes (60.9%). The rate is the same for every policy (58–64%), "so it is the log, not the policies" (`dense_annotations.md:1151-1185`).

**2. A dense state-based failure taxonomy over all 6,000 episodes** (P117, `dense_annotations.md:1112-1149`, proxy contact, validated on only 6 short gold episodes for 4 of the 5 policies).

| Metric | Cosmos3 | G0.5 | GR00T | MolmoAct2 | π0.5 |
|---|---|---|---|---|---|
| Picks per success | 15.8 | 49.0 | 39.8 | 27.9 | 20.3 |
| Place pass | 30% | 12% | 19% | 18% | 24% |
| Uncommanded release | 31% | 13% | 12% | 23% | 20% |
| Time in no_completed_subtask | 22% | 36% | 45% | 37% | 35% |
| Median first contact | 6.9 s | 5.8 s | 2.9 s | 5.8 s | 5.4 s |

- Wrong-object *picks* are 35–47%.
- "g05 and gr00t both score 10% and fail at opposite ends of the pipeline": GR00T has grasp failures, G0.5 has placement failures.
- Blog extras:
  - π0.5's lift height does not separate success from failure (−0.8 cm vs +5.9 to +13.3 cm for the others).
  - MolmoAct2, GR00T and G0.5 are "within 4.6 points of one another on every cut".
  - On the counting slice π0.5 leads, 68.6% vs Cosmos 54.3%.
  - The private notes once held "840 reviewed episodes across four policies" (`dense_annotations.md:90-93`); the blog later reports 1,150.

**3. VLM-pointing caveats beyond those listed.**
- The 0/4 vs 6/6 contrast is significant despite small n (Fisher exact p = 1/210 ≈ 0.005), but it is not a clean A/B: unequal n, an unknown delta interface, earlier harness revisions, and verification marked "RUNTIME (fork)", i.e. in the other repository merged in commit `16f32df`. It is not reproducible from public code. The README says the controller "is released as a separate package", but the repo and PyPI both return 404.
- **TCP inconsistency, important for any LLM pose-targeting harness.** Three numbers coexist:
  - the connector adds 0.1034 m and calls it "Franka flange origin to the fingertip" (the Franka-Hand value);
  - RoboLab's DROID `ee_pos` is the Robotiq `base_link` (`droid.py:279-293,478-480`);
  - `droid.py:419-421` says "Robotiq 2F-85 max height base flange -> fingertip is 162.8mm";
  - recordings put the TCP about 15 cm ahead of `base_link`.

  Fed RoboLab's `ee_pos`, a 0.1034 m offset would aim about 5–6 cm short of the fingertips. Either the private harness uses a different `ee_pos`, or this is a latent bug (UNVERIFIED which).
- The arm controller is Isaac Lab's high-PD reference (`disable_gravity=True`, PD 400/80, EEF offset (0,0,0); findings C2). π0.5 must be served as `pi05_droid_jointpos`; OpenPI's `--env DROID` "serves delta actions, and the arm wanders" (`policies/README.md:24-26`). Action-space mismatch is a real failure mode for any new harness.

**4. Real2Sim protocol details.**
- Only the can runs received `launch-paced.md` (candidate at ~20 min, stop at ~60 min). Corkscrew and chair used `launch-unpaced.md` (`prompts/README.md:1`). Wall times are therefore not comparable across objects.
- `object.json` names the exact product: "Coca-Cola Zero Sugar, 12 fl oz (355 mL) aluminium can, US market, sealed and full". The can's mass and dimension accuracy reflects product knowledge, not visual inference. On the measured objects, every agent overestimated the chair mass by 55–85% (11.5–13.8 vs 7.44 kg) and Astra overestimated the corkscrew by 68% (0.24 vs 0.143 kg).
- Gemini runs served `gemini-3-flash-preview;gemini-3.8-flash`, not 3.8 only.
- Codex costs are estimated from "Codex rollout token counts"; Claude and Gemini costs are "CLI result" (`runs.csv` `cost_basis`). Cost comparisons mix the two bases.
- The task targets "Isaac Sim 6.0 (PhysX 5)".
- In the private corkscrew evaluation, Fable's passive wing moved 81.6° and "Astra/Gemini coupling rejected by PhysX at load".
- Opus 5.5 and GPT-6 Sol were not run on the corkscrew, so "only Fable got the coupling" is among three agents.

**5. Labeler details useful for an LLM verifier.**
- Prompt line: "Those stamps are the only valid source of timestamps" (`segment_v2.md:5`).
- Media goes before text, but contact-sheet images follow the text (`llm.py:165-183`).
- Code comments say gemini-3.x deprecates `temperature` and recommends `None`, yet the default is 0.0 (`config.py:74`, `llm.py:193-195`).
- A state-input mode, segmentation from per-frame robot state (`s_state`), scored F1 0.660 at $2.32 per video-hour.
- The best WGO-protocol score is `f1_wgo` 0.476 vs Macrodata's published 0.306.
- At current gemini-3.7-flash list price ($0.75 / $3.75 per M tokens), `balanced` costs $4.12 per video-hour (`results.md:70-74`).
- ADR-001: no human review loop, "We cannot match a human-in-the-loop vendor on boundary accuracy and must not claim to."

**6. Physics post limits.** "Unless stated otherwise, each configuration is one matched trial." There is no hardware ground truth. In the controller-tuning section, "Task success transfers here, but trajectory quality does not."
