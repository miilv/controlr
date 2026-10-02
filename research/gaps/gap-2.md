# Gap 2: Pointing and grounding accuracy, in millimetres, on robot camera images

*Research date 2026-10-02. All numbers below were read from the cited primary source on that date unless marked UNVERIFIED or [derived].*

**Execution status.** Task 1 (compile every published number) is done, and the protocols behind those numbers were audited. Tasks 2–5 (a 300-target millimetre evaluation) were **not run**. Four things blocked them:
- there are no Gemini or OpenAI keys in this environment;
- there is no GPU, so PhysBrain or MolmoPoint cannot run locally;
- the root disk sat at 100% (0–229 MB free) during the session;
- no human was available for click annotation.

The only Claude credential present is the session's own harness token, and it was not used for benchmarking. Section 5 is a ready-to-run protocol with exact request shapes, costs and statistical power.

## 0. Bottom line

1. **No published pointing number exists for Claude Opus 5.5, Gemini Robotics-ER 2 or ER 1.6.**
   - ER 2's model card says only "See Figure 1-4 in the release post". Those four figures cover success detection, ERQA, instrument reading, progress, orchestration and safety. None of them is pointing.
   - The ER 1.6 blog has a pointing chart with no values.
   - Anthropic's Opus 5.5 page has no pointing, spatial or ScreenSpot row.
2. **Measured in millimetres, no model has ever been evaluated on a robot cell with real-robot ground truth.** The closest data is arXiv 2609.28184:
   - 151 tabletop RGB-D scenes, 1920×1080 frames.
   - Direct VLM localization error, median: GPT-5.4 **16.8 px** (RMSE 48.6); Claude Sonnet 4.6 **29.3 px** (RMSE 70.9).
   - That is about **7–15 mm** and **13–26 mm** at 0.5–1 m [derived].
   - The measurement floor, set by the reference definition, is about 14–16 px.
3. **Every benchmark in the debate scores point-in-mask hits, not distance.**
   - Masks are object-, part- or free-space-sized, so the scores bound *semantic* grounding, not millimetre precision.
   - In robot-cell terms, at 0.5 m on a D435 a 5 mm radius is only **14 px**.
4. **The "Claude −22 on RefSpatial" headline is protocol-confounded.**
   - The public PhysBrainEvalKit API path downsizes images to `image_max_side=336`.
   - It asks API models for **0–1 normalized** coordinates on all ten point benchmarks.
   - It has no Claude backend.
   - It always sends `temperature`/`top_p`, which the native Claude API rejects on Opus 5.
   - So the actual Claude configuration is unreleased (UNVERIFIED). If the public defaults were used, they contradict Anthropic's docs ("Claude does not work well when you ask for normalized coordinates").
5. **On the same table, Opus 5 leads on parts, trails on relational and free-space pointing.**
   - It leads every frontier model on Part-Affordance (78.1 vs Astra 55.0 and Gemini 3.6 Flash 64.7).
   - It is within 3.3 points of the best frontier model on PointBench, RoboAfford, RoboRefIt, VABench-Point and RoboSpatial-Home.
   - It trails by 8–22 points on PIOBench, Where2Place, PixMo-Points and RefSpatial.
   - Mean over the nine pointing sets [derived]: Astra 74.6, Gemini 3.6 Flash 74.4, Embodied-R1.5 71.5, PhysBrain 1.5 70.8, Opus 5 70.1.
6. **GroundingPI-4B cannot be "run locally" today.**
   - Its GitHub repo holds only `LICENSE` (Apache-2.0).
   - The Hugging Face repo `GroundingPI/GroundingPI` (created 2026-09-30) holds only a 28-byte `README.md`.
   - These local models do have public weights: PhysBrain 1.5-8B (no licence tag), MolmoPoint-8B (Apache-2.0) and Embodied-R1.5-8B (Apache-2.0).
7. **Design consequence.** For 1–8 mm contact work, a VLM point is a *seed*, not the target. The millimetre has to come from mask/depth/CAD refinement and relative visual servoing (piper-astra-jev: 0.6 mm) or from contact. Choose the pointer by the Section 5 eval. Until then, do not route `part_hint` away from Claude.

## 1. Published pointing numbers (with effort settings and splits)

### 1a. PhysBrain 1.5 (arXiv 2609.14973), Table 4: the only table containing Claude

**Settings, quoted.** "We evaluate the proprietary models using the lowest available official thinking setting: minimal thinking for Gemini, low thinking for GPT, and adaptive thinking with low effort for Claude."

**Metric.** A unified micro-F1 point-in-mask score (§B.1.2). For single-point, single-target items it equals hit rate. RefSpatial pools the Location, Placement and Unseen splits (100 + 100 + 77 = 277 items; the kit loads all three).

| Benchmark | Gemini 3.6 Flash | GPT-6 Astra | Claude Opus 5 | PhysBrain 1.5-8B | Embodied-R1.5-8B | Δ Opus vs best frontier |
|---|---|---|---|---|---|---|
| Part-Affordance | 64.7 | 55.0 | **78.1** | 84.0 | 83.4 | **+13.4** |
| PIOBench | 80.9 | 79.7 | 71.6 | 68.3 | 61.5 | −9.3 |
| PixMo-Points | 74.8 | 75.2 | 56.7 | 62.2 | 65.3 | −18.5 |
| PointBench | 69.6 | 71.9 | 68.6 | 64.3 | 64.5 | −3.3 |
| RefSpatial-Bench | 76.4 | 78.0 | 56.3 | 50.9 | 55.2 | −21.7 |
| RoboAfford | 83.4 | 84.6 | 82.3 | 80.4 | 76.9 | −2.3 |
| RoboRefIt | 85.9 | 85.1 | 83.5 | 89.6 | 86.1 | −2.4 |
| VABench-Point | 60.7 | 65.3 | 64.7 | 65.2 | 75.7 | −0.6 |
| Where2Place | 73.4 | 76.9 | 69.0 | 72.1 | 75.0 | −7.9 |
| RoboSpatial-Home* | 71.0 | 73.7 | 72.0 | 73.9 | 72.0 | −1.7 |

\*RoboSpatial-Home mixes point F1 with binary-question accuracy.

**Significance [derived].**
- RefSpatial: n = 277, so the 21.7-point gap is z ≈ 5.6. It is real *under this protocol*.
- PointBench: n = 982, so the 3.3-point gap is z ≈ 1.6. It is not significant.

### 1b. Other primary sources

| Source (settings) | Numbers |
|---|---|
| Gemini Robotics 1.5, arXiv 2510.03342, Table 19 (GPT-5 via API, Sept 2025) | **Point-Bench**: GR-ER 1.5 thinking 71.6, no thinking 73.3; original GR-ER 75.7; Gemini 2.5 Pro 62.7; GPT-5 43.6<br>**RefSpatial**: 48.5 / 41.8 / 49.3 / 33.6 / 23.5<br>**Where2Place**: 59.0 / 48.0 / 41.0 / 37.0 / 37.0<br>**RoboSpatial-Pointing**: 31.1 / 25.3 / 30.3 / 8.3 / 19.0<br>Metric is in-mask %. Table 21 lists Gemini 2.5 Pro Where2Place as 22.0, an internal inconsistency |
| GroundingPI, arXiv 2609.39601, Tables 2 and 29 (Astra at **High** effort; locally run baselines averaged over 3 runs, GroundingPI over 10) | **RefSpatial Location / Placement / Unseen / RoboSpatial-Context**: Astra 86.00 / 85.86 / 81.93 / 65.69; GroundingPI-4B 76.00 / 75.00 / 75.32 / 73.77; Qwen3.7-Max 71.50 / 66.00 / 57.14 / 69.67; RoboRefer 51 / 49 / 39<br>**ScreenSpot-Pro**: Astra 93.17 vs GroundingPI 65.78<br>Headline average over 34 benchmarks: GroundingPI 73.68 vs Astra 71.54<br>**No Claude or Gemini 3.x rows** |
| MolmoPoint, arXiv 2603.28069, Table 1 (Point-Bench leaderboard values) | ER 1.5 67.1; Gemini 2.5 Pro 62.8; Molmo2-8B 68.7; **MolmoPoint-8B 70.7** |
| MolmoPoint, Table 2 (PixMo-Points F1, authors' own runs) | GPT-5.2 31.6; Gemini 3 Pro 77.8; MolmoPoint-8B 89.2 |
| Embodied-R1.5, arXiv 2606.11324, Table 2 (EmbodiedEvalKit; RefSpatial all splits) | Embodied-R1.5: RefSpatial 54.2, PointBench 74.6, Part-Afford 82.9, Where2Place 74.0<br>**ER 1.5: 39.7 / 70.7 / 27.0 / 48.3**<br>GPT-5.4: 15.7 / 37.9 / 17.2 / 34.7 |
| RoboPoint, arXiv 2406.10721, Table 2 | Where2Place: RoboPoint 46.77 vs GPT-4o 29.06<br>RoboRefIt: 49.82 vs 15.28 |
| PointArena, arXiv 2505.09990 | Point-Bench has 982 pairs, binary in-mask scoring, first point only<br>Chain-of-thought *hurt* pointing: GPT-4o −2.9, Gemini 2.5 Flash 46.8 → 30.9 |
| Roboflow Vision Evals, Opus 5.5 page (box localization, not points) | Object detection mAP@50: 74.4% at low effort (#5 of 61); 76.8% at high effort (#4 of 27) |

**Missing cells.**
- Opus 5.5: none anywhere.
- ER 2: no Point-Bench, RefSpatial or Where2Place number, confirmed by checking the model card, release post, DeepMind page and API docs.
- ER 1.6: chart only.
- MolmoPoint: no RefSpatial number.
- GPT-6 Astra: no vendor pointing numbers; only third-party (PhysBrain low, GroundingPI high).

## 2. Why these scores cannot choose the pointer

1. **Hit-in-mask is not millimetres.**
   - PointArena's masks are SAM-initialized and grid-refined. The authors say this "often results in coarse and imprecise boundaries".
   - A point anywhere on a mug counts as a hit; so does any point on a 10 cm free-space patch.
   - None of these benchmarks reports pixel error.
2. **The protocol was probably hostile to Claude (UNVERIFIED for the actual Table 4 runs).**
   - `DeepCybo-PhysAI/PhysBrainEvalKit`, `core/api_engine.py`:
     - `encode_image_to_base64(image_input, target_size=336)` with `image_max_side: int = 336`, JPEG quality 85;
     - `detect_api_backbone()` supports only `gpt`, `gemini_robotics` and `gemini-2.5`;
     - the `gpt` path posts to `{base_url}/chat/completions` with `temperature` and `top_p`.
   - Every point benchmark's `gpt` prompt says "The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points" (`benchmark/refspatial.py:65`, and likewise in `pointbench.py`, `pixmo_points.py`, `partafford.py`, `where2place.py`, `vabench_point.py`, `pio.py`).
   - The upstream `pickxiguapi/EmbodiedEvalKit` has the same 336 px default.
   - `AGENTS.md` scopes the public kit to "the public Qwen3-VL benchmark suite". The closed-model configuration was not released.
   - At 336 px, one pixel on a D435 frame is 2.05 mm at 0.5 m [derived].
   - Because the format was uniform across benchmarks while Claude's deltas range from −22 to +13, normalized output alone does not explain the pattern. Small-target and relational sets (RefSpatial, PixMo) suffer most.
3. **Effort confound.**
   - In GR-ER 1.5, thinking *lowered* Point-Bench (73.3 → 71.6) but *raised* RefSpatial (+6.7) and Where2Place (+11.0).
   - Astra on RefSpatial: 78.0 at low effort (PhysBrain) vs 84.8 at high effort (GroundingPI Table 29, item-weighted over 277 [derived]). These are different harnesses.
   - Claude's largest deficit sits on exactly the benchmark where low effort hurts.
4. **The same model moves 5–9 points between harnesses.**
   - ER 1.5 RefSpatial: 48.5 (Google) vs 39.7 (EmbodiedEvalKit).
   - ER 1.5 Point-Bench: 71.6 or 73.3 vs 67.1 (leaderboard) vs 70.7.
   - The specialist ER 1.5 scored **27.0** on Part-Afford in Embodied-R1.5's evaluation. That is the only part-level ER number in existence.

## 3. The only metric data for frontier VLM localization on a robot tabletop

**arXiv 2609.28184, "VLMs Can Describe, But Not Measure" (Trento / Mondragón, 2026-09-23).**

*Setup.*
- 151 scenes, 26 objects, 1–5 objects per scene.
- Orbbec Femto Mega: RGB 1920×1080 (80°×51°, so fx ≈ 1144 px [derived]); depth 640×576.
- Reference: ArUco markers in a second, marker-augmented image.
- Error = bbox centre vs marker position.
- Sonnet 4.6 was run with `max_tokens=8192` via Azure; GPT-5.4 with seed 42. Effort was not stated.

*Direct VLM results (Table I).*

| Model | Recall | Position median (px) | Position RMSE (px) | Depth median (mm) | Depth RMSE (mm) |
|---|---|---|---|---|---|
| Sonnet 4.6 | 92.8% | 29.3 | 70.9 | 96.7 | 166.4 |
| GPT-5.4 | 95.9% | 16.8 | 48.6 | 151.6 | 241.0 |
| Qwen3-VL-4B | 79.2% | 221.7 | 238.1 | 413.9 | 519.5 |

*SAM3 + VLM label + RGB-D mask-median (Table II).*
- Position 15.2–16.1 px.
- Depth **11.4–11.6 mm median**, but RMSE 84–91 mm (outliers).
- SAM2-l reaches 13.6 px. That is effectively the floor set by "bbox centre vs marker".

*Reading.*
- GPT-5.4 is near the floor; Sonnet 4.6 is about 2× the floor with heavy tails.
- The paper does not say whether Sonnet 4.6's standard-tier server-side resize (1920×1080 → 1456×819) was rescaled. UNVERIFIED.
- No Opus 5/5.5, ER or Astra was tested.

**Simulation datapoint (Manda RoboLab-Verified, `docs/verified/README.md`).**
- An unnamed Gemini pointer plus a private geometric controller (`vlm-pinpoint`, 404) scored: BananaInBowl 6/6, BananasInCrate 5/6, OneBottleInSquarePail 2/6, FruitsOnion 1/6.
- Driven as Cartesian deltas, the same model scored 0/4.

**PointArena Point-Act.** xArm 6 Lite, 10 remote participants × 3 trials, three agents. Its "R² = 0.92" between Point-Bench and real success is fitted on **three** points.

## 4. Pixel-to-millimetre budget for the planned cell [derived]

Pinhole model, f = (W/2)/tan(HFOV/2). Specs are from realsenseai.com:
- D435 RGB: 1920×1080, 69°×42°, rolling shutter, depth <2% at 2 m.
- D405: 1280×720, 87°×58°, ideal range 7–50 cm, depth ±2% at 50 cm.

| Camera / image the model sees | mm per px at 0.3 m | at 0.5 m | at 1 m | Radius in px for 5 mm at 0.5 m |
|---|---|---|---|---|
| D435 1920×1080 (fx ≈ 1397), unresized by Opus 5.5 | 0.21 | 0.36 | 0.72 | 14.0 |
| Same, standard-tier resize to 1456×819 (Haiku 4.5, Sonnet 4.6) | 0.28 | 0.47 | 0.94 | 10.6 |
| Same, PhysBrain-kit 336 px | 1.23 | 2.05 | 4.09 | 2.4 |
| Same, Robocurve 224 px squash | 1.84 | 3.07 | 6.14 | 1.6 |
| D405 1280×720 (fx ≈ 674) | 0.44 | 0.74 | 1.48 | 6.7 |

**Claude image limits (vision docs).**
- High-resolution tier ("Claude 4.7 and later"): 2576 px long edge or 4784 visual tokens, where tokens = ⌈w/28⌉·⌈h/28⌉.
  - 1920×1080 costs 2,691 tokens and is not resized.
  - The design's 1920×480 tile costs 1,242.
- Standard tier: 1568 / 1568.
- Coordinates are returned in the resized image's pixel space. The guard is per image block: `"transformations": {"oversized_image": "error"}`. HARNESS_DESIGN writes it as a bare `"oversized_image"`; fix that.
- Computer-use and zoom tool-result images are *never* auto-resized; they are rejected instead.

**ER 2 quantization.** 0–1000 units are 1.92 px on a 1920-wide frame, about 0.7 mm at 0.5 m, which is negligible. ER 2's internal image resolution is undocumented (UNVERIFIED).

**Error budget.**
- A GPT-5.4-class 16.8 px error is about 6 mm at 0.5 m on the D435. That misses a 5 mm radius and passes 10–20 mm.
- D405 depth tolerance alone is ±10 mm at 50 cm. Axial error can therefore exceed the lateral budget.
- Pointing must feed a mask- and geometry-refinement step. Use patch-median depth, mask snapping, then a plane, CAD or edge fit.

## 5. Protocol to run (the deciding measurement)

**Data.** The fallbacks named in the gap brief cannot give millimetres:
- Robocurve Tier 1 captured 640×480 but sent 224×224 squashed frames. No depth or intrinsics are published, and no depth appears in any transcript.
- The DROID RLDS release is 180×320 with no depth or calibration fields. The raw stereo HD release is 8.7 TB.

Usable sources, in order:
1. **Ilia's cell.** Ground truth by TCP touch-probe: the robot touches the target and records it directly in the base frame. Cross-check with ChArUco plus D435/D405 deprojection.
2. **RoboRefIt.** 640×480 *aligned* RGB-D, 10,872 images with instance masks. Intrinsics are undocumented (UNVERIFIED).
3. **The 2609.28184 scenes**, if released (UNVERIFIED).

**Set.**
- 300 targets: 75 each of object centroid, graspable part (handle, rim), slot or groove, and place point.
- Split across overhead D435 at 0.5–1.0 m and wrist D405 at 0.15–0.4 m.

**Queries.**
- **Claude:**
  - `claude-opus-5-5` at effort `low` and `medium`. Thinking cannot be disabled; the default effort is `medium`.
  - Unresized 1920×1080 PNG with `transformations.oversized_image:"error"`.
  - Prompt: "Return the point as `[x, y]` integer pixel coordinates."
  - Use structured outputs via `output_config.format`. Forced `tool_choice` returns a 400 on Opus 5.5.
- **Gemini:**
  - `gemini-robotics-er-2-preview`, `[y, x]` normalized 0–1000, `thinking_level` `low` and `medium` (the docs recommend medium).
  - k = 1 and k = 3, aggregated by both mean and median.
  - It needs a *restricted* API key; unrestricted keys get a 403.
- **GPT-6 Astra:** the native point convention is undocumented here, so test both absolute pixels and 0–1.
- **Local models:** PhysBrain 1.5-8B, MolmoPoint-8B and Embodied-R1.5-8B, each in its own point format. GroundingPI waits for its weights.
- **Variants:** plain; zoom crop (offset back to the full frame); Set-of-Mark ids on SAM3 masks; a 5 cm metric grid overlay.

**Metrics.**
- Pixel error, and 3D error from both 5×5 patch-median depth and mask-snapped depth.
- Hit rate within 5/10/20 mm.
- Parse failure, refusal and no-point rates.
- Latency (p50/p90) and dollars per query.

**Power [derived].**
- n = 300 gives a ±5.7-point 95% CI at p = 0.5.
- A paired McNemar test detects a 10-point difference with about 155–194 targets and 8 points with about 243, at 80% power with 20–25% discordance.
- A 5-point difference needs about 470 targets.

**Cost [derived].**
- Opus 5.5 per query: about 3k input tokens × $4/M plus 0.1–0.6k output × $20/M, so about $0.015–0.025.
- 300 targets × 4 variants × 2 efforts: about $35–60.
- ER 2 costs less per call (the price table is in the landscape note; Gemini image-token counts are UNVERIFIED here).

## 6. Corrections to carry into HARNESS_DESIGN

1. **R3, §3.3 and §4.2.** Replace "route parts and grooves to ER 2 / GroundingPI" with: "Default to Opus 5.5 points in absolute pixels on unresized frames, with a mandatory `zoom` tool and geometric refinement. Run ER 2 (k = 3) as an A/B arm; decide by the Section 5 hit rate at 5 mm." Part-level evidence favours Claude: 78.1 vs Astra 55.0 and Gemini 3.6 Flash 64.7, and ER 1.5 scored 27.0 in a different kit.
2. Remove "GroundingPI-4B (run locally)" until weights ship. Use PhysBrain 1.5-8B, MolmoPoint-8B or Embodied-R1.5-8B instead.
3. Fix the field name to `transformations.oversized_image`.
4. Never pass a VLM point directly as a ≤5 mm contact target. Seed SAM3, take the mask, run the depth or CAD fit, then close the last millimetres with relative servoing or contact.

## Sources

- PhysBrain 1.5, Table 4 and §B.1.2: https://arxiv.org/html/2609.14973v1
- PhysBrainEvalKit (`core/api_engine.py`, `core/inference.py`, `benchmark/*.py`, `AGENTS.md`): https://github.com/DeepCybo-PhysAI/PhysBrainEvalKit
- EmbodiedEvalKit `core/api_engine.py`: https://github.com/pickxiguapi/EmbodiedEvalKit
- GroundingPI, Tables 2, 4, 29 and §12: https://arxiv.org/html/2609.39601v1
- GroundingPI code and weights: https://github.com/groundingpi/GroundingPI · https://huggingface.co/GroundingPI/GroundingPI
- Gemini Robotics 1.5, Tables 19 and 21, Fig. 10: https://arxiv.org/html/2510.03342v1
- MolmoPoint, Tables 1 and 2: https://arxiv.org/html/2603.28069v1 · https://huggingface.co/allenai/MolmoPoint-8B
- Embodied-R1.5, Table 2: https://arxiv.org/html/2606.11324v1 · https://huggingface.co/IffYuan/Embodied-R1.5-8B-SFT
- RoboPoint, Table 2: https://arxiv.org/html/2406.10721
- PointArena (Point-Bench, Point-Act, CoT ablation): https://arxiv.org/html/2505.09990
- RefSpatial-Bench card (splits 100/100/77, metric): https://huggingface.co/datasets/BAAI/RefSpatial-Bench
- VLMs Can Describe, But Not Measure: https://arxiv.org/html/2609.28184v1
- Orbbec Femto Mega specs: https://www.orbbec.com/documentation-mega/femto-mega-hardware-specifications/
- Gemini Robotics ER docs: https://ai.google.dev/gemini-api/docs/robotics-overview
- ER 2 model card: https://deepmind.google/models/model-cards/gemini-robotics-er-2/
- ER 2 DeepMind page: https://deepmind.google/models/gemini-robotics/gemini-robotics-er/
- ER 2 release post: https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/
- ER 1.6 blog: https://deepmind.google/blog/gemini-robotics-er-1-6/
- Claude vision and coordinates docs: https://platform.claude.com/docs/en/build-with-claude/vision · https://platform.claude.com/docs/en/build-with-claude/vision-coordinates
- Opus 5.5 launch page: https://www.anthropic.com/claude-opus-5-5
- Roboflow Vision Evals, Opus 5.5: https://playground.roboflow.com/models/anthropic/claude-opus-5-5
- RealSense D435 and D405 specs: https://www.realsenseai.com/products/depth-camera-d435/ · https://www.realsenseai.com/products/stereo-depth-camera-d405/
- DROID dataset format: https://droid-dataset.github.io/droid/the-droid-dataset
- RoboRefIt / VL-Grasp: https://arxiv.org/pdf/2308.00640 · https://github.com/luyh20/vl-grasp
- Local: `repos/manda-RoboLab-Verified/docs/verified/README.md`; `sources/robocurve-gpt6-astra.md`
