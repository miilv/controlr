# Gap 5: Accuracy of the independent success verifier that gates `finish(success)`

*Run on 2026-10-02. Primary data:*
- *335 labelled episodes: 235 real bimanual YAM episodes from two Robocurve reports, plus 100 RoboDojo-sim episodes;*
- *1,782 verifier calls made for this note.*

*Per-episode verdicts are in `gaps/gap-5-verifier-results.csv`. The runner is `gaps/gap-5-verify.py`. Frames and logs are in `/dev/shm/gap5/`, because the root disk was full.*

## 0. Bottom line

**The verifier is reliable only where vision is reliable.**

- Setup: Opus 5.5 at `effort:"low"`, reading a Manda contact sheet plus the full-resolution final frame, on 235 real episodes.
- Overall: balanced accuracy (BA) 0.871 [95% CI 0.808–0.923]. False-success rate (FSR) 13.3% (26/195). AUROC 0.969.
- Non-contact half (block-in-bowl, store-in-safe, sort, pack): **0/92 false accepts**, BA 0.968.
- Contact/alignment half (puzzle groove, stack bowls, cap pen, stand bottles): **BA 0.707, FSR 25.2% (26/103)**. Every false accept came from this half.
- This reproduces FailBench's contact gap, now measured on Claude.

**A different vendor does not de-correlate errors.**

- GPT-6 Astra as verifier: BA 0.886 [0.828–0.937], FSR 10.3%. That is statistically the same as Opus 5.5, at 2× the price ($0.035 vs $0.017 per check).
- GPT-5.6 Sol: BA 0.869.
- Opus 5.5 and Astra agree on 96.6% of real episodes (κ 0.91) and share 75% of their errors (error-set IoU 0.75). FailBench's panel mean was 0.28.
- A 2-of-3 vote gives BA 0.891 and FSR 11.8%.

**What lowers false accepts is a different *view*, not a different vendor.**

- Opus 5.5 on the end frame alone, AND Astra on the contact sheet plus end frame: **FSR 2.1% (4/195)**.
- Cost: it **rejects 22.5% of true successes (9/40)**. A unanimous 3-model gate gets 5.6% at the same 22.5% false-reject rate.
- Caveat: this pairing was chosen post hoc, from 10 pairs tested on the same data.

**The gate repeats the agent's mistakes on exactly the claims that matter.**

- Tier 1 has 22 coded false success claims; 19 are Stack Bowls trials graded 0.75 ("overlapping", not "stacked").
- The Opus 5.5 verifier accepted **19/22**, Astra 18/22, the unanimous gate 10/22.
- At 224 px the verifier sees what the agent saw.

**M2 ("overclaim < 5%, verifier-gated") is reachable only by giving up most true successes on ambiguous tasks.**

- Baseline: an Opus 5.5 agent makes false success claims on 10/119 failed Tier 1 trials (8.4%).
- With an Opus 5.5 verifier, the rate stays at **5.9% (7/119)**.
- With the EF-Opus AND Astra gate it falls to 0.8% (1/119), but that gate also rejects 4/6 true Tier 1 `done`s.
- Every verifier rejected Opus 5.5's only completion, `118b151b`.

**Input format and effort.**

- End-frame-only Opus mistook the puzzle board's painted blue groove for a seated piece in 13/56 failed puzzle trials. The contact sheet cut that to 6/56.
- The contact sheet adds *process inference* instead. It accepted 20/20 Stack Bowls trials graded 0.75, against 3/20 with the end frame only.
- `effort:"high"` did not help. On the 175 hardest episodes BA was 0.724 vs 0.754 at low effort, only 5/175 verdicts changed, and output tokens were 2.9×.

**A no-vision gripper predicate is a strong partner on single-object pick-place.**

- The predicate: "held (aperture 0.05–0.9, effort > 0.7), then ended open".
- On the Astra report it passed 31/32 successes and rejected 64/88 failures.
- AND Opus 5.5: BA 0.942, false accepts 2/88.
- The same thresholds passed 0/5 true Tier 1 bowl-stack completions. Calibrate per object.

**Not measured:**

- ER 2: no Gemini key, and the proxy's Gemini routes reject images.
- Robometer-4B: no GPU, about 4 GB free RAM.
- The 3-rater blind re-scoring of RoboDojo-Real.
- Ilia's YAM trials: none exist yet.

ER 2's 82.4 and Robometer's F1 0.81 remain vendor and author numbers.

## 1. Published evidence

| Source | Numbers | Caveat |
|---|---|---|
| **FailBench**, arXiv 2609.03611. 2,197 attempts, 14 sources, 13 detectors; video, or 32 even frames | Best macro BA 0.77 (Gemini 3 Flash; micro 0.74); Gemma-4-31B 0.75. `reassemble` (NIST board) slice mean 0.52, best 0.60. Top-3 majority vote 0.78 vs 0.76, and 0.55 on contact assembly. Mean error-set IoU over 78 pairs 0.28. Localize-and-crop 0.773 → 0.797 (223 fixed, 160 broken, McNemar p = 0.0015). Every failure-tuned specialist was below its own base model | No Claude, no GPT-5.x/6. The abstract says the success bias "persists even with increased reasoning effort". In the text this rests on trace length (Gemma wrong answers 1,868 vs right 1,464 characters), not an effort sweep |
| **DeepMind ER 2 page** | Video success detection: ER 2 82.4, Opus 5 81.0, ER 1.6 76.0, Gemini 3.6 Flash 75.4, GPT 5.6 Sol 74.7. Image: ER 2 87.7, Opus 5 83.6. Progress (5 buckets): ER 2 57.4, Opus 5 37.1 | Vendor chart. No n, no metric definition, no frame rate. The API exposes video success detection as moment-finding, returning `{"completion_time_seconds": null}` if the task is not completed |
| **Robometer**, arXiv 2603.02115 | Failure-detection F1 on 100 DROID trajectories (30 success / 70 failure, 7 tasks, unseen scenes): Robometer 0.81, RoboReward-4B 0.74, token uncertainty 0.48, GPT-5-mini 0.33 (perfect TNR, very low TPR), VLAC 0.16 | Authors' own evaluation. The `lerobot/Robometer-4B` card gives no accuracy and no VRAM figure |
| **RoboRMBench**, arXiv 2609.05401. 2,390 real trajectories, 21,673 paraphrases | Rewording the instruction flips success/failure. Flip rate: Sonnet 4.6 0.013–0.051, Haiku 4.5 0.166–0.199, Gemini 3 Flash 0.096–0.136. Reasoning *reduced* robustness for Sonnet 4.6 | Progress scoring; no Opus 5.x |
| **Gemini Robotics 1.5**, arXiv 2510.03342 | Success-detection failures were 6% of long-horizon episodes with 2.5 Flash vs 4% with ER 1.5. At 5 Hz, "stale success predictions quickly become irrelevant" | Accuracy shown only in a figure |
| **Pigey** `real/agent.ts:347` `runVerifier()` | Context-isolated Claude call, `max_tokens: 200`, returns `{"ok","reason"}` after each DropAbove/VLARollout/Release and at Done | **Fails open** on API or parse errors |

No prior measurement existed of Opus 5.5 or Astra as a *separate* verifier on real robot video.

## 2. Corpus and protocol

| Corpus | Episodes | Successes | Label | Pixels given to the verifier |
|---|---|---|---|---|
| **A**: Robocurve "GPT-6 Astra on robotic manipulation". Agents: Astra, Fable 5.1, Fable 5 | 120 (bowl 60, puzzle 60) | 32 (bowl 28, puzzle 4) | Published stage = 4 | Published MP4 at 1344×448: three 448 px views, 10 fps |
| **B**: Robocurve "Opus 5.5 on RoboDojo-RC Tier 1" | 115: all 113 `done` terminations plus the 2 `give_up` completions | 8 | Rubric score = 1.0 | Left/top/right MP4s at 224×224, the frames the agent saw |
| **C**: `YuMoool/astra-robodojo-rollouts`, RoboDojo **simulation**, Astra direct/hybrid | 100 | 37 | `evaluation_success` | Saved previews, 480×360 per camera, read by HTTP range from the Hugging Face zips |

**Label provenance**
- All 115 Tier 1 transcripts carry operator judgement "n/a", so those scores were assigned after the run.
- In corpus A, the published puzzle success `rig-4_fd86a832` was live-judged "partial".
- Both reports state that grading was "operator-judged with the model known".
- The real **contact subset** is the A puzzle plus Tier 1 Cap Pen, Stack Bowls and Stand Up Bottles: 112 episodes, 9 successes.

**Inputs**
- **EF**: the final frame, all cameras side by side, at native resolution.
- **CS**: a contact sheet built like `manda-robot-episode-labeler/src/rel/video/contact_sheet.py`: 20 frames, 5 columns, 3 px pad, 18 px dark gutter, yellow timestamp. Frames are evenly spaced over the episode, and the last tile is stamped FINAL.

**Prompt**
- The system prompt is independent of the agent. It contains Manda's rule verbatim: "Judge failure by what you can see. Do not infer success from the task description."
- It adds: "An object that is near, touching, or resting on the edge of its target is not 'in' or 'on' it. A held object is not a placed object." It also asks for `uncertain` when the decisive region is not visible.
- The user turn gives the instruction, the success criterion (for Tier 1, the top rubric line), the rubric or stages, and a description of the images.
- Output is JSON: `{final_state, decisive_region_visible, verdict, p_success, progress}`.

**Fail-closed scoring**: only `"success"` accepts; `uncertain` counts as a reject.

**Models**, all through the user's OpenAI/Anthropic-compatible proxy (omniroute):
- `claude-opus-5-5` on Messages, `output_config:{effort:"low"}`. Thinking is adaptive and cannot be disabled. `effort:"high"` was also run on the 175 hardest episodes (A puzzle plus all of B).
- `gpt-6-astra` on chat completions with `reasoning_effort:"low"`. Whether the proxy forwards that parameter is UNVERIFIED.
- `gpt-5.6-sol-low`.

There was one call per episode per condition. Opus CS-only was not run on corpus C.

## 3. Results

**Real episodes (A + B): n = 235, 40 successes, 195 failures**

| Verifier (fail-closed) | BA [95% CI] | TPR | FSR (k) | AUROC | FSR at FRR ≤ 5% | BA contact (112) | BA non-contact (123) | $/check |
|---|---|---|---|---|---|---|---|---|
| Opus 5.5 CS | 0.851 [0.786–0.911] | 0.825 | 12.3% (24) | 0.958 | 18.5% | 0.661 | 0.952 | 0.016 |
| Opus 5.5 EF | 0.846 [0.779–0.914] | 0.775 | 8.2% (16) | 0.945 | 26.2% | 0.645 | 0.935 | 0.009 |
| **Opus 5.5 CS+EF** | **0.871 [0.808–0.923]** | 0.875 | 13.3% (26) | 0.969 | 17.4% | 0.707 | 0.968 | 0.017 |
| Astra CS+EF | 0.886 [0.828–0.937] | 0.875 | 10.3% (20) | 0.967 | 14.9% | 0.736 | 0.968 | 0.035 |
| Sol CS+EF | 0.869 [0.806–0.929] | 0.850 | 11.3% (22) | 0.952 | 19.5% | 0.690 | 0.946 | UNVERIFIED |
| 2-of-3 (Opus, Astra, Sol) | 0.891 [0.836–0.939] | 0.900 | 11.8% (23) | – | – | 0.722 | 0.984 | – |
| 3-of-3 unanimous | 0.859 [0.787–0.924] | 0.775 | 5.6% (11) | – | – | 0.669 | 0.935 | – |
| Opus EF AND Astra CS+EF (post hoc) | 0.877 | 0.775 | **2.1% (4)** | – | – | 0.703 | – | 0.043 |

**Thresholding `p_success` does not rescue the gate.** At the threshold that keeps 95% of true successes, each verifier still passes **15–26% of failures**.

**Opus CS+EF by slice**
- A bowl: BA 0.982, 0/32 false accepts.
- A puzzle (4 successes): BA 0.696, 6/56 false accepts.
- Tier 1: BA 0.782, 20/107 false accepts.
- Sim C: BA 0.785 [0.699–0.866], but TPR only 0.649. In sim the verifiers mostly *reject true successes*: Astra 16/37, Sol 14/37. Partly this is because the success criterion was generic.

**Progress estimation** (rubric level or stage, 208 real episodes)
- Opus CS+EF: exact 61.5%, within one level 94.7%.
- Astra: exact 64.9%, within one level 97.6%.

**Gate on the agent's own claims**

Across A + B there were 157 `done`s: 36 true, 121 false. The Opus CS+EF verifier passed 32/36 true and 24/121 false.

On the 22 Tier 1 false success claims coded in gap-4:

| Verifier | Opus 5.5 agent: 10 false claims passed | Astra agent: 12 false claims passed (all Stack Bowls) | True Tier 1 `done`s passed (6) |
|---|---|---|---|
| Opus 5.5 CS+EF | 7 | 12 | 4 |
| Opus 5.5 EF | 1 | 2 | 2 |
| Astra CS+EF | 7 | 11 | 4 |
| 3-of-3 | 5 | 5 | 2 |
| Opus EF AND Astra | 1 | 2 | 2 |

Overclaim after gating, per failed trial for an Opus 5.5 agent:
- With an Opus 5.5 verifier: 7/119 = 5.9%.
- With the unanimous gate: 4.2%.
- With Opus EF AND Astra: 0.8%.

None of these differences is resolvable at n = 10 claims.

## 4. Why it fails (from the verdict texts)

1. **End-frame illusions.** The puzzle groove is painted blue.
   - EF-only Opus said "the blue round piece sitting inside the circular groove" at p = 0.70–0.85 while the piece lay on the table, in `a685676a`, `b90fd8b1`, `bb22303f` and `fb1601c3`.
   - The contact sheet showed the piece elsewhere, and CS+EF said "failure" in each case.
2. **Process inference.** With the contact sheet, the models reasoned that "the white and purple bowls are no longer visible elsewhere, so they are nested" (`570715d2`, `72a90a0a`, `75a90497`).
   - They accepted 0.75 "overlapping" stacks. Gap-4 found these indistinguishable from the 1.0 stacks at 224 px.
   - Manda's rule does not prevent this, because the inference comes from the trajectory, not from the task text.
3. **Occlusion.** When the gripper still holds the disc over the groove, verifiers correctly answer `uncertain` or `decisive_region_visible:false`. They said the same about 2 of the 4 true puzzle successes:
   - `d3d2906e` was still held at the end;
   - `fd86a832` was graded 4 but live-judged "partial".
4. **Label ambiguity caps measurable accuracy.** In `118b151b` (Store In Safe, graded 1.0) the monkey plush sticks out under the closed lid. All verifiers rejected it, and the agent itself doubted it. About 3 of the 40 real successes are arguable.

## 5. Sensors, agreement, effort, latency

**Sensor predicate** (corpus A only, read from transcript `state[eef_state]`/`state[joint_eff]`)
- "Held, then released" passed 28/28 bowl successes and rejected 27/32 bowl failures.
- On the puzzle it passed 3/4 successes and rejected 37/56 failures.
- AND vision: BA 0.942 with Opus (false accepts 2/88, false rejects 3/32) and 0.963 with Astra (1/88 and 2/32).
- On Tier 1 the same thresholds missed bowl grasps, because the gripper closes almost fully on a rim. Use per-object thresholds or a lift test.

**Agreement** (κ on accept/reject, real episodes)

| Pair | κ | Error-set IoU |
|---|---|---|
| Opus CS+EF vs Astra | 0.91 | 0.75 |
| Opus CS+EF vs Sol | 0.72 | 0.41 |
| Astra vs Sol | 0.71 | 0.36 |
| Opus EF vs Astra | 0.60 | 0.22 |
| Opus EF vs Opus CS | 0.56 | 0.22 |

Changing the input view de-correlates errors about as much as changing the vendor, and AND-ing across both works best. Opus EF AND Opus CS+EF, a single vendor, reaches FSR 4.6% (9/195) at TPR 0.75.

**Effort** (Opus 5.5 CS+EF, same 175 episodes: A puzzle plus B)

| Effort | BA | AUROC | False accepts | False rejects | Output tokens | $/check |
|---|---|---|---|---|---|---|
| low | 0.754 [0.616–0.882] | 0.902 | 26/163 | 4/12 | 181 | 0.018 |
| high | 0.724 [0.584–0.861] | 0.889 | 22/163 | 5/12 | 526 | 0.025 |

Only 5/175 verdicts changed. κ between the two settings was 0.90.

**Latency**
- Sequential calls through the proxy (n = 8 each), median: Opus 5.5 CS+EF 3.8 s (range 2.6–9.6), Opus EF 3.3 s, Astra 10.2 s, Sol 9.5 s.
- Under this run's parallel load, the median was 22–26 s for every model.
- These are proxy upper bounds, not API latencies (see Gap 1).

## 6. Implications for the design (R5, §3.10, §4.2, M2)

1. **Keep fail-closed, but stop relying on "a separate model" for de-correlation.** On identical pixels, Opus 5.5 and Astra make the same mistakes. De-correlate the **evidence** instead:
   - a scripted verification pose before `finish`;
   - a second camera view at ≥ 448 px, cropped to the target (FailBench's crop gave +2.3 pp);
   - per-skill sensor predicates: grasp then release, a lift test, and a contact/effort stall at insertion depth.
2. **Route by predicate class.**
   - `object_in` / containment / placement: one Opus 5.5 CS+EF check at low effort ($0.017, 0/92 false accepts on the non-contact slice), AND the gripper predicate.
   - Seating, stacking, capping, upright: vision is advisory only (BA about 0.7). Require a geometric or force signal. Otherwise return `uncertain` and request the verification view.
3. **Send both a contact sheet and the full-resolution final frame.** Add to the prompt: "Judge the end state from the FINAL tile and final frame only; earlier tiles only help locate objects." Then re-measure; this fix is untested.
4. **Do not tune the false-accept/false-reject trade-off with `p_success` thresholds.** Their calibration is too coarse. Use AND-gates across views or models instead.
5. **Restate M2 per predicate class**, and measure it on at least 100 claimed successes. On Tier 1-like tasks, vision-only gating trades overclaim directly against false rejects:
   - 5.9% overclaim with one Opus verifier;
   - 0.8% with Opus EF AND Astra, at a cost of rejecting 4 of 6 true `done`s.
6. **Evaluate ER 2 and Robometer-4B on this exact harness before adopting either.** `verify.py` needs only a new `CONDS` entry.
   - ER 2 leads Opus 5 by 1.4 points on its own chart.
   - Robometer's only failure-detection number is F1 0.81 on 100 DROID trajectories, and it needs a GPU.

## 7. Caveats

- **Few real successes (40).** The puzzle has 4 and Tier 1 has 8, so slice confidence intervals are wide. The AND-gate winner was selected post hoc.
- **Grader bias.** Labels came from graders who knew the model, and 2–3 of the successes are disputable.
- **Calls went through a third-party proxy.**
  - Response `model` fields were checked and match (`claude-opus-5-5`, `gpt-6-astra`).
  - Whether OpenAI-route reasoning parameters are passed through is UNVERIFIED.
  - Latencies include proxy overhead.
- **Recorded video, not live cameras.** The verifiers saw at most 448 px per view (A) or 224 px (B).
- **Corpus C is simulation with a generic success criterion.** Its low TPR partly reflects that criterion.

## Sources

- Robocurve, "Opus 5.5 on RoboDojo-RC Tier 1" (2026-09-23): https://robocurve.org/opus-5-5-robodojo-rc-tier-1/. Includes `data/runs/*.json`, videos at `robodojo-tier1-artifacts.pages.dev/media/{host}/{rig}/{run}_{left,top,right}.mp4`, and logs at `…/log/{host}/{rig}/{run}/`.
- Robocurve, "GPT-6 Astra on robotic manipulation" (2026-09-04): https://openai.robocurve.org/gpt-6-astra/ (`runs/videos/*.mp4`, `runs/transcripts/*.html`).
- GPT-as-Policy rollouts: https://huggingface.co/datasets/YuMoool/astra-robodojo-rollouts (`SCHEMA.md`, `episodes.jsonl`; local clone `repos/astra-robodojo-rollouts`).
- FailBench: https://arxiv.org/abs/2609.03611 (full HTML text read).
- Gemini Robotics ER 2: https://deepmind.google/models/gemini-robotics/embodied-reasoning/ and https://ai.google.dev/gemini-api/docs/robotics-video-progress
- Robometer: https://arxiv.org/abs/2603.02115 and https://huggingface.co/lerobot/Robometer-4B
- RoboRMBench: https://arxiv.org/abs/2609.05401
- Gemini Robotics 1.5: https://arxiv.org/abs/2510.03342
- Manda labeler, `repos/manda-robot-episode-labeler` @ `31f0b19`: `src/rel/video/contact_sheet.py`, `src/rel/prompts/label_v2.md`, `src/rel/config.py`.
- Pigey: `repos/Pigey/real/agent.ts:347`.
- Peer notes: `gaps/gap-4.md` and `gaps/gap-4-trials.csv` (`false_success_claim`).
- Design under test: `HARNESS_DESIGN.md` §3.10, §4.2 and M2; `CRITIQUE.md` Gap 5.
