# Gap 4: Forensic analysis of the 360 public Robocurve "Opus 5.5 on RoboDojo-RC Tier 1" trials

*Written 2026-10-02 from primary data fetched the same day. Per-trial table: `gaps/gap-4-trials.csv` (360 rows: rig, order on rig, score, termination, calls, API latency, close events, failure class, false-success flag).*

## 1. What was retrieved and how it was analysed

**Retrieved from https://robocurve.org/opus-5-5-robodojo-rc-tier-1/** (a curl browser User-Agent works):
- all 360 `data/runs/*.json` files, plus `data/trials.csv` and `data/cells.csv`;
- all 360 per-trial "Log" pages, `https://robodojo-tier1-artifacts.pages.dev/log/{host}/{rig}/{run_id}/`. Each is a self-contained HTML file of 9–12 MB. Embedded PNGs were stripped and the text was parsed. Each log contains:
  - the full system prompt and tool schemas;
  - each observation's `eef_state`/`joint_eff` readback;
  - every `move_to` call with its `note`;
  - the final `done`/`give_up` text, including `summary`/`reason` and `hindsight`;
  - one wire line per call (`call N attempt M · /messages · status 200 · X s`).
- 78 camera MP4s (top/left/right for 26 trials), `…/media/{host}/{rig}/{run_id}_{cam}.mp4`. Each is 224×224 at 30 fps, one frame per 10 Hz control step.

**Harness facts, read from each log's spec strip.**
- Inspect Robots 0.58.0, `policy=agent`, `effort=medium`, `max_llm_calls=40`, `max_speed_frac=0.25`, `image_horizon=2`, `images=always`, max steps 900.
- Opus runs used `wire=messages` with `max_output_tokens=16000`. Astra used `wire=responses` with `null`.
- Tools: `move_to` (absolute EEF targets), `done`, `give_up`.
  - `done` is described as "Declare the task finished. The trial ends; a scorer judges success."
  - The system prompt says: "When the goal is achieved call done; if it cannot be achieved call give_up."
- When the budget runs out, the harness calls `_forced_give_up(..., "LLM call budget exhausted")`. That synthetic call is "never appended to _messages" (`plugins/inspect-robots-agent/src/inspect_robots_agent/policy.py`, budget check at ~L997 and `_forced_give_up` at L1209–1217 in the local clone, v0.60.0 / `095172f`; the trials ran 0.58.0).

**Method.**
- Each of the 234 non-completed Opus 5.5 and Astra trials was hand-coded with one *primary blocking failure*: the reason the next unmet rubric milestone was not reached. The coding used the rubric score, the model's final self-report, and the last agent notes. I read all 234 texts.
- Video and frame inspection was done only for the 22 false-success trials and 4 reference trials (§3).
- Objective log metrics (empty closes, tracking error, latency) were computed for all 360 trials.
- **Limitations:**
  - one non-blind coder;
  - self-reports can be wrong (§3 shows cases);
  - Astra writes terser, more "safety-stop" reports than Opus 5.5, which may inflate its no-grasp share;
  - Opus 5 is not hand-coded.

## 2. Design facts that bound every comparison

1. **Each task ran on its own rig, and models ran in blocks rather than interleaved.**
   - The rigs were `omen/rig-1` Pack & Pour, `rig-2` Stack Bowls, `rig-6` Classify (the page calls it "Sort Objects"), `omen-2/rig-3` Safe, `rig-4` Cap Pen, and `rig-5` Bottles.
   - On every rig the order by `started_at` is 20× Astra, then 20× Opus 5.5, then 20× Opus 5.
   - Everything ran from 2026-09-22 16:13 to 2026-09-23 03:19 UTC.
   - So the model comparison is confounded with order, time of night, operator fatigue and motor heat.
2. **Overheat is an order artefact, not a model property.**
   - All 10 Opus 5 overheats occurred at positions 41–60 of their rig's 60-trial sequence.
   - Astra's 2 overheats were at positions 14 and 17. Opus 5.5's 2 were at 25 and 30.
   - Correct the design's §0/§2 "Opus 5 had 10/120 overheats" citation accordingly.
3. **Graders were not blind and tasks were not neutral.** The page states "human operators who knew the model" scored the trials, and that the six tasks were "the six easiest" for Astra out of 18.
4. **Completions: 1/120 for Opus 5.5 vs 5/120 for Astra, Fisher p = 0.21.**
   - Two of Astra's five completions (Store In Safe `29b81aa0`, `8deefa55`) ended in a *forced* budget give_up while the model was still pressing the lid. The model never declared success.

## 3. "Premature done" is mostly a misreading of the termination field

The critique counts 44 of 45 Opus 5.5 `done` terminations as incomplete and treats them as overclaims (critique Error 2, and Error 3's "44/119 vs 12/115, p ≈ 1.5×10⁻⁶"). **The `done` summaries do not support that reading.**

**Opus 5.5 (45 `done`s):**
- 34 are explicit partial reports (76%, Wilson 61–86%). For example `fbc95441`: "I only finished half the task … I ran out of my call budget before I could move them."
- 10 claim success and were scored below 1.
- 1 is the single completion (`118b151b`), where the model *doubted* its own success: "the brown bear's legs stick out the front edge". The grader gave it 1.0.

**Astra:**
- It used `done` only when it believed it had succeeded: 15 `done`s, 12 of them false claims and 3 true.
- It never used `done` for partial results.

**False-success claims among non-completed trials:**

| Model | False claims | Rate | Wilson 95% CI |
|---|---|---|---|
| Opus 5.5 | 10/119 | 8.4% | 4.6–14.8% |
| Astra | 12/115 | 10.4% | 6.1–17.4% |

Fisher p = 0.66, so there is **no detectable difference**. Opus 5.5 uses `done` as "end of episode, honest report" when it runs out of budget. That is a protocol deviation, not a calibration failure.

**19 of the 22 false claims are Stack Bowls trials graded 0.75.** The rubric gives 0.75 for "three bowls overlapping in one connected arrangement" and 1.0 for "three bowls stacked".

**What the agent could see.** For every false claim I extracted the done-step frame from all three cameras:
- **Stack Bowls (7 Opus 5.5, 12 Astra).** Stated evidence: "In the final overhead view there's a single stack with the pink bowl on top and no other bowls on the table" (`570715d2`); Astra says "Stacked all three bowls together … parked both arms clear".
  - In all 19 frames, top_cam shows one pink bowl about 25–30 px wide, and both wrist cameras look past it at bare table.
  - **They are visually indistinguishable from the done frames of Astra's three 1.0 trials** (`8987b031`, `2190daaf`, `b7f0d5d3`).
  - Whether the lower bowls are nested is hidden under the top bowl at 224 px. Opus 5.5 even said so: "From that angle I couldn't see how deeply each bowl nests" (`fa110597`).
  - The decisive region was **not observable**. These are unverifiable claims under a rubric distinction the agent could not perceive, not hallucinations.
- **Cap Pen `6afdf1a7` (scored 0.75).** Claim: "In both wrist cameras the tip disappeared into the cap … I didn't pull-test the fit."
  - Both wrist cameras show the pen edge-on as a line a few pixels wide, so the cap/tip junction cannot be resolved. **Not observable.**
- **Stand Up Bottles `14f73148` (scored 0).** Claim: "all three look upright now."
  - The bottles lie with caps pointing forward. The forward-looking wrist camera and the 45° top camera both render a lying, forward-pointing bottle as a vertical elongated blob, the same shape a standing bottle makes.
  - **Observable but ambiguous.** This is a perspective misperception.
- **Classify `0fab5548` (scored 0.83).** The teddy on the basket rim is clearly visible in left_cam, and the model flagged it ("may be resting on the rim"). It still called `done`, at 39/40 calls.
  - **Visible and noticed.** This was budget pressure, not perception.

**Implication for R5/§3.10.** A 224-px verifier would have reproduced most of these errors. A success gate needs one of:
- a dedicated verification view: an oblique or side shot at higher resolution, zoomed on the container;
- a physical probe: a pull-test, a lift-and-check, or a gripper-width check;
- a structured `claimed_success: bool` field separate from episode termination. Opus 5.5's `done` conflates the two.

## 4. Terminal failure classes, task × class

**Codes:**
- NG — no grasp: empty closes, or the object was pushed away.
- SD — slip or drop in transport.
- PM — placement/release miss: bounced out, on the rim, landed beside, or not seated.
- AL — bimanual alignment or insertion.
- FS — final articulated or dynamic step: lid or pour.
- RK — reach or kinematic limit.
- HW — arm stall, unresponsive arm, or overheat.
- CO — collision: container tipped or dragged, lid knocked shut.
- BU — budget ran out while progressing.

Trials marked `*` are false-success claims (§3).

**Opus 5.5 (n = 119 non-completed)**

| Task | NG | SD | PM | AL | FS | RK | HW | CO | BU | n | `*` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Cap Pen | 1 | 0 | 0 | **18** | 0 | 1 | 0 | 0 | 0 | 20 | 1 |
| Classify (Sort) | 2 | 2 | **8** | 0 | 0 | 0 | 2 | 2 | 4 | 20 | 1 |
| Pack & Pour | 2 | 0 | **11** | 0 | 1 | 0 | 2 | 0 | 4 | 20 | 0 |
| Stack Bowls | 0 | 0 | **14** | 0 | 0 | 3 | 1 | 0 | 2 | 20 | 7 |
| Stand Up Bottles | 7 | 0 | 5 | 0 | 0 | **8** | 0 | 0 | 0 | 20 | 1 |
| Store In Safe | 3 | 1 | 4 | 0 | **5** | 2 | 1 | 2 | 1 | 19 | 0 |
| **All, % [Wilson 95%]** | 13 [8–20] | 3 [1–7] | **35 [27–44]** | 15 [10–23] | 5 [2–11] | 12 [7–19] | 5 [2–11] | 3 [1–8] | 9 [5–16] | 119 | 10 |

**GPT-6 Astra (n = 115 non-completed)**

| Task | NG | SD | PM | AL | FS | RK | HW | CO | BU | n | `*` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Cap Pen | 4 | 0 | 0 | **15** | 0 | 1 | 0 | 0 | 0 | 20 | 0 |
| Classify (Sort) | 7 | 0 | **8** | 0 | 0 | 2 | 0 | 1 | 2 | 20 | 0 |
| Pack & Pour | 2 | 2 | **7** | 0 | 1 | 1 | 3 | 3 | 1 | 20 | 0 |
| Stack Bowls | 4 | 0 | **13** | 0 | 0 | 0 | 0 | 0 | 0 | 17 | 12 |
| Stand Up Bottles | **16** | 1 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 20 | 0 |
| Store In Safe | **10** | 0 | 2 | 0 | 5 | 1 | 0 | 0 | 0 | 18 | 0 |
| **All, % [Wilson 95%]** | **37 [29–47]** | 3 [1–7] | 26 [19–35] | 13 [8–20] | 5 [2–11] | 7 [4–13] | 3 [1–7] | 3 [1–9] | 3 [1–7] | 115 | 12 |

**Key per-task cells:**

| Task | Opus 5.5 | Astra |
|---|---|---|
| Cap Pen, AL | 18/20 = 90% [70–97] | 15/20 = 75% [53–89] |
| Stack Bowls, PM | 14/20 = 70% [48–85] | 13/17 = 76% [53–90] |
| Stand Up Bottles | NG 7/20 = 35% [18–57]; RK 8/20 = 40% [22–61] | NG 16/20 = 80% [58–92] |
| Store In Safe, NG | 3/19 | 10/18 = 56% [34–75] |
| Store In Safe, FS (lid) | 5/19 = 26% [12–49] | 5/18 = 28% [12–51] |
| Pack & Pour, PM | 11/20 = 55% [34–74] | 7/20 = 35% [18–57] |

## 5. What the data say about the "last-centimetre" reading

The critique says the last-cm diagnosis is unsupported. The data show it is **partly right, but for the wrong reasons**:

- **Placement/alignment is Opus 5.5's largest blocker.** PM + AL = 60/119 = 50% [42–59] vs Astra 45/115 = 39% [31–48], p = 0.09.
- **Astra's largest blocker is earlier in the pipeline:** no-grasp 37% vs 13% for Opus 5.5 (p = 1.7×10⁻⁵).
- So the per-task flips (Cap Pen 57.0 vs 41.0, Stack Bowls 42.5 vs 65.0) are consistent with Opus 5.5 getting further and then failing at placement.

**Composition of Opus 5.5's 42 PM trials.** Keyword-assigned, so approximate:
- about 11 drop-from-height bounce-outs, nearly all in Pack & Pour. Example, `f9147747`: "I let go of each one from above the bowl without seating it first, and both bounced out";
- about 14 landed on a rim or edge (teddy over the safe's front wall, pen on the basket rim);
- 8 stacks graded 0.75 ("not seated");
- about 4 landed beside or short;
- about 5 other, mostly bottles that fell back after release.

Only the "not seated" and on-rim cases are millimetre-to-centimetre precision problems. The bounce-outs are a *release-strategy* failure.

**AL (Cap Pen, 18) is mostly not a precision failure either.** Among the 18 final texts:
- 13 mention a joint, reach or tracking stall;
- 8 mention arm-body interference ("the two gripper housings and wrists hit each other while the grasp points were still about 22–24 cm apart", `9ea13997`);
- 8 mention unknown cap orientation ("I don't know which way the cap's open end faces");
- 10 mention the cap not seating or falling.

Only two report a residual misalignment of about 1–1.5 cm (`82007e65`, `80d8f14e`). The blocker is bimanual geometry plus missing axial perception, not millimetre control. Astra fails the same way: 10/15 mention interference. But Astra then *returns both parts to the table* (15 Astra final texts across tasks report returning objects to the table). Because the rubric's "lifting … milestones retain credit", this costs it nothing.

**Stand Up Bottles is embodiment-limited.**
- Only in this task are the bounds `left_pitch: [0, 0]`, `left_roll: [0, 0]`, and the same for the right arm. Every other task allows ±0.6 pitch and ±1.571 roll.
- The bottle necks sit at x ≈ 0.45–0.56 m. Table-level reach is about 0.49 m, per the rig facts in the prompt.
- 12/20 Opus 5.5 reasons cite the locked pitch and 12/20 cite reach. Example, `9fc8a745`: "Pitch and roll are locked, so the gripper can't tip a lying bottle upright."
- The ≤5% scores mostly measure the harness configuration, not the model.

**Store In Safe lid.** Example, `e3463608`: "The open lid stands up behind the box, about x ≥ 0.6 … the right arm stalls at z≈0.15–0.21." The lid sits outside the reachable box. That is a scene-layout limit.

## 6. Objective log metrics, all trials

**Empty closes.** An empty close is a `move_to` with gripper ≤ 0.1 from an open gripper (≥ 0.3) whose next observation reads below 0.04, the prompt's own "empty" threshold. Plush toys can read about 0.01 while partly held, so these rates are upper bounds.

| Task | Opus 5.5 | Astra | Opus 5 |
|---|---|---|---|
| All tasks | **260/543 = 48% [44–52]** | **328/574 = 57% [53–61]** (p = 0.002 vs Opus 5.5) | 270/525 = 51% |
| Stack Bowls | 70% | 76% | 76% |
| Stand Up Bottles | 64% | 75% | 48% (n = 25) |
| Store In Safe | 55% | 71% | 51% |
| Pack & Pour | 38% | 34% | 49% |
| Cap Pen | 37% | 46% | 74% |
| Classify | 32% | 42% | 20% |

Roughly every second close misses. That is the dominant consumer of the 40-call budget.

**Free-space tracking.** Moves with all of x, y, z commanded, target z ≥ 0.10 and start z ≥ 0.08:

| Model | Median 3D error | p90 | Moves > 2 cm | Median z error | z error p10 | n |
|---|---|---|---|---|---|---|
| Opus 5.5 | 2.2 cm | 5.2 cm | 56% | −1.6 cm | −3.0 cm | 1,058 |
| Astra | 2.0 cm | 5.1 cm | 50% | −1.6 cm | −2.8 cm | 1,056 |

**Wrist sag and stall notes.**
- When pitch was commanded or held at 0, readback |pitch| exceeded 0.15 rad in 31% of observations, with p90 0.34 rad.
- 95/120 Opus 5.5 trials contain at least one "stuck/stalled/not responding/stopped tracking" note (276 notes in total). Astra: 32/120 trials, 45 notes. The difference is partly style.

**Budget.**
- 100/120 Opus 5.5 final texts cite the call budget [76–89%], against 8/120 for Astra. Opus 5.5 averaged 34.2 calls, and 66 trials used 35 or more.
- Astra averaged 28.6 calls and gave up within 15 calls in 9 trials. In 28 final texts it asked the operator to reposition objects. Example, `16c5f77d` at 8 calls: "Please move that plush inward".
- Forced budget give_ups: Opus 5.5 6, Astra 5, Opus 5 29.

**Latency, tokens and cache** (wire lines, all 200, 0 retries). These match Gap 1:

| Model | API latency p50 / p90 / p99 / max | Wall time per call (p50) | API share of wall time | Output tokens per call | Cache-read share |
|---|---|---|---|---|---|
| Opus 5.5 | 5.28 / 11.72 / 21.10 / 60.3 s (n = 4,105) | 7.59 s | 0.84 | 428 | 0.865 |
| Astra | 6.53 / 11.28 / 19.59 / 38.9 s (n = 3,437) | 8.80 s | 0.83 | 137 | 0.328 recorded |
| Opus 5 | 8.77 / 20.79 / 40.49 / 67.8 s (n = 4,226) | 12.31 s | 0.90 | 674 | 0.865 |

Opus 5.5 per-task p50/p90 and output tokens per call:

| Task | p50 / p90 | Output tokens per call |
|---|---|---|
| Pack & Pour | 4.6 / 9.3 s | 345 |
| Classify | 4.8 / 10.6 s | 379 |
| Safe | 4.9 / 8.8 s | 329 |
| Stack Bowls | 5.5 / 11.2 s | 420 |
| Bottles | 6.2 / 13.4 s | 510 |
| Cap Pen | 6.9 / 14.3 s | 578 |

Latency tracks output length, and output length tracks task difficulty.

**Cost.** The page's $0.90 per trial is reproduced exactly ($0.898) from the mean token fields (620,537 cache-read, 96,670 cache-write, 137 uncached, 14,496 output). It uses $4/$20 per MTok, cache reads at $0.20/MTok (the bundled Anthropic reference's Opus 5.5 rate), and cache writes at 1.25× ($5/MTok). The write rate is my assumption, and it fits. Opus 5 reproduces $1.758 at $5/$25 with 0.1× reads and 1.25× writes.

## 7. Which proposed contact skills would have addressed which failures (Opus 5.5 counts / Astra counts)

| Failure (O55 / Astra) | `center_grasp` | `descend_until_contact` | `insert` | `place_release` | `regrasp` | Not covered; what is needed |
|---|---|---|---|---|---|---|
| NG 15 / 43 (plus about 50% empty closes everywhere) | **primary** | yes (table-level grasps) | – | – | **yes** | Bottle necks beyond reach; plush needs a compliant or enveloping grasp |
| SD 3 / 3 | – | – | – | – | **yes**, with lift-verify | – |
| PM bounce-out about 11 / 6 | – | **yes** | – | **yes** (lower into the container, open at contact) | – | – |
| PM rim/beside about 18 / 12 | – | yes | – | partial (needs a container-centre target) | yes | Container localisation |
| PM "not seated" stacks 8 / 12 | – | **yes** (seat until stall) | – | yes | – | A *verifier* with a side or high-resolution view (§3) |
| AL 18 / 15 | – | – | partial | – | yes (re-orient cap) | Collision-aware bimanual pose planning, cap-orientation perception, an axial view; or a learned bimanual head |
| FS lid 5 / 5, pour 1 / 1 | – | – | – | – | – | Articulated push/pull and pour skills; reachable scene layout |
| RK 14 / 8 | – | – | – | – | – | Reach-aware planning, drag-to-reach, handoff; unpin pitch/roll |
| HW 6 / 3, CO 4 / 4 | – | – | – | – | – | Stall detect/resync, thermal budget, collision checking |
| BU 11 / 3 | indirect | indirect | – | indirect | – | Macro skills that fold approach–descend–close–lift–verify into 1 call instead of about 5 |

**How much the five skills would cover (my judgment).** For Opus 5.5:
- They touch NG + SD + PM + AL = 78/119 = 66% of non-completed trials.
- They *directly* fix about 19/119 = 16%: the bounce-outs and the not-seated stacks, which `descend_until_contact` + `place_release` handle.
- About 35/119 = 29% (RK, FS, HW, CO) need things outside the skill set: workspace layout, articulated skills, collision-aware bimanual planning, and stall handling.

For Astra, `center_grasp`/`regrasp` matter most (NG 37%).

## 8. Corrections to the design and the critique

1. **Replace** "Opus 5.5 declared `done` in 44/119 vs Astra 12/115" with **false-success claims 10/119 vs 12/115 (p = 0.66)**.
   - 19/22 of those claims are Stack Bowls 0.75 vs 1.0 cases, indistinguishable at 224 px.
   - Opus 5.5's other 34 `done`s are honest partial reports.
2. **"Last centimetre"** may be kept only in this narrower form: "Opus 5.5's blocking failures are mostly at placement/assembly (50%), but these are dominated by release strategy, bimanual interference and unobservable seating, not millimetre accuracy."
3. **Treat Tier 1 as non-interleaved, unblinded and order-confounded.** Opus 5's 10 overheats sit entirely in each rig's last block.
4. **Bottles scores reflect pinned `pitch/roll = [0, 0]` and reach.** Do not use them as a model signal.
5. **Harness requirements that follow from this data:**
   - separate `claimed_success` from termination;
   - add a verification view;
   - add a grasp verifier, since about half of closes are empty;
   - add `place_release` with contact detection;
   - add stall detection;
   - set per-task call budgets with a remaining-calls counter in each observation. The models counted calls themselves.

## Sources

- Robocurve, "Opus 5.5 on RoboDojo-RC Tier 1: higher scores at lower cost", 2026-09-23. https://robocurve.org/opus-5-5-robodojo-rc-tier-1/ (fetched 2026-10-02). Sections used: results tables, rubrics, Technical specifications, Limitations.
- Same site: `data/trials.csv`, `data/cells.csv`, and 360 × `data/runs/{host}_{rig}_{run_id}.json`.
- Per-trial logs: `https://robodojo-tier1-artifacts.pages.dev/log/{host}/{rig}/{run_id}/` (all 360, Inspect Robots 0.58.0 HTML reports with wire capture).
- Per-trial videos: `https://robodojo-tier1-artifacts.pages.dev/media/{host}/{rig}/{run_id}_{top,left,right}.mp4`. 26 trials inspected: all 22 false-success trials, Astra Stack Bowls 1.0 trials `8987b031`/`2190daaf`/`b7f0d5d3`, and Opus 5.5 `118b151b`.
- `robocurve/inspect-robots` local clone, `research/repos/inspect-robots` (HEAD `095172f`, v0.60.0): `plugins/inspect-robots-agent/src/inspect_robots_agent/policy.py` (budget check and `_forced_give_up`).
- Bundled Anthropic Claude API reference (claude-api skill, models table cached 2026-06-24): Opus 5.5 at $4/$20 per MTok, cache reads $0.20.
- `research/CRITIQUE.md` Errors 2, 3, 6 and 17; `research/gaps/gap-1.md` (independent latency extraction, consistent with this note).
- Derived data from this work: `research/gaps/gap-4-trials.csv`.
