# 2026-10-28 — Perception at grasp time: the metric, its baseline on the saved runs, and the frame-0 ceiling (stream M)

**Question.** Ilia's goal is perception that is right in >= 95 % of grasp-time events, judged from the point of view of
the manipulator doing the grasp. This report:
1. defines that metric as code;
2. measures it on every saved integ run with a report and an action list;
3. computes how far a perfect per-object choice between all frame-0 sources could go;
4. picks the neighbour radius R from the data.

Team brief: `/tmp/dojo-team/PERCEPTION_95.md`. Code: `dojo/perception/grasp_metric.py`,
`scripts/perception/m95_{replay,score,ceiling}.py`. 0 LLM calls.

## The metric (`grasp_metric.score_event`)

- **Event:** every grasp and every place / release, taken from the executed gripper commands (`grip_events`).
  - A grasp is the LAST closing command (< 0.5) of a closing run before the jaws open again. A pre-shape on the
    approach is not the grasp (hang_mugs' wall grasp: 1.0 -> 0.43, then 0.0 about 30 steps later).
  - A release is the first opening after a grasp.
  - Both belief and GT are taken at the state before that action.
- **Required objects:**
  - the grasp target, or the held object for a place;
  - for a place, its reference (support / container / slot / base).
  - The target is the label of the last logged pick refine of that arm since its previous grip event. Otherwise it is
    the believed object within 6 cm of the TCP. A close with neither is `close_other` (a press with closed jaws) and
    is not scored.
- **Required-object classes:**
  - `ok`: the believed body centre is within the task tolerance (eval_g1 `TOLERANCE`; 15 mm where none) of the
    nearest GT object of its group, the rotation is within the task's rotation tolerance, and the asset is right;
  - `overshoot`, `gross` (> max(15, 2 x tol) mm), `rot`;
  - `identity`: right place, wrong asset;
  - `wrong_object`: gross for its own object but within tolerance of another one;
  - `not_believed`;
  - `not_held`: at a place, the GT object moved > 30 mm in the TCP frame since its grasp, so it is not in the jaws.
  - Distances are between body centres (mesh bbox centre), not origins, so a 180-degree flip of an off-centre origin
    is a rotation error and not a 50 mm one (plug_in_charger's charger).
- **Neighbours:**
  - every GT object whose footprint (convex hull of its catalogue mesh) lies within R (xy) of the corridor (the event
    point plus the last 10 cm of the TCP's path);
  - it is clean when a believed object of any label is within 15 mm of it.
- **Phantoms:** believed objects within R of the corridor that match no neighbour and are > 50 mm from every GT object.
- **Clean** = every required object ok, every neighbour within R found, no phantom within R. Reported at R = 5 / 10 /
  15 cm, plus required-only and target-only.
- **The runtime's belief at the event** (`episode_events`):
  - the report's `dev_belief_probe` snapshot where the run has one (new: `dojo.runtime.eval.BeliefProbe`);
  - otherwise rebuilt from the report: the planning record, then every accepted refine (pick / place / held), the
    jaws (an attached object rides the commanded TCP), `belief_drop`, `post_place`, and held_look drops.
  - The rebuild cannot see a divergence's re-estimate (events after it are flagged `after_divergence`) or the
    conveyor tracker's riders (`moving`).
- **GT:** a physics-only replay of the episode's executed actions on its settled layout (`m95_replay.py`, compute3
  light1-2). Fidelity: every close that has a `dev_gt_probe` reproduced the episode's GT exactly (177 of 177 at 0.0 mm,
  including the c4 genuine-5.1 rendered runs).
- **Rebuild corrections found on the way.** Each would otherwise have biased the numbers:
  - stream C: the pre-shape;
  - stream H: the runtime's own attach rule; 18 of 98 `not_held` were closes the runtime never attached;
  - stream W: post_place moves logged as from / to;
  - 3-D matching for stacked blocks, which had produced false identity errors in 34 of 38 stack_blocks episodes.

## Data

- Every perception-mode integ episode with a report and an action list:
  - compute3 `~/dojo-data/integ/runs` (proxy: 5.1 physics + 6.0 render);
  - compute4 `~/dojo-data/integ-c4/runs` (genuine 5.1 rendered, copied).
- Ids 100000-09 / 100020-49 only.
- The c4 episodes the 429 marks list as agent-unavailable are tagged `agent_unavailable`. Excluding them moves the
  latest-round R10 by about 1 point (71 -> 70 %).
- Rounds, newest first:

| round | when | preset / code |
|---|---|---|
| c4 | 27 Oct 17-22Z, genuine 5.1 | submit_v2/v3 |
| P3r3 | 27 Oct 12Z | submit_v2, 13aaf9a |
| TL | 27 Oct 11Z | |
| P3n | 27 Oct 08Z | eeb94de |
| P3 | 27 Oct 05Z | 1f7e965 |
| S79 | 27 Oct 03Z | 48600bf |
| breadth2 | 26 Oct 21Z | ed8ca42 |
| heldpose / breadth | 26 Oct 16Z | f9b0d75 |
| early stack / tower / AG | 24-26 Oct | |

## Results

### 1. Baseline: latest round per task (the runtime's own belief; clean %, n events)

| task (round) | events | R5 | **R10** | R15 | required only | frame-0 record, R10 |
|---|---|---|---|---|---|---|
| build_tower (P3r3) | 90 | 100 | **100** | 100 | 100 | 24 |
| insert_tubes (P3r3) | 32 | 100 | **100** | 100 | 100 | 78 |
| stack_blocks (P3r3) | 28 | 100 | **100** | 100 | 100 | 39 |
| cover_blocks (AG) | 72 | 100 | **100** | 100 | 100 | 75 |
| plug_in_charger (P3r3) | 19 | 95 | **95** | 95 | 95 | 63 |
| stack_blocks_by_language (breadth2) | 30 | 93 | **93** | 93 | 93 | 43 |
| fasten_screws (S79) | 147 | 87 | **86** | 86 | 88 | 28 |
| deposit_coin (S79) | 21 | 71 | **71** | 71 | 71 | 29 |
| organize_table (breadth2) | 26 | 69 | **69** | 69 | 69 | 35 |
| play_Xylophone (breadth2) | 12 | 67 | **67** | 67 | 67 | 33 |
| fill_pen_holder (breadth2) | 34 | 62 | **62** | 53 | 65 | 50 |
| hang_mugs (P3r3) | 24 | 67 | **58** | 58 | 71 | 58 |
| make_toast (c4, 5.1) | 33 | 58 | **52** | 45 | 61 | 45 |
| store_tools_in_toolbox (breadth2) | 38 | 61 | **55** | 53 | 63 | 47 |
| play_tic_tac_toe (breadth2) | 107 | 61 | **55** | 40 | 75 | 25 |
| swap_blocks (breadth2) | 48 | 54 | **50** | 46 | 54 | 13 |
| put_bottles_into_dustbin (AG) | 48 | 50 | **50** | 44 | 50 | 44 |
| sweep_blocks (c4, 5.1) | 45 | 40 | **38** | 33 | 42 | 33 |
| stack_bowls (AG) | 21 | 29 | **14** | 14 | 29 | 14 |
| general_pickup (AG) | 13 | 100 | **0** (100 without clutter) | 0 | 100 | 0 |
| make_kong (breadth2) | 6 | 0 | **0** | 0 | 0 | 0 |
| align_blocks (AG) | 6 | 0 | **0** | 0 | 0 | 0 |
| **all, latest rounds** | **900** | 74 | **71** | 67 | **77** | 37 |

- **Coverage.**
  - press_by_number presses with closed jaws (no grasp events).
  - The conveyor tasks' riders are tracked between reads and are not logged, so their rebuilt beliefs are not
    perception numbers. The Integration c4b belt runs had no gripper close at all.
  - general_pickup's distractors (clutter) are real neighbours that B1 never estimates.
- **All rounds:** 4767 events in 538 episodes: 75 % clean at R10, 79 % required-only. The older rounds alone: 3867
  events, 76 %. That mix is stack / tower-heavy, and those tasks are clean.
- **By round** (all tasks, R10): P3r3 94 %, S79 88 %, P3n 85 %, TL 85 %, P3 82 %, early 78 %, AG 66 %, breadth 63 %,
  breadth2 58 %, c4 5.1 44 %, heldpose 27 %. The task mix differs per round.

### 2. Why events fail, and what it takes to reach 95 % (latest rounds)

Of the 900 events, 264 are unclean at R10. The first failure, by blocker:

| blocker | events | owner |
|---|---|---|
| released: an object the robot let go earlier, believed where the release put it | 72 | W (release read), post_place |
| not_held: the belief carries an object the jaws are not on | 56 | H |
| neighbour / phantom | 40 | C (tic_tac_toe pieces), general_pickup clutter |
| in-hand pose | 35 | H |
| target pose (frame-0 / refine) | 30 | C, V, S |
| identity (fasten bolts / nuts, toaster, make_kong tiles, bottles) | 27 | S, Identity |
| reference pose | 4 | |

- 219 more clean events reach 95 %. The largest per-task gaps: tic_tac_toe 43, sweep 26, put_bottles 22, swap 22,
  stack_bowls 17, store_tools 16, toast 15, general_pickup 13, fasten 13, fill_pen 12.
- Per task: `m95_score.py table --latest --gap`.
- The runtime improves on frame 0 by a lot. On the same events, the frame-0 record would be clean in 333 of 900 at
  R10, the runtime's belief in 636 (312 fixed / 9 broken). The remaining failures are mostly state the robot itself
  changed (held, released), not first reads.

### 3. The threshold R

| variant | AUC of an episode's clean rate for success, all rounds (538 episodes) | latest rounds (107) |
|---|---|---|
| R5 | 0.81 | 0.80 |
| R10 | 0.81 | 0.75 |
| R15 | 0.82 | 0.76 |
| required only | 0.81 | 0.79 |
| target only | 0.79 | 0.76 |

- Neighbour / phantom problems alone are rare. Among 258 episodes whose required objects were always right, R10 adds
  a neighbour / phantom problem to 9: 5 of those 9 failed, against 43 of the 249 without. At R5 it is 3 of 4; at R15,
  7 of 11.
- **Decision:**
  - R = 10 cm stays the headline. It is the jaw span plus clearance and separates outcomes about as well as R5 / R15.
  - Required-only and R5 are reported alongside.
  - Neighbours are a weak but real failure signal, not noise.

### 4. The frame-0 ceiling: a perfect per-object selector over every source

- **Sources:**
  - the SAM bake-off arms (B1 main, SAM 2.1-S / 2.1-L / 3.1 proposers, trim, union, gate, combo);
  - the Identity stream's g1 runs (embed on / off, partial coverage);
  - the perceiver-models arms (Sonnet x2, Sonnet 16, Opus, Fable, raw / H2 / arbitration; partial coverage).
- **Stereo** is a verification signal, not a pose source, so it is not a selectable arm. Its slot-base reject is in
  the stereo report.
- **Scoring:** strict = eval_g1 `failure_class == ok` (within tolerance + right asset). 217 scenes / 1073 objects
  (d0_51 + d0_ext + coin + laptop).

| source | objects ok | scenes all ok (strict) | scenes all ok, make_kong tile faces excused |
|---|---|---|---|
| B1 (main, series min) | 66 % | 51 % | 55 % |
| SAM 2.1-S proposer | 69 % | 57 % | 61 % |
| best single arm (sam_combo) | 70 % | 60 % | 65 % |
| perfect selector, 9 B1 / SAM arms | 72 % (92 % excused) | 66 % | 71 % |
| + agent arms + Identity (partial coverage) | 74 % (93 %) | **70 %** | **75 %** |

- **Per task, the selector's scenes all ok** (excused):
  - organize 1/9, pack 3/15, build_tower 8/15, stack_blocks 9/15, plug 7/10, stack_by_language 7/10,
    put_bottles 12/17, deposit_coin 8/10, align 8/10, bowls 13/15;
  - cover, fill_pen, general_pickup, insert_key, press, store_tools and swap_T at 100 %.
- Restricting to grasp-relevant objects (the labels the runs grasp or place on) changes little: 71 / 75 % of scenes.
- **No source gets 74 objects right:** gross 46, missed 12, identity 11, overshoot 5. 22 of them are >= 50 % hidden,
  13 occluded, 7 thin.
  - pack 17 (shoe / car / toothbrush 40-260 mm off);
  - organize 12 (mouse / alarm / garage 100-210 mm);
  - build_tower 10 (hidden blocks, rotation);
  - stack_blocks 10 (edge blocks missed; colour identity);
  - put_bottles 7;
  - push_T 5 (under the home grippers);
  - stack_by_language 4 (letter identity);
  - plug 3, coin stand 2;
  - make_kong's 207 tile faces: no source reads them.
- **The 6.0 proxy hard sets** (d0, 92 scenes / 400 objects; only the perceiver-models arms ran there): B1 39 % of
  scenes, the selector with the agents 46 %.
- **Verdict.** The best possible pick among today's sources is 70-75 % of scenes all right at frame 0, far from 95 %.
  - The missing classes are information-limited: hidden or out-of-view objects, look-alike identity, far misreads in
    clutter.
  - At grasp time, the classes no frame-0 source addresses at all (held state, released objects) are the largest
    runtime blockers (section 2).
  - These define what streams C (continuous re-reads), V (multi-view geometry), S (a trained identity / detector),
    H (held state) and W (release read) must deliver.

## Spend and resources
- 0 LLM calls.
- GPU: compute3 light1-2, one physics-only replay per slot, never more than 4 of our GPU jobs on the box.
- 691 episodes replayed in 2 h 40 min on two slots, on a box at load 13-30. Replays stop after the last gripper change.
- Data: compute3 `~/dojo-data/m95` (85 MB).

## Reproduce
```
python3 scripts/perception/m95_score.py jobs "<run dirs>" > jobs.jsonl
scripts/simlab/rdpy.sh -u scripts/perception/m95_replay.py --task T --jobs jobs.jsonl --out-dir REPLAYS
python scripts/perception/m95_score.py score "<run dirs>" --replays REPLAYS --out OUT
python scripts/perception/m95_score.py table OUT/events.jsonl [--latest] [--gap] [--taxonomy] [--corr] [--frame0]
python scripts/perception/m95_ceiling.py --g1 NAME=DIR ... --pmodels CASES --pm-lists DIR --pm-tags ... \
    --combine d0_51,d0_ext,coin,laptop [--identity-ok make_kong] [--targets OUT/events.jsonl]
```
