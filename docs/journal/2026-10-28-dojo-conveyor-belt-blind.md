# 2026-10-28: the conveyor's "nothing moves" on compute4 was a missing belt background (Conveyor stream)

## 1. The failure (Integration, genuine-5.1 rendered, main ad5b50e, submit_v3)
`compute4:~/dojo-data/integ-c4/runs2/c4b_matchbelt_*`: 5 episodes of match_and_pick_from_conveyor, DINOv2 identity on
(`emb`, 100000-02) and off (`off`, 100000-01).
- Every episode: score 0, `perception_gross`, 507 / 700 steps, `{"stage": "belt", "note": "nothing moves"}`.
- `actor_hold` was on, so the scan, the looks and the search were skipped. The tracker read from step 0, every 4 steps.
- At step 4 the tracker dropped all five frame-0 labels after 2 reads (`track_drop`).
- After that, the only reads were unlabelled `rider<k>` blobs, and no track ever had a velocity.

## 2. Root cause: compute4's `bg_folds_51` had no `scene/` backgrounds
- compute4's `~/dojo-perception-data/bg_folds_51` matches compute3's file for file (md5), except that it has no
  `fold{0,1}/scene/`. It was copied on 26 Oct, before the scene backgrounds were built (27 Oct 03:37-11:43Z):
  `Dynamic_conveyor_00000`, `Geometry_basket_00005` (imitate), `Geometry_cube_cushion_00002` (organize_table).
- Without the belt's own background, `segment.with_scene` falls back to the table's. The dark belt then segments
  as foreground: 12-20 head blobs.
- What that did to the reads:
  - off arm: 125 of 127 belt reads found nothing, at 0.15 s a read (3-5 s on compute5, which had the background);
  - DINOv2 arm: `_embed_pick` fitted its best asset on fixture blobs. These were static "bottle" riders at
    (0.48, 0.31), (-0.24, 0.25) and (-0.355, -0.01) for 240 steps, at confidence 0.25-0.4 (below the newcomer gate
    of 0.5, so none joined);
  - frame-0 B1 was blind too: 5 labels on junk at confidence 0.04-0.15. The agentic perceiver put target0 right
    (0.43, 0.04), but as "bag".
- No read could match a frame-0 label, so the never-confirmed rule (`track_confirm_reads` 2) dropped all five at the
  second read. That rule exists for the scan's frame-0 ghosts; here the reads simply could not see.
- None of the brief's suspects caused it: newcomer gate, identity vote (#186), divergence, riders at the edge.

## 3. Offline replay on genuine-5.1 belt frames (compute3, CPU, 0 LLM calls)
Frames: `compute3:~/dojo-data/perception2/memcopy/belt51/match_and_pick_from_conveyor` (belt_bg record of our dev
layouts, robot at home, a frame every 10 steps, steps 0-250; 100003 is a partial copy). Same code (this branch),
table background (= compute4) vs the belt's scene background.

| background | belt reads | empty | head blobs | read time (median) | first rider read right (<= 3 cm, identity) | frame-0 B1 |
|---|---|---|---|---|---|---|
| table | 79 | 79 | 12-20 | 0.23 s | 0 / 22 | 5 labels on junk, conf 0.04-0.30 |
| scene | 84 | 12 (no rider in view) | 0-11 | 2.3 s | 24 / 24 | the in-view rider right in 4 / 4 (<= 9 mm, conf 0.81-0.95), the 4 upstream labels not found |

Tracker replay (`RuntimeProvider._track` on the recorded scene reads):
- target0 matched at every read while it is in view (x <= 0.70); no `track_drop`.
- `belt_velocity` from the 4th read (step 30) on:

  | layout | v (mm / step) |
  |---|---|
  | 100000 | 3.93-3.97 |
  | 100001 | 3.96-3.97 |
  | 100002 | 3.84-3.99 |

  The dev GT is 4.00, along +x.
- So with the belt's background the tracker works as built.
- What remains is identity at the far, upstream end of the belt. 100000's match target1 (a puppet) reads as
  "camera", with split fragments. The DINOv2 path (#192 / #193) exists for this case; there is no offline check of it
  here.
- Before its first fit (3 reads over >= 8 steps) a tracked rider's belief keeps the frame-0 pose. That is 8 steps at
  the runtime's read every 4, before `match_and_pick` has a velocity, so it does no harm.
- Riding out of view past x ~0.72, the border cuts a rider and its reads jump 4-11 cm across the belt (#175's case).

### 3.1 To step 440, plain and DINOv2, through match_and_pick's decision (compute3 light4, lead-approved)
- Inputs: the same frames, steps 260-440 at a read every 20 steps, the scene background, plain reads and
  `embed_model dinov2_s`.
- A replay harness runs the real `match_and_pick` (first object, velocity, match rule) on the tracker. The robot is at
  home and the pick is stubbed.

| layout | plain | DINOv2 |
|---|---|---|
| 100000 (puppet) | no match by step 440: the puppet behind reads as "camera" | the same |
| 100001 (ice cream) | target1 at step 386, 2 mm off | target1 at step 266, 13 mm off |
| 100002 (phone) | target1 at step 346, 95 mm off across the belt (a flat phone at the far end; its reads are right from step 420) | the same |

- Reads of the match rider at x in (-0.45, 0.70]:

  | reads | at the rider | right identity |
  |---|---|---|
  | plain | 16 / 25 | 11 / 25 |
  | DINOv2 | 13 / 25 | 13 / 25 |

  DINOv2 is the same as plain for the first rider (23 / 23), and has fewer phantoms (34 against 52 over steps 0-250).
- Two tracker / skill gaps showed up, and are fixed in §4:
  - **The belt velocity fell to zero between riders.** Once the first rider's track left the 40-step window, no track
    was trusted until the next rider had 4 reads: zeros for ~100 steps, and the next riders' predictions stood still.
  - **A ghost of the first object was "the next one".** 100002, plain: the first phone's reads at the view's border
    jumped 11 cm across the belt. Two of them started a second phone track there. Once the first's own track was lost,
    match_and_pick's "comes from behind the first" guard was off (`first not in w.objects`), and it chose that ghost,
    riding out of view at x 0.97, at step 126.
- Far-end reads are noisy:
  - newcomer tracks there give belt velocities of 2.1-5.5 mm / step while they are the only trusted tracks;
  - identity there is the weak point: puppet -> camera, the flat phone's y.

## 4. The change (branch `feat/dojo-conveyor-belt-blind`)
A missing belt background is now loud and harmless:
- `segment.scene_bg_key(fold_dir, keys)`: the scene key whose background a fold has (`with_scene` uses it).
- `belt.moving_bases(task)`: the Dynamic bases the task's riders ride. `belt.blind_reason(task, scene_bg)`: why a belt
  read cannot see them. Generic: keyed by the scene's assets, never by the task name.
- Perception service:
  - `bg_scene(fold)` returns the background in use;
  - the `belt_read` op returns `_perception.scene_bg` and, when its belt has no background, `_perception.blind`;
  - a `WARNING ... belt reads are blind` line once per fold and scene when a Dynamic base has no scene background.
- Tracker (`RuntimeProvider._track`): a blind read is no read. It neither confirms nor drops, adds no newcomer, and
  logs one `belt_blind` event (`stats.track_blind` counts these reads). `RuntimeProvider.belt_blind` holds the reason.
- Policy: the `track_read` event carries `blind: true`.
- Skills: `match_and_pick` / `pick_into_held` answer `nothing moves (belt reads blind: ...)`. G3 is unchanged, since
  its provider has no `belt_blind`.

Two fixes from §3.1:
- `RuntimeProvider.belt_velocity` keeps the belt's last measured velocity when no track is trusted. Only in a scene
  with a moving base: a belt runs at one speed.
- `match_and_pick`'s "behind the first" now uses the first's last belief moved on with the belt once the first has
  left the belief. While the first is in the belief nothing changes. In G3, where the GT world keeps every object,
  nothing changes either.

Tests:
- `tests/dojo/test_belt_replay.py` replays the recorded sequences (`tests/fixtures/belt_reads_51.json`):
  - compute4's reads on main's rule reproduce the five drops at step 4;
  - the same reads marked blind keep the labels and name the cause;
  - the scene-background reads track the first rider from frame 0 and give 4.0 mm / step by the 4th read, and keep
    it between riders;
  - match_and_pick on the replayed reads takes target1 in 100001 / 100002 (plain and DINOv2) and nothing in 100000.
    The previous skill takes 100002's ghost;
  - `blind_reason`, the service's warning, and the `belt_read` op through the service process.
- `test_runtime.py::test_a_blind_belt_read_is_logged_and_the_belief_keeps_its_objects` covers the policy end to end.

## 5. Data, and what is still open
- Done by the lead: compute4 now has `~/dojo-data/bg_folds_51_ilia` and `~/dojo-data/bg_folds_ilia` (table files
  md5-identical, plus compute3's `scene/`). Integration sets `DOJO_BG_FOLDS` to them. The shared
  `~/dojo-perception-data` stays untouched because Mikhail's pinned run reads it.
- **pick_from_conveyor_by_image has no genuine-5.1 belt background anywhere.** Its belt is `Dynamic/conveyor/00001`.
  Only 6.0-proxy leave-one-out folds exist, in `compute3:~/dojo-data/pbake-identity/bg_byimg{A,B}`. Every genuine-5.1
  run of that task is blind until a 5.1 `belt_bg record` + `build` exists. It waits for compute2 (BACKLOG).
- Next: the paired rendered re-run of matchbelt 100000-04 with the scene backgrounds (Integration).
- Open (BACKLOG): far-end identity (puppet -> camera) and position (flat phone); far-end velocity noise.
- Spend: 0 LLM calls.
  - compute3: ~10 min of 4 CPU processes x 2 threads, then ~35 min on light4 (flock): the DINOv2 arm and the
    260-440 reads.
  - B1 / belt_read render through pyrender EGL, so those processes also held a ~315 MiB graphics context on the GPU
    for ~5 min, outside a slot. Stopped when seen, and reported to the lead. There is no OSMesa on the box.
