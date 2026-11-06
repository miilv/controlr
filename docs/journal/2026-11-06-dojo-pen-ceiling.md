# PEN: fill_pen_holder ceiling work

## Scope and resource envelope

Physics-only G3, perfect perception, 0 LLM calls/tokens, compute6 as robotics.
Deployment `~/controlr-pen`, data `~/dojo-data/pen`. At most two light locks
(`~/dojo-gpu-light.lock`, `~/dojo-gpu-light2.lock`); no camera lock. Every process
checks used VRAM <=24576 MiB after acquiring its lock, before starting Isaac.
MemoryMax=10G / CPUQuota=600%. No compute4 or dc19 data access, no system changes.
Local environment, tests and source stay below 1 GB.

Read the fill-pen history first: lift clear 12.5→19.0, measured joints 19.0→24.7,
upright-first 24.7→26.0 (null); those are not new camera candidates. Historical
camera rejections of lift/joint-limit arms remain rejected.

## Baseline replication and numbered failures

Base on settled development IDs 100000–04: **34.0 points, sample SD 37.6,
1/5 success**, per-ID scores **25/100/10/25/10**. It exactly reproduces the
historical 34-point ceiling. Steps median 491, max 619/1100; wall median 278.9 s,
planning median 202.2 s. Files `base5/fill_pen_holder_episodes.jsonl`, action
trajectories and failed-episode traces under the data root.

Final per-pen outcomes, 20 total, taken from each pen's last `[fill]` event:

| Outcome | Pens |
|---|---:|
| Inside | 10 |
| Released outside / place miss | 4 |
| No grasp-place transfer | 2 |
| Grasp slip | 4 |

Pen/00000 accounts for **all four final slips**, 0/4 inside, and nine failed
pick attempts. The two nearly identical rolled box pinches repeatedly rise
0–3 mm despite an 80 mm commanded lift. By asset: oil_pen/00001 5 inside,
2 misses, 1 unreachable; oil_pen/00000 2 inside; pen/00001 2 inside;
pen/00004 1 inside, 2 misses, 1 unreachable. No holder ends tipped in this
five-layout baseline. This is a diagnosis of these layouts, not a claim that
tipping is absent on other draws.

## Exploratory screens and mechanisms

- Vertical-only narrow pinches at five stations: 17.0 mean, 0/5 success,
  scores 40/25/10/10/0 versus A 34.0. Final outcomes 7 inside, 10 unreachable,
  3 misses, one failed pick. Rejected: reduced slip traded away too much reach.
  This was an exploratory uncommitted implementation, superseded before the
  frozen paired validation; it is not an adoption measurement.
- The replacement `fill_grasp=vertical` prefers those pinches but retains every
  original candidate at +2 score cost. Width is projected into horizontal jaw
  coordinates, so a rolled pen does not use an incorrect body-frame width.
  Each station has a distinct retry note. `fill_grasp=rolled` restricts the
  preference to horizontal shafts whose body faces are >10 degrees from flat;
  flat-object candidates and scores stay exactly unchanged.
- `fill_holder=stay` bypasses relocation. `fill_drop=center` uses only the
  maximum-clearance fit with occupied-space padding, refusing a full opening
  instead of progressively ignoring already inserted pens.
- `fill_insert=safe` scopes the measured joint range to the fill operation
  (restored on exceptions), lowers the servo hover clearance to 3 mm and enables
  the existing 4 mm xy correction. It does not introduce new reads.
- An older all-switch screen used the generic `fingers_out` tail as well. It
  completed 100000–02 at 10/25/0 versus A 25/100/10, then was stopped during
  100003 by checked PID 162068 (exact module and output directory checked before
  TERM/KILL). Only completed episodes are counted. One holder tipped. The tail
  raised some leaning release targets despite a planned insertion; it was
  removed before final paired validation. The failed screen is not evidence for
  the replacement. Data `combined5`.

## Frozen validation design

Ten fresh generator variant-2 draws, IDs 100030–39, generated and presettled in
`pen/simlab/layouts/fill_pen_holder` without changing shared records. All ten
pass both stability passes. No 100005–09, 100020–29 or >=100100 was used for
training, tuning or diagnostics. Raw, settled records and index remain together.

A uses default switches; B uses vertical preference + stay + center + safe.
Both use the same layout files, G3 entry point and physics configuration.
Both final arms set OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1.
Earlier exploratory screens used host thread defaults; their wall times are not
compared to the final arms. Code freeze f3e04c28; 0a0915da adds the independent
rolled mode without changing the vertical/off arms' behavior.

Reproduce a final arm from the deployed checkout (under one of the light-lock
wrappers with the admission checks above):

```sh
scripts/simlab/rdpy.sh -u -m dojo.skills.g3 run --task fill_pen_holder \
  --seeds 100030-100039 \
  --layouts "$HOME/dojo-data/pen/simlab/layouts/fill_pen_holder" \
  --trace --out "$HOME/dojo-data/pen/base10"
# B adds:
# --skill-attr fill_grasp=vertical --skill-attr fill_holder=stay
# --skill-attr fill_drop=center --skill-attr fill_insert=safe
```

Rendered validation is separately pre-registered in
[the camera specification](../experiments/2026-11-06-pen-camera-prereg.md).
No camera adoption is claimed from G3.

## Further diagnostics after baseline replication

The rolled-only screen on 100000–04 is exactly null: 25/100/10/25/10,
34.0 mean, 1/5 success. Retaining the original candidates prevented the
vertical-only reach regressions, but did not fix pen0.

A CPU IK probe at the recorded pre-pick pose of pen0 on 100002 explains why.
Of ten horizontal-jaw, straight-down candidates, 0/10 reach with either arm.
Tilting the approach while keeping its closing axis horizontal gives left
10/10 at 25 degrees and 10/10 at 40; right 4/10 at 25 and 10/10 at 40.
These counts are IK feasibility only, not collision checks or physical grasps.
`fill_grasp=rolled_tilt` adds those approaches, still only for rolled horizontal
shafts, with the original candidate fallback. The flat-object behavior and
all previous arms are unchanged. First physical probe: development ID 100002.
The probe code and logs live in `~/controlr-pen/pen-grasp-diag.py` on compute6.

The grasp-plus-insertion screen (`precision5`) completed 100000–02 at
40/10/25 versus A 25/100/10. It was stopped during the next episode after
checking its exact module/output-directory PID; no unfinished episode is counted.
This is a rejected exploratory screen, not a completed five-layout comparison.

Local full suite: 1795 passed, 187 skipped (18m48s); additional rolled/tilted
focused cases pass. Initial parity failures came from adding configuration
fields while a suite was already running (old in-process module versus new
subprocess source); all 36 affected/focused tests passed on rerun. CI passed
on the first pushed PR head. Pyflakes reports only the pre-existing unused
imports and placeholder-free f-string in library.py, no new undefined names.

## Completed fresh-layout comparison

Defaults: **6.5 ± 8.2** (sample SD), 0/10 success. Rolled-only: exactly the same
scores and step counts on all ten, 0/10 success. Combination (vertical preference,
stay, center, safe): **18.0 ± 11.8**, 0/10 success; paired gain **11.5** points,
paired-delta sample SD 14.9, 6 score wins / 1 loss / 3 ties. Full-success flips
0/0. Holders tipped at the final trace frame: 4/10 → 0/10. This is a real
partial-score gain on this development cohort, not camera adoption or proof of
a higher original-five full-task ceiling. Per-ID table and failure classes:
[experiment report](../experiments/2026-11-06-pen-g3.md).

The tilted preference's 100002 physical probe remained null (10 points): despite
reachable pick IK, the transfer planner chose the old slipping direct pinch.
`rolled_relay` excludes that fallback only while the shaft is rolled, to test
whether the existing relay search can pair the better pick with a drop.
`fill_drop=inset` uses 7 mm wall packing before centered fitting, retaining the
occupied-space constraints in both attempts. These are separate default-off
screens; the completed combination result above remains unchanged.

The strict relay probe on 100002 still scores 10, but **physically picks pen0**:
`top vertical axis 2 tilt 40 station +24mm`, pick ok, drift 2.4 mm. The table
relay then refuses a descent whose palm would reach z=0.753 (below the table).
It puts the object down through regrasp and banks no additional tier. This is
proof of an improved pick, not a task-score improvement. `rolled_stage` adds a
20 mm release gap plus the existing short-drop allowance to intermediate relay
legs only. Its final insertion is unchanged. The staging option is covered by
planning/execution forwarding tests, including default None.

Local second full-suite run: **1800 passed, 187 skipped, 17m36s**; focused tests
cover subsequent relay options. The final PR CI must validate the final source.

The short-drop staging probe on 100002 completed a pen0 pick and table relay,
then missed the final holder drop (53.4 mm final xy error, large post-release
rotation); score stayed 10. Adding stay + inset + safe to that stage mode scored
0 on the same development layout: reachable pick candidates still had no complete
relay-to-holder path. These single-layout probes are rejected, not adoption
measurements. No further source variants are introduced after e9428765.

Rejected `rolled_relay` / `rolled_stage` and their generic intermediate-place
option were removed from the final implementation. Their code is reproducible
from commits 69505043 / e9428765 respectively. The retained production source
returns to the positive-arm implementation at 42d84ea4; the failed probes remain
in the results. This cleanup introduces no new experimental behavior.
