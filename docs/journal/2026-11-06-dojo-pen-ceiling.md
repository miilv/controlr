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
