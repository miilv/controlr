# 2026-10-27 — SAM proposers, guarded: which click model, flip / cover guards, speed, and the config for a rendered A/B

Follow-up to [2026-10-27-sam-bakeoff](2026-10-27-sam-bakeoff.md). Same scenes, same B1, same scorer. Journal:
[../journal/2026-10-27-dojo-sam.md](../journal/2026-10-27-dojo-sam.md) §4.

## Question

The first batch had two regressions. With them fixed, does REAL_RSI's mask proposer still help, and which arm goes
to a rendered A/B?

- The proposer flipped 4 lying bottles by mask IoU alone. Now its 180-degree flip needs the texture to agree.
- `trim` cut merged blobs. Now it cuts only blobs the masks cover by >= 50 %.

The click models compared: SAM 2.1 small / large, SAM 3.1.

## Data and protocol

- **Scenes:** 222 frame-0 scenes, 1098 scored objects (d0_51 92, d0_ext 115, deposit_coin 10, store_laptop 5; ids
  100000-09 / 100020-49; details in the first report). sweep_blocks is left out (its count fix is in the first report).
- **Baseline:** B1 = main with all SAM flags off: 961 within tolerance / 85 gross / 25 missed / 49 phantoms.
- **Runs:** one flag changed per arm, eval_g1's scorer (`scripts/perception/sam/bake.py`). 4 CPU workers on compute4,
  the SAM server in light1.
- **The proposer:** for every chosen object, SAM clicks at its head-view centre + 4 offsets (half its projected
  radius). The object is re-fitted (same shape, local) inside each mask 0.3-3x its silhouette. The fit whose
  silhouette agrees best with its mask wins, B1's own pose included, by >= 0.05 IoU. The group's other shapes (<= 8
  candidates) are tried in the winning mask, and B1's flip only where the texture agrees.

## Results (paired, 1098 objects)

| arm | within tol | gross | missed | phantoms | fixed / broken |
|---|---|---|---|---|---|
| B1 | 961 | 85 | 25 | 49 | - |
| `sam_propose=sam2.1_small` | **988** | **70** | 25 | **38** | **28 / 1** |
| `sam_propose=sam2.1_large` | 989 | 71 | 25 | 39 | 31 / 3 |
| `sam_propose=sam3.1` (tracker clicks) | 984 | 74 | 23 | 40 | 28 / 5 |
| `sam_fg=trim` (cover >= 0.5) | 979 | 82 | 18 | 45 | 26 / 8 |
| `sam_propose=sam2.1_small` + `sam_fg=trim` | 997 | 73 | 18 | 39 | 45 / 9 |

Per task, SAM 2.1-S proposer (within, B1 -> arm):

| task | n | B1 -> proposer | notes |
|---|---|---|---|
| stack_bowls | 45 | 38 -> 43 | phantoms 7 -> 2, identity 8 -> 5 |
| build_tower | 120 | 100 -> 104 | gross 14 -> 12 |
| align_blocks | 40 | 34 -> 37 | |
| stack_blocks | 45 | 35 -> 38 | |
| pack_objects_into_box_random (clutter) | 75 | 52 -> 55 | gross 20 -> 18 |
| organize_table | 81 | 61 -> 63 | gross 14 -> 10, phantoms 13 -> 10 (SAM 2.1-L: 66) |
| store_laptop_and_headphones (genuine 5.1) | 25 | 16 -> 18 | headset identity + pose 2 of 3, headphone stand 5 of 5 to 1-4 mm |
| plug_in_charger | 20 | 15 -> 17 | |
| deposit_coin (shadows) | 30 | 15 -> 16 | the piggy bank's shadow-biased fit; the coin / stand phantoms stay (SAM 3.1 clicks: 18) |
| put_bottles_into_dustbin | 85 | 76 -> 76 | 1 fixed, 1 broken (the flip guard took the other 3 regressions away) |
| all other tasks | 507 | unchanged or +1 | |

- **Click models:** all three proposers land within 5 objects of each other. SAM 2.1-S breaks the fewest (1) and is
  the smallest. SAM 2.1-L does better on organize_table but breaks 3. SAM 3.1's tracker clicks break 3 of the 5
  laptop-scene objects.
- **Trim:**
  - It recovers objects B1 missed (25 -> 18: put_bottles 3, deposit_coin's piggy bank) and fixes cast-shadow fits
    (stack_bowls).
  - It still breaks 8, on build_tower, pack and store_tools: a mask covering half a blob cuts a neighbour's pixels.
  - With the proposer on top, +36 net but 9 broken. It stays off, as decided.
- **What the proposer doesn't do:** find missed objects (missed 25 -> 25) or move a far phantom (deposit_coin's
  580-805 mm coin / stand reads). It refits locally around what B1 chose.

## Speed and VRAM (RTX 5090, bf16, one model per process; `scripts/perception/sam/bench.py`)

| model | load | peak VRAM | 5-click proposal per object (incl. the image encode) | text, 8 prompts per scene |
|---|---|---|---|---|
| SAM 2.1 small | 2.1 s | 0.55 GB | 86 ms | - |
| SAM 2.1 large | 1.4 s (cached) | 1.35 GB | 114 ms | - |
| SAM 3.1 | 17.6 s | 4.70 GB | 137 ms | 230 ms |

- **Cost per scene:** the proposer adds a median 1.3 s (p90 4.2 s) on CPU, mostly the refits. B1 alone takes a median
  3.4 s per scene (1-70 s by task).
- **Client:** the estimator's side needs no torch. `dojo.perception.sam_masks` imports numpy only and talks to the
  server over a unix socket.

## Recommendation: a rendered A/B of the SAM 2.1-S proposer

```bash
# once per box (torch cu128 + Meta's sam2 / sam3 code + weights, ~2.8 GB):
scripts/perception/sam/setup_venv.sh ~/controlr-sam
# next to the perception service (one more GPU process, 0.55 GB VRAM; part of the episode's footprint):
~/controlr-sam/.venv/bin/python -m dojo.perception.sam_masks serve --sock ~/dojo-data/sam/sam.sock --models sam2.1_small &
# the perception service's environment:
export DOJO_SAM_ADDR=$HOME/dojo-data/sam/sam.sock
export DOJO_PERCEPTION_CFG='{"sam_propose": "sam2.1_small"}'
```

- **Expected effect:** fewer gross pose / identity errors on found objects (frame 0: +27 within, -15 gross, -11
  phantoms per 1098 objects), none on missing objects.
- **Where to look first:** stack_bowls, build_tower, organize_table, pack_random, store_laptop.
- **Without a server:** the flag is a no-op (`_perception.sam.propose.error`).
- **Not measured:** the moving belt. The belt frames are on compute3 only.

## Spend

- 0 LLM calls.
- GPU: the SAM server in compute4 light1 (stopped after the runs) and a 1-min bench.
- Runs: `~/dojo-data/sam/v_prop_s2`, `v_prop_l2`, `v_prop_31b`, `v_trim2`, `v_combo2` (vs `b1` + `s_b1`) on compute4.
