# 2026-10-28: stream N, fixes after #268 (tile carry, dolls rows, align free rows, clutter on a pad)

compute7 was closed (VRAM incident). These G3 runs are on compute3: one job at a time, under `timeout -k`, starting
only at MemAvailable ≥ 12 GB, on a box with load ~20. That makes them 5-8× slower than on compute7.

## Changes
- **tiles**:
  - The carry uses the uneven-object speed (0.09 rad/step instead of CARRY 0.14), a 4 cm lift, and a 12° tilt next
    to the 25° one. A frictionless numeral slid 52 mm in the pads and fell 13 cm off its pad (v0 100001, ceiling run
    at dd0f9bf).
  - A clutter object on a pad now blocks it like a tile does (arrange_largest_number_random).
- **rows** (dolls):
  - Rows built around the widest doll where it stands: it fits the open gripper worst, and moving it costs 0.4 m more.
  - The row is filled outward from that doll (else from the left end). The last doll of 100030 went into a gap
    between two placed ones and its approach hit one.
  - 30 mm gaps; rows near the middle of both arms' reach cost less.
  - No in-air handovers: a doll held 53 mm off its axis in a handover slid out (100032).
- **align**: free rows that move every block when no anchored plan's slides plan (v0 100002: every slide crossed
  another block).

## G3 mode gt (dev = v1 draws of 100030-39; compute3)
| task | commit | n | success | before |
|---|---|---|---|---|
| solve_equation | eec3a85 (tiles = head) | 10 | 9 | 18/18 at 004b501 on 100030-49 |
| sort_nesting_dolls | 1ed92d3 | 10 | 8 | 1/3 (e7626c8), 0/2 (1911fbc) |
| arrange_largest_number | 1ed92d3 | see PR | | 2/6 (094d219, bbox meshes) |

The remaining failures:
- dolls: ik_fail on the last one or two dolls (100033, 100034);
- equation: a cross-table tile with no grasp-place pair (100038);
- arrange: an "8" numeral flung off the hand during its carry (100030).
