# 2026-10-27: make_kong in mode B on genuine 5.1: why topples were never confirmed, and what still fails

Long-Horizon stream (journal [2026-10-27-dojo-longh](../journal/2026-10-27-dojo-longh.md)). Question: breadth2 (proxy B)
had make_kong at 2/5, with "did not tip over" in the losses although #70 sends every topple's expected pose to
`provider.confirm_pose`. Why were the confirms rejected, and what is left once they are not?

**Setup.**
- Replay: `scripts/dojo/replay_kits.py run --save-obs-at 120,141,170,197` of breadth2's 100001 action list on the
  genuine 5.1 rendered kit (compute8); true object poses per step and the contract frames at the confirm steps.
- Offline re-check: perception service `check` (CPU silhouettes) on those frames, the episode's belief record, prior
  (standing) vs expected (toppled) pose of the tile.
- Episodes: genuine 5.1 B on compute8 (`integ_eval.sh --renderer isaac51 --preset submit_v1`), settled dev layouts,
  one repeat. Run dirs `compute8:~/dojo-data/longh/runs/B51_kong*`.

## The confirm

- The topple worked physically: the tile lies flat from step 135, 2 mm from the expected pose.
- At the confirm (step 141) the toppling arm's gripper hangs over the tile and covers it in the head view. The runtime's
  head precision read 0.96 (standing) vs 1.0 (toppled), below its +0.15 margin.
- Silhouette IoU prior vs expected on the replayed frames: head 0.59 vs 0.57 (step 141), 0.62 vs 0.66 (step 170); the
  toppling arm's own wrist camera 0.075 vs 0.93 and 0.08 vs 0.92.
- A skill-side second look with the arm sent home (`confirm_moved(arm=)`, branch `feat/dojo-longh-kong`) did not help on
  5.1 frames: with the head view unoccluded it still read standing 0.91-0.96 vs toppled 0.98-0.99 (100000-02: 0/3, the
  same as main's 0/5 on 100000-04 with pre-#102 / #105 code). Not merged.
- The fix is runtime-side: #105 (Integration-3) lets a wrist view with IoU >= 0.6 and >= prior + 0.15 confirm.

## Results

| code | layouts | success | score | failures |
|---|---|---|---|---|
| breadth2 (proxy B, integ7) | 100000-04 | 2/5 | 40.0 | "did not tip over" 2, perception roles 1 |
| main 48600bf (5.1 B) | 100000-04 | 0/5 | 0.0 | "did not tip over" 5 |
| main a3dc7e1 (#102 no scan / looks during the discard, #105 wrist confirm) | 100000-09 | **6/10** | 60.0 | topple plan collision 4 |
| main 13a28b6 + `snap_row` (#141) | 100000-09 | **10/10** | **100.0** | none (steps 337-384 of 600) |

The four losses on a3dc7e1 are the same planning failure: "push collision: waypoint 1/7: hits <neighbour>". Perception
puts a hand tile up to 11.5 mm off its slot on 5.1 frames (p90 11.5 mm; 100009: mahjong0_2 believed at x -0.2215, slot
-0.233), into the 13 mm gap the topple's fingertips need. `snap_row` moves the standing hand row onto its fitted slots
in the planning world (the belief is untouched) before each topple plan: the 4 collision losses are the 4 rescued
layouts (the two bases differ only by unrelated merges). G3 (GT): mode A 10/10, mode B `--confirm-pose` 10/10, unchanged.

## Conclusions (n = 10 rendered, one repeat)

- The topple confirm needed a view that sees the toppled tile: the toppling arm's wrist (#105), not the head view, which
  cannot tell standing from toppled on 5.1 frames even with the arm away.
- With #102 + #105 + `snap_row`, make_kong is 10/10 on genuine 5.1 B (from 0/5 on the same renderer before).

**Spend.** LLM calls (agentic perceiver): 2-4 per episode on a3dc7e1 (no looks before declare_kong since #102), 8 per
episode before it. GPU: compute8 heavy lock.
