# 2026-11-07 — MATCH2 camera screen dispatched on compute6

The lead approved the 10-pair diagnostic screen ([pre-registration](../experiments/2026-11-07-match2-screen.md),
on main as `2a8cd037` before any queue write). It is not an adoption test.

- **Deployment:** merged main `ec865e93` (contains #420) at `compute6:~/dojo-integ-c6-ec865e93`, by the KONG / POUR
  recipe: `git archive` of the commit, an isolated Python 3.12.13 venv from the lane interpreter with the segab
  requirements (own uv cache under `~/dojo-data/match2-cam`), package parity `diff={}`, candidate parity true,
  catalogue gate OK, `.deployed.json` with weights and lane hashes. Resolved RuntimeConfig against POUR's `066e9c8f`:
  203 existing fields unchanged; the 8 new fields default off; A vs defaults: `save_belt_look_frames` only; B vs A:
  the six switches only.
- **Inputs:** the screen block copied from compute3 through the agent box, SHA256-verified byte for byte (11 / 11
  files), raw-checked again on compute6 under an admitted free light lock in a 20 G / 600 % scope: 10 / 10 raw ok,
  records unchanged. The conveyor runs RAW records, so there is no settle pass.
- **Insertion (14:40:43Z):** 20 rows at positions **2-21** of `~/dojo-integ-c6-state/queue2.txt` under
  `wave-start.lock` then `queue2.txt.lock`. The started tie pair's queued mate `c6b_imitatesg_tieBr1pb1_100001`
  stays first; the next unstarted tie `c6b_imitatesg_tieBr2pb1_100001` follows the block. All 43 old rows keep bytes
  and order. Every row pins `AGENTIC_MODEL=claude/claude-sonnet-5-5`, the deployment and its frozen layout hash.
  Queue SHA256 `225f7d71…` -> `3cf66d20…`; manifest `d5b56cc8…`. Rough ETA: start ~14:50Z, finish ~16:00Z.
  [Receipt](../experiments/data/match2-screen/dispatch.json).
- Preparation made zero model calls; no process was killed; nothing outside `~/dojo-integ-c6-ec865e93` and
  `~/dojo-data/match2-cam` was written except the queue insertion itself.
