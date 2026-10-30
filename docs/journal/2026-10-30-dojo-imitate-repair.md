# 2026-10-30: imitate's misread twin closes on air; `imitate_repair` (stream W)

Context: the dev row c4b_imitatesg_dev_home_100001 (#357, B perception + home + keep_sides) scored 0. The first
twin's pick closed on air 6 times. Was it a frame-0 localisation error on a far object, or something keep_sides made
sticky? Dev only, 0 LLM calls.

**What it was**
- B1's frame 0 read our watch (GT t2, (-0.202, -0.180)) as a garage, 17 mm off: est "t1", conf 0.53. So our side held
  two "garages": "t1" and our real garage "t3".
- The demonstrator's first object was its garage. Its one-to-one twin was the tie between "t1" and "t3", and the
  assignment took "t1".
- The pick was planned on a garage model where a watch lies, so the jaws closed on air ("rose 0 mm", grip check
  "empty"). The pick refines never fit: IoU 0.03-0.23, one 61 mm / 61 deg fit rejected by the low-IoU-turn rule.
- 3 tries, then a re-plan (the re-entry), then 3 more on the same twin.
- Cause: a near object, not far or occluded, so not the bundle's class.
- Not keep_sides: the twin assignment (`st["twins"]`) persists across re-entry anyway. keep_sides only kept the halves.

**What the flag does: `RuntimeConfig.imitate_repair` (default False; G3 skill attr)**
- A twin not put into basket0 after `REPAIR_TRIES` (2) is dropped as a misread or unreachable object (`st["bad"]`).
- The next candidate like the demonstrated object (same asset, else category) is tried.
- The demonstrator's objects still to copy are paired again among ours that are left.
- Off: today's behaviour (3 tries, then the copy stops).
- Expected on dev 100001: the garage is copied with "t3" (#1). The demonstrator's watch then finds no watch on our side,
  since ours is the misread "t1", so the copy stops at 1/5 (the first transition's credit) instead of 0.
- A misread whose correct twin exists elsewhere on our side (the tie case) continues further.
- What remains is the frame-0 identity of small near objects (watch vs garage / action_camera), the same limit noted
  in 2026-10-29-dojo-imit-detection.

Tests: `tests/dojo/test_skills6.py::test_imitate_repair_drops_a_twin_that_cannot_be_put_and_takes_the_next` (greedy /
one-to-one, on / off).
