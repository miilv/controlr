# 2026-10-28 — Stream P: pour_balls in camera mode — the whole roll profile (`ported_pour`, default v1)

**Found.** On the camera arm (compute4, the "sg" reports, pre-#278), pour_balls_into_vase scored 0/3, against 8/10 in G3.
- Perception was right: cup 1.3 mm, vase 2.2 mm.
- Every ball was "found: false". They sit inside the cup, hidden from every camera, so the belief has none.
- Mikhail's pour stops tilting at "cup empty": no believed ball inside the cup, from the first hold on (roll 88 deg,
  cup tilt ~82 deg).
- With no balls believed, that stop fired at once. The cup went back upright with the balls still inside
  (stages_left 1).
- In G3 the balls are ground truth, so the stop waited until they had left.

**Done** (branch `fix/dojo-port-pour-full`, on top of #278):
- `pour_p1` is a copy of robot-harness's pour without the "cup empty" stop. The whole roll profile (to 130 deg) always
  runs.
- The route uses it with `RuntimeConfig.ported_pour="full"` / G3 `--skill-attr ported_pour=full`; the default "v1" is
  unchanged.
- Test: a camera-mode belief with no balls gives v1 `pour_stop` "cup empty" and p1 "profile end".

**Spend:** 0 LLM, 0 GPU.
