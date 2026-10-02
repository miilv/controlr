# 2026-10-02 — key architecture decisions for v0

**Decisions (and why):**
- **The model is the controller, no skills and no VLA.** The project's thesis: large API models
  with the right prompt are enough; the literature review's recommendation "the LLM picks
  skills" was deliberately rejected; alternatives are config switches only.
- **A turn loop, not real time.** The robot holds still while the model thinks; ~1 s/turn is a
  target, not a constraint; a long first "thinking" call is fine → a separate planner
  (`claude/claude-opus-5-5-xhigh`) whose text goes into the first turn.
- **Python from scratch, not a fork of a coding harness.** Pi (TypeScript, an excellent provider
  layer) would need a bridge to the robot's Python stack; Hermes (≈490k lines, compresses
  history) is too heavy and breaks append-only. At the start the core was estimated at ≈1–2k
  lines (by the end of the day the whole package had grown to ≈8.8k with the Isaac backend and
  safety); Pi and Hermes served as references (cache-marker placement), no code was borrowed.
  Rust buys nothing: a turn is bound by the network and the model (0.5–5 s), not the harness.
- **One OpenAI-compatible router (omniroute)** instead of provider SDKs; Claude caching through
  it is verified (≈95 % read).
- **Isaac Sim with PHANTOM's scene, not MuJoCo.** A calibrated copy of the real rig (UR3 CB3,
  Robotiq, D435) → conclusions about prompts/representations transfer to the hardware. The
  MuJoCo scene was dropped.
- **Default actions: `ee_delta` in mm, fixed orientation, binary gripper**; everything else
  (`ee_abs`, joint modes, `rotation=yaw/full`, chunks) is a config axis.
- **Default control model: `claude/claude-sonnet-5-5`** (Ilia's request; works through the
  router although it isn't listed in `/models`).

**Not done:** see [BACKLOG](../BACKLOG.md) P1 — cameras, carry corridor, arm–box collisions.
