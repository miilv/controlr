# 2026-10-25 — perception R&D: frozen agentic perceiver runner (D-C6), failure taxonomy, ranked options

Report with all numbers: [../experiments/2026-10-25-perception-options.md](../experiments/2026-10-25-perception-options.md).

**Done**

- `dojo/perception/agentic/` on `feat/dojo-perception-rnd`:
  - `j1.py`: the J1 interface (`perceive`, `held_look`; caller side stdlib + numpy, worker in our venv).
  - `libero.py`: LIBERO picks.json per capture.
  - `runner.py`: the agent loop, JSON-text replies.
  - `tools_rd.py`: the protocol's scripts in a per-call workspace + `locate` / `project`.
  - `prompt.py`: preamble + vendored protocol, md5-checked.
  - `seed_refine.py`: H2, B1's render-and-compare from the agent's points.
  - `flag_model.json`: calibrated risk for the budget gate.
  - `llm.py`: httpx client or a stdlib streaming fallback.
  - `dev_d0.py`: the dev measurement on our D0.
  - Tests: `tests/dojo/test_agentic_perceiver.py` (FakeChat, fake HTTP server, fake LIBERO helpers).
- Vendored from robot-harness @ 444e49f, byte-identical: `docs/PERCEIVER-PROTOCOL.md` (b668358, md5 c12d7875...),
  their perceiver tools, and `j1_schema.py`. They are Mikhail's side's files (D-C6: they own the protocol, the tool
  contract and the measurement).
- Installed pinned on isr-lab: `~/data2/ilia/agentic/` (e2ec222, 1.8 MB, README with the env). It reuses the J1
  venv; nothing was run there and the omniroute key was not installed.

**Decisions**

- **Replies are one JSON object in plain text, with no provider tool calling.** Tool results come back as user turns,
  because images cannot travel in tool messages. This works the same on claude/, cx/ and auto/ routes.
- **The agent never types a pose by eye.** Every number comes from `locate` or the protocol's scripts. A `keep: i`
  entry adopts B1's object i unchanged.
- **B1 runs first as the hint, and is the fallback when the agent gives no answer.** held_look is B1-first, with the
  agent only when B1 is unsure (confidence < 0.5 or dark_risk: J1b's 5 operator asks).
- **The budget gate (`AGENTIC_GATE=budget`):** the agent on the episode's first perceive call and at most once more,
  when B1 misses a wanted object or a calibrated risk is >= 0.2. Otherwise B1 tracks with the prior. The default stays
  `always` for Misha's measurement.

**Found along the way**

- **Sonnet on claude/ emits `<invoke>` / `<parameter>` markup** and can loop to max_tokens. It is now parsed and
  stopped early, and DEVELOPMENT.md has the pitfall.
- **`dva/gemini-*` answers 401; `auto/gemini` works but is slow** (about 150 s per perceive call, 5 calls).
- **My first dev pass of the push_T / stack cases ran without the B1 hint.** b1.json had not been synced locally yet.
  Those outputs were kept as the `*_nohint` ablation (agent alone); the hinted pass was rerun.
- **B1 flags 65 % of objects against 26 % actually wrong.** The calibrated model at threshold 0.2 flags 25 % with
  recall 0.77.

**Numbers** (report §3)

- Sonnet 5.5 with the B1 hint, B1's unmentioned objects kept, and H2, on 28 hard D0 scenes (102 objects):
  gross 24 -> 11 (13 fixed / 0 broken), within tolerance 63 -> 66, 3 LLM calls and 35 s per call
  (duplicate-safe merge).
- Sol: 19 gross, 7 fixed / 2 broken. Astra-medium: 18 gross, 10 / 4, within tolerance 66. auto/gemini (12 scenes):
  13 -> 6, but 150 s per call.
- The 6 push_T gross errors left are invisible at frame 0.
- Spend: about 486 LLM calls, 6.0 M prompt tokens.

**Not done / next step**

- The dark-object held_look case is unmeasured: we have no held-object frames of our own. Next: simlab renders of
  dark blocks in the jaws, then B1 vs agent.
- The park-arms scan pose and the episode-level tracking filter (report §4.1–4.2).
- SAM2.1 mask proposer and the synthetic-render detector belong on tomorrow's big-disk boxes.
