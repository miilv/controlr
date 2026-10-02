# 2026-10-01 — literature review: LLMs as robot controllers

**Done:** a multi-agent teardown of Ilia's 14 sources (Robocurve GPT-6 Astra, RoboDojo,
GPT-as-Policy, EmbodiedSWE, quackd, innate-os PR#817, …) + 8 landscape sweeps (frontier models,
VLAs and action heads, LLM planners, direct control, benchmarks, industry, engineering
challenges, a catalogue of open-source harnesses), an adversarial fact-check of every note,
6 gap studies. Everything is in [research/](../../research/README.md): `REPORT.md`,
`HARNESS_DESIGN.md`, `sources/`, `landscape/`, `gaps/`.

**Decisions / findings that shaped the project:**
- The bottleneck is waiting for the model, not "intelligence": on real rigs Opus 5.5 ≈ 5.3 s p50 per call.
- Direct numeric control by a model has been tried (Robocurve, RoboDojo's RoboProbe,
  GPT-as-Policy), but not in the regime "short replies, no reasoning, the whole history cached,
  ablations of the frame representation" — that is controlr's niche.
- RoboDojo: Astra #7/51 (22.48 % SR), near zero on precision; without the head camera 6/8,
  without head + wrist 3/8.

**Found along the way:** unrestricted subagents filled the disk and spent money —
[incident](../incidents/2026-10-02-research-run-side-effects.md); StructuredOutput is unavailable
to workflow subagents.

**Next step:** the harness (see [2026-10-02-design-decisions](2026-10-02-design-decisions.md)).
