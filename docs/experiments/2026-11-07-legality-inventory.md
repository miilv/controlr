# Legality inventory: main and sealed D-C19

This is a source audit, not a robot experiment or an eligibility certificate.
**Board entry is blocked.** The fixed launch command still selects task-specific
schemas, controllers, priors and checker-shaped predicates during inference.
No LLM calls, tokens, simulator jobs or remote box access were used.

## Reproducible scope

Main snapshot: `daae902cc90f89442e70df5770232b9fb25c73bb` (fetched before inventory).
D-C19 implementation: `127b65c4`, matching the partner audit; candidate provenance
`5509d7be`. Flags are the complete tracked
[`2026-11-06-candidate.txt`](../../scripts/dojo/queues/2026-11-06-candidate.txt)
from main, including `submit_v3`; an immutable [copy](data/legality/dc19-candidate.txt) accompanies this report. This avoids the truncated JSON limitation in the
audit. These are static inventories of both snapshots, not executed flag configurations.
The same scanner and declared roots cover planner, supervisor, runtime, agent,
perception and all local registered skills (including candidates). The legacy library
module is also a root so dispatch-table-only skills are followed.

Run `uv run python scripts/legality/scan.py --root CHECKOUT --output /tmp/report.json`.
For the historical snapshot, extract `git archive 127b65c4 dojo controlr skills` into
an empty temporary directory and pass it as CHECKOUT. The scanner itself comes from
this PR. Task terms are frozen audit data: all 42 canonical names, their random
variants (covering the 54-task collections), and vocabulary aliases. No task-schema
file is loaded by the tooling or the runtime tests.

## Counts and full inventory

Counts are `(file, line, enclosing symbol, category)` signals, merged across roots.
A line can have several categories. `checker_tolerance` overlaps `checker`;
`instruction_boundary` includes explanatory text as well as actual reads. Unresolved
calls are uncertainty, **not confirmed violations**. Attribute matching is conservative
and reaches some development/evaluation helpers; those rows need feasibility review.
These numbers must not be quoted as a count of proven board violations.

| Category | Main | D-C19 |
|---|---:|---:|
| unresolved | 34373 | 32779 |
| termination | 10 | 10 |
| checker | 220 | 210 |
| task_name | 513 | 498 |
| checker_tolerance | 69 | 67 |
| task_config | 127 | 127 |
| task_switch | 104 | 79 |
| benchmark_composite | 6 | 6 |
| instruction_boundary | 51 | 38 |

Every signal, including unresolved calls, is in the compressed CSVs:
[main inventory](data/legality/main.csv.gz) and
[D-C19 inventory](data/legality/dc19.csv.gz). Columns give file, line, category,
symbol, all reachable entry points and exact source evidence. Decompress with
`gzip -dc docs/experiments/data/legality/main.csv.gz`.

## Confirmed blockers to retire

| Mechanism / our location | Candidate activation | Board classification / audit crosswalk |
|---|---|---|
| `runtime/supervisor.py:VOCABULARIES`, `perception/j1_adapter.py:infer_task`, `runtime/policy.py:_setup` | Always: instruction becomes vocabulary/scene key | **Blocks**. Audit §1.6–1.8, §4.2. Replacing official task identity with an inferred identity does not cure routing. |
| `perception/inverse_layout.py:task_map`, `label_groups`, `instance_range`, `_placement_prior`, `fixed_slot_labels`; `runtime/vocab.py:load_sets` | Always task-schema path; `series_count=seg`, `pad_row_extend=true`, `series_labels=numbered` keep generator priors | **Blocks**. Audit §3.1–3.6: candidates, counts/quotas, placement ranges, slots, label prefixes. Removing only a wrapper or mask prompt leaves the deeper read. |
| `skills/planner.py:ported_plan`; `skills/ported/__init__.py:TIER_A`; `skills/library.py:ROUTE_ONLY` | `ported_routes=true`; swap=reach, drag=turn, pickup=library, pour=full, aim=b10, classify=p2 | **Blocks**. Audit §1.4–1.5, §4.1; glib findings 1–2, 7. Fixed per-family best variants remain policy routing. |
| `skills/routes.py:NEW_ROUTES`, `skills/planner.py:plan` | equation, largest_number, align, row_by_size, pour_order | **Blocks** for benchmark-specific recipes. Audit §1.3; glib findings 1, 6–7. A generic name alone does not make a complete benchmark solution general. |
| `skills/ported/harness_robot.py:HarnessRobot.cmd_obs` | Internal ported observation always includes `str(world.instruction or "")` | **Blocks** for a claimed explicit-argument boundary. glib findings 3–5, 20: instruction leaks below wrappers. A runtime negative control reproduces the read here. |
| `perception/select_ties.py:tie_mode`, `TIE_OFF` | Default `dc17` | **Blocks** under the new hard rule, despite historical D-C17 permission. Audit §3.7, §4.7. Task-keyed assignment cannot survive the general-system move. |
| `skills/align.py`, `skills/store.py`, `skills/artgeom.py` | Align route active; other skills reachable through registry/library | **Blocks** where checker-only goals replace instruction/physical goals (notably omitted set square and staged credit). Audit §2.4, §2.6; glib 6–7. |
| `skills/planner.py:waits_on_actor`, `runtime/policy.py:_setup`, `skills/library.py:WAITS_ON_ACTOR` | Instruction-recognized skill suppresses scan/look/search | **Blocks**. Audit §2.7, §4.3. Must be triggered uniformly by observed motion/safety, not task fingerprint. |
| `perception/agentic/vendor/PERCEIVER-PROTOCOL.md` | Agentic fallback when enabled by preset/runtime | **Blocks when used**: instruction permits checker rules, including blue presses. Audit §2.1, §4.6. Known prompt resources are explicit inventory roots; arbitrary dynamic file loading remains unresolved. |
| `perception/sam_routing.py`, `data/sam_task_banks.json`; RuntimeConfig `watch_relocate_task`, `key_scope_task` | Not enabled in D-C19 candidate; SAM routes diagnostic-only/default-off | **Blocks if adopted**. New hard-rule journal, audit §4.2/§4.7 pattern. These are current-main retirement debt, not evidence that D-C19 enabled them. |

Locations above are symbol anchors in our tree; exact per-snapshot lines are in the
CSVs (the audit's old line numbers do not match current main).

## Risky evidence requiring provenance review

- **Checker-shaped tolerances:** `skills/tiles.py` (`ON_PAD`, heading/reach banks,
  exact/nonnegative arithmetic), `rows.py` (uprightness/spacing), `pack.py` (facing
  and containment), `sweep.py` (collection/final arrangement), `garment.py`,
  `holders.py`, `insertion.py`, `precision.py:insert_each`, `regrasp.py` and
  `kinematics.py`. Audit §2.2–2.6; glib findings 8–11. Generic physical contracts
  are legitimate only with independent provenance; existing checker comments
  are evidence to retire/rejustify. Bare numeric literals cannot be certified
  mechanically. The `checker_tolerance` category is a provenance-review aid.
- **Near one-to-one benchmark verbs:** tower, kong, demonstration, game, align,
  row-by-size, pack and sweep. Audit §1.1–1.2 and glib findings 1, 6–7. Require
  parameter-complete geometric verbs and model composition; test transfer across
  unrelated layouts before promotion. The scanner labels recognized composites;
  it cannot decide semantic generality from a verb name.
- **Asset grounding:** globally shared mesh/description data can be legitimate;
  `ported/skills/colour_order.py:CATALOGUE_COLOURS` copied from task generators
  needs removal or independent pixel/metadata grounding (audit §3.9).
- **Termination:** the inventory surfaces `episode_over`, `env_end` and success
  latch mentions. Transport shutdown/resource exhaustion is not itself illegal.
  No item is cleared merely because it is named `ended`: trace whether it causes
  retries or completion feedback. Evaluator success/termination as a retry signal
  is prohibited. No live evaluator was used to establish dynamic feasibility.
- **Active perception:** tracking based on actual measured motion and held-pose
  reads based on uncertainty can be general. Existing task-conditioned candidate
  pools contaminate that argument (audit §4.4–4.6).

## Partner-review comparison cases and runtime evidence

The scanner fixtures exercise alias imports, dispatch tables, deeper helper calls,
unresolved callbacks, immutable exemptions and forbidden general-path exemptions.
They are deliberately independent of either team's library layout. Unlike the old
MATCH2 wrapper-only test, they require a leak in a deeper callee to appear at the
original root. Comments/docstrings and long strings are included (glib 4–5).

Runtime tests run perception-mode episodes on the kinematic mock through both the
rule planner and FakeLLM agent, covering stacking/nesting and explicit arrangement.
A poisoned task-identity object cannot be printed, compared, hashed or tested for
truth. Instruction text is available to planning. Schema functions are trapped by
code identity/name at call time, so pre-imported aliases do not evade the guard;
file reads are separately guarded and swallowed failures remain sticky hits.

Two explicit negative controls document current violations: unknown instruction
routing enters `load_sets`, and internal `HarnessRobot.cmd_obs` reads the instruction
sentinel. They assert the precise forbidden access, rather than using broad xfail.
They do not make these paths acceptable. The external perception mock cannot prove
that the real service is schema-free; the static inventory already shows it is not.
Only representative skills are exercised; full reachable-path coverage is not claimed.

Glib findings 12–19 (cardinality, duplicate/self pairs, movable roles, default frames,
NaN, false look success, overstated outcome contracts and occupancy geometry) are
correctness review cases for the shared library, not additional proven defects in
our snapshot. Do not translate partner line numbers into claims about our code.

## CI and shared-tool handoff

The inventory job is report-only. The separate ratchet forbids new evidence hashes
and allowlist growth; new declared general-system directories cannot be exempted,
even through legacy dependencies. Existing exemptions are debt, not board permission.
See [the development policy](../DEVELOPMENT.md#inference-legality-inventory-and-ratchet).
The initial allowlist is exactly this main snapshot's findings, including unresolved
calls. Shrink it as the listed code paths are retired.

Compare the partner implementation on these same fixtures and mock negative controls
when it arrives; retain one tool in robot-harness after the repo move. No second
library architecture or deployment path is introduced here.
