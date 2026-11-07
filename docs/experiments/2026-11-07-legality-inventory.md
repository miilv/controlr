# Legality inventory: our main and D-C19 using the shared partner tool

**Board entry is blocked.** This is a static inventory of inference dependencies,
not a scored experiment or an eligibility certificate. Zero live LLM calls/tokens,
simulator jobs, robot access or remote box access.

## Shared implementation and reproducible scope

The base is `dojo/legality/audit.py`, **shared with the partner team**, from their
`night/JOINT-PLANNER` commit `4c4d36db`, delivered in the 18:09Z bundle. Original
source SHA256: `334f617d48de0a803247c64c8d92006cea62da1c98e18213ea6094fab9f5af48`.
Our separate prototype scanner was removed. `scripts/legality/scan.py` is only
report formatting and a CI debt ratchet over this shared engine.

Main snapshot: `27eb254521d66cf50123d730afc95a43b2e13850`.
D-C19 source: `127b65c4` (the exact partner-audited source), candidate provenance
`5509d7be`. Complete flags: [immutable candidate](data/legality/dc19-candidate.txt)
and [parsed overrides/hash](data/legality/candidate-config.json), including submit_v3,
ported routes and all five new_routes. This removes the audit's truncated-JSON limit.

`dojo/legality/profiles.py:controlr_inference` indexes our `dojo/` and `controlr/`
code without importing it. Main has 527 declared roots, D-C19 502: planner, supervisor,
runtime entry/setup, model-agent runtime, perception, all local registered skills,
every callable/module in the path-loaded ported subtree, known adjacent file-loader
targets, all cmd_* handlers and the internal observation boundary. Imported
global dispatch tables and known agent/perceiver prompt resources are also scanned.
This is a conservative superset of candidate reachability, not a claim that every
flagged branch ran. Candidate overrides are recorded, never executed. Old source
line numbers in either review must not be applied to current main.

Reproduce by extracting `git archive REF dojo controlr skills scripts/skills/articulation_links.py` into an empty
CHECKOUT, then running:

```bash
uv run python scripts/legality/scan.py --root CHECKOUT \
  --candidate docs/experiments/data/legality/dc19-candidate.txt --output /tmp/inventory.json
```

The shared engine's direct CLI is `python -m dojo.legality.audit --profile
controlr-inference --json`. Its default profile here is our tree; the upstream
`controlr-glib` profile remains available when the partner library is present.
No glib code, models or environment were imported/copied into inference.

## Inventory counts

Each count is one `(function, reason)` signal with a concrete file/line and call
path from its entry point. Task names include snake, space, dash and Camel forms;
strings/docstrings have no length exemption. A line can have multiple categories.
`checker_tolerance` overlaps checker provenance. Instruction text is allowed to the
planner: an instruction-read signal there is **not** itself a board violation.
Unresolved external/dynamic calls and reflection are review debt, **not confirmed
violations**. Member-name overapproximation can reach development helpers; provenance
comments are review evidence. These counts must not be reported as proven board violations.

| Category | Main | D-C19 |
|---|---:|---:|
| reflection | 53 | 39 |
| unresolved | 28172 | 26174 |
| checker_module | 1 | 1 |
| task_data | 101 | 96 |
| checker_provenance | 175 | 165 |
| checker_tolerance | 60 | 57 |
| task_switch | 88 | 67 |
| checker_state | 53 | 51 |
| termination | 3 | 3 |
| task_vocabulary | 75 | 75 |
| task_name | 335 | 321 |
| dynamic_code | 51 | 45 |
| identity_source | 3 | 3 |
| instruction_or_identity | 98 | 96 |
| benchmark_composite | 6 | 6 |
| checker_predicate | 2 | 2 |

Complete machine-readable, grouped inventories:
[main CSV](data/legality/main.csv.gz), [D-C19 CSV](data/legality/dc19.csv.gz).
Both include category, audit/review group, identity-source disposition, file:line,
entry point, deep symbol, full call path, reason and exact source evidence.
Use `gzip -dc docs/experiments/data/legality/main.csv.gz` to inspect them.

## Path-loaded code coverage

Every `.py` module under `ported/skills/**` is an explicit extra root, including
helper files and functions without `run()`. Every `HarnessRobot.cmd_*` handler is
rooted, including the `cmd_skill` dispatcher and nested loaders. Known local
file-loaded helpers outside that subtree are included too. No ordinary import
edge is assumed for those files. See the complete [module/root/hash manifest](data/legality/path-loaded-modules.json)
for both snapshots. Dynamic import/spec/exec sites are separately inventoried as
`dynamic_code`; arbitrary/external targets remain unresolved and cannot be
certified by this audit. The manifest is a conservative superset, not a claim
that all these modules execute in the candidate.

## Perception checkpoint training scope

**Joint decision supplied by the lead:** a globally fixed checkpoint is not sufficient
for general-system admission. Training scope is a separate provenance requirement.
These classifications apply wherever the candidate uses the weights, regardless of
service name, socket alias or whether launch flags are identical across episodes.
The [machine-readable reader/service inventory](data/legality/perception-services.json)
and [candidate summary](data/legality/candidate-config.json) record both decisions.
They are additional model-provenance findings; the AST signal counts above are unchanged.

- **`kong_v1b` — EXCLUDED from the general system; blocks board entry.** It was
  trained on a single task's tile set and counts as task-shaped perception. The
  historical candidate selects `--glyph-kong kong_v1b`, loading
  `kong_v1b/model.pt` through the `glyphk` service. Freezing that checkpoint globally
  does not make it general. This supersedes any favorable reading of the earlier
  audit's observation that glyph components were globally pinned.
- **`digit_v2` — UNDER REVIEW.** It may qualify as a general digit perception
  service. The candidate selects `--glyph digit_v2`, loading `digit_v2/model.pt`
  through `glyph`. Its admission is undecided, not approved and not classified as
  excluded. The planned comparison against model-read crops is on 8 November;
  no crop comparison, training or model call was performed for this inventory.

### Candidate, service and environment wiring

The dated [candidate flags](data/legality/dc19-candidate.txt) enable both models and
set `DOJO_PERCEPTION_CFG` to `glyph_read=on`. Keep those historical bytes unchanged
for reproducibility; their presence is not authorization for a general-system run.

- `scripts/dojo/queues/lane2.sh:181-229` parses both flags and either starts per-job
  services or reuses shared ones. `glyphk` exports **`DOJO_GLYPH_ADDR_KONG`**;
  `glyph` exports **`DOJO_GLYPH_ADDR_DIGIT`** and legacy **`DOJO_GLYPH_ADDR`**.
  `LANE_MODELS` selects their model directory (default
  `~/dojo-data/segtrain/models`), with the model name selecting `model.pt`.
- `scripts/dojo/queues/shared_side.sh:37-40` explicitly starts **excluded**
  `glyphk/kong_v1b` and **under-review** `glyph/digit_v2`.
  `sam_shared.sh:19` starts this shared bundle too: enabling a shared SAM service
  does not clear the glyph reader's exclusion or review status.
- `scripts/dojo/official_eval.sh:160-161,296-298` uses the same `glyph`/`glyphk`
  environment mapping and `glyph_read serve --model ...`. Here
  **`DOJO_SEG_MODELS`** selects the model root. The frozen file hashes/provenance
  and the matching mapping in `scripts/dojo/candidate_parity.py:32-33` establish
  identity/parity only, not general-system eligibility.
- `dojo/perception/glyph_read.py:_GlyphModel` loads the checkpoint with
  `torch.load`. `MODEL_OF` routes numerals/operators to `digit` and mahjong tiles
  to `kong`; `ENV_ADDR_OF` chooses the dedicated endpoints. `client()` falls back
  to **`DOJO_GLYPH_ADDR`**, and the legacy category-less client uses that address
  without a family check. `model_fits()` checks model family, not training scope.
  Thus clearing only `DOJO_GLYPH_ADDR_KONG`, renaming a socket, using generic
  `--glyph kong_v1b`, or passing those weights directly to `serve --model` does
  **not** make them admissible. The exclusion follows the actual weights.
- `dojo/perception/inverse_layout.py` consumes slot/numeral glyph evidence when
  `EstimatorConfig.glyph_read` is enabled. Its own task-config candidate/identity
  dependencies remain separate blockers even if the excluded reader is removed
  or the digit service is later approved.

## Identity-source policy

**Legitimate:** robot/camera/scene `env_cfg`, `X5A.urdf`, and intrinsic asset files
(`object.usd`, `metadata.json`, `description.json`), derived catalogue-wide by the
same procedure for every episode.

**Illegitimate:** `task/RoboDojo/config/**`, `tasks/**` checkers, `Eval_Layout`,
`Traj`, logs, and per-task candidate lists/counts. “Same catalogue” is insufficient:
task-conditioned candidate pruning encodes the answer. A globally stored task schema
is still a task schema. The task-map, quota, count, slot, placement and background
paths below therefore remain blocked even when their asset meshes are public.

### Our two catalogue provenance questions

1. **`dojo/perception/catalog.py:695-704`, “Checker-relevant attributes”.** The
   place frames come from `metadata.json` `active.place` (read at 475–485);
   `annotations` receives that same `md` at 515 and copies `passive.functional`
   and `passive.support` at 699–705. Articulation axes are measured from intrinsic
   USD joints by `articulation_info`, not loaded from a task checker. **These
   specific fields have legitimate intrinsic-file provenance** under this policy,
   despite the misleading heading, provided extraction remains uniform and the
   asset metadata itself is the intrinsic source. This is source tracing, not
   independent verification of external asset contents. The function also mixes
   in the digit rule below: do not promote the entire annotation blob as clean.
2. **`catalog.py:614-616`, number-card index → digit.**
   `CATEGORY_ATTRIBUTES = {"number_card": lambda idx: {"digit": int(idx)}}` is a
   handwritten category-specific semantic mapping. Its comment claims decal
   textures were checked, but the code does not read texture/OCR or an intrinsic
   digit metadata field to produce the value. **Exclude it from a clean identity
   catalogue** until digit extraction is asset-texture/OCR or intrinsic-metadata
   based, with reproducible provenance. A manual “idx equals digit” assertion is
   not that derivation. Do not use perceived asset index as a substitute for an
   unread card digit.

The inventory marks the mixed annotation pipeline and the category-index rule;
legitimate intrinsic-file names are deliberately not flagged as task data. Scanner
rules cannot establish numeric/semantic provenance from a bare literal.

## Confirmed blockers to retire

The excluded `kong_v1b` checkpoint is an additional blocker; `digit_v2` remains
under review as detailed above. A fixed service/checkpoint is not a clean bill of health.

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

### Concrete path-loaded checker blocker

`ported_plan` with candidate `ported_pour=full` selects `pour_p1`; the reflective
`HarnessRobot.cmd_skill` loads `ported/skills/pour_p1/skill_robodojo.py`. Its
`_in_target` at line 182 supplies the ball-membership count at 424; the `verified`
assignment at 436 requires all balls in that footprint, cup tilt <=15 degrees,
and home (or ended). This matches the concrete checker-shaped path in the partner
hybrid re-check and **blocks board entry**. Both predicate and outcome assignment
now have `checker_predicate` inventory signals. Generic instruction sentinels alone
cannot detect this geometric/checker knowledge.

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








## Proposed additions to send back to the partner

All additions extend the shared engine/profile; there is no second call-graph engine.

- **Imported global constants and dispatch tables:** opt-in module nodes connect
  reachable functions to imported globals. A clean wrapper cannot hide a task
  table in another module. Relative/package imports and re-exported registry roots
  are resolved; a planted imported global demonstrates this case. All path-loaded
  modules/helpers and cmd_* handlers are explicit roots; unknown targets have
  dynamic-loader findings. The reviewed pour_p1 predicate has a regression case.
- **Resolved leaves:** builtin float/list, specifically imported dataclasses.field
  and numpy.ascontiguousarray, and field-only stdlib dataclass constructors are
  recognized operations, not caller exemptions. Shadowed callbacks and file-loading
  calls remain unresolved. This removes false positives without growing exemptions.
- **Opaque/dynamic calls:** our profile reports unresolved calls rather than
  silently certifying them. This is deliberately conservative, including external
  dependencies; it explains the large separate unresolved inventory.
- **Task switches, schema accessors and checker provenance:** extra signals catch
  RuntimeConfig task fields, ported variant routing, checker-shaped tolerance
  comments, known loader/accessor symbols, near one-to-one composites and the
  category-index digit rule. Known prompt resources share the same audit rules.
  Historical logs/Traj paths and evaluator-termination surfaces are also signals;
  writing logs or transport shutdown alone is not a violation.
- **Pin scope:** reflective and carrier exemptions match their exact stripped
  source line. One approved reflective call cannot exempt a second dynamic read
  elsewhere in the function. Duplicate matching lines fail closed as ambiguous.
  Multiline literals pin each offending line, not the opening quote. Changed pins
  are themselves findings.
- **General-path enforcement:** first-path deduplication must not hide a dependency
  also reached from a new general root. The wrapper separately checks reachability
  from those roots; no legacy exemption can certify that dependency. Exemption
  occurrence budgets cannot grow by duplicating an already-known source line.
- **Shared runtime sentinel plus task-data tripwires:** use the partner's
  `InstructionSentinel` / `InstructionRead` (BaseException); our test adapter also
  poisons task identity, clears cached schema accessors and traps alias calls and
  task-data file reads. Sticky hits survive code that catches BaseException.
  Internal `HarnessRobot.cmd_obs` is tested below the wrapper. The shared sentinel's
  documented C-level serialization limitation remains; serialized observations
  must also be inspected for `SENTINEL_TEXT`.

The partner's glib-specific tests are preserved with explicit skips when glib is
absent; their planted-violation, spelling and sentinel tests run in this tree.
Our added fixtures cover the gaps above. Mock runtime episodes exercise the rule
planner and FakeLLM agent with raw instruction text permitted; stacking/arrangement
also run with the shared sentinel **inside** explicit-argument skills. Production
`RuntimePolicy` tests poison `RuntimeProvider.instruction` and subsequent bridge
observations after planning, rebuild Skills through the production factory, and
exercise real `RuntimeProvider.resync -> _Checker` calls with a mocked perception
service. GT access raises; a planted perception-checkpoint read raises
`InstructionRead`. Both `error` and `error_handled` must be absent in positive cases. Negative
controls still expose unknown-instruction schema fallback and the internal ported
instruction leak. They document blockers, not exemptions. The external perception
mock cannot certify the real perception service; static findings show it is not clean.

Glib findings 12–19 (cardinality, duplicate/self pairs, movable roles, default frames,
NaN, false look success, overstated outcomes and occupancy geometry) are shared-library
review cases, not additional proven defects in our snapshot.

## CI and policy

The inventory job is report-only and uploads all findings. The separate ratchet
rejects new findings, growing exemption keys/budgets and **any** finding reachable
from the declared new general-system paths. The initial baseline is historical debt,
not board permission. Retire exemptions as paths disappear; never regenerate it to
make a PR green. Exemptions pin exact source evidence/reason, independent of harmless
line-number moves. See [DEVELOPMENT](../DEVELOPMENT.md#inference-legality-inventory-and-ratchet).

One shared implementation now serves this repo and the partner profile. Send the
proposed additions/tests back for the robot-harness move; do not retain competing scanners.
