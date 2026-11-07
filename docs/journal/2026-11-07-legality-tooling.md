# Inference legality tooling and baseline

The LEGAL stream adopts the partner team’s shared `dojo/legality/audit.py`
(delivery 18:09Z, upstream `4c4d36db`), replacing its separate prototype. It adds
our inference profile, an offline transitive source inventory and mock-only runtime
sentinels after Ilia's general-system hard rule. No inference defaults changed.
The [baseline report](../experiments/2026-11-07-legality-inventory.md) inventories
main and sealed D-C19, cross-checks both partner audits, distinguishes confirmed
blockers from risky evidence and retains all unresolved calls for review.

CI uploads the inventory without failing on historical findings. A separate ratchet
rejects new findings, growing exemptions and any finding reachable from the declared
new general-system directories. Runtime tests keep task instruction text available
to the planner while poisoning task identity and trapping schema accesses. Negative
controls reproduce the legacy schema fallback and the transitive ported instruction
leak; their presence is not permission to use those paths in a board entry.

The partner implementation is now the single base. The inventory’s proposed-additions
section records our small extensions and tests for the move to robot-harness. This PR does
not introduce another general library or attempt to retire the blocked controllers.

The hybrid re-check adds explicit roots for every vendored Python module/callable,
including helpers without run(), all cmd_* handlers and known adjacent file-loader
targets. The inventory publishes module/root hashes and dynamic-loader findings.
The active candidate pour_p1 containment/tilt verified predicate is a concrete
checker-shaped blocker, now tested and labelled. Carrier/reflective pins are unique
single lines; multiline literals point to each offending line. The production
RuntimeProvider perception/checkpoint path is exercised with poisoned instructions
and a negative-control read, rather than relying only on GTProvider fixtures.

The asset-path firewall test permits only the auditor's explicit denial-symbol set
and rejection regex; it still rejects Path/open/list constructions in that module.
Validation: full unit suite on four workers, 2,317 passed / 214 skipped; final
combined legality/provider/firewall coverage, 41 passed / 8 glib-only skips.

CI exposed two main advances during this first baseline's review. The bootstrap now
pins actual merged-main source `9fa62eda`, including existing CLASSIFY3/evidence code.
New legacy rows remain debt, not board permission; no clean-path exemptions were added.
The resolver recognizes only named numeric/record leaves, with shadowed callbacks and
file-loading calls still unresolved. The updated branch ratchet passes locally and
combined tooling/provider/firewall tests pass: 43 passed / 8 glib-only skips.

The joint reader decision excludes kong_v1b (single-task tile training) from the
general system. digit_v2 is under review pending the model-read crop comparison
on 8 November. The inventory and machine-readable candidate/service records mark
both statuses, covering dedicated/shared/official launchers, model-root variables
and the legacy generic glyph-address fallback. Historical candidate bytes and
runtime behavior are unchanged; no model was loaded or evaluated.


The bootstrap was refreshed on main `3539749a409cb56fb8895bb99d58750c8059291c`
after #427/#428/#430/#431/#432/#433. Recent general-library additions have 734
exact transition review pins, separate from 28,591 legacy pins. Reasons distinguish
physical options, lexical false positives and unresolved dependencies; sort's p2
instruction boundary remains a blocker. General masks/prompts/filters and base
refit are now explicit roots even before adoption (603 roots on main). Both debt
files shrink after the first merge; neither can clear a clean-root finding. Tests
cover review pin edits, duplicate growth, overlap, missing reasons and clean-root
bypasses. The inventory begins with ten legacy retirement items for the joint
library hand-off. The immutable D-C19 source/flags and reader provenance decisions
are unchanged.
