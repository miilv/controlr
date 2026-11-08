# Shared audit engine convergence

Base: partner `1413f37401adba053920675c7a407c4337226446`. Controlr source before
convergence: `985d889dbb6d099e38ef477f31fcdfa2b5b5ada1`.

The engine uses the partner implementation and retains controlr's inference
profile, path-loaded roots, unresolved reporting, resources, record/numeric
leaf resolution, provenance signals and unique exact-line reflective/carrier pins.
No inference defaults or runtime code change. No LLM calls or remote runs.

Semantic differences reconciled:

- Partner rapid type analysis, inherited methods, constructor/dunder edges and
  class-body nodes replace the constructor-blind graph. The inference inventory
  retains conservative all-method traversal through `scan_globals`; partner
  profiles retain rapid type analysis. Both use the same edge builder.
- Partner global `=NAME` nodes and module docstrings coexist with controlr's
  synthetic `<module>` nodes and imported-module edges when `scan_globals` is on.
- Package-relative imports and re-exported root resolution retain controlr's
  package initializer handling. Flat aliases remain supported for the partner.
- Partner prefixes for detached modules, literal method getattr, linked dispatch,
  injected classes, callbacks and configuration gates are retained.
- Partner task identifiers/comments, Camel acronym splitting, folded f-strings,
  concatenation/join/format/percent strings, data-file names, reflection builtins,
  introspection attributes and mapping identity reads are retained. Controlr's
  task accessors, historical data paths, checker predicates/provenance/tolerances,
  composites, policy switches, termination and category-index signals remain.
- Inventory strings retain every match and offending multiline source line;
  partner profiles retain their original reason strings for planted-test parity.
- Partner ignores attribute stores as reads and permits planner instruction
  access via profile-specific instruction/identity attributes. These are retained.
- Controlr's unresolved callback reporting and shadow-aware numeric/record leaves
  remain opt-in, including dynamic-loader signals. Unknown file operations remain
  unresolved. Field-only dataclass defaults are still scanned.
- Carrier and reflective pins retain controlr's unique exact stripped-line
  matching. Partner dispatch pins retain upstream behavior pending A2; their
  accepted-version tuples, required guards and gates are retained.
- Partner glib/hybrid profiles narrow carrier exceptions, add hash-listed data
  names, injected classes, dispatch edges and callbacks. They do not replace our
  inference profile. The CLI default remains controlr-inference; JSON now also
  reports function count and node count separately.
- Partner index forks/shared-index API and instantiable-class inspection remain.
  Runtime sentinel behavior is unchanged. Resource scanning and first-path
  deduplication remain, with independent clean-root reachability in the wrapper.

Known upstream gaps A1 (nested module / literal getattr constructors) and A2
(function-wide dispatch reflection exemption) are strict expected failures.
The existing controlr reflective_ok protection is retained, rather than regressed
to A2's behavior. No duplicate upstream fix is introduced.

Partner tests requiring absent glib/runtime modules are explicitly skipped;
standalone planted tests run locally. Detection gains are reported separately
from existing legacy and transition debt, with frozen-source evidence.

Same-source comparison: old engine 29,812 findings, converged engine 31,441.
The 16 retired old signals are attribute stores previously labelled reads.
Global binding nodes are not trusted as callable targets; unresolved callback
reporting remains conservative. The conservative
inference profile retains explicit dunder-method traversal and its original
self-method fallback; the partner's rapid type analysis remains profile-specific.
D-C19 changes from the published 27,201 signals to 28,769 (category table in the
inventory). No finding is reachable from declared clean general-system roots.
The new `detection-gains.json` holds 1,645 added rows under 1,643 distinct keys;
all reproduce on previous main with identical source lines. Two rows (function
and docstring nodes) in agentic/__init__.py are outside the original legacy set.

Validation: full local unit suite passes; the focused engine/ratchet tests pass
with 32 passed, nine absent-glib skips and three strict expected failures (A1's
two constructor spellings and A2). The ratchet against previous main and compileall
pass. The partner delivery is `/home/agent/dojo-lead/shared/to-misha/audit-converged.py`;
its opening docstring lists the controlr additions. LLM spend: $0.

## A1/A2 follow-up

Previous main: `84fb8c6e93df69ad59a6223810730b9011e044bd`. Partner patch
`e3a0cb19` is already present: ProcSession._serve is linked to the same robot API
list. The follow-up ports `b30b37ed` by hand around our exact-line pin protection.
Dotted module chains and literal getattr module function/class references now
reach their targets. Both new class spellings also reach class-body nodes,
including field-only classes, preserving the converged engine's existing feature.

Dispatch pins now exempt only reflection starting on their exact matching source
lines. Reflective_ok already had that scope and retains it; both mechanisms reject
duplicated exact lines rather than treating a substring as approval. Accepted
worker API/signature spellings are retained. The robot_api guard parses every
rdgen module's ProcSession calls and rejects explicit robot_api keywords, kwargs
and extra positional arguments; comments containing robot_api= no longer trip it.

The three A1/A2 strict xfails become ordinary passing tests. Both partner-planted
tests are ported, with the pin test also covering reflective_ok. Additional planted
cases preserve field-only class-body edges and verify the worker guard across
modules. Glib-only skips retain their explicit absent-module reasons.

Main inventory: 31,441 → 31,486. D-C19: 28,769 → 28,814. Each adds 45 unresolved
call signals in already-inventoried glyph_read.py; no signals retire and no new
non-legacy or clean-root findings appear. All 45 new baseline keys reproduce on
a separate git archive of previous main with identical finding/source-line evidence.
They join the separate detection-gain ledger; existing gain pins, legacy allowlist
and transition review are unchanged. A narrowly reviewed bootstrap permits these
new pins only against actual base 84fb8c6e; later bases cannot reuse that exception.

No inference defaults or runtime code change. No LLM calls, remote machines or
sealed data; spend $0. The partner delivery remains
`/home/agent/dojo-lead/shared/to-misha/audit-converged.py`.

Follow-up validation: full local suite 2,447 passed / 224 skipped / no xfails;
focused legality suite 44 passed / nine absent-glib skips. Compileall and the
ratchet against frozen previous main pass. Both partner-planted tests run and
pass; the three former strict xfails are ordinary passing tests.
