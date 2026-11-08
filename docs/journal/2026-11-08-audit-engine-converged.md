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
