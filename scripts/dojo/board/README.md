# Durable RoboDojo research board

Stream M's recovered rules and STRAT's read-only fetcher. Python standard library only;
run from any working directory. Git ancestry is evaluated against `--repo` (default: this
checkout), so keep full history and `origin/main` available. No robot, GPU or LLM calls.

```bash
mkdir -p ~/dojo-board
cp scripts/dojo/board/snapshot/*.json ~/dojo-board/
python scripts/dojo/board/cells.py --state-dir ~/dojo-board
# Optional read-only refresh of the permitted compute4 integration tree:
python scripts/dojo/board/fetch_rows.py --state-dir ~/dojo-board
python scripts/dojo/board/cells.py --state-dir ~/dojo-board
```

Outputs: `cells.json`, `cells.md`, `northstar.json`, `cells_noceil.md`, `v15_state.json`.
Fetch errors exit nonzero and preserve the previous rows; inspect the fetch log before
claiming freshness. Run the board after a successful fetch. Reports are parsed remotely
only when their mtimes change; only compact row/config fields cross SSH. The initial
fetch has no report cache and may take longer. `snapshot/manifest.json` hashes the checked-in
inputs; these are a small, sanitized recovery checkpoint, not sealed results.

The fetcher reads only `~/dojo-data/integ-c4/runs2` on compute4. Directory ids are checked
before opening reports/episodes; unknown ids, D-C19 names and ids >=100100 are excluded.
Rows are filtered again before transfer and at board ingestion. Sealed scoring is
intentionally unsupported (`--dc19` fails argument parsing). No poller or remote writes.

`cells.py` preserves historical candidate/strict selection and pre-registered decisions;
it does not retroactively replace old A/B rules with the new Camera A/B default. Its
embedded dated references and flags describe the measurement candidate, not current
runtime defaults. Missing historical proxy/breadth/other-system ledgers are omitted and
not imputed; the surviving checkpoint still reproduces 51.53 strict. G3 references in
code and `g3c_rows.json` are dated estimates, sometimes thin, stale, or below camera results.
The journal points table discloses those limitations. Do not read negative differences
as negative recoverable opportunity, or dev-only G3 upside as an adopted ceiling.

Before refreshing, fetch repository history (`git fetch origin`). For exact replay use the
same repository revision/ref (`--main-ref` from `snapshot/manifest.json`) and snapshot inputs; headlines remain observations of older
candidates, not an evaluation of current main. Review new unclassified families explicitly.
