# 2026-11-08: a read-only audit scanner walked sealed roots on compute4 and compute7

**What happened.**
- At about 23:15Z on 7 Nov the lead delegated PUBAUDIT, a read-only Codex audit (gpt-6.1-sol). Its question was our side's
  use of public RoboDojo layouts, needed for the joint committee disclosure. The report is
  `docs/journal/2026-11-08-public-layout-use-audit.md` (81aed7cd).
- The brief listed the sealed roots as never to be read: `~/dojo-data/dc19`, `~/coord`, `~/rh-misha`,
  `~/dojo-official-rh-*`, and any `scored` dir.
- **compute4:** the agent's first text scanner applied its sealed-name filter only to child directories, not to the
  glob-expanded starting roots. It therefore walked `~/dojo-official-rh-*` and `~/controlr-dc19`.
- **compute7:** its filter matched `dojo-official-rh-` with a trailing dash, so it also walked `~/dojo-official-rh`.
- The scanner was a keyword grep over text files: seed / layout / episode fields and similar.
- The agent says it discarded that output and rebuilt the scanner to reject the roots too. It disclosed the error in
  its own report ("Audit procedure errors").

**Impact.**
- Keyword-matched lines from sealed D-C19 / official-rh records entered the audit agent's context, and so its T3
  thread transcript. They were not used in any finding.
- The agent writes no code and has finished. No skill, prompt, config or experiment was changed.
- The lead did not see the discarded output.
- No sealed file was modified and no process was touched.
- Exposure is limited to that one finished agent and its stored transcript. It is still a breach of the sealed-data
  boundary and is reported to the partner team: the sealed process is joint (D-C19).

**Fix.**
- Audit and scan briefs name the roots to read as an explicit allowlist, not "everything except the sealed list".
- A scanner must check its starting roots, not only descendants, before walking.
- Prefixes are matched without a trailing separator.
- The PUBAUDIT thread is not reused for further work.

**Follow-up (8 Nov, ~00:00Z).**
- The partner team asked for the matched file paths (no contents), the patterns, and whether tranche B (ids 100105–100109) was touched.
- The same finished agent was asked once to answer from its own record, with no re-scan.
- Its answer: the record is incomplete and individual paths were not retained. Tranche B and whether results were in the matched lines are both **unknown**.
- Its patterns file was forwarded to the partner unread by the lead.
- We will not re-scan the sealed roots to rebuild the list.
- Until the founders decide, tranche B is "under review". Our suggestion: treat every sealed block under those roots as possibly exposed, and cut a fresh sealed block for the hidden-layout rehearsal.
