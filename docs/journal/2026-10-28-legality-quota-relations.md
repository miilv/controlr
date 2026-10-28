# 2026-10-28 — legality triage: B1's imitate quota and copy-list identities came from the config's structure (Legality stream)

Source: an independent perception reviewer, via the lead: `dojo/perception/inverse_layout.py` `_swap_pairs` /
`_category_quota` (~2112-2330) may decide object IDENTITY from the task config's region / list structure, close to the
audit's "config identity relations" rule (robot-harness `docs/notes/ILIA-POLICY-LEGALITY-AUDIT-2026-10-28.md`, P1-2).
The lead's rule: region priors on POSITION from the public config are fine; identity assigned from config structure
instead of observation is not.

**Which tasks.** `quota_lists` over the public catalogue (compute3, read-only): only imitate_sorting_sequence has quota
lists, `t0..t4` (hierarchical, 5 categories x 1: action_camera, garage, phone, toy_car, watch) and its `same_as_label`
copy `aim0..aim4` (the demonstrator's objects).

**Verdicts.**
- `_swap_pairs` (`swap_pairs`, on): NOT a relation. Two chosen objects of different label groups trade labels only
  where each label's generator region admits the other's place (a public position prior) and both fit the images
  better swapped. No select_mode / tie / quota is read. Kept.
- `_category_quota` (`category_quota`): a config identity relation. "Each category once" comes from the select list's
  structure; when two objects read as one category, the quota re-fits them as the missing category and assigns by
  cost, and `_quota_balance` proposes a missing category on unexplained foreground. Identity (and existence) is forced
  by the structure, not observed. OFF by default now.
- `_quota_copy_assets` (`quota_copy_assets`): the audit's P1-2 class exactly. `same_as_label` hands the demonstrator's
  objects our objects' assets (the experiment's own words: "the config says aim_i is t_i's asset"). OFF by default now.
- Not changed, flagged for a ruling (BACKLOG): B1's identity step pools evidence across a `same` select list (every
  member takes the key with the least summed cost) and gives a `unique` / `hierarchical` list distinct keys (Hungarian);
  `_options` restricts a `same` group to one asset and a single-category `hierarchical` group to one entry. These span
  many tasks (often the instruction itself says "different"), so a change needs a ruling and a G1 + rendered A/B.

**Behaviour change: imitate_sorting_sequence only** (no other task has a quota list; G1 sets unaffected by construction,
as in the quota's own journal). Its offline numbers with both on were (13 genuine-5.1 demo sequences, docs/experiments
2026-10-27-imitate-frame0 / -imitate-copy-assets): plan right 5/13 (region fix) -> 7/13 (+ quota) -> 10/13 (+ copy
assets); demo category 53 -> 57 -> 63 of 65. Expect imitate back near the region-fix level (~5/13 plan right offline)
in perception modes. GT mode / G3 unchanged. The functions stay for dev A/Bs behind their flags.

**Tests** `tests/dojo/test_legality_quota.py` (need cv2: they skip in CI; run locally with a perception venv, 4 pass):
defaults off; `estimate` reaches the two functions only behind their flags; `_swap_pairs` reads no select relation;
`quota_lists` changes with the hidden relation alone (same labels / candidates / regions, `same_as_label` vs `unique`
copy), which is what the flags would consume. `test_legality_kong.py`'s scan keeps `quota_lists` as the one reviewed
use, now noted as behind a default-off flag. No LLM calls, no GPU job.
