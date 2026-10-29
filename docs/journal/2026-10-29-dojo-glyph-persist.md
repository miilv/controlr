# 2026-10-29 — a slot's glyph read persists across later records (stream W)

**Problem.** 5.1 solve_equation 100005 (Integration, `c4b_glyphchk2_solve_100005` at 3ad78f28): frame 0 read t1
"minus" (cam_head, margin 0.948). The scan's re-estimate had its own views disagree (cam_head minus, cam_head@1 minus,
cam_right_wrist "equal"), so it returned t1 unread with the placeholder "division". agentic0, fused and agentic2 built on
that. The skill saw two unknown t-tiles (t0 being the empty pad), #291's one-unknown-equals-the-empty-pad rule could not
apply, and it stopped at "values unknown: ['t0','t1']".

**Fix** (`dojo/perception/read_persist.py`, wired in `RuntimePolicy._persist_reads` at every record stage: initial, scan,
agentic0, fused, searched, agentic2; event `glyph_persist`). Per slot label, with `_merge_glyph`'s semantics:
- a later record without a read keeps the earlier read (`carried`);
- an agreeing read refreshes it;
- a contradicting one makes the label unknown and drops the earlier read.

Slot labels only. The first version also persisted free numerals by label. Replaying 100005's records showed why that is
wrong: each re-estimate assigns the free labels to tiles anew (the scan's num_5 sat where frame 0's num_6 was), which gave
three false "disagree" events and one digit carried onto another tile. The rule acts only on classifier reads, so it is a
no-op with `glyph_read` off. It has no config relation and no task keying.

**Offline replay of 100005's records** (slim records from the report): t1 stays Rigid/minus/0, identity read, through
scan, agentic0, fused and agentic2. t0 is the only unknown t-tile, so the empty-pad rule applies: a - 3 = 6, a = 9, and
num_9 reads 9 with conf 1.0.

**Side finding, num_3 / num_5 reading "not a digit".** This comes from B1 placement, not the classifier. Frame 0's
num_3 sits at (-0.048, -0.274), 2 mm from the config-pinned minus tile (-0.05, -0.27). num_5 sits at (0.049, -0.275),
on multiplication (0.05, -0.27). The classifier correctly reads the operator under each (margin 0.95). The free-numeral
fit does not exclude the pinned operator tiles' footprints: the free numerals land on those footprints, rather than
swapping places with the operators.
