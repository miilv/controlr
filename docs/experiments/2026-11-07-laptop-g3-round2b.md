# Laptop G3 round 2b: denser stable pinches plus staged carry

## Registration before panel episodes

Continue the lead's reliability target after the [staged panel](2026-11-07-laptop-g3-round2.md)
reaches only 6/10 standard and 7/10 random. No individual panel failure is used
for tuning. Development returns only to original pilots 100030/31.

The original standard-100030 loose-hold pilot is rescued by a denser grasp
bank: 44 mm pinch, 3.0 mm lift drift and 1.3 mm carry shift, headset credit.
The existing antipodal generator samples 200 faces and six approaches; six
approaches omit straight down (±18 degrees are the closest). The new bank
samples 400 faces and thirteen approaches (including zero), retains box/slice
proposals, keeps only widths >=35 mm, and screens up to 24 full hang plans.
Native IK/collision and the original physical lift/loose verification remain.
The two random pilots also score with this bank. All four pilot-only production
smokes must complete before panel launch; no panel feedback alters the candidate.

A: `hang_cradle=straddle`, `headset_seat=off`, `headset_carry=legacy`,
`headset_grasp=legacy`. B: same straddle allowance plus `headset_seat=on`,
`headset_carry=staged`, `headset_grasp=dense`. No laptop route in either arm.
New grasp field defaults to legacy; every new behavior is default-off.
Freeze source before panel launch and retain its commit in every row.

Fresh v15 draws, variant 13, salt
`laptop-round2b-dense-2026-11-07-88c48b74`, ids **100050–64** per cell, never
used by this stream in earlier rounds. Settle once, pick the first ten stable
ids ascending (before policy outcomes), hash exact settled records and share
between A/B. No reserved/sealed ids or rendered/proxy use of these ids.
Ten pairs per cell; alternate arm order by id. Report points mean/sample SD,
headset-tier credits, full successes, score wins/losses/ties, steps and exclusions.
Reliability: **at least 7/10 headset credits in each cell**. Also assess the
registered score gate: pooled gain >=5 points, wins >=2*losses, >=2 net wins.
Keep this panel separate from earlier panels; no selected pooling.

Two concurrent physics-only jobs maximum on compute3, using free light2/light8
slots (another stream owns light7), who/GPU admissions, <=24 GiB used VRAM,
>=10 GB free disk, systemd user scopes 8G / 600% CPU and KILL-tail timeout.
Only `~/controlr-laptop` and `~/dojo-data/laptop` are written. Zero LLM spend.
No camera dispatch or board candidate change. If this qualifies, write a
prospective camera spec only.
