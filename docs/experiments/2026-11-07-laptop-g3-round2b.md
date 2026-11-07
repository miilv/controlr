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

## Completed dense-bank panel

Source `473d64cf`. All four original production pilots scored before launch;
none is in this panel. All 30 generated records settled; first ten stable
100050–59 are used per cell. Forty valid panel episodes, no exclusions,
missing meshes, infrastructure failures or budget exhaustion.
[Rows, hashes, admissions, recipes and pilot drift audit](2026-11-07-laptop-g3-round2b-results.json).
Numbers below are 0–100 points and sample SD; raw score is 0–1.

| Cell | n pairs | A mean ± SD | B mean ± SD | Headset credits A/B | Full successes A/B | Score W/L/T |
|---|---:|---:|---:|---:|---:|---:|
| standard | 10 | 0 ± 0 | 12 ± 10.33 | 0/10 / 6/10 | 0/10 / 0/10 | 6 / 0 / 4 |
| random | 10 | 2 ± 6.32 | 18 ± 6.32 | 1/10 / 9/10 | 0/10 / 0/10 | 8 / 0 / 2 |

Pooled +14 points, 14 wins / 0 losses passes the paired score gate. It does
**not** meet the lead's reliability target: standard remains 6/10. Do not stop
or call this a reliable standard-cell hold. Development continues on a newly
registered independent [pilot bank](2026-11-07-laptop-g3-round2c.md); no individual
panel failure is used for tuning. No selected pooling with the earlier panel.

Steps median/max: standard A 84.5/149, B 203.5/357; random A 145.5/204,
B 249/365. Full success flips 0/0. Laptop tier unattempted, no laptop score.
Production pilots have continuous post-lift-to-hover max attachment drift
1.66/3.36 mm standard and 6.79/2.50 mm random; their success is not just a
small final residual after a large transient displacement.

Local full suite on this exact action source: **2063 passed / 191 skipped**
(1041.26 s); focused suite 16 passed. GitHub CI for source 473d64cf is green.
All new flags remain default off/legacy. A prospective camera spec is written
but blocked pending reliability and a separately assigned compatible host;
no camera jobs, board changes or model calls/tokens/spend (0/0/$0).
