# KONG eligible re-test moved to compute4 after COIN

The lead requested moving the whole unstarted 60-row KONG scoped re-test from
compute6's tail to compute4 after COIN. Amendment `7e16003e` preceded both
queue mutations. At 11:28:30Z the locked no-start check passed and compute6
positions 104–163 were removed, preserving all other bytes. At 11:29:19Z the
validated c4b_ block was appended in compute4 positions 45–104 behind 44 COIN
rows. No running lane/job was changed. No KONG outcome was inspected, and MOVE
launched no model/GPU job.

New source deploy `3eefbfc5` contains #410/#411 and #417. Frozen inputs,
calibration/models/backgrounds/cache, full package parity, six candidate
checks, 60 canonical configs, existing Skills fields, catalogue coverage and
17 focused tests pass. Seven new config fields and six new Skills attributes
are explicitly inactive and recorded; no frozen field changed.

The [prereg receipt](../experiments/2026-11-07-kong-scoped-retest-prereg.md#move-receipt--compute6-to-compute4-2026-11-07)
contains both queues' full hashes, byte-verified manifest, code/instrument pins,
checks and recovery evidence. ETA is approximately 14:00–16:25Z, subject to
shared resource/startup gates and COIN finishing. No verdict or adoption.
