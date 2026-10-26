# 2026-10-26: a colon-list DOJO_CATALOG crashed perception for 7 proxy jobs

**Symptom.** From 00:25 UTC on compute3, seven mode-B integ jobs (the scan A/B tower chunks off / trigger a-c, off d)
failed every episode at step 0. Each one hit `perception service: PermissionError: [Errno 13] Permission denied:
'/root/dojo-data/per...'` and finished in 2-3 min instead of ~30.

**Root cause.** At 00:05 integration-2 restarted the queue runner with
`DOJO_CATALOG=~/dojo-data/skills4/catalog:~/dojo-data/skills3/catalog:~/dojo-perception-data/catalog`. The aim was to
get skills4's `links.json` (make_toast's lever geometry) to the skills' `AssetDB`, which reads DOJO_CATALOG as a path
list.
- `dojo.perception.render.catalog_root` reads it as ONE directory. It found no `index.json` there and fell through
  to its next candidate, `/root/dojo-data/perception/catalog`.
- On compute3 `/root` is unreadable, so `Path.exists()` raised PermissionError instead of returning False.
- Every perception `estimate` therefore failed.
- Mode gt jobs start no perception service, so they were unaffected.
- Cost: ~35 min of heavy-lane GPU, 35 errored episodes, 0 LLM calls.

**How it was fixed** (00:45-00:47, ~5 min):
1. Stopped the runner.
2. Copied skills4's three `links.json` (toaster 00000 / 00001, laptop 00000) into `~/dojo-perception-data/catalog`.
3. Renamed the errored run dirs `*__catalog_env_error`.
4. Re-queued the 8 jobs at the front.
5. Restarted the runner with `DOJO_CATALOG=~/dojo-perception-data/catalog` (single dir).

The first rerun episode was fine.

**What changed so it won't repeat.**
- `scripts/dojo/queue_runner.sh` has a pre-flight: perception must load its catalogue with the runner's own
  environment, or the runner refuses to start.
- perception-R&D makes `catalog_root` treat any OSError on a candidate as "absent" and accept a list.

**Lessons.**
- DOJO_CATALOG has two readers with different contracts (skills: a list; perception: one dir). Keep it one dir, and
  put extra skills files into perception's catalogue.
- Probe candidate paths under `try / except OSError`. The same PermissionError hit `dojo.runtime.vocab` on CI (fixed
  in #32).
- After changing a long-running runner's environment, watch its first episode before walking away.
