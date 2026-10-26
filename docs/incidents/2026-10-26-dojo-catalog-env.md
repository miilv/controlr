# 2026-10-26: a colon-list DOJO_CATALOG crashed perception for 7 proxy jobs

**What happened.** integration-2 restarted the compute3 integ queue runner at 00:05 UTC with
`DOJO_CATALOG=~/dojo-data/skills4/catalog:~/dojo-data/skills3/catalog:~/dojo-perception-data/catalog`, so that
make_toast's lever geometry (skills4's `links.json`) would reach the skills' `AssetDB`, which reads DOJO_CATALOG as a
path list.

The next seven mode-B jobs (the scan A/B tower chunks off / trigger a-c, off d) failed every episode at step 0:
`perception service: PermissionError: [Errno 13] Permission denied: '/root/dojo-data/per...'`.

**Cause.** `dojo.perception.render.catalog_root` reads DOJO_CATALOG as ONE directory.
- With the list it found no `index.json`, so it fell through to its next candidate, `/root/dojo-data/perception/catalog`.
- `Path.exists()` there raises PermissionError on compute3, where `/root` is unreadable, instead of returning False.
- The perception service died on every `estimate`. Mode gt jobs were unaffected because they start no perception
  service.

**Cost.** About 35 min of heavy-lane GPU time, 35 errored episodes, 0 LLM calls.

**Fix.**
- skills4's three `links.json` files (toaster 00000 / 00001, laptop 00000) were copied into
  `~/dojo-perception-data/catalog`.
- The runner was restarted with `DOJO_CATALOG=~/dojo-perception-data/catalog` (single dir).
- The errored run dirs were renamed `*__catalog_env_error` and the jobs re-queued.

**Lessons.**
- DOJO_CATALOG has two readers with different contracts: skills reads it as a list, perception as a single dir. Until
  perception accepts a list, keep it a single dir and put extra skills files (`links.json`) into perception's
  catalogue.
- Probe candidate paths with `try / except OSError`. The same PermissionError hit `dojo.runtime.vocab` on CI (fixed
  in #32); `render.catalog_root` needs the same guard (perception-R&D).
- Check the first episode of a queue after any change to the runner's environment, before walking away.
