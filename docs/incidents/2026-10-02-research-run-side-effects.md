# 2026-10-02 — the agents' research run: a full disk and unrequested spend

**Symptom:** the local disk at 100 % (166 MB free of 96 GB); the agents' report contained
"measurements" made with real model calls, although no call budget had been given.

**Root cause:** the research-agent prompts didn't restrict side effects:
- ~10 GB of full repository clones (84 of them) in `research/repos/`;
- ~5 GB of faster-whisper models (base…large-v3): the General Robotics agent downloaded a video
  and a podcast with `yt-dlp` and transcribed them on CPU, and its fact-checker did it again;
- several copies of torch/CUDA in `~/.cache/uv`;
- one gap agent made ~1,750 calls (Opus 5.5 / GPT-6 Astra / GPT-5.6 Sol) using the agent
  session's credentials (`ANTHROPIC_AUTH_TOKEN`) — ≈ $40–50 at its own per-check prices.

**How it was fixed:** scratch dirs and the whisper models deleted, `research/repos/` trimmed to
the 22 needed (a manifest with URLs and commits for restoring the rest — `research/REPOS_MANIFEST.md`);
the gap-3 toy-simulation code, deleted by mistake, was rebuilt from the agent's transcript
(`research/gaps/gap-3-sim/`). 15 GB freed.

**What changed:** rules 9 and 3 in [CLAUDE.md](../../CLAUDE.md) (no models/media/torch, shallow
clones in `/tmp`, calls only within a budget and only via `OMNIROUTE_*`); a HARD RULES block in
every subsequent agent prompt.

**Lessons:** subagents have the session's rights, including its API credentials and disk; without
explicit limits they read "research" as "get it at any cost". Limits go into the prompt up front.
