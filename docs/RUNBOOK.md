# Runbook: when something breaks

Machines and paths — [ENVIRONMENT.md](ENVIRONMENT.md). Past cases — [incidents/](incidents/).
compute3 is a shared machine: read-only by default, ours is only `~/controlr*`.

## 1. compute3: no GPU (`nvidia-smi` can't reach the driver)

Typical cause: an automatic kernel upgrade without the NVIDIA module for the new kernel
([incident 2026-10-02](incidents/2026-10-02-compute3-nvidia-driver.md)).

```bash
ssh compute3 'uname -r; lsmod | grep -c nvidia; dpkg -l | grep linux-modules-nvidia | awk "{print \$2}"'
# no module for the running kernel -> (only on the owner's explicit request; passwordless sudo exists)
ssh compute3 'sudo apt-get install -y linux-modules-nvidia-595-open-$(uname -r) && sudo modprobe nvidia nvidia_uvm && nvidia-smi'
```
No reboot needed if `modprobe` succeeds. A reboot kills other people's sessions/jobs — only with consent.

## 2. Isaac server

- Usually `scripts/remote_run.sh` manages it: starts it if port `7801` is free, waits for
  readiness (up to 600 s), stops it after the run. Log: `~/controlr/runs/isaac_server.log`.
- Tests start their own server on port `7821` (`CONTROLR_ISAAC_TEST_PORT`), never on the main one.
- Manual start (from `~/controlr`): `setsid nohup bash scripts/isaac_server.sh --port 7801 > runs/isaac_server.log 2>&1 &`
- Who holds the port / how to stop our server:
  ```bash
  ssh compute3 'ss -ltnp "sport = :7801"; pgrep -af "controlr/robot/isaac/server.py"'
  ssh compute3 'pkill -f "controlr/robot/isaac/server.py --phantom"'   # our process only
  ```
- "Isaac server died" immediately → check the log tail: most often the GPU (§1) or `PHANTOM_ROOT` / `ISAAC_SIM_ROOT`.
- An episode ended `unstable` → PhysX diverged (squeezed object / impact). That's an episode
  outcome, not a harness bug; if it repeats on one seed, inspect contacts in `turns.jsonl` and BACKLOG.

## 3. Router (omniroute) and models

| Symptom | Cause / action |
|---|---|
| reply in ~0.4 s with no usage | the router replayed a cached **response** to an identical request; check `llm.request_nonce: true` |
| `400 unsupported_image_block` | a route without image support (`dva/*`); use `claude/`, `cc/`, `no-think/`, `cx/` |
| empty reply, `finish_reason` = length | thinking used up `llm.max_tokens`; raise it or use a `no-think/` route |
| model not in `/models` but needed | try a direct call — the router accepts some unlisted ids (that was the case for `claude-sonnet-5-5`) |
| 429 / 5xx | the client retries with backoff; on subscription routes the real limit is the subscription quota |
| LLM calls from compute3 take 5–20 s and grow with the turn number (headers late, 0 reasoning) | (history: the Happ VPN, removed 2026-10-04, uploaded at ~50 KB/s; now see §3a) the route to the router uploads slowly, and every turn re-sends all images. Check: `curl -w '%{time_total} %{speed_upload}\n' --data-binary @<~740 KB file>` to `$OMNIROUTE_BASE_URL/chat/completions` from compute3 (< 1 s is healthy; 5–17 s seen). Workaround: from the dev box `ssh -N -R 127.0.0.1:18809:127.0.0.1:<dev box HTTP proxy port> compute3` (keep it up for the whole round), then `CONTROLR_LLM_PROXY=http://127.0.0.1:18809 scripts/remote_run.sh …` |
| `401 No active credentials for provider: codex-app-server` | `cxa/` routes have no account; use `cx/` |

Model list: `uv run controlr models --filter claude`. Is the network the problem? `bash scripts/net_check.sh` on
compute3 (no LLM calls): 740 KB upload to the router should take < 0.75 s. Where a call's time goes (router only vs
upstream vs model, chat vs Responses, tiny vs a replayed real turn): `scripts/router_latency_stand.py`.

## 3a. compute3 network (controlr-vpn)

compute3's VPN is `controlr-vpn`: sing-box, machine-wide like the dev box's (`tun0` + auto_route;
`.ru`/`.su`/`.рф`/Yandex/geoip-ru, private IPs, torrents, **the router** (direct is faster) and the
subscription host go direct; everything else through urltest over the Nofox WiFi nodes). Nothing needs
`HTTPS_PROXY`; HTTP/SOCKS proxies exist anyway on `127.0.0.1:10809/10808` and `18810/18811`. tailscale
(our ssh) keeps its own policy rules, which come before sing-box's.

| Task | Command |
|---|---|
| health / speed | `bash scripts/net_check.sh` — 740 KB to the router < 0.75 s, chatgpt.com / api.anthropic.com 401 (403 = the VPN is down) |
| status / logs | `systemctl status controlr-vpn`, `journalctl -u controlr-vpn` |
| refresh nodes now | `sudo systemctl start controlr-vpn-refresh` (timer: every 6 h; restarts the proxy only if nodes changed) |
| turn compute3's VPN off | on compute3: `sudo systemctl disable --now controlr-vpn controlr-vpn-refresh.timer` (tun0 and its rules go away, the machine goes direct). Revoke the link: delete the `compute3` `[[profiles]]` block in the dev box's `~/vpn-retranslator/config.toml` + `sudo systemctl restart vpn-retranslator` (refreshes then fail and keep the last config) |
| reinstall | `scripts/compute3_vpn/install.sh <token file>` from the dev box |
| the old VPN (Happ, AmneziaVPN, OpenVPN — removed 2026-10-04) | backup + `rollback.sh` in `compute3:~/controlr-netbackup-20261004/` |

Files on compute3: `~/controlr-vpn/{bin/sing-box, config.json (0600), sub_token (0600), refresh.py, geoip-ru.srs}`
(the earlier proxy-only variant: `config.proxy-only.json`, `refresh.proxy-only.py`, `controlr-vpn.service.proxy-only`),
units `/etc/systemd/system/controlr-vpn{,-refresh}.service`, `controlr-vpn-refresh.timer`.

## 4. Cache not working (cost grows, `cache_read` ≈ 0)

1. `summary.json` → `cache_regressions` (turns where the cache dropped) and the per-turn read share in `turns.jsonl`.
2. Common causes: an early byte of the transcript changed (non-deterministic text/image), effort
   changed mid-episode, a turn added > 20 blocks (lookback), the prefix is below the model's
   minimum (Haiku 4.5 — 4096 tokens), requests landed on different upstream router accounts.
3. Isolated check: `uv run controlr bench-cache --model <id> --turns 20` (⚠️ ~20 calls).

## 5. Disk

Locally, runs (`runs/`) and clones (`research/repos/`) are the main consumers; on compute3 —
`~/controlr/runs`. Before heavy work: `df -h /`. The `~/.cache/uv` and `~/.cache/huggingface`
caches are shared — clean only our own and only by agreement
([incident 2026-10-02](incidents/2026-10-02-research-run-side-effects.md)).

## 6. Real UR3 (once the backend exists)

Only with explicit per-session permission and a human at the e-stop. Drivers and `SafetyMonitor`
come from PHANTOM. Emergency order: e-stop → `robot.hold()` → read the logs → incident report.
