# 2026-10-04 — compute3 network: old VPNs removed, router direct, own proxy for chatgpt/anthropic

**Why:** every router turn re-uploads the transcript (≈ 0.75 MB by turn 16) and compute3's Happ VPN
uploaded at 42–138 KB/s → 5–17 s per LLM call ([luna-latency](2026-10-04-luna-latency.md)). Ilia:
delete all VPN configs on compute3, start clean, use the same VPN as the dev box, with a separate
subscription link for compute3 that can be turned off.

**Removed (with backup):** Happ 3.3.6 (`happd` root service, GUI + xray core, sing-box tun `tun0`,
policy table 2022), AmneziaVPN (root service, no tunnel up), OpenVPN + NetworkManager plugin (no
configs). Two phases: (A) stop/disable with a 5-min dead-man timer that would re-enable everything if
ssh (tailscale) did not survive — it did; (B) back up, `apt purge` (4 packages, nothing else),
delete `/opt/AmneziaVPN`, `/opt/happ`, user configs, `/etc/openvpn`. Kept: Wi-Fi profiles (the only
uplink — `eno1` has no carrier), tailscale (our ssh path), docker. Backup + `rollback.sh`:
`compute3:~/controlr-netbackup-20261004/` (218 MB; the Happ .deb moved there from `~/Downloads`).
The disk is 100 % full (15 GB free) — backups kept small for that reason.

**Subscription:** the dev box's own `vpn-retranslator` (re-serves the Nofox Remnawave subscription
under one HWID / device slot) got a `compute3` profile: WiFi nodes, no US/CA/LTE (13 nodes),
`https://vpn.smrtx.ru/sub/<token>`; turning it off = delete the profile block + restart the service.
Config backup on the dev box: `~/vpn-retranslator/config.toml.bak-20261004-compute3`.

**New:** `controlr-vpn` — sing-box 1.12.17 (the dev box's version) in `~/controlr-vpn`, HTTP proxy
127.0.0.1:18810 / SOCKS 18811, **no tun** (nothing on compute3 changes unless a process sets
`HTTPS_PROXY`), urltest over the 13 nodes; `controlr-vpn-refresh.timer` re-fetches every 6 h and
restarts only when the node list changes. Units and scripts: `scripts/compute3_vpn/`, RUNBOOK §3a.

**Measured (`scripts/net_check.sh`, 740 KB body to the router):**

| path | upload | 740 KB | router first byte | chatgpt.com / anthropic |
|---|---|---|---|---|
| Happ (before) | ~50 KB/s | 14.7–15.5 s | 2–3 s | reachable |
| controlr-vpn (exit FI) | 0.9–1.2 MB/s | 0.61–0.82 s | 0.31–0.94 s | reachable (401) |
| direct, Skoltech Wi-Fi | 1.1–1.5 MB/s | 0.48–0.69 s | 0.18–0.20 s | **403 (geo-block)** |

So: router runs need no proxy at all (direct is fastest); direct Codex / Anthropic calls use
`HTTPS_PROXY=http://127.0.0.1:18810`. The ssh tunnel to the dev box is no longer needed.

**Side effect for other users of compute3:** the machine no longer has a system-wide VPN; anything
that relied on Happ for blocked sites now goes direct (Ilia's decision).

**Next:** ethernet on `eno1` (Ilia) — `controlr-vpn` is not bound to an interface, so it follows the
default route automatically; re-run `scripts/net_check.sh` after the switch.

## Correction (same day): machine-wide, like the dev box

The proxy-only variant above was my choice while Happ still held the tun; Ilia asked for "the same VPN
as the dev box", which is machine-wide. Switched at 17:32 MSK (dead-man rollback to proxy-only armed,
not needed): `tun0` + auto_route, the dev box's rules (RU domains / geoip-ru / private / torrent
direct, DNS 1.1.1.1 direct), plus compute3-specific direct rules for the router and the subscription
host, tailscale's ranges excluded; proxies on 10809/10808 (Happ's old ports) and 18810/18811; runs
as root (tun). This also undoes the side effect noted above — other users of compute3 have a VPN again.

| path (no proxy variables) | 740 KB to the router | chatgpt.com / anthropic |
|---|---|---|
| machine-wide `controlr-vpn` | 0.43–0.48 s (1.5–1.7 MB/s; router direct by rule) | 401 in 0.29–0.32 s |

Egress Helsinki (urltest), ya.ru direct (32 ms), tailscale direct path to the dev box intact.
