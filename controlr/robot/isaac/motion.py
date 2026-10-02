"""Joint-space trajectory timing and interpolation for the Isaac backend (numpy only).

Imported by both the client (segment durations) and the server (interpolation),
so the speed limits can be unit-tested without a GPU.

WHY a separate module: the review (control-safety #3, #7) found that
  * one joint-linear segment per MOVE bends the TCP path (up to 115 mm off the
    commanded line near a wrist singularity) — the safety envelope now sends IK
    waypoints along the straight TCP line, and the server must pass THROUGH them
    in one smooth motion (not stop at each 5 mm waypoint);
  * durations were computed from AVERAGE speeds, but the min-jerk profile peaks
    at 15/8 of the average — 0.28 m/s against a 0.15 m/s TCP limit.

A "group" is one action: q_from -> waypoints... -> q_target. The time scaling is
one min-jerk profile s(t) over the whole group; the joint path q(u) is the
polyline through the waypoints, parameterised by cumulative joint-space arc
length, so the peak joint speed is (15/8) * L / T and the duration below keeps
it at or under the limits.
"""

from __future__ import annotations

import numpy as np

MIN_JERK_PEAK = 15.0 / 8.0          # peak / average speed of s = 10s^3 - 15s^4 + 6s^5


def min_jerk(s: float) -> float:
    s = min(max(float(s), 0.0), 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def group_duration(q_from: np.ndarray, waypoints: np.ndarray, tcp_path_len: float, rot_angle: float, *,
                   tcp_speed: float, joint_speed: float, rot_speed: float, min_s: float) -> float:
    """Duration (s) of one group such that the PEAK TCP, joint and tool rotation speeds
    of the min-jerk profile stay at or below the limits. ``waypoints`` (N,6) ends with
    the target; ``tcp_path_len`` (m) and ``rot_angle`` (rad) describe the TCP motion."""
    pts = np.vstack([np.asarray(q_from, float)[None], np.asarray(waypoints, float).reshape(-1, 6)])
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(seg_len.sum())
    # joint j moves at |seg_ij| / seg_len_i of the path speed du/dt on segment i
    ok = seg_len > 1e-12
    share = float(np.max(np.abs(seg[ok]) / seg_len[ok, None])) if np.any(ok) else 0.0
    t = max(tcp_path_len / tcp_speed, share * total / joint_speed, rot_angle / rot_speed)
    return max(MIN_JERK_PEAK * t, min_s)


def interpolate_group(q_from: np.ndarray, waypoints: np.ndarray, n_steps: int) -> np.ndarray:
    """(n_steps, 6) joint targets for steps k = 1..n_steps of one group: min-jerk in
    time, piecewise linear in joint space through the waypoints (arc-length param)."""
    pts = np.vstack([np.asarray(q_from, float)[None], np.asarray(waypoints, float).reshape(-1, 6)])
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    out = np.empty((max(1, n_steps), 6))
    for k in range(1, len(out) + 1):
        if total <= 1e-12:
            out[k - 1] = pts[-1]
            continue
        d = min_jerk(k / len(out)) * total
        i = int(np.clip(np.searchsorted(cum, d, side="right") - 1, 0, len(seg_len) - 1))
        f = 0.0 if seg_len[i] <= 1e-12 else (d - cum[i]) / seg_len[i]
        out[k - 1] = pts[i] + min(max(f, 0.0), 1.0) * (pts[i + 1] - pts[i])
    return out
