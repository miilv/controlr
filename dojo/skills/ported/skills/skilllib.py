# Vendored unchanged from robot-harness (Mikhail's side) a21dd6f6f2db323bd0fe84f8afd20312cb8dffc6, skills/skilllib.py.
# Ported into controlr by stream P (2026-10-28) through dojo/skills/ported/harness_robot.py; logic not modified.
"""Shared math for LIBERO skills. Pure numpy over server replies (no sim internals).
Geometry comes from obs() in privileged mode: objects[name] = {pos, quat(wxyz), aabb[lo,hi], size(body frame), center, kind},
regions[name] = {pos, quat, half_size, parent}, eef.single = {pos, quat, finger_gap_m, ...}, robot_base."""
import numpy as np

R_DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
MAX_OPEN = 0.079            # Panda finger gap fully open (m) [measured, ADAPTER-libero 3]
ENCLOSE_MAX = 0.072         # narrower side above this -> rim pinch (probe v1-v4)


class SkillError(Exception):
    pass


def quat2mat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def yaw_for_closing_axis(u):
    """Top-down tool yaw (deg) whose finger-closing axis is along horizontal u. Yaw 0 closes along world y."""
    yaw = np.arctan2(u[1], u[0]) - np.pi / 2
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2           # 2-finger grasp is symmetric under 180 deg
    return float(np.degrees(yaw))


def need_privileged(o):
    if "objects" not in o:
        raise SkillError("this skill needs object poses: start the server with --privileged (tier A) "
                         "or add a perception adapter")
    return o["objects"]


def resolve(o, name, allow_regions=False):
    """Exact name, else a unique substring match over objects (+ regions). Returns (kind, name, info)."""
    objs = need_privileged(o)
    regs = o.get("regions", {}) if allow_regions else {}
    if name in objs:
        return "object", name, objs[name]
    if name in regs:
        return "region", name, regs[name]
    cand = [n for n in objs if name in n] + [n for n in regs if name in n]
    if len(cand) == 1:
        n = cand[0]
        return ("object", n, objs[n]) if n in objs else ("region", n, regs[n])
    raise SkillError(f"{name!r} matches {cand if cand else 'nothing'}; objects: {sorted(objs)}"
                     + (f"; regions: {sorted(regs)}" if allow_regions else ""))


def aabb(info):
    lo, hi = info["aabb"]
    return np.asarray(lo, float), np.asarray(hi, float)


def eef_pos(o):
    return np.asarray(o["eef"]["single"]["pos"], float)


def horizontal_axes(info):
    """World horizontal directions of the body x and y axes plus body-frame widths along them."""
    R = quat2mat(info["quat"]); sx, sy = float(info["size"][0]), float(info["size"][1])
    ax, ay = R[:, 0].copy(), R[:, 1].copy()
    if abs(R[2, 2]) < 0.7:                                 # body not upright: fall back to world AABB
        lo, hi = aabb(info)
        return np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), hi[0] - lo[0], hi[1] - lo[1]
    ax[2] = 0; ay[2] = 0
    return ax / np.linalg.norm(ax), ay / np.linalg.norm(ay), sx, sy


def obstacle_boxes(o, exclude, min_top=-np.inf, fixtures=True):
    out = []
    for n, info in need_privileged(o).items():
        if n in exclude or "aabb" not in info:
            continue
        if not fixtures and info.get("kind") == "fixture":
            continue
        lo, hi = aabb(info)
        if hi[2] > min_top:
            out.append((n, lo, hi))
    return out


def xy_clearance(points, boxes):
    clr = 1.0
    for _, lo, hi in boxes:
        for p in points:
            dx = max(lo[0] - p[0], 0.0, p[0] - hi[0]); dy = max(lo[1] - p[1], 0.0, p[1] - hi[1])
            clr = min(clr, float(np.hypot(dx, dy)))
    return clr


def carry_height(o, a_xy, b_xy, exclude, extra_tops=(), margin=0.05, corridor=0.12):
    """Max top z of non-fixture objects within `corridor` m of the a->b segment (+ extra tops) + margin."""
    a, b = np.asarray(a_xy, float), np.asarray(b_xy, float)
    tops = list(extra_tops)
    for n, lo, hi in obstacle_boxes(o, exclude, fixtures=False):
        c = (lo[:2] + hi[:2]) / 2.0
        t = np.clip(np.dot(c - a, b - a) / max(np.dot(b - a, b - a), 1e-9), 0, 1)
        if np.linalg.norm(c - (a + t * (b - a))) < corridor:
            tops.append(hi[2])
    return (max(tops) if tops else -np.inf) + margin
