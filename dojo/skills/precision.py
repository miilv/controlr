"""Precision skills (skills-5): tool strikes in order, many-object insertions, nuts screwed onto bolts.

Generic library calls, chosen by ``planner.plan`` from the instruction words and the objects (asset geometry and
functional frames), never from a task name:

* ``strike_in_order(tool, instrument, points)``  pick a tool by its handle and strike the instrument's target points
                                                 one after another with the tool's working end, lifting it
                                                 clearly between strikes (a mallet on xylophone keys);
* ``insert_each(objects, receptacle)``           insert several identical objects into free holes of one receptacle,
                                                 one by one, upright, banking after each (tubes into a rack);
* ``screw_on(pairs)``                            put each nut on its bolt and turn it down the thread with re-grasps
                                                 (a nut onto the bolt of the same colour).

Every skill returns {"ok", "stage", ...}; failures carry a ``SkillResult`` (status = G3 category).
"""

from __future__ import annotations

import math
import re

import numpy as np

from dojo.skills import grasps as G
from dojo.skills.executor import ArmPlan, BudgetExceeded, EpisodeEnded
from dojo.skills.geometry import T_from, axis_angle_R, inv_T
from dojo.skills.kinematics import ARMS, TABLE_Z, X5Kin
from dojo.skills.motion import FAST, MEDIUM, held_box
from dojo.skills.skills import CLOSED, OPEN, SkillResult, Skills

# ============================================================================ geometry helpers


def frame_points_w(o, tag: str, active: bool = False) -> np.ndarray:
    """World positions of a functional point's frames (N, 3)."""
    raw = ((o.asset.active_functional if active else o.asset.functional) or {}).get(tag) or {}
    P = np.asarray([np.asarray(f, float)[:3] for f in (raw.get("frame") or [])], float).reshape(-1, 3)
    return (o.R @ P.T).T + o.pos if len(P) else P


def surface_z(o, xy, r: float = 0.0) -> float | None:
    """World z of the object's mesh surface seen from above at world ``xy`` (the highest triangle there; with
    ``r`` > 0 the highest over a disc of that radius, 9 rays). None without a mesh or when no triangle is hit."""
    if o.asset.mesh is None:
        return None
    V, F = o.asset.mesh
    Vw = (o.R @ np.asarray(V, float).T).T + o.pos
    a, b, c = Vw[F[:, 0]], Vw[F[:, 1]], Vw[F[:, 2]]
    pts = [np.asarray(xy, float)[:2]]
    if r > 0:
        pts += [pts[0] + r * np.array([math.cos(t), math.sin(t)]) for t in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
    best = None
    for p in pts:
        lo = np.minimum(np.minimum(a[:, :2], b[:, :2]), c[:, :2])
        hi = np.maximum(np.maximum(a[:, :2], b[:, :2]), c[:, :2])
        m = np.all((lo <= p + 1e-9) & (hi >= p - 1e-9), axis=1)
        if not m.any():
            continue
        ta, tb, tc = a[m], b[m], c[m]
        d = (tb[:, 1] - tc[:, 1]) * (ta[:, 0] - tc[:, 0]) + (tc[:, 0] - tb[:, 0]) * (ta[:, 1] - tc[:, 1])
        ok = np.abs(d) > 1e-14
        d = np.where(ok, d, 1.0)
        l1 = ((tb[:, 1] - tc[:, 1]) * (p[0] - tc[:, 0]) + (tc[:, 0] - tb[:, 0]) * (p[1] - tc[:, 1])) / d
        l2 = ((tc[:, 1] - ta[:, 1]) * (p[0] - tc[:, 0]) + (ta[:, 0] - tc[:, 0]) * (p[1] - tc[:, 1])) / d
        l3 = 1 - l1 - l2
        ins = ok & (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
        if not ins.any():
            continue
        z = (l1 * ta[:, 2] + l2 * tb[:, 2] + l3 * tc[:, 2])[ins].max()
        best = float(z) if best is None else max(best, float(z))
    return best


def section_points(mesh, n, s: float) -> np.ndarray:
    """Points where the mesh's triangle edges cross the plane ``x . n = s`` (object frame): its cross-section."""
    V, F = np.asarray(mesh[0], float), np.asarray(mesh[1], int)
    d = V @ np.asarray(n, float) - s
    out = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = F[:, i], F[:, j]
        m = (d[a] * d[b]) < 0
        if m.any():
            t = (d[a[m]] / (d[a[m]] - d[b[m]]))[:, None]
            out.append(V[a[m]] + t * (V[b[m]] - V[a[m]]))
    return np.vstack(out) if out else np.zeros((0, 3))


_HM_CACHE: dict = {}


def height_map(o, cell: float = 0.002) -> dict | None:
    """Top-surface height map of an object's mesh in its own frame (max z of dense surface samples per ``cell``,
    dilated by one cell), for ``Collision.heightmaps``. Cached per asset."""
    if o.asset.mesh is None:
        return None
    key = (o.asset.key, cell)
    if key in _HM_CACHE:
        return _HM_CACHE[key]
    V, F = np.asarray(o.asset.mesh[0], float), np.asarray(o.asset.mesh[1], int)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    L = np.maximum(np.maximum(np.linalg.norm(b - a, axis=1), np.linalg.norm(c - a, axis=1)), np.linalg.norm(c - b, axis=1))
    nseg = np.clip(np.ceil(L / (0.7 * cell)), 1, 24).astype(int)
    pts = [V]
    for n in np.unique(nseg):
        m = nseg == n
        u, v = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))
        keep = (u + v) <= 1.0
        u, v = u[keep], v[keep]
        P = a[m][:, None, :] + u[None, :, None] * (b[m] - a[m])[:, None, :] + v[None, :, None] * (c[m] - a[m])[:, None, :]
        pts.append(P.reshape(-1, 3))
    P = np.vstack(pts)
    x0, y0 = float(V[:, 0].min()) - cell, float(V[:, 1].min()) - cell
    nx = int(math.ceil((float(V[:, 0].max()) + cell - x0) / cell)) + 1
    ny = int(math.ceil((float(V[:, 1].max()) + cell - y0) / cell)) + 1
    Z = np.full((ny, nx), -np.inf)
    ix = ((P[:, 0] - x0) / cell).astype(int)
    iy = ((P[:, 1] - y0) / cell).astype(int)
    np.maximum.at(Z, (iy, ix), P[:, 2])
    Zd = Z.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            Zd = np.maximum(Zd, np.roll(np.roll(Z, dy, 0), dx, 1))
    hm = {"x0": x0, "y0": y0, "cell": cell, "Z": Zd, "center": np.asarray(o.asset.center, float)}
    _HM_CACHE[key] = hm
    return hm


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _align_R(a, b) -> np.ndarray:
    """Rotation taking unit vector ``a`` onto unit vector ``b`` (shortest arc)."""
    a, b = _unit(a), _unit(b)
    v = np.cross(a, b)
    s, c = float(np.linalg.norm(v)), float(a @ b)
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        p = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
        return axis_angle_R(_unit(p), math.pi)
    return axis_angle_R(v / s, math.atan2(s, c))


# ============================================================================ strike_in_order

LOW_TIP = 0.0006              # m: fingertips over the table for a pick of a part lying on it (a coin, a fallen handle)
PINCH_OFFSET = 0.004          # m: a flat disc is pinched this far off its centre (it hangs below the pads, on edge;
                              # twin: 6.5 mm off, the contacts near the pads' edge, 2/5 slid out at the lift)
STRIKE_PRESS = 0.0015         # the working end is commanded this far into the surface (a light touch)
STRIKE_LIFT = 0.07            # between strikes the working end's lowest point rises this far above the surface
STRIKE_DOWN_RAD = 0.06        # rad/step of the stroke down (a strike, not a press)
STRIKE_PITCHES = (5.0, 20.0, 35.0, 50.0)     # deg: the handle rises from the working end towards the hand
STRIKE_HEADINGS = tuple(range(0, 360, 30))   # deg, world heading of the handle (working end -> hand)
STRIKE_ROLLS = tuple(range(0, 360, 30))      # deg about the handle


def tool_work_point(tool, tag: str | None = None) -> tuple[np.ndarray, str | None]:
    """The tool's working point in its own frame: the mean of an active functional point's frames (a mallet's
    "beat"), else the end of its long axis farther from the centre of mass."""
    af = tool.asset.active_functional or {}
    tags = [tag] if tag else [t for t in af if re.search(r"beat|hit|strike|head|tip|work", t.lower())] or list(af)
    for t in tags:
        fr = (af.get(t) or {}).get("frame") or []
        if fr:
            return np.mean([np.asarray(f, float)[:3] for f in fr], axis=0), t
    ext = tool.asset.extents
    k = int(np.argmax(ext))
    p = tool.asset.center.copy()
    com = G.mesh_com(tool.asset) if tool.asset.mesh is not None else tool.asset.center
    sgn = -1.0 if com[k] >= tool.asset.center[k] else 1.0
    p[k] += sgn * ext[k] / 2
    return p, None


def handle_grasps(sk: Skills, o, arm: str, work_loc, dists=(0.13, 0.15, 0.11, 0.17, 0.09, 0.07),
                  tilts=(0.0, 20.0, -20.0)) -> list[G.Grasp]:
    """Top grasps across a stick-like object's handle at ``dists`` (m) from its working point along its long axis:
    the pads centred on the handle's cross-section there (from the mesh), the closing axis horizontal and across
    the handle, the approach from above (tilted about the handle by ``tilts``). Ranked by distance order."""
    ext = o.asset.extents
    k = int(np.argmax(ext))
    u_loc = np.zeros(3)
    u_loc[k] = 1.0 if o.asset.center[k] >= work_loc[k] else -1.0
    V = np.asarray(o.asset.mesh[0], float) if o.asset.mesh is not None else None
    out = []
    for rank, d in enumerate(dists):
        p_loc = np.asarray(work_loc, float) + u_loc * d
        if not (o.asset.bbox_min[k] + 0.005 <= p_loc[k] <= o.asset.bbox_max[k] - 0.005):
            continue
        width = float(np.delete(ext, k).min())
        sl = None
        if V is not None:
            sl = section_points(o.asset.mesh, u_loc, float(p_loc @ u_loc))
            if len(sl) < 3:
                continue
            c = sl.mean(0)
            p_loc = p_loc + (c - p_loc) - u_loc * float((c - p_loc) @ u_loc)
        p_w = o.R @ p_loc + o.pos
        u_w = o.R @ u_loc
        if abs(float(u_w[2])) > 0.85:
            continue
        closing = _unit(np.cross(u_w, [0.0, 0.0, 1.0]))
        appr0 = _unit(np.cross(closing, u_w))
        if appr0[2] > 0:
            appr0 = -appr0
        if sl is not None:
            cl_loc = o.R.T @ closing
            width = float(np.ptp(sl @ cl_loc))
        if width > G.MAX_WIDTH:
            continue
        for tilt in tilts:
            appr = axis_angle_R(u_w, math.radians(tilt)) @ appr0
            for flip in (1.0, -1.0):
                R = G._frame(appr, flip * closing)
                T = T_from(p_w, R)
                # fingertips >= 2.5 mm over the table: raise the TCP (the pads still cover a handle a few mm thick);
                # a handle lying on the table (its tool fell off its stand): fingertips 0.6 mm over the table, the
                # pads still cover its whole section (``LOW_TIP``; the pick is planned with no fingertip margin)
                g_open = G.gap_to_grip(min(G.MAX_GAP, width + 2 * G.CLEAR + 0.006))
                low = float(sk.kin.finger_points(sk.kin.flange_from_tcp(T), g_open)[:, 2].min())
                need = TABLE_Z + 0.0025 - low
                tag = ""
                if need > 0:
                    if need > 0.4 * width + 0.002:
                        if float(p_w[2]) - TABLE_Z > 0.6 * width + 0.004:
                            continue
                        need = TABLE_Z + LOW_TIP - low
                        tag = " low"
                    T[2, 3] += need
                out.append(G.Grasp("handle", T, width, pre_dist=0.06, score=0.1 * rank + abs(tilt) / 100.0 + (0.3 if tag else 0),
                                   note=f"handle {d*1000:.0f} mm from the working end, tilt {tilt:+.0f}{tag}"))
    return out


def strike_orientations(u_loc, R_tool_tcp, min_down: float = 0.2) -> list[tuple[np.ndarray, tuple]]:
    """Tool orientations for a strike: the handle (``u_loc``, tool frame, working end -> hand) at every world heading
    and rise of the grid, rolled about itself; kept when the gripper (tool -> TCP rotation ``R_tool_tcp``) points at
    least ``min_down`` downwards. Sorted most-downward first."""
    out = []
    for psi in STRIKE_HEADINGS:
        for pitch in STRIKE_PITCHES:
            pr, h = math.radians(pitch), math.radians(psi)
            u_w = np.array([math.cos(pr) * math.cos(h), math.cos(pr) * math.sin(h), math.sin(pr)])
            R0 = _align_R(u_loc, u_w)
            for phi in STRIKE_ROLLS:
                R = axis_angle_R(u_w, math.radians(phi)) @ R0
                az = float((R @ R_tool_tcp)[2, 0])
                if az < -min_down:
                    out.append((R, (psi, pitch, phi, az)))
    return sorted(out, key=lambda x: x[1][3])


def _strike_targets(R, T_tcp_tool, work_loc, head, p, zs) -> tuple[np.ndarray, np.ndarray]:
    """TCP poses over a target (``STRIKE_LIFT`` above the touch) and at the touch, for tool orientation ``R``."""
    low = float((head @ R.T)[:, 2].min()) if len(head) else 0.0          # lowest working-end point below the root
    wp = R @ work_loc
    T_dn = T_from(np.array([p[0] - wp[0], p[1] - wp[1], zs - STRIKE_PRESS - low]), R) @ inv_T(T_tcp_tool)
    T_up = T_dn.copy()
    T_up[2, 3] += STRIKE_LIFT + STRIKE_PRESS
    return T_up, T_dn


def plan_strikes(sk: Skills, a: str, T_tcp_tool, work_loc, head, targets, ins, q0, max_orient: int = 400):
    """One tool orientation that strikes every target with arm ``a`` (IK over each target, a straight stroke down,
    the hand clear of the instrument), chained from ``q0``; the fewest joint travel wins among the first feasible
    ones. Returns (orientation key, [(q_up, stroke_down, stroke_up), ...]) or (None, note)."""
    R_tool_tcp = inv_T(T_tcp_tool)[:3, :3]
    u_loc = _unit(inv_T(T_tcp_tool)[:3, 3] - work_loc)
    best, n_ok_ends = None, 0
    for R, key in strike_orientations(u_loc, R_tool_tcp)[:max_orient]:
        # the two outermost targets first (cheap rejection), then the chain
        ok = True
        for (_, p, zs) in (targets[0], targets[-1]):
            T_up, T_dn = _strike_targets(R, T_tcp_tool, work_loc, head, p, zs)
            if float(sk.kin.finger_points(sk.kin.flange_from_tcp(T_dn), CLOSED)[:, 2].min()) < ins.top_z + 0.004:
                ok = False
                break
            if not sk.kin.ik(a, T_up, q0, n_random=1).ok:
                ok = False
                break
        if not ok:
            continue
        n_ok_ends += 1
        q, chain, travel = np.asarray(q0, float), [], 0.0
        for (_, p, zs) in targets:
            T_up, T_dn = _strike_targets(R, T_tcp_tool, work_loc, head, p, zs)
            r = sk.kin.ik(a, T_up, q, n_random=1)
            if not r.ok:
                ok = False
                break
            dn = sk.motion.line(a, r.q, T_dn, STRIKE_DOWN_RAD, "strike_down", gripper=CLOSED, check=False)
            up = sk.motion.line(a, dn.q_end if dn.segments else r.q, T_up, MEDIUM, "strike_up", gripper=CLOSED,
                                check=False)
            if not (dn.ok and up.ok):
                ok = False
                break
            travel += float(np.abs(r.q - q).max())
            q = r.q
            chain.append((r.q, dn, up))
        if not ok:
            continue
        cost = travel - 0.3 * abs(key[3])
        if best is None or cost < best[0]:
            best = (cost, key, chain)
        if n_ok_ends >= 12 and best is not None:
            break
    if best is None:
        return None, f"{a}: no orientation strikes every target ({n_ok_ends} reach both ends)"
    return best[1], best[2]


def strike_in_order(sk: Skills, tool: str, instrument: str, points: list[str], work_tag: str | None = None,
                    arm: str | None = None) -> dict:
    """Pick ``tool`` by its handle (``handle_grasps``: the hand well away from the working end) and strike the
    instrument's target points (passive functional tags of ``instrument``, in the given order) with the tool's
    working end (``work_tag``, default the tool's striking point): over each target, a quick stroke down until the
    working end touches the surface (``STRIKE_PRESS`` into it), and back up ``STRIKE_LIFT`` above it before the
    next. One tool orientation serves every target (``plan_strikes``); the tool is re-measured in the hand after
    every strike and the next one re-planned from it. Keeps the tool at the end (the caller's ``finish`` puts it
    down)."""
    w = sk.refresh()
    to, ins = w.obj(tool), w.obj(instrument)
    work_loc, wtag = tool_work_point(to, work_tag)
    sk.coll.include.add(instrument)
    tgt = []
    for tag in points:
        P = frame_points_w(ins, tag)
        if not len(P):
            return {"ok": False, "stage": "targets", "note": f"{instrument} has no frames for {tag}"}
        p = P.mean(0)
        zs = surface_z(ins, p[:2], r=0.004)
        tgt.append((tag, p, float(zs) if zs is not None else float(p[2])))
    V = np.asarray(to.asset.mesh[0], float) if to.asset.mesh is not None else None
    head = V[np.linalg.norm(V - work_loc, axis=1) < 0.03] if V is not None else np.asarray([work_loc])
    # the arm: a pick that leaves one tool orientation striking every target (planned with the pick's grasp)
    choice, notes = None, []
    arms = [arm] if arm else list(ARMS)
    fm = sk.coll.finger_margin
    for a in arms:
        cands = handle_grasps(sk, to, a, work_loc)
        sk.coll.finger_margin = 0.0 if any(g.note.endswith("low") for g in cands) else fm
        try:
            picks = sk.plan_pick(tool, a, grasps=cands, lift=0.06, keep=6, max_eval=60, return_all=True)
        finally:
            sk.coll.finger_margin = fm
        if not isinstance(picks, list) or not picks:
            notes.append(f"{a}: no pick ({getattr(picks, 'note', '')[:120]})")
            continue
        for pp in picks[:3]:
            key, chain = plan_strikes(sk, a, pp.T_tcp_obj, work_loc, head, tgt, ins, pp.plan.q_end)
            if key is None:
                notes.append(chain)
                continue
            travel = sum(float(np.abs(c[0] - q).max()) for c, q in zip(chain, [pp.plan.q_end] + [c[0] for c in chain]))
            cost = pp.steps + 20.0 * travel
            if choice is None or cost < choice[0]:
                choice = (cost, pp, key)
            break
    if choice is None:
        return {"ok": False, "stage": f"plan strikes with {tool}",
                "result": sk._record(SkillResult(False, "strike", "ik_fail", 0, None, note="; ".join(notes)[:600]))}
    _, pp, key = choice
    sk.log(f"[strike] {pp.arm} picks {tool} ({pp.grasp.note}); orientation heading/pitch/roll {key[:3]}")
    r = sk._record(sk.run_pick(pp))
    if not r.ok:
        return {"ok": False, "stage": f"pick {tool}", "result": r}
    a = pp.arm
    done = []
    for i, (tag, p, zs) in enumerate(tgt):
        if sk.ex.ended:
            break
        w = sk.refresh()
        to = w.obj(tool)
        T_tcp_tool = inv_T(sk.kin.tcp(a, w.arms[a].q)) @ to.T
        res = _strike_one(sk, a, tool, T_tcp_tool, work_loc, head, p, zs, ins, key)
        sk._record(res)
        if not res.ok:
            return {"ok": False, "stage": f"strike {tag} ({i + 1}/{len(tgt)})", "result": res, "struck": done}
        done.append(tag)
    return {"ok": True, "stage": "struck", "struck": done, "work_tag": wtag}


def _strike_one(sk: Skills, a: str, tool: str, T_tcp_tool, work_loc, head, p, zs: float, ins, key) -> SkillResult:
    """One strike with the measured in-hand pose: the planned orientation (``key``) if it still works, else the
    nearest feasible one; over ``p``, stroke down to touch the surface at ``zs``, back up."""
    to = sk.world.obj(tool)
    q0 = sk.ex.q[a]
    key_l, chain = plan_strikes(sk, a, T_tcp_tool, work_loc, head, [(None, p, zs)], ins, q0, max_orient=1000)
    ors = strike_orientations(_unit(inv_T(T_tcp_tool)[:3, 3] - work_loc), inv_T(T_tcp_tool)[:3, :3])
    # prefer the planned orientation: same heading / pitch / roll when it is feasible from here
    planned = [(R, k) for R, k in ors if k[:3] == key[:3]]
    use = None
    for R, k in planned:
        T_up, T_dn = _strike_targets(R, T_tcp_tool, work_loc, head, p, zs)
        r = sk.kin.ik(a, T_up, q0, n_random=1)
        if not r.ok or float(sk.kin.finger_points(sk.kin.flange_from_tcp(T_dn), CLOSED)[:, 2].min()) < ins.top_z + 0.004:
            continue
        dn = sk.motion.line(a, r.q, T_dn, STRIKE_DOWN_RAD, "strike_down", gripper=CLOSED, check=False)
        up = sk.motion.line(a, dn.q_end if dn.segments else r.q, T_up, MEDIUM, "strike_up", gripper=CLOSED, check=False)
        if dn.ok and up.ok:
            use = (k, (r.q, dn, up))
    if use is None:
        if key_l is None:
            return SkillResult(False, "strike", "ik_fail", 0, a, note=str(chain)[:300])
        use = (key_l, chain[0])
    k, (q_up, dn, up) = use
    box = held_box(to, T_tcp_tool)
    go = sk.motion.to_pose(a, q0, sk.kin.tcp(a, q_up), FAST, "strike_over", gripper=CLOSED, held=box, ignore=(tool,))
    if not go.ok:
        go = sk.motion.joint(a, q0, q_up, FAST, "strike_over", gripper=CLOSED, check=False)
    pl = ArmPlan(a, np.asarray(q0, float), sk.ex.g[a]).move(go, "strike_over").hold(1, "aim")
    pl.move(dn, "strike_down").hold(1, "touch").move(up, "strike_up")
    try:
        r = sk.ex.run({a: pl}, name=f"strike {tool}")
    except BudgetExceeded as e:
        return SkillResult(False, "strike", "step_budget", 0, a, note=str(e))
    except EpisodeEnded as e:
        return SkillResult(True, "strike", "ok", pl.steps, a, note=f"episode ended ({e})")
    return SkillResult(True, "strike", "ok", r["steps"], a, note=f"heading/pitch/roll {k[:3]}")


# ============================================================================ thin discs into slots (insert_into)

DISC_THIN = 0.005             # m: an object this thin with two equal other extents is a disc (a coin)


def disc_axis(o) -> int | None:
    """The normal axis (object frame index) of a thin disc: smallest extent < ``DISC_THIN`` and the other two
    within 5 % of each other. None otherwise."""
    ext = np.asarray(o.asset.extents, float)
    k = int(np.argmin(ext))
    a, b = np.delete(ext, k)
    if ext[k] < DISC_THIN and abs(a - b) <= 0.05 * max(a, b):
        return k
    return None


def spin_about_axis(o, T_obj, k: int, n: int = 12) -> list[np.ndarray]:
    """``T_obj`` turned about the object's own axis ``k`` through its bbox centre in ``n`` steps (a disc is the same
    disc at every turn about its normal; an insert frame fixes one of them)."""
    c = o.asset.center
    out = []
    for i in range(n):
        ax = np.zeros(3)
        ax[k] = 1.0
        Rk = axis_angle_R(ax, 2 * math.pi * i / n)
        Tl = np.eye(4)
        Tl[:3, :3] = Rk
        Tl[:3, 3] = c - Rk @ c
        out.append(np.asarray(T_obj, float) @ Tl)
    return out


def edge_grasps(sk: Skills, o, arm: str, floor_z: float, tilts=(0.0, 12.0, -12.0, 20.0, -20.0, 28.0, -28.0),
                min_cover: float = 0.005) -> list[G.Grasp]:
    """Grasps of a disc standing on its edge (a coin in a holder) by its upper rim: the pads on its two faces, the
    approach in the disc's plane from above (tilted in that plane by ``tilts``), as deep as the fingers may go with
    every finger point at or above ``floor_z`` (the holder's top + clearance); at least ``min_cover`` of the disc
    between the pads."""
    k = disc_axis(o)
    if k is None:
        return []
    n = o.R[:, k]
    up = np.array([0.0, 0.0, 1.0]) - n * float(n[2])
    if np.linalg.norm(up) < 0.5:
        return []                                     # lying flat: not on its edge
    up = _unit(up)
    side = np.cross(n, up)
    r = float(np.delete(o.asset.extents, k).max()) / 2
    c = o.center_w
    width = float(o.asset.extents[k])
    g_open = G.gap_to_grip(min(G.MAX_GAP, width + 2 * G.CLEAR + 0.006))
    out = []
    for tilt in tilts:
        t = math.radians(tilt)
        appr = -(math.cos(t) * up + math.sin(t) * side)
        for flip in (1.0, -1.0):
            R = G._frame(appr, flip * n)
            best = None
            for s_ in np.arange(r + 0.012, r - 0.013, -0.0005):      # TCP this far from the centre, against the approach
                T = T_from(c - appr * s_, R)
                low = float(sk.kin.finger_points(sk.kin.flange_from_tcp(T), g_open)[:, 2].min())
                if low >= floor_z:
                    best = (s_, T)
                elif best is not None:
                    break
            if best is None:
                continue
            s_, T = best
            cover = r - s_ + 0.0126
            if cover < min_cover:
                continue
            out.append(G.Grasp("edge", T, width, pre_dist=0.05, score=abs(tilt) / 100.0 - 10.0 * cover,
                               note=f"edge tilt {tilt:+.0f} cover {cover*1000:.0f} mm"))
    return out


def pinch_grasps(sk: Skills, o, arm: str, tip_clear: float = LOW_TIP, n_dir: int = 6,
                 offsets=(PINCH_OFFSET, -PINCH_OFFSET)) -> list[G.Grasp]:
    """Grasps of a disc lying flat (a coin on the table): straight down, the pads closing across a diameter with
    the fingertips ``tip_clear`` above the table (the disc is thinner than the usual fingertip clearance), the hand
    ``offsets`` (m) off the centre along the pads' width: turned on edge, the disc then hangs below the pads' line
    (its centre of mass under the two rim contacts) and reaches further below the fingers (twin: pinched at its
    centre a coin went 2 mm into the slot and fell over on the bank, 3/3)."""
    k = disc_axis(o)
    if k is None:
        return []
    nz = o.R[:, k]
    if abs(float(nz[2])) < 0.9:
        return []
    c = o.center_w
    width = float(np.delete(o.asset.extents, k).max())
    out = []
    from dojo.skills.kinematics import down_R
    for i in range(n_dir):
        yaw = math.pi * i / n_dir
        R = down_R(yaw)
        for e in offsets:
            p = np.array([c[0], c[1], max(TABLE_Z, o.bottom_z) + tip_clear + 0.0126]) + R[:, 2] * e
            out.append(G.Grasp("pinch", T_from(p, R), width, pre_dist=0.05, score=0.01 * i,
                               note=f"pinch yaw {math.degrees(yaw):.0f} off {e*1000:+.1f} mm"))
    return out


def disc_hangs(o):
    """``pair_ok`` for ``plan_transfer``: the disc's centre below the TCP at the place (it hangs under the
    pads' line, so most of it is below the fingers)."""
    def ok(g, T_tcp_obj, T_obj, T_tcp) -> bool:
        if g.method != "pinch":
            return True
        c = T_obj[:3, :3] @ o.asset.center + T_obj[:3, 3]
        return float(c[2]) < float(T_tcp[2, 3]) - 0.5 * PINCH_OFFSET
    return ok


def holder_of(sk: Skills, label: str) -> str | None:
    """The fixed object ``label`` stands in (its bbox centre inside the other's footprint, within its height)."""
    w = sk.world
    o = w.obj(label)
    p = o.center_w
    for r in w.objects.values():
        if r.label == label or r.clutter or r.obj_type == "Rigid" or r.category == "camera_stand":
            continue
        mn, mx = r.aabb()
        if np.all(p[:2] >= mn[:2] - 0.003) and np.all(p[:2] <= mx[:2] + 0.003) and mn[2] - 0.01 <= p[2] <= mx[2] + 0.02:
            return r.label
    return None


DISC_OUT = 0.03               # m: a disc's lift out of its holder before the carry (8 cm straight up left the reach of
                              # an arm 0.52 m out)
DISC_LIFT = 0.10              # m: a disc no arm can insert is still lifted this far from its holder and set back
DISC_TABLE_RELAY = False      # the table relay (laid flat, pinched by the other arm) is off: a pinched coin hangs from
                              # the pads' line, and a hand pointing sideways keeps its 37 mm finger roots over the
                              # bank's back, so the coin stays 7 mm above the slot (twin 0/7, it fell on the bank)
DISC_SQUEEZE = 4              # still steps after the close before a disc is lifted (twin: lifted at once, it stayed)
DISC_LIFT_RAD = 0.03          # rad/step of a disc's lift out of its holder (twin: at 0.10 it stayed in the holder 5/5,
                              # at 0.03 and 0.01 it came out 3/3)
DISC_PLACE_KW = {"pre_h": 0.04, "pre_clear": 0.03, "descend_rad": 0.03, "release_gap": 0.0, "settle": 2, "pre_settle": 3, "servo": True,
                 "fingers_out": True, "probe": False, "hover_clear": 0.006, "seat_tol": 0.05}


def disc_into_slot(sk: Skills, obj: str, rec: str, targets: list[np.ndarray]) -> dict:
    """A thin disc (``disc_axis``) edge first into a slot of ``rec`` (``targets``: insert poses from the assets, the
    disc's centre at or below the opening): picked by its upper rim with the fingers clear of what it stands in
    (``edge_grasps``), carried with its plane along the slot, lowered with the fingertips just above the
    receptacle's top and let go (it drops in; ``DISC_PLACE_KW``). Every turn of the disc about its normal is a
    target. When no arm reaches both ends: a table relay (laid flat near the middle by one arm, picked flat with a
    pinch by the other, ``pinch_grasps``)."""
    from dojo.skills.library import place_oriented, free_spots
    w = sk.refresh()
    o = w.obj(obj)
    k = disc_axis(o)
    tg = [Tk for T in targets for Tk in spin_about_axis(o, T, k)]
    hold = holder_of(sk, obj)
    floor = max(TABLE_Z, o.bottom_z)
    if hold is not None:
        sk.coll.include.add(hold)
        sk.coll.margin_of[hold] = 0.002
        floor = max(floor, w.obj(hold).top_z)
    sk.coll.include.add(rec)
    hm = height_map(w.obj(rec))
    if hm is not None:
        sk.coll.heightmaps[rec] = hm
    place_kw = {**DISC_PLACE_KW, "servo_ref": rec, "sym_axis": k}

    def grasps(a):
        oo = sk.world.obj(obj)
        if abs(float(oo.R[2, k])) > 0.9:
            return pinch_grasps(sk, oo, a)
        return edge_grasps(sk, oo, a, floor + 0.003)
    tkw = {"grasps": grasps, "screen_fingers_out": True, "lift": DISC_OUT,
           "pick_kw": {"squeeze": DISC_SQUEEZE, "lift_rad": DISC_LIFT_RAD}}
    sk.log(f"[disc] {obj} -> {rec}: {len(tg)} targets, holder {hold}")
    direct = sk.plan_transfer(obj, tg, ignore=(rec,), place_kw=place_kw, **tkw)
    if isinstance(direct, SkillResult) and not DISC_TABLE_RELAY:
        sk.log(f"[disc] no one-arm plan ({direct.note[:200]}); lift it from its holder and set it back")
        r = disc_lift_and_return(sk, obj, floor, tkw)
        return {"ok": False, "stage": f"insert {obj}: no arm reaches both ends", "result": r}
    if isinstance(direct, SkillResult):
        sk.log(f"[disc] no one-arm plan ({direct.note[:200]}); table relay")
        r1 = disc_lay_flat(sk, obj, rec, hold, floor, tkw)
        if not r1.ok:
            return {"ok": False, "stage": f"relay {obj}", "result": r1}
    else:
        sk.log(f"[disc] {obj}: direct, {direct[0].arm} ({direct[0].grasp.note})")
    # the insertion leg (re-planned from the coin as it is now: in its holder, or lying flat after the relay)
    old_fm = sk.coll.finger_margin
    flat = abs(float(sk.refresh().obj(obj).R[2, k])) > 0.9
    sk.coll.finger_margin = 0.0 if flat else old_fm
    if flat:                                         # a pinch off the table lifts at the usual rate
        tkw = {**tkw, "pick_kw": {"squeeze": DISC_SQUEEZE}, "pair_ok": disc_hangs(sk.world.obj(obj))}
    try:
        res = place_oriented(sk, obj, tg, yaw_sym=1, tol_xy=0.02, tol_tilt=30.0, tol_z=0.06, ignore=(rec,),
                             place_kw=place_kw, air_handover=False, transfer_kw=tkw)
    finally:
        sk.coll.finger_margin = old_fm
    w = sk.refresh()
    if inside_bbox(w.obj(obj), w.obj(rec)):
        return {"ok": True, "stage": "inserted", "result": res}
    return {"ok": False, "stage": f"insert {obj}", "result": res}


def disc_lift_and_return(sk: Skills, obj: str, floor: float, tkw: dict, lift: float = DISC_LIFT) -> SkillResult:
    """Take a disc from its holder by the rim (``edge_grasps``), lift it ``lift``, lower it straight back and let go."""
    k = disc_axis(sk.world.obj(obj))
    best = None
    for a in ARMS:
        gs = edge_grasps(sk, sk.world.obj(obj), a, floor + 0.003)
        pp = sk.plan_pick(obj, a, grasps=gs, lift=lift, squeeze=DISC_SQUEEZE, lift_rad=DISC_LIFT_RAD)
        if not isinstance(pp, SkillResult) and (best is None or pp.steps < best.steps):
            best = pp
    if best is None:
        return sk._record(SkillResult(False, "lift", "ik_fail", 0, None, note=f"no rim grasp of {obj} (axis {k})"))
    r = sk._record(sk.run_pick(best))
    if r.ok:
        sk._record(sk.regrasp(best.arm))
    return r


DISC_LEAN = 25.0              # deg: a disc let go on its edge leans this much, so it falls flat to a known side


def disc_lay_flat(sk: Skills, obj: str, rec: str, hold: str | None, floor: float, tkw: dict) -> SkillResult:
    """Relay leg of a disc that no arm can carry from its holder to its slot: the arm that reaches it takes it by
    the rim (``edge_grasps``) and stands it on its edge, leaning ``DISC_LEAN``, on a free table spot between the
    arms, and lets go: it falls flat, where the other arm can pinch it (``pinch_grasps``). Banks nothing."""
    from dojo.skills.library import free_spots
    w = sk.world
    o = w.obj(obj)
    k = disc_axis(o)
    geo = [g for g in w.task_objects() if g.obj_type == "Geometry" and g.category != "camera_stand"]
    spots = [xy for xy in free_spots(sk, obj, near_xy=(0.0, -0.22), clearance=0.03)
             if all(float(np.linalg.norm(np.subtract(xy, g.center_w[:2]))) > 0.05 + float(np.max(g.asset.extents[:2])) / 2
                    for g in geo)][:4]
    n = o.R[:, k]
    up = _unit(np.array([0.0, 0.0, 1.0]) - n * float(n[2]))
    side = np.cross(n, up)
    tg = []
    for xy in spots:
        for lean in (DISC_LEAN, -DISC_LEAN):
            R = axis_angle_R(side, math.radians(lean)) @ o.R
            for T in spin_about_axis(o, T_from(np.zeros(3), R), k, n=8):
                Rr = T[:3, :3]
                h = float(np.abs(Rr[2, :]) @ (o.asset.extents / 2))
                c = np.array([xy[0], xy[1], TABLE_Z + 0.002 + h])
                tg.append(T_from(c - Rr @ o.asset.center, Rr))
    if not tg:
        return sk._record(SkillResult(False, "relay", "ik_fail", 0, None, note="no free spot between the arms"))
    kw = {"pre_h": 0.04, "release_gap": 0.0, "settle": 1, "descend_rad": 0.04}
    r = sk.plan_transfer(obj, tg, place_kw=kw, **tkw)
    if isinstance(r, SkillResult):
        return sk._record(r)
    sk.log(f"[disc] relay: {r[0].arm} lays {obj} down at {np.round(r[1].T_obj_target[:2, 3], 3).tolist()}")
    res = sk.run_transfer(*r, tol_xy=0.05, tol_tilt=90.0, tol_z=0.05)
    sk.bank()
    o2 = sk.refresh().obj(obj)
    flat = abs(float(o2.R[2, k])) > 0.9
    return sk._record(SkillResult(flat, "relay", "ok" if flat else "place_miss", res.steps, r[0].arm,
                                  note=f"{obj} {'lies flat' if flat else 'not flat'} at {np.round(o2.pos, 3).tolist()} "
                                       f"({res.note[:80]})"))


def inside_bbox(o, c, margin: float = 0.0) -> bool:
    """Is ``o``'s bbox centre inside ``c``'s oriented bbox?"""
    loc = c.R.T @ (o.center_w - c.pos)
    return bool(np.all(loc >= c.asset.bbox_min - margin) and np.all(loc <= c.asset.bbox_max + margin))


# ============================================================================ insert_each

EACH_PLACE_KW = {"pre_h": 0.02, "descend_rad": 0.02, "release_gap": 0.0, "settle": 3, "pre_settle": 3,
                 "release": True, "servo": True, "seat_tol": 0.008}
EACH_HOLES = 4                # free holes tried per object (best first)
EACH_PAD_BAND = 0.010         # m: along a long object's axis the pads cover this much each side of the TCP ...
EACH_SECTION_VAR = 0.0015     # ... where its section radius may vary this much (no grasp on a lip, a cap, a taper:
                              # twin, a tube held at its flange slid 5 mm and fell beside the rack)


def radius_profile(o, step: float = 0.002) -> tuple[int, np.ndarray, np.ndarray]:
    """(long axis k, positions along k, max section radius about the bbox axis) of a long object, from its mesh."""
    k = int(np.argmax(o.asset.extents))
    n = np.zeros(3)
    n[k] = 1.0
    c = o.asset.center
    ss = np.arange(float(o.asset.bbox_min[k]) + step / 2, float(o.asset.bbox_max[k]), step)
    rr = []
    for s_ in ss:
        P_ = section_points(o.asset.mesh, n, float(s_))
        rr.append(float(np.max(np.linalg.norm(np.delete(P_ - c, k, axis=1), axis=1))) if len(P_) else np.nan)
    return k, ss, np.asarray(rr)


def uniform_section_grasps(sk: Skills, o, arm: str, tilts=(0.0, 25.0, -25.0, 40.0, -40.0)) -> list[G.Grasp]:
    """Box grasps of a long object whose pads sit on a stretch of near-constant section (``EACH_SECTION_VAR`` over
    ``EACH_PAD_BAND`` each side of the TCP along its long axis)."""
    gs = G.box_grasps(o, sk.kin, arm, tilts=tuple(tilts))
    if o.asset.mesh is None:
        return gs
    key = ("rprof", o.asset.key)
    prof = sk._hq_cache.get(key)
    if prof is None:
        prof = radius_profile(o)
        sk._hq_cache[key] = prof
    k, ss, rr = prof
    out = []
    for g in gs:
        s_ = float((inv_T(o.T) @ np.r_[g.T[:3, 3], 1.0])[k])
        m = np.abs(ss - s_) <= EACH_PAD_BAND
        band = rr[m]
        if len(band) and np.all(np.isfinite(band)) and float(band.max() - band.min()) <= EACH_SECTION_VAR:
            out.append(g)
    return out


EACH_OVER = 0.07              # m: a filled hole this close to the line from the carrying arm's base to a hole is "in
                              # the way"


def _seg_dist(p, a, b) -> float:
    """Distance (xy) from point ``p`` to the segment ``a``-``b``."""
    p, a, b = (np.asarray(v, float)[:2] for v in (p, a, b))
    ab = b - a
    t = float(np.clip((p - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


EACH_SPREAD = 0.06            # m: holes at least this far from a filled one first (open fingers next to a filled hole)


def hole_frames(rec) -> list[tuple[str, np.ndarray, float]]:
    """A receptacle's holes from its asset: passive support frames (tag, world 4x4, radius), the frame at the hole's
    mouth (a test-tube rack's ``.../k`` support points)."""
    out = []
    for tag, lst in (rec.asset.support or {}).items():
        for c, r in lst:
            c = np.asarray(c, float)
            q = c[3:7] if len(c) >= 7 else np.array([1.0, 0, 0, 0])
            out.append((tag, rec.T @ T_from(c[:3], quat=q / np.linalg.norm(q)), float(r)))
    return out


def hole_floor(rec, xy, r: float) -> float | None:
    """World z of the floor of a hole of radius ``r`` at world ``xy`` (the lowest of 5 rays near its centre)."""
    zs = [surface_z(rec, np.asarray(xy, float) + d) for d in
          (np.zeros(2), np.array([0.3 * r, 0]), np.array([-0.3 * r, 0]), np.array([0, 0.3 * r]), np.array([0, -0.3 * r]))]
    zs = [z for z in zs if z is not None]
    return min(zs) if zs else None


def upright_targets(o, p_mouth, floor_z: float | None, depth: float, n_yaw: int = 12) -> list[np.ndarray]:
    """Object poses with its leading end (``insertion.insertion_axis``: the narrow end of its long axis) straight
    down over ``p_mouth`` (world), the end ``depth`` below the mouth (not below ``floor_z``), turned about the
    vertical in ``n_yaw`` steps."""
    from dojo.skills.insertion import R_axis_down, insertion_axis
    from dojo.skills.geometry import rot_z
    V = np.asarray(o.asset.mesh[0], float) if o.asset.mesh is not None else None
    if V is None:
        return []
    k, sg = insertion_axis(V)
    R0 = R_axis_down(k, sg)
    out = []
    for i in range(n_yaw):
        R = rot_z(2 * math.pi * i / n_yaw) @ R0
        W = V @ R.T
        tip = W[W[:, 2] <= W[:, 2].min() + 0.003]
        c_tip = tip.mean(0)
        z_end = float(p_mouth[2]) - depth
        if floor_z is not None:
            z_end = max(z_end, floor_z + 0.001)
        pos = np.array([p_mouth[0] - c_tip[0], p_mouth[1] - c_tip[1], z_end - float(W[:, 2].min())])
        out.append(T_from(pos, R))
    return out


def insert_each(sk: Skills, objects: list[str], receptacle: str, depth: float | None = None) -> dict:
    """Insert every object, one by one, leading end first and upright, into a free hole of ``receptacle`` (holes from
    its asset's support frames, ``hole_frames``): the hole farthest from the filled ones first (``EACH_SPREAD``) and on
    the object's side, the object re-oriented in the hand (lying -> standing) or relayed, a closed-loop descent into
    the hole's mouth (``EACH_PLACE_KW``: sag-corrected, depth check, probing), let go, both grippers opened after each
    (partial credit per object). ``depth``: how far the leading end goes below the mouth (default: to the floor, at
    most 0.8 of the object's length)."""
    from dojo.skills.library import place_oriented
    w = sk.refresh()
    rec = w.obj(receptacle)
    holes = hole_frames(rec)
    if not holes:
        return {"ok": False, "stage": "holes", "note": f"{receptacle} has no hole frames"}
    sk.coll.include.add(receptacle)
    hm = height_map(rec)
    if hm is not None:
        sk.coll.heightmaps[receptacle] = hm
    st = w.notes.setdefault("insert_each", {"filled": {}, "done": []})
    failed = []
    for _ in range(len(objects)):
        w = sk.refresh()
        todo = [l for l in objects if l in w.objects and l not in st["done"] and l not in failed]
        if not todo:
            break
        # the object nearest to the receptacle first (shortest carries; the far ones keep their room)
        lab = min(todo, key=lambda l: float(np.linalg.norm(w.obj(l).center_w[:2] - rec.center_w[:2])))
        o = w.obj(lab)
        L = float(np.max(o.asset.extents))
        filled = [np.asarray(v, float) for v in st["filled"].values()]
        free = []
        picker = min(ARMS, key=lambda a_: float(np.linalg.norm(sk.kin.base(a_)[:2, 3] - o.center_w[:2])))
        base = sk.kin.base(picker)[:2, 3]
        for tag, T, r in holes:
            p = T[:3, 3]
            if any(float(np.linalg.norm(p[:2] - f[:2])) < r for f in filled):
                continue
            dmin = min([float(np.linalg.norm(p[:2] - f[:2])) for f in filled] or [1.0])
            side = float(np.linalg.norm(p[:2] - o.center_w[:2]))
            # the arm carrying it in must not reach over an object already standing in a hole (twin: a hand passing
            # over a filled hole was held 34 mm off its command, the tube landed on the rack)
            over = any(_seg_dist(f[:2], base, p[:2]) < EACH_OVER for f in filled)
            free.append(((dmin < EACH_SPREAD), over, side, tag, T, r))
        free.sort(key=lambda x: (x[0], x[1], x[2]))
        free = [x[:1] + x[2:] for x in free]
        res = None
        pk = {**EACH_PLACE_KW, "grip_only": (receptacle,)}
        tkw = {"grasps": lambda a, lab=lab: uniform_section_grasps(sk, sk.world.obj(lab), a)}
        cands = []
        for spread_bad, _, tag, T, r in free[:EACH_HOLES]:
            p = T[:3, 3]
            fl = hole_floor(rec, p[:2], r)
            d = depth if depth is not None else (min(0.8 * L, float(p[2]) - fl) if fl is not None else 0.5 * L)
            tg = upright_targets(o, p, fl, d, n_yaw=8)
            if tg:
                cands.append((tag, d, tg))
        # a one-arm plan into any of the best holes first; a relay only when none has one
        for tag, d, tg in cands:
            plan = sk.plan_transfer(lab, tg, ignore=(receptacle,), place_kw=pk, **tkw)
            if isinstance(plan, SkillResult):
                sk.log(f"[insert_each] {lab} -> {tag}: no one-arm plan ({plan.note[:120]})")
                continue
            sk.log(f"[insert_each] {lab} -> {receptacle} {tag}: {plan[0].arm} ({plan[0].grasp.note}), {d*1000:.0f} mm in")
            from dojo.skills.library import _run_leg
            res = _run_leg(sk, lab, tg, 1, (receptacle,), pk, first=plan, tol=(0.008, 20.0, 0.03), transfer_kw=tkw)
            break
        if res is None and cands:
            tag, d, tg = cands[0]
            sk.log(f"[insert_each] {lab} -> {receptacle} {tag}: relay")
            res = place_oriented(sk, lab, tg, yaw_sym=1, tol_xy=0.008, tol_tilt=20.0, tol_z=0.03, ignore=(receptacle,),
                                 place_kw=pk, air_handover=False, transfer_kw=tkw)
        for a in ARMS:
            if sk.held[a] is not None:
                sk._record(sk.regrasp(a))
        sk.bank()
        w = sk.refresh()
        o = w.obj(lab)
        ins = [p for _, T, r in holes for p in [T[:3, 3]] if float(np.linalg.norm(o.pos[:2] - p[:2])) < r + 0.004]
        up = abs(float((o.R @ _unit(np.eye(3)[int(np.argmax(o.asset.extents))]))[2])) > math.cos(math.radians(30))
        if ins and up:
            st["done"].append(lab)
            st["filled"][lab] = ins[0].tolist()
        else:
            failed.append(lab)
            sk.log(f"[insert_each] {lab} not in a hole ({None if res is None else res.status}: "
                   f"{'' if res is None else res.note[:120]})")
        if res is not None and res.status in ("step_budget", "ended"):
            break
    if len(st["done"]) < len(objects):
        return {"ok": False, "stage": "insert_each", "note": f"in: {st['done']}, failed: {failed}"}
    return {"ok": True, "stage": "all inserted"}


# ============================================================================ screw_on

SCREW_PITCH = 0.003           # m per turn, first guess (a 24 mm thread's coarse pitch); re-measured after each sweep
SCREW_RATE = 0.15             # rad/step of the wrist while turning a nut (twin: 0.08 and 0.15 both went down 2.2 mm per
                              # 270 deg sweep)
SCREW_SWEEPS_DEG = (300.0, 270.0, 240.0, 180.0, 120.0)
SCREW_DEPTH = 0.0105          # m: a nut is turned until its bottom is this far below the bolt's top
SCREW_MAX_SWEEPS = 8


def n_fold(o) -> int:
    """Rotational symmetry about the vertical of a prism-like part from its footprint: a hexagon's across-corners /
    across-flats ratio is 1.155 (6), a square's 1.0 (4), else 2."""
    a, b = sorted(np.asarray(o.asset.extents[:2], float))
    r = b / max(a, 1e-6)
    if abs(r - 2 / math.sqrt(3)) < 0.03:
        return 6
    if abs(r - 1.0) < 0.03:
        return 4
    return 2


def _flat_yaw(o) -> float:
    """World yaw of the axis across a prism's narrower footprint extent (across the flats of a hex nut)."""
    k = int(np.argmin(o.asset.extents[:2]))
    ax = o.R[:, k]
    return math.atan2(float(ax[1]), float(ax[0]))


def plan_sweep(sk: Skills, a: str, nut, pitch: float, sign: float):
    """The longest wrist sweep that turns ``nut`` about its axis by ``sign`` (-1 = clockwise from above) while
    following the thread down: the pads across a pair of flats (any of the ``n_fold`` turns), TCP at the nut's mid
    height. Returns (T_grasp, [q per step], sweep rad) or None."""
    from dojo.skills.geometry import rot_z
    from dojo.skills.kinematics import down_R
    nf = n_fold(nut)
    yaw0 = _flat_yaw(nut)
    c = nut.center_w
    best = None
    for sweep in SCREW_SWEEPS_DEG:
        th_tot = math.radians(sweep)
        n = max(2, int(math.ceil(th_tot / SCREW_RATE)))
        for k in range(nf):
            psi = yaw0 + 2 * math.pi * k / nf
            T = T_from(np.array([c[0], c[1], c[2]]), down_R(psi))
            r = sk.kin.ik(a, T, sk.ex.q[a], n_random=2)
            if not r.ok:
                continue
            q, qs = r.q.copy(), []
            for j in range(1, n + 1):
                th = th_tot * j / n
                Tj = T.copy()
                Tj[:3, :3] = rot_z(sign * th) @ T[:3, :3]
                Tj[2, 3] -= pitch * th / (2 * math.pi)
                rr = sk.kin.solve(a, Tj, q)
                if not rr.ok or float(np.abs(rr.q - q).max()) > 0.5:
                    break
                q = rr.q
                qs.append(q)
            if len(qs) == n:
                cost = float(np.abs(r.q - sk.ex.q[a]).max())
                if best is None or cost < best[0]:
                    best = (cost, T, r.q, qs, th_tot)
        if best is not None:
            return best[1], best[2], best[3], best[4]
    return None


def screw_on(sk: Skills, pairs: list[list[str]], depth: float = SCREW_DEPTH, sign: float = -1.0) -> dict:
    """Screw each movable threaded part onto its fixed partner (``pairs`` = [[nut, bolt], ...]): the nut is put on the
    bolt's top, upright and on its axis (re-oriented / relayed as needed, ``place_oriented``), let go so it seats on
    the thread start, then turned down the thread in wrist sweeps (``plan_sweep``: the pads across its flats, the
    hand following the pitch down, ``SCREW_RATE``), let go and re-gripped between sweeps, until its bottom is
    ``depth`` below the bolt's top. ``sign`` -1 = clockwise from above (a right-hand thread tightens). The pitch is
    re-measured from the nut's descent after every sweep. Both grippers are opened after each pair (partial credit
    per pair)."""
    from dojo.skills.library import place_oriented
    from dojo.skills.kinematics import gap_to_grip
    done, failed = [], []
    st = sk.refresh().notes.setdefault("screw_on", {"done": []})
    for nut, bolt in pairs:
        if nut in st["done"]:
            continue
        w = sk.refresh()
        o, b = w.obj(nut), w.obj(bolt)
        # 1. on the bolt's top: upright, centred on its axis, bottom 1 mm over the top
        R = o.R
        from dojo.skills.targets import level_R
        Rl = level_R(R)
        c_b = b.center_w
        h = float(np.abs(Rl[2, :]) @ (o.asset.extents / 2))
        p_c = np.array([c_b[0], c_b[1], b.top_z + 0.001 + h])
        T = T_from(p_c - Rl @ o.asset.center, Rl)
        pk = {"pre_h": 0.03, "descend_rad": 0.03, "release_gap": 0.0, "settle": 2, "pre_settle": 2, "servo": True,
              "servo_ref": bolt, "seat_tol": 0.02, "probe": False}
        res = place_oriented(sk, nut, [T], yaw_sym=n_fold(o), tol_xy=0.004, tol_tilt=8.0, tol_z=0.02, ignore=(bolt,),
                             place_kw=pk, air_handover=False)
        for a in ARMS:
            if sk.held[a] is not None:
                sk._record(sk.regrasp(a))
        w = sk.refresh()
        o, b = w.obj(nut), w.obj(bolt)
        off = float(np.linalg.norm(o.center_w[:2] - b.center_w[:2]))
        if off > 0.004 or o.bottom_z > b.top_z + 0.004:
            failed.append(nut)
            sk.log(f"[screw] {nut} not on {bolt} ({off*1000:.1f} mm off, {res.status}: {res.note[:100]})")
            if res.status in ("step_budget", "ended"):
                break
            continue
        # 2. sweeps down the thread
        arm = min(ARMS, key=lambda a_: float(np.linalg.norm(sk.kin.base(a_)[:2, 3] - b.center_w[:2])))
        other = "right" if arm == "left" else "left"
        if sk.held[other] is None:
            try:
                sk.ex.run({other: sk.plan_home(other)}, name=f"clear {other}")
            except (BudgetExceeded, EpisodeEnded):
                pass
        pitch = SCREW_PITCH
        n_sw = 0
        g_open = gap_to_grip(min(G.MAX_GAP, float(np.min(o.asset.extents[:2])) + 0.022))
        while n_sw < SCREW_MAX_SWEEPS:
            w = sk.refresh()
            o, b = w.obj(nut), w.obj(bolt)
            down = b.top_z - o.bottom_z
            if down >= depth:
                break
            sw = plan_sweep(sk, arm, o, pitch, sign)
            if sw is None:
                sk.log(f"[screw] {nut}: no sweep for {arm}")
                break
            T, q_g, qs, th = sw
            T_hi = T.copy()
            T_hi[2, 3] += 0.008
            r_hi = sk.kin.ik(arm, T_hi, q_g, n_random=0)
            go = sk.motion.joint(arm, sk.ex.q[arm], r_hi.q if r_hi.ok else q_g, FAST, "screw_over", gripper=g_open,
                                 check=False)
            dn = sk.motion.line(arm, go.q_end, T, 0.05, "screw_down", gripper=g_open, check=False)
            pl = ArmPlan(arm, sk.ex.q[arm].copy(), sk.ex.g[arm]).grip(g_open, 2, "open").move(go, "over").move(dn, "down")
            pl.grip(CLOSED, 4, "close").hold(2, "squeeze")
            from dojo.skills.motion import Segment
            pl.move([Segment(np.vstack([dn.q_end if dn.segments else q_g] + qs), SCREW_RATE, "turn")], "turn")
            T_end = sk.kin.tcp(arm, qs[-1])
            T_up = T_end.copy()
            T_up[2, 3] += 0.006
            up = sk.motion.line(arm, qs[-1], T_up, MEDIUM, "screw_up", gripper=g_open, check=False)
            pl.grip(g_open, 2, "let_go").move(up, "up")
            z0 = o.pos[2]
            try:
                sk.ex.run({arm: pl}, name=f"screw {nut}")
            except BudgetExceeded as e:
                sk._record(SkillResult(False, "screw", "step_budget", 0, arm, note=str(e)))
                break
            except EpisodeEnded as e:
                sk._record(SkillResult(True, "screw", "ended", pl.steps, arm, note=str(e)))
                return {"ok": True, "stage": "ended"}
            n_sw += 1
            o2 = sk.refresh().obj(nut)
            dz = float(z0 - o2.pos[2])
            if dz > 0.0003:
                pitch = float(np.clip(dz * 2 * math.pi / th, 0.0005, 0.006))
            sk._record(SkillResult(True, "screw", "ok", pl.steps, arm,
                                   note=f"{nut}: sweep {n_sw} {math.degrees(th):.0f} deg, down {dz*1000:.1f} mm, "
                                        f"{(b.top_z - o2.bottom_z)*1000:.1f} mm below the top, pitch {pitch*1000:.2f}"))
        sk.bank()
        w = sk.refresh()
        o, b = w.obj(nut), w.obj(bolt)
        if b.top_z - o.bottom_z >= depth - 0.001:
            st["done"].append(nut)
            done.append(nut)
        else:
            failed.append(nut)
        if sk.ex.steps_left - sk.ex.reserve < 40:
            break
    if len(done) + len(st["done"]) - len(done) < len(pairs) and failed:
        return {"ok": False, "stage": "screw_on", "note": f"done {st['done']}, failed {failed}"}
    return {"ok": len(st["done"]) >= len(pairs), "stage": "screwed", "note": f"done {st['done']}"}


SKILLS = {"strike_in_order": strike_in_order, "insert_each": insert_each, "screw_on": screw_on}
