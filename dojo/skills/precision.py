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
                # fingertips >= 2.5 mm over the table: raise the TCP (the pads still cover a handle a few mm thick)
                g_open = G.gap_to_grip(min(G.MAX_GAP, width + 2 * G.CLEAR + 0.006))
                low = float(sk.kin.finger_points(sk.kin.flange_from_tcp(T), g_open)[:, 2].min())
                need = TABLE_Z + 0.0025 - low
                if need > 0:
                    if need > 0.4 * width + 0.002:
                        continue
                    T[2, 3] += need
                out.append(G.Grasp("handle", T, width, pre_dist=0.06, score=0.1 * rank + abs(tilt) / 100.0,
                                   note=f"handle {d*1000:.0f} mm from the working end, tilt {tilt:+.0f}"))
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
    for a in arms:
        cands = handle_grasps(sk, to, a, work_loc)
        picks = sk.plan_pick(tool, a, grasps=cands, lift=0.06, keep=6, max_eval=60, return_all=True)
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


SKILLS = {"strike_in_order": strike_in_order}
