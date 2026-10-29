"""Hang the headphones on their stand, close the laptop, then stand it in its vertical rack (stream N;
store_laptop_and_headphones' instruction: "Hang the headphones on the headphone stand, close the laptop, then place it
into the vertical laptop stand.").

Three stages, each credited on its own by the checker (paired 20 / 80): ``hang_on_hooks`` for the headset (its
``headband_bridge`` onto the stand's ``support_cradle``), ``holders.swing_closed`` for the laptop's hinge, then the closed
laptop re-oriented by ``place_oriented``: its body y axis up (the checker's ``is_axis_up(laptop, [0, 1, 0], 10)``), its
``bottom`` point down in the rack, its width along the rack's long side.

Route "store_laptop", off by default (``RuntimeConfig.new_routes``). Route-only (``library.ROUTE_ONLY``).
"""

from __future__ import annotations

import re

import numpy as np

CLOSE = re.compile(r"\bclose\s+the\s+([a-z ]+?)\s*(?:,|\bthen\b|\band\b|\.|$)")
INTO = re.compile(r"\bplace\s+it\s+(?:into|in|on)\s+the\s+([a-z ]+?)\s*(?:,|\bthen\b|\.|$)")
RACK_DROP = 0.006       # m: the laptop's bottom edge this far over the rack's floor (its bbox bottom) at the release
YAWS_DEG = (0.0, 180.0, 8.0, -8.0, 188.0, 172.0)
EDGE_PICK_TILT = 95.0   # deg: the shut laptop's pick approach may be horizontal (an edge pinch across the slab)
EDGE_OFF = 0.010        # m: an edge pinch this close to the middle of its edge (G3: 51 mm off-centre pinches dropped it)
EDGE_SQUEEZE = 4        # still steps after the close before the lift
LID_REPUSH_HEIGHTS = ((0.6, 0.9, 0.75), (0.9, 0.6, 0.75))   # swing_closed's lid heights for the extra pushes


def _words(s: str) -> set:
    return {w for w in re.split(r"[^a-z]+", s.lower()) if w}


def route(text: str, world, routes) -> tuple | None:
    """("store_laptop", {...}, why) when the route is on and the words ask to hang something, close something hinged and
    place it into a named rack."""
    if "store_laptop" not in set(routes or ()) or not re.search(r"\bhang\b", text):
        return None
    mc, mi = CLOSE.search(text), INTO.search(text)
    if not (mc and mi):
        return None
    objs = [o for o in world.task_objects() if o.category != "camera_stand"]
    hinged = [o for o in objs if o.obj_type == "Articulation" and (_words(o.category) & _words(mc.group(1)))
              and any((j or {}).get("type") == "revolute" for j in (o.asset.joints or {}).values())]
    if len(hinged) != 1:
        return None
    lap = hinged[0]
    hinge = next((t for t, v in (lap.asset.functional or {}).items() if (v or {}).get("parent_joint")), None)
    want = _words(mi.group(1))

    def under(o) -> bool:                       # what the laptop rests on now (its stand) is not where it goes
        return (float(np.linalg.norm(o.center_w[:2] - lap.center_w[:2])) < 0.12 and o.top_z <= lap.center_w[2])
    racks = sorted((o for o in objs if o.label != lap.label and not under(o) and _words(o.category) & want),
                   key=lambda o: -len(_words(o.category) & want))
    if not racks:
        return None
    hang = None
    for rk in objs:
        tags = [t for t, v in (rk.asset.functional or {}).items()
                if re.search(r"hook|support_point|hang|cradle", t.lower()) and (v or {}).get("frame")]
        if not tags:
            continue
        movers = [o for o in objs if o.obj_type == "Rigid" and not o.clutter and o.label != rk.label
                  and len(o.asset.active_functional or {}) == 1]
        if movers:
            hang = {"objects": [o.label for o in movers], "rack": rk.label, "hook_tag": tags[0],
                    "handle_tag": next(iter(movers[0].asset.active_functional))}
            break
    if hang is None:
        return None
    return ("store_laptop", {"hang": hang, "laptop": lap.label, "hinge_tag": hinge, "rack": racks[0].label},
            f"hang {hang['objects']} on {hang['rack']}, close {lap.label}, stand it in {racks[0].label}")


def upright_targets(sk, laptop: str, rack: str) -> list:
    """Closed-laptop poses (asset frame) in the rack, as the checker reads them (``artgeom``): the root frame's y axis
    up (``is_axis_up(laptop, [0, 1, 0])`` reads the root prim), the width (root x) along the rack's long side, the
    ``bottom`` point, placed on its base link (``checker_point``: the body's front edge), RACK_DROP over the rack's
    bottom at the rack's centre (``is_A_functional_point_in_B_bbox``)."""
    from dojo.skills.artgeom import checker_point, root_axis
    from dojo.skills.geometry import T_from, rot_z
    w = sk.world
    o, rk = w.obj(laptop), w.obj(rack)
    bt = checker_point(o, "bottom", active=True)
    p_b = bt[0][:3, 3] if bt else np.asarray(o.asset.center, float)
    ext = np.asarray(rk.asset.extents, float)
    long_ax = rk.R[:, 0] if ext[0] >= ext[1] else rk.R[:, 1]
    yaw0 = float(np.arctan2(long_ax[1], long_ax[0]))
    y_ = root_axis(o, [0.0, 1.0, 0.0])
    x_ = root_axis(o, [1.0, 0.0, 0.0])
    # the asset rotation that takes the root's y to world z and its x to world x (then turned to the rack's long side)
    A = np.stack([x_, y_, np.cross(x_, y_)], axis=1)               # root axes in the asset frame (columns)
    W = np.stack([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]], axis=1)  # where they go in the world, before the yaw
    up = W @ A.T
    c = rk.center_w
    target = np.array([c[0], c[1], float(rk.bottom_z) + RACK_DROP])
    out = []
    for dy in YAWS_DEG:
        R = rot_z(yaw0 + np.radians(dy)) @ up
        out.append(T_from(target - R @ p_b, R))
    return out


def checker_ok(o, T_obj, rk, up_tol_deg: float = 10.0) -> bool:
    """The checker's laptop conditions at pose ``T_obj`` (asset frame) but the hinge: root y within ``up_tol_deg`` of
    up, and the ``bottom`` point (on its base link) inside the rack's box."""
    from dojo.skills.artgeom import checker_point, root_axis
    y_w = T_obj[:3, :3] @ root_axis(o, [0.0, 1.0, 0.0])
    if np.degrees(np.arccos(np.clip(float(y_w[2]) / float(np.linalg.norm(y_w)), -1.0, 1.0))) > up_tol_deg:
        return False
    bt = checker_point(o, "bottom", active=True)
    if not bt:
        return False
    p = (T_obj @ bt[0])[:3, 3]
    q = rk.R.T @ (p - rk.pos)
    return bool(np.all(q >= rk.asset.bbox_min - 1e-6) and np.all(q <= rk.asset.bbox_max + 1e-6))


def mid_edge(o):
    """``plan_transfer``'s ``pair_ok``: the TCP within EDGE_OFF of the middle of one of the slab's horizontal edges (in
    the object frame, along the edge it pinches)."""
    c = np.asarray(o.asset.center, float)

    def ok(g, T_tcp_obj, T_obj, T_tcp) -> bool:
        p = np.linalg.inv(np.asarray(T_tcp_obj, float))[:3, 3] - c          # the TCP in the object frame, from its centre
        return min(abs(float(p[0])), abs(float(p[1]))) <= EDGE_OFF
    return ok


def store_laptop(sk, hang: dict, laptop: str, hinge_tag: str | None, rack: str) -> dict:
    """Hang, close, store: each stage banked (grippers open) before the next; the later stages run even when the hang
    failed (the checker credits them on their own)."""
    from dojo.skills.holders import swing_closed
    from dojo.skills.library import hang_on_hooks, place_oriented
    from dojo.skills.kinematics import ARMS
    notes = {}
    r1 = hang_on_hooks(sk, **hang)
    notes["hang"] = r1.get("note") or r1.get("stage")
    sk.bank()
    r2 = swing_closed(sk, laptop, hinge_tag, ratio=0.92)
    for hts in LID_REPUSH_HEIGHTS:
        # the measured lid short of shut (G3: 8 of 20 rows stopped at 0.30-0.89 after the first two pushes; the checker
        # wants >= 0.85): push again, at other heights of the lid
        if r2.ok or sk.ex.steps_left < 120:
            break
        r2 = swing_closed(sk, laptop, hinge_tag, ratio=0.92, heights=hts)
    notes["close"] = f"{r2.status} {r2.note[:60]}"
    sk.bank()
    sk.refresh()
    if not r2.ok:
        return {"ok": False, "stage": "close", "note": str(notes)}
    # the shut laptop as it is now (``Skills.artic_geom`` "measured": the lid turned onto the body, a 9 mm slab; the
    # authored asset is an OPEN laptop), picked by an edge (a side pinch across the slab: top-down pinches would span
    # 126-180 mm), turned to stand in the rack
    saved = getattr(sk, "artic_geom", "authored")
    sk.artic_geom = "measured"
    try:
        sk.refresh()
        tg = upright_targets(sk, laptop, rack)
        r3 = place_oriented(sk, laptop, tg, yaw_sym=1, tol_xy=0.03, tol_tilt=10.0, tol_z=0.04, ignore=(rack,),
                            place_kw={"pre_h": 0.06, "release_gap": 0.003}, max_spots=2,
                            transfer_kw={"max_pick_tilt": EDGE_PICK_TILT, "pair_ok": mid_edge(sk.world.obj(laptop)),
                                         "pick_kw": {"squeeze": EDGE_SQUEEZE}})
    finally:
        sk.artic_geom = saved
    notes["store"] = f"{r3.status} {r3.note[:80]}"
    for a in ARMS:
        if sk.held[a] is not None:
            sk._record(sk.regrasp(a))
    sk.bank()
    return {"ok": bool(r1.get("ok") and r3.ok), "stage": "store_laptop", "note": str(notes)}


ROUTE_SKILLS = {"store_laptop": store_laptop}

__all__ = ["route", "upright_targets", "checker_ok", "store_laptop", "ROUTE_SKILLS"]
