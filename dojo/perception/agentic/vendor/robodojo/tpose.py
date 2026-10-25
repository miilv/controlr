"""tpose.py: T-block pose from top-face corners. Corners P1..P8 (P1,P2 stem end; P3 stem-right/bar junction; P4,P5 bar
right end inner/outer; P6,P7 bar left end outer/inner; P8 stem-left/bar junction), in the wrist view (dense) and some in
the head view (for the top height by triangulation)."""
import json, math, sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from geom import *


def solve(calib_head, calib_wrist, wcam, wpx, hpx, z_override=None):
    dh, dw = load(calib_head), load(calib_wrist)
    Kh, Th = calib(dh, 'cam_head'); Kw, Tw = calib(dw, wcam)
    zs = []
    for k, ph in hpx.items():
        if k in wpx:
            p, gap = tri(Kh, Th, ph, Kw, Tw, wpx[k]); zs.append(p[2])
            print(f'  tri {k}: {p.round(4)} gap {gap*1000:.1f} mm')
    zt = float(np.median(zs)) if z_override is None else z_override
    W = {k: plane(Kw, Tw, v, zt) for k, v in wpx.items()}
    se = (W['P1'] + W['P2']) / 2                 # stem end centre
    bo = (W['P5'] + W['P6']) / 2                 # bar outer edge centre
    ax = bo - se; L = np.linalg.norm(ax[:2]); u = ax[:2] / L
    yaw_axis = math.degrees(math.atan2(u[1], u[0]))
    # bar outer edge direction (P6 -> P5) should be perpendicular to the stem axis
    e = W['P5'] - W['P6']; yaw_bar = math.degrees(math.atan2(e[1], e[0]))
    # heading perpendicular from bar edge: rotate by +/-90 to align with the axis
    cand = [yaw_bar + 90, yaw_bar - 90]
    yb = min(cand, key=lambda c: abs((c - yaw_axis + 180) % 360 - 180))
    yb = (yb + 180) % 360 - 180
    stem_w = np.linalg.norm((W['P1'] - W['P2'])[:2]); bar_len = np.linalg.norm(e[:2])
    yaw = math.degrees(math.atan2(math.sin(math.radians(yaw_axis)) + math.sin(math.radians(yb)),
                                  math.cos(math.radians(yaw_axis)) + math.cos(math.radians(yb))))
    out = {'top_z': round(zt, 4), 'tri_z': [round(z, 4) for z in zs], 'stem_end': se[:2].round(4).tolist(),
           'bar_outer': bo[:2].round(4).tolist(), 'length_m': round(float(L), 4), 'stem_w_m': round(float(stem_w), 4),
           'bar_len_m': round(float(bar_len), 4), 'yaw_axis': round(yaw_axis, 2), 'yaw_bar': round(yb, 2),
           'yaw': round(yaw, 2), 'corners': {k: v[:2].round(4).tolist() for k, v in W.items()}}
    return out, W, se, u


def grasp_point(se, u, L, frac):
    return (se[:2] + u * frac).round(4).tolist()


if __name__ == '__main__':
    pass
