"""tp.py <calib_head.json> <calib_wrist.json|-> <wrist cam|-> '<json picks>' [z_top]
picks = {"w": {"P1": [u,v], "P2":.., "P5":.., "P6":..}, "h": {...}}  (P1,P2 stem-end top corners; P5,P6 bar outer
top corners). Wrist picks present -> wrist plane hits at z (median head+wrist triangulation of shared labels, else
z_top); else head plane hits at z_top (default 0.780). Prints grasp point g (axis midpoint), yaw (heading stem->bar,
mean of stem axis and bar-edge normal), sizes for sanity (T: length ~0.079, stem 0.0197, bar 0.060)."""
import json, math, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from geom import *

ch, cw, wcam, picks = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
z0 = float(sys.argv[5]) if len(sys.argv) > 5 else 0.780
dh = load(ch); Kh, Th = calib(dh, 'cam_head')
w, h = picks.get('w') or {}, picks.get('h') or {}
zs, gaps = [], []
if w:
    dw = load(cw); Kw, Tw = calib(dw, wcam)
    for k in w:
        if k in h:
            p, gap = tri(Kh, Th, h[k], Kw, Tw, w[k]); zs.append(round(float(p[2]), 4)); gaps.append(round(gap * 1000, 1))
    zt = float(np.median(zs)) if zs else z0
    W = {k: plane(Kw, Tw, v, zt) for k, v in w.items()}; src = 'wrist'
else:
    zt = z0
    W = {k: plane(Kh, Th, v, zt) for k, v in h.items()}; src = 'head'
se = (W['P1'] + W['P2']) / 2; bo = (W['P5'] + W['P6']) / 2
ax = bo - se; L = float(np.linalg.norm(ax[:2])); ya = math.degrees(math.atan2(ax[1], ax[0]))
e = W['P5'] - W['P6']; yb0 = math.degrees(math.atan2(e[1], e[0]))
yb = min([yb0 + 90, yb0 - 90], key=lambda c: abs((c - ya + 180) % 360 - 180)); yb = (yb + 180) % 360 - 180
yaw = math.degrees(math.atan2(math.sin(math.radians(ya)) + math.sin(math.radians(yb)), math.cos(math.radians(ya)) + math.cos(math.radians(yb))))
g = (se + bo) / 2
print(json.dumps({'src': src, 'top_z': round(zt, 4), 'tri_z': zs, 'ray_gap_mm': gaps, 'g': g[:2].round(4).tolist(),
                  'yaw': round(yaw, 2), 'yaw_axis': round(ya, 2), 'yaw_bar': round(yb, 2), 'len': round(L, 4),
                  'stem_w': round(float(np.linalg.norm((W['P1'] - W['P2'])[:2])), 4), 'bar_len': round(float(np.linalg.norm(e[:2])), 4)}))
