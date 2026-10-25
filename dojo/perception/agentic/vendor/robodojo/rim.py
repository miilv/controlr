"""rim.py <calib_head.json> <calib_wrist.json> <wrist cam> '<json>' : rim circle from head + wrist ellipse boxes.
json = {"h": [ul, ur, vt, vb], "w": [ul, ur, vt, vb]}  (ellipse extremes of the rim's outer edge)
Prints the triangulated rim centre, ray gap, and the rim diameter from each view at that height."""
import json, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from geom import *
dh, dw, cam, b = load(sys.argv[1]), load(sys.argv[2]), sys.argv[3], json.loads(sys.argv[4])
Kh, Th = calib(dh, 'cam_head'); Kw, Tw = calib(dw, cam)
ch = ((b['h'][0] + b['h'][1]) / 2, (b['h'][2] + b['h'][3]) / 2); cw = ((b['w'][0] + b['w'][1]) / 2, (b['w'][2] + b['w'][3]) / 2)
p, g = tri(Kh, Th, ch, Kw, Tw, cw); z = p[2]
dhm = np.linalg.norm(plane(Kh, Th, (b['h'][0], ch[1]), z) - plane(Kh, Th, (b['h'][1], ch[1]), z))
dwm = np.linalg.norm(plane(Kw, Tw, (b['w'][0], cw[1]), z) - plane(Kw, Tw, (b['w'][1], cw[1]), z))
print(json.dumps({"centre": p.round(4).tolist(), "ray_gap_mm": round(g * 1000, 1), "diam_head_mm": round(dhm * 1000, 1),
                  "diam_wrist_mm": round(dwm * 1000, 1), "px_head": ch, "px_wrist": cw}))
