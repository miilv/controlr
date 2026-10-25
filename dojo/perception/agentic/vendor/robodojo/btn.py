"""btn.py <calib.json> '<json boxes>' [z]: boxes = {"red0":[ul,ur,vt,vb],...} silhouette boxes of the button caps (red/blue
part) in cam_head; plane hit of the box centre at z (default 0.789 = the cap's widest circle, triangulated on practice L9)."""
import json, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from geom import *
d = load(sys.argv[1]); K, T = calib(d, 'cam_head'); bx = json.loads(sys.argv[2]); z = float(sys.argv[3]) if len(sys.argv) > 3 else 0.789
out = {}
for k, (ul, ur, vt, vb) in bx.items():
    p = plane(K, T, ((ul + ur) / 2, (vt + vb) / 2), z); out[k] = [round(float(p[0]), 4), round(float(p[1]), 4)]
print(json.dumps(out))
