"""sil.py <calib.json> <cam> '<json {name: [u_left, u_right, v_top, v_bottom]}>': silhouette box -> centre (plane hit at
mid-height, iterated with the size), edge from the horizontal and vertical extents (plane hits at mid-height)."""
import json, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from geom import *
d = load(sys.argv[1]); K, T = calib(d, sys.argv[2]); B = json.loads(sys.argv[3])
for n, (ul, ur, vt, vb) in B.items():
    uc, vc = (ul + ur) / 2, (vt + vb) / 2
    e = 0.035
    for _ in range(20):
        zm = 0.765 + e / 2
        wl, wr = plane(K, T, (ul, vc), zm), plane(K, T, (ur, vc), zm)
        e = float(np.linalg.norm((wr - wl)[:2]))
    c = plane(K, T, (uc, vc), 0.765 + e / 2)
    print(n, json.dumps({"c": c[:2].round(4).tolist(), "edge": round(e, 4), "top_z": round(0.765 + e, 4)}))
