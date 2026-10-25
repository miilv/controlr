"""cube.py <calib.json> <cam> '<json: [[u,v] x4 top-face corners in order]>' [cam2 calib2 '<json corners2 same order>']
Single view: plane hits at z, edge = mean side length, iterate z = 0.765 + edge (cube). Two views: triangulate each
corner, z = median. Prints top-face centre, top z, edge, yaw (mod 90, from the side directions)."""
import json, math, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from geom import *
d = load(sys.argv[1]); K, T = calib(d, sys.argv[2]); C = json.loads(sys.argv[3])
out = {}
if len(sys.argv) > 6:
    d2 = load(sys.argv[5]); K2, T2 = calib(d2, sys.argv[4]); C2 = json.loads(sys.argv[6])
    P = []; gaps = []
    for a, b in zip(C, C2):
        if a is None or b is None: continue
        p, g = tri(K, T, a, K2, T2, b); P.append(p); gaps.append(round(g * 1000, 1))
    z = float(np.median([p[2] for p in P])); out['tri_z'] = [round(float(p[2]), 4) for p in P]; out['gap_mm'] = gaps
else:
    z = 0.80
    for _ in range(30):
        W = [plane(K, T, c, z) for c in C]
        L = np.mean([np.linalg.norm((W[i] - W[(i + 1) % 4])[:2]) for i in range(4)])
        z = 0.765 + L
W = [plane(K, T, c, z) for c in C]
L = float(np.mean([np.linalg.norm((W[i] - W[(i + 1) % 4])[:2]) for i in range(4)]))
ctr = np.mean(W, axis=0)
angs = [math.degrees(math.atan2(*(W[(i + 1) % 4] - W[i])[1::-1])) % 90 for i in range(4)]
# circular mean mod 90
yaw = math.degrees(math.atan2(np.mean([math.sin(math.radians(4 * a)) for a in angs]), np.mean([math.cos(math.radians(4 * a)) for a in angs]))) / 4
out.update(c=ctr[:2].round(4).tolist(), top_z=round(z, 4), edge=round(L, 4), yaw=round(yaw, 1), side_angles=[round(a, 1) for a in angs])
print(json.dumps(out))
