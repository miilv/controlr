"""cubez.py <calib.json> <cam> <top_z> '<json [[u,v] x4 top-face corners in order]>' : plane hits at the given top z ->
top-face centre, mean edge, yaw mod 90 (deg, circular mean of the 4 side directions). next3 agent (geom.py math)."""
import json, math, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])  # [tools/perceiver copy, 25 Oct 2026] sys.path: geom.py from this file's directory (was the scratchpad rgbpp/tools)
from geom import *
d = load(sys.argv[1]); K, T = calib(d, sys.argv[2]); z = float(sys.argv[3]); C = json.loads(sys.argv[4])
W = [plane(K, T, c, z) for c in C]
L = [float(np.linalg.norm((W[i] - W[(i + 1) % 4])[:2])) for i in range(4)]
ctr = np.mean(W, axis=0)
angs = [math.degrees(math.atan2(*(W[(i + 1) % 4] - W[i])[1::-1])) % 90 for i in range(4)]
yaw = math.degrees(math.atan2(np.mean([math.sin(math.radians(4 * a)) for a in angs]), np.mean([math.cos(math.radians(4 * a)) for a in angs]))) / 4
print(json.dumps({"c": ctr[:2].round(4).tolist(), "top_z": z, "edges_mm": [round(1000 * l, 1) for l in L],
                  "yaw_mod90": round(yaw, 1), "side_angles": [round(a, 1) for a in angs]}))
