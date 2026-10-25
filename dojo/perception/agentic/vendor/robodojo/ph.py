"""ph.py <calib> <cam> '{name: [u, v, z], ...}' : plane hits (pixel ray at height z). next3 agent."""
import json, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])  # [tools/perceiver copy, 25 Oct 2026] sys.path: geom.py from this file's directory (was the scratchpad rgbpp/tools)
from geom import *
d = load(sys.argv[1]); K, T = calib(d, sys.argv[2])
for n, (u, v, z) in json.loads(sys.argv[3]).items():
    p = plane(K, T, (u, v), z)
    print(n, [round(float(x), 4) for x in p[:2]], z)
