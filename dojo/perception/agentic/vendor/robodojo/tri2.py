"""tri2.py <calibA> <camA> <calibB> <camB> '{name: [[uA,vA],[uB,vB]], ...}' : two-view triangulation per named point
(geom.tri: midpoint of the closest approach), prints xyz and the ray gap in mm. next3 agent."""
import json, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])  # [tools/perceiver copy, 25 Oct 2026] sys.path: geom.py from this file's directory (was the scratchpad rgbpp/tools)
from geom import *
dA = load(sys.argv[1]); KA, TA = calib(dA, sys.argv[2]); dB = load(sys.argv[3]); KB, TB = calib(dB, sys.argv[4])
for n, (a, b) in json.loads(sys.argv[5]).items():
    p, g = tri(KA, TA, a, KB, TB, b)
    print(n, [round(float(v), 4) for v in p], round(g * 1000, 1))
