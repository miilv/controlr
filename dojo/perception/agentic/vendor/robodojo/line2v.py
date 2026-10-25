"""line2v.py <calibA> <camA> '[[u,v],[u,v]]' <calibB> <camB> '[[u,v],[u,v]]' : the 3D line seen as an image line in two
views (two points on it in each view, not necessarily the same points): intersect the two back-projected planes.
Prints a point on the line (at x = the mean of the view-A rays' x at their closest approach) and its direction.
next3 agent; geom.py math only."""
import json, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])  # [tools/perceiver copy, 25 Oct 2026] sys.path: geom.py from this file's directory (was the scratchpad rgbpp/tools)
from geom import *


def pl(calib, cam, pts):
    d = load(calib); K, T = calib_(d, cam)
    o, d1 = ray(K, T, pts[0]); _, d2 = ray(K, T, pts[1])
    n = np.cross(d1, d2); n /= np.linalg.norm(n)
    return o, n


def calib_(d, cam):
    return calib(d, cam)


oA, nA = pl(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]))
oB, nB = pl(sys.argv[4], sys.argv[5], json.loads(sys.argv[6]))
dirv = np.cross(nA, nB); dirv /= np.linalg.norm(dirv)
# point: solve nA.p = nA.oA, nB.p = nB.oB, dirv.p = dirv.x0 with x0 = 0
A = np.vstack([nA, nB, dirv]); b = np.array([nA @ oA, nB @ oB, 0.0])
p = np.linalg.solve(A, b)
print(json.dumps({"point": p.round(4).tolist(), "dir": dirv.round(4).tolist()}))
