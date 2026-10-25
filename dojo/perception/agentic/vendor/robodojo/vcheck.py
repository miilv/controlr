"""vcheck.py <calib.json> <arm> '<json points [[x,y,z],...]>' x,y,z,yaw ... : project world points into the wrist camera
for candidate TCP poses (top-down tool at yaw), using rgbpp/tools/geom.py (calibration + flange pose only)."""
import json, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from geom import *
d = load(sys.argv[1]); arm = sys.argv[2]; P = json.loads(sys.argv[3])
Tfc, Kw = cam_in_flange(d, arm)
for c in sys.argv[4:]:
    x, y, z, yaw = [float(v) for v in c.split(',')]
    Twc = flange_T_from_tcp([x, y, z], tool_quat(yaw)) @ Tfc
    px = project(Kw, Twc, P)
    print(c, [None if p is None else (round(p[0]), round(p[1])) for p in px])
