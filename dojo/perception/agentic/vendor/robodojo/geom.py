"""Mac-side geometry for the RGB-only runs: wrist-camera pose for a planned TCP pose, projection, plane hits.
Uses only camera calibration (K, T_world_cam) recorded by `obs calib_out=true` and the arm's own flange pose."""
import json, math, sys
import numpy as np

GRASP_OFF = 0.1501
TOP_DOWN = np.array([0.5, -0.5, 0.5, 0.5])


def qmul(a, b):
    w0, x0, y0, z0 = a; w1, x1, y1, z1 = b
    return np.array([w0*w1-x0*x1-y0*y1-z0*z1, w0*x1+x0*w1+y0*z1-z0*y1, w0*y1-x0*z1+y0*w1+z0*x1, w0*z1+x0*y1-y0*x1+z0*w1])


def qmat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def axis_q(axis, rad):
    q = np.zeros(4); q[0] = math.cos(rad/2); q[axis+1] = math.sin(rad/2); return q


def tool_quat(yaw=0.0, pitch=0.0, roll=0.0):
    r = axis_q(0, math.radians(pitch)); r = qmul(axis_q(1, math.radians(roll)), r); r = qmul(axis_q(2, math.radians(yaw)), r)
    return qmul(r, TOP_DOWN)


def T_of(pos, quat):
    T = np.eye(4); T[:3, :3] = qmat(quat); T[:3, 3] = pos; return T


def flange_T_from_tcp(tcp, quat):
    R = qmat(quat); return T_of(np.asarray(tcp, float) - GRASP_OFF * R[:, 0], quat)


def load(path):
    d = json.load(open(path))
    return d


def cam_in_flange(d, arm):
    c = d["calib"][f"cam_{arm}_wrist"]
    fl = d["eef"][arm]["flange"]
    return np.linalg.inv(T_of(fl["pos"], fl["quat"])) @ np.asarray(c["T_world_cam"], float), np.asarray(c["K"], float)


def project(K, Twc, pts):
    pts = np.asarray(pts, float).reshape(-1, 3)
    Pc = (np.linalg.inv(Twc) @ np.c_[pts, np.ones(len(pts))].T)[:3].T
    return [None if p[2] <= 0 else (K[0, 0]*p[0]/p[2]+K[0, 2], K[1, 1]*p[1]/p[2]+K[1, 2]) for p in Pc]


def ray(K, Twc, px):
    d = Twc[:3, :3] @ np.array([(px[0]-K[0, 2])/K[0, 0], (px[1]-K[1, 2])/K[1, 1], 1.0])
    return Twc[:3, 3], d/np.linalg.norm(d)


def plane(K, Twc, px, z):
    o, d = ray(K, Twc, px); s = (z-o[2])/d[2]; return o+s*d


def tri(K1, T1, px1, K2, T2, px2):
    o1, d1 = ray(K1, T1, px1); o2, d2 = ray(K2, T2, px2)
    w0 = o1-o2; b = d1@d2; den = 1-b*b
    s1 = (b*(d2@w0)-(d1@w0))/den; s2 = ((d2@w0)-b*(d1@w0))/den
    p1, p2 = o1+s1*d1, o2+s2*d2
    return (p1+p2)/2, float(np.linalg.norm(p1-p2))


def calib(d, cam):
    c = d["calib"][cam]; return np.asarray(c["K"], float), np.asarray(c["T_world_cam"], float)
