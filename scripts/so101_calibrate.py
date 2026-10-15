"""Wrist-camera calibration for the SO-101 rig: intrinsics + hand-eye (camera pose on the TCP).

Needs OpenCV with aruco (not a controlr dependency):

    uv run --with opencv-contrib-python python scripts/so101_calibrate.py board board.png
        # print board.png at 100 % on A4, measure one square, pass it as --square-mm
    uv run --with opencv-contrib-python python scripts/so101_calibrate.py intrinsics --square-mm 30
        # NO arm motion: hold the printed board in front of the camera at different angles
        # and distances; a frame is taken every --every-s seconds (needs the server running)
    uv run --with opencv-contrib-python python scripts/so101_calibrate.py handeye --square-mm 30 \
        --board-x 0.26 [--intrinsics out.json]
        # board flat on the table, its centre about --board-x m in front of the base: the ARM
        # MOVES through ~15 poses above it (slowly; a human at the power switch)

Writes ``~/.config/controlr/so101_wrist.json`` (``robot.params.camera_calib``) with ``K``,
``dist``, ``width``, ``height``, ``T_tcp_cam`` (4x4, OpenCV camera frame in the TCP frame)
and the check numbers: the board's position in the base frame from every view (spread =
hand-eye + kinematics error; its z should be ~0, the table).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_OUT = Path("~/.config/controlr/so101_wrist.json").expanduser()


def _cv():
    try:
        import cv2
    except ImportError as e:
        raise SystemExit("needs OpenCV with aruco: uv run --with opencv-contrib-python python "
                         "scripts/so101_calibrate.py ...") from e
    return cv2


def make_board(cv2, square_mm: float, cols: int = 5, rows: int = 7):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    return cv2.aruco.CharucoBoard((cols, rows), square_mm / 1000.0, 0.75 * square_mm / 1000.0, d)


def detect(cv2, board, rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    if ids is None or len(ids) < 8:
        return None
    obj, img = board.matchImagePoints(corners, ids)
    return obj.reshape(-1, 3), img.reshape(-1, 2)


def connect(port: int):
    from controlr.robot.so101.client import So101Robot
    return So101Robot({"port": port, "reset_duration_s": 5.0, "view_note": False}, tcp_speed_m_s=0.03)


def calibrate_intrinsics(cv2, samples, size):
    objs = [s[0].astype(np.float32) for s in samples]
    imgs = [s[1].astype(np.float32) for s in samples]
    rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, size, None, None)
    return float(rms), K, dist.reshape(-1)


def cmd_board(a) -> None:
    cv2 = _cv()
    board = make_board(cv2, a.square_mm)
    px_per_mm = 300 / 25.4
    img = board.generateImage((int(5 * a.square_mm * px_per_mm), int(7 * a.square_mm * px_per_mm)),
                              marginSize=int(5 * px_per_mm))
    cv2.imwrite(a.out, img)
    print(f"{a.out}: 5x7 ChArUco, {a.square_mm} mm squares at 300 dpi — print at 100 %, measure a square")


def cmd_intrinsics(a) -> None:
    cv2 = _cv()
    board = make_board(cv2, a.square_mm)
    robot = connect(a.port)
    samples, size = [], None
    print(f"hold the board in front of the camera; {a.n} good views, one every {a.every_s} s (Ctrl-C ends)")
    try:
        while len(samples) < a.n:
            obs = robot.observe()
            rgb = obs.images["wrist"]
            size = (rgb.shape[1], rgb.shape[0])
            det = detect(cv2, board, rgb)
            print(f"  view {len(samples) + 1}: {'ok, ' + str(len(det[0])) + ' corners' if det else 'board not found'}")
            if det:
                samples.append(det)
            time.sleep(a.every_s)
    except KeyboardInterrupt:
        pass
    finally:
        robot.close()
    if len(samples) < 6:
        raise SystemExit("fewer than 6 good views")
    rms, K, dist = calibrate_intrinsics(cv2, samples, size)
    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = json.loads(out.read_text()) if out.exists() else {}
    rec.update(K=K.tolist(), dist=dist.tolist(), width=size[0], height=size[1], intrinsics_rms_px=rms,
               intrinsics_views=len(samples))
    out.write_text(json.dumps(rec, indent=1) + "\n")
    hfov = 2 * math.degrees(math.atan(size[0] / 2 / K[0, 0]))
    print(f"rms {rms:.2f} px, fx {K[0, 0]:.0f} fy {K[1, 1]:.0f} (hfov {hfov:.0f} deg) -> {out}")


def calib_poses(board_x: float, n: int = 15, width: int = 1280, height: int = 720, hfov_deg: float = 80.0):
    """TCP poses (position, pitch deg, roll deg) from which the DEFAULT camera model sees the
    board centre (on the table at ``board_x``, y 0) well inside the image, reachable from the
    home pose by a straight move, picked greedily for rotation diversity (hand-eye needs it)."""
    from controlr.robot.so101 import kinematics as K
    from controlr.robot.so101.client import _intrinsics

    Kc = _intrinsics(width, height, hfov_deg)
    target = np.array([board_x, 0.0, 0.0, 1.0])
    R0 = K.tool_rotation(math.radians(60), 0.0, math.radians(90))
    q_home, _, _ = K.ik([0.24, 0.0, 0.10], R0, np.radians([0, -30, 60, 60, -90]), iters=300)
    cands = []
    for dx in (-0.08, -0.05, -0.02, 0.01):
        for dy in (-0.05, 0.0, 0.05):
            for z in (0.07, 0.10, 0.13):
                for pitch in (50, 60, 70, 80):
                    for roll in (65, 90, 115):
                        p = np.array([board_x + dx, dy, z])
                        az = math.atan2(p[1] - K.PAN_AXIS_XY[1], p[0] - K.PAN_AXIS_XY[0])
                        R = K.tool_rotation(math.radians(pitch), az, math.radians(roll))
                        q, pe, ae = K.ik(p, R, q_home, iters=150)
                        if pe > 1e-3 or ae > 0.02:
                            continue
                        c = K.camera_T_cam_base(q, K.DEFAULT_T_TCP_CAM) @ target
                        if c[2] < 0.08:
                            continue
                        u, v = (Kc @ c[:3])[:2] / c[2]
                        if 0.2 * width < u < 0.8 * width and 0.2 * height < v < 0.8 * height:
                            cands.append((p, pitch, roll, R))
    if not cands:
        raise SystemExit("no calibration pose sees the board: move it (--board-x)")
    from controlr.robot.kinematics import rotation_angle
    chosen = [min(cands, key=lambda c: abs(c[1] - 60) + abs(c[2] - 90) + np.linalg.norm(c[0][:2] - [0.24, 0]))]
    while len(chosen) < min(n, len(cands)):
        best = max(cands, key=lambda c: min(rotation_angle(c[3] @ d[3].T) + 2 * np.linalg.norm(c[0] - d[0])
                                            for d in chosen))
        chosen.append(best)
    return [(p, pitch, roll) for p, pitch, roll, _ in chosen]


def cmd_handeye(a) -> None:
    cv2 = _cv()
    from controlr.robot.kinematics import matrix_to_rotvec
    from controlr.robot.so101 import kinematics as K
    from controlr.types import Action, ActionMode

    board = make_board(cv2, a.square_mm)
    out = Path(a.out).expanduser()
    intr = json.loads(Path(a.intrinsics or out).expanduser().read_text()) if (a.intrinsics or out.exists()) else {}
    robot = connect(a.port)
    shots = Path(a.shots)
    shots.mkdir(parents=True, exist_ok=True)
    samples = []
    input(f"board flat on the table, centre ~{a.board_x * 1000:.0f} mm in front of the base. The arm will move "
          "through 15 poses. Hand on the power switch, Enter to start ")
    try:
        robot.reset({})
        for i, (p, pitch, roll) in enumerate(calib_poses(a.board_x)):
            az = math.atan2(p[1] - K.PAN_AXIS_XY[1], p[0] - K.PAN_AXIS_XY[0])
            R = K.tool_rotation(math.radians(pitch), az, math.radians(roll))
            tgt = tuple(p) + tuple(matrix_to_rotvec(R))
            rep = robot.execute([Action(ActionMode.EE_ABS, tuple(p), raw=f"calib pose {i}", tcp_target=tgt)])
            if rep.stopped:
                raise SystemExit("the arm was blocked: aborting")
            obs = robot.observe()
            from PIL import Image
            Image.fromarray(obs.images["wrist"]).save(shots / f"pose_{i:02d}.jpg", quality=92)
            det = detect(cv2, board, obs.images["wrist"])
            reached = float(np.linalg.norm(obs.state.tcp_pos - p)) * 1000
            print(f"pose {i:2d}: reached within {reached:.0f} mm, {'board: ' + str(len(det[0])) + ' corners' if det else 'board NOT seen'}"
                  f" {[e.brief for e in rep.events]}")
            if det:
                samples.append((det, K.fk(obs.state.q), obs.images["wrist"].shape[1::-1]))
    finally:
        robot.hold()
        robot.close()
    if len(samples) < 6:
        raise SystemExit(f"only {len(samples)} poses saw the board: move it so the camera sees it from the home pose")
    size = samples[0][2]
    if "K" in intr:
        Kc, dist = np.asarray(intr["K"], float), np.asarray(intr["dist"], float)
        rms = intr.get("intrinsics_rms_px")
    else:
        rms, Kc, dist = calibrate_intrinsics(cv2, [s[0] for s in samples], size)
    Rg, tg, Rt, tt = [], [], [], []
    for (obj, img), T_base_tcp, _ in samples:
        ok, rvec, tvec = cv2.solvePnP(obj.astype(np.float32), img.astype(np.float32), Kc, dist)
        if not ok:
            continue
        Rg.append(T_base_tcp[:3, :3])
        tg.append(T_base_tcp[:3, 3])
        Rt.append(cv2.Rodrigues(rvec)[0])
        tt.append(tvec.reshape(3))
    R_cg, t_cg = cv2.calibrateHandEye(Rg, tg, Rt, tt, method=cv2.CALIB_HAND_EYE_PARK)
    T_tcp_cam = np.eye(4)
    T_tcp_cam[:3, :3], T_tcp_cam[:3, 3] = R_cg, t_cg.reshape(3)
    # check: the board origin in the base frame from every view
    origins = []
    for R_g, t_g, R_t, t_t in zip(Rg, tg, Rt, tt):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R_g, t_g
        Tb = np.eye(4)
        Tb[:3, :3], Tb[:3, 3] = R_t, t_t
        origins.append((T @ T_tcp_cam @ Tb)[:3, 3])
    origins = np.array(origins)
    spread = float(np.max(np.linalg.norm(origins - origins.mean(0), axis=1))) * 1000
    rec = json.loads(out.read_text()) if out.exists() else {}
    rec.update(K=Kc.tolist(), dist=np.asarray(dist).reshape(-1).tolist(), width=size[0], height=size[1],
               intrinsics_rms_px=rms, T_tcp_cam=T_tcp_cam.tolist(), handeye_views=len(Rg),
               board_origin_base_mm=(origins.mean(0) * 1000).round(1).tolist(), board_spread_mm=round(spread, 1),
               calibrated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"T_tcp_cam:\n{np.round(T_tcp_cam, 4)}\nboard origin (base, mm) {rec['board_origin_base_mm']} "
          f"(z should be ~0), spread over views {spread:.1f} mm -> {out}")


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("board")
    b.add_argument("out")
    b.add_argument("--square-mm", type=float, default=30.0)
    for name in ("intrinsics", "handeye"):
        s = sub.add_parser(name)
        s.add_argument("--square-mm", type=float, required=True)
        s.add_argument("--port", type=int, default=7811)
        s.add_argument("--out", default=str(DEFAULT_OUT))
        if name == "intrinsics":
            s.add_argument("--n", type=int, default=20)
            s.add_argument("--every-s", type=float, default=1.5)
        else:
            s.add_argument("--board-x", type=float, default=0.26)
            s.add_argument("--intrinsics", default=None)
            s.add_argument("--shots", default=f"runs/so101_calib_{time.strftime('%Y%m%dT%H%M%S')}")
    a = ap.parse_args(argv)
    {"board": cmd_board, "intrinsics": cmd_intrinsics, "handeye": cmd_handeye}[a.cmd](a)


if __name__ == "__main__":
    main(sys.argv[1:])
