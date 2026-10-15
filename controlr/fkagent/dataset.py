"""Teleop recordings -> few-shot sessions in the FK agent's step format.

A recording (``scripts/so101_teleop_record.py``): ``joints.jsonl`` lines
``{"t", "leader": {joint: v}, "follower": {joint: v}, "frame"?: "<t>.jpg"}`` (+ ``{"event"}``
lines), ``frames/*.jpg`` (wrist camera, ~10 Hz) and ``calib_so101_follower.json``. Joint values
are LeRobot RANGE_M100_100 (gripper RANGE_0_100).

Grasp detection (gripper only, then checked on the frames):
* the follower's jaws STALL above the leader's command: ``F - L > stall_min`` (default 1.5) while
  the follower's jaws stand still (``|dF/dt| < still_rate``) — held for >= ``hold_min_s``. An empty
  close ends with F within ~1 of L, an opening has F below L (lag), a closing in progress has
  the follower still moving. The grasp moment = the start of that hold (jaw contact).
* frame check: when the gripper lifts the object, the region between the jaws moves with the
  camera (small change) while the rest of the image changes: ``jaw_diff / background_diff``
  between the grasp and the lift frame < 0.8 = consistent.

One session per grasp: t = 0 when the arm leaves the start pose (the last time before the grasp
that shoulder_lift was folded at its lower stop, as in the demos' start pose), key points
``start``, ``approach`` (TCP ~150 mm from the grasp), ``pre_grasp`` (~60 mm), any empty
closes (``miss``), ``grasp``, ``lift`` (+40 mm while holding). Each key point: MODEL joint angles
(measured + offsets), the motion (joint delta) from the previous key point, gripper, FK TCP and
tool pitch, and the wrist photo nearest in time.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from controlr.fkagent import geometry as G
from controlr.robot.so101.zero_fit import norm_to_deg

ARM = G.ARM


@dataclass
class Recording:
    name: str
    path: Path
    t: np.ndarray            # s
    q_meas: np.ndarray       # (N, 5) LeRobot DEGREES
    gF: np.ndarray           # follower gripper 0..100
    gL: np.ndarray           # leader gripper 0..100
    frame_t: np.ndarray      # times of the frames
    frame_names: list

    @property
    def q_model(self) -> np.ndarray:
        return G.to_model(self.q_meas)

    def frame_at(self, t: float, max_dt: float = 0.3) -> tuple[str | None, float]:
        if not len(self.frame_t):
            return None, math.inf
        i = int(np.argmin(np.abs(self.frame_t - t)))
        dt = float(abs(self.frame_t[i] - t))
        return (self.frame_names[i] if dt <= max_dt else None), dt

    def idx(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))


def load(path) -> Recording:
    d = Path(path)
    cal = json.loads((d / "calib_so101_follower.json").read_text())
    takes: list[list[dict]] = [[]]
    for line in (d / "joints.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") == "start" and takes[-1]:
            takes.append([])            # a restarted recorder appended another take
        if "event" in r or not r.get("follower") or not r.get("leader"):
            continue
        takes[-1].append(r)
    rows = max(takes, key=lambda rs: (rs[-1]["t"] - rs[0]["t"]) if rs else -1.0)
    t, q, gf, gl, ft, fn = [], [], [], [], [], []
    for r in rows:
        tt = float(r["t"])
        if t and tt <= t[-1]:
            continue
        f, lead = r["follower"], r["leader"]
        t.append(tt)
        q.append([float(norm_to_deg(f[j], cal[j]["range_min"], cal[j]["range_max"])) for j in ARM])
        gf.append(float(f.get("gripper", math.nan)))
        gl.append(float(lead.get("gripper", math.nan)))
        if r.get("frame") and (d / "frames" / r["frame"]).exists():
            ft.append(tt)
            fn.append(r["frame"])
    if not t:
        raise ValueError(f"{d}: no follower samples in joints.jsonl")
    return Recording(d.name, d, np.array(t), np.array(q).reshape(-1, 5), np.array(gf), np.array(gl),
                     np.array(ft), fn)


# --------------------------------------------------------------------------- grasp detection

@dataclass
class Hold:
    t0: float          # jaw contact (grasp moment)
    t1: float          # release (leader opens / stall ends)
    stall: float       # median F - L over the hold
    f: float           # follower gripper at contact
    l: float           # leader command at contact
    release: float = math.nan   # the jaws open again (F > f + 5): the object is let go

    @property
    def weak(self) -> bool:
        return self.stall < 1.5


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def detect_holds(rec: Recording, *, stall_min: float = 1.2, still_rate: float = 8.0, hold_min_s: float = 0.4,
                 gap_s: float = 0.25) -> list[Hold]:
    t, F, L = rec.t, rec.gF, rec.gL
    dF = np.gradient(F, t) if len(t) > 2 else np.zeros_like(F)
    mask = ((F - L) > stall_min) & (np.abs(dF) < still_rate)
    runs = _runs(mask)
    merged: list[list[int]] = []
    for i, j in runs:
        if merged and t[i] - t[merged[-1][1]] <= gap_s:
            merged[-1][1] = j
        else:
            merged.append([i, j])
    holds = []
    for i, j in merged:
        if t[j] - t[i] < hold_min_s:
            continue
        opened = np.where(F[j:] > F[i] + 5.0)[0]
        release = float(t[j + int(opened[0])]) if len(opened) else float(t[-1])
        holds.append(Hold(float(t[i]), float(t[j]), float(np.median(F[i:j + 1] - L[i:j + 1])),
                          float(F[i]), float(L[i]), release))
    return holds


def detect_empty_closes(rec: Recording, t_from: float, t_to: float, *, min_drop: float = 8.0,
                        window_s: float = 1.5, empty_tol: float = 1.2, closed_max: float = 8.0) -> list[float]:
    """Times in [t_from, t_to) where the follower jaws closed by >= ``min_drop`` within
    ``window_s`` and ended within ``empty_tol`` of the command, nearly shut (nothing between them)."""
    t, F, L = rec.t, rec.gF, rec.gL
    out: list[float] = []
    for i in range(rec.idx(t_from), rec.idx(t_to)):
        lo, hi = rec.idx(t[i] - 0.5), rec.idx(t[i] + 0.5)
        if F[i] > np.min(F[lo:hi + 1]) + 1e-9 or F[i] >= closed_max or abs(F[i] - L[i]) >= empty_tol:
            continue
        before = F[rec.idx(t[i] - window_s):i + 1]
        if float(np.max(before)) - F[i] < min_drop:
            continue
        if not out or t[i] - out[-1] > 1.0:
            out.append(float(t[i]))
    return out


def frame_check(rec: Recording, t_grasp: float, t_lift: float) -> dict:
    """Is the grasped object moving with the gripper? Mean abs grey difference between the grasp
    and lift frames inside the jaw region vs the top band of the image."""
    from PIL import Image

    fa, _ = rec.frame_at(t_grasp)
    fb, _ = rec.frame_at(t_lift)
    if fa is None or fb is None or fa == fb:
        return {"ok": None, "why": "no frames"}
    a = np.asarray(Image.open(rec.path / "frames" / fa).convert("L").resize((160, 90)), float)
    b = np.asarray(Image.open(rec.path / "frames" / fb).convert("L").resize((160, 90)), float)
    jaw = float(np.mean(np.abs(a[35:80, 20:100] - b[35:80, 20:100])))
    bg = float(np.mean(np.abs(a[:25] - b[:25])))
    ratio = jaw / max(bg, 1e-6)
    return {"ok": bool(ratio < 0.8), "jaw_diff": round(jaw, 1), "background_diff": round(bg, 1),
            "ratio": round(ratio, 2), "frames": [fa, fb]}


# --------------------------------------------------------------------------- sessions

def _start_index(rec: Recording, i_grasp: int, prev_end: int, start_lift_deg: float) -> int:
    """Last sample before the grasp with shoulder_lift folded at the start pose; the arm leaves
    the start pose right after it. None found: the end of the previous cycle / the recording start."""
    q = rec.q_meas
    folded = np.where(q[prev_end:i_grasp, 1] <= start_lift_deg)[0]
    if len(folded):
        return prev_end + int(folded[-1])
    return prev_end


def _index_at_distance(rec: Recording, i_from: int, i_to: int, target_tcp, dist_mm: float) -> int:
    """The last sample in [i_from, i_to) whose TCP is at least ``dist_mm`` from ``target_tcp``."""
    best = i_from
    for i in range(i_from, i_to):
        if float(np.linalg.norm(G.tcp_mm(rec.q_model[i]) - target_tcp)) >= dist_mm:
            best = i
    return best


def keypoint(rec: Recording, i: int, name: str, t0: float, prev_q=None) -> dict:
    q = rec.q_model[i]
    fr, dt = rec.frame_at(float(rec.t[i]))
    return {"name": name, "t": round(float(rec.t[i]) - t0, 3), "t_rec": round(float(rec.t[i]), 3),
            "q_deg": [round(float(v), 2) for v in q],
            "q_measured_deg": [round(float(v), 2) for v in rec.q_meas[i]],
            "motion_deg": None if prev_q is None else [round(float(v), 2) for v in q - np.asarray(prev_q)],
            "gripper": round(float(rec.gF[i]), 1), "gripper_cmd": round(float(rec.gL[i]), 1),
            **G.pose_summary(q), "frame": fr, "frame_dt": None if fr is None else round(dt, 3)}


def build_sessions(rec: Recording, *, labels: dict | None = None, start_lift_deg: float = -90.0,
                   approach_mm: float = 150.0, pre_mm: float = 60.0, lift_mm: float = 40.0) -> list[dict]:
    labels = labels or {}
    holds = detect_holds(rec)
    sessions = []
    prev_end = 0
    for k, h in enumerate(holds, 1):
        ig, ir = rec.idx(h.t0), rec.idx(h.release)
        i0 = _start_index(rec, ig, prev_end, start_lift_deg)
        regrip = any(o.t0 < rec.t[i0] <= o.release for o in holds[:k - 1])
        t0 = float(rec.t[i0])
        q = rec.q_model
        g_tcp = G.tcp_mm(q[ig])
        # the jaws open widest before the grasp = the approach opening
        g_open = float(np.max(rec.gF[i0:ig + 1])) if ig > i0 else float(rec.gF[ig])
        misses = detect_empty_closes(rec, t0, h.t0 - 0.2)
        kps = [keypoint(rec, i0, "start", t0)]
        first_close = rec.idx(misses[0]) if misses else ig
        ia = _index_at_distance(rec, i0, first_close, g_tcp, approach_mm)
        ip = _index_at_distance(rec, ia, first_close, g_tcp, pre_mm)   # before the first close attempt
        events = [(ia, "approach")] + [(rec.idx(m), "miss") for m in misses] + [(ip, "pre_grasp"), (ig, "grasp")]
        # lift: first time while holding the TCP is lift_mm above the grasp (or the highest point)
        z = np.array([G.tcp_mm(q[i])[2] for i in range(ig, ir + 1)])
        up = np.where(z >= g_tcp[2] + lift_mm)[0]
        il = ig + (int(up[0]) if len(up) else int(np.argmax(z)))
        if il > ig:
            events.append((il, "lift"))
        seen, prev_i = {i0}, i0
        for i, name in sorted(events, key=lambda e: e[0]):
            if i in seen and name != "grasp":
                continue
            seen.add(i)
            kps.append(keypoint(rec, i, name, t0, prev_q=q[prev_i]))
            prev_i = i
        fc = frame_check(rec, h.t0, float(rec.t[il])) if il > ig else {"ok": None, "why": "no lift"}
        sid = f"{rec.name}#{k}"
        sessions.append({
            "session_id": sid, "recording": rec.name, "cycle": k,
            "object": labels.get(sid) or labels.get(rec.name) or "object",
            "regrip": bool(regrip), "t0_rec": round(t0, 3),
            "grasp": {"t": round(h.t0 - t0, 3), "q_deg": kps_q(q[ig]), **G.pose_summary(q[ig]),
                      "gripper_open": round(g_open, 1), "gripper_stall": round(h.f, 1),
                      "gripper_cmd": round(h.l, 1), "stall": round(h.stall, 2), "weak_stall": h.weak,
                      "hold_s": round(h.t1 - h.t0, 2), "frame_check": fc},
            "misses": [round(m - t0, 3) for m in misses],
            "keypoints": kps,
        })
        prev_end = ir
    return sessions


def kps_q(q) -> list:
    return [round(float(v), 2) for v in q]


def export(sessions: list[dict], recs: dict[str, Recording], out_dir, *, long_edge: int = 640,
           quality: int = 85) -> Path:
    """Write ``sessions.json`` + ``images/<session>_<kp>.jpg`` (resized) and a contact sheet per
    session (``sheets/``) for checking the key points by eye."""
    from PIL import Image, ImageDraw

    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "sheets").mkdir(exist_ok=True)
    for s in sessions:
        rec = recs[s["recording"]]
        tiles = []
        for j, kp in enumerate(s["keypoints"]):
            kp["photo"] = None
            if not kp.get("frame"):
                continue
            im = Image.open(rec.path / "frames" / kp["frame"]).convert("RGB")
            sc = long_edge / max(im.size)
            im = im.resize((round(im.size[0] * sc), round(im.size[1] * sc)))
            name = f"{s['session_id'].replace('#', '_')}_{j:02d}_{kp['name']}.jpg"
            im.save(out / "images" / name, quality=quality)
            kp["photo"] = f"images/{name}"
            t = im.copy()
            ImageDraw.Draw(t).text((6, 6), f"{kp['name']} t={kp['t']:.2f} tcp={kp['tcp_mm']}", fill=(255, 0, 0))
            tiles.append(t)
        if tiles:
            w, h = tiles[0].size
            sheet = Image.new("RGB", (w * min(3, len(tiles)), h * ((len(tiles) + 2) // 3)))
            for j, t in enumerate(tiles):
                sheet.paste(t.resize((w, h)), ((j % 3) * w, (j // 3) * h))
            sheet.save(out / "sheets" / f"{s['session_id'].replace('#', '_')}.jpg", quality=80)
    (out / "sessions.json").write_text(json.dumps({"format": "fkagent-sessions-v1",
                                                   "joint_offsets_deg": G.JOINT_OFFSETS_DEG.tolist(),
                                                   "sessions": sessions}, indent=1))
    return out / "sessions.json"


def load_sessions(path) -> tuple[list[dict], Path]:
    p = Path(path)
    if p.is_dir():
        p = p / "sessions.json"
    return json.loads(p.read_text())["sessions"], p.parent


def summary_lines(sessions: list[dict]) -> list[str]:
    lines = [f"{'session':26s} {'object':12s} {'grasp t':>7s} {'TCP mm (x, y, z)':>20s} {'pitch':>6s} "
             f"{'open':>5s} {'stall':>6s} {'miss':>4s} {'frame chk':>9s} keypoints"]
    for s in sessions:
        g = s["grasp"]
        fc = g["frame_check"]
        lines.append(f"{s['session_id']:26s} {s['object'][:12]:12s} {g['t']:7.2f} "
                     f"{'(%.0f, %.0f, %.0f)' % tuple(g['tcp_mm']):>20s} {g['tool_pitch_deg']:6.0f} "
                     f"{g['gripper_open']:5.0f} {g['stall']:6.1f} {len(s['misses']):4d} "
                     f"{str(fc.get('ok')) + (' %.2f' % fc['ratio'] if 'ratio' in fc else ''):>9s} "
                     + ",".join(k["name"] for k in s["keypoints"]) + (" REGRIP" if s["regrip"] else ""))
    return lines
