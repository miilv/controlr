"""Fit the SO-101's joint zero offsets from teleop recordings of table touches.

Record (on the rig, scripts/so101_teleop_record.py): close the jaws, then touch the closed jaw
tips to the table at 8-12 spots — near and far (150-350 mm in front of the base), left and right,
the tool steep (pointing down) and shallow (tilted ~30-45 deg) — holding each touch still ~1 s.
Then here:

    uv run python scripts/so101_fit_offsets.py /path/to/touch_rec [more recs] [--fit-dz]

Prints the detected touches (time, q, FK TCP before the fit), the fit (offsets deg with rough
std, residual per touch), the TCP of two reference poses with the offsets, and writes a JSON with
``joint_offsets_deg`` for ``robot.params.joint_offsets_deg`` (docs/SO101.md "Joint zero
calibration"). The model is controlr/robot/so101/zero_fit.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from controlr.robot.so101 import zero_fit as Z

# reference poses from the ball demo (deg): jaws closed around a ~65 mm ball on the table
# (TCP expected ~15-40 mm up), and the folded rest pose (TCP above the table)
GRASP_DEG = [0.4, -11.2, 58.1, 40.3, -94.4]
REST_DEG = [1.0, -101.6, 98.1, 83.3, -89.5]


def _fmt(v, nd=1) -> str:
    return "[" + ", ".join(f"{x:7.{nd}f}" for x in v) + "]"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="+",
                    help="teleop recording dirs (joints.jsonl + calib_so101_follower.json); DIR@H: the touches "
                         "in DIR are on a surface H mm above the table (default: --tip-z-mm)")
    ap.add_argument("--tip-z-mm", type=float, default=0.0, help="height of the touched surface (table = 0)")
    ap.add_argument("--fit-dz", action="store_true",
                    help="also fit a base height offset dz (mm); needs touches on two or more surface heights")
    ap.add_argument("--joints", default=",".join(Z.FITTABLE), help="joints whose offsets are fitted")
    ap.add_argument("--prior-sigma-deg", type=float, default=10.0)
    ap.add_argument("--noise-mm", type=float, default=2.0, help="assumed height scatter per touch")
    ap.add_argument("--still-deg", type=float, default=0.5, help="max joint motion within a dwell")
    ap.add_argument("--min-dwell-s", type=float, default=0.6)
    ap.add_argument("--gripper-max", type=float, default=8.0, help="jaws closed: gripper <= this (0..100)")
    ap.add_argument("--min-lift-deg", type=float, default=-90.0, help="shoulder_lift at or below = rest pose")
    ap.add_argument("--out", default="runs/so101_joint_offsets.json")
    a = ap.parse_args(argv)

    dwells: list[Z.Dwell] = []
    for arg in a.recordings:
        rec, _, h = arg.partition("@")
        surface = float(h) if h else a.tip_z_mm
        try:
            t, q, g = Z.load_recording(rec)
        except (OSError, KeyError, ValueError) as e:
            print(f"{rec}: cannot read ({type(e).__name__}: {e})")
            continue
        found, rej = Z.find_dwells(t, q, g, still_deg=a.still_deg, min_s=a.min_dwell_s, gripper_max=a.gripper_max,
                                   min_lift_deg=a.min_lift_deg, source=str(rec))
        print(f"{rec}: {len(t)} ticks over {t[-1] - t[0] if len(t) else 0:.1f} s; still windows "
              f"{rej['still_windows']}, rejected: jaws open {rej['gripper_open']}, rest pose {rej['rest_pose']}; "
              f"touches {len(found)}" + (f" (surface z {surface:.1f} mm)" if surface else ""))
        for d in found:
            d.surface_z_mm = surface
        dwells += found
    if not dwells:
        print("no touch dwells found: record closed-jaw touches on the table, still for >= "
              f"{a.min_dwell_s} s each, away from the rest pose (docs/SO101.md 'Joint zero calibration')")
        return 2

    print(f"\n{'#':>2} {'t0..t1 s':>13}  q deg [pan, lift, elbow, wrist, roll]{'':6} grip  TCP mm before fit")
    for k, d in enumerate(dwells):
        print(f"{k:2d} {d.t0:6.2f}..{d.t1:6.2f}  {_fmt(d.q_deg)}  {d.gripper:4.1f}  {_fmt(Z.tcp_mm(d.q_deg))}")

    joints = tuple(j.strip() for j in a.joints.split(",") if j.strip())
    Q = np.array([d.q_deg for d in dwells])
    tips = np.array([d.surface_z_mm for d in dwells])
    fit = Z.fit_offsets(Q, tip_z_mm=tips, joints=joints, fit_dz=a.fit_dz,
                        prior_sigma_deg=a.prior_sigma_deg, noise_mm=a.noise_mm)

    heights = ", ".join(f"{h:.1f}" for h in sorted(set(tips.tolist())))
    print(f"\nfit ({len(Q)} touches, surface z = {heights} mm, prior 0 +- {a.prior_sigma_deg:.0f} deg, "
          f"noise {a.noise_mm:.1f} mm):")
    for nm, v, s, sh in zip(fit.names, fit.params, fit.std, fit.prior_share):
        unit = "mm" if nm == "dz" else "deg"
        print(f"  {nm:14s} {v:+7.2f} +- {s:4.2f} {unit}   (posterior/prior std {sh:.2f})")
    print(f"  condition number {fit.cond:.0f}; weakest direction {fit.weakest}")
    print(f"\n{'#':>2}  z before  z after (mm, rel. to surface)")
    for k, (rb, ra) in enumerate(zip(fit.residuals_before_mm, fit.residuals_mm)):
        print(f"{k:2d}  {rb:+8.1f}  {ra:+8.1f}")
    print(f"  rms {np.sqrt(np.mean(fit.residuals_before_mm ** 2)):.1f} -> {np.sqrt(np.mean(fit.residuals_mm ** 2)):.1f} mm")
    for w in fit.warnings:
        print("WARNING:", w)

    print("\nreference poses, TCP mm before -> after:")
    for nm, q, expect in (("grasp (ball ~65 mm)", GRASP_DEG, "z ~15-40"), ("rest", REST_DEG, "z > 0")):
        print(f"  {nm:20s} {_fmt(Z.tcp_mm(q))} -> {_fmt(Z.tcp_mm(q, fit.offsets_deg, fit.dz_mm))}   expected {expect}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {"joint_offsets_deg": np.round(fit.offsets_deg, 3).tolist(), "dz_mm": round(fit.dz_mm, 2) if a.fit_dz else None,
           "fitted": fit.names, "std": np.round(fit.std, 3).tolist(), "condition": fit.cond,
           "residuals_mm": np.round(fit.residuals_mm, 2).tolist(),
           "residuals_before_mm": np.round(fit.residuals_before_mm, 2).tolist(), "warnings": fit.warnings,
           "surface_z_mm": sorted(set(tips.tolist())), "prior_sigma_deg": a.prior_sigma_deg, "noise_mm": a.noise_mm,
           "recordings": list(a.recordings), "dwells": [d.as_dict() for d in dwells]}
    out.write_text(json.dumps(res, indent=1))
    print(f"\nsaved {out}\nuse: robot.params.joint_offsets_deg: {res['joint_offsets_deg']}"
          + (f"  (and dz {fit.dz_mm:+.1f} mm: the table is not at base z = 0)" if a.fit_dz else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
