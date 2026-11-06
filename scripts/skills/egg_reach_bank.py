"""Scene-free sensitivity of captured base-frame flange requests; never executes.

Input: PLAN's stripped request bank (q0, flange, limits). Rotation perturbations
preserve the TCP point, but are not object-geometry-valid release candidates.
"""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path

import numpy as np

from dojo.skills.egg_reach import practical_limits
from dojo.skills.geometry import T_from
from dojo.skills.holders import _rot
from dojo.skills.kinematics import X5Kin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bank', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    kin = X5Kin(roots={'left': ((0, 0, 0), (1, 0, 0, 0))})
    rows = []
    for index, item in enumerate(json.loads(args.bank.read_text())):
        req = item['request']
        kin.limits = np.asarray(req['limits'], float).copy()
        f = req['flange']
        tcp = kin.tcp_from_flange(T_from(f[:3], quat=f[3:]))
        out = dict(index=index, seed=item['seed'], phase=item['name'],
                   distance_m=float(np.linalg.norm(tcp[:2, 3])), variants=[])
        for profile in ('wide', 'practical'):
            with practical_limits(kin) if profile == 'practical' else nullcontext():
                variants = [('nominal', tcp)]
                for axis in range(3):
                    for deg in (-60, -30, -15, 15, 30, 60):
                        T = tcp.copy()
                        T[:3, :3] = _rot(np.eye(3)[axis], math.radians(deg)) @ T[:3, :3]
                        variants.append((f'axis{axis}:{deg}', T))
                for tag, T in variants:
                    r = kin.ik('left', T, req['q0'], n_random=4)
                    local = kin.solve('left', T, req['q0']) if not r.ok else r
                    out['variants'].append(dict(profile=profile, tag=tag, ok=r.ok, residual_m=r.pos_err,
                        local_residual_m=local.pos_err,
                        local_margin_rad=np.minimum(local.q - kin.limits[:, 0],
                                                    kin.limits[:, 1] - local.q).tolist(),
                        margin_rad=None if not r.ok else np.minimum(r.q - kin.limits[:, 0],
                                                                    kin.limits[:, 1] - r.q).tolist()))
        rows.append(out)
        args.out.write_text(json.dumps(rows, indent=2))
        print(index, {p: sum(v['ok'] for v in out['variants'] if v['profile'] == p)
                      for p in ('wide', 'practical')}, flush=True)


if __name__ == '__main__':
    main()
