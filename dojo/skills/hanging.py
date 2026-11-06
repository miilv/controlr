"""Geometry candidates for holding a loop while carrying it to a hook."""
from __future__ import annotations

import numpy as np

from dojo.skills import grasps as G
from dojo.skills.geometry import T_from
from dojo.skills.library import functional_frames


def handle_grasps(obj):
    """Pinch a handle's outer bar across its loop plane, with mesh-confirmed opposed faces.

    The functional origin is outside the bar; sample inward offsets rather than
    assuming its thickness. Normal grasp IK, finger clearance and transfer
    collision checks still decide whether any candidate can be executed.
    """
    if obj.asset.mesh is None:
        return []
    out = []
    frames = functional_frames(obj, 'handle', active=True) or functional_frames(obj, 'handle')
    for frame in frames:
        H = obj.T @ frame
        closing = H[:3, 0]
        approach = np.array([0., 0., -1.])
        approach -= closing * float(approach @ closing)
        if np.linalg.norm(approach) < .5:
            continue
        approach /= np.linalg.norm(approach)
        for offset in (.003, .006, .009):
            point = H[:3, 3] - H[:3, 2] * offset
            for sign in (1., -1.):
                R = np.column_stack((approach, sign * closing, np.cross(approach, sign * closing)))
                T = T_from(point, R)
                width = G.mesh_width_at(obj, T, depths=(0., .004, .008))
                if width is None or not .004 <= width <= G.MAX_WIDTH:
                    continue
                out.append(G.Grasp('handle', T, width, pre_dist=.06, score=-.5,
                                   note=f'outer bar inset {offset*1000:.0f}mm width {width*1000:.1f}mm'))
    return out
