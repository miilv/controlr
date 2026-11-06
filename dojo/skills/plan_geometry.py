"""cuRobo proxies from existing belief collision geometry, never simulator state."""
from __future__ import annotations

import numpy as np

from dojo.skills.geometry import T_from


def heightmap_boxes(obj, hm, margin, capacity=512):
    """Vertical columns of the native height field; max-pool when capacity requires it."""
    z = np.asarray(hm["Z"])
    stride = 1
    while int(np.ceil(z.shape[0]/stride))*int(np.ceil(z.shape[1]/stride)) > capacity:
        stride += 1
    world = "R" in hm
    rotation = np.eye(3) if world else obj.R
    low = float(obj.bottom_z-obj.pos[2]-margin) if world else float(obj.asset.center[2]-obj.asset.extents[2]/2-margin)
    result = []
    for y in range(0, z.shape[0], stride):
        for x in range(0, z.shape[1], stride):
            upper = float(np.max(z[y:y+stride, x:x+stride])) + margin
            if not np.isfinite(upper) or upper <= low:
                continue
            nx, ny = min(stride, z.shape[1]-x), min(stride, z.shape[0]-y)
            center = np.array([hm["x0"]+(x+nx/2)*hm["cell"],
                               hm["y0"]+(y+ny/2)*hm["cell"], (low+upper)/2])
            half = np.array([nx*hm["cell"]/2, ny*hm["cell"]/2, (upper-low)/2])
            result.append((obj.pos+rotation@center, rotation, half, obj.label))
    return result


def mesh_spheres(held, tcp_offset, capacity=256):
    """Cover the believed mesh surface with clustered spheres; retain concavities."""
    mesh = getattr(held, "mesh", None)
    if mesh is None:
        return None
    from dojo.skills.insertion import _surface_samples
    vertices, faces = map(np.asarray, mesh[:2])
    if not len(vertices) or not len(faces):
        return None
    if not hasattr(held, "_surface_cover"):
        step = .008
        points = _surface_samples(vertices, faces, step=step)
        longest = max(float(np.linalg.norm(vertices[faces[:, a]]-vertices[faces[:, b]], axis=1).max())
                      for a, b in ((0, 1), (1, 2), (0, 2)))
        # _surface_samples caps at 40 edge subdivisions. This radius covers the
        # gaps between its triangle-lattice samples, rather than only vertices.
        padding = max(step, longest/40) + .002
        cell = .012
        while True:
            keys, index = np.unique(np.floor(points/cell).astype(int), axis=0, return_inverse=True)
            if len(keys) <= capacity:
                break
            cell *= 1.25
        low = np.full((len(keys), 3), np.inf); high = -low.copy()
        np.minimum.at(low, index, points); np.maximum.at(high, index, points)
        centers = (low+high)/2
        radii = np.linalg.norm((high-low)/2, axis=1)+padding
        held._surface_cover = (centers, radii)
    centers, radii = held._surface_cover
    transform = T_from((tcp_offset, 0, 0)) @ held.T_tcp_obj
    return np.c_[centers@transform[:3, :3].T+transform[:3, 3], radii]
