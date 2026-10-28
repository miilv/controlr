"""Stream C dev: coarse CPU silhouettes (no GL context, no GPU job). Every catalogue asset's silhouette mesh is
re-decimated with a cell of extent / BGPERC_COARSE_DIV (default 30): a 25k-face mug becomes ~1-2k faces and a CPU
wrist refine drops from ~13 s to ~1 s; outlines move by at most cell * sqrt(3) / 2 (~3 mm on a 115 mm mug).
``install()`` before the first catalogue load. The runtime uses GL silhouettes (full meshes): this is an offline
stand-in, stated as such in every number it produces."""
import os
from functools import lru_cache


def install(div: float | None = None):
    import numpy as np

    from dojo.perception import inverse_layout as IL
    from dojo.perception import render as R
    div = float(div or os.environ.get("BGPERC_COARSE_DIV", "30"))

    @lru_cache(maxsize=64)
    def load_asset(key: str):
        a = R.Asset(key)
        ext = float(np.max(a.info["extents"]))
        if len(a.Fs) > 1500:
            a.Vs, a.Fs = R.cluster_decimate(a.V, a.F, ext / div)
            a.sil_cell = ext / div
        return a

    R.load_asset = load_asset
    if hasattr(IL, "load_asset"):
        IL.load_asset = load_asset
    IL.shape_models.cache_clear()
    return div
