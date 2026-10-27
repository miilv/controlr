"""Infrastructure problems that only Kit's own log shows, per episode / call.

Two physics-only Isaac processes that start the same SDF task at the same moment can fail to cook its SDF colliders
(``omni.physx.cooking.plugin``: "UjitsoMeshCookingContext: cooking failure for .../collision/collisions"; compute6,
2026-10-27: fasten_screws G3, 14 of 15 episodes with uncooked nuts fell through, 1/15 against 1.0 for the same layout
run alone). The episode then measures the infrastructure, not the skill. ``KitLogWatch`` reads this process's Kit log
(carb setting ``/log/file``) from a mark and reports such lines.

    w = KitLogWatch()          # marks the log's current end
    ...                        # load, run
    w.since()                  # {} or {"cooking_failures": n, "first": "..."}
"""

from __future__ import annotations

import os

INFRA_PATTERNS = {"cooking_failures": "cooking failure"}


def kit_log_path() -> str | None:
    """This process's Kit log file (None outside Kit)."""
    try:
        import carb  # noqa: PLC0415 (only inside Isaac)
        p = carb.settings.get_settings().get("/log/file")
        return str(p) if p else None
    except Exception:  # noqa: BLE001
        return None


class KitLogWatch:
    def __init__(self, path: str | None = None):
        self.path = path if path is not None else kit_log_path()
        self.pos = self._size()

    def _size(self) -> int:
        try:
            return os.path.getsize(self.path) if self.path else 0
        except OSError:
            return 0

    def mark(self) -> None:
        self.pos = self._size()

    def since(self) -> dict:
        """Infrastructure lines logged since the mark: {kind: count, "first": the first such line} or {}."""
        if not self.path:
            return {}
        try:
            with open(self.path, "rb") as f:
                f.seek(self.pos)
                txt = f.read().decode("utf-8", errors="replace")
        except OSError:
            return {}
        out, first = {}, None
        for line in txt.splitlines():
            for kind, pat in INFRA_PATTERNS.items():
                if pat in line:
                    out[kind] = out.get(kind, 0) + 1
                    first = first or line.strip()[:240]
        if out:
            out["first"] = first
        return out
