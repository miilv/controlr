"""Objects the task config pins in a support slot of another object (pure python: no catalogue, no images).

relative_plane "<base label>/<support key>" (make_toast: "bread_shelf/bread_shelf/0"; play_Xylophone:
"mallet_stand/mallet_stand/0"), a pinned x/y offset, no z range, no random turn. The base asset's public
``passive.support`` frame (catalogue info ``support``, written by ``catalog annotate``) places the object; without it
perception cannot (``dojo.runtime.catalog_check`` fails loudly).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


def support_slot(placement: dict) -> tuple[str, str] | None:
    """(base label, support key) for a slot-pinned placement, else None."""
    rp = placement.get("relative_plane", "Table")
    if not isinstance(rp, str) or "/" not in rp or placement.get("zlim") is not None or placement.get("rotate_rand"):
        return None
    xl, yl = placement.get("xlim") or [0, 0], placement.get("ylim") or [0, 0]
    if not (isinstance(xl, list) and len(xl) == 2 and not isinstance(xl[0], list) and xl[0] == xl[1]
            and isinstance(yl, list) and len(yl) == 2 and not isinstance(yl[0], list) and yl[0] == yl[1]):
        return None
    base, _, skey = rp.partition("/")
    return base, skey


@lru_cache(maxsize=1)
def settle_table() -> dict:
    """``slot_settle.json``: {"<base asset key>|<support key>|<object asset key>": {"pos", "quat", ...}}, keyed by the
    physical asset pair only (never a task or a layout): where physics puts that object in that slot of that base."""
    p = Path(__file__).with_name("slot_settle.json")
    return {k: v for k, v in json.loads(p.read_text()).items() if not k.startswith("_")} if p.exists() else {}


def settled(base_key: str, support_key: str, child_key: str) -> dict | None:
    return settle_table().get(f"{base_key}|{support_key}|{child_key}")


def slot_requirements(task_labels: dict) -> list[tuple[str, str, str, str, bool]]:
    """[(slot label, base candidate key, support key, object candidate key, validated)] a vocabulary needs from the
    catalogue (``task_labels`` = tasks.json's ``labels`` of one vocabulary); validated = the pair has settle data, so
    perception's slot path places it (and the base's support frame is REQUIRED)."""
    out = []
    for lab, d in task_labels.items():
        sl = support_slot(d.get("placement") or {})
        if sl is None or sl[0] not in task_labels or sl[0] == lab:
            continue
        for key in task_labels[sl[0]].get("candidates") or []:
            for ck in d.get("candidates") or []:
                out.append((lab, key, sl[1], ck, settled(key, sl[1], ck) is not None))
    return out
