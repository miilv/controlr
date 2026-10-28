"""Where a layout record's asset identity comes from (the policy's identity contract; legality audit 2026-10-28 P1-2).

B1 places a label the public task config pins to a slot (make_kong's 22 tiles; a label stacked on another or held in a
support slot; a fixed-pose fixture) at that slot with the label's first candidate as a placeholder asset. It reads which
candidate is there only when the views can tell (``inverse_layout._slot_identity``: at most
``EstimatorConfig.slot_identity_max`` candidates and a margin over the runner-up), and then marks the entry
``_perception.identity = "read"``. A slot entry whose label has several candidates and no read is marked "unknown":
its ``category_idx`` is a placeholder that says nothing, and two unknown entries with the same index are NOT known to
match. ``runtime.estimate.objects_from_record`` carries this to ``ObjectState.identity_known``; skills that match
identities (``library.declare_kong``) use only known ones.

No task-config relation (select_mode ``same`` / ``same_index_as_label`` / ``same_as_label``) fills an identity here:
which unread objects share an asset is the answer of a matching task (make_kong: the discard matches one hand triple),
not a prior. Of the config, only the candidate POOL of each label is read (how many candidates it has).

Pure python (no cv2 / scipy): the runtime and the tests import it without the perception extras.
"""
from __future__ import annotations

RECORD_TYPES = ("Rigid", "Geometry", "Articulation", "Dynamic")
# "structural": records written before this contract, whose placeholder indices the config's identity ties set
UNREAD = ("unknown", "structural")


def identity_known(p: dict | None) -> bool:
    """Whether an entry's ``_perception`` leaves its asset identity known: read from the images, the label's only
    candidate, or chosen among the candidates by appearance / shape (every non-slot B1 entry). False for placeholders."""
    return (p or {}).get("identity") not in UNREAD


def _placed_from_config(p: dict) -> bool:
    """Entries B1 places from the config without choosing the asset among the candidates: config slots (incl. labels
    stacked on another one or pinned in a support slot) and fixed-pose fixtures."""
    return bool(p.get("slot")) or p.get("place_tag") == "fixed"


def identity_marks(record: dict, task: str, labels: dict | None = None) -> dict:
    """Mark, in place, every entry B1 placed from the config: "read" stays; a label with a single candidate needs no
    read; any other becomes "unknown" (its ``category_idx`` stays as the placeholder: the twin and the meshes need an
    asset). ``labels``: the task config's labels (default ``inverse_layout.task_map()[task]["labels"]``), of which
    only the candidate pools are read. Returns ``record``."""
    if labels is None:
        from .inverse_layout import task_map
        labels = (task_map().get(task) or {}).get("labels") or {}
    for typ in RECORD_TYPES:
        for lst in (record.get(typ) or {}).values():
            for r in lst:
                p = r.get("_perception") or {}
                if not _placed_from_config(p) or p.get("identity") == "read":
                    continue
                cands = set(p.get("geom_group") or (labels.get(r.get("label")) or {}).get("candidates") or [])
                if len(cands) == 1:
                    continue
                r.setdefault("_perception", {})["identity"] = "unknown"
    return record
