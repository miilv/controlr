"""make_kong glyph classifier on tile crops (``kong_crops.py``): 42 glyphs + "back" (a face-down tile shows no glyph).
DEV training / eval script (torch + torchvision; ImageNet ResNet-18 fine-tuned). 0 LLM calls.

    python scripts/perception/seg/kong_cls.py train --crops ~/dojo-data/segtrain/kong_crops --out ~/dojo-data/segtrain/kong_cls \\
        [--epochs 30] [--precision 0.98]
    python scripts/perception/seg/kong_cls.py eval --crops C --model ~/dojo-data/segtrain/kong_cls/model.pt [--split test]

Labels: the tile's asset index when its face (local +z) turns toward the camera (within 80 deg), "back" when it turns
away (beyond 95 deg); grazing views are left out. The read rule: top-1 is accepted when its
softmax margin over the runner-up is >= ``margin`` (chosen on the val split, ids 100049 variants: the smallest margin
whose accepted reads reach ``--precision``); below it the tile is "unread". Thresholds are never chosen on the test
split (held-out ids 100000-09 / 100020-24).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

N_GLYPH = 42          # set from <crops>/classes.json when present (digit_op: 10 digits + 5 operators)
BACK = N_GLYPH
NAMES: list = []
ROT_DEG = 180.0


def set_classes(crops: Path) -> None:
    global N_GLYPH, BACK, NAMES
    p = Path(crops) / "classes.json"
    if p.exists():
        NAMES = json.loads(p.read_text())["names"]
        N_GLYPH = len(NAMES)
        BACK = N_GLYPH


def label_of(r: dict) -> int | None:
    """The glyph when the tile's face turns toward the camera (cos > 0.17, i.e. within 80 deg), "back" when it turns
    away (cos < -0.09), None at grazing angles (left out). Old crops without ``face_dot``: by tilt from face-up."""
    fd = r.get("face_dot")
    if fd is not None:
        return int(r["cls"]) if fd > 0.17 else (BACK if fd < -0.09 else None)
    return BACK if r["tilt"] > 105 else int(r["cls"])


def view_type(r: dict) -> str:
    """D-C17's two view types: (2) the tile held in the jaws, seen by the holding arm's or another camera
    (``held:<view>``), and (1) every other tile (``free:<view>``: frame-0 hand / discard / wall tiles, arm poses)."""
    v = "head" if r["view"] == "cam_head" else "wrist"
    return f"{'held' if r.get('is_held') else 'free'}:{v}"


VAL_IDS: set = set()   # train-split layouts moved to val (``--val-ids``): more held views to calibrate on


def layout_id(r: dict) -> str | None:
    import re

    m = re.search(r"_(1000\d\d)_v\d+_f", r["path"])
    return m.group(1) if m else None


def load_rows(crops: Path, split: str) -> list[dict]:
    rows = [json.loads(line) for line in open(crops / "meta.jsonl")]
    for r in rows:
        if r["split"] == "train" and layout_id(r) in VAL_IDS:
            r["split"] = "val"
    return [r for r in rows if r["split"] == split and label_of(r) is not None]


def view_kind(r: dict) -> str:
    """The margin's view kind: head / wrist, and held_head / held_wrist for the tile in the jaws (the draw read)."""
    v = "head" if r["view"] == "cam_head" else "wrist"
    return f"held_{v}" if r.get("is_held") else v


class Crops:
    def __init__(self, crops: Path, rows: list[dict], train: bool, size: int = 128):
        import torchvision.transforms as T

        self.crops, self.rows = crops, rows
        norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if train:   # tiles lie at any yaw; sizes vary with the view; mild colour change
            self.tf = T.Compose([T.RandomRotation(ROT_DEG), T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
                                 T.ColorJitter(0.25, 0.25, 0.2, 0.02), T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.3),
                                 T.ToTensor(), norm])
        else:
            self.tf = T.Compose([T.Resize((size, size)), T.ToTensor(), norm])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image

        r = self.rows[i]
        return self.tf(Image.open(self.crops / r["path"]).convert("RGB")), label_of(r)


def build_model(pretrained: bool = True):
    import torch.nn as nn
    import torchvision

    m = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, N_GLYPH + 1)
    return m


def predict(model, ds, device, bs: int = 256) -> np.ndarray:
    import torch

    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4)
    out = []
    model.eval()
    with torch.no_grad():
        for x, _ in dl:
            out.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, N_GLYPH + 1))


def margins(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.sort(P, axis=1)
    return P.argmax(1), s[:, -1] - s[:, -2]


def choose_margin(pred, marg, y, precision: float) -> float:
    """Smallest margin whose accepted reads are >= ``precision`` right (val split)."""
    for t in np.round(np.arange(0.0, 1.0, 0.01), 2):
        ok = marg >= t
        if ok.sum() == 0 or (pred[ok] == y[ok]).mean() >= precision:
            return float(t)
    return 1.0


def report(rows, P, margin) -> dict:
    """Per view type: top-1, the read rate and the precision among reads. ``margin``: one value or {"head", "wrist"}."""
    pred, marg = margins(P)
    if isinstance(margin, dict):
        thr = np.array([margin.get(view_kind(r), margin["head" if r["view"] == "cam_head" else "wrist"]) for r in rows])
    else:
        thr = np.full(len(rows), float(margin))
    y = np.array([label_of(r) for r in rows])
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        kind = "back" if y[i] == BACK else "glyph"
        groups[(view_type(r), kind)].append(i)
        groups[("all", kind)].append(i)
    out = {}
    for k, idx in sorted(groups.items()):
        idx = np.array(idx)
        acc = float((pred[idx] == y[idx]).mean())
        read = marg[idx] >= thr[idx]
        out[f"{k[0]}|{k[1]}"] = {"n": int(len(idx)), "top1": round(acc, 3), "read": round(float(read.mean()), 3),
                                 "read_right": round(float((pred[idx][read] == y[idx][read]).mean()) if read.any() else 0.0, 3)}
    return out


def cmd_train(a):
    import torch

    global ROT_DEG
    ROT_DEG = a.rot_deg
    set_classes(Path(a.crops).expanduser())
    VAL_IDS.update(x.strip() for x in (a.val_ids or "").split(",") if x.strip())
    crops, out = Path(a.crops).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    tr, va = load_rows(crops, "train"), load_rows(crops, "val")
    ys = np.array([label_of(r) for r in tr])
    # class-balanced sampling: glyphs are 42 classes with uneven counts, backs are many
    w = 1.0 / np.bincount(ys, minlength=N_GLYPH + 1)[ys]
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(ys), replacement=True)
    dl = torch.utils.data.DataLoader(Crops(crops, tr, True), batch_size=a.batch, sampler=sampler, num_workers=6, drop_last=True)
    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl))
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for x, y in dl:
            opt.zero_grad()
            loss = lossf(model(x.to(dev)), y.to(dev))
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        P = predict(model, Crops(crops, va, False), dev)
        acc = float((P.argmax(1) == np.array([label_of(r) for r in va])).mean())
        print(f"[kong] epoch {ep + 1}/{a.epochs} loss {tot / len(dl):.3f} val top1 {acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
    P = predict(model, Crops(crops, va, False), dev)
    pred, marg = margins(P)
    yv = np.array([label_of(r) for r in va])
    margin = choose_margin(pred, marg, yv, a.precision)
    # per view kind (the runtime reads head and wrist crops): its own margin when val has >= 20 crops of it
    per_view = {}
    kinds = np.array([view_kind(r) for r in va])
    for kind in ("head", "wrist", "held_head", "held_wrist"):
        sel = kinds == kind
        per_view[kind] = choose_margin(pred[sel], marg[sel], yv[sel], a.precision) if sel.sum() >= 20 else margin
    torch.save({"state": model.state_dict(), "margin": margin, "margins": per_view, "precision": a.precision,
                "n_glyph": N_GLYPH, "names": NAMES or None, "val_ids": sorted(VAL_IDS)}, out / "model.pt")
    (out / "val_report.json").write_text(json.dumps({"margin": margin, "margins": per_view,
                                                     "report": report(va, P, margin)}, indent=1))
    print(f"[kong] margin {margin}, per view {per_view} (val precision target {a.precision}); {time.time() - t0:.0f}s",
          flush=True)


def cmd_eval(a):
    import torch

    crops = Path(a.crops).expanduser()
    set_classes(crops)
    ck = torch.load(Path(a.model).expanduser(), map_location="cpu")
    VAL_IDS.update(ck.get("val_ids") or [])
    model = build_model(pretrained=False)
    model.load_state_dict(ck["state"])
    model = model.to("cuda")
    rows = load_rows(crops, a.split)
    P = predict(model, Crops(crops, rows, False), "cuda")
    mv = ck.get("margins") or {"head": ck["margin"], "wrist": ck["margin"]}
    rep = report(rows, P, mv)
    print(f"split {a.split}, margins {mv} (chosen on val for precision {ck['precision']})")
    print("| view (pose) | tiles | n | top-1 | read (margin) | right among read |")
    print("|---|---|---|---|---|---|")
    for k, v in rep.items():
        view, kind = k.split("|")
        print(f"| {view} | {kind} | {v['n']} | {v['top1']:.2f} | {v['read']:.2f} | {v['read_right']:.2f} |")
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--crops", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--precision", type=float, default=0.98)
    t.add_argument("--rot-deg", type=float, default=180.0,
                   help="rotation augmentation (digits: small, a 6 turned over is a 9)")
    t.add_argument("--val-ids", default="", help="training layouts (ids 100030-49) moved to val, e.g. 100046,100047,100048")
    e = sub.add_parser("eval")
    e.add_argument("--crops", required=True)
    e.add_argument("--model", required=True)
    e.add_argument("--split", default="test")
    e.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    {"train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
