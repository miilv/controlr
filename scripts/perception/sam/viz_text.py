"""DEV: SAM 3.1 text masks for a frame's vocabulary, drawn over the head image (colour per mask, prompt + score)."""
import sys, json, cv2, numpy as np
from pathlib import Path
from dojo.perception import sam_masks
from dojo.perception.inverse_layout import label_groups, restrict_by_instruction

ep, out = Path(sys.argv[1]), sys.argv[2]
rec = json.loads((ep / "frame_0.json").read_text())
img = cv2.cvtColor(cv2.imread(str(ep / "frame_0_cam_head.png")), cv2.COLOR_BGR2RGB)
task = ep.parent.name
groups = restrict_by_instruction(label_groups(task), rec["obs"]["instruction"])
voc = sam_masks.vocabulary(groups, rec["obs"]["instruction"])
ms = sam_masks.client().text(img, voc, thr=float(sys.argv[3]) if len(sys.argv) > 3 else 0.4)
vis = img.copy()
for i, m in enumerate(ms):
    col = np.array([(67 * i + 40) % 255, (151 * i + 90) % 255, (211 * i + 170) % 255])
    vis[m.mask > 0] = (0.45 * vis[m.mask > 0] + 0.55 * col).astype(np.uint8)
    ys, xs = np.nonzero(m.mask)
    cv2.putText(vis, f"{m.prompt[:14]} {m.score:.2f}", (int(xs.min()), max(10, int(ys.min()) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
cv2.imwrite(out, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
print(voc); print([(m.prompt, round(m.score, 2), int(m.mask.sum())) for m in ms])
