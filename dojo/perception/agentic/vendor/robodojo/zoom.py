"""zoom.py <png> <cx> <cy> <half> <out> [scale] [mark u,v ...]: crop a (2*half)^2 window around (cx,cy), upscale,
draw a labelled grid every 10 px (bold every 50) in ORIGINAL pixel coordinates, optional red cross marks."""
import sys
from PIL import Image, ImageDraw
src, cx, cy, half, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
scale = int(sys.argv[6]) if len(sys.argv) > 6 else 6
marks = [tuple(float(v) for v in m.split(",")) for m in sys.argv[7:]]
im = Image.open(src).convert("RGB")
x0, y0 = max(cx - half, 0), max(cy - half, 0)
x1, y1 = min(cx + half, im.width), min(cy + half, im.height)
c = im.crop((x0, y0, x1, y1)).resize(((x1 - x0) * scale, (y1 - y0) * scale), Image.NEAREST)
d = ImageDraw.Draw(c)
step = 10 if half <= 60 else 20
for x in range((x0 // step) * step, x1 + 1, step):
    if x < x0: continue
    X = (x - x0) * scale
    d.line([(X, 0), (X, c.height)], fill=(0, 255, 255) if x % 50 else (255, 255, 0), width=1)
    d.text((X + 2, 2), str(x), fill=(255, 255, 0))
for y in range((y0 // step) * step, y1 + 1, step):
    if y < y0: continue
    Y = (y - y0) * scale
    d.line([(0, Y), (c.width, Y)], fill=(0, 255, 255) if y % 50 else (255, 255, 0), width=1)
    d.text((2, Y + 2), str(y), fill=(255, 255, 0))
for (u, v) in marks:
    X, Y = (u - x0 + 0.5) * scale, (v - y0 + 0.5) * scale
    d.line([(X - 8, Y), (X + 8, Y)], fill=(255, 0, 0), width=2)
    d.line([(X, Y - 8), (X, Y + 8)], fill=(255, 0, 0), width=2)
c.save(out)
print(out, c.size)
