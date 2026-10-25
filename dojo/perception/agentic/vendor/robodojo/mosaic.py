"""mosaic.py <png> <out> <half> <scale> cx,cy[:label] ... : crops around each centre with a grid every 5 px (labels every 10)
in ORIGINAL pixel coordinates, tiled 2 per row."""
import sys
from PIL import Image, ImageDraw
src, out, half, scale = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
im = Image.open(src).convert("RGB")
tiles = []
for spec in sys.argv[5:]:
    lab = spec.split(":")[1] if ":" in spec else ""
    cx, cy = [int(float(v)) for v in spec.split(":")[0].split(",")]
    x0, y0 = cx - half, cy - half
    c = Image.new("RGB", (2 * half, 2 * half))
    c.paste(im.crop((x0, y0, x0 + 2 * half, y0 + 2 * half)), (0, 0))
    c = c.resize((2 * half * scale, 2 * half * scale), Image.NEAREST)
    d = ImageDraw.Draw(c)
    for x in range((x0 // 5) * 5, x0 + 2 * half + 1, 5):
        if x < x0: continue
        X = (x - x0) * scale
        d.line([(X, 0), (X, c.height)], fill=(255, 255, 0) if x % 10 == 0 else (0, 200, 200), width=1)
        if x % 10 == 0: d.text((X + 2, 2), str(x), fill=(255, 255, 0))
    for y in range((y0 // 5) * 5, y0 + 2 * half + 1, 5):
        if y < y0: continue
        Y = (y - y0) * scale
        d.line([(0, Y), (c.width, Y)], fill=(255, 255, 0) if y % 10 == 0 else (0, 200, 200), width=1)
        if y % 10 == 0: d.text((2, Y + 2), str(y), fill=(255, 255, 0))
    d.text((c.width - 40, c.height - 14), lab, fill=(255, 0, 255))
    tiles.append(c)
W = tiles[0].width
rows = (len(tiles) + 1) // 2
M = Image.new("RGB", (2 * W + 6, rows * W + 6 * (rows - 1)), (255, 255, 255))
for i, t in enumerate(tiles):
    M.paste(t, ((i % 2) * (W + 6), (i // 2) * (W + 6)))
M.save(out); print(out, M.size)
