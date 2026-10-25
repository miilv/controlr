#!/usr/bin/env python3
"""Grid-overlaid views for picking pixels by eye.
  grid.py full  IMG OUT [--step 32] [--scale 2]
  grid.py crops IMG OUT --box NAME:u0,v0,u1,v1[:step:scale] [--box ...] [--step 8] [--scale 5]
Labels are ORIGINAL image pixel coordinates (u = column, v = row)."""
import argparse
from PIL import Image, ImageDraw, ImageFont


def font(sz):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def gridded(img, u0, v0, u1, v1, step, scale, title=""):
    crop = img.crop((u0, v0, u1, v1)).resize(((u1 - u0) * scale, (v1 - v0) * scale), Image.NEAREST)
    pad = 34
    W, H = crop.size
    out = Image.new("RGB", (W + pad, H + pad + (18 if title else 0)), (255, 255, 255))
    oy = 18 if title else 0
    out.paste(crop, (pad, pad + oy))
    d = ImageDraw.Draw(out); f = font(11)
    if title:
        d.text((pad, 2), title, fill=(0, 0, 0), font=font(13))
    first_u = ((u0 + step - 1) // step) * step
    for u in range(first_u, u1 + 1, step):
        x = pad + (u - u0) * scale
        major = (u % (step * 4) == 0)
        d.line([(x, pad + oy), (x, pad + oy + H)], fill=(255, 255, 0) if major else (0, 255, 255), width=1)
        d.text((x - 8, oy + 2 + (12 if (u // step) % 2 else 0)), str(u), fill=(0, 0, 0), font=f)
    first_v = ((v0 + step - 1) // step) * step
    for v in range(first_v, v1 + 1, step):
        y = pad + oy + (v - v0) * scale
        major = (v % (step * 4) == 0)
        d.line([(pad, y), (pad + W, y)], fill=(255, 255, 0) if major else (0, 255, 255), width=1)
        d.text((1, y - 6), str(v), fill=(0, 0, 0), font=f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["full", "crops"])
    ap.add_argument("img"); ap.add_argument("out")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--scale", type=int, default=None)
    ap.add_argument("--box", action="append", default=[])
    a = ap.parse_args()
    img = Image.open(a.img).convert("RGB")
    if a.mode == "full":
        out = gridded(img, 0, 0, img.width, img.height, a.step or 32, a.scale or 2)
    else:
        tiles = []
        for b in a.box:
            parts = b.split(":")
            name, rest = parts[0], parts[1]
            u0, v0, u1, v1 = map(int, rest.split(","))
            st = int(parts[2]) if len(parts) > 2 else (a.step or 8)
            sc = int(parts[3]) if len(parts) > 3 else (a.scale or 5)
            u0, v0 = max(u0, 0), max(v0, 0); u1, v1 = min(u1, img.width), min(v1, img.height)
            tiles.append(gridded(img, u0, v0, u1, v1, st, sc, f"{name} u{u0}-{u1} v{v0}-{v1}"))
        W = sum(t.width for t in tiles) + 10 * (len(tiles) - 1); H = max(t.height for t in tiles)
        out = Image.new("RGB", (W, H), (255, 255, 255)); x = 0
        for t in tiles:
            out.paste(t, (x, 0)); x += t.width + 10
    out.save(a.out)
    print(a.out, out.size)


if __name__ == "__main__":
    main()
