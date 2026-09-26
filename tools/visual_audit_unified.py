#!/usr/bin/env python
"""Render annotated unified-dataset images for human visual label audit.

Covers, per split: one image per class (preferring idd_ for road classes and
coco_ for general classes, both when available), zoomed crops around the
smallest box per class, the most crowded images, and random images across
sources. Output PNGs to reports/visual_audit_unified/.

Usage:
    venv/bin/python tools/visual_audit_unified.py [--split train] [--out reports/visual_audit_unified]
"""
import argparse
import glob
import os
import random

from PIL import Image, ImageDraw

NAMES = ["person", "car", "motorcycle", "bus", "truck", "autorickshaw",
         "bicycle", "chair", "table", "couch", "bed", "backpack", "bag",
         "suitcase", "bottle", "cup", "bowl", "laptop", "cell_phone", "book",
         "keyboard", "mouse", "tv_monitor", "bench", "umbrella", "dog"]
COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
          (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
          (210, 245, 60), (250, 190, 212), (0, 128, 128), (128, 0, 0),
          (70, 100, 255), (255, 0, 128), (0, 200, 255), (128, 128, 0),
          (150, 0, 150), (0, 180, 100), (255, 150, 0), (30, 90, 180),
          (220, 60, 200), (90, 160, 30), (30, 30, 160), (160, 130, 60),
          (0, 90, 90), (255, 80, 80)]


def load_meta(lbl_dir):
    meta = {}
    for p in glob.glob(os.path.join(lbl_dir, "*.txt")):
        stem = os.path.basename(p)[:-4]
        classes, count, min_a, min_c = set(), 0, None, None
        for line in open(p):
            parts = line.split()
            if len(parts) != 5:
                continue
            c = int(parts[0])
            _, _, w, h = map(float, parts[1:5])
            classes.add(c)
            count += 1
            a = w * h
            if min_a is None or a < min_a:
                min_a, min_c = a, c
        meta[stem] = {"classes": classes, "count": count,
                      "min_area": min_a, "min_cls": min_c}
    return meta


def render(img_path, lbl_path, name, out_dir):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    fw, fh = img.size
    for line in open(lbl_path):
        parts = line.split()
        if len(parts) != 5:
            continue
        c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
        x1, y1 = (cx - w / 2) * fw, (cy - h / 2) * fh
        x2, y2 = (cx + w / 2) * fw, (cy + h / 2) * fh
        draw.rectangle([x1, y1, x2, y2],
                       outline=COLORS[c % len(COLORS)], width=2)
        draw.text((x1, max(0, y1 - 12)), NAMES[c],
                  fill=COLORS[c % len(COLORS)])
    img.save(os.path.join(out_dir, name))


def zoom(out_path, img_path, lbl_path, cls):
    """Crop a square around the smallest instance of cls and upscale (x4)."""
    img = Image.open(img_path).convert("RGB")
    fw, fh = img.size
    best = None
    for line in open(lbl_path):
        parts = line.split()
        if len(parts) != 5 or int(parts[0]) != cls:
            continue
        _, cx, cy, w, h = map(float, parts)
        x1, y1 = (cx - w / 2) * fw, (cy - h / 2) * fh
        x2, y2 = (cx + w / 2) * fw, (cy + h / 2) * fh
        if best is None or (x2 - x1) * (y2 - y1) < best[0]:
            best = ((x2 - x1) * (y2 - y1), x1, y1, x2, y2)
    if best is None:
        return None
    _, x1, y1, x2, y2 = best
    side = max(x2 - x1, y2 - y1) * 3
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    box = (max(0, int(cx - side / 2)), max(0, int(cy - side / 2)),
           min(fw, int(cx + side / 2)), min(fh, int(cy + side / 2)))
    crop = img.crop(box)
    s = 600 / max(crop.size)
    crop = crop.resize((int(crop.size[0] * s), int(crop.size[1] * s)),
                       Image.NEAREST)
    crop.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="reports/visual_audit_unified")
    args = ap.parse_args()

    img_dir = f"datasets/lensvoice-unified/images/{args.split}"
    lbl_dir = f"datasets/lensvoice-unified/labels/{args.split}"
    os.makedirs(args.out, exist_ok=True)
    random.seed(7)

    meta = load_meta(lbl_dir)
    img_of = {os.path.basename(p)[:-4]: p
              for p in glob.glob(os.path.join(img_dir, "*.jpg"))}
    lbl_of = {os.path.basename(p)[:-4]: p
              for p in glob.glob(os.path.join(lbl_dir, "*.txt"))}
    stems = [s for s in meta if s in img_of and s in lbl_of]
    src = lambda s: "idd" if s.startswith("idd_") else "coco"
    made = []

    def emit(stem, tag):
        render(img_of[stem], lbl_of[stem], f"{tag}.png", args.out)
        made.append(tag)

    # one per class, both sources when available
    for c in range(26):
        pool_idd = [s for s in stems if c in meta[s]["classes"] and src(s) == "idd"]
        pool_coco = [s for s in stems if c in meta[s]["classes"] and src(s) == "coco"]
        if c <= 5:
            pick = pool_idd or pool_coco
        else:
            pick = pool_coco or pool_idd
        if pick:
            emit(random.choice(pick), f"class_{c:02d}_{NAMES[c]}_{src(pick[0])}")
            if pool_idd and pool_coco and c > 5:
                emit(random.choice(pool_idd),
                     f"class_{c:02d}_{NAMES[c]}_idd_extras")

    # zoomed crops around the smallest box per class
    for c in range(26):
        cand = [s for s in stems if meta[s]["min_cls"] == c]
        if not cand:
            continue
        best = min(cand, key=lambda s: meta[s]["min_area"])
        out = os.path.join(args.out, f"tiny_{c:02d}_{NAMES[c]}_{src(best)}.png")
        if zoom(out, img_of[best], lbl_of[best], c):
            made.append(f"tiny_{c:02d}_{NAMES[c]}")

    # most crowded
    for i, s in enumerate(sorted(stems, key=lambda s: -meta[s]["count"])[:6]):
        emit(s, f"crowded_{i}_{meta[s]['count']}_{src(s)}")

    # random across sources
    for i, s in enumerate(random.sample(stems, 10)):
        emit(s, f"random_{i}_{src(s)}")

    print(f"rendered {len(made)} samples to {args.out}")


if __name__ == "__main__":
    main()