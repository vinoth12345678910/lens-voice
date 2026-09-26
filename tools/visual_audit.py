#!/usr/bin/env python
"""Render annotated IDD images for human visual label audit.

Saves PNG samples into reports/visual_audit/ covering: one image per class,
random images, the smallest-box images, the most crowded images, and images
containing the contested 'vehicle fallback' and 'rider' classes.

Usage:
    venv/bin/python tools/visual_audit.py [--data datasets/original/idd-1] [--split train] [--out reports]
"""
import argparse
import glob
import os
import random

from PIL import Image, ImageDraw

NAMES = ["animal", "autorickshaw", "bicycle", "bus", "car", "motorcycle",
         "person", "rider", "traffic light", "traffic sign", "truck",
         "vehicle fallback"]
COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
          (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
          (210, 245, 60), (250, 190, 212), (0, 128, 128), (128, 0, 0)]


def load_meta(lbl_dir):
    """Return {stem: {"classes": set, "count": int, "min_box": (area, cls)}}."""
    meta = {}
    for p in glob.glob(os.path.join(lbl_dir, "*.txt")):
        stem = os.path.basename(p)[:-4]
        classes = set()
        count = 0
        min_area, min_cls = None, None
        for line in open(p):
            parts = line.split()
            if len(parts) != 5:
                continue
            c = int(parts[0])
            _, _, w, h = map(float, parts[1:5])
            classes.add(c)
            count += 1
            a = w * h
            if min_area is None or a < min_area:
                min_area, min_cls = a, c
        meta[stem] = {"classes": classes, "count": count,
                      "min_area": min_area, "min_cls": min_cls}
    return meta


def render(image_path, label_path, name, out_dir, crop=None):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    fw, fh = img.size
    for line in open(label_path):
        parts = line.split()
        if len(parts) != 5:
            continue
        c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
        x1, y1 = (cx - w / 2) * fw, (cy - h / 2) * fh
        x2, y2 = (cx + w / 2) * fw, (cy + h / 2) * fh
        draw.rectangle([x1, y1, x2, y2], outline=COLORS[c % len(COLORS)], width=2)
        draw.text((x1, max(0, y1 - 12)), f"{NAMES[c]}", fill=COLORS[c % len(COLORS)])
    if crop:
        x, y, s = crop
        img = img.crop((x, y, x + s, y + s))
    img.save(os.path.join(out_dir, name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/original/idd-1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="reports/visual_audit")
    args = ap.parse_args()

    img_dir = os.path.join(args.data, args.split, "images")
    lbl_dir = os.path.join(args.data, args.split, "labels")
    os.makedirs(args.out, exist_ok=True)
    random.seed(7)

    meta = load_meta(lbl_dir)
    img_of = {stem: p for p in glob.glob(os.path.join(img_dir, "*.jpg"))
              for stem in [os.path.basename(p)[:-4]]}
    lbl_of = {stem: p for p in glob.glob(os.path.join(lbl_dir, "*.txt"))
              for stem in [os.path.basename(p)[:-4]]}
    stems = list(meta.keys())
    made = []

    def emit(stem, tag):
        if stem not in img_of:
            return
        s = meta[stem]
        render(img_of[stem], lbl_of[stem], f"{tag}.png", args.out)
        made.append((tag, len(s["classes"] & set(range(12)))))

    # one per class: image whose first box is that class
    for c in range(12):
        pick = next((st for st in stems if c in meta[st]["classes"]), None)
        if pick:
            emit(pick, f"class_{c:02d}_{NAMES[c].replace(' ', '_')}")

    # random images
    for i, st in enumerate(random.sample(stems, 6)):
        emit(st, f"random_{i}")

    # smallest object per class (crop around it)
    for c in range(12):
        best = min((st for st in stems if meta[st]["min_cls"] == c),
                   key=lambda st: meta[st]["min_area"]) if any(
            meta[st]["min_cls"] == c for st in stems) else None
        emit(best, f"smallest_{c:02d}_{NAMES[c]}") if best else None

    # most crowded
    crowded = sorted(stems, key=lambda st: -meta[st]["count"])[:4]
    for i, st in enumerate(crowded):
        emit(st, f"crowded_{i}_{meta[st]['count']}")

    # vehicle-fallback containing
    vf = [st for st in stems if 11 in meta[st]["classes"]]
    for i, st in enumerate(vf[:4]):
        emit(st, f"vehicle_fallback_{i}")
    # rider containing
    rd = [st for st in stems if 7 in meta[st]["classes"]]
    for i, st in enumerate(rd[:4]):
        emit(st, f"rider_{i}")

    print(f"rendered {len(made)} samples to {args.out}")


if __name__ == "__main__":
    main()