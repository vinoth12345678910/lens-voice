#!/usr/bin/env python
"""Dataset audit for a YOLO-format detection dataset.

Checks image/label pairing, class ids, and bounding-box validity for every
label file, plus summary stats. Writes reports/dataset_audit.json and
reports/dataset_audit.md.

YOLO box format is normalized: cx, cy, w, h in [0, 1]. All geometry checks
below are based on that (corners = cx +/- w/2).

Usage:
    venv/bin/python tools/audit_dataset.py [--data datasets/original/idd-1] [--out reports]
"""
import argparse
import collections
import glob
import json
import os

EPS = 1e-4


def audit_split(root, split, out, nc=12):
    # support both Roboflow ("<root>/<split>/images") and nested ("<root>/images/<split>")
    if os.path.isdir(os.path.join(root, split, "images")):
        img_dir = os.path.join(root, split, "images")
        lbl_dir = os.path.join(root, split, "labels")
    elif os.path.isdir(os.path.join(root, "images", split)):
        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)
    else:
        raise SystemExit(f"cannot find split layout for {root}/{split}")
    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    labels = sorted(glob.glob(os.path.join(lbl_dir, "*.txt")))

    img_stems = {os.path.splitext(os.path.basename(p))[0] for p in images}
    lbl_stems = {os.path.splitext(os.path.basename(p))[0] for p in labels}

    empty = 0
    insts_per_class = collections.Counter()
    imgs_per_class = collections.Counter()
    bad_class = 0
    bad_neg = 0            # any cx,cy,w,h < 0
    zero_area = 0
    out_of_bounds = 0      # a corner exceeds [0,1]
    tiny = 0               # area < 1e-4 (~6px at 640; ~41px^2)
    small = 0              # w or h < 1e-2
    huge = 0               # w or h > 0.95
    dup_boxes = 0
    dup_file_lines = 0
    total_inst = 0
    total_files = 0
    area_buckets = collections.Counter()
    images_any_box = 0

    for p in labels:
        lines = [l for l in open(p, encoding="utf-8").read().splitlines() if l.strip()]
        if not lines:
            empty += 1
            continue
        total_files += 1
        boxes = collections.Counter()
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue  # counted elsewhere as malformed
            try:
                cx, cy, w, h = (float(parts[1]), float(parts[2]),
                                float(parts[3]), float(parts[4]))
                c = float(parts[0])
            except ValueError:
                bad_class += 1
                continue
            if c != int(c) or c < 0 or c > nc - 1:
                bad_class += 1
            ci = int(c)
            total_inst += 1
            insts_per_class[ci] += 1
            key = (ci, round(cx, 4), round(cy, 4), round(w, 4), round(h, 4))
            boxes[key] += 1

            if any(v < 0 for v in (cx, cy, w, h)):
                bad_neg += 1
            if w <= 0 or h <= 0:
                zero_area += 1
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
            if x1 < -EPS or y1 < -EPS or x2 > 1 + EPS or y2 > 1 + EPS:
                out_of_bounds += 1
            if w * h < 1e-4:
                tiny += 1
            if w < 1e-2 or h < 1e-2:
                small += 1
            if w > 0.95 or h > 0.95:
                huge += 1
            area_buckets[bucket(w * h)] += 1

        dup_boxes += sum(n - 1 for n in boxes.values() if n > 1)
        dup_file_lines += len(lines) - len(set(lines))
        for ci in {k[0] for k in boxes}:
            imgs_per_class[ci] += 1

    images_with_box = total_files
    out = {
        "split": split,
        "images": len(images),
        "labels": len(labels),
        "labels_missing_image": len(lbl_stems - img_stems),
        "images_missing_label": len(img_stems - lbl_stems),
        "empty_label_files": empty,
        "nonempty_label_files": total_files,
        "images_with_any_box": images_with_box,
        "instances": total_inst,
        "instances_per_class": dict(sorted(insts_per_class.items())),
        "images_per_class": dict(sorted(imgs_per_class.items())),
        "issues": {
            "invalid_class_id": bad_class,
            "negative_coords": bad_neg,
            "zero_area": zero_area,
            "out_of_bounds": out_of_bounds,
            "tiny_boxes_area_lt_1e4": tiny,
            "small_side_lt_0_01": small,
            "huge_side_gt_0_95": huge,
            "duplicate_boxes_within_file": dup_boxes,
            "duplicate_dupe_lines_within_file": dup_file_lines,
        },
        "area_buckets": dict(sorted(area_buckets.items())),
    }
    out["issues_total"] = sum(out["issues"].values())
    assert isinstance(out, dict)
    out = dict(out)  # plain dict for json
    return out


def bucket(area):
    """Area buckets that matter for small-object analysis (~640px images)."""
    if area < 4e-5:
        return "<4e-5 (~<16px area @640)"
    if area < 1e-4:
        return "1e-4-4e-5"
    if area < 4e-4:
        return "4e-5-1e-3"
    if area < 1e-2:
        return "<1% "
    if area < 0.1:
        return "<10%"
    return ">=10%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/original/idd-1")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--splits", default="train,valid,test")
    ap.add_argument("--nc", type=int, default=12)
    args = ap.parse_args()

    results = {}
    for split in args.splits.split(","):
        results[split] = audit_split(args.data, split, args.out, nc=args.nc)

    total = {"instances": sum(r["instances"] for r in results.values()),
             "images": sum(r["images"] for r in results.values()),
             "issues": sum(r["issues_total"] for r in results.values())}
    payload = {"data_root": args.data, "total": total, "splits": results}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "dataset_audit.json"), "w") as f:
        json.dump(payload, f, indent=2)

    md = ["# Dataset Audit Report", "",
          f"- data root: `{args.data}`",
          f"- total images: {total['images']}",
          f"- total instances: {total['instances']}",
          f"- total flagged label issues: {total['issues']}",
          ""]
    for split, r in results.items():
        md += [
            f"## {split}", "",
            f"- images: {r['images']} | labels: {r['labels']} | instances: {r['instances']}",
            f"- empty label files: {r['empty_label_files']} | images with a box: {r['images_with_any_box']}",
            f"- pairing: images missing label = {r['images_missing_label']}, "
            f"labels missing image = {r['labels_missing_image']}", "",
            "| class | instances | images |", "|---|---|---|",
        ]
        names = ["animal", "autorickshaw", "bicycle", "bus", "car", "motorcycle",
                 "person", "rider", "traffic light", "traffic sign", "truck",
                 "vehicle fallback"]
        for ci in sorted(r["instances_per_class"]):
            md.append(f"| {ci} {names[ci] if ci < len(names) else '?'} | "
                      f"{r['instances_per_class'][ci]} | {r['images_per_class'][ci]} |")
        md += ["", "### Label issues", "", "| issue | count |", "|---|---|"]
        for k, v in r["issues"].items():
            md.append(f"| {k} | {v} |")
        md += ["", "### Box area buckets", "", "| bucket | count |", "|---|---|"]
        for k, v in r["area_buckets"].items():
            md.append(f"| {k} | {v} |")
        md += [""]
    with open(os.path.join(args.out, "dataset_audit.md"), "w") as f:
        f.write("\n".join(md))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()