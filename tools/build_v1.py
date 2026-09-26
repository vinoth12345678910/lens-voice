#!/usr/bin/env python
"""Build the cleaned 6-class Dataset V1 from the immutable original.

Mapping (old id -> new id); all others are dropped:
    person(6), rider(7) -> person(0)
    car(4) -> car(1)
    motorcycle(5) -> motorcycle(2)
    bus(3) -> bus(3)
    truck(10) -> truck(4)
    autorickshaw(1) -> autorickshaw(5)

Dropped: animal(0), bicycle(2), traffic light(8), traffic sign(9),
vehicle fallback(11).

Images whose boxes all disappear are KEPT as hard negatives (empty label
file), capped so negatives never exceed a configurable fraction of positives.

Reads ONLY from datasets/original/idd-1. Writes datasets/cleaned/v1.

Usage:
    venv/bin/python tools/build_v1.py
"""
import argparse
import glob
import json
import os
import random
import shutil

NAMES = ["person", "car", "motorcycle", "bus", "truck", "autorickshaw"]
OLD_TO_NEW = {6: 0, 7: 0, 4: 1, 5: 2, 3: 3, 10: 4, 1: 5}
# (source split dir, dest split dir) - dest uses canonical ultralytics 'val'
SPLITS = [("train", "train"), ("valid", "val"), ("test", "test")]


def build_split(src, dst, split, dst_split=None, neg_cap=0.20, seed=7):
    img_src = os.path.join(src, split, "images")
    lbl_src = os.path.join(src, split, "labels")
    dst_split = dst_split or split
    img_dst = os.path.join(dst, "images", dst_split)
    lbl_dst = os.path.join(dst, "labels", dst_split)
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)

    positives = 0
    negatives = 0
    kept = 0
    empty_src = 0
    stats = {"images_copied": 0, "instances": 0, "pos": 0, "neg": 0,
             "per_class": {}, "deduped_boxes": 0}
    neg_pool = []

    for lbl_path in sorted(glob.glob(os.path.join(lbl_src, "*.txt"))):
        stem = os.path.basename(lbl_path)[:-4]
        img_path = os.path.join(img_src, stem + ".jpg")
        if not os.path.exists(img_path):
            continue
        lines = [ln for ln in open(lbl_path, encoding="utf-8").read().splitlines() if ln.strip()]
        if not lines:
            empty_src += 1
        out = []
        seen = set()
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                continue
            c = int(parts[0])
            if c not in OLD_TO_NEW:
                continue
            new = OLD_TO_NEW[c]
            tail = " ".join(parts[1:])
            box = (new, tail) if (new, tail) not in seen else None
            if box:
                seen.add((new, tail))
                out.append(f"{new} {tail}")
            else:
                stats["deduped_boxes"] += 1
        stats["instances"] += len(out)
        for cls_id in set(o.split()[0] for o in out):
            stats["per_class"][int(cls_id)] = stats["per_class"].get(int(cls_id), 0) + 1
        if out:
            positives += 1
            shutil.copy2(img_path, os.path.join(img_dst, stem + ".jpg"))
            with open(os.path.join(lbl_dst, stem + ".txt"), "w") as f:
                f.write("\n".join(out) + "\n")
            stats["images_copied"] += 1
        else:
            negatives += 1
            neg_pool.append(stem)
    stats["pos"] = positives
    stats["neg"] = negatives
    stats["empty_in_source"] = empty_src

    # cap negatives
    rng = random.Random(seed)
    rng.shuffle(neg_pool)
    cap = int(positives * neg_cap)
    keep_neg = neg_pool if negatives <= cap else neg_pool[:cap]
    for stem in keep_neg:
        shutil.copy2(os.path.join(img_src, stem + ".jpg"), os.path.join(img_dst, stem + ".jpg"))
        open(os.path.join(lbl_dst, stem + ".txt"), "w").close()
    stats["neg_kept"] = len(keep_neg)
    stats["neg_dropped"] = negatives - len(keep_neg)
    stats["images_copied"] += len(keep_neg)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/original/idd-1")
    ap.add_argument("--dst", default="datasets/cleaned/v1")
    ap.add_argument("--neg-cap", type=float, default=0.20,
                    help="max negatives as a fraction of positives per split")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if os.path.exists(args.dst):
        print(f"destination {args.dst} already exists; refusing to clobber a built dataset")
        return
    os.makedirs(args.dst)

    report = {"classes": NAMES, "neg_cap": args.neg_cap}
    for src_split, dst_split in SPLITS:
        # read from the immutable source <src_split>, write under canonical <dst_split>
        report[dst_split] = build_split(args.src, args.dst, src_split,
                                        dst_split=dst_split, neg_cap=args.neg_cap,
                                        seed=args.seed)

    ns = "0: person\n1: car\n2: motorcycle\n3: bus\n4: truck\n5: autorickshaw"
    yaml = f"# Dataset V1 — 6 classes (built from datasets/original/idd-1)\ntrain: images/train\nval: images/val\ntest: images/test\n\nnc: 6\nnames:\n  {ns.replace(':', ':\n', 1)}\n"
    # Cleaner: write the block explicitly instead of the trick above
    yaml = (
        "# Dataset V1 - 6 classes (derived, read-only source: datasets/original/idd-1)\n"
        "path: .                # anchor: relative paths below resolve against this YAML's dir\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 6\n"
        "names:\n"
        "  0: person\n"
        "  1: car\n"
        "  2: motorcycle\n"
        "  3: bus\n"
        "  4: truck\n"
        "  5: autorickshaw\n"
    )
    with open(os.path.join(args.dst, "data.yaml"), "w") as f:
        f.write(yaml)

    with open(os.path.join(args.dst, "STATS.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()