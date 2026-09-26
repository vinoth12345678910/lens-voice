"""
tools/coco_sample.py — deterministic COCO subsampling for the LensVoice unified dataset.

Reads COCO instance annotations + a quotas JSON, and selects a subset of COCO
images per split using:

  1. multi-class images first (richest frames), bounded by global caps
  2. then per-class top-up of images until each class image-quota is met
  3. global caps resolve conflicts; fully deterministic (sorted image ids)

Emits:
  - reports/coco_sampling_proposal.{json,md}  (expected retained counts)
  - datasets/proposals/coco_selected_{split}.txt (selected COCO image ids)

No images are downloaded; this is analysis only.

args:
  --ann-dir   annotations dir        [datasets/source-coco/coco2017/annotations]
  --quotas    quotas json file       [datasets/proposals/coco_quotas.json]
  --out       report prefix          [reports/coco_sampling_proposal]
  --est-bytes assumed avg image size [210000]
"""
import argparse, json, os, collections, datetime

SPLITS = ("train", "val")
ANN_FILES = {"train": "instances_train2017.json", "val": "instances_val2017.json"}
LV_NAMES = {0: "person", 1: "car", 2: "motorcycle", 3: "bus", 4: "truck",
            5: "autorickshaw", 6: "bicycle", 7: "chair", 8: "table", 9: "couch",
            10: "bed", 11: "backpack", 12: "bag", 13: "suitcase", 14: "bottle",
            15: "cup", 16: "bowl", 17: "laptop", 18: "cell_phone", 19: "book",
            20: "keyboard", 21: "mouse", 22: "tv_monitor", 23: "bench",
            24: "umbrella", 25: "dog"}
COCO_NAMES = {
    "person": 0, "car": 1, "motorcycle": 2, "bus": 3, "truck": 4, "bicycle": 6,
    "chair": 7, "dining table": 8, "couch": 9, "bed": 10, "backpack": 11,
    "handbag": 12, "suitcase": 13, "bottle": 14, "cup": 15, "bowl": 16,
    "laptop": 17, "cell phone": 18, "book": 19, "keyboard": 20, "mouse": 21,
    "tv": 22, "bench": 23, "umbrella": 24, "dog": 25,
}


def select_split(split, class_sets, classes, quotas, cap):
    cls = [c for c in classes if c in class_sets]
    selected = []
    def contains(img_id):
        return frozenset(c for c in cls if img_id in class_sets[c])
    coverage = collections.defaultdict(int)
    # 1. multi-class images, richest first; reject if any class would exceed its quota
    multi = sorted(
        ((img, contains(img)) for img in set().union(*[class_sets[c] for c in cls])),
        key=lambda t: (-len(t[1]), t[0]))
    for img, img_c in multi:
        if len(selected) >= cap:
            break
        if len(img_c) < 2:
            continue
        if any(coverage[c] >= quotas[c] for c in img_c):
            continue
        selected.append(img)
        for c in img_c:
            coverage[c] += 1
    # 2. per-class top-up, largest deficit first; quota is a hard ceiling
    for c in sorted(cls, key=lambda c: -(quotas[c] - coverage[c])):
        if quotas[c] - coverage[c] <= 0:
            continue
        for img in class_sets[c]:
            if len(selected) >= cap or coverage[c] >= quotas[c]:
                break
            if img in selected:
                continue
            img_c = contains(img)
            if any(coverage[c2] >= quotas[c2] for c2 in img_c):
                continue
            selected.append(img)
            for c2 in img_c:
                coverage[c2] += 1
    return selected, coverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", default="datasets/source-coco/coco2017/annotations")
    ap.add_argument("--quotas", default="datasets/proposals/coco_quotas.json")
    ap.add_argument("--out", default="reports/coco_sampling_proposal")
    ap.add_argument("--est-bytes", type=int, default=210000)
    args = ap.parse_args()

    q = json.load(open(args.quotas))
    caps = q["global_image_caps"]
    tags = q["tags"]

    data = {}
    for split in SPLITS:
        with open(os.path.join(args.ann_dir, ANN_FILES[split])) as f:
            data[split] = json.load(f)

    cat_id_to_lv = {}
    for split in SPLITS:
        for c in data[split]["categories"]:
            if c["name"] in COCO_NAMES:
                cat_id_to_lv[c["id"]] = COCO_NAMES[c["name"]]

    ann_by_split = {}
    for split in SPLITS:
        class_sets = collections.defaultdict(set)
        img_lvs = collections.defaultdict(set)
        for a in data[split]["annotations"]:
            lv = cat_id_to_lv.get(a["category_id"])
            if lv is None:
                continue
            class_sets[lv].add(a["image_id"])
            img_lvs[a["image_id"]].add(lv)
        ann_by_split[split] = (class_sets, img_lvs)

    classes = sorted(COCO_NAMES.values())
    selected, coverage, report = {}, {}, {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                                          "quotas_file": args.quotas, "est_image_size_bytes": args.est_bytes}
    for split in SPLITS:
        class_sets, img_lvs = ann_by_split[split]
        quotas_lv = {c: q["image_quotas"][split][LV_NAMES[c]] for c in classes}
        sel, cov = select_split(split, class_sets, classes, quotas_lv, caps[split])
        selected[split] = sel
        coverage[split] = cov
        os.makedirs("datasets/proposals", exist_ok=True)
        with open(f"datasets/proposals/coco_selected_{split}.txt", "w") as f:
            f.write("\n".join(f"{i:012d}" for i in sel))

    # expected instance counts after selection
    split_inst = {}
    for split in SPLITS:
        class_sets, img_lvs = ann_by_split[split]
        sel = set(selected[split])
        inst = collections.Counter()
        for a in data[split]["annotations"]:
            lv = cat_id_to_lv.get(a["category_id"])
            if lv is not None and a["image_id"] in sel:
                inst[lv] += 1
        split_inst[split] = inst

    report["selected"] = {}
    for split in SPLITS:
        report["selected"][split] = {
            "images": len(selected[split]),
            "instances": sum(split_inst[split].values()),
            "est_size_mib": round(len(selected[split]) * args.est_bytes / 2 ** 20, 1),
        }
    report["instance_counts"] = {str(c): {s: split_inst[s][c] for s in SPLITS} for c in classes}
    report["image_coverage"] = {str(c): {s: coverage[s][c] for s in SPLITS} for c in classes}

    with open(args.out + ".json", "w") as f:
        json.dump(report, f, indent=2)

    md = []
    md.append(f"# COCO Sampling Proposal — LensVoice Unified Dataset\n")
    md.append(f"Generated {report['generated_at']}. Global caps: train `{caps['train']:,}`,"
              f" val `{caps['val']:,}` images. Selection: multi-class-first, then per-class top-up;"
              f" deterministic.\n")
    md.append("\n## Expected retained (COCO side only)\n")
    md.append("| split | images | instances | est. size |")
    md.append("|---|---|---|---|")
    for split in SPLITS:
        s = report["selected"][split]
        md.append(f"| {split} | {s['images']:,} | {s['instances']:,} |"
                  f" {s['est_size_mib']:,} MiB |")
    md.append("\n## Per-class (selected images / instances), train | val\n")
    md.append("| lv id | class | tag | train imgs | train inst | val imgs | val inst | quota train |")
    md.append("|---|---|---|---|---|---|---|---|")
    for c in classes:
        try:
            src_name = [n for n, lv in COCO_NAMES.items() if lv == c][0]
        except IndexError:
            src_name = "-"
        md.append(f"| {c} | {LV_NAMES[c]} | {tags[LV_NAMES[c]]} |"
                  f" {coverage['train'][c]:,} | {split_inst['train'][c]:,} |"
                  f" {coverage['val'][c]:,} | {split_inst['val'][c]:,} |"
                  f" {q['image_quotas']['train'][LV_NAMES[c]]:,} |")
    md.append("\n## Unified dataset (IDD + selected COCO)\n")
    idd = {"train": (31032, 332410), "val": (8866, 95396), "test": (4433, 46550)}
    md.append("| split | IDD imgs | COCO imgs | total imgs | total instances | est disk |")
    md.append("|---|---|---|---|---|---|")
    coco_inst = {s: sum(split_inst[s].values()) for s in SPLITS}
    for split in ["train", "val", "test"]:
        if split == "test":
            ci, cc = idd[split][0], 0
        else:
            ci = idd[split][0] + report["selected"][split]["images"]
            cc = report["selected"][split]["instances"]
        ti = idd[split][0] + (report["selected"][split]["images"] if split != "test" else 0)
        coco_gib = (report["selected"][split]["est_size_mib"] / 1024) if split != "test" else 0
        idd_gib = {"train": 1.64, "val": 0.47, "test": 0.24}[split]
        total_inst = (idd[split][1] if split != "test" else 0) + (cc if split != "test" else 0)
        inst_str = f"{total_inst:,}" if split != "test" else "—"
        md.append(f"| {split} | {idd[split][0]:,} | {report['selected'][split]['images'] if split != 'test' else 0:,} |"
                  f" {ti:,} | {inst_str} | {coco_gib+idd_gib:.2f} GiB |")
    md.append("\n## Notes\n")
    md.append("- No image downloads performed for this analysis; COCO avg size estimated at"
              f" {args.est_bytes/1024:.0f} KiB/image and will be confirmed at build time.")
    md.append(f"- Autorickshaw (id 5) is IDD-only; COCO contributes 0 instances to it.")
    md.append("- Val is the official COCO val2017 (disjoint from train); COCO test2017 is unlabeled"
              " → excluded; unified test = IDD test only (LensVoice-Val later).")
    open(args.out + ".md", "w").write("\n".join(md))
    print("wrote", args.out + ".json", "+ .md ; selected", {s: len(selected[s]) for s in SPLITS})


if __name__ == "__main__":
    main()