"""
tools/inspect_coco.py — COCO coverage / union / imbalance analysis for LensVoice.

Reads COCO 2017 instance annotations (train + val), maps the LensVoice-required
categories by *name*, and emits:

  * category statistics (instances, unique images, train/val split)
  * union statistics (unique required images, multi-class images, est. size)
  * class imbalance
  * provenance tracking id: COCO category id -> LensVoice class id

args:
  --ann-dir    dir containing instances_{train,val}2017.json   [cwd]
  --out        output path prefix (json + md)                  [reports/coco_coverage_analysis]
  --est-bytes  assumed avg image size in bytes                 [210000]
"""
import argparse, json, os, collections, datetime

COCO_NAMES = {
    "person": 0, "car": 1, "motorcycle": 2, "bus": 3, "truck": 4,
    "bicycle": 6, "chair": 7, "dining table": 8, "couch": 9, "bed": 10,
    "backpack": 11, "handbag": 12, "suitcase": 13, "bottle": 14, "cup": 15,
    "bowl": 16, "laptop": 17, "cell phone": 18, "book": 19, "keyboard": 20,
    "mouse": 21, "tv": 22, "bench": 23, "umbrella": 24, "dog": 25,
}
LV_NAMES = {0: "person", 1: "car", 2: "motorcycle", 3: "bus", 4: "truck",
            5: "autorickshaw", 6: "bicycle", 7: "chair", 8: "table", 9: "couch",
            10: "bed", 11: "backpack", 12: "bag", 13: "suitcase", 14: "bottle",
            15: "cup", 16: "bowl", 17: "laptop", 18: "cell_phone", 19: "book",
            20: "keyboard", 21: "mouse", 22: "tv_monitor", 23: "bench",
            24: "umbrella", 25: "dog"}
COCO_NAMES_REV = {v: k for k, v in COCO_NAMES.items()}
SPLITS = ("train", "val")
ANN_FILES = {"train": "instances_train2017.json", "val": "instances_val2017.json"}


def load_coco(ann_dir):
    data = {}
    for split in SPLITS:
        p = os.path.join(ann_dir, ANN_FILES[split])
        with open(p) as f:
            data[split] = json.load(f)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", default="datasets/source-coco/coco2017/annotations")
    ap.add_argument("--out", default="reports/coco_coverage_analysis")
    ap.add_argument("--est-bytes", type=int, default=210000)
    args = ap.parse_args()

    data = load_coco(args.ann_dir)
    cat_id_to_name = {}
    for split in SPLITS:
        for c in data[split]["categories"]:
            cat_id_to_name.setdefault(c["id"], c["name"])

    req_coco_ids = {cid: name for cid, name in cat_id_to_name.items()
                    if name in COCO_NAMES}
    name_to_lv = {name: COCO_NAMES[name] for name in set(COCO_NAMES) if name in req_coco_ids.values()}

    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset_ref": "awsaf49/coco-2017-dataset",
        "annotation_files": {s: os.path.join(args.ann_dir, ANN_FILES[s]) for s in SPLITS},
        "est_image_size_bytes": args.est_bytes,
        "mapping": {str(lv): {"name": LV_NAMES[lv], "coco_name": n}
                    for n, lv in sorted(name_to_lv.items(), key=lambda kv: kv[1])},
        "not_in_coco": [LV_NAMES[5]],
        "categories": {},
        "union": {},
        "imbalance": {},
    }

    per_split_stat = {s: {"instances": collections.Counter(),
                          "images": collections.defaultdict(set)}
                      for s in SPLITS}

    for split in SPLITS:
        anns = data[split]["annotations"]
        for a in anns:
            name = cat_id_to_name.get(a["category_id"])
            if name not in name_to_lv:
                continue
            lv = name_to_lv[name]
            per_split_stat[split]["instances"][lv] += 1
            per_split_stat[split]["images"][lv].add(a["image_id"])

    for lv in sorted(set(COCO_NAMES.values())):
        cname = LV_NAMES[lv]
        cname_lv = LV_NAMES[lv]
        entry = {"lensvoice_id": lv, "lensvoice_name": cname_lv}
        inst_total, img_total = 0, set()
        for s in SPLITS:
            inst = per_split_stat[s]["instances"][lv]
            imgs = per_split_stat[s]["images"][lv]
            entry[f"{s}_instances"] = inst
            entry[f"{s}_images"] = len(imgs)
            inst_total += inst; img_total |= imgs
        entry["instances"] = inst_total
        entry["unique_images"] = len(img_total)
        report["categories"][str(lv)] = entry

    req_img_ids = {s: set() for s in SPLITS}
    multi_req = {s: 0 for s in SPLITS}
    for split in SPLITS:
        for a in data[split]["annotations"]:
            name = cat_id_to_name.get(a["category_id"])
            if name in name_to_lv:
                req_img_ids[split].add(a["image_id"])
    for split in SPLITS:
        imgid_ann = collections.defaultdict(set)
        for a in data[split]["annotations"]:
            name = cat_id_to_name.get(a["category_id"])
            if name in name_to_lv:
                imgid_ann[a["image_id"]].add(name_to_lv[name])
        n_multi = sum(1 for s in imgid_ann.values() if len(s) > 1)
        multi_req[split] = n_multi
        report["union"][f"{split}_required_images"] = len(req_img_ids[split])
        report["union"][f"{split}_multi_class_images"] = n_multi

    total_inst = sum(report["categories"][k]["instances"] for k in report["categories"])
    total_imgs = sum(report["union"][k] for k in ("train_required_images", "val_required_images"))
    report["union"]["total_required_images"] = total_imgs
    report["union"]["total_required_instances"] = total_inst
    report["union"]["train_est_download_mib"] = round(
        report["union"]["train_required_images"] * args.est_bytes / 2 ** 20, 1)
    report["union"]["val_est_download_mib"] = round(
        report["union"]["val_required_images"] * args.est_bytes / 2 ** 20, 1)
    report["union"]["total_est_download_mib"] = round(total_imgs * args.est_bytes / 2 ** 20, 1)
    report["union"]["avg_objects_per_required_image"] = round(
        total_inst / total_imgs, 2) if total_imgs else 0

    smallest = min(report["categories"].values(), key=lambda c: c["instances"])["instances"]
    for k, c in sorted(report["categories"].items(), key=lambda kv: kv[1]["instances"], reverse=True):
        report["imbalance"][c["lensvoice_name"]] = {
            "instances": c["instances"],
            "ratio_to_smallest": round(c["instances"] / smallest, 1) if smallest else None,
            "pct_of_total": round(100 * c["instances"] / total_inst, 1),
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(report, f, indent=2)

    md = []
    md.append("# COCO Coverage Analysis — LensVoice Unified Dataset (pre-download)\n")
    md.append(f"Source: `{report['dataset_ref']}` (27.6 GiB full package; images stored as individual"
              f" files → supports per-image selective download).\n")
    md.append(f"COCO category → LensVoice id mapping (by name, `{len(report['mapping'])}` matched).\n")
    for lv, m in sorted(report["mapping"].items(), key=lambda kv: int(kv[0])):
        md.append(f"- COCO `{m['coco_name']}` → LensVoice `{lv} {m['name']}`")
    if report["not_in_coco"]:
        md.append(f"- **NOT in COCO** (IDD-only): {', '.join(report['not_in_coco'])}\n")

    md.append("\n## Category statistics\n")
    md.append("| lv id | lensvoice class | instances | unique imgs | train imgs | val imgs |")
    md.append("|---|---|---|---|---|---|")
    for k, c in sorted(report["categories"].items(), key=lambda kv: int(kv[0])):
        md.append(f"| {c['lensvoice_id']} | {c['lensvoice_name']} | {c['instances']:,} |"
                  f" {c['unique_images']:,} | {c['train_images']:,} | {c['val_images']:,} |")

    u = report["union"]
    md.append("\n## Union statistics\n")
    md.append(f"- COCO images with ≥1 required class: train `{u['train_required_images']:,}`,"
              f" val `{u['val_required_images']:,}` → **total {u['total_required_images']:,}**")
    md.append(f"- Total required instances: **{u['total_required_instances']:,}**")
    md.append(f"- Multi-class images: train `{u['train_multi_class_images']:,}`,"
              f" val `{u['val_multi_class_images']:,}`")
    md.append(f"- Avg objects per required image: **{u['avg_objects_per_required_image']}**")
    md.append(f"- Estimated download size (@ {args.est_bytes/1024:.0f} KiB/image):"
              f" train `{u['train_est_download_mib']:,} MiB`,"
              f" val `{u['val_est_download_mib']:,} MiB`,"
              f" total `{u['total_est_download_mib']:,} MiB` "
              f"({u['total_est_download_mib']/1024:.1f} GiB) — *estimate; final confirmed at build*")

    md.append("\n## Class imbalance (COCO side only)\n")
    md.append("| lensvoice class | instances | ratio vs smallest | % of total |")
    md.append("|---|---|---|---|")
    for name, im in report["imbalance"].items():
        md.append(f"| {name} | {im['instances']:,} | {im['ratio_to_smallest']:,.0f}x | {im['pct_of_total']}% |")

    md.append("\n## Notes\n")
    md.append("- **No leakage:** COCO train/val official splits are disjoint; we preserve them "
              "as-is (each image appears in exactly one split).")
    md.append("- **Small objects:** no tiny-box filtering in this analysis; box size / tiny-object "
              "statistics are computed at build+audit time (per spec §13).")
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(md))
    print("wrote", args.out + ".json", "and", args.out + ".md")
    print("TOTAL required COCO images:", total_imgs, "instances:", total_inst)


if __name__ == "__main__":
    main()