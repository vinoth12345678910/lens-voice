"""
tools/build_unified.py — build datasets/lensvoice-unified from IDD V1 + selective COCO.

Merges two sources into one 26-class YOLO dataset with full provenance:

  source          ->  unified split          class handling
  IDD V1 train    ->  train                  ident (ids 0-5 already match)
  IDD V1 val      ->  val                    ident
  IDD V1 test     ->  test                   ident
  COCO train2017  ->  train                  remap by category NAME
  COCO val2017    ->  val                    remap by category NAME

Filenames are prefixed source_ (coco_/idd_) so every image's origin is visible;
COCO-origin val is therefore distinguishable from IDD-origin val. Tiny/small
objects are retained (never filtered). Official splits are preserved.

Outputs (per spec §11):
  images/{train,val,test}/  labels/{train,val,test}/
  manifests/{train,val,test}.csv   (per-instance provenance rows)
  reports/class_distribution.json   reports/source_distribution.json
  reports/dataset_audit.{json,md}   data.yaml   README.md
"""
import argparse, copy, json, os, shutil, sys

LV = {
    "person": 0, "car": 1, "motorcycle": 2, "bus": 3, "truck": 4,
    "autorickshaw": 5, "bicycle": 6, "chair": 7, "table": 8, "couch": 9,
    "bed": 10, "backpack": 11, "bag": 12, "suitcase": 13, "bottle": 14,
    "cup": 15, "bowl": 16, "laptop": 17, "cell_phone": 18, "book": 19,
    "keyboard": 20, "mouse": 21, "tv_monitor": 22, "bench": 23, "umbrella": 24,
    "dog": 25,
}
LV_NAMES = [k for k in sorted(LV, key=LV.get)]

# COCO category name -> LV class id (name-based remap). No autorickshaw in COCO.
COCO_MAP = {
    "person": 0, "car": 1, "motorcycle": 2, "bus": 3, "truck": 4,
    "bicycle": 6, "chair": 7, "dining table": 8, "couch": 9, "bed": 10,
    "backpack": 11, "handbag": 12, "suitcase": 13, "bottle": 14, "cup": 15,
    "bowl": 16, "laptop": 17, "cell phone": 18, "book": 19, "keyboard": 20,
    "mouse": 21, "tv": 22, "bench": 23, "umbrella": 24, "dog": 25,
}
# autoshi = 5 -> IDD-only (0 COCO images; reported in audit, never fabricated).

IDD_V1 = "datasets/cleaned/v1"
IDD_SPLITS = {"train": "train", "val": "val", "test": "test"}
COCO_ROOT = "datasets/source-coco/coco2017"
COCO_ANN = {
    "train": ("train2017", "annotations/instances_train2017.json"),
    "val": ("val2017", "annotations/instances_val2017.json"),
}
OUT = "datasets/lensvoice-unified"


def load_coco(ann_path):
    with open(ann_path) as f:
        ann = json.load(f)
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    imgs = {im["id"]: im for im in ann["images"]}
    by_img = {}
    for a in ann["annotations"]:
        cat_name = cats[a["category_id"]]
        lv = COCO_MAP.get(cat_name)
        if lv is None:
            continue  # category outside the 26-class taxonomy
        x, y, w, h = a["bbox"]
        by_img.setdefault(a["image_id"], []).append(
            (lv, x, y, w, h, a["id"], a["category_id"], cat_name))
    return imgs, by_img


def to_yolo(w, h, x, y, bw, bh):
    xc = max(0.0, min(1.0, (x + bw / 2) / w))
    yc = max(0.0, min(1.0, (y + bh / 2) / h))
    nw = max(0.0, min(1.0, bw / w))
    nh = max(0.0, min(1.0, bh / h))
    return xc, yc, nw, nh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="idd,coco", help="comma-separated: idd,coco")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--prune-file", default="datasets/proposals/idd_train_leak_prune.txt",
                    help="train stems to skip (near-dup vs val/test); 'none' to disable")
    ap.add_argument("--clean", action="store_true",
                    help="wipe --out first (removes stale files from prior builds)")
    args = ap.parse_args()
    sources = {s.strip() for s in args.sources.split(",")}

    if args.clean and os.path.isdir(args.out):
        print("cleaning", args.out)
        shutil.rmtree(args.out)

    prune = set()
    if args.prune_file and args.prune_file.lower() != "none":
        for l in open(args.prune_file):
            l = l.strip()
            if l:
                prune.add(l)
    print(f"prune-file={args.prune_file or 'none'} entries={len(prune)}")

    for sub in ("images", "labels", "manifests"):
        for sp in ("train", "val", "test"):
            os.makedirs(os.path.join(args.out, sub, sp), exist_ok=True)

    # aggregated stats
    cls_inst = {sp: {i: 0 for i in range(26)} for sp in ("train", "val", "test")}
    cls_img = {sp: {i: 0 for i in range(26)} for sp in ("train", "val", "test")}
    src_imgs = {sp: {} for sp in ("train", "val", "test")}
    tiny = {sp: {"<32px": 0, "<10px": 0, "total": 0} for sp in ("train", "val", "test")}
    n_img = {sp: 0 for sp in ("train", "val", "test")}
    empty_labels = {sp: 0 for sp in ("train", "val", "test")}
    pruned_train = 0
    problems = []

    # ---------------- IDD ----------------
    if "idd" in sources:
        import numpy as np
        from PIL import Image
        for sp, d in IDD_SPLITS.items():
            idir = os.path.join(IDD_V1, "images", d)
            ldir = os.path.join(IDD_V1, "labels", d)
            ids = sorted(os.listdir(idir))
            for fn in ids:
                if not fn.endswith(".jpg"):
                    continue
                src_img = os.path.join(idir, fn)
                src_lbl = os.path.join(ldir, os.path.splitext(fn)[0] + ".txt")
                base = "idd_" + fn
                stem = os.path.splitext(base)[0]
                if stem in prune:
                    pruned_train += 1
                    continue
                out_img = os.path.join(args.out, "images", sp, base)
                if not os.path.exists(src_lbl):
                    problems.append(f"IDD {sp} {fn}: missing label")
                    continue
                lines = [l for l in open(src_lbl).read().splitlines() if l.strip()]
                if not lines:
                    empty_labels[sp] += 1
                lbl_out = os.path.join(args.out, "labels", sp, os.path.splitext(base)[0] + ".txt")
                with open(lbl_out, "w") as f:
                    f.write("\n".join(lines) + ("\n" if lines else ""))
                shutil.copy2(src_img, out_img)
                with Image.open(src_img) as im:
                    W, H = im.size
                with open(os.path.join(args.out, "manifests", sp, os.path.splitext(base)[0] + ".csv"), "w") as f:
                    f.write("image_path,source_dataset,original_image_id,original_category_id,"
                            "original_category_name,lensvoice_class_id,lensvoice_class_name,split\n")
                    seen = set()
                    for ln in lines:
                        c, xc, yc, w, h = [float(t) for t in ln.split()]
                        ci = int(c)
                        f.write(f"{base},idd,{fn},{ci},{LV_NAMES[ci]},{ci},"
                                f"{LV_NAMES[ci]},{sp}\n")
                        cls_inst[sp][ci] += 1
                        seen.add(ci)
                        tiny[sp]["total"] += 1
                    prev = tiny[sp]["<32px"]
                    tiny[sp]["<32px"] = prev + sum(
                        1 for ln in lines if float(ln.split()[3]) < 32 / W
                        and float(ln.split()[4]) < 32 / H)
                    tiny[sp]["<10px"] = tiny[sp]["<32px"]
                    for lv in seen:
                        cls_img[sp][lv] += 1
                src_imgs[sp]["idd"] = src_imgs[sp].get("idd", 0) + 1
                n_img[sp] += 1
            print(f"idd {sp}: {len(ids)}")

    # ---------------- COCO ----------------
    if "coco" in sources:
        from PIL import Image
        for sp, (folder, ann_rel) in COCO_ANN.items():
            imgs, by_img = load_coco(os.path.join(COCO_ROOT, ann_rel))
            idir = os.path.join(COCO_ROOT, folder)
            cnt = 0
            for img_id in sorted(imgs):
                fn = f"{img_id:012d}.jpg"
                src_img = os.path.join(idir, fn)
                if not os.path.exists(src_img):
                    continue  # not in approved subset
                info = imgs[img_id]
                W, H = info["width"], info["height"]
                with Image.open(src_img) as im:
                    if im.size != (W, H):
                        problems.append(f"COCO {fn}: ann WxH {W}x{H} != img {im.size}")
                rows = by_img.get(img_id, [])
                if not rows:
                    problems.append(f"COCO {fn}: no labels")
                base = "coco_" + fn
                out_img = os.path.join(args.out, "images", sp, base)
                lbl = []
                for (lv, x, y, w, h, aid, ocat, oname) in rows:
                    xc, yc, nw, nh = to_yolo(W, H, x, y, w, h)
                    lbl.append(f"{lv} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
                with open(os.path.join(args.out, "labels", sp,
                                       os.path.splitext(base)[0] + ".txt"), "w") as f:
                    f.write("\n".join(lbl) + ("\n" if lbl else ""))
                with open(os.path.join(args.out, "manifests", sp,
                                       os.path.splitext(base)[0] + ".csv"), "w") as f:
                    f.write("image_path,source_dataset,original_image_id,original_category_id,"
                            "original_category_name,lensvoice_class_id,lensvoice_class_name,split\n")
                    seen = set()
                    for (lv, x, y, w, h, aid, ocat, oname) in rows:
                        xc, yc, nw, nh = to_yolo(W, H, x, y, w, h)
                        pxw, pxh = nw * W, nh * H
                        f.write(f"{base},coco,{fn},{ocat},{oname},{lv},"
                                f"{LV_NAMES[lv]},{sp}\n")
                        cls_inst[sp][lv] += 1
                        seen.add(lv)
                        tiny[sp]["total"] += 1
                        if pxw < 32 and pxh < 32:
                            tiny[sp]["<32px"] += 1
                        if pxw < 10 and pxh < 10:
                            tiny[sp]["<10px"] += 1
                    for lv in seen:
                        cls_img[sp][lv] += 1
                shutil.copy2(src_img, out_img)
                src_imgs[sp]["coco"] = src_imgs[sp].get("coco", 0) + 1
                n_img[sp] += 1
                cnt += 1
            print(f"coco {sp}: {cnt} images written")

    # ---------------- reports ----------------
    os.makedirs(os.path.join(args.out, "reports"), exist_ok=True)
    class_dist = {sp: [{"class_id": i, "name": LV_NAMES[i],
                        "images": cls_img[sp][i], "instances": cls_inst[sp][i]}
                        for i in range(26)] for sp in ("train", "val", "test")}
    with open(os.path.join(args.out, "reports", "class_distribution.json"), "w") as f:
        json.dump(class_dist, f, indent=2)
    with open(os.path.join(args.out, "reports", "source_distribution.json"), "w") as f:
        json.dump(src_imgs, f, indent=2)

    audit = {
        "splits": {
            sp: {
                "images": int(n_img[sp]),
                "instances": int(sum(cls_inst[sp].values())),
                "empty_label_images": empty_labels[sp],
                "leak_pruned_train_images": pruned_train if sp == "train" else 0,
                "sources": src_imgs[sp],
                "class_instances": cls_inst[sp],
                "class_images": cls_img[sp],
                "tiny": tiny[sp],
            } for sp in ("train", "val", "test")
        },
        "problems": problems[:50],
        "problem_count": len(problems),
        "taxonomy": LV,
    }

    with open(os.path.join(args.out, "reports", "dataset_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)
    with open(os.path.join(args.out, "reports", "dataset_audit.md"), "w") as f:
        f.write("# LensVoice-Unified Audit\n\n")
        for sp in ("train", "val", "test"):
            s = audit["splits"][sp]
            f.write(f"## {sp}\n- images: {s['images']}\n- instances: {s['instances']}\n")
            f.write(f"- sources: {s['sources']}\n")
            if sp == "train":
                f.write(f"- leak-pruned (near-dup vs val/test): "
                        f"{s['leak_pruned_train_images']} — see leakage_report.md\n")
            f.write(f"- tiny: {s['tiny']}\n")
            f.write("| class | name | images | instances |\n|---|---|---|---|\n")
            for r in class_dist[sp]:
                f.write(f"| {r['class_id']} | {r['name']} | {r['images']} | {r['instances']} |\n")
            f.write("\n")
        f.write("## Problems\n- count: {}\n".format(audit["problem_count"]))
        for p in audit["problems"]:
            f.write(f"- {p}\n")

    # data.yaml
    yml = ["# LensVoice-Unified — 26 classes (IDD V1 + selective COCO 2017)",
           "train: ./images/train", "val: ./images/val", "test: ./images/test",
           f"nc: {len(LV)}", "names:"]
    for i, n in enumerate(LV_NAMES):
        yml.append(f"  {i}: {n}")
    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        f.write("\n".join(yml) + "\n")

    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write("# LensVoice-Unified Dataset\n\n26-class merged dataset: "
                "IDD V1 (Indian roads) + selective COCO 2017 (general objects).\n\n"
                "- Splits preserve official sources: IDD train/val/test and "
                "COCO train2017/val2017 (COCO test2017 is unlabeled, excluded).\n"
                "- COCO-origin content is prefixed `coco_`, IDD `idd_`.\n"
                "- Tiny/small objects retained by design.\n"
                "- See reports/ for class/source distribution and audit.\n")

    print("done. reports:", os.path.join(args.out, "reports"))


if __name__ == "__main__":
    main()