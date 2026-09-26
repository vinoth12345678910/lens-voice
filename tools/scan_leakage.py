"""
tools/scan_leakage.py — full near-duplicate scan of IDD train frames vs val/test.

Checks every IDD-origin TRAIN image against every IDD-origin VAL+TEST image
(cross-split temporal-frame leakage inherited from the source's image-wise
split). Uses 64-bit dHash bucket-indexed by a 16-bit prefix so each train image
only compares against true near-candidates, then confirms candidates with a
64x64 grayscale MSE (threshold 350) to avoid pruning merely-similar frames.

Outputs (writes):
  datasets/proposals/idd_train_leak_prune.txt   train stems to drop from the
                                                  unified TRAIN split (val/test
                                                  untouched, still the eval truth)
  datasets/lensvoice-unified/reports/leakage_report.json|md

Run: venv/bin/python tools/scan_leakage.py
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

UNIFIED = "datasets/lensvoice-unified"
MSE_THR = 350.0
HAM_THR = 10


def dhash(path, size=8):
    a = np.asarray(Image.open(path).convert("L").resize((size + 1, size)),
                   dtype=np.uint8)
    bits = (a[:, :-1] > a[:, 1:]).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def mse64(path_a, path_b):
    x = np.asarray(Image.open(path_a).convert("L").resize((64, 64)), dtype=np.float32)
    y = np.asarray(Image.open(path_b).convert("L").resize((64, 64)), dtype=np.float32)
    return float(np.mean((x - y) ** 2))


def list_idd(root, split):
    d = os.path.join(root, "images", split)
    return [os.path.join(d, f) for f in os.listdir(d)
            if f.endswith(".jpg") and f.startswith("idd_")]


def main():
    train = list_idd(UNIFIED, "train")
    evals = list_idd(UNIFIED, "val") + list_idd(UNIFIED, "test")
    print(f"train={len(train)}  val+test={len(evals)}")

    with ThreadPoolExecutor(max_workers=8) as ex:
        train_h = list(ex.map(dhash, train))

    # bucket the eval side by 16-bit prefix
    buckets = {}
    for p in evals:
        h = dhash(p)
        buckets.setdefault(h >> 48, []).append((h, p))

    flags = []  # (train_path, hamming, mse, eval_path, eval_split)
    examined = 0
    for tp, th in zip(train, train_h):
        pfx = th >> 48
        for q in ((pfx - 1) & 0xFFFF, pfx, (pfx + 1) & 0xFFFF):
            for eh, ep in buckets.get(q, []):
                examined += 1
                d = (th ^ eh).bit_count()
                if d <= HAM_THR:
                    m = mse64(tp, ep)
                    if m < MSE_THR:
                        split = "val" if "/val/" in ep else "test"
                        flags.append((tp, d, m, ep, split))

    # keep the best (lowest mse) record per train image
    by_train = {}
    for tp, d, m, ep, split in flags:
        prev = by_train.get(tp)
        if prev is None or m < prev[0]:
            by_train[tp] = (m, ep, split, d)
    prune = sorted(by_train)

    print(f"buckets={len(buckets)} candidate-pairs-examined={examined}")
    print(f"confirmed leaked train images: {len(prune)}"
          f" of {len(train)} ({(100*len(prune)/len(train)):.2f}%)")
    for tp in prune[:15]:
        m, ep, split, d = by_train[tp]
        print(f"  {os.path.basename(tp)}  ~ {os.path.basename(ep)} [{split}] ham={d} mse={m:.0f}")

    os.makedirs("datasets/proposals", exist_ok=True)
    with open("datasets/proposals/idd_train_leak_prune.txt", "w") as f:
        for tp in prune:
            f.write(os.path.splitext(os.path.basename(tp))[0] + "\n")

    report = {
        "method": "dHash(64-bit, prefix-bucketed) + 64x64 MSE confirm",
        "thresholds": {"hamming": HAM_THR, "mse": MSE_THR},
        "train_idd_images": len(train),
        "val_plus_test_idd_images": len(evals),
        "leaked_train_images": len(prune),
        "leak_rate_pct": round(100 * len(prune) / len(train), 2),
        "prune_file": "datasets/proposals/idd_train_leak_prune.txt",
        "examples": [{"train": os.path.basename(tp).split("_rf")[0],
                      "eval": os.path.basename(by_train[tp][1]),
                      "eval_split": by_train[tp][2],
                      "hamming": by_train[tp][3],
                      "mse": round(by_train[tp][0], 1)}
                     for tp in prune[:20]],
    }
    rdir = os.path.join(UNIFIED, "reports")
    with open(os.path.join(rdir, "leakage_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(rdir, "leakage_report.md"), "w") as f:
        f.write("# Leakage Report\n\n")
        f.write(f"- IDD train images scanned: {len(train)}\n")
        f.write(f"- IDD val+test index: {len(evals)}\n")
        f.write(f"- **Leaked train frames pruned: {len(prune)} "
                f"({report['leak_rate_pct']}%)**\n")
        f.write("- Val/test kept untouched (eval truth).\n")


if __name__ == "__main__":
    main()