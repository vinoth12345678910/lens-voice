"""
ALREADY APPLIED — do not run this again.

The labels in idd-1/ are the post-remap 12-class version. This file is kept
only as the record of how the mapping was done.

Running it a second time CORRUPTS the dataset: the mapping is not idempotent,
so a second pass would send motorcycle (5) into vehicle fallback, person (6)
into motorcycle, rider (7) into person, and so on down the list. The guard in
__main__ below refuses to run if the labels already look remapped, but do not
rely on it — just leave this alone.

One-time remap: merge near-empty classes (caravan, trailer, train — 98/14/39
instances in the train split, statistically unlearnable and dragging down mAP)
into the existing 'vehicle fallback' catch-all class, and renumber the
remaining 12 classes contiguously. Run once against idd-1/{train,valid,test}/labels.
Originals are backed up in idd-1_labels_backup/ before this runs.
"""
import os

OLD_TO_NEW = {
    0: 0,   # animal
    1: 1,   # autorickshaw
    2: 2,   # bicycle
    3: 3,   # bus
    4: 4,   # car
    5: 11,  # caravan -> vehicle fallback
    6: 5,   # motorcycle
    7: 6,   # person
    8: 7,   # rider
    9: 8,   # traffic light
    10: 9,  # traffic sign
    11: 11, # trailer -> vehicle fallback
    12: 11, # train -> vehicle fallback
    13: 10, # truck
    14: 11, # vehicle fallback
}

NEW_NAMES = [
    'animal', 'autorickshaw', 'bicycle', 'bus', 'car', 'motorcycle',
    'person', 'rider', 'traffic light', 'traffic sign', 'truck',
    'vehicle fallback',
]

def remap_dir(labels_dir):
    n_files = 0
    n_lines = 0
    for fn in os.listdir(labels_dir):
        if not fn.endswith(".txt"):
            continue
        path = os.path.join(labels_dir, fn)
        with open(path) as f:
            lines = f.readlines()
        out_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            old_cls = int(parts[0])
            new_cls = OLD_TO_NEW[old_cls]
            parts[0] = str(new_cls)
            out_lines.append(" ".join(parts))
            n_lines += 1
        with open(path, "w") as f:
            f.write("\n".join(out_lines) + ("\n" if out_lines else ""))
        n_files += 1
    return n_files, n_lines

def already_remapped(base):
    """True if no label file anywhere uses a class id above 11."""
    for split in ["train", "valid", "test"]:
        d = os.path.join(base, split, "labels")
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith(".txt"):
                continue
            with open(os.path.join(d, fn)) as f:
                for line in f:
                    line = line.strip()
                    if line and int(line.split()[0]) > 11:
                        return False
    return True


if __name__ == "__main__":
    import sys

    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "idd-1")

    if already_remapped(base) and "--force" not in sys.argv:
        sys.exit(
            "Refusing to run: every label in idd-1/ is already within 0-11, so the\n"
            "remap has already been applied. Running it again would corrupt the\n"
            "dataset (motorcycle -> vehicle fallback, person -> motorcycle, ...).\n"
            "Pass --force only if you have genuinely restored the original 15-class\n"
            "labels from idd-1_labels_backup/ first."
        )

    for split in ["train", "valid", "test"]:
        d = os.path.join(base, split, "labels")
        nf, nl = remap_dir(d)
        print(f"{split}: remapped {nl} labels across {nf} files")

    # invalidate Ultralytics label caches so they get rebuilt from new labels
    for split in ["train", "valid"]:
        cache = os.path.join(base, split, "labels.cache")
        if os.path.exists(cache):
            os.remove(cache)
            print(f"removed stale cache: {cache}")

    print("New class list:", NEW_NAMES)
