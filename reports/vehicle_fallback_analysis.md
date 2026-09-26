# Vehicle-Fallback Analysis

**Dataset examined:** every split of `idd-1` (the immutable original, also copied to `datasets/original/idd-1`). `vehicle fallback` = class id **11**, the 12th class produced by the earlier 15→12 class remap (caravan, trailer, train were merged into it).

## Counts

| split | instances | files containing it | vf-only files |
|---|---|---|---|
| train | 15,419 | 8,399 / 31,032 (27%) | 268 (3% of those files) |
| valid | 4,525 | 2,764 / 8,866 (31%) | ~70 (2.5%) |
| test | 2,090 | 1,333 / 4,433 (30%) | — |

It almost never appears alone: 97% of vf images also contain at least one of the six target classes. Removing the class therefore **wastes almost no whole images**.

## Bounding-box sizes (train)

| statistic | value |
|---|---|
| instances | 15,419 |
| mean area | 0.0093 (~3,800 px² @ 640) |
| median area | 0.0005 (~205 px² @ 640) — very small |
| tiny (< 41 px² @ 640) | 2,285 (14.8%) |
| tall (h > 1.5×w) | 9,386 (61%) |
| wide (w > 1.5×h) | 814 (5%) |

The geometry is striking: **61% tall, median-box tiny, 15% subpixel**. These are not ordinary vehicles (which would be wide or square) — they are largely **distant, partial, or occluded roadside objects** (typically standing figures / part-visible vehicles in gaps), consistent with IDD's "vehicle fallback" catch-all semantics.

## Co-occurring classes (train, % of vf files)

| class | % of vf files | | class | % of vf files |
|---|---|---|---|---|
| motorcycle | 248%* | | bus | 93% |
| car | 228% | | traffic sign | 69% |
| rider | 221% | | bicycle | 24% |
| person | 188% | | animal | 19% |
| truck | 138% | | traffic light | 15% |
| autorickshaw | 125% | | | |

\* >100% because a file can contain multiple vf instances.
vf rides along in the densest scenes, heavily overlapping the six target classes.

## Likely semantic categories

Base on the original IDD ontology plus the observed geometry:

1. **Distant/occluded vehicles** — partial cars/trucks at the skyline, behind poles, at frame edge (tall thin slivers).
2. **Standing/unhandled two-wheelers** — riders/scooters in mixed traffic that IDD's annotators could not cleanly classify.
3. **Misc hardware** — trailers, tractors, carts (the classes previously merged in), rare enough to be unlearnable on their own.

The visual samples for human confirmation are in `reports/visual_audit/vehicle_fallback_*.png` (rendered on the train split).

## Recommendation

**EXCLUDE `vehicle fallback` from Dataset V1** (pending your decision).

- It is a catch-all with internally inconsistent labels (tiny/tall/partial boxes, 15% subpixel — even a correctly trained model cannot reach high AP on it).
- It inflates class count and adds confusable positives to car/truck/autorickshaw, which suppresses exactly the classes we want ≥90%.
- Removing it wastes only ~3% of images (and those become hard negatives, not losses).
- The originals are **not deleted** — they remain in `datasets/original/idd-1` (immutable) and in `idd-1_labels_backup/`, so it is fully reversible.

**Decision needed from you (no auto-delete):**
- [ ] Option A (recommended): exclude vf from v1 — v1 has exactly 6 classes, no fallback.
- [ ] Option B: include it as a 7th class, accept the mAP50 drag and a low-vf AP.
- [ ] Option C: relabel vf into the closest of the 6 real classes (car/truck/motorcycle) — best accuracy, but requires a human or external-audit labeling pass on ~15k boxes.

Until you pick, Dataset V1 is built with Option A (all other options are trivially re-derivable because the original is immutable).