# COCO Coverage Analysis — LensVoice Unified Dataset (pre-download)

Source: `awsaf49/coco-2017-dataset` (27.6 GiB full package; images stored as individual files → supports per-image selective download).

COCO category → LensVoice id mapping (by name, `25` matched).

- COCO `person` → LensVoice `0 person`
- COCO `car` → LensVoice `1 car`
- COCO `motorcycle` → LensVoice `2 motorcycle`
- COCO `bus` → LensVoice `3 bus`
- COCO `truck` → LensVoice `4 truck`
- COCO `bicycle` → LensVoice `6 bicycle`
- COCO `chair` → LensVoice `7 chair`
- COCO `dining table` → LensVoice `8 table`
- COCO `couch` → LensVoice `9 couch`
- COCO `bed` → LensVoice `10 bed`
- COCO `backpack` → LensVoice `11 backpack`
- COCO `handbag` → LensVoice `12 bag`
- COCO `suitcase` → LensVoice `13 suitcase`
- COCO `bottle` → LensVoice `14 bottle`
- COCO `cup` → LensVoice `15 cup`
- COCO `bowl` → LensVoice `16 bowl`
- COCO `laptop` → LensVoice `17 laptop`
- COCO `cell phone` → LensVoice `18 cell_phone`
- COCO `book` → LensVoice `19 book`
- COCO `keyboard` → LensVoice `20 keyboard`
- COCO `mouse` → LensVoice `21 mouse`
- COCO `tv` → LensVoice `22 tv_monitor`
- COCO `bench` → LensVoice `23 bench`
- COCO `umbrella` → LensVoice `24 umbrella`
- COCO `dog` → LensVoice `25 dog`
- **NOT in COCO** (IDD-only): autorickshaw


## Category statistics

| lv id | lensvoice class | instances | unique imgs | train imgs | val imgs |
|---|---|---|---|---|---|
| 0 | person | 273,469 | 66,808 | 64,115 | 2,693 |
| 1 | car | 45,799 | 12,786 | 12,251 | 535 |
| 2 | motorcycle | 9,096 | 3,661 | 3,502 | 159 |
| 3 | bus | 6,354 | 4,141 | 3,952 | 189 |
| 4 | truck | 10,388 | 6,377 | 6,127 | 250 |
| 6 | bicycle | 7,429 | 3,401 | 3,252 | 149 |
| 7 | chair | 40,282 | 13,354 | 12,774 | 580 |
| 8 | table | 16,411 | 12,338 | 11,837 | 501 |
| 9 | couch | 6,040 | 4,618 | 4,423 | 195 |
| 10 | bed | 4,355 | 3,831 | 3,682 | 149 |
| 11 | backpack | 9,091 | 5,756 | 5,528 | 228 |
| 12 | bag | 12,894 | 7,133 | 6,841 | 292 |
| 13 | suitcase | 6,495 | 2,507 | 2,402 | 105 |
| 14 | bottle | 25,367 | 8,880 | 8,501 | 379 |
| 15 | cup | 21,549 | 9,579 | 9,189 | 390 |
| 16 | bowl | 14,984 | 7,425 | 7,111 | 314 |
| 17 | laptop | 5,201 | 3,707 | 3,524 | 183 |
| 18 | cell_phone | 6,696 | 5,017 | 4,803 | 214 |
| 19 | book | 25,876 | 5,562 | 5,332 | 230 |
| 20 | keyboard | 3,008 | 2,221 | 2,115 | 106 |
| 21 | mouse | 2,368 | 1,964 | 1,876 | 88 |
| 22 | tv_monitor | 6,093 | 4,768 | 4,561 | 207 |
| 23 | bench | 10,251 | 5,805 | 5,570 | 235 |
| 24 | umbrella | 11,844 | 4,142 | 3,968 | 174 |
| 25 | dog | 5,726 | 4,562 | 4,385 | 177 |

## Union statistics

- COCO images with ≥1 required class: train `90,813`, val `3,847` → **total 94,660**
- Total required instances: **587,066**
- Multi-class images: train `53,032`, val `2,300`
- Avg objects per required image: **6.2**
- Estimated download size (@ 205 KiB/image): train `18,187.3 MiB`, val `770.4 MiB`, total `18,957.7 MiB` (18.5 GiB) — *estimate; final confirmed at build*

## Class imbalance (COCO side only)

| lensvoice class | instances | ratio vs smallest | % of total |
|---|---|---|---|
| person | 273,469 | 116x | 46.6% |
| car | 45,799 | 19x | 7.8% |
| chair | 40,282 | 17x | 6.9% |
| book | 25,876 | 11x | 4.4% |
| bottle | 25,367 | 11x | 4.3% |
| cup | 21,549 | 9x | 3.7% |
| table | 16,411 | 7x | 2.8% |
| bowl | 14,984 | 6x | 2.6% |
| bag | 12,894 | 5x | 2.2% |
| umbrella | 11,844 | 5x | 2.0% |
| truck | 10,388 | 4x | 1.8% |
| bench | 10,251 | 4x | 1.7% |
| motorcycle | 9,096 | 4x | 1.5% |
| backpack | 9,091 | 4x | 1.5% |
| bicycle | 7,429 | 3x | 1.3% |
| cell_phone | 6,696 | 3x | 1.1% |
| suitcase | 6,495 | 3x | 1.1% |
| bus | 6,354 | 3x | 1.1% |
| tv_monitor | 6,093 | 3x | 1.0% |
| couch | 6,040 | 3x | 1.0% |
| dog | 5,726 | 2x | 1.0% |
| laptop | 5,201 | 2x | 0.9% |
| bed | 4,355 | 2x | 0.7% |
| keyboard | 3,008 | 1x | 0.5% |
| mouse | 2,368 | 1x | 0.4% |

## Notes

- **No leakage:** COCO train/val official splits are disjoint; we preserve them as-is (each image appears in exactly one split).
- **Small objects:** no tiny-box filtering in this analysis; box size / tiny-object statistics are computed at build+audit time (per spec §13).