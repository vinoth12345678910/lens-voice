# COCO Sampling Proposal — LensVoice Unified Dataset

Generated 2026-09-26T14:48:42. Global caps: train `30,000`, val `3,000` images. Selection: multi-class-first, then per-class top-up; deterministic.


## Expected retained (COCO side only)

| split | images | instances | est. size |
|---|---|---|---|
| train | 14,943 | 142,459 | 2,992.7 MiB |
| val | 1,521 | 13,046 | 304.6 MiB |

## Per-class (selected images / instances), train | val

| lv id | class | tag | train imgs | train inst | val imgs | val inst | quota train |
|---|---|---|---|---|---|---|---|
| 0 | person | road | 7,000 | 37,046 | 700 | 3,648 | 7,000 |
| 1 | car | road | 4,000 | 14,725 | 383 | 1,369 | 4,000 |
| 2 | motorcycle | road | 1,479 | 3,457 | 106 | 255 | 1,800 |
| 3 | bus | road | 1,500 | 2,255 | 140 | 225 | 1,500 |
| 4 | truck | road | 2,500 | 4,112 | 191 | 319 | 2,500 |
| 6 | bicycle | road | 1,679 | 3,728 | 109 | 216 | 4,000 |
| 7 | chair | general | 3,000 | 9,388 | 300 | 1,006 | 3,000 |
| 8 | table | general | 3,200 | 4,817 | 320 | 484 | 3,200 |
| 9 | couch | general | 1,700 | 2,167 | 143 | 194 | 1,700 |
| 10 | bed | general | 1,500 | 1,696 | 90 | 97 | 1,500 |
| 11 | backpack | general | 2,099 | 3,334 | 153 | 252 | 2,100 |
| 12 | bag | general | 2,200 | 4,259 | 210 | 409 | 2,200 |
| 13 | suitcase | general | 1,154 | 2,742 | 80 | 235 | 1,300 |
| 14 | bottle | general | 2,600 | 7,233 | 260 | 727 | 2,600 |
| 15 | cup | general | 2,700 | 7,122 | 270 | 674 | 2,700 |
| 16 | bowl | general | 2,100 | 4,416 | 210 | 412 | 2,100 |
| 17 | laptop | general | 1,600 | 2,385 | 160 | 202 | 1,600 |
| 18 | cell_phone | general | 1,900 | 2,678 | 135 | 168 | 1,900 |
| 19 | book | general | 2,200 | 10,312 | 182 | 1,010 | 2,200 |
| 20 | keyboard | general | 1,100 | 1,563 | 105 | 151 | 1,100 |
| 21 | mouse | general | 950 | 1,151 | 87 | 105 | 950 |
| 22 | tv_monitor | general | 1,900 | 2,577 | 175 | 241 | 1,900 |
| 23 | bench | general | 2,100 | 3,537 | 146 | 248 | 2,100 |
| 24 | umbrella | general | 1,249 | 3,382 | 107 | 240 | 1,600 |
| 25 | dog | general | 1,900 | 2,377 | 125 | 159 | 1,900 |

## Unified dataset (IDD + selected COCO)

| split | IDD imgs | COCO imgs | total imgs | total instances | est disk |
|---|---|---|---|---|---|
| train | 31,032 | 14,943 | 45,975 | 474,869 | 4.56 GiB |
| val | 8,866 | 1,521 | 10,387 | 108,442 | 0.77 GiB |
| test | 4,433 | 0 | 4,433 | — | 0.24 GiB |

## Notes

- No image downloads performed for this analysis; COCO avg size estimated at 205 KiB/image and will be confirmed at build time.
- Autorickshaw (id 5) is IDD-only; COCO contributes 0 instances to it.
- Val is the official COCO val2017 (disjoint from train); COCO test2017 is unlabeled → excluded; unified test = IDD test only (LensVoice-Val later).