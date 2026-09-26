# Dataset Audit Report

- data root: `datasets/cleaned/v1`
- total images: 44331
- total instances: 474356
- total flagged label issues: 105382

## train

- images: 31032 | labels: 31032 | instances: 332410
- empty label files: 1409 | images with a box: 29623
- pairing: images missing label = 0, labels missing image = 0

| class | instances | images |
|---|---|---|
| 0 animal | 134301 | 23900 |
| 1 autorickshaw | 65797 | 19143 |
| 2 bicycle | 74898 | 20795 |
| 3 bus | 13746 | 7830 |
| 4 car | 20432 | 11574 |
| 5 motorcycle | 23236 | 10481 |

### Label issues

| issue | count |
|---|---|
| invalid_class_id | 0 |
| negative_coords | 0 |
| zero_area | 0 |
| out_of_bounds | 0 |
| tiny_boxes_area_lt_1e4 | 18971 |
| small_side_lt_0_01 | 52593 |
| huge_side_gt_0_95 | 2103 |
| duplicate_boxes_within_file | 0 |
| duplicate_dupe_lines_within_file | 0 |

### Box area buckets

| bucket | count |
|---|---|
| 1e-4-4e-5 | 14650 |
| 4e-5-1e-3 | 56772 |
| <1%  | 176878 |
| <10% | 64483 |
| <4e-5 (~<16px area @640) | 4321 |
| >=10% | 15306 |

## val

- images: 8866 | labels: 8866 | instances: 95396
- empty label files: 403 | images with a box: 8463
- pairing: images missing label = 0, labels missing image = 0

| class | instances | images |
|---|---|---|
| 0 animal | 38658 | 6856 |
| 1 autorickshaw | 18889 | 5439 |
| 2 bicycle | 21580 | 5944 |
| 3 bus | 3756 | 2195 |
| 4 car | 5787 | 3319 |
| 5 motorcycle | 6726 | 3006 |

### Label issues

| issue | count |
|---|---|
| invalid_class_id | 0 |
| negative_coords | 0 |
| zero_area | 0 |
| out_of_bounds | 0 |
| tiny_boxes_area_lt_1e4 | 5620 |
| small_side_lt_0_01 | 15455 |
| huge_side_gt_0_95 | 582 |
| duplicate_boxes_within_file | 0 |
| duplicate_dupe_lines_within_file | 0 |

### Box area buckets

| bucket | count |
|---|---|
| 1e-4-4e-5 | 4245 |
| 4e-5-1e-3 | 16650 |
| <1%  | 50524 |
| <10% | 18243 |
| <4e-5 (~<16px area @640) | 1375 |
| >=10% | 4359 |

## test

- images: 4433 | labels: 4433 | instances: 46550
- empty label files: 186 | images with a box: 4247
- pairing: images missing label = 0, labels missing image = 0

| class | instances | images |
|---|---|---|
| 0 animal | 18922 | 3437 |
| 1 autorickshaw | 9115 | 2655 |
| 2 bicycle | 10484 | 2964 |
| 3 bus | 1857 | 1093 |
| 4 car | 2875 | 1611 |
| 5 motorcycle | 3297 | 1501 |

### Label issues

| issue | count |
|---|---|
| invalid_class_id | 0 |
| negative_coords | 0 |
| zero_area | 0 |
| out_of_bounds | 0 |
| tiny_boxes_area_lt_1e4 | 2642 |
| small_side_lt_0_01 | 7139 |
| huge_side_gt_0_95 | 277 |
| duplicate_boxes_within_file | 0 |
| duplicate_dupe_lines_within_file | 0 |

### Box area buckets

| bucket | count |
|---|---|
| 1e-4-4e-5 | 2031 |
| 4e-5-1e-3 | 7620 |
| <1%  | 24900 |
| <10% | 9238 |
| <4e-5 (~<16px area @640) | 611 |
| >=10% | 2150 |
