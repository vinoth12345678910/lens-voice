# Dataset Audit Report

- data root: `datasets/original/idd-1`
- total images: 44331
- total instances: 524841
- total flagged label issues: 126551

## train

- images: 31032 | labels: 31032 | instances: 367706
- empty label files: 62 | images with a box: 30970
- pairing: images missing label = 0, labels missing image = 0

| class | instances | images |
|---|---|---|
| 0 animal | 4644 | 1564 |
| 1 autorickshaw | 23236 | 10481 |
| 2 bicycle | 2346 | 2025 |
| 3 bus | 13747 | 7830 |
| 4 car | 65798 | 19143 |
| 5 motorcycle | 74898 | 20795 |
| 6 person | 63716 | 15825 |
| 7 rider | 70588 | 18562 |
| 8 traffic light | 2674 | 1245 |
| 9 traffic sign | 10207 | 5763 |
| 10 truck | 20433 | 11574 |
| 11 vehicle fallback | 15419 | 8399 |

### Label issues

| issue | count |
|---|---|
| invalid_class_id | 0 |
| negative_coords | 0 |
| zero_area | 0 |
| out_of_bounds | 0 |
| tiny_boxes_area_lt_1e4 | 24047 |
| small_side_lt_0_01 | 62043 |
| huge_side_gt_0_95 | 2135 |
| duplicate_boxes_within_file | 6 |
| duplicate_dupe_lines_within_file | 6 |

### Box area buckets

| bucket | count |
|---|---|
| 1e-4-4e-5 | 18146 |
| 4e-5-1e-3 | 66852 |
| <1%  | 193595 |
| <10% | 67589 |
| <4e-5 (~<16px area @640) | 5901 |
| >=10% | 15623 |

## valid

- images: 8866 | labels: 8866 | instances: 105583
- empty label files: 16 | images with a box: 8850
- pairing: images missing label = 0, labels missing image = 0

| class | instances | images |
|---|---|---|
| 0 animal | 1320 | 460 |
| 1 autorickshaw | 6726 | 3006 |
| 2 bicycle | 640 | 550 |
| 3 bus | 3756 | 2195 |
| 4 car | 18889 | 5439 |
| 5 motorcycle | 21580 | 5944 |
| 6 person | 18255 | 4589 |
| 7 rider | 20403 | 5307 |
| 8 traffic light | 721 | 332 |
| 9 traffic sign | 2981 | 1737 |
| 10 truck | 5787 | 3319 |
| 11 vehicle fallback | 4525 | 2420 |

### Label issues

| issue | count |
|---|---|
| invalid_class_id | 0 |
| negative_coords | 0 |
| zero_area | 0 |
| out_of_bounds | 0 |
| tiny_boxes_area_lt_1e4 | 7191 |
| small_side_lt_0_01 | 18273 |
| huge_side_gt_0_95 | 594 |
| duplicate_boxes_within_file | 0 |
| duplicate_dupe_lines_within_file | 0 |

### Box area buckets

| bucket | count |
|---|---|
| 1e-4-4e-5 | 5335 |
| 4e-5-1e-3 | 19604 |
| <1%  | 55205 |
| <10% | 19114 |
| <4e-5 (~<16px area @640) | 1856 |
| >=10% | 4469 |

## test

- images: 4433 | labels: 4433 | instances: 51552
- empty label files: 9 | images with a box: 4424
- pairing: images missing label = 0, labels missing image = 0

| class | instances | images |
|---|---|---|
| 0 animal | 718 | 237 |
| 1 autorickshaw | 3297 | 1501 |
| 2 bicycle | 340 | 301 |
| 3 bus | 1857 | 1093 |
| 4 car | 9115 | 2655 |
| 5 motorcycle | 10484 | 2964 |
| 6 person | 9123 | 2304 |
| 7 rider | 9799 | 2656 |
| 8 traffic light | 355 | 157 |
| 9 traffic sign | 1499 | 782 |
| 10 truck | 2875 | 1611 |
| 11 vehicle fallback | 2090 | 1174 |

### Label issues

| issue | count |
|---|---|
| invalid_class_id | 0 |
| negative_coords | 0 |
| zero_area | 0 |
| out_of_bounds | 0 |
| tiny_boxes_area_lt_1e4 | 3428 |
| small_side_lt_0_01 | 8546 |
| huge_side_gt_0_95 | 282 |
| duplicate_boxes_within_file | 0 |
| duplicate_dupe_lines_within_file | 0 |

### Box area buckets

| bucket | count |
|---|---|
| 1e-4-4e-5 | 2539 |
| 4e-5-1e-3 | 9008 |
| <1%  | 27244 |
| <10% | 9670 |
| <4e-5 (~<16px area @640) | 889 |
| >=10% | 2202 |
