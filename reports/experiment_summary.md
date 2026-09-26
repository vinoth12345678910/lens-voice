# Experiment Summary (living document)

| exp | model | imgsz | epochs | data | mAP50 | mAP50-95 | P | R | GPU | status |
|---|---|---|---|---|---|---|---|---|---|---|
| exp001 | yolo26n | 640 | 75 | v1 (6cls) | — | — | — | — | pending | **not run - awaiting GPU box** |

Verdict chain (gate): nothing scales up unless the previous verdict is GREEN.

- Prior 12-class run for reference (the baseline to beat): `results.csv` at repo
  root, 150 epochs, yolo26n/640 → mAP50 0.153, R 0.167. This polluted the
  dataset's signal with animal/bicycle/traffic-light/traffic-sign/vehicle-fallback.
- V1 removes that and merges rider→person; extra `reports/experiment_summary.md`
  rows will be appended as each exp finishes.