# Final Model Report (template — populated after the best model is locked)

## Candidate model
- model: TBD (expect yolo26l on V1, or a targeted-fix variant)
- input: 640x640 (native; do not upscale without hi-res source)
- classes: `[{0: person}, {1: car}, {2: motorcycle}, {3: bus}, {4: truck}, {5: autorickshaw}]`

## Accuracy (val)
| metric | value |
|---|---|
| mAP50 | TBD |
| mAP50-95 | TBD |
| precision | TBD |
| recall | TBD |
| per-class AP50 | TBD |

## Deployment (LensVoice compatible output)
```json
{"objects": [{"label": "person", "confidence": 0.94, "bbox": [100, 120, 300, 500]}]}
```
- bbox = `[x1, y1, x2, y2]` in input pixels. YOLO26 head is NMS-free: 300
  decoded rows, client applies only a confidence filter (see
  `training/export_model.py` + `training/test_inference.py`).
- The pipeline derives left/center/right from bboxes downstream; the model does
  NOT output distance or direction.
- Export formats: onnx/tflite/coreml/torchscript via `training/export_model.py`.

## Benchmark (target hardware)
| metric | value |
|---|---|
| model load time | TBD |
| inference latency | TBD |
| FPS (single image / webcam) | TBD |
| CPU / GPU / RAM / VRAM | TBD |
| model size | TBD |

## Where the best model lives
`models/best/` — copied from `experiments/<best>/weights/best.pt` + exported
formats, once the benchmark candidate is locked.