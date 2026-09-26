# LensVoice model layer

## The 26-class detection taxonomy (LOCKED)

| id | class | id | class | id | class |
|----|-------|----|-------|----|-------|
| 0 | person | 9 | couch | 18 | cell_phone |
| 1 | car | 10 | bed | 19 | book |
| 2 | motorcycle | 11 | backpack | 20 | keyboard |
| 3 | bus | 12 | bag | 21 | mouse |
| 4 | truck | 13 | suitcase | 22 | tv_monitor |
| 5 | autorickshaw | 14 | bottle | 23 | bench |
| 6 | bicycle | 15 | cup | 24 | umbrella |
| 7 | chair | 16 | bowl | 25 | dog |
| 8 | table | 17 | laptop | | |

The single source of truth is **`configs/default.yaml`** (`class_names`). The
runtime never hardcodes ids in code — adapters map the model's own `names` map
back to this taxonomy.

## Interfaces (the integration contract)

Adapters convert framework outputs into `lensvoice.models.schemas`. Nothing
framework-specific leaks past this boundary.

| interface | method | returns |
|-----------|--------|---------|
| `VisionModel` | `predict(frame)` | `ModelOutput` → `DetectedObject[]` |
| `OCRModel` | `read(frame)` | `OCRResult[]` |
| `TrafficSignModel` | `detect(frame)` | `TrafficSignResult[]` |
| `DepthModel` | `estimate(frame)` | `DepthResult` (relative only) |
| `SceneModel` | `classify(frame)` | `SceneResult` |

Output types: `lensvoice/models/schemas.py` — plain dataclasses, NumPy only.

## Bounding-box contract

`BBox(x1, y1, x2, y2)` — absolute pixels, `x2 >= x1, y2 >= y1` (validated).
Helpers: `width`, `height`, `center_x/y`, `area`, `iou()`, `size_ratio_to()`.

## Depth contract

`DepthResult.depth` is **relative** `[0, 1]`, 1.0 = closest, and is explicitly
marked `calibrated: False`. The pipeline buckets it into
`very_near / near / medium / far` and never claims metres.

## Adapters

| slot | adapter | backend | status |
|------|---------|---------|--------|
| yolo | `yolo_adapter.YOLOAdapter` | Ultralytics YOLO | Wired; needs trained weight |
| ocr | `ocr_adapter.PaddleOCRAdapter` | PaddleOCR PP-OCRv4 | Wired; needs model |
| traffic_sign | `traffic_sign_adapter` (detector+classifier skeleton) | Ultralytics | Classifier TODO (Friend 1) |
| depth | `depth_adapter.DepthAdapter` | MiDaS (torch.hub) | Socket ready |
| scene | `scene_adapter.SceneAdapter` | transformers CLIP | Socket ready |

All real adapters **refuse to run** with an empty `weight` (explicit
`RuntimeError`) — safest when someone flips `enabled: true` without weights.

## Mock models

`lensvoice.models.mock_models` provides deterministic, weight-free stubs for
every slot so the full runtime runs in a CI box. They are pure functions of
frame id/dimensions (no randomness).

## Weights locations

`models/` at the repo root is the **weights dir** (often `*.pt`, gitignored,
transferred out-of-band). Paths are given to the pipeline only via
`configs/default.yaml → models.<slot>.weight` — never hardcoded.

## Factory

`lensvoice.models.create_models(settings)` returns one instance per slot
(real adapter if `enabled`, else mock). `Pipeline` uses only this factory.