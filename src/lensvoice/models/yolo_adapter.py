"""YOLOAdapter — bridges a trained Ultralytics unified YOLO detector into the
pipeline's VisionModel interface.

No Ultralytics types escape this class: output is always `ModelOutput` /
`DetectedObject` (lensvoice.models.schemas). Weights path is configurable
(ModelConfig.weight) — never assumed from a filename.

Class names come from the model's own `names` map when available and fall
back to the config taxonomy otherwise.
"""
from __future__ import annotations

from typing import Dict

from lensvoice.config.settings import ModelConfig
from lensvoice.models.interfaces import VisionModel
from lensvoice.models.schemas import BBox, DetectedObject, Frame, ModelOutput


class YOLOAdapter(VisionModel):
    def __init__(self, cfg: ModelConfig, class_names: Dict[int, str] | None = None):
        self.cfg = cfg
        self._class_names = class_names or {}
        self._model = None
        self._names = None

    # -- lazy backend load ------------------------------------------------
    def _ensure_model(self):
        if self._model is not None:
            return self._model
        if not self.cfg.weight:
            raise RuntimeError(
                "YOLOAdapter is enabled but 'models.yolo.weight' is unset. "
                "Point it at a trained weight file (Friend 1 artifact).")
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("ultralytics not installed; cannot load YOLO weights") from e
        self._model = YOLO(self.cfg.weight)
        self._names = getattr(self._model, "names", None) or self._class_names
        return self._model

    # -- VisionModel ------------------------------------------------------
    def predict(self, frame: Frame) -> ModelOutput:
        model = self._ensure_model()
        results = model.predict(
            frame.image,
            conf=self.cfg.confidence,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device or None,
            verbose=False,
        )
        detections = []
        names = self._names
        for res in results:
            if res.boxes is None or len(res.boxes) == 0:
                continue
            for box in res.boxes:
                xyxy = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if conf < self.cfg.confidence:
                    continue
                try:
                    bbox = BBox(xyxy[0], xyxy[1], xyxy[2], xyxy[3])
                except ValueError:
                    continue
                detections.append(DetectedObject(
                    class_id=cls_id,
                    class_name=str(names.get(cls_id, cls_id)) if names else str(cls_id),
                    confidence=conf,
                    bbox=bbox,
                    timestamp=frame.timestamp,
                ))
        return ModelOutput(timestamp=frame.timestamp, frame_id=frame.frame_id,
                           detections=detections)


__all__ = ["YOLOAdapter"]