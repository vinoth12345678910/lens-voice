"""TrafficSignAdapter — Skeleton for the two-stage traffic-sign pipeline
(Friend 1 deliverable): a sign detector + a sign classifier.

Architecture is finalised here; the classifier weights are not available yet,
so loading them raises until Friend 1 provides `classifier_weight`. When
`enabled: false` the pipeline uses MockTrafficSignModel instead.
"""
from __future__ import annotations

from lensvoice.config.settings import ModelConfig
from lensvoice.models.interfaces import TrafficSignModel
from lensvoice.models.schemas import BBox, Frame, TrafficSignResult


class TrafficSignAdapter(TrafficSignModel):
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self._detector = None
        self._classifier = None

    def _ensure(self):
        if self._detector is not None:
            return self._detector
        if not self.cfg.weight:
            raise RuntimeError(
                "traffic_sign enabled but 'models.traffic_sign.weight' unset")
        from ultralytics import YOLO
        self._detector = YOLO(self.cfg.weight)
        if self.cfg.classifier_weight:
            # TODO(Friend 1): wire classifier here once weights + labels ready.
            raise NotImplementedError(
                "sign classifier not implemented yet; provide classifier_weight "
                "together with the classifier code.")
        return self._detector

    def detect(self, frame: Frame) -> list[TrafficSignResult]:
        det = self._ensure()
        results = det.predict(frame.image, conf=self.cfg.confidence,
                              device=self.cfg.device or None, verbose=False)
        out = []
        for res in results:
            if res.boxes is None:
                continue
            for box in res.boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                if conf < self.cfg.confidence:
                    continue
                try:
                    bbox = BBox(xyxy[0], xyxy[1], xyxy[2], xyxy[3])
                except ValueError:
                    continue
                out.append(TrafficSignResult(
                    sign_class="sign", confidence=conf,
                    timestamp=frame.timestamp, bbox=bbox))
        return out


__all__ = ["TrafficSignAdapter"]