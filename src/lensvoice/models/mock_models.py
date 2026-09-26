"""Deterministic mock perception models.

Used whenever a model slot is `enabled: false`. They make the entire pipeline
exercisable with no GPU, no weights, no webcam — so the runtime can be built
and integration-tested before Friend 1's trained artifacts land.

Everything here is a pure function of input dimensions. Mocks deliberately
render *simple* moving objects so spatial/temporal reasoning exercises
real code paths.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from lensvoice.models.interfaces import (
    DepthModel, OCRModel, SceneModel, TrafficSignModel, VisionModel,
)
from lensvoice.models.schemas import (
    BBox, DepthResult, DetectedObject, Frame, ModelOutput, OCRResult,
    SceneResult, TrafficSignResult,
)


class MockVisionModel(VisionModel):
    def __init__(self, class_names: Dict[int, str] | None = None):
        self.class_names = class_names or {}

    def _names(self, class_id: int) -> str:
        return self.class_names.get(class_id, f"class_{class_id}")

    def predict(self, frame: Frame) -> ModelOutput:
        w, h = frame.width, frame.height
        cx, cy = w / 2.0, h / 2.0
        # Three deterministic objects: a central person, a car on the left,
        # a dog on the right. Sizes scale slightly with frame index to let
        # the temporal module observe movement/approach over time.
        k = 1.0 + 0.0004 * (frame.frame_id % 120)
        boxes = [
            DetectedObject(class_id=0, class_name=self._names(0), confidence=0.93,
                           bbox=BBox(cx - 40 * k, cy - 90 * k, cx + 40 * k, cy + 90 * k),
                           timestamp=frame.timestamp),
            DetectedObject(class_id=1, class_name=self._names(1), confidence=0.88,
                           bbox=BBox(30, cy - 40, 30 + 120 * k, cy + 40),
                           timestamp=frame.timestamp),
            DetectedObject(class_id=25, class_name=self._names(25), confidence=0.80,
                           bbox=BBox(w - 140, h - 90, w - 20, h - 30),
                           timestamp=frame.timestamp),
        ]
        return ModelOutput(timestamp=frame.timestamp, frame_id=frame.frame_id,
                           detections=boxes)


class MockOCRModel(OCRModel):
    def read(self, frame: Frame) -> list[OCRResult]:
        return [OCRResult(text="STOP", confidence=0.9, timestamp=frame.timestamp,
                          bbox=BBox(0.2 * frame.width, 0.1 * frame.height,
                                    0.45 * frame.width, 0.2 * frame.height))]


class MockTrafficSignModel(TrafficSignModel):
    def detect(self, frame: Frame) -> list[TrafficSignResult]:
        return [TrafficSignResult(sign_class="stop", confidence=0.91,
                                  timestamp=frame.timestamp,
                                  bbox=BBox(0.6 * frame.width, 0.05 * frame.height,
                                            0.85 * frame.width, 0.25 * frame.height))]


class MockDepthModel(DepthModel):
    def estimate(self, frame: Frame) -> DepthResult:
        # Relative depth map: centre-bottom is closest (1.0), top far (0.3).
        h, w = frame.height, frame.width
        yy = np.linspace(0.25, 1.0, h).reshape(h, 1)
        xx = np.linspace(0.5, 1.0, w).reshape(1, w)
        depth = yy * xx
        return DepthResult(timestamp=frame.timestamp, depth=depth,
                           metadata={"model": "mock", "calibrated": False})


class MockSceneModel(SceneModel):
    def __init__(self, label: str = "outdoor city street"):
        self.label = label

    def classify(self, frame: Frame) -> SceneResult:
        return SceneResult(scene_label=self.label, confidence=0.95,
                           timestamp=frame.timestamp)


__all__ = [
    "MockVisionModel", "MockOCRModel", "MockTrafficSignModel",
    "MockDepthModel", "MockSceneModel",
]