"""Unit: mock models are deterministic, weight-free, and frame-shape driven."""
from __future__ import annotations

import numpy as np

from lensvoice.models.schemas import Frame
from lensvoice.models.mock_models import (MockDepthModel, MockOCRModel,
                                          MockSceneModel, MockTrafficSignModel,
                                          MockVisionModel)


def _frame(frame_id=1, ts=10.0) -> Frame:
    return Frame(image=np.zeros((300, 400, 3), dtype=np.uint8),
                 timestamp=ts, frame_id=frame_id)


def test_mock_vision_box_counts_deterministic():
    v = MockVisionModel({0: "person", 1: "car", 25: "dog"})
    out1 = v.predict(_frame())
    out2 = v.predict(_frame())
    assert len(out1.detections) == 3 == len(out2.detections)
    assert [d.class_id for d in out1.detections] == [0, 1, 25]
    assert out1.detections[0].bbox == out2.detections[0].bbox


def test_mock_vision_boxes_stay_in_frame():
    v = MockVisionModel({0: "person"})
    f = _frame()
    for d in v.predict(f).detections:
        assert 0 <= d.bbox.x1 <= d.bbox.x2 <= f.width
        assert 0 <= d.bbox.y1 <= d.bbox.y2 <= f.height


def test_mock_ocr_sign_scene_depth():
    f = _frame()
    assert MockOCRModel().read(f)[0].text == "STOP"
    assert MockTrafficSignModel().detect(f)[0].sign_class == "stop"
    assert MockSceneModel().classify(f).scene_label == "outdoor city street"
    d = MockDepthModel().estimate(f)
    assert d.depth is not None and d.depth.max() <= 1.0
    assert d.metadata["model"] == "mock"