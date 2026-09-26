"""Unit: data contracts (BBox, Frame)."""
from __future__ import annotations

import numpy as np
import pytest

from lensvoice.models.schemas import BBox, Frame


def test_bbox_validation():
    with pytest.raises(ValueError):
        BBox(10, 10, 5, 5)
    b = BBox(0, 0, 2, 2)
    assert b.width == 2 and b.height == 2
    assert b.center_x == 1.0 and b.center_y == 1.0
    assert b.area == 4.0


def test_bbox_xywh_roundtrip():
    b = BBox.from_xywh(10, 20, 30, 40)
    assert (b.x1, b.y1, b.x2, b.y2) == (10, 20, 40, 60)


def test_bbox_iou():
    a = BBox(0, 0, 10, 10)
    assert a.iou(a) == 1.0
    b = BBox(10, 0, 20, 10)          # touches edge
    assert a.iou(b) == 0.0
    c = BBox(5, 0, 15, 10)           # half overlap
    assert a.iou(c) == pytest.approx(50 / 150)


def test_bbox_center_delta_and_ratio():
    a = BBox(0, 0, 10, 10)
    b = BBox(0, 0, 20, 20)
    assert a.center_delta_to(a) == 0.0
    assert a.size_ratio_to(b) == pytest.approx(0.25)


def test_frame_derives_shape():
    img = np.zeros((24, 80, 3), dtype=np.uint8)
    f = Frame(image=img, timestamp=1.0, frame_id=3)
    assert f.width == 80 and f.height == 24