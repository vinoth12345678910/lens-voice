"""Unit: spatial reasoning."""
from __future__ import annotations

import numpy as np

from lensvoice.config.settings import SpatialConfig
from lensvoice.models.schemas import BBox
from lensvoice.reasoning.spatial import SpatialAnalyzer

CFG = SpatialConfig()
SA = SpatialAnalyzer(CFG)


def test_horizontal_zones():
    assert SA._h_zone(0.1) == "left"
    assert SA._h_zone(0.4) == "center"
    assert SA._h_zone(0.9) == "right"


def test_vertical_zones():
    assert SA._v_zone(0.1) == "upper"
    assert SA._v_zone(0.5) == "middle"
    assert SA._v_zone(0.9) == "lower"


def test_depth_buckets():
    # depth of 0.05 folds into very_near (bin 0.15)
    assert SA.depth_of(np.full((10, 10), 0.05), BBox(1, 1, 6, 6)) == "very_near"
    assert SA.depth_of(np.full((10, 10), 0.25), BBox(1, 1, 6, 6)) == "near"
    assert SA.depth_of(np.full((10, 10), 0.45), BBox(1, 1, 6, 6)) == "medium"
    assert SA.depth_of(np.full((10, 10), 0.99), BBox(1, 1, 6, 6)) == "far"


def test_depth_none():
    assert SA.depth_of(None, BBox(1, 1, 6, 6)) == "unknown"


def test_classify_integrates():
    depth = np.full((100, 100), 0.02)
    info = SA.classify(BBox(35, 10, 65, 30), width=100, height=100, depth=depth)
    assert info.h_zone == "center"
    assert info.distance == "very_near"