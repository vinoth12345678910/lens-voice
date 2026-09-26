"""Spatial reasoning.

Maps each detected object into a coarse 3x3 zone of the frame and into a
relative-depth bucket (very_near/near/medium/far). The depth bucket is derived
from the RELATIVE depth map — the pipeline never claims real-world distance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from lensvoice.config.settings import SpatialConfig
from lensvoice.models.schemas import BBox


@dataclass
class SpatialInfo:
    h_zone: str = "unknown"      # left | center | right
    v_zone: str = "unknown"      # upper | middle | lower
    distance: str = "unknown"    # very_near | near | medium | far | unknown


DISTANCE_ORDER = ["far", "medium", "near", "very_near"]


class SpatialAnalyzer:
    def __init__(self, cfg: SpatialConfig):
        self.cfg = cfg

    # -- zones ------------------------------------------------------------
    def _h_zone(self, fx: float) -> str:
        if fx < self.cfg.left_boundary:
            return "left"
        if fx > self.cfg.right_boundary:
            return "right"
        return "center"

    def _v_zone(self, fy: float) -> str:
        if fy < self.cfg.upper_boundary:
            return "upper"
        if fy > self.cfg.lower_boundary:
            return "lower"
        return "middle"

    def zones(self, box: BBox, width: int, height: int) -> tuple[str, str]:
        fx = box.center_x / max(1.0, width)
        fy = box.center_y / max(1.0, height)
        return self._h_zone(fx), self._v_zone(fy)

    # -- relative depth bucket --------------------------------------------
    def depth_of(self, depth: Optional[np.ndarray], box: BBox) -> str:
        if depth is None or depth.size == 0:
            return "unknown"
        h, w = depth.shape
        x1 = max(0, int(round(box.x1)))
        y1 = max(0, int(round(box.y1)))
        x2 = min(w - 1, int(round(box.x2)))
        y2 = min(h - 1, int(round(box.y2)))
        if x2 <= x1 or y2 <= y1:
            return "unknown"
        region = depth[y1:y2 + 1, x1:x2 + 1]
        value = float(np.median(region))
        for label, bound in sorted(self.cfg.distance_bins.items(),
                                   key=lambda kv: kv[1]):
            if value <= bound:
                return label
        return "far"

    def classify(self, box: BBox, width: int, height: int,
                 depth: Optional[np.ndarray]) -> SpatialInfo:
        hz, vz = self.zones(box, width, height)
        return SpatialInfo(h_zone=hz, v_zone=vz, distance=self.depth_of(depth, box))


__all__ = ["SpatialAnalyzer", "SpatialInfo", "DISTANCE_ORDER"]