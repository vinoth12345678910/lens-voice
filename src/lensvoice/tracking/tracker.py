"""Multi-object tracking.

The skeleton ships a deterministic mock tracker (greedy IoU association) so
the pipeline and integration tests run end-to-end. ByteTrack / BoT-SORT are
the PLANNED replacements behind the same API:
    update(detections: list[DetectedObject]) -> list[DetectedObject]

The tracker also retains per-track history (Trajectory) that spatial/temporal
reasoning and change detection read from.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional

from lensvoice.config.settings import TrackingConfig
from lensvoice.models.schemas import BBox, DetectedObject


@dataclass
class Trajectory:
    """History retained for one track id."""
    track_id: int
    bboxes: list[BBox] = field(default_factory=list)
    class_ids: list[int] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    last_center: tuple[float, float] = (0.0, 0.0)
    last_area: float = 0.0
    frames_missing: int = 0

    def last_bbox(self) -> Optional[BBox]:
        return self.bboxes[-1] if self.bboxes else None


class MockTracker:
    """Deterministic greedy-IoU tracker (no external dependencies)."""

    def __init__(self, cfg: TrackingConfig):
        self.cfg = cfg
        self._next_id = itertools.count(1)
        self.tracks: dict[int, Trajectory] = {}

    def update(self, detections: list[DetectedObject]) -> list[DetectedObject]:
        if not detections:
            for tr in self.tracks.values():
                tr.frames_missing += 1
            return detections
        old_ids = list(self.tracks.keys())
        old_boxes = [self.tracks[i].last_bbox() for i in old_ids]
        old_boxes = [b for b in old_boxes if b is not None]

        pairs: list[tuple[int, int]] = []  # (det_index, track_index)
        if old_boxes:
            costs = [[1.0 - det.bbox.iou(o) for o in old_boxes] for det in detections]
            try:
                from scipy.optimize import linear_sum_assignment
            except ImportError:  # pragma: no cover
                linear_sum_assignment = None
            if linear_sum_assignment is not None:
                import numpy as np
                rows, cols = linear_sum_assignment(np.asarray(costs))
                for r, c in zip(rows, cols):
                    if costs[r][c] < 1.0 - self.cfg.iou_threshold:
                        pairs.append((r, int(c)))
            else:  # pragma: no cover
                used: set[int] = set()
                for r, det in enumerate(detections):
                    best, best_iou = None, -1.0
                    for c, o in enumerate(old_boxes):
                        if c in used:
                            continue
                        iou = det.bbox.iou(o)
                        if iou > self.cfg.iou_threshold and iou > best_iou:
                            best, best_iou = c, iou
                    if best is not None:
                        pairs.append((r, best))
                        used.add(best)

        matched_ids: set[int] = set()
        new_ids: set[int] = set()
        matched_dets = [False] * len(detections)
        for r, c in pairs:
            det = detections[r]
            tid = old_ids[c]
            tr = self.tracks[tid]
            tr.bboxes.append(det.bbox)
            tr.class_ids.append(det.class_id)
            tr.last_seen = det.timestamp
            tr.last_center = (det.bbox.center_x, det.bbox.center_y)
            tr.last_area = det.bbox.area
            tr.frames_missing = 0
            det.track_id = tid
            matched_dets[r] = True
            matched_ids.add(tid)

        for r, det in enumerate(detections):
            if matched_dets[r]:
                continue
            tid = next(self._next_id)
            self.tracks[tid] = Trajectory(
                track_id=tid, bboxes=[det.bbox], class_ids=[det.class_id],
                first_seen=det.timestamp, last_seen=det.timestamp,
                last_center=(det.bbox.center_x, det.bbox.center_y),
                last_area=det.bbox.area)
            det.track_id = tid
            new_ids.add(tid)

        for tid, tr in self.tracks.items():
            if tid not in matched_ids and tid not in new_ids:
                tr.frames_missing += 1
        return detections


__all__ = ["MockTracker", "Trajectory"]