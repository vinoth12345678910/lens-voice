"""Ultralytics ByteTrack adapter for the LensVoice tracking contract."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from lensvoice.config.settings import TrackingConfig
from lensvoice.models.schemas import BBox, DetectedObject
from lensvoice.tracking.tracker import Trajectory


class _ByteTrackResults:
    """Minimal Results-like object required by Ultralytics BYTETracker."""

    def __init__(
        self,
        xywh: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
    ):
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask):
        return _ByteTrackResults(
            self.xywh[mask],
            self.conf[mask],
            self.cls[mask],
        )


class ByteTrackTracker:
    """Adapter around Ultralytics BYTETracker."""

    def __init__(self, cfg: TrackingConfig):
        self.cfg = cfg
        self.tracks: dict[int, Trajectory] = {}

        args = SimpleNamespace(
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            new_track_thresh=0.5,
            track_buffer=cfg.max_age_frames,
            match_thresh=0.8,
            fuse_score=True,
        )

        from ultralytics.trackers.byte_tracker import BYTETracker

        self._tracker = BYTETracker(args)

    @staticmethod
    def _to_results(
        detections: list[DetectedObject],
    ) -> _ByteTrackResults:
        """Convert LensVoice detections to ByteTrack input format."""

        xywh = []
        conf = []
        cls = []

        for det in detections:
            xywh.append(
                [
                    det.bbox.center_x,
                    det.bbox.center_y,
                    det.bbox.width,
                    det.bbox.height,
                ]
            )

            conf.append(det.confidence)
            cls.append(det.class_id)

        return _ByteTrackResults(
            np.asarray(xywh, dtype=np.float32).reshape(-1, 4),
            np.asarray(conf, dtype=np.float32),
            np.asarray(cls, dtype=np.float32),
        )

    def update(
        self,
        detections: list[DetectedObject],
    ) -> list[DetectedObject]:
        """Update ByteTrack and assign track IDs to detections."""

        if not detections:
            self._tracker.update(
                self._to_results([]),
                img=None,
            )

            for track in self.tracks.values():
                track.frames_missing += 1

            return detections

        results = self._to_results(detections)

        tracked = self._tracker.update(
            results,
            img=None,
        )

        for row in tracked:
            x1, y1, x2, y2, track_id, score, class_id, det_index = row

            det_index = int(det_index)
            track_id = int(track_id)
            class_id = int(class_id)

            if det_index < 0 or det_index >= len(detections):
                continue

            det = detections[det_index]

            det.track_id = track_id

            bbox = BBox(
                float(x1),
                float(y1),
                float(x2),
                float(y2),
            )

            det.bbox = bbox

            now = float(det.timestamp)

            if track_id not in self.tracks:
                self.tracks[track_id] = Trajectory(
                    track_id=track_id,
                    bboxes=[bbox],
                    class_ids=[class_id],
                    first_seen=now,
                    last_seen=now,
                    last_center=(
                        bbox.center_x,
                        bbox.center_y,
                    ),
                    last_area=bbox.area,
                    frames_missing=0,
                )
            else:
                track = self.tracks[track_id]

                track.bboxes.append(bbox)
                track.class_ids.append(class_id)
                track.last_seen = now
                track.last_center = (
                    bbox.center_x,
                    bbox.center_y,
                )
                track.last_area = bbox.area
                track.frames_missing = 0

        active_ids = {
            int(row[4])
            for row in tracked
        }

        for track_id, track in self.tracks.items():
            if track_id not in active_ids:
                track.frames_missing += 1

        return detections