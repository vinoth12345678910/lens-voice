"""Temporal reasoning.

Classifies movement / approach / recede / disappearance from tracker history.
All statements are relativistic (bbox-size ratios over time), never absolute
speed — the pipeline cannot measure real velocity.
"""
from __future__ import annotations

from lensvoice.config.settings import TemporalConfig
from lensvoice.models.schemas import DetectedObject, Frame, ObjectState
from lensvoice.tracking.tracker import Trajectory


class TemporalAnalyzer:
    def __init__(self, cfg: TemporalConfig):
        self.cfg = cfg

    def analyze(self, detections: list[DetectedObject],
                tracks: dict[int, Trajectory],
                frame: Frame) -> dict[int, ObjectState]:
        states: dict[int, ObjectState] = {}
        for det in detections:
            tid = det.track_id
            if tid is None:
                continue
            tr = tracks.get(tid)
            state = ObjectState(track_id=tid)
            state.observations = len(tr.bboxes) if tr else 1
            state.status = "new" if state.observations <= 2 else "persistent"
            if state.observations >= self.cfg.min_observations and tr is not None:
                state.movement = self._movement(tr, frame.width)
            states[tid] = state
        return states

    def _movement(self, tr: Trajectory, frame_width: int) -> str:
        hist = tr.bboxes
        if len(hist) < 2:
            return "unknown"
        prev = hist[-2]
        cur = hist[-1]
        dx = abs(cur.center_x - prev.center_x) / max(1.0, frame_width)
        ratio = cur.size_ratio_to(prev)
        if ratio <= self.cfg.approach_ratio and abs(ratio - 1.0) > 0.02:
            return "approaching"
        if ratio >= self.cfg.recede_ratio:
            return "receding"
        if dx > self.cfg.movement_ratio:
            return "moving"
        return "stationary"

    # -- disappearance (survival) -----------------------------------------
    def disappeared(self, tracks: dict[int, Trajectory]) -> list[tuple[int, Trajectory]]:
        return [(tid, tr) for tid, tr in tracks.items()
                if tr.frames_missing >= self.cfg.disappear_frames]


__all__ = ["TemporalAnalyzer"]