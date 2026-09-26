"""Context engine.

Assembles all evidence of a single frame into one structured Context. This is
the "no natural language" layer — the narrator is the ONLY place sentences
are made, and it reads exclusively from Context/Event data.
"""
from __future__ import annotations

from lensvoice.models.schemas import (
    Context, DetectedObject, DepthResult, Frame, ObjectState, OCRResult,
    SceneResult, TrafficSignResult,
)
from lensvoice.reasoning.spatial import SpatialAnalyzer
from lensvoice.reasoning.temporal import TemporalAnalyzer


class ContextEngine:
    def __init__(self, spatial: SpatialAnalyzer, temporal: TemporalAnalyzer):
        self.spatial = spatial
        self.temporal = temporal

    def build(self, frame: Frame,
              detections: list[DetectedObject],
              tracks: dict,
              scene: SceneResult | None,
              ocr: list[OCRResult],
              signs: list[TrafficSignResult],
              depth: DepthResult | None) -> Context:
        states = self.temporal.analyze(detections, tracks, frame)
        det_by_id = {d.track_id: d for d in detections if d.track_id is not None}
        for tid, st in states.items():
            det = det_by_id.get(tid)
            if det is None:
                continue
            info = self.spatial.classify(det.bbox, frame.width, frame.height,
                                         depth.depth if depth else None)
            st.h_zone = info.h_zone
            st.v_zone = info.v_zone
            st.distance = info.distance
        return Context(timestamp=frame.timestamp, frame_id=frame.frame_id,
                       scene=scene, objects=detections,
                       object_states=states, text=ocr, signs=signs)


__all__ = ["ContextEngine"]