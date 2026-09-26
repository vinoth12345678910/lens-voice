"""Safety reasoning.

Emits evidence-driven hazard events. Hard rule: no statement is ever made
that is not supported by the structured evidence in Context. In particular we
do NOT infer a person is "going to cross", does "not see" us, or that a
vehicle "will collide" — we only report what is measured:
  - an object is in the collision band and is RELATIVELY close
  - an object classified as approaching a vehicle while near
  - text/signs (stop signs, pedestrian crossings) are present
"""
from __future__ import annotations

from lensvoice.config.settings import SafetyConfig
from lensvoice.models.schemas import BBox, Context, DetectedObject, Event


class SafetyAnalyzer:
    VEHICLE_IDS = (1, 2, 3, 4, 5, 6)  # car, motorcycle, bus, truck, autorickshaw, bicycle

    def __init__(self, cfg: SafetyConfig):
        self.cfg = cfg

    def analyze(self, ctx: Context, frame_width: int) -> list[Event]:
        events: list[Event] = []
        w = max(1, frame_width)
        now = ctx.timestamp
        for det in ctx.objects:
            st = ctx.object_states.get(det.track_id)
            if st is None:
                continue
            if st.distance not in self.cfg.close_depth_categories:
                continue
            fx = det.bbox.center_x / w
            close = (1.0 - self.cfg.center_band) / 2.0
            in_band = fx < close or fx > 1.0 - close
            is_vehicle = det.class_id in self.VEHICLE_IDS
            if in_band:
                if is_vehicle and self.cfg.require_approaching_for_vehicle:
                    # only call a hazard if we saw approach behaviour
                    if st.movement == "approaching":
                        events.append(Event(
                            type="vehicle_in_path", source="safety",
                            timestamp=now, confidence=det.confidence,
                            track_id=det.track_id, bbox=det.bbox,
                            detail={"class": det.class_name, "distance": st.distance,
                                    "movement": st.movement}))
                else:
                    events.append(Event(
                        type="hazard_path", source="safety",
                        timestamp=now, confidence=det.confidence,
                        track_id=det.track_id, bbox=det.bbox,
                        detail={"class": det.class_name, "distance": st.distance}))
        for sign in ctx.signs:
            if sign.sign_class in ("stop", "pedestrian_crossing", "yield"):
                events.append(Event(
                    type="sign_attention", source="safety",
                    timestamp=now, confidence=sign.confidence,
                    bbox=sign.bbox, detail={"sign": sign.sign_class}))
        return events


__all__ = ["SafetyAnalyzer"]