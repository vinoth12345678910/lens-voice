"""Change detection.

Emits evidence (Event objects) only when the world demonstrably changed:
a track appeared/disappeared, moved across the frame, got noticeably closer,
or new OCR/sign text arrived. Cooldowns prevent narration spam.

Anti-hallucination rule honored here: no event without a supporting
measurement that survived the configured thresholds.
"""
from __future__ import annotations

from typing import Optional

from lensvoice.config.settings import ChangeDetectionConfig, TemporalConfig
from lensvoice.models.schemas import (
    DetectedObject, Event, Frame, ObjectState,
)
from lensvoice.reasoning.spatial import DISTANCE_ORDER
from lensvoice.tracking.tracker import Trajectory as Traj


class ChangeDetector:
    def __init__(self, cfg: ChangeDetectionConfig, temporal_cfg: TemporalConfig):
        self.cfg = cfg
        self.temporal_cfg = temporal_cfg
        self._last_center: dict[int, tuple[float, float]] = {}
        self._last_distance: dict[int, str] = {}
        self._cooldowns: dict[str, float] = {}
        self._last_text: dict[tuple[str, str], float] = {}

    def _cooldown_active(self, key: str, now: float, ms: float) -> bool:
        return now - self._cooldowns.get(key, -1e9) < ms

    def _touch(self, key: str, now: float) -> None:
        self._cooldowns[key] = now

    # -- object-level events ----------------------------------------------
    def update(self, frame: Frame, detections_by_id: dict[int, DetectedObject],
               object_states: dict[int, ObjectState],
               trajectories: dict[int, Traj]) -> list[Event]:
        events: list[Event] = []
        now = frame.timestamp
        seen: set[int] = set()

        for tid, st in object_states.items():
            seen.add(tid)
            det = detections_by_id.get(tid)
            key = f"obj:{tid}"
            if st.status == "new":
                if not self._cooldown_active(key + ":new", now, self.cfg.cooldown_ms):
                    events.append(Event(
                        type="new_object", source="change_detection",
                        timestamp=now, track_id=tid,
                        bbox=det.bbox if det else None,
                        detail={"class": det.class_name if det else "",
                                "distance": st.distance}))
                    self._touch(key + ":new", now)
                continue

            tr = trajectories.get(tid)
            if tr is None or len(tr.bboxes) < 2:
                continue
            prev = tr.bboxes[-2]
            cur = tr.bboxes[-1]
            center_dx = abs(cur.center_x - prev.center_x) / max(1.0, frame.width)
            if center_dx > self.cfg.moved_center_ratio and \
                    not self._cooldown_active(key + ":move", now, self.cfg.cooldown_ms):
                events.append(Event(type="object_moved", source="change_detection",
                                    timestamp=now, track_id=tid,
                                    bbox=cur, detail={"dx_normalised": round(center_dx, 3)}))
                self._touch(key + ":move", now)

            prev_dist = self._last_distance.get(tid, st.distance)
            drop = self._index(st.distance) - self._index(prev_dist)
            if st.distance != prev_dist and drop < 0 and \
                    self._cooldown_active(key + ":dist", now, self.cfg.cooldown_ms) is False and \
                    -drop >= self.cfg.distance_drop_levels:
                events.append(Event(type="object_closer", source="change_detection",
                                    timestamp=now, track_id=tid,
                                    bbox=cur,
                                    detail={"distance": st.distance}))
                self._touch(key + ":dist", now)
            self._last_distance[tid] = st.distance

        # disappearance is only announced for tracks we ever narrated
        for tid, tr in trajectories.items():
            if tid not in seen and tr.frames_missing >= self.temporal_cfg.disappear_frames:
                key = f"obj:{tid}"
                if not self._cooldown_active(key + ":gone", now, self.cfg.cooldown_ms):
                    events.append(Event(type="object_gone", source="change_detection",
                                        timestamp=now, track_id=tid))
                    self._touch(key + ":gone", now)
        return events

    @staticmethod
    def _index(distance: str) -> int:
        if distance not in DISTANCE_ORDER:
            return 0
        return DISTANCE_ORDER.index(distance)

    # -- text/sign-level events ------------------------------------------
    def text_changed(self, now: float, items: list[tuple[str, str]]) -> list[Event]:
        events: list[Event] = []
        for text, kind in items:
            key = f"text:{kind}:{text}"
            if self._cooldown_active(key, now, self.cfg.text_cooldown_ms):
                continue
            self._touch(key, now)
            events.append(Event(type="text_detected", source="change_detection",
                                timestamp=now, detail={"text": text, "kind": kind}))
        return events

    def sign_ok(self, sign_class: str, now: float) -> bool:
        """Cooldown gate for recurring sign_attention events."""
        key = f"sign:{sign_class}"
        if self._cooldown_active(key, now, self.cfg.cooldown_ms):
            return False
        self._touch(key, now)
        return True


__all__ = ["ChangeDetector"]