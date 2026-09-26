"""Unit: change detection + safety + priority."""
from __future__ import annotations

import numpy as np

from lensvoice.config.settings import (ChangeDetectionConfig, PriorityConfig,
                                       SafetyConfig, TemporalConfig)
from lensvoice.models.schemas import BBox, Context, DetectedObject, Event, Frame
from lensvoice.reasoning.change_detection import ChangeDetector
from lensvoice.reasoning.priority import PriorityScorer
from lensvoice.reasoning.safety import SafetyAnalyzer

CD = ChangeDetector(ChangeDetectionConfig(cooldown_ms=5000,
                                          text_cooldown_ms=8000),
                    TemporalConfig())


def _frame(w=640, h=360, ts=1000.0, fid=1) -> Frame:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    return Frame(image=img, timestamp=ts, frame_id=fid)


def _det(box: BBox, track_id=1, cls="person", cid=0, ts=1000.0) -> DetectedObject:
    return DetectedObject(class_id=cid, class_name=cls, confidence=0.9,
                          bbox=box, timestamp=ts, track_id=track_id)


# -- change detection ----------------------------------------------------

def test_new_object_event_and_cooldown():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    frame = Frame(image=img, timestamp=1000.0, frame_id=1)
    dets = {1: _det(BBox(0, 0, 10, 10), ts=1000.0)}
    states = {1: {"status": "new"} | {}}
    # build proper ObjectState
    from lensvoice.models.schemas import ObjectState
    states = {1: ObjectState(track_id=1, status="new", distance="near")}
    ev = CD.update(frame, dets, states, {})
    assert any(e.type == "new_object" for e in ev)

    ev2 = CD.update(frame, dets, states, {})
    assert not any(e.type == "new_object" for e in ev2)  # cooldown active


def test_moved_above_threshold():
    tr = MockTrackerLocal()
    # reuse a simple trajectory trace: previous + current bbox
    tr.update([_det(BBox(0, 0, 10, 10), ts=1000.0)])
    tr.update([_det(BBox(80, 0, 90, 10), ts=1100.0)])  # dx=.8/640 .. below ratio? uses width
    tr.update([_det(BBox(300, 0, 320, 10), ts=1200.0)])  # dx=2.2/6.4=0.34 > 0.1
    frame = _frame(ts=1200.0, fid=3)
    states = {1: {"status": "persistent"} | {}}
    from lensvoice.models.schemas import ObjectState
    states = {1: ObjectState(track_id=1, status="persistent")}
    ev = CD.update(frame, {1: _det(BBox(300, 0, 320, 10), ts=1200.0)},
                   states, tr.tracks)
    assert any(e.type == "object_moved" for e in ev)


class MockTrackerLocal:
    """Minimal trajectory stand-in for CD tests."""

    def __init__(self):
        from lensvoice.tracking.tracker import Trajectory
        self.tracks: dict = {}
        self._Trajectory = Trajectory

    def update(self, dets):
        for d in dets:
            tid = d.track_id if d.track_id is not None else 1
            tr = self.tracks.setdefault(tid, self._Trajectory(track_id=tid))
            tr.bboxes.append(d.bbox)
            tr.class_ids.append(d.class_id)
            tr.last_seen = d.timestamp
        return dets


# -- safety ---------------------------------------------------------------

def test_hazard_when_close_in_band():
    sa = SafetyAnalyzer(SafetyConfig())
    ctx = Context(timestamp=1.0, frame_id=1)
    det = _det(BBox(5, 100, 60, 200), track_id=1, cls="person", cid=0)
    ctx.objects = [det]
    from lensvoice.models.schemas import ObjectState
    ctx.object_states = {1: ObjectState(track_id=1, status="persistent",
                                        distance="near", h_zone="left")}
    ev = sa.analyze(ctx, frame_width=640)
    assert any(e.type == "hazard_path" for e in ev)


def test_far_object_no_hazard():
    sa = SafetyAnalyzer(SafetyConfig())
    ctx = Context(timestamp=1.0, frame_id=1)
    from lensvoice.models.schemas import ObjectState
    det = _det(BBox(100, 100, 200, 200), track_id=1, cls="car", cid=1)
    ctx.objects = [det]
    ctx.object_states = {1: ObjectState(track_id=1, status="persistent", distance="far")}
    assert sa.analyze(ctx, 640) == []


def test_vehicle_needs_approach():
    sa = SafetyAnalyzer(SafetyConfig(require_approaching_for_vehicle=True))
    ctx = Context(timestamp=1.0, frame_id=1)
    from lensvoice.models.schemas import ObjectState
    det = _det(BBox(100, 100, 200, 200), track_id=1, cls="car", cid=1)
    ctx.objects = [det]
    ctx.object_states = {1: ObjectState(track_id=1, status="persistent", distance="very_near",
                                        movement="stationary")}
    # vehicle in path but stationary -> NOT a hazard per policy
    assert sa.analyze(ctx, 640) == []


# -- priority -------------------------------------------------------------

def test_score_levels():
    ps = PriorityScorer(PriorityConfig())
    events = [
        Event(type="hazard_path", source="safety", timestamp=1.0),
        Event(type="text_detected", source="change_detection", timestamp=1.0,
              detail={"text": "STOP", "kind": "text"}),
    ]
    scored = ps.score(events)
    by_type = {s.event.type: s for s in scored}
    assert by_type["hazard_path"].level == "CRITICAL"
    assert by_type["text_detected"].level == "LOW"


def test_repetition_penalty():
    ps = PriorityScorer(PriorityConfig(), cooldown_ms=5000)
    ev = Event(type="text_detected", source="change_detection", timestamp=1.0,
               detail={"text": "STOP"})
    s1 = ps.score([ev], ctx=None)[0]
    ps.remember(s1, 1.0)
    s2 = ps.score([ev], ctx=None)[0]
    assert s2.score < s1.score


def test_approaching_bonus():
    ps = PriorityScorer(PriorityConfig())
    ev = Event(type="hazard_path", source="safety", timestamp=1.0,
               detail={"movement": "approaching"})
    scored = ps.score([ev])
    assert scored[0].score == ps.cfg.weights["safety"] + ps.cfg.weights["approaching"]