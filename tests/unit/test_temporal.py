"""Unit: temporal reasoning (approach/movement/disappearance)."""
from __future__ import annotations

from lensvoice.config.settings import TemporalConfig
from lensvoice.models.schemas import BBox, DetectedObject, Frame
from lensvoice.reasoning.temporal import TemporalAnalyzer
from lensvoice.tracking.tracker import MockTracker, TrackingConfig

CFG = TemporalConfig(min_observations=3, disappear_frames=3)
TA = TemporalAnalyzer(CFG)


def _frame(w=640, h=360, ts=1.0, fid=1) -> Frame:
    import numpy as np
    return Frame(image=np.zeros((h, w, 3), dtype=np.uint8), timestamp=ts, frame_id=fid)


def _dets(inner: list[float], ts: float, cls="car", cid=1, conf=0.9) -> list[DetectedObject]:
    x1, y1, x2, y2 = inner
    return [DetectedObject(class_id=cid, class_name=cls, confidence=conf,
                           bbox=BBox(x1, y1, x2, y2), timestamp=ts)]


def test_status_new_then_persistent():
    tr = MockTracker(TrackingConfig())
    states = {}
    for i in range(4):
        dets = tr.update(_dets([0, 0, 20, 20], float(i)))
        states = TA.analyze(dets, tr.tracks, _frame(ts=float(i), fid=i))
        tid = 1
        if i <= 1:
            assert states.get(tid).status == "new"
    assert states.get(tid).status == "persistent"


def test_approaching_detected():
    tr = MockTracker(TrackingConfig())
    last_states = None
    boxes = [[100, 100, 200, 200],   # area 10000
             [100, 100, 181, 181],   # area 6561 (ratio .656 <= .85 -> approaching)
             [100, 100, 172, 172]]
    for i, b in enumerate(boxes):
        dets = tr.update(_dets(b, float(i)))
        last_states = TA.analyze(dets, tr.tracks, _frame(ts=float(i), fid=i))
    assert last_states.get(1).movement == "approaching"


def test_unknown_without_history():
    tr = MockTracker(TrackingConfig())
    dets = tr.update(_dets([0, 0, 20, 20], 0.0))
    states = TA.analyze(dets, tr.tracks, _frame(ts=0.0))
    assert states.get(1).movement == "unknown"


def test_disappeared():
    tr = MockTracker(TrackingConfig())
    tr.update(_dets([0, 0, 20, 20], 0.0))
    tr.update([])
    tr.update([])
    tr.update([])
    gone = TA.disappeared(tr.tracks)
    assert any(tid == 1 for tid, _ in gone)