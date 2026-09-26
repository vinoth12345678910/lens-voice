"""Unit: mock tracker (greedy IoU association)."""
from __future__ import annotations

from lensvoice.config.settings import TrackingConfig
from lensvoice.models.schemas import BBox, DetectedObject
from lensvoice.tracking.tracker import MockTracker

DEFAULT_CFG = TrackingConfig()


def _det(bbox: BBox, class_id=0, conf=0.9, cls_name="person", ts=1.0) -> DetectedObject:
    return DetectedObject(class_id=class_id, class_name=cls_name, confidence=conf,
                          bbox=bbox, timestamp=ts)


def test_new_tracks_assigned():
    tr = MockTracker(DEFAULT_CFG)
    dets = [_det(BBox(0, 0, 10, 10)), _det(BBox(100, 0, 200, 100), class_id=1, cls_name="car")]
    out = tr.update(dets)
    assert {d.track_id for d in out} == {1, 2}
    assert len(tr.tracks) == 2


def test_same_objects_reuse_tracks():
    tr = MockTracker(DEFAULT_CFG)
    dets1 = [_det(BBox(0, 0, 10, 10)), _det(BBox(100, 0, 200, 100))]
    dets2 = [_det(BBox(0, 0, 10, 10), ts=1.1), _det(BBox(100, 0, 200, 100), ts=1.1)]
    out1 = tr.update(dets1)
    out2 = tr.update(dets2)
    assert {d.track_id for d in out2} == {d.track_id for d in out1}
    assert out2[0].track_id is not None


def test_new_object_gets_new_id():
    tr = MockTracker(DEFAULT_CFG)
    tr.update([_det(BBox(0, 0, 10, 10))])
    out = tr.update([_det(BBox(300, 300, 400, 400), ts=1.1)])
    assert len(out) == 1
    assert out[0].track_id == 2


def test_missing_track_ages():
    tr = MockTracker(DEFAULT_CFG)
    tr.update([_det(BBox(0, 0, 10, 10))])
    tr.update([])
    tr.update([])
    assert tr.tracks[1].frames_missing == 2


def test_trajectory_history():
    tr = MockTracker(DEFAULT_CFG)
    tr.update([_det(BBox(0, 0, 10, 10), ts=1.0)])
    tr.update([_det(BBox(0, 0, 12, 12), ts=1.1)])
    traj = tr.tracks[1]
    assert len(traj.bboxes) == 2