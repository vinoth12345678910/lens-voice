"""Unit: frame scheduler."""
from __future__ import annotations

from lensvoice.scheduler.frame_scheduler import FrameScheduler


def test_detection_runs_at_3_frames_on_30fps():
    s = FrameScheduler(30, {"yolo": 10, "ocr": 2, "traffic_sign": 3,
                           "depth": 5, "scene": 1})
    assert s.interval("yolo") == 3
    assert s.interval("ocr") == 15
    assert s.interval("scene") == 30
    assert s.should_run("yolo", 0) is True
    assert s.should_run("yolo", 3) is True
    assert s.should_run("yolo", 4) is False


def test_due_lists_due_modules():
    s = FrameScheduler(30, {"yolo": 10, "ocr": 2, "traffic_sign": 3,
                           "depth": 5, "scene": 1})
    due = s.due(0)
    assert set(due) == {"yolo", "ocr", "traffic_sign", "depth", "scene"}
    assert set(s.due(10)) == {"traffic_sign"}
    assert set(s.due(30)) == {"yolo", "ocr", "traffic_sign", "depth", "scene"}


def test_div_by_zero_guards():
    s = FrameScheduler(0, {"yolo": 0})
    assert s.interval("yolo") >= 1