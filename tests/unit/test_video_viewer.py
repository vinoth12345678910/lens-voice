"""Unit: video-file camera source + LiveViewer drawing (no window, synthetic video)."""
from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from lensvoice.camera.camera import EndOfStream
from lensvoice.camera.webcam import WebcamCamera
from lensvoice.config.settings import CameraConfig, SafetyConfig, SpatialConfig
from lensvoice.display.viewer import LiveViewer
from lensvoice.models.schemas import BBox, DetectedObject, ObjectState


@pytest.fixture
def clip(tmp_path):
    path = str(tmp_path / "t.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(10):
        vw.write(np.full((48, 64, 3), i * 20, np.uint8))
    vw.release()
    return path


def test_file_ends_cleanly_and_uses_video_time(clip):
    cam = WebcamCamera(CameraConfig(source=clip), realtime=False)
    frames = []
    with pytest.raises(EndOfStream):
        while True:
            frames.append(cam.read())
    cam.stop()
    assert len(frames) == 10 and cam.is_file
    assert [round(f.timestamp) for f in frames[:3]] == [0, 100, 200]       # video time (ms)
    assert [f.frame_id for f in frames] == list(range(1, 11))


def test_realtime_paces_at_video_speed(clip):
    cam = WebcamCamera(CameraConfig(source=clip), realtime=True)
    t0 = time.monotonic()
    n = 0
    with pytest.raises(EndOfStream):
        while True:
            cam.read(); n += 1
    assert n == 10 and 0.8 <= time.monotonic() - t0 <= 1.6                  # 10 frames @ 10 fps ~ 1 s


def test_realtime_skips_frames_when_processing_is_slow(clip):
    cam = WebcamCamera(CameraConfig(source=clip), realtime=True)
    got = []
    with pytest.raises(EndOfStream):
        while True:
            got.append(cam.read()); time.sleep(0.25)                        # slower than the 0.1 s frame period
    assert 2 <= len(got) < 10                                               # frames dropped, like a live camera
    assert got[-1].timestamp > got[0].timestamp


def _det(tid=1, name="cup", box=(100, 100, 200, 200)):
    return DetectedObject(class_id=15, class_name=name, confidence=0.9, bbox=BBox(*box),
                          timestamp=0.0, track_id=tid)


def test_render_draws_boxes_depth_and_subtitles():
    v = LiveViewer(SpatialConfig(), SafetyConfig(), depth_label="test", show=False)
    base = np.zeros((480, 640, 3), np.uint8)
    centred = _det(box=(270, 100, 370, 200))                                 # centre x = 50 % -> inside the in-path band
    v.objs = [(centred, ObjectState(track_id=1, distance="very_near", h_zone="center", movement="approaching"))]
    v._objs_t = time.monotonic()
    v.depth = np.linspace(0, 1, 480 * 640).reshape(480, 640)
    v.on_spoken("cell_phone, close, right", "CRITICAL")
    out = v.render(base)
    assert out.shape == base.shape and out.any()
    assert tuple(out[100, 320]) == (60, 60, 255)                            # close + in-path -> red box edge
    assert v.subs[-1][2] == "cell phone, close, right"                      # underscores spoken/shown as words


def test_render_scales_big_frames_down_and_holds_boxes_briefly():
    v = LiveViewer(show=False, max_size=(640, 480))
    out = v.render(np.zeros((1080, 1920, 3), np.uint8))
    assert out.shape[1] <= 640 and out.shape[0] <= 480
    v.objs = [(_det(), None)]
    v._objs_t = time.monotonic() - 10                                       # stale boxes are not drawn
    assert not v.render(np.zeros((480, 640, 3), np.uint8))[60:110, 165:210].any()   # box top edge + label not drawn (guide lines are at x=160, 213)


def test_viewer_ignores_immediate_repeat_sentence():
    v = LiveViewer(show=False)
    v.on_spoken("cup, ahead", "LOW"); v.on_spoken("cup, ahead", "LOW")
    assert len(v.subs) == 1
