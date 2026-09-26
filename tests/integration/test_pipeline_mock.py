"""Integration: the whole pipeline runs end-to-end on mocks only.

No GPU, no weights, no webcam, no framework state.
"""
from __future__ import annotations

import numpy as np

from lensvoice.audio.tts import SilentTTS
from lensvoice.camera.mock_camera import MockCamera
from lensvoice.config.settings import Settings
from lensvoice.models.schemas import Frame, ObjectState
from lensvoice.runtime.pipeline import Pipeline


def _settings() -> Settings:
    return Settings.from_dict({
        "class_names": {0: "person", 1: "car", 2: "motorcycle", 25: "dog"},
        "camera": {"width": 640, "height": 360, "fps": 30},
        "scheduler": {"camera_fps": 30, "detection_fps": 10, "ocr_fps": 2,
                      "traffic_sign_fps": 3, "depth_fps": 5, "scene_fps": 1},
    })


def test_process_frame_produces_evidence():
    settings = _settings()
    spoken: list[str] = []
    pipe = Pipeline(settings, MockCamera(settings.camera),
                    tts=SilentTTS(),
                    on_utterance=lambda lvl, text: spoken.append(text))
    pipe.start()

    img = np.zeros((360, 640, 3), dtype=np.uint8)
    ctx = pipe.process_frame(Frame(image=img, timestamp=0.0, frame_id=0))  # every module due

    pipe.stop()
    assert len(ctx.objects) == 3
    assert len(ctx.object_states) == 3
    states = list(ctx.object_states.values())
    assert all(isinstance(s, ObjectState) for s in states)
    assert all(s.status == "new" for s in states)  # first sighting
    assert all(s.h_zone in ("left", "center", "right") for s in states)
    assert ctx.scene is not None
    assert ctx.signs and ctx.text  # mock OCR + signs ran on their schedules


def test_movement_resolves_over_frames():
    settings = _settings()
    pipe = Pipeline(settings, MockCamera(settings.camera), tts=SilentTTS())
    pipe.start()
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    seen_movement = False
    for fid in range(0, 15, 3):  # YOLO due frames
        ctx = pipe.process_frame(Frame(image=img, timestamp=float(fid), frame_id=fid))
        if any(s.movement != "unknown" for s in ctx.object_states.values()):
            seen_movement = True
    pipe.stop()
    # mock person/car boxes grow each frame -> approaching should resolve
    assert seen_movement


def test_audio_speaks_without_weights():
    settings = _settings()
    spoken: list[str] = []
    pipe = Pipeline(settings, MockCamera(settings.camera),
                    tts=SilentTTS(persist=spoken))
    pipe.start()
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    for fid in range(1, 10):
        pipe.process_frame(Frame(image=img, timestamp=float(fid), frame_id=fid))
    pipe.audio.drain()  # mirror run()'s drain-at-exit
    pipe.stop()
    assert any(("person" in s or "car" in s) for s in spoken)
    assert pipe.metrics.count("detections") > 0


def test_run_loop_exits_cleanly_on_bounded_camera():
    settings = _settings()
    cam = MockCamera(settings.camera, max_frames=25)
    pipe = Pipeline(settings, cam, tts=SilentTTS())
    pipe.run()  # must return without exceptions
    assert True


def test_enabled_yolo_without_weight_refuses():
    import numpy as np
    from lensvoice.models.yolo_adapter import YOLOAdapter
    cfg = _settings().model("yolo")
    cfg.enabled = True
    cfg.weight = ""
    adapter = YOLOAdapter(cfg)
    f = Frame(image=np.zeros((10, 10, 3), dtype=np.uint8), timestamp=1.0, frame_id=1)
    try:
        adapter.predict(f)
        raise AssertionError("expected RuntimeError (weight unset)")
    except RuntimeError as e:
        assert "weight" in str(e)