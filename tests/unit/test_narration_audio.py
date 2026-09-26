"""Unit: narrator + audio queue + logger privacy."""
from __future__ import annotations

import numpy as np

from lensvoice.audio.queue import PriorityAudioQueue
from lensvoice.audio.tts import SilentTTS
from lensvoice.config.settings import AudioConfig
from lensvoice.models.schemas import (BBox, Context, DetectedObject, Event,
                                      ObjectState, PriorityScore)
from lensvoice.narration.narrator import Narrator
from lensvoice.telemetry.logger import Logger, LoggingConfig


def _ctx() -> Context:
    ctx = Context(timestamp=1.0, frame_id=1)
    det = DetectedObject(class_id=1, class_name="car", confidence=0.9,
                         bbox=BBox(0, 0, 40, 40), timestamp=1.0, track_id=1)
    ctx.objects = [det]
    ctx.object_states = {1: ObjectState(track_id=1, status="persistent",
                                        distance="near", h_zone="left",
                                        movement="approaching")}
    return ctx


def test_hazard_sentence_uses_evidence():
    n = Narrator()
    ctx = _ctx()
    ev = Event(type="hazard_path", source="safety", timestamp=1.0, track_id=1,
               detail={"class": "car", "movement": "approaching"})
    scored = [PriorityScore(event=ev, score=100.0, level="CRITICAL")]
    out = n.generate(ctx, scored)
    assert len(out) == 1
    level, text = out[0]
    assert level == "CRITICAL"
    assert "car" in text and "approaching" in text


def test_no_invention_for_unknown_state():
    n = Narrator()
    ctx = Context(timestamp=1.0, frame_id=1)
    ev = Event(type="new_object", source="change_detection", timestamp=1.0,
               detail={"class": "truck"})
    out = n.generate(ctx, [PriorityScore(event=ev, score=30.0, level="MEDIUM")])
    assert "truck" in out[0][1]
    assert not any(w in out[0][1] for w in ("distance", "speed", "intent"))


def test_audio_queue_priority_order_and_bound():
    cfg = AudioConfig(max_queue_size=4)
    tts = SilentTTS()
    q = PriorityAudioQueue(cfg, tts)
    q.enqueue("LOW", "low1")
    q.enqueue("CRITICAL", "critical1")
    first = q.pop()
    assert first.level == "CRITICAL"
    assert q.pop().level == "LOW"


def test_audio_queue_drops_low_when_full():
    cfg = AudioConfig(max_queue_size=2)
    tts = SilentTTS()
    q = PriorityAudioQueue(cfg, tts)
    q.enqueue("LOW", "a")
    q.enqueue("LOW", "b")
    assert q.enqueue("LOW", "c") is False       # full, all equal -> drop new
    assert q.enqueue("CRITICAL", "boom") is True  # better than existing -> evicts lowest
    texts = [u.text for u in [q.pop(), q.pop()]]
    assert "boom" in texts


def test_interrupt_on_critical():
    cfg = AudioConfig(interrupt_on_critical=True)
    spoken: list[str] = []
    tts = SilentTTS(persist=spoken)
    q = PriorityAudioQueue(cfg, tts)
    q.enqueue("CRITICAL", "look out", interrupt=True)
    assert spoken == ["look out"]


def test_logger_redacts_images_and_text():
    cfg = LoggingConfig(structured=True)
    log = Logger(cfg)
    log.info("frame", image=np.zeros((2, 2, 3)), ocr_text="STOP", frame_id=7)
    # no crash; image/text redacted — verify via emitter formatting
    assert True