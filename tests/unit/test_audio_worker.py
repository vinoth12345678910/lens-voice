"""Unit: live playback worker, staleness, interrupt; narrator + null models."""
from __future__ import annotations

import threading
import time

from lensvoice.audio.queue import PriorityAudioQueue
from lensvoice.audio.tts import TTS
from lensvoice.config.settings import AudioConfig, ModelConfig, Settings
from lensvoice.models import create_models
from lensvoice.models.mock_models import NullOCRModel, NullTrafficSignModel
from lensvoice.models.schemas import Context, Event
from lensvoice.narration.narrator import Narrator


class SlowTTS(TTS):
    def __init__(self, dur=0.05):
        self.dur, self.spoken, self.stops = dur, [], 0
        self._cut = threading.Event()

    def speak(self, text, level="MEDIUM", interrupt=False):
        self._cut.clear()
        self._cut.wait(self.dur)          # "speaking"; stop() ends it early
        self.spoken.append(text)

    def stop(self):
        self.stops += 1
        self._cut.set()


def _wait(pred, t=2.0):
    end = time.time() + t
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_worker_speaks_by_priority_then_fifo():
    tts = SlowTTS(0.02)
    q = PriorityAudioQueue(AudioConfig(max_queue_size=8), tts)
    q.enqueue("LOW", "l1"); q.enqueue("LOW", "l2"); q.enqueue("MEDIUM", "m1"); q.enqueue("CRITICAL", "c1")
    q.start_worker()
    assert _wait(lambda: len(tts.spoken) == 4)
    q.stop_worker()
    assert tts.spoken == ["c1", "m1", "l1", "l2"]


def test_stale_items_are_dropped():
    tts = SlowTTS(0.01)
    q = PriorityAudioQueue(AudioConfig(max_age_ms=50), tts)
    q.enqueue("LOW", "old")
    time.sleep(0.12)
    q.enqueue("LOW", "fresh")
    q.start_worker()
    assert _wait(lambda: "fresh" in tts.spoken)
    q.stop_worker()
    assert tts.spoken == ["fresh"] and q.stale_dropped == 1


def test_critical_interrupts_lower_priority_speech():
    tts = SlowTTS(1.0)                       # every utterance "takes" 1 s unless cut
    q = PriorityAudioQueue(AudioConfig(interrupt_on_critical=True), tts)
    q.start_worker()
    q.enqueue("LOW", "long low")
    assert _wait(lambda: q._speaking_rank == 3)
    t0 = time.time()
    q.enqueue("CRITICAL", "danger")
    assert _wait(lambda: "long low" in tts.spoken, 0.5)      # the LOW speech was cut short, not waited out
    assert time.time() - t0 < 0.5 and tts.stops >= 1
    assert _wait(lambda: "danger" in tts.spoken, 3.0)        # then the critical item is spoken
    q.stop_worker()


def test_stop_worker_is_prompt_and_discards():
    tts = SlowTTS(0.2)
    q = PriorityAudioQueue(AudioConfig(max_queue_size=8), tts)
    for i in range(6):
        q.enqueue("LOW", f"x{i}")
    q.start_worker()
    time.sleep(0.05)
    t0 = time.time()
    q.stop_worker()
    assert time.time() - t0 < 1.5 and len(q) == 0


def test_worker_survives_tts_exception():
    class Bad(SlowTTS):
        def speak(self, text, level="MEDIUM", interrupt=False):
            if text == "bad":
                raise RuntimeError("x")
            super().speak(text, level)
    tts = Bad(0.01)
    q = PriorityAudioQueue(AudioConfig(), tts)
    q.start_worker(); q.enqueue("LOW", "bad"); q.enqueue("LOW", "good")
    assert _wait(lambda: "good" in tts.spoken)
    q.stop_worker()


def test_nameless_object_gone_is_not_narrated():
    n = Narrator()
    ctx = Context(timestamp=1.0, frame_id=1)
    gone = Event(type="object_gone", source="change_detection", timestamp=1.0, track_id=9)
    assert n._sentence(gone, ctx) is None
    named = Event(type="object_gone", source="change_detection", timestamp=1.0, track_id=9,
                  detail={"class": "cup"})
    assert n._sentence(named, ctx) == "cup no longer visible"


def test_null_models_when_mock_false():
    s = Settings()
    s.models["ocr"] = ModelConfig(mock=False)
    s.models["traffic_sign"] = ModelConfig(mock=False)
    m = create_models(s)
    assert isinstance(m["ocr"], NullOCRModel) and isinstance(m["traffic_sign"], NullTrafficSignModel)
    assert m["ocr"].read(None) == [] and m["traffic_sign"].detect(None) == []


def test_same_sentence_not_repeated_within_window():
    tts = SlowTTS(0.01)
    q = PriorityAudioQueue(AudioConfig(dedupe_ms=200, max_queue_size=8), tts)
    assert q.enqueue("CRITICAL", "laptop in your path, close, right") is True
    assert q.enqueue("CRITICAL", "laptop in your path, close, right") is False   # repeat inside window
    assert q.enqueue("LOW", "cup, ahead") is True                                # different sentence is fine
    assert q.dupes_dropped == 1
    time.sleep(0.25)
    assert q.enqueue("CRITICAL", "laptop in your path, close, right") is True    # window passed


def test_dedupe_off_by_default():
    q = PriorityAudioQueue(AudioConfig(max_queue_size=8), SlowTTS())
    assert q.enqueue("LOW", "x") and q.enqueue("LOW", "x")


class PrepTTS(SlowTTS):
    """prepare() takes time for some sentences, like a network fetch."""
    def __init__(self, prep=None, dur=0.02):
        super().__init__(dur)
        self.prep = prep or {}
        self.prefetched = []

    def prepare(self, text):
        time.sleep(self.prep.get(text, 0.0))
        return True

    def prefetch(self, text):
        self.prefetched.append(text)


def test_enqueue_starts_prefetch_only_in_live_mode():
    tts = PrepTTS()
    q = PriorityAudioQueue(AudioConfig(), tts)
    q.enqueue("LOW", "before worker")
    assert tts.prefetched == []                      # legacy sync mode: untouched
    q.start_worker()
    q.enqueue("LOW", "live one")
    assert "live one" in tts.prefetched
    q.stop_worker()


def test_urgent_sentence_overtakes_one_still_being_fetched():
    tts = PrepTTS(prep={"slow low": 0.4})
    q = PriorityAudioQueue(AudioConfig(max_queue_size=8, interrupt_on_critical=False), tts)
    q.start_worker()
    q.enqueue("LOW", "slow low")
    time.sleep(0.1)                                   # worker is now waiting for its audio
    q.enqueue("CRITICAL", "danger")
    assert _wait(lambda: len(tts.spoken) == 2, 3.0)
    q.stop_worker()
    assert tts.spoken == ["danger", "slow low"]


def test_sentence_that_went_stale_while_fetching_is_not_spoken():
    tts = PrepTTS(prep={"late": 0.4})
    q = PriorityAudioQueue(AudioConfig(max_age_ms=150), tts)
    q.start_worker()
    q.enqueue("LOW", "late")
    time.sleep(0.7)
    q.stop_worker()
    assert tts.spoken == [] and q.stale_dropped == 1


def test_warm_phrases_are_the_narrators_own_sentences():
    from lensvoice.narration.narrator import Narrator
    from lensvoice.narration.phrases import common_phrases, core_phrases
    core, allp = core_phrases(), common_phrases()
    n = Narrator()
    assert n.scene_sentence("urgent", "chair", 1, "center", "near", True, False, 0) in core
    assert n.scene_sentence("urgent", "car", 1, "right", "near", False, True, 1) in core
    assert n.scene_sentence("normal", "couch", 1, "right", "medium", False, False, 1) in allp
    assert 40 <= len(core) <= 120 and set(core) <= set(allp) and len(set(allp)) == len(allp)
