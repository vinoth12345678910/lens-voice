"""Unit: Sarvam TTS provider against a local fake server (no internet, no real key)."""
from __future__ import annotations

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from lensvoice.audio.sarvam_tts import SarvamTTS
from lensvoice.audio.tts import SilentTTS, create_tts
from lensvoice.config.settings import TTSConfig

WAV = b"RIFF....WAVEfake"


class _Fake:
    def __init__(self, mode="ok", delay=0.0, limit_first=0):
        self.mode, self.requests, self.delay, self.limit_first = mode, [], delay, limit_first
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((dict(self.headers), body))
                time.sleep(outer.delay)
                if outer.mode == "rate_limited" and len(outer.requests) <= outer.limit_first:
                    self.send_response(429); self.send_header("Retry-After", "1"); self.end_headers()
                    self.wfile.write(b'{"error":"rate_limit_exceeded_error"}'); return
                if outer.mode == "error500":
                    self.send_response(500); self.end_headers(); self.wfile.write(b"boom"); return
                if outer.mode == "legacy_only" and "text" in body:
                    self.send_response(422); self.end_headers(); self.wfile.write(b"bad field"); return
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                self.wfile.write(json.dumps({"audios": [base64.b64encode(WAV).decode()]}).encode())

            def log_message(self, *a):
                pass

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_port}/text-to-speech"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


@pytest.fixture
def fake():
    f = _Fake(); yield f; f.close()


def _tts(url, **kw):
    cfg = TTSConfig(provider="sarvam", voice="priya", pace=1.1, **kw)
    return SarvamTTS(cfg, api_key="test-key-123", player=["true"], url=url)


def test_request_matches_documented_api(fake):
    t = _tts(fake.url)
    assert t._synthesize("bottle, right") == WAV
    headers, body = fake.requests[0]
    headers = {k.lower(): v for k, v in headers.items()}      # header names are case-insensitive
    assert headers["api-subscription-key"] == "test-key-123"
    assert body == {"text": "bottle, right", "language_code": "en-IN", "speaker": "priya",
                    "model": "bulbul:v3", "pace": 1.1, "speech_sample_rate": 22050,
                    "output_audio_codec": "wav"}


def test_underscores_are_spoken_as_spaces(fake):
    t = _tts(fake.url)
    t._synthesize("cell_phone, close, right")
    assert fake.requests[0][1]["text"] == "cell phone, close, right"


def test_repeated_phrase_is_cached(fake):
    t = _tts(fake.url)
    t._synthesize("chair, ahead"); t._synthesize("chair, ahead")
    assert len(fake.requests) == 1


def test_speak_plays_audio_via_player(fake):
    t = _tts(fake.url)
    t.speak("cup, left")          # player is `true`: returns immediately, must not raise
    assert len(fake.requests) == 1


def test_legacy_field_fallback_after_422():
    f = _Fake("legacy_only")
    try:
        t = _tts(f.url)
        assert t._synthesize("hello") == WAV
        assert "inputs" in f.requests[-1][1] and "target_language_code" in f.requests[-1][1]
        assert t._legacy_fields is True
    finally:
        f.close()


def test_api_error_does_not_raise():
    f = _Fake("error500")
    try:
        t = _tts(f.url)
        t.speak("anything")           # skipped, not raised
        assert t.failures == 1
    finally:
        f.close()


def test_unreachable_server_does_not_raise():
    t = _tts("http://127.0.0.1:9/text-to-speech", timeout_s=0.5)
    t.speak("anything")
    assert t.failures == 1


def test_missing_key_is_explicit(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SARVAM_API_KEY"):
        SarvamTTS(TTSConfig(provider="sarvam"), player=["true"])


def test_key_never_in_error_or_repr(fake, monkeypatch):
    t = _tts(fake.url)
    assert "test-key-123" not in repr(t.cfg)


def test_create_tts_selects_provider(monkeypatch):
    assert isinstance(create_tts(TTSConfig()), SilentTTS)
    monkeypatch.setenv("SARVAM_API_KEY", "k")
    monkeypatch.setattr("lensvoice.audio.sarvam_tts._find_player", lambda: ["true"])
    assert isinstance(create_tts(TTSConfig(provider="sarvam")), SarvamTTS)
    with pytest.raises(ValueError):
        create_tts(TTSConfig(provider="nope"))


def test_prefetch_overlaps_and_speak_is_then_instant():
    f = _Fake(delay=0.3)
    try:
        t = _tts(f.url)
        t.prefetch("chair, ahead")
        t0 = time.time()
        assert t.prepare("chair, ahead") is True            # waits for the in-flight fetch, no second request
        assert 0.2 < time.time() - t0 < 1.5
        t1 = time.time()
        t.speak("chair, ahead")                              # cached: no network wait
        assert time.time() - t1 < 0.2 and len(f.requests) == 1
    finally:
        f.close()


def test_concurrent_requests_for_same_phrase_hit_the_api_once():
    f = _Fake(delay=0.3)
    try:
        t = _tts(f.url)
        ths = [threading.Thread(target=t._synthesize, args=("cup, left",)) for _ in range(4)]
        [x.start() for x in ths]; [x.join() for x in ths]
        assert len(f.requests) == 1
    finally:
        f.close()


def test_disk_cache_survives_restart(tmp_path):
    f = _Fake()
    try:
        a = _tts(f.url, cache_dir=str(tmp_path))
        assert a._synthesize("book, right") == WAV
        b = _tts(f.url, cache_dir=str(tmp_path))             # a new process would start like this
        assert b._synthesize("book, right") == WAV
        assert len(f.requests) == 1 and b.api_calls == 0
    finally:
        f.close()


def test_playback_start_records_voice_lag_and_fires_hook(fake):
    t = _tts(fake.url)
    seen = []
    t.on_play = lambda text, level: seen.append((text, level))
    t.enq_time = time.monotonic() - 0.5
    t.speak("mouse, ahead", level="LOW")
    assert seen == [("mouse, ahead", "LOW")] and 450 < t.lags[-1] < 3000


def test_rate_limit_backs_off_and_retries():
    f = _Fake("rate_limited", limit_first=1)                 # first request -> 429, then fine
    try:
        t = _tts(f.url, max_rps=50.0)
        t0 = time.time()
        assert t._synthesize("hello") == WAV                 # retried after the backoff
        assert t.rate_limited == 1 and len(f.requests) == 2
        assert time.time() - t0 >= 0.9                       # honoured Retry-After ~1 s
        assert t._min_interval > t._base_interval            # rate was halved
    finally:
        f.close()


def test_requests_are_spaced_by_the_global_limiter():
    f = _Fake()
    try:
        t = _tts(f.url, max_rps=5.0)                         # >= 0.2 s between requests
        t0 = time.time()
        ths = [threading.Thread(target=t._synthesize, args=(f"phrase {i}",)) for i in range(4)]
        [x.start() for x in ths]; [x.join() for x in ths]
        assert time.time() - t0 >= 0.55 and len(f.requests) == 4
    finally:
        f.close()


def test_slow_fetch_falls_back_to_system_voice_without_waiting(tmp_path):
    f = _Fake(delay=1.5)                                     # API is slow
    try:
        t = _tts(f.url, fallback_local_ms=300)
        spoken = []
        t._say = "true"                                       # stand-in for macOS `say` (exits at once)
        t._begin = lambda args, gen, text, level: (spoken.append((args[0], text)) or None)
        t0 = time.time()
        t.speak("tv_monitor, close, left", level="CRITICAL")     # urgent: the shortest wait
        assert time.time() - t0 < 0.9                         # did not wait for the 1.5 s fetch
        assert spoken == [("true", "tv_monitor, close, left")]
        time.sleep(1.8)
        assert t._synthesize("tv_monitor, close, left") == WAV    # the fetch finished and is cached
    finally:
        f.close()


def test_warm_up_yields_to_live_and_stops_when_rate_limited():
    f = _Fake("rate_limited", limit_first=10**6)             # API always says 429
    try:
        t = _tts(f.url, max_rps=50.0)
        t.rate_limited = 0
        t._last_live = 0.0
        done = threading.Event()
        orig = t._warm_run
        t._warm_run = lambda ph: (orig(ph), done.set())
        t.warm([f"p{i}" for i in range(20)])
        assert done.wait(60), "warm-up never gave up"
        assert t.rate_limited >= 3 and len(f.requests) < 20  # gave up after repeated 429s, did not hammer
    finally:
        f.close()


def test_routine_sentences_may_wait_longer_than_urgent_ones_for_sarvam_audio():
    f = _Fake(delay=0.45)                                    # API takes 0.45 s (the fake serves one request at a time)
    try:
        t = _tts(f.url, fallback_local_ms=300, max_rps=50.0)      # fast limiter: measure only the wait budgets
        said = []
        t._say = "true"
        t._begin = lambda args, gen, text, level: (said.append(("local", level)) or None)
        t._play = lambda audio, gen, text="", level="MEDIUM": said.append(("sarvam", level))
        t.speak("person in your path, close", level="CRITICAL")   # budget 0.3 s < 0.45 s -> system voice now
        t.speak("chair, far away, left", level="MEDIUM")          # budget 0.9 s -> waits for Sarvam
        t.speak("tv monitor left, couch right", level="LOW")      # budget 1.8 s -> waits for Sarvam
        assert said == [("local", "CRITICAL"), ("sarvam", "MEDIUM"), ("sarvam", "LOW")]
    finally:
        f.close()
