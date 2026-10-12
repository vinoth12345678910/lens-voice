"""Sarvam AI text-to-speech provider (bulbul) with local playback.

Sarvam TTS is a cloud API (needs internet + an API key); "local" here means the
whole LensVoice pipeline and audio playback run on this machine. The key is read
from the environment variable named by `tts.api_key_env` (default SARVAM_API_KEY)
and is never stored, logged, or put in a config file.

  POST https://api.sarvam.ai/text-to-speech        header: api-subscription-key
  body  {text, language_code, speaker, model, pace, speech_sample_rate, output_audio_codec}
  reply {"audios": [<base64 WAV>]}

Latency design (a sentence must be heard while it is still true):
  * prefetch(text): the fetch starts in a background pool the moment a sentence is
    queued, overlapping with whatever is playing now;
  * persistent HTTPS connections per worker thread;
  * memory + optional disk cache: a phrase is paid for and fetched once, ever;
  * hard cap: if audio is not ready within tts.fallback_local_ms, the sentence is
    spoken AT ONCE with the system voice (macOS `say`); the fetch continues into the
    cache, so next time the same sentence is heard in the Sarvam voice.

Rate limits: every request passes one global limiter (starts at tts.max_rps). An HTTP
429 halves the rate and pauses ALL requests (honouring Retry-After); the rate then
recovers slowly. warm() runs on one low-priority thread that yields to live sentences
and stops after repeated 429s, so it can never starve the live voice.

speak() is blocking (like every TTS here). It never raises on network/API problems:
a failed sentence is skipped (or spoken locally) and logged. stop() cuts off the
current utterance (CRITICAL interrupt).

Smoke test (speaks one sentence, prints latency):
    SARVAM_API_KEY=... venv/bin/python -m lensvoice.audio.sarvam_tts "bottle, right"
"""
from __future__ import annotations

import base64
import hashlib
import http.client
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.parse
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Optional

from lensvoice.audio.tts import TTS
from lensvoice.config.settings import TTSConfig

log = logging.getLogger("lensvoice.sarvam")

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
_CACHE_MAX = 1024
_PLAYERS = (("afplay",), ("paplay",), ("aplay", "-q"),
            ("ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"))
_MAX_INTERVAL = 6.0
# how long a sentence may wait for Sarvam audio before the system voice takes over, as a
# multiple of tts.fallback_local_ms: urgent warnings must not wait, a scene summary can
_WAIT_MULT = {"CRITICAL": 1, "HIGH": 1, "MEDIUM": 3, "LOW": 6}


class _HTTPStatus(Exception):
    def __init__(self, code: int, body: bytes, retry_after: float = 0.0):
        super().__init__(f"HTTP {code}: {body!r}")
        self.code = code
        self.retry_after = retry_after


def _find_player() -> list[str]:
    for cmd in _PLAYERS:
        if shutil.which(cmd[0]):
            return list(cmd)
    raise RuntimeError("no audio player found (need afplay / paplay / aplay / ffplay)")


class SarvamTTS(TTS):
    def __init__(self, cfg: TTSConfig, api_key: Optional[str] = None,
                 player: Optional[list[str]] = None, url: str = SARVAM_TTS_URL):
        key = api_key or os.environ.get(cfg.api_key_env, "")
        if not key:
            raise RuntimeError(
                f"tts.provider is 'sarvam' but environment variable {cfg.api_key_env} is not set. "
                f"Run:  export {cfg.api_key_env}=...   (never put the key in a file)")
        self._api_key = key
        self.cfg = cfg
        self.url = url
        self._u = urllib.parse.urlparse(url)
        self.speaker = cfg.voice or "priya"
        self._player = player if player is not None else _find_player()
        self._cache: dict[tuple, bytes] = {}
        self._inflight: dict[tuple, Future] = {}
        self._cache_lock = threading.Lock()
        self._cache_dir = os.path.expanduser(cfg.cache_dir) if cfg.cache_dir else ""
        if self._cache_dir:
            os.makedirs(self._cache_dir, exist_ok=True)
        self._pool: Optional[ThreadPoolExecutor] = None
        self._local = threading.local()          # one persistent connection per thread
        self._legacy_fields = False              # flips on if the API rejects the documented field names
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._gen = 0                            # bumped by stop(); a stale speak() skips playback
        # global request limiter
        self._rate_lock = threading.Lock()
        self._min_interval = 1.0 / max(0.1, cfg.max_rps)
        self._base_interval = self._min_interval
        self._next_slot = 0.0
        self._blocked_until = 0.0
        self._ok_streak = 0
        self._last_live = 0.0                    # last time a live (non warm-up) sentence needed audio
        self._say = shutil.which("say") if cfg.fallback_local_ms > 0 else None
        # counters / telemetry
        self.failures = 0
        self.api_calls = 0
        self.cache_hits = 0
        self.rate_limited = 0                    # HTTP 429 responses seen
        self.fallback_count = 0                  # sentences spoken with the system voice
        self.enq_time: Optional[float] = None    # set by the audio queue before speak()
        self.lags: list[float] = []              # ms from "sentence queued" to "audio started"

    # -- TTS ----------------------------------------------------------------
    def speak(self, text: str, level: str = "MEDIUM", interrupt: bool = False) -> None:
        if not text or not text.strip():
            return
        if interrupt:
            self.stop()
        gen = self._gen
        audio = self._wait_audio(text, _WAIT_MULT.get(level, 1))
        if gen != self._gen:
            return
        if audio is not None:
            self._play(audio, gen, text, level)
        elif self._say:                          # not ready in time (or failed): speak it NOW
            self._speak_local(text, gen, level)

    def stop(self) -> None:
        with self._lock:
            self._gen += 1
            proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass

    def prefetch(self, text: str) -> None:
        """Start fetching audio now so it is ready when the sentence's turn comes."""
        if self.cfg.prefetch and text.strip():
            self._lookup(text)

    def prepare(self, text: str) -> bool:
        """Wait for the audio, but never longer than the local-fallback budget."""
        return self._wait_audio(text) is not None

    def is_ready(self, text: str) -> bool:
        """Cached in memory or on disk (no network wait)."""
        key = self._cache_key(text.replace("_", " "))
        with self._cache_lock:
            if key in self._cache:
                return True
        return bool(self._cache_dir) and os.path.exists(self._disk_path(key))

    def warm(self, phrases: list[str]) -> None:
        """Pre-fetch phrases politely in the background (see class docstring)."""
        threading.Thread(target=self._warm_run, args=(list(phrases),),
                         name="sarvam-warm", daemon=True).start()

    # -- cache ------------------------------------------------------------------
    def _cache_key(self, text: str) -> tuple:
        c = self.cfg
        return (c.language, self.speaker, c.model, c.pace, c.sample_rate, text)

    def _disk_path(self, key: tuple) -> str:
        return os.path.join(self._cache_dir, hashlib.sha1(repr(key).encode()).hexdigest() + ".wav")

    def _put(self, key: tuple, audio: bytes) -> None:
        if len(self._cache) >= _CACHE_MAX:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = audio

    def _ensure_pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=max(1, self.cfg.prefetch_workers),
                                            thread_name_prefix="sarvam-synth")
        return self._pool

    def _lookup(self, text: str, live: bool = True) -> tuple[Optional[bytes], Optional[Future]]:
        """(audio, None) if cached; else (None, future) for a fetch that is running or just started."""
        text = text.replace("_", " ")          # class names like cell_phone / tv_monitor are spoken as words
        if live:
            self._last_live = time.monotonic()
        key = self._cache_key(text)
        with self._cache_lock:
            hit = self._cache.get(key)
            if hit is not None:
                self.cache_hits += 1
                return hit, None
            fut = self._inflight.get(key)
            if fut is not None:
                return None, fut
            if self._cache_dir:
                try:
                    with open(self._disk_path(key), "rb") as f:
                        disk = f.read()
                    self._put(key, disk)
                    self.cache_hits += 1
                    return disk, None
                except OSError:
                    pass
            fut = self._inflight[key] = Future()
        self._ensure_pool().submit(self._fetch_store, key, text, fut, live)
        return None, fut

    def _wait_audio(self, text: str, mult: float = 1.0) -> Optional[bytes]:
        audio, fut = self._lookup(text)
        if audio is not None or fut is None:
            return audio
        budget = self.cfg.fallback_local_ms * mult / 1000.0 if self._say else None
        try:
            return fut.result(timeout=budget)
        except FutureTimeout:
            return None                          # still fetching: caller falls back to the system voice

    def _synthesize(self, text: str) -> Optional[bytes]:
        """Blocking fetch-or-cache (used by the smoke test and unit tests)."""
        audio, fut = self._lookup(text)
        return audio if audio is not None else fut.result()

    def _fetch_store(self, key: tuple, text: str, fut: Future, live: bool) -> None:
        audio = None
        try:
            audio = self._fetch(text, live)
            if audio is not None:
                with self._cache_lock:
                    self._put(key, audio)
                if self._cache_dir:
                    try:
                        tmp = self._disk_path(key) + ".part"
                        with open(tmp, "wb") as f:
                            f.write(audio)
                        os.replace(tmp, self._disk_path(key))
                    except OSError:
                        pass
        except Exception as e:            # never leave a waiter hanging
            self._fail(f"{type(e).__name__}: {str(e)[:200]}")
        finally:
            with self._cache_lock:
                self._inflight.pop(key, None)
            fut.set_result(audio)

    # -- request limiter ---------------------------------------------------------
    def _acquire(self) -> None:
        with self._rate_lock:
            now = time.monotonic()
            start = max(now, self._next_slot, self._blocked_until)
            self._next_slot = start + self._min_interval
        if start > now:
            time.sleep(start - now)

    def _on_rate_limited(self, retry_after: float) -> float:
        with self._rate_lock:
            self.rate_limited += 1
            self._ok_streak = 0
            self._min_interval = min(_MAX_INTERVAL, self._min_interval * 2.0)
            wait = max(retry_after, self._min_interval, 1.0)
            self._blocked_until = max(self._blocked_until, time.monotonic() + wait)
            return wait

    def _on_ok(self) -> None:
        with self._rate_lock:
            self._ok_streak += 1
            if self._ok_streak % 10 == 0:
                self._min_interval = max(self._base_interval, self._min_interval * 0.8)

    # -- synthesis ------------------------------------------------------------
    def _body(self, text: str) -> dict:
        c = self.cfg
        body = {"text": text, "language_code": c.language, "speaker": self.speaker,
                "model": c.model, "pace": c.pace, "speech_sample_rate": c.sample_rate,
                "output_audio_codec": "wav"}
        if self._legacy_fields:   # field names used by the earlier LensVoice mobile app
            body = {"inputs": [text], "target_language_code": c.language,
                    "speaker": self.speaker, "model": c.model}
        return body

    def _conn(self) -> http.client.HTTPConnection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            cls = http.client.HTTPSConnection if self._u.scheme == "https" else http.client.HTTPConnection
            conn = cls(self._u.hostname, self._u.port, timeout=self.cfg.timeout_s)
            self._local.conn = conn
        return conn

    def _post(self, body: dict) -> bytes:
        payload = json.dumps(body).encode()
        headers = {"api-subscription-key": self._api_key, "Content-Type": "application/json"}
        for attempt in (0, 1):
            conn = self._conn()
            try:
                conn.request("POST", self._u.path or "/", body=payload, headers=headers)
                resp = conn.getresponse()
                data = resp.read()
                break
            except TimeoutError:
                conn.close(); self._local.conn = None
                raise
            except (http.client.HTTPException, OSError):    # stale keep-alive connection: reconnect once
                conn.close(); self._local.conn = None
                if attempt:
                    raise
        if resp.status != 200:
            try:
                ra = float(resp.getheader("Retry-After") or 0.0)
            except ValueError:
                ra = 0.0
            raise _HTTPStatus(resp.status, data[:200], ra)
        audios = json.loads(data).get("audios") or []
        if not audios:
            raise ValueError("empty 'audios' in response")
        return base64.b64decode(audios[0])

    def _fetch(self, text: str, live: bool = True) -> Optional[bytes]:
        for attempt in range(3):
            self._acquire()
            self.api_calls += 1
            try:
                try:
                    audio = self._post(self._body(text))
                except _HTTPStatus as e:
                    if e.code == 422 and not self._legacy_fields:
                        self._legacy_fields = True          # retry once with the legacy field names
                        audio = self._post(self._body(text))
                    else:
                        raise
            except _HTTPStatus as e:
                if e.code == 429:
                    wait = self._on_rate_limited(e.retry_after)
                    if live and attempt < 2 and wait <= 3.0:
                        continue                            # the limiter holds us back; try again
                    self._fail("HTTP 429 rate limit (backing off)")
                    return None
                self._fail(str(e))
                return None
            except Exception as e:       # network down, timeout, bad JSON ...
                self._fail(f"{type(e).__name__}: {str(e)[:200]}")
                return None
            self.failures = 0
            self._on_ok()
            return audio
        return None

    def _fail(self, msg: str) -> None:
        self.failures += 1
        if self.failures <= 2 or self.failures % 25 == 0:
            log.warning("sarvam tts problem (%d in a row): %s", self.failures, msg)

    # -- warm-up --------------------------------------------------------------------
    def _warm_run(self, phrases: list[str]) -> None:
        strikes = 0
        for ph in phrases:
            while time.monotonic() - self._last_live < 3.0 or time.monotonic() < self._blocked_until:
                time.sleep(0.5)                  # live sentences and rate-limit pauses come first
            before = self.rate_limited
            audio, fut = self._lookup(ph, live=False)
            if fut is not None:
                fut.result()
            strikes = strikes + 1 if self.rate_limited > before else 0
            if strikes >= 3:
                log.warning("warm-up stopped: rate limited by the API (it will fill up during use instead)")
                return
            time.sleep(max(1.2, self._min_interval))

    # -- playback ---------------------------------------------------------------
    def _begin(self, popen_args: list[str], gen: int, text: str, level: str):
        with self._lock:
            if gen != self._gen:
                return None
            self._proc = subprocess.Popen(popen_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            proc = self._proc
        self.last_play_wall = time.monotonic()
        if self.enq_time is not None:
            self.lags.append((time.monotonic() - self.enq_time) * 1000.0)
            del self.lags[:-2000]
        if self.on_play:
            try:
                self.on_play(text, level)
            except Exception:        # a display problem must never stop speech
                pass
        return proc

    def _play(self, audio: bytes, gen: int, text: str = "", level: str = "MEDIUM") -> None:
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="lv_tts_")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(audio)
            proc = self._begin(self._player + [path], gen, text, level)
            if proc is not None:
                proc.wait()
        except OSError as e:
            log.warning("audio playback failed: %s", e)
        finally:
            with self._lock:
                self._proc = None
            try:
                os.unlink(path)
            except OSError:
                pass

    def _speak_local(self, text: str, gen: int, level: str) -> None:
        """Immediate fallback: the system voice, so a sentence is never late."""
        cmd = [self._say, "-r", str(self.cfg.fallback_rate)]
        if self.cfg.fallback_voice:
            cmd += ["-v", self.cfg.fallback_voice]
        try:
            proc = self._begin(cmd + [text.replace("_", " ")], gen, text, level)
            if proc is not None:
                self.fallback_count += 1
                proc.wait()
        except OSError as e:
            log.warning("local voice failed: %s", e)
        finally:
            with self._lock:
                self._proc = None


def _main() -> int:
    import argparse
    from lensvoice.config.settings import load_settings
    ap = argparse.ArgumentParser(description="Speak one sentence through Sarvam TTS")
    ap.add_argument("text", nargs="?", default="Bottle, close, on your right")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    tts = SarvamTTS(load_settings(a.config).tts)
    t0 = time.monotonic()
    audio = tts._synthesize(a.text)
    if audio is None:
        print("synthesis FAILED (see warning above)")
        return 1
    print(f"synthesis ok: {len(audio)} bytes in {(time.monotonic() - t0) * 1000:.0f} ms")
    tts._play(audio, tts._gen)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(_main())

__all__ = ["SarvamTTS", "SARVAM_TTS_URL"]
