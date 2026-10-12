"""Bounded, priority-ordered speech queue.

Design:
  - enqueue(level, text): critical can interrupt current speech; otherwise
    items queue up to max_queue_size. When full, the lowest-priority new item
    is dropped instead of evicting higher-priority speech.
  - pop() returns the highest-priority item first (first-in-first-out within a level).
  - dedup + cooldown happen upstream (PriorityScorer / ChangeDetector);
    this queue only orders and bounds.
  - Live mode (start_worker): a background thread speaks the queue while the
    camera loop keeps running. Queued speech older than audio.max_age_ms is
    dropped (stale narration is wrong narration) and a CRITICAL arrival cuts off
    lower-priority speech in progress when audio.interrupt_on_critical is set.
    Without a worker the queue behaves exactly as before (drain() at shutdown).
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field

from lensvoice.config.settings import AudioConfig

_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
# the depth bucket flickers between neighbours, so repeats are judged without the distance words
_DIST_WORDS = re.compile(r",?\s*(very close|close|a short way off|far away)\b")


def _story(text: str) -> str:
    return _DIST_WORDS.sub("", text).strip(" ,")
log = logging.getLogger("lensvoice.audio")


@dataclass
class Utterance:
    level: str
    text: str
    t: float = field(default_factory=time.monotonic, compare=False)

    @property
    def rank(self) -> int:
        return _ORDER.get(self.level, 3)


class PriorityAudioQueue:
    def __init__(self, cfg: AudioConfig, tts):
        self.cfg = cfg
        self.tts = tts
        self._q: list[Utterance] = []
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._speaking_rank: int | None = None
        self._recent: dict[str, float] = {}      # sentence -> last accepted time
        self.stale_dropped = 0
        self.dupes_dropped = 0
        self.low_dropped = 0
        self._last_low = -1e9
        self.spoken = 0

    def enqueue(self, level: str, text: str, interrupt: bool = False) -> bool:
        new = Utterance(level, text)
        if interrupt and self.cfg.interrupt_on_critical:
            self.tts.speak(text, level=level, interrupt=True)
            return True
        with self._lock:
            if self.cfg.dedupe_ms > 0:
                last = self._recent.get(_story(text))
                if last is not None and (new.t - last) * 1000.0 < self.cfg.dedupe_ms:
                    self.dupes_dropped += 1
                    return False
            if self.cfg.low_gap_ms > 0 and new.level == "LOW":
                if (new.t - self._last_low) * 1000.0 < self.cfg.low_gap_ms:
                    self.low_dropped += 1
                    return False
            if len(self._q) >= self.cfg.max_queue_size:
                # drop the least important (highest rank) item, keep new only if better
                worst = max(self._q, key=lambda u: u.rank)
                if new.rank >= worst.rank and len(self._q) >= self.cfg.max_queue_size:
                    return False
                self._q.remove(worst)
            self._q.append(new)
            self._q.sort(key=lambda u: (u.rank, u.t))
            self._recent[_story(text)] = new.t
            if new.level == "LOW":
                self._last_low = new.t
            if len(self._recent) > 256:
                self._recent.pop(next(iter(self._recent)))
            cut = (self._worker is not None and new.rank == 0 and self.cfg.interrupt_on_critical
                   and self._speaking_rank is not None and self._speaking_rank > 0)
        if self._worker is not None and (new.rank <= 1 or len(self) <= 2):
            self.tts.prefetch(text)          # start fetching audio now, overlapping with current speech
        if cut:
            self.tts.stop()      # lower-priority speech in progress is cut; the worker then speaks the critical item
        return True

    def pop(self) -> Utterance | None:
        with self._lock:
            if not self._q:
                return None
            return self._q.pop(0)

    def drain(self) -> None:
        while (u := self.pop()) is not None:
            self.tts.speak(u.text, level=u.level)

    def __len__(self) -> int:
        with self._lock:
            return len(self._q)

    # -- live playback worker ------------------------------------------------
    def start_worker(self) -> None:
        if self._worker is not None:
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, name="lensvoice-audio", daemon=True)
        self._worker.start()

    def stop_worker(self, discard: bool = True) -> None:
        """Stop speaking. Pending items are discarded by default (shutdown must be prompt)."""
        if self._worker is None:
            return
        self._stop.set()
        self.tts.stop()
        self._worker.join(timeout=3.0)
        self._worker = None
        if discard:
            with self._lock:
                self._q.clear()

    def _fresh(self, u: Utterance) -> bool:
        max_age = self.cfg.max_age_ms
        if max_age > 0 and (time.monotonic() - u.t) * 1000.0 > max_age:
            self.stale_dropped += 1
            return False
        return True

    def _next_fresh(self) -> Utterance | None:
        while (u := self.pop()) is not None:
            if self._fresh(u):
                return u
        return None

    def _ready(self, u: Utterance) -> Utterance | None:
        """Wait until `u` has audio, then re-check: a more urgent sentence may have arrived
        and `u` may now be stale. Returns the utterance to speak, or None."""
        while True:
            self.tts.prepare(u.text)                 # instant when prefetched / cached
            with self._lock:
                if self._q and self._q[0].rank < u.rank:      # something more urgent is waiting
                    self._q.append(u)
                    self._q.sort(key=lambda x: (x.rank, x.t))
                    u = self._q.pop(0)
                    if not self._fresh(u):
                        return None
                    continue
            return u if self._fresh(u) else None

    def _run(self) -> None:
        while not self._stop.is_set():
            u = self._next_fresh()
            if u is None:
                time.sleep(0.01)
                continue
            self._speaking_rank = u.rank
            try:
                u = self._ready(u)
                if u is None:
                    continue
                self._speaking_rank = u.rank
                self.tts.enq_time = u.t
                self.tts.speak(u.text, level=u.level)
                self.spoken += 1
            except Exception as e:      # a TTS failure must never kill the worker
                log.warning("tts.speak failed: %s", e)
            finally:
                self._speaking_rank = None


__all__ = ["PriorityAudioQueue", "Utterance"]
