"""Bounded, priority-ordered speech queue.

Design:
  - enqueue(level, text): critical can interrupt current speech; otherwise
    items queue up to max_queue_size. When full, the lowest-priority new item
    is dropped instead of evicting higher-priority speech.
  - pop() returns the highest-priority item first.
  - dedup + cooldown happen upstream (PriorityScorer / ChangeDetector);
    this queue only orders and bounds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from lensvoice.config.settings import AudioConfig

_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


@dataclass
class Utterance:
    level: str
    text: str

    @property
    def rank(self) -> int:
        return _ORDER.get(self.level, 3)


class PriorityAudioQueue:
    def __init__(self, cfg: AudioConfig, tts):
        self.cfg = cfg
        self.tts = tts
        self._q: list[Utterance] = []

    def enqueue(self, level: str, text: str, interrupt: bool = False) -> bool:
        new = Utterance(level, text)
        if interrupt and self.cfg.interrupt_on_critical:
            self.tts.speak(text, level=level, interrupt=True)
            return True
        if len(self._q) >= self.cfg.max_queue_size:
            # drop the least important (highest rank) item, keep new only if better
            worst = max(self._q, key=lambda u: u.rank)
            if new.rank >= worst.rank and len(self._q) >= self.cfg.max_queue_size:
                return False
            self._q.remove(worst)
        self._q.append(new)
        self._q.sort(key=lambda u: (u.rank, u.text))
        return True

    def pop(self) -> Utterance | None:
        if not self._q:
            return None
        return self._q.pop(0)

    def drain(self) -> None:
        while (u := self.pop()) is not None:
            self.tts.speak(u.text, level=u.level)

    def __len__(self) -> int:
        return len(self._q)


__all__ = ["PriorityAudioQueue", "Utterance"]