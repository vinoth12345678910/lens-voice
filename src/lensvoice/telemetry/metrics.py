"""Sliding-window metrics collector (no external deps)."""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Optional


class MetricsCollector:
    """Records numeric named metrics; exposes counts, rates and percentiles
    over a sliding window. Thread-tolerant for future multiprocessing."""

    def __init__(self, window_seconds: int = 60, enabled: bool = True):
        self.enabled = enabled
        self.window = window_seconds
        self._samples: dict[str, list[float]] = {}

    def record(self, name: str, value: float, now: Optional[float] = None) -> None:
        if not self.enabled:
            return
        t = time.monotonic() if now is None else now
        samples = self._samples.setdefault(name, [])
        # naive prune to keep O(1)-ish for our scale
        cutoff = t - self.window
        samples[:] = [x for x in samples if x[0] >= cutoff]
        samples.append((t, value))

    def count(self, name: str) -> int:
        if name not in self._samples:
            return 0
        now = time.monotonic()
        cutoff = now - self.window
        return sum(1 for t, _ in self._samples[name] if t >= cutoff)

    def recent(self, name: str, n: int = 5) -> list[float]:
        vals = [v for _, v in self._samples.get(name, [])]
        return vals[-n:]

    def snapshot(self) -> dict:
        out = {}
        for name, samples in self._samples.items():
            now = time.monotonic()
            cutoff = now - self.window
            vals = [v for t, v in samples if t >= cutoff]
            if not vals:
                continue
            stats = {"count": len(vals), "mean": statistics.fmean(vals)}
            if len(vals) > 1:
                stats["p50"] = statistics.median(vals)
                stats["min"] = min(vals)
                stats["max"] = max(vals)
            out[name] = stats
        return out


__all__ = ["MetricsCollector"]