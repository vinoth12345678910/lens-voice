"""Time helpers.

A single monotonic clock source is injected through the pipeline so tests can
drive deterministic timestamps. All pipeline modules should use
`monotonic_ms()` (or the injected `Clock`) rather than calling
`time.monotonic()` directly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass


def monotonic_ms() -> float:
    """Monotonic milliseconds since an arbitrary, fixed origin."""
    return time.monotonic() * 1000.0


def wallclock_ms() -> float:
    """Wall-clock milliseconds (for display/telemetry only, not logic)."""
    return time.time() * 1000.0


@dataclass
class Clock:
    """Injectable clock. Replace `now` with a fake to make tests deterministic."""

    now: float = float("nan")

    def __post_init__(self) -> None:
        if self.now != self.now:  # NaN -> use real time
            self.now = monotonic_ms()

    def time_ms(self) -> float:
        return self.now

    def advance(self, ms: float) -> None:
        """Test helper: move the fake clock forward."""
        self.now += ms


__all__ = ["Clock", "monotonic_ms", "wallclock_ms"]