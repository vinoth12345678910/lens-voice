"""Speech synthesizer interface + stub implementations.

Pipeline depends only on TTS. SilentTTS records what would be spoken (used by
mock mode and tests) and is the shipping default. `create_tts` selects the
provider from config (`tts.provider`: silent | sarvam).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from lensvoice.config.settings import TTSConfig


class TTS(ABC):
    # Optional hook: called as on_play(text, level) at the moment audio STARTS
    # playing (lets a viewer show the subtitle exactly when it is heard).
    on_play = None
    last_play_wall: float | None = None     # monotonic time audio last STARTED (set by real providers)

    @abstractmethod
    def speak(self, text: str, level: str = "MEDIUM", interrupt: bool = False) -> None:
        """Synthesise and play `text` (blocking for the stub)."""

    def stop(self) -> None:
        """Cut off speech in progress (no-op for providers that cannot)."""

    def prefetch(self, text: str) -> None:
        """Start preparing audio for `text` in the background (no-op by default)."""

    def prepare(self, text: str) -> bool:
        """Block until audio for `text` is ready; True if speak() will start immediately."""
        return True

    def duration_hint(self, text: str) -> float:
        """Seconds a non-blocking TTS pretends speech takes (0 for real, blocking providers)."""
        return 0.0

    def is_ready(self, text: str) -> bool:
        """True if speak(text) would start without waiting (audio cached / nothing to fetch)."""
        return True


class SilentTTS(TTS):
    """Absorbs utterances so mock mode needs no audio device."""

    def __init__(self, persist: list[str] = None) -> None:
        self.history: list[str] = persist if persist is not None else []

    def speak(self, text: str, level: str = "MEDIUM", interrupt: bool = False) -> None:
        self.history.append(text)

    def duration_hint(self, text: str) -> float:
        return 0.3 * len(text.split()) + 0.25      # ~ real speech pace, so pull-mode timing is realistic


def create_tts(cfg: TTSConfig, persist: list[str] | None = None) -> TTS:
    """Build the configured TTS. Real providers refuse to start when misconfigured."""
    provider = (cfg.provider or "silent").lower()
    if provider == "silent":
        return SilentTTS(persist=persist)
    if provider == "sarvam":
        from lensvoice.audio.sarvam_tts import SarvamTTS   # lazy: no network deps for mock runs
        return SarvamTTS(cfg)
    raise ValueError(f"Unsupported tts.provider: {cfg.provider!r} (silent | sarvam)")


__all__ = ["TTS", "SilentTTS", "create_tts"]