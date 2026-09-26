"""Speech synthesizer interface + stub implementations.

Pipeline depends only on TTS. SilentTTS records what would be spoken (used by
mock mode and tests) and is the shipping default until a Piper/system provider
is wired.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class TTS(ABC):
    @abstractmethod
    def speak(self, text: str, level: str = "MEDIUM", interrupt: bool = False) -> None:
        """Synthesise and play `text` (blocking for the stub)."""


class SilentTTS(TTS):
    """Absorbs utterances so mock mode needs no audio device."""

    def __init__(self, persist: list[str] = None) -> None:
        self.history: list[str] = persist if persist is not None else []

    def speak(self, text: str, level: str = "MEDIUM", interrupt: bool = False) -> None:
        self.history.append(text)


__all__ = ["TTS", "SilentTTS"]