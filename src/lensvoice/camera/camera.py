"""Camera interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

from lensvoice.models.schemas import Frame


class EndOfStream(Exception):
    """Raised by a camera when no more frames exist (bounded mocks)."""


class Camera(ABC):
    @abstractmethod
    def start(self) -> None:
        """Open the source (no-op for mock cameras)."""

    @abstractmethod
    def read(self) -> Frame:
        """Grab the next frame. Blocks until one is available."""

    @abstractmethod
    def stop(self) -> None:
        """Release the source."""

    @property
    def resolution(self) -> tuple[int, int]:
        return (0, 0)


__all__ = ["Camera", "EndOfStream"]