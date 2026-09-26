"""Perception model interfaces.

The pipeline depends ONLY on these ABCs — never directly on Ultralytics,
PaddleOCR, Depth Anything, etc. Adapters translate library outputs into the
schemas in `lensvoice.models.schemas`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from lensvoice.models.schemas import (
    DepthResult, Frame, ModelOutput, OCRResult, SceneResult, TrafficSignResult,
)


class VisionModel(ABC):
    """Unified object detection (single 26-class detector)."""

    @abstractmethod
    def predict(self, frame: Frame) -> ModelOutput:
        """Detect objects in a frame."""


class OCRModel(ABC):
    @abstractmethod
    def read(self, frame: Frame) -> list[OCRResult]:
        """Read text present in a frame."""


class TrafficSignModel(ABC):
    @abstractmethod
    def detect(self, frame: Frame) -> list[TrafficSignResult]:
        """Detect and recognise traffic signs in a frame."""


class DepthModel(ABC):
    @abstractmethod
    def estimate(self, frame: Frame) -> DepthResult:
        """Estimate RELATIVE depth for a frame (never claims metric distance)."""


class SceneModel(ABC):
    @abstractmethod
    def classify(self, frame: Frame) -> SceneResult:
        """Classify the scene."""


__all__ = ["VisionModel", "OCRModel", "TrafficSignModel", "DepthModel", "SceneModel"]