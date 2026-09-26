"""Model adapters and mock models for the LensVoice perception layer.

`create_models(settings)` is the single factory the pipeline uses. When a
model is `enabled: false` the corresponding mock is returned, so the whole
pipeline runs without GPU, weights, or installed inference libraries.
"""
from __future__ import annotations

from typing import Dict

from lensvoice.config.settings import ModelConfig, Settings
from lensvoice.models.interfaces import (
    DepthModel, OCRModel, SceneModel, TrafficSignModel, VisionModel,
)
from lensvoice.models.mock_models import (
    MockDepthModel, MockOCRModel, MockSceneModel, MockTrafficSignModel,
    MockVisionModel,
)

# Real adapters are imported lazily (their backend deps are optional).
from lensvoice.models.ocr_adapter import PaddleOCRAdapter  # noqa: E402
from lensvoice.models.traffic_sign_adapter import TrafficSignAdapter  # noqa: E402
from lensvoice.models.yolo_adapter import YOLOAdapter  # noqa: E402


def _build(name: str, cfg: ModelConfig, class_names: Dict[int, str]) -> object:
    if name == "yolo":
        return YOLOAdapter(cfg, class_names) if cfg.enabled else MockVisionModel(class_names)
    if name == "ocr":
        return PaddleOCRAdapter(cfg) if cfg.enabled else MockOCRModel()
    if name == "traffic_sign":
        return TrafficSignAdapter(cfg) if cfg.enabled else MockTrafficSignModel()
    if name == "depth":
        from lensvoice.models.depth_adapter import DepthAdapter
        return DepthAdapter(cfg) if cfg.enabled else MockDepthModel()
    if name == "scene":
        from lensvoice.models.scene_adapter import SceneAdapter
        return SceneAdapter(cfg) if cfg.enabled else MockSceneModel()
    raise KeyError(f"unknown model slot: {name}")


def create_models(settings: Settings) -> Dict[str, object]:
    """Build one perception model per slot (mock when disabled)."""
    names = settings.class_names if isinstance(settings.class_names, dict) else settings.class_names
    return {
        key: _build(key, settings.model(key), names)
        for key in ("yolo", "ocr", "traffic_sign", "depth", "scene")
    }


__all__ = [
    "create_models", "YOLOAdapter", "PaddleOCRAdapter", "TrafficSignAdapter",
    "VisionModel", "OCRModel", "TrafficSignModel", "DepthModel", "SceneModel",
]