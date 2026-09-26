"""DepthAdapter — Socket for a monocular depth model (MiDaS / Depth Anything).

Contract: output is RELATIVE depth in [0, 1] where 1.0 = closest. The pipeline
turns these values into coarse categories (very_near/near/medium/far) and
NEVER converts them to metric distance (no calibration in scope).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from lensvoice.config.settings import ModelConfig
from lensvoice.models.interfaces import DepthModel
from lensvoice.models.schemas import DepthResult, Frame


class DepthAdapter(DepthModel):
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self._model = None
        self._transform = None

    def _ensure(self):
        if self._model is not None:
            return self._model
        if not self.cfg.weight:
            raise RuntimeError("depth enabled but 'models.depth.weight' unset")
        try:
            import torch  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("torch not installed; depth adapter unavailable") from e
        # MiDaS via torch.hub is the reference path; swap in Depth Anything
        # the same way once weights are decided (Friend 1 / later).
        self._model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        device = self.cfg.device
        if device in ("auto", ""):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device).eval()
        self._device = device
        transform = torch.hub.load("intel-isl/MiDaS", "transforms")
        self._transform = transform.default_transform
        return self._model

    def estimate(self, frame: Frame) -> DepthResult:
        import torch
        model = self._ensure()
        with torch.no_grad():
            img = self._transform(frame.image).to(self._device)
            pred = model(img)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=frame.image.shape[:2],
                mode="bicubic", align_corners=False,
            ).squeeze().cpu().numpy()
        depth = (pred - pred.min()) / (pred.max() - pred.min() + 1e-9)
        return DepthResult(timestamp=frame.timestamp, depth=depth,
                           metadata={"calibrated": False})


__all__ = ["DepthAdapter"]