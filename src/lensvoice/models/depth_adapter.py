"""DepthAdapter — Socket for a monocular depth model (MiDaS / Depth Anything).

Contract: output is RELATIVE depth in [0, 1] where 1.0 = closest. The pipeline
turns these values into coarse categories (very_near/near/medium/far) and
NEVER converts them to metric distance (no calibration in scope).

`weight` selects the backend:
  * "depth-anything/Depth-Anything-V2-Small-hf" (any HF "org/name" id, or a local
    directory) -> Depth Anything V2 via transformers. `imgsz` is the input SHORT
    side in pixels (rounded to a multiple of 14; smaller = faster). The map is
    normalised per frame (2nd..98th percentile -> 0..1), so "close" means close
    RELATIVE TO THE REST OF THIS SCENE, not an absolute distance.
  * anything else -> the original MiDaS torch.hub path.
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
        self._hf = False

    def _ensure(self):
        if self._model is not None:
            return self._model
        if not self.cfg.weight:
            raise RuntimeError("depth enabled but 'models.depth.weight' unset")
        try:
            import torch  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("torch not installed; depth adapter unavailable") from e
        if self._is_hf(self.cfg.weight):
            return self._ensure_hf()
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

    @staticmethod
    def _is_hf(weight: str) -> bool:
        import os
        return "depth-anything" in weight.lower() or os.path.isdir(weight)

    def _ensure_hf(self):
        import torch
        from transformers import AutoModelForDepthEstimation
        device = self.cfg.device
        if device in ("auto", ""):
            device = "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu")
        self._model = AutoModelForDepthEstimation.from_pretrained(self.cfg.weight).to(device).eval()
        self._device = device
        self._hf = True
        return self._model

    def _estimate_hf(self, frame: Frame) -> DepthResult:
        import cv2
        import torch
        model = self._ensure()
        h, w = frame.image.shape[:2]
        short = max(14, int(round(self.cfg.imgsz / 14)) * 14)
        scale = short / min(h, w)
        nh, nw = max(14, int(round(h * scale / 14)) * 14), max(14, int(round(w * scale / 14)) * 14)
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)          # pipeline frames are BGR
        x = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self._device)
        with torch.no_grad():
            pred = model(pixel_values=t).predicted_depth             # larger = closer
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False,
            ).squeeze().float().cpu().numpy()
        lo, hi = np.percentile(pred, [2, 98])
        depth = np.clip((pred - lo) / (hi - lo + 1e-9), 0.0, 1.0)
        return DepthResult(timestamp=frame.timestamp, depth=depth,
                           metadata={"calibrated": False, "model": self.cfg.weight})

    def estimate(self, frame: Frame) -> DepthResult:
        import torch
        model = self._ensure()
        if getattr(self, "_hf", False):
            return self._estimate_hf(frame)
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