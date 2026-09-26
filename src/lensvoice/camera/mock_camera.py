"""Mock camera: synthetic frames with no hardware.

Deterministic per resolution; queries only that the same frame is produced
again. Used by the pipeline's mock mode and by integration tests.
"""
from __future__ import annotations

import numpy as np

from lensvoice.camera.camera import Camera, EndOfStream
from lensvoice.config.settings import CameraConfig
from lensvoice.models.schemas import Frame


class MockCamera(Camera):
    def __init__(self, cfg: CameraConfig, max_frames: int | None = None):
        self.cfg = cfg
        self.max_frames = max_frames
        self._frame_id = 0
        self._started = False

    def start(self) -> None:
        self._started = True

    def read(self) -> Frame:
        if self.max_frames is not None and self._frame_id >= self.max_frames:
            raise EndOfStream()
        self._frame_id += 1
        # Noise-free synthetic gradient frame so mocks stay deterministic.
        h, w = self.cfg.height, self.cfg.width
        yy, xx = np.mgrid[0:h, 0:w]
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[..., 0] = (xx / w * 255).astype(np.uint8)
        img[..., 1] = (yy / h * 255).astype(np.uint8)
        img[..., 2] = 128
        import time
        return Frame(image=img, timestamp=time.monotonic() * 1000.0,
                     frame_id=self._frame_id)

    def stop(self) -> None:
        self._started = False

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.cfg.width, self.cfg.height)


__all__ = ["MockCamera"]