"""Webcam camera source (OpenCV VideoCapture).

Only instantiated when the user explicitly asks for a camera; the mock
pipeline never touches OpenCV capture. Frames arrive as BGR uint8 (from
cv2.VideoCapture) — the perception stage does not assume a channel order.
"""
from __future__ import annotations

import numpy as np

from lensvoice.camera.camera import Camera
from lensvoice.config.settings import CameraConfig
from lensvoice.models.schemas import Frame


class WebcamCamera(Camera):
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self._cap = None
        self._frame_id = 0

    def start(self) -> None:
        try:
            import cv2
        except ImportError as e:
            raise RuntimeError("cv2 not installed; webcam unavailable") from e
        self._cap = cv2.VideoCapture(self.cfg.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera source {self.cfg.source}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)

    def read(self) -> Frame:
        if self._cap is None:
            self.start()
        ok, image = self._cap.read()
        if not ok or image is None:
            raise RuntimeError("camera read failed (source disconnected?)")
        self._frame_id += 1
        import time
        return Frame(image=image, timestamp=time.monotonic() * 1000.0,
                     frame_id=self._frame_id)

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.cfg.width, self.cfg.height)


__all__ = ["WebcamCamera"]