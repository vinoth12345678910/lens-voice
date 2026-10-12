"""Webcam / video-file camera source (OpenCV VideoCapture).

Only instantiated when the user explicitly asks for a camera; the mock
pipeline never touches OpenCV capture. Frames arrive as BGR uint8 (from
cv2.VideoCapture) — the perception stage does not assume a channel order.

`camera.source` is a camera index (live webcam) or a video file path:
  * live camera : frames are stamped with the monotonic wall clock, as before.
  * video file  : end of file raises EndOfStream (clean exit, not an error); with
    `realtime` (default) frames are paced at the file's own fps and frames are
    SKIPPED when processing falls behind, exactly like a live camera, so
    narration stays in time with what is on screen. Timestamps are video time.
"""
from __future__ import annotations

import time

import numpy as np

from lensvoice.camera.camera import Camera, EndOfStream
from lensvoice.config.settings import CameraConfig
from lensvoice.models.schemas import Frame


class WebcamCamera(Camera):
    def __init__(self, cfg: CameraConfig, realtime: bool = True):
        self.cfg = cfg
        self.realtime = realtime
        self._cap = None
        self._frame_id = 0
        self._is_file = isinstance(cfg.source, str)
        self._fps = 30.0
        self._idx = 0                 # video-frame index of the next frame to read (files)
        self._t0: float | None = None

    @property
    def is_file(self) -> bool:
        return self._is_file

    @property
    def fps(self) -> float:
        return self._fps

    def start(self) -> None:
        try:
            import cv2
        except ImportError as e:
            raise RuntimeError("cv2 not installed; webcam unavailable") from e
        self._cap = cv2.VideoCapture(self.cfg.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera source {self.cfg.source}")
        if self._is_file:
            self._fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
            self._idx, self._t0 = 0, None
        else:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)

    def _pace_file(self) -> None:
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        due = int((now - self._t0) * self._fps)          # video frame that should be showing now
        while self._idx < due:                            # behind: drop frames to catch up
            if not self._cap.grab():
                raise EndOfStream()
            self._idx += 1
        wait = self._t0 + self._idx / self._fps - time.monotonic()
        if wait > 0:                                      # ahead: wait for the frame's time
            time.sleep(wait)

    def read(self) -> Frame:
        if self._cap is None:
            self.start()
        if self._is_file and self.realtime:
            self._pace_file()
        ok, image = self._cap.read()
        if not ok or image is None:
            if self._is_file:
                raise EndOfStream()
            raise RuntimeError("camera read failed (source disconnected?)")
        self._frame_id += 1
        if self._is_file:
            ts = self._idx / self._fps * 1000.0           # video time (ms)
            self._idx += 1
        else:
            ts = time.monotonic() * 1000.0
        return Frame(image=image, timestamp=ts, frame_id=self._frame_id)

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.cfg.width, self.cfg.height)


__all__ = ["WebcamCamera"]
