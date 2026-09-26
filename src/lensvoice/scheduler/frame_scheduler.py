"""Per-module frame scheduling.

Deterministic frame-count scheduler: each perception module gets its own
FPS budget (e.g. YOLO 10 Hz, OCR 2 Hz, depth 5 Hz on a 30 fps camera) run
against the incoming frame id. Whether a module runs on a given frame is a
pure function of (frame_id, camera_fps, module_fps) — trivially testable.
"""
from __future__ import annotations

from typing import Dict


class FrameScheduler:
    MODULES = ("yolo", "ocr", "traffic_sign", "depth", "scene")

    def __init__(self, camera_fps: int, fps: Dict[str, int]):
        self.camera_fps = max(1, int(camera_fps))
        self.fps = {m: max(1, int(fps.get(m, 1))) for m in self.MODULES}

    def interval(self, module: str) -> int:
        """Frames between two executions of `module` (>=1)."""
        return max(1, round(self.camera_fps / self.fps[module]))

    def should_run(self, module: str, frame_id: int) -> bool:
        return (frame_id % self.interval(module)) == 0

    def due(self, frame_id: int) -> list[str]:
        return [m for m in self.MODULES if self.should_run(m, frame_id)]


__all__ = ["FrameScheduler"]