"""LiveViewer — watch the pipeline work: boxes, distance, depth and subtitles.

Plugs into Pipeline(on_frame=viewer). Per frame it draws, on the video:
  * a box per tracked object: class, track id, confidence, distance bucket, zone,
    movement; boxes that are close AND inside the "in your path" band turn red;
  * faint guide lines for the left / ahead / right zones and the in-path band;
  * a depth inset (red = close, blue = far; press 'd' for a full overlay);
  * a HUD (fps, YOLO ms, tracks, depth source) and bottom subtitles of what is
    being SPOKEN (shown when the audio starts, via tts.on_play).
Keys: q / Esc quit, d depth overlay, p pause.  Optional --record writes an .mp4.

`render()` is pure (no window) so it is unit-testable; the window is only opened
by __call__ when show=True. Must run on the main thread (macOS Cocoa).
"""
from __future__ import annotations

import collections
import time
from typing import Optional

import cv2
import numpy as np

_LEVEL_COLOR = {"CRITICAL": (60, 60, 255), "HIGH": (0, 140, 255),
                "MEDIUM": (0, 200, 255), "LOW": (240, 240, 240)}
_GROUP_COLOR = {   # BGR
    "road": (255, 200, 0),          # person, vehicles, bicycle
    "furniture": (80, 200, 80),     # chair table couch bed bench
    "item": (0, 215, 255),          # everything small
}
_ROAD = {"person", "car", "motorcycle", "bus", "truck", "autorickshaw", "bicycle"}
_FURN = {"chair", "table", "couch", "bed", "bench"}
_LEVEL_SHORT = {"CRITICAL": "CRIT", "HIGH": "HIGH", "MEDIUM": "MED", "LOW": "LOW"}
_CLOSE = ("very_near", "near")
_HOLD_S = 1.5       # keep the last boxes this long if YOLO has not refreshed them


def _group(name: str) -> str:
    return "road" if name in _ROAD else "furniture" if name in _FURN else "item"


class LiveViewer:
    def __init__(self, spatial_cfg=None, safety_cfg=None, depth_label: str = "mock",
                 voice_label: str = "silent", show: bool = True,
                 record: Optional[str] = None, record_fps: float = 15.0,
                 title: str = "LensVoice", max_size: tuple[int, int] = (1280, 860)):
        self.spatial, self.safety = spatial_cfg, safety_cfg
        self.depth_label, self.voice_label = depth_label, voice_label
        self.show, self.record_path, self.record_fps = show, record, record_fps
        self.title, self.max_w, self.max_h = title, *max_size
        self.objs: list = []                       # [(DetectedObject, ObjectState|None)]
        self._objs_t = 0.0
        self.depth: Optional[np.ndarray] = None
        self.subs: collections.deque = collections.deque(maxlen=4)   # (t_start, level, text)
        self.show_depth = False
        self._writer = None
        self._win = False
        self._times: collections.deque = collections.deque(maxlen=30)
        self._info: dict = {}
        self.on_summary = None                      # () -> None, called on key 's' (scene summary)
        self.lag_fn = None                          # () -> last voice lag in ms (or None)

    # -- inputs -----------------------------------------------------------------
    def on_spoken(self, text: str, level: str = "LOW") -> None:
        """Thread-safe: call when a sentence starts being spoken (or queued, if silent)."""
        now, text = time.monotonic(), text.replace("_", " ")
        if any(t == text and now - ts < 6.0 for ts, _, t in self.subs):
            return                                 # ignore an immediate repeat (silent mode shows every produced sentence)
        self.subs.append((now, level, text))

    def __call__(self, frame, ctx, info: dict) -> None:
        now = time.monotonic()
        self._times.append(now)
        if info.get("detections_updated"):
            self.objs = [(d, ctx.object_states.get(d.track_id)) for d in ctx.objects]
            self._objs_t = now
        if info.get("depth") is not None:
            self.depth = info["depth"].depth
        self._info = info
        img = self.render(frame.image)
        if self.record_path:
            self._write(img)
        if self.show:
            self._show(img)

    # -- drawing ------------------------------------------------------------------
    def render(self, image: np.ndarray, now: Optional[float] = None) -> np.ndarray:
        now = time.monotonic() if now is None else now
        out = image.copy()
        h, w = out.shape[:2]
        k = max(0.5, max(h, w) / 1100.0)                 # text/line scale from resolution
        th = max(1, int(round(1.5 * k)))

        if self.show_depth and self.depth is not None:
            out = cv2.addWeighted(out, 0.45, self._colormap(self.depth, (w, h)), 0.55, 0)

        self._guides(out, k)
        if now - self._objs_t <= _HOLD_S:
            for det, st in self.objs:
                self._box(out, det, st, k, th)
        if self.depth is not None and not self.show_depth:
            self._inset(out, k)
        self._hud(out, k)
        self._subtitles(out, now, k)

        scale = min(self.max_w / w, self.max_h / h, 1.0)
        if scale < 1.0:
            out = cv2.resize(out, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return out

    @staticmethod
    def _colormap(depth: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        cm = cv2.applyColorMap((np.clip(depth, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        return cv2.resize(cm, size, interpolation=cv2.INTER_LINEAR)

    def _text(self, img, txt, org, k, color=(255, 255, 255), bg=(0, 0, 0), alpha=0.6, scale=0.55):
        font, s, th = cv2.FONT_HERSHEY_SIMPLEX, scale * k, max(1, int(round(k)))
        (tw, tht), base = cv2.getTextSize(txt, font, s, th)
        x, y = org
        x, y = max(0, min(x, img.shape[1] - tw - 6)), max(tht + 6, y)
        ov = img.copy()
        cv2.rectangle(ov, (x - 3, y - tht - 4), (x + tw + 3, y + base + 2), bg, -1)
        cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
        cv2.putText(img, txt, (x, y), font, s, color, th, cv2.LINE_AA)
        return tht + base + 8

    def _guides(self, img, k):
        h, w = img.shape[:2]
        lb = getattr(self.spatial, "left_boundary", 0.333)
        rb = getattr(self.spatial, "right_boundary", 0.666)
        band = getattr(self.safety, "center_band", 0.5)
        ov = img.copy()
        for fx in (lb, rb):
            cv2.line(ov, (int(fx * w), 0), (int(fx * w), h), (255, 255, 255), 1)
        x0, x1 = int((0.5 - band / 2) * w), int((0.5 + band / 2) * w)
        cv2.rectangle(ov, (x0, 0), (x1, h - 1), (0, 255, 255), 1)
        cv2.addWeighted(ov, 0.25, img, 0.75, 0, img)
        for fx, name in ((lb / 2, "left"), (0.5, "ahead"), ((1 + rb) / 2, "right")):
            cv2.putText(img, name, (int(fx * w) - int(18 * k), int(h * 0.11)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4 * k, (200, 200, 200), 1, cv2.LINE_AA)

    def _in_path(self, det, w) -> bool:
        band = getattr(self.safety, "center_band", 0.5)
        cx = det.bbox.center_x / max(1.0, w)
        return abs(cx - 0.5) <= band / 2

    def _box(self, img, det, st, k, th):
        h, w = img.shape[:2]
        dist = getattr(st, "distance", "unknown")
        danger = dist in _CLOSE and self._in_path(det, w)
        color = (60, 60, 255) if danger else _GROUP_COLOR[_group(det.class_name)]
        b = det.bbox
        p1, p2 = (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2))
        cv2.rectangle(img, p1, p2, color, th * (2 if danger else 1))
        tid = f"#{det.track_id} " if det.track_id is not None else ""
        parts = [f"{det.class_name.replace('_', ' ')} {tid}{det.confidence:.2f}"]
        if st is not None:
            tags = [t for t in (dist if dist != "unknown" else "",
                                {"center": "ahead"}.get(st.h_zone, st.h_zone) if st.h_zone != "unknown" else "",
                                st.movement if st.movement in ("approaching", "receding", "moving") else "") if t]
            if tags:
                parts.append(" | ".join(tags).replace("_", " "))
        y = p1[1] - 4
        for line in reversed(parts):
            self._text(img, line, (p1[0], y), k, color=(255, 255, 255), bg=color, alpha=0.75, scale=0.5)
            y -= int(20 * k)
        if danger:
            cv2.putText(img, "IN PATH", (p1[0] + 4, p2[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55 * k, (60, 60, 255), th, cv2.LINE_AA)

    def _inset(self, img, k):
        h, w = img.shape[:2]
        iw = int(w * 0.24)
        ih = int(iw * h / w)
        cm = self._colormap(self.depth, (iw, ih))
        x, y = w - iw - 8, 8
        img[y:y + ih, x:x + iw] = cm
        cv2.rectangle(img, (x, y), (x + iw, y + ih), (255, 255, 255), 1)
        self._text(img, f"depth ({self.depth_label}) red=close", (x + 4, y + ih - 6), k,
                   scale=0.42, alpha=0.5)

    def _hud(self, img, k):
        i = self._info
        t = self._times
        fps = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 0.0
        parts = [f"{fps:4.1f} fps"]
        if i.get("yolo_ms") is not None:
            parts.append(f"YOLO {i['yolo_ms']:.0f} ms")
        parts += [f"objects {len(self.objs)}", f"depth: {self.depth_label}", f"voice: {self.voice_label}"]
        lag = self.lag_fn() if self.lag_fn else None
        if lag is not None:
            parts.append(f"voice lag {lag / 1000:.1f}s")
        self._text(img, " | ".join(parts), (8, int(34 * k)), k, scale=0.5)
        h = img.shape[0]
        self._text(img, "q quit   d depth   s summary   p pause", (8, h - int(10 * k)), k, scale=0.4, alpha=0.4,
                   color=(200, 200, 200))

    def _subtitles(self, img, now, k):
        h, w = img.shape[:2]
        y = h - int(46 * k)
        for t, level, text in reversed(list(self.subs)):
            age = now - t
            life = max(3.0, 0.09 * len(text) + 1.5)
            if age > life:
                continue
            fade = 1.0 if age < life - 1.0 else max(0.0, life - age)
            col = _LEVEL_COLOR.get(level, (255, 255, 255))
            self._text(img, f"{_LEVEL_SHORT.get(level, level[:4])}  {text}", (int(0.04 * w), y), k, color=col, bg=(0, 0, 0),
                       alpha=0.65 * fade + 0.05, scale=0.8 if level == "CRITICAL" else 0.7)
            y -= int(38 * k)
            if y < h * 0.6:
                break

    # -- window / recording --------------------------------------------------------
    def _show(self, img) -> None:
        if not self._win:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            self._win = True
        cv2.imshow(self.title, img)
        key = cv2.waitKey(1) & 0xFF
        while True:
            if key in (ord("q"), 27):
                raise KeyboardInterrupt           # Pipeline.run treats this as a clean stop
            if key == ord("d"):
                self.show_depth = not self.show_depth
            if key == ord("s") and self.on_summary:
                self.on_summary()
            if key == ord("p"):
                key = cv2.waitKey(0) & 0xFF       # paused until any key
                if key == ord("p"):
                    break
                continue
            break

    def _write(self, img) -> None:
        if self._writer is None:
            h, w = img.shape[:2]
            self._writer = cv2.VideoWriter(self.record_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                           self.record_fps, (w, h))
        self._writer.write(img)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._win:
            cv2.destroyWindow(self.title)
            cv2.waitKey(1)
            self._win = False


__all__ = ["LiveViewer"]
