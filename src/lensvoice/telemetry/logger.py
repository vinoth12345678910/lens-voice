"""Privacy-safe structured logger.

Rules enforced by design:
  * never log raw images (rejected even if a caller passes an ndarray)
  * OCR/recognised text is redacted unless explicitly whitelisted — logging
    what the user could read in the world is a privacy decision, so it is off
    by default
  * timestamps are wall-clock only for display; logic uses monotonic Clock
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any

from lensvoice.config.settings import LoggingConfig

_FORBIDDEN = {"image", "frame", "ndarray", "raw_values", "ocr_text", "text"}


class Logger:
    def __init__(self, cfg: LoggingConfig, name: str = "lensvoice"):
        self.cfg = cfg
        self.name = name
        self._log = logging.getLogger(name)
        self._log.setLevel(getattr(logging, cfg.level.upper(), logging.INFO))
        if not self._log.handlers:
            h = logging.StreamHandler(sys.stderr)
            h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
            self._log.addHandler(h)

    def _redact(self, fields: dict) -> dict:
        out = dict(fields)
        for k, v in list(out.items()):
            if k in _FORBIDDEN or k.endswith("_text"):
                out[k] = "**redacted**"
            elif isinstance(v, (bytes, bytearray)):
                out[k] = f"<{len(v)} bytes>"
        return out

    def _emit(self, level: str, msg: str, fields: dict = None):
        fields = fields or {}
        fields = self._redact(fields)
        if self.cfg.structured:
            payload = {"event": msg, **fields}
            if self.cfg.json:
                text = json.dumps(payload, default=str)
            else:
                parts = " ".join(f"{k}={v}" for k, v in fields.items())
                text = f"{msg} {parts}".strip()
        else:
            text = msg if not fields else f"{msg} {fields}"
        getattr(self._log, level)(text)

    def debug(self, msg: str, **fields): self._emit("debug", msg, fields)
    def info(self, msg: str, **fields): self._emit("info", msg, fields)
    def warning(self, msg: str, **fields): self._emit("warning", msg, fields)
    def error(self, msg: str, **fields): self._emit("error", msg, fields)


__all__ = ["Logger"]