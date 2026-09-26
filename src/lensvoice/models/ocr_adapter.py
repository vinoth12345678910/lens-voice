"""PaddleOCRAdapter — bridges PaddleOCR into the OCRModel interface.

OCR output feeds the Context's `text` field as evidence. Privacy contract:
OCR text must not be persisted by default (see pipeline), and no raw frame is
ever logged.
"""
from __future__ import annotations

from lensvoice.config.settings import ModelConfig
from lensvoice.models.interfaces import OCRModel
from lensvoice.models.schemas import BBox, Frame, OCRResult


class PaddleOCRAdapter(OCRModel):
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self._ocr = None

    def _ensure_ocr(self):
        if self._ocr is not None:
            return self._ocr
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "OCR model enabled but paddleocr not installed") from e
        use_gpu = self.cfg.device not in ("cpu", "")
        self._ocr = PaddleOCR(
            lang=self.cfg.language or "en",
            use_gpu=use_gpu,
            ocr_version="PP-OCRv4"
            if self.cfg.extra.get("ocr_version") == "PP-OCRv4" else None,
        )
        return self._ocr

    def read(self, frame: Frame) -> list[OCRResult]:
        ocr = self._ensure_ocr()
        result = ocr.ocr(frame.image)
        out = []
        for block in result or []:
            if not block:
                continue
            for item in block:
                box, (text, conf) = item[0], item[1]
                if not text:
                    continue
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                try:
                    bbox = BBox(min(xs), min(ys), max(xs), max(ys))
                except ValueError:
                    bbox = None
                out.append(OCRResult(text=str(text), confidence=conf,
                                     timestamp=frame.timestamp, bbox=bbox))
        return out


__all__ = ["PaddleOCRAdapter"]