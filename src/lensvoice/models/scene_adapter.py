"""SceneAdapter — Socket for scene/context classification (e.g. CLIP).

`weight` may be a CLIP model name (e.g. "openai/clip-vit-base-patch32") or a
fine-tuned model dir. Returns the top class label + confidence.
"""
from __future__ import annotations

from lensvoice.config.settings import ModelConfig
from lensvoice.models.interfaces import SceneModel
from lensvoice.models.schemas import Frame, SceneResult


class SceneAdapter(SceneModel):
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self._model = None
        self._processor = None

    def _ensure(self):
        if self._model is not None:
            return self._model
        if not self.cfg.weight:
            raise RuntimeError("scene enabled but 'models.scene.weight' unset")
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("transformers not installed") from e
        self._model = CLIPModel.from_pretrained(self.cfg.weight)
        self._processor = CLIPProcessor.from_pretrained(self.cfg.weight)
        return self._model

    def classify(self, frame: Frame) -> SceneResult:
        model = self._ensure()
        texts = self.cfg.extra.get(
            "candidate_labels",
            ["outdoor city street", "indoor room", "park", "highway", "sidewalk"],
        )
        inputs = self._processor(text=texts, images=frame.image,
                                 return_tensors="pt")
        out = model(**inputs)
        probs = out.logits_per_image.softmax(dim=1)[0]
        idx = int(probs.argmax())
        return SceneResult(scene_label=texts[idx], confidence=float(probs[idx]),
                           timestamp=frame.timestamp)


__all__ = ["SceneAdapter"]