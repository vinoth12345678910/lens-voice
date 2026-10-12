"""The sentences the narrator says most often, for pre-fetching voice audio.

Enumerated from the narrator's own wording code (never retyped), so they cannot drift apart.
  core_phrases   : every URGENT sentence (the ones that must be instant), a few dozen
  common_phrases : core + the routine "there's a table on your left, close by." family
"""
from __future__ import annotations

from lensvoice.config.settings import NarrationConfig
from lensvoice.narration.narrator import _BAND_PHRASE, _MOVERS, Narrator

_ZONES = ("left", "center", "right")


def core_phrases(class_names=None, cfg: NarrationConfig | None = None) -> list[str]:
    cfg, n, out = cfg or NarrationConfig(), Narrator(), []
    for cls in cfg.hazard_classes:
        for v in (0, 1):
            out.append(n.scene_sentence("urgent", cls, 1, "center", "near", True, False, v))      # in your way
            if cls in _MOVERS:
                for z in _ZONES:
                    out.append(n.scene_sentence("urgent", cls, 1, z, "near", False, True, v))    # coming closer
    return list(dict.fromkeys(out))


def common_phrases(class_names=None, cfg: NarrationConfig | None = None) -> list[str]:
    cfg, n = cfg or NarrationConfig(), Narrator()
    out = core_phrases(class_names, cfg)
    for cls in cfg.hazard_classes:
        for z in _ZONES:
            for b in _BAND_PHRASE:
                for v in (0, 1):
                    out.append(n.scene_sentence("normal", cls, 1, z, b, False, False, v))
    return list(dict.fromkeys(out))


__all__ = ["common_phrases", "core_phrases"]
