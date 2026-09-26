"""Priority scoring.

Converts evidence Events into PriorityScores. Scoring is a pure function of
Event fields + Context evidence; the narrator only ever speaks from an
Event + its annotations. A repeated story gets a repetition penalty so the
user is not nagged with the same sentence on a loop.
"""
from __future__ import annotations

from typing import Optional

from lensvoice.config.settings import PriorityConfig
from lensvoice.models.schemas import Context, Event, PriorityScore


class PriorityScorer:
    LEVELS = ("CRITICAL", "HIGH", "MEDIUM", "LOW")

    def __init__(self, cfg: PriorityConfig, cooldown_ms: float = 5000.0):
        self.cfg = cfg
        self.cooldown_ms = cooldown_ms
        self._last_announced: dict[tuple, float] = {}

    def score(self, events: list[Event], ctx: Context | None = None) -> list[PriorityScore]:
        scored: list[PriorityScore] = []
        now = ctx.timestamp if ctx is not None else 0.0
        for ev in events:
            base = self._base_score(ev)
            if ctx is not None:
                base = self._context_bonus(ev, ctx, base)
            key = (ev.type, ev.track_id)
            if now - self._last_announced.get(key, -1e9) < self.cooldown_ms:
                base -= self.cfg.repetition_penalty
            base = max(0.0, base)
            level = self._level_for(base)
            scored.append(PriorityScore(event=ev, score=base, level=level))
        return sorted(scored, key=lambda s: s.score, reverse=True)

    def _base_score(self, ev: Event) -> float:
        w = self.cfg.weights
        if ev.type in ("hazard_path", "vehicle_in_path"):
            base = w["safety"]
            # approach is measurable from detail, so it amplifies even without ctx
            if ev.detail.get("movement") == "approaching":
                base += w["approaching"]
            return base
        if ev.type == "sign_attention":
            return w["navigation"]
        if ev.type == "object_closer":
            return w["proximity"]
        if ev.type == "object_moved":
            return w["navigation"]
        if ev.type == "new_object":
            return w["new_object"] + w["information"]
        if ev.type == "text_detected":
            return w["information"]
        if ev.type == "object_gone":
            return w["information"]
        return w["information"]

    def _context_bonus(self, ev: Event, ctx: Context, base: float) -> float:
        return base

    def _level_for(self, score: float) -> str:
        if score >= self.cfg.levels["critical"]:
            return "CRITICAL"
        if score >= self.cfg.levels["high"]:
            return "HIGH"
        if score >= self.cfg.levels["medium"]:
            return "MEDIUM"
        return "LOW"

    def remember(self, score: PriorityScore, now: float) -> None:
        self._last_announced[(score.event.type, score.event.track_id)] = now


__all__ = ["PriorityScorer"]