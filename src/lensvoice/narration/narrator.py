"""Narrator — the ONLY place sentences are generated.

Starts from structured evidence only (Context + Events). Hard rules:
  * never invent distance, speed, intent, emotion, or identity
  * never say "I think / maybe" about something we did not measure
  * a hazard is only described with the fields it actually has
"""
from __future__ import annotations

from typing import Optional

from lensvoice.models.schemas import Context, Event, PriorityScore

_ZONE_TEXT = {"left": "left", "center": "ahead", "right": "right",
              "upper": "up ahead", "middle": "ahead", "lower": "lower"}
_DIST_TEXT = {"very_near": "very close", "near": "close", "medium": "a short way off",
              "far": "far away", "unknown": ""}


class Narrator:
    def __init__(self, terse: bool = False):
        self.terse = terse

    def generate(self, ctx: Context, scored: list[PriorityScore]) -> list[tuple[str, str]]:
        """Return (level, sentence) pairs — only for scored, non-stale events."""
        out: list[tuple[str, str]] = []
        for s in scored:
            text = self._sentence(s.event, ctx)
            if text:
                out.append((s.level, text))
        return out

    # -- sentence builders -------------------------------------------------
    def _sentence(self, ev: Event, ctx: Context) -> Optional[str]:
        if ev.type in ("hazard_path", "vehicle_in_path"):
            return self._hazard(ev, ctx)
        if ev.type == "sign_attention":
            kind = ev.detail.get("sign", "sign")
            return f"{kind.replace('_', ' ')} sign ahead" if not self.terse \
                else f"{kind} sign"
        if ev.type == "new_object":
            return self._new_object(ev, ctx)
        if ev.type == "object_moved":
            return self._moved(ev, ctx)
        if ev.type == "object_closer":
            name = self._name(ev, ctx)
            return f"{name} getting closer"
        if ev.type == "object_gone":
            name = self._name(ev, ctx)
            return f"{name} no longer visible"
        if ev.type == "text_detected":
            kind = ev.detail.get("kind", "text")
            text = ev.detail.get("text", "")
            return f"text says {text}" if not self.terse else text
        return None

    def _new_object(self, ev: Event, ctx: Context) -> Optional[str]:
        name = self._name(ev, ctx)
        zone = self._zone(ev, ctx)
        st = ctx.object_states.get(ev.track_id)
        dist = _DIST_TEXT.get(st.distance, "") if st else ""
        if self.terse:
            return f"{name}{', ' + zone if zone else ''}"
        parts = [p for p in (name, dist, zone) if p]
        return ", ".join(parts)

    def _moved(self, ev: Event, ctx: Context) -> Optional[str]:
        name = self._name(ev, ctx)
        zone = self._zone(ev, ctx)
        if self.terse:
            return f"{name} moved"
        return f"{name} moved to the {zone}" if zone else f"{name} moved"

    def _class_of(self, ev: Event, ctx: Context) -> Optional[str]:
        name = (ev.detail.get("class") or "").strip() or None
        if name:
            return name
        for det in ctx.objects:
            if det.track_id == ev.track_id:
                return det.class_name
        return None

    def _name(self, ev: Event, ctx: Context) -> str:
        return self._class_of(ev, ctx) or "an object"

    def _zone(self, ev: Event, ctx: Context) -> str:
        st = ctx.object_states.get(ev.track_id)
        return _ZONE_TEXT.get(st.h_zone, "") if st else ""

    def _hazard(self, ev: Event, ctx: Context) -> Optional[str]:
        name = self._name(ev, ctx)
        st = ctx.object_states.get(ev.track_id)
        dist = _DIST_TEXT.get(st.distance, "") if st else ""
        zone = self._zone(ev, ctx)
        if self.terse:
            return f"{name}, {zone}" if zone else name
        verb = "approaching" if ev.detail.get("movement") == "approaching" else "in your path"
        return f"{name} {verb}, {dist}, {zone}" if dist and zone else f"{name} {verb}"


__all__ = ["Narrator"]