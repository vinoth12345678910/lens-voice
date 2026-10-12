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


_NAME = {"tv_monitor": "TV", "cell_phone": "phone", "autorickshaw": "auto", "motorcycle": "motorbike",
         "person": "someone"}
_PLURAL = {"person": "people", "couch": "couches", "bench": "benches", "bus": "buses",
           "tv_monitor": "TVs", "cell_phone": "phones", "autorickshaw": "autos"}
_MOVERS = {"person", "car", "motorcycle", "bus", "truck", "autorickshaw", "bicycle", "dog"}
# relative words only (never metres): the band comes from the relative depth map
_BAND_PHRASE = {"very_near": "very close", "near": "close by", "medium": "a little further off",
                "far": "far off"}
_ZONE_PHRASE = {"left": "on your left", "right": "on your right", "center": "straight ahead"}


def _plural(name: str) -> str:
    return _PLURAL.get(name, _NAME.get(name, name.replace("_", " ")) + "s")


def _a(name: str) -> str:
    """'a chair', 'an auto', 'a TV', 'someone'."""
    n = _NAME.get(name, name.replace("_", " "))
    if name == "person":
        return n
    return f"{'an' if n[0].lower() in 'aeiou' else 'a'} {n}"


def _cap(t: str) -> str:
    return t[:1].upper() + t[1:]


def _quantity(count: int) -> str:
    """Spoken the way people say it: never an exact count we might have wrong."""
    return "a couple of" if count == 2 else "a few" if count <= 5 else "several"


class Narrator:
    def __init__(self, terse: bool = False):
        self.terse = terse

    # -- pull-mode wording: built from the CURRENT scene state at the moment of speaking --------
    # Spoken like a person guiding you, but only measured facts are stated: what it is, which
    # side, how close RELATIVELY, whether it is in your way, whether it is coming closer.
    def scene_sentence(self, kind: str, name: str, count: int = 1, zone: str = "",
                       band: str = "", in_path: bool = False, approaching: bool = False,
                       variant: int = 0) -> str:
        zone_t, band_t = _ZONE_PHRASE.get(zone, ""), _BAND_PHRASE.get(band, "")
        if count > 1:
            subject = f"{_quantity(count)} {_plural(name)}"
            return _cap(", ".join(p for p in (f"{subject} {zone_t}".strip(), band_t) if p)) + "."
        who = _a(name)
        if kind == "urgent":
            mover = name in _MOVERS
            if approaching and mover:
                where = {"left": ", on your left", "right": ", on your right"}.get(zone, "")
                if zone == "center":
                    return (f"Watch out, {who} is coming straight toward you." if variant % 2 == 0
                            else f"{_cap(who)} is getting closer, straight ahead.")
                return (f"Watch out, {who} is coming toward you{where}." if variant % 2 == 0
                        else f"{_cap(who)} is getting closer{where}.")
            if name == "person":
                return "Careful, someone is right in front of you." if variant % 2 == 0 \
                    else "Heads up, someone is right ahead of you."
            return (f"Careful, there's {who} right in front of you." if variant % 2 == 0
                    else f"Heads up, {who} right ahead of you.")
        subject = "someone" if name == "person" else who
        if variant % 2 == 0:
            text = ", ".join(p for p in (f"There's {subject} {zone_t}".strip(), band_t) if p)
        else:
            text = ", ".join(p for p in (f"{_cap(subject)} {zone_t}".strip(), band_t) if p)
        return text + "."

    def summary_sentence(self, parts: list[tuple[str, int, str]], small: int = 0) -> str:
        """parts: (class, count, zone) by importance; small: number of small handheld items."""
        bits = []
        for name, count, zone in parts:
            who = f"{_quantity(count)} {_plural(name)}" if count > 1 else _a(name)
            if name == "person" and count == 1:
                who = "someone"
            bits.append(f"{who} {_ZONE_PHRASE.get(zone, '')}".strip())
        if small >= 2:
            bits.append("a few smaller things")
        if not bits:
            return ""
        body = bits[0] if len(bits) == 1 else ", ".join(bits[:-1]) + " and " + bits[-1]
        return f"I can see {body}."

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
            # a nameless "an object no longer visible" carries no evidence worth speaking
            name = self._class_of(ev, ctx)
            return f"{name} no longer visible" if name else None
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