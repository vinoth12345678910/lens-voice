"""Scene picture + pull-based narration scheduler.

Instead of pushing a sentence into a queue for every detection event, keep a small,
stable picture of the scene and, ONLY WHEN THE VOICE IS FREE, ask "what is the single
most important thing to say right now?" - the sentence is built from the state at that
moment, so nothing can be stale and nothing is queued.

SceneTracker
  * objects are stable: one only "exists" after being seen in `confirm_hits` of the last
    `confirm_window` detection ticks and is dropped after `drop_after_s` unseen; a new
    tracker id on top of a recent same-class object (IoU >= reid_iou) is the SAME object, so
    tracker id churn cannot cause re-announcements;
  * distance band, zone and "in path" use hysteresis (a change must hold `band_hold_ticks`);
  * every candidate is scored  class_weight * proximity * (2 if in path) * (1.5 if
    approaching) * novelty  and the top one is spoken;
  * small handheld classes (and other non-hazards) are never announced unless near;
  * three levels: urgent (cuts non-urgent speech, per-object cooldown), normal (>= min gap
    between sentences), ambient (scene summary on start, on request, or after long silence);
  * same-class neighbours are grouped ("three chairs, ahead"); silence is a valid output.

PullVoice runs the voice loop: pick -> speak (blocking) -> pick ... in its own thread.
Sentence wording lives in Narrator (the only place sentences are made).
"""
from __future__ import annotations

import collections
import itertools
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from lensvoice.config.settings import NarrationConfig
from lensvoice.models.schemas import BBox
from lensvoice.narration.narrator import Narrator

_RANK = {"far": 0, "medium": 1, "near": 2, "very_near": 3}
_PROX = {"very_near": 1.0, "near": 0.7, "medium": 0.35, "far": 0.15}
_NEAR = ("near", "very_near")
_LEVEL = {"urgent": "CRITICAL", "normal": "MEDIUM", "ambient": "LOW"}
_MOVERS = {"person", "car", "motorcycle", "bus", "truck", "autorickshaw", "bicycle", "dog"}


class _Stable:
    """A value that only changes after the new value was seen `hold` ticks in a row."""

    def __init__(self, hold: int):
        self.hold, self.value, self._cand, self._n = max(1, hold), None, None, 0

    def push(self, v) -> None:
        if self.value is None:
            self.value = v
        elif v == self.value:
            self._cand, self._n = None, 0
        else:
            self._n = self._n + 1 if v == self._cand else 1
            self._cand = v
            if self._n >= self.hold:
                self.value, self._cand, self._n = v, None, 0


@dataclass
class _Said:
    rank: int
    in_path: bool
    approaching: bool
    t: float                       # wall time


@dataclass
class SceneObject:
    sid: int
    cls: str
    cfg_hold: int
    bbox: Optional[BBox] = None
    last_ts: float = 0.0           # frame timestamp (ms)
    track_ids: set = field(default_factory=set)
    hits: collections.deque = field(default_factory=collections.deque)
    confirmed: bool = False
    band: _Stable = None
    zone: _Stable = None
    in_path: _Stable = None
    area_hist: collections.deque = field(default_factory=collections.deque)   # (ts, area)
    band_moves: collections.deque = field(default_factory=lambda: collections.deque(maxlen=8))  # (ts, from, to)
    said: Optional[_Said] = None
    last_urgent: float = -1e9

    def __post_init__(self):
        self.band, self.zone, self.in_path = _Stable(self.cfg_hold), _Stable(self.cfg_hold), _Stable(self.cfg_hold)


@dataclass
class Pick:
    kind: str                      # urgent | normal | ambient
    text: str
    level: str                     # CRITICAL | MEDIUM | LOW  (viewer/audio vocabulary)
    ready_wall: float              # when this became pickable (for latency telemetry)
    objs: list = field(default_factory=list)
    score: float = 0.0


class SceneTracker:
    def __init__(self, cfg: NarrationConfig, narrator: Optional[Narrator] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self.narr = narrator or Narrator()
        self.clock = clock
        self._lock = threading.RLock()
        self._ids = itertools.count(1)
        self.objs: dict[int, SceneObject] = {}
        self.W = self.H = 1
        self.t_start: Optional[float] = None
        self.last_ts = 0.0
        self._summary_req = False
        self._start_summary_done = False
        self._sum_text = ""                # the summary we will say: frozen when first needed, so its audio
        self._sum_due_at: Optional[float] = None   # can be fetched ahead of time (wall time it was frozen)
        self.ready_fn: Optional[Callable[[str], bool]] = None    # tts.is_ready: is the audio for this text fetched?
        self.last_spoken_end = -1e9
        self._last_urgent_wall = -1e9
        self._speaking = False
        self._hazard = set(cfg.hazard_classes)
        self._handheld = set(cfg.handheld_classes)
        # telemetry
        self.candidates_seen: set = set()
        self._first_ready: dict = {}
        self.spoken = 0
        self.summaries = 0
        self.interrupts = 0
        self.urgent_latency_ms: list[float] = []        # urgent: confirmed -> audio actually started
        self.pick_to_audio_ms: list[float] = []         # any sentence: picked -> audio started (voice responsiveness)
        self.ego_suppressed = 0

    # ------------------------------------------------------------------ scene picture
    def update(self, ctx, width: int, height: int, ts: float, tick: bool) -> None:
        """Feed one frame. `tick` = detection ran on this frame (ctx.objects is fresh)."""
        c = self.cfg
        with self._lock:
            if self.t_start is None:
                self.t_start = self.clock()
            self.W, self.H, self.last_ts = max(1, width), max(1, height), ts
            if tick:
                matched: set[int] = set()
                for det in ctx.objects:
                    obj = self._associate(det, ts, matched)
                    matched.add(obj.sid)
                    st = ctx.object_states.get(det.track_id)
                    obj.bbox, obj.last_ts = det.bbox, ts
                    obj.hits.append(True)
                    obj.area_hist.append((ts, det.bbox.area))
                    while obj.area_hist and ts - obj.area_hist[0][0] > 2500:
                        obj.area_hist.popleft()
                    if st is not None:
                        if st.distance != "unknown":
                            before = obj.band.value
                            obj.band.push(st.distance)
                            if before is not None and obj.band.value != before:
                                obj.band_moves.append((ts, _RANK[before], _RANK[obj.band.value]))
                        if st.h_zone != "unknown":
                            obj.zone.push(st.h_zone)
                    obj.in_path.push(self._raw_in_path(det.bbox))
                for sid, obj in self.objs.items():
                    if sid not in matched:
                        obj.hits.append(False)
                    while len(obj.hits) > c.confirm_window:
                        obj.hits.popleft()
                    if not obj.confirmed and sum(obj.hits) >= c.confirm_hits:
                        obj.confirmed = True
            for sid in [s for s, o in self.objs.items() if ts - o.last_ts > c.drop_after_s * 1000.0]:
                del self.objs[sid]

    def _associate(self, det, ts: float, matched: set) -> SceneObject:
        for o in self.objs.values():
            if o.sid not in matched and o.cls == det.class_name and det.track_id is not None \
                    and det.track_id in o.track_ids:
                return o
        best, best_iou = None, self.cfg.reid_iou
        for o in self.objs.values():
            if o.sid in matched or o.cls != det.class_name or o.bbox is None or ts - o.last_ts > 1500:
                continue
            iou = det.bbox.iou(o.bbox)
            if iou >= best_iou:
                best, best_iou = o, iou
        if best is None:
            best = SceneObject(sid=next(self._ids), cls=det.class_name, cfg_hold=self.cfg.band_hold_ticks)
            self.objs[best.sid] = best
        if det.track_id is not None:
            best.track_ids.add(det.track_id)
        return best

    def _raw_in_path(self, b: BBox) -> bool:
        half = self.cfg.path_center_frac / 2.0
        return abs(b.center_x / self.W - 0.5) <= half and b.y2 / self.H >= self.cfg.path_bottom_min

    def _approaching(self, o: SceneObject) -> bool:
        if any(to > frm and o.last_ts - t <= 1000 for t, frm, to in o.band_moves):
            return True
        target = o.last_ts - self.cfg.approach_s * 1000.0
        then = None
        for ts, area in o.area_hist:
            if abs(ts - target) <= 300 and (then is None or abs(ts - target) < abs(then[0] - target)):
                then = (ts, area)
        return bool(then and then[1] > 0 and o.area_hist[-1][1] / then[1] >= self.cfg.approach_area_ratio)

    # ------------------------------------------------------------------ scoring
    def _weight(self, cls: str) -> float:
        if cls in _MOVERS and cls != "dog":
            return 3.0
        if cls == "dog":
            return 2.5
        if cls in self._hazard:
            return 2.0
        if cls in self._handheld:
            return 0.5
        return 1.0

    def _candidates(self, now: float) -> list[dict]:
        out = []
        c = self.cfg
        live = [o for o in self.objs.values() if o.confirmed and o.band.value is not None]
        raw_appr = {o.sid: self._approaching(o) for o in live}
        # "approaching" only means something for things that MOVE. A static chair 'approaching' is
        # the camera moving; and if most objects approach at once it is ego-motion for all of them.
        n_appr = sum(raw_appr.values())
        ego = c.ego_motion_filter and n_appr >= 3 and n_appr >= 0.6 * len(live)
        if ego:
            self.ego_suppressed += 1
        urgent_ok = now - self._last_urgent_wall >= c.urgent_min_gap_s
        for o in live:
            band = o.band.value
            hazard = o.cls in self._hazard
            if not hazard and band not in _NEAR:
                continue                       # clutter / small items: only when near, never auto at range
            in_path = bool(o.in_path.value)
            appr = raw_appr[o.sid] and o.cls in _MOVERS and not ego
            rank, said = _RANK[band], o.said
            if said is None:
                novelty = 1.0
            elif rank > said.rank or (in_path and not said.in_path) or (appr and not said.approaching):
                novelty = 1.0
            elif now - said.t >= c.repeat_after_s:
                novelty = 0.5
            else:
                novelty = 0.0
            if novelty <= 0.0:
                continue
            score = self._weight(o.cls) * _PROX[band] * (2.0 if in_path else 1.0) * (1.5 if appr else 1.0) * novelty
            urgent = (hazard and band in _NEAR and (in_path or appr) and urgent_ok
                      and now - o.last_urgent >= c.urgent_cooldown_s)
            key = (o.sid, rank, in_path, appr)
            self._first_ready.setdefault(key, now)
            self.candidates_seen.add(key)
            out.append({"obj": o, "score": score, "urgent": urgent, "band": band, "in_path": in_path,
                        "appr": appr, "zone": o.zone.value or "center", "key": key})
        return out

    # ------------------------------------------------------------------ pull API
    def request_summary(self) -> None:
        with self._lock:
            self._summary_req = True

    def has_urgent(self) -> bool:
        with self._lock:
            return any(c["urgent"] for c in self._candidates(self.clock()))

    def peek(self) -> Optional[Pick]:
        """What pick() would say now (no side effects) - used to pre-fetch voice audio."""
        with self._lock:
            return self._choose(self.clock())

    def pick(self) -> Optional[Pick]:
        """Called by the voice when it is free. Commits the choice (marks objects as told)."""
        with self._lock:
            if self._speaking:
                return None
            now = self.clock()
            p = self._choose(now)
            if p is None:
                return None
            for o in p.objs:
                o.said = _Said(rank=_RANK.get(o.band.value, 0), in_path=bool(o.in_path.value),
                               approaching=self._approaching(o), t=now)
                if p.kind == "urgent":
                    o.last_urgent = now
                    self._last_urgent_wall = now
            if p.kind == "ambient":
                self._summary_req = False
                self._start_summary_done = True
                self._sum_text, self._sum_due_at = "", None
                self.summaries += 1
            self._speaking = True
            self.spoken += 1
            return p

    def speech_done(self, p: Pick, started_wall: Optional[float] = None,
                    audio_wall: Optional[float] = None) -> None:
        """started_wall: when the voice began handling this pick; audio_wall: when audio actually started."""
        with self._lock:
            self._speaking = False
            self.last_spoken_end = self.clock()
            if audio_wall is not None:
                if started_wall is not None:
                    self.pick_to_audio_ms.append((audio_wall - started_wall) * 1000.0)
                if p.kind == "urgent":
                    self.urgent_latency_ms.append((audio_wall - p.ready_wall) * 1000.0)

    def _choose(self, now: float) -> Optional[Pick]:
        c = self.cfg
        confirmed = [o for o in self.objs.values() if o.confirmed]
        cands = self._candidates(now)
        urgent = [x for x in cands if x["urgent"]]
        if urgent:
            top = max(urgent, key=lambda x: x["score"])
            text = self.narr.scene_sentence("urgent", top["obj"].cls, 1, top["zone"], top["band"],
                                            top["in_path"], top["appr"], top["obj"].sid)
            return Pick("urgent", text, _LEVEL["urgent"], self._first_ready[top["key"]], [top["obj"]], top["score"])
        started = self.t_start if self.t_start is not None else now
        if self._summary_req and confirmed:                      # asked for with the 's' key
            self._freeze_summary(now, confirmed)
            return self._summary(now, confirmed, use_fixed=True) if self._summary_ok(now) else None
        if c.summary_on_start and not self._start_summary_done and len(confirmed) >= 2:
            # several objects in view at the start: describe the room ONCE first (routine sentences wait,
            # urgent ones never do). The text is frozen at ~1.5 s and pre-fetched; it is spoken at >= 2.5 s,
            # as soon as its audio is ready (or after summary_max_wait_s).
            if now - started >= 1.5:
                self._freeze_summary(now, confirmed)
            if now - started >= 2.5 and self._sum_due_at is not None and self._summary_ok(now):
                return self._summary(now, confirmed, use_fixed=True)
            return None
        if cands and now - self.last_spoken_end >= c.min_gap_normal_s:
            top = max(cands, key=lambda x: x["score"])
            same = [x for x in cands if x["obj"].cls == top["obj"].cls and x["zone"] == top["zone"]
                    and x["obj"].cls not in self._handheld or x is top]
            count = len(same)
            near = max(same, key=lambda x: _RANK[x["band"]])
            text = self.narr.scene_sentence("normal", top["obj"].cls, count, top["zone"], near["band"],
                                            variant=top["obj"].sid)
            return Pick("normal", text, _LEVEL["normal"], self._first_ready[top["key"]],
                        [x["obj"] for x in same], top["score"])
        if c.ambient_silence_s > 0 and confirmed and \
                now - max(self.last_spoken_end, started) >= c.ambient_silence_s:
            self._freeze_summary(now, confirmed)
            return self._summary(now, confirmed, use_fixed=True) if self._summary_ok(now) else None
        return None

    def _freeze_summary(self, now: float, confirmed: list[SceneObject]) -> None:
        """Fix the summary text the first time it is needed, so the voice can fetch its audio meanwhile."""
        if self._sum_due_at is None:
            self._sum_text = self._summary(now, confirmed).text
            self._sum_due_at = now

    def _summary_ok(self, now: float) -> bool:
        """The frozen summary can be spoken: its audio is ready, or it has waited long enough."""
        if self._sum_due_at is None:
            return False
        if self.ready_fn is None or not self._sum_text or self.ready_fn(self._sum_text):
            return True
        return now - self._sum_due_at >= self.cfg.summary_max_wait_s

    def prefetch_text(self) -> Optional[str]:
        """Text worth pre-fetching audio for: what would be said now, else a summary that is about to be due."""
        with self._lock:
            now = self.clock()
            p = self._choose(now)
            if p is not None and p.text:
                return p.text
            confirmed = [o for o in self.objs.values() if o.confirmed]
            if confirmed and self._summary_req:
                self._freeze_summary(now, confirmed)
            elif (confirmed and self.cfg.summary_on_start and not self._start_summary_done
                  and len(confirmed) >= 2 and now - (self.t_start or now) >= 1.5):
                self._freeze_summary(now, confirmed)
            return self._sum_text or None

    def _summary(self, now: float, confirmed: list[SceneObject], use_fixed: bool = False) -> Pick:
        groups: dict[tuple, list] = collections.defaultdict(list)
        small = 0
        for o in confirmed:
            if o.cls in self._handheld:
                small += 1
            else:
                groups[(o.cls, o.zone.value or "center")].append(o)
        ranked = sorted(groups.items(), key=lambda kv: -max(
            self._weight(kv[0][0]) * _PROX.get(x.band.value, 0.2) for x in kv[1]))
        parts = [(cls, len(objs), zone) for (cls, zone), objs in ranked[:5]]
        if not parts and not small:
            return Pick("ambient", "", _LEVEL["ambient"], now, [])
        text = self.narr.summary_sentence(parts, small if small >= 2 else 0)
        if use_fixed and self._sum_text:
            text = self._sum_text                  # the exact text whose audio was pre-fetched
        return Pick("ambient", text, _LEVEL["ambient"], now, list(confirmed))

    # ------------------------------------------------------------------ telemetry
    def stats(self) -> dict:
        with self._lock:
            lat, pa = sorted(self.urgent_latency_ms), sorted(self.pick_to_audio_ms)
            return {"candidates": len(self.candidates_seen), "spoken": self.spoken,
                    "summaries": self.summaries, "interrupts": self.interrupts,
                    "ego_motion_frames": self.ego_suppressed,
                    "urgent_n": len(lat),
                    "urgent_latency_ms_median": lat[len(lat) // 2] if lat else None,
                    "urgent_latency_ms_max": lat[-1] if lat else None,
                    "pick_to_audio_ms_median": pa[len(pa) // 2] if pa else None,
                    "pick_to_audio_ms_p90": pa[min(len(pa) - 1, int(0.9 * len(pa)))] if pa else None,
                    "pick_to_audio_ms_max": pa[-1] if pa else None}


class PullVoice:
    """The voice loop: ask the scene what to say next, say it, repeat. Nothing is queued."""

    def __init__(self, scene: SceneTracker, tts, on_utterance: Optional[Callable[[str, str], None]] = None):
        self.scene, self.tts, self.on_utterance = scene, tts, on_utterance
        scene.ready_fn = getattr(tts, "is_ready", None)
        self._stop = threading.Event()
        self._cut = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.current_kind: Optional[str] = None

    def start(self) -> None:
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="lensvoice-voice", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set(); self._cut.set()
        self.tts.stop()
        self._thread.join(timeout=3.0)
        self._thread = None

    def after_update(self) -> None:
        """Called by the pipeline after each scene update: interrupt for urgent news, pre-fetch audio."""
        sc = self.scene
        if sc.cfg.interrupt_urgent and self.current_kind in ("normal", "ambient") and sc.has_urgent():
            sc.interrupts += 1
            self._cut.set()
            self.tts.stop()
        text = sc.prefetch_text()
        if text:
            self.tts.prefetch(text)

    def _run(self) -> None:
        sc = self.scene
        while not self._stop.is_set():
            p = sc.pick()
            if p is None:
                time.sleep(0.03)
                continue
            if not p.text:
                sc.speech_done(p)
                continue
            self.current_kind = p.kind
            self._cut.clear()
            started = sc.clock()
            self.tts.enq_time = p.ready_wall if p.kind == "urgent" else started
            audio_wall = None
            try:
                if self.on_utterance:
                    self.on_utterance(p.level, p.text)
                self.tts.speak(p.text, level=p.level)
                played = getattr(self.tts, "last_play_wall", None)
                # real providers stamp when audio started; a silent stand-in "starts" immediately
                audio_wall = played if (played is not None and played >= started) else (
                    started if played is None else None)
                hint = self.tts.duration_hint(p.text)         # silent stand-in: pretend speech takes time
                if hint > 0:
                    self._cut.wait(hint)
            except Exception:
                pass
            finally:
                self.current_kind = None
                sc.speech_done(p, started, audio_wall)


__all__ = ["SceneTracker", "SceneObject", "Pick", "PullVoice"]
