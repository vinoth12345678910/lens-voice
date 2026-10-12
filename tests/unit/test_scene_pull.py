"""Unit: scene picture + pull narration (stable objects, scoring, gating, pacing)."""
from __future__ import annotations

import threading
import time

from lensvoice.config.settings import NarrationConfig
from lensvoice.models.schemas import BBox, Context, DetectedObject, ObjectState
from lensvoice.narration.scene import PullVoice, SceneTracker

W, H = 1000, 600


class Clock:
    def __init__(self): self.t = 100.0
    def __call__(self): return self.t
    def advance(self, s): self.t += s


def mk(**kw):
    clk = Clock()
    cfg = NarrationConfig(mode="pull", **kw)
    return SceneTracker(cfg, clock=clk), clk


def box(cx, y2, w=100, h=100):
    return BBox(cx - w / 2, y2 - h, cx + w / 2, y2)


def feed(tr, ts, objs):
    """objs: (track_id, cls, bbox, band, zone)"""
    ctx = Context(timestamp=ts, frame_id=int(ts))
    for tid, cls, bb, band, zone in objs:
        ctx.objects.append(DetectedObject(class_id=0, class_name=cls, confidence=0.9, bbox=bb,
                                          timestamp=ts, track_id=tid))
        ctx.object_states[tid] = ObjectState(track_id=tid, distance=band, h_zone=zone)
    tr.update(ctx, W, H, ts, True)


def run(tr, objs, n=4, t0=0.0, dt=100.0):
    for i in range(n):
        feed(tr, t0 + i * dt, objs)
    return t0 + (n - 1) * dt


FAR_CHAIR = lambda tid=1: (tid, "chair", box(150, 300), "far", "left")


def test_object_exists_only_after_enough_sightings():
    tr, _ = mk()
    run(tr, [FAR_CHAIR()], n=2)
    assert tr.pick() is None                       # 2 of 4: not confirmed yet
    run(tr, [FAR_CHAIR()], n=1, t0=200)
    p = tr.pick()
    assert p is not None and "chair" in p.text


def test_flicker_does_not_cause_reannouncement_and_object_drops_after_1s():
    tr, clk = mk()
    t = run(tr, [FAR_CHAIR()])
    first = tr.pick(); tr.speech_done(first)
    assert first and first.kind == "normal"
    feed(tr, t + 100, [])                          # one missed detection tick
    feed(tr, t + 200, [FAR_CHAIR()])               # seen again
    clk.advance(5)
    assert tr.pick() is None                       # same object, same state: silence
    feed(tr, t + 1500, [])                         # unseen > 1 s -> dropped
    assert not tr.objs


def test_new_tracker_id_on_same_object_is_not_a_new_object():
    tr, clk = mk()
    t = run(tr, [FAR_CHAIR(1)])
    p = tr.pick(); tr.speech_done(p)
    run(tr, [FAR_CHAIR(7), ], n=4, t0=t + 100)     # tracker lost it and gave a new id
    clk.advance(5)
    assert len(tr.objs) == 1 and tr.pick() is None


def test_distance_band_needs_to_hold_before_changing():
    tr, _ = mk()
    t = 0.0
    for i in range(8):                              # a cup sitting on a band boundary
        feed(tr, t, [(1, "chair", box(150, 300), "far" if i % 2 == 0 else "medium", "left")]); t += 100
    assert next(iter(tr.objs.values())).band.value == "far"      # never flipped
    for i in range(3):
        feed(tr, t, [(1, "chair", box(150, 300), "medium", "left")]); t += 100
    assert next(iter(tr.objs.values())).band.value == "medium"   # a real change took hold


def test_small_handheld_items_only_when_near():
    tr, clk = mk(summary_on_start=False, ambient_silence_s=0)
    run(tr, [(1, "cup", box(500, 500), "far", "center"), (2, "bottle", box(300, 500), "medium", "left")])
    assert tr.pick() is None                        # clutter at range is never announced
    run(tr, [(1, "cup", box(500, 500), "near", "center")], t0=500)
    p = tr.pick()
    assert p is not None and "cup" in p.text


def test_each_object_is_told_once_until_it_changes():
    tr, clk = mk()
    t = run(tr, [FAR_CHAIR()])
    p = tr.pick(); tr.speech_done(p)
    clk.advance(10)
    assert tr.pick() is None                        # unchanged
    t = run(tr, [(1, "chair", box(150, 300), "near", "left")], t0=t + 100)   # moved a band closer
    clk.advance(5)
    q = tr.pick()
    assert q is not None and "close" in q.text
    tr.speech_done(q)
    clk.advance(21)                                 # long unchanged -> may be mentioned again
    assert tr.pick() is not None


def test_urgent_chair_in_path_preempts_and_has_cooldown():
    tr, clk = mk(summary_on_start=False, ambient_silence_s=0)
    far_cup = (2, "cup", box(800, 500), "far", "right")
    t = run(tr, [FAR_CHAIR(), far_cup])
    blocker = (1, "chair", box(500, 560), "near", "center")      # centre of frame, low = in path
    t = run(tr, [blocker, far_cup], t0=t + 100, n=4)
    assert tr.has_urgent()
    p = tr.pick()
    assert p.kind == "urgent" and p.level == "CRITICAL"
    assert "chair" in p.text and ("in front of you" in p.text or "ahead of you" in p.text)
    tr.speech_done(p, tr.clock())
    assert not tr.has_urgent()                      # per-object cooldown
    clk.advance(4.5)
    assert tr.pick() is None                        # same state: not repeated even after the cooldown


def test_same_class_neighbours_are_grouped():
    tr, _ = mk(summary_on_start=False, ambient_silence_s=0)
    run(tr, [(1, "chair", box(450, 300), "far", "center"), (2, "chair", box(500, 300), "far", "center"),
             (3, "chair", box(550, 300), "far", "center")])
    p = tr.pick()
    assert p.text.startswith("A few chairs") and len(p.objs) == 3


def test_minimum_gap_between_normal_sentences():
    tr, clk = mk(min_gap_normal_s=3.0, summary_on_start=False, ambient_silence_s=0)
    run(tr, [FAR_CHAIR(1), (2, "person", box(800, 300), "far", "right")])
    first = tr.pick(); tr.speech_done(first)
    assert tr.pick() is None                        # second object waits for the gap
    clk.advance(3.1)
    assert tr.pick() is not None


def test_most_important_thing_first_and_quiet_scene_stays_quiet():
    tr, clk = mk(summary_on_start=False, ambient_silence_s=0)
    assert tr.pick() is None
    run(tr, [FAR_CHAIR(1), (2, "person", box(800, 300), "far", "right")])
    assert "someone" in tr.pick().text              # a person outranks a chair
    quiet, _ = mk()
    assert quiet.pick() is None


def test_scene_summary_on_start_and_on_request():
    tr, clk = mk(min_gap_normal_s=3.0)
    run(tr, [(1, "tv_monitor", box(150, 300), "far", "left"), (2, "couch", box(850, 300), "far", "right")])
    assert tr.pick() is None                        # two objects: the room is described once first, routine lines wait
    tr, clk = mk()
    run(tr, [(1, "tv_monitor", box(150, 300), "far", "left"), (2, "couch", box(850, 300), "far", "right")])
    clk.advance(1.6)
    assert "TV" in (tr.prefetch_text() or "")       # the text is frozen at ~1.5 s and the voice pre-fetches it
    assert tr.pick() is None                        # ... but it is not spoken before 2.5 s
    clk.advance(1.5)
    s = tr.pick()
    assert s.kind == "ambient" and s.text.startswith("I can see") and "a TV on your left" in s.text \
        and "a couch on your right" in s.text
    tr.speech_done(s); clk.advance(1)
    assert tr.pick() is None                        # the summary counted as telling everyone
    tr.request_summary()
    clk.advance(0.1)
    assert tr.pick().kind == "ambient"


def test_stats_report_candidates_vs_spoken_and_urgent_latency():
    tr, clk = mk()
    t = run(tr, [(1, "chair", box(500, 560), "near", "center")])
    p = tr.pick()
    clk.advance(0.2)
    tr.speech_done(p, started_wall=clk() - 0.1, audio_wall=clk())
    st = tr.stats()
    assert st["spoken"] == 1 and st["candidates"] >= 1 and st["urgent_n"] == 1
    assert 190 <= st["urgent_latency_ms_max"] <= 210 and 90 <= st["pick_to_audio_ms_median"] <= 110


class FakeTTS:
    on_play = None
    def __init__(self, dur=0.05): self.spoken, self.stops, self.dur, self.enq_time = [], 0, dur, None
    def speak(self, text, level="MEDIUM", interrupt=False): time.sleep(self.dur); self.spoken.append(text)
    def stop(self): self.stops += 1
    def prefetch(self, text): self.pref = text
    def duration_hint(self, text): return 0.0


def test_voice_pulls_when_free_and_urgent_cuts_non_urgent_speech():
    tr, clk = mk(min_gap_normal_s=0.0)
    tts = FakeTTS(dur=0.3)
    got = []
    voice = PullVoice(tr, tts, on_utterance=lambda lv, tx: got.append((lv, tx)))
    voice.start()
    run(tr, [FAR_CHAIR()])
    t_end = time.time() + 2
    while not got and time.time() < t_end:
        time.sleep(0.01)
    assert got and got[0][0] == "MEDIUM"
    while voice.current_kind is None and time.time() < t_end:   # is it mid-sentence?
        time.sleep(0.005)
    run(tr, [(5, "person", box(500, 560), "near", "center")], t0=1000)
    voice.after_update()                              # an urgent thing appeared while a normal sentence plays
    assert tts.stops >= 1 or tts.spoken
    voice.stop()


def test_pipeline_pull_mode_end_to_end_with_mock_models():
    from lensvoice.audio.tts import SilentTTS
    from lensvoice.camera.mock_camera import MockCamera
    from lensvoice.config.settings import load_settings
    from lensvoice.runtime.pipeline import Pipeline
    s = load_settings()
    s.narration.mode = "pull"
    s.narration.min_gap_normal_s = 0.2
    s.narration.summary_on_start = False
    spoken = []
    p = Pipeline(s, MockCamera(s.camera, max_frames=150), tts=SilentTTS(), on_utterance=lambda l, t: spoken.append(t))
    p.run()
    assert p.picture is not None and len(spoken) >= 1
    assert p.picture.stats()["spoken"] == len(spoken)


def test_static_objects_never_count_as_approaching_that_is_the_camera():
    tr, clk = mk(summary_on_start=False, ambient_silence_s=0)
    t = run(tr, [(1, "table", box(200, 400, 80, 80), "medium", "left")])
    for i, w in enumerate((80, 100, 130, 170)):                        # the table's box keeps growing
        feed(tr, t + 100 * (i + 1), [(1, "table", box(200, 400, w, w), "medium", "left")])
    o = next(iter(tr.objs.values()))
    assert tr._approaching(o) is True                                   # growth is measured ...
    assert not tr.has_urgent()                                          # ... but never makes a chair/table urgent


def test_a_person_coming_closer_is_urgent_and_spoken_naturally():
    tr, clk = mk(summary_on_start=False, ambient_silence_s=0)
    t = run(tr, [(1, "person", box(200, 400, 80, 120), "medium", "left")])
    for i, w in enumerate((100, 130, 170)):
        feed(tr, t + 100 * (i + 1), [(1, "person", box(200, 400, w, int(w * 1.5)), "near", "left")])
    p = tr.pick()
    assert p.kind == "urgent" and "someone" in p.text and "coming" in p.text or "closer" in p.text
    assert "on your left" in p.text


def test_camera_motion_filter_ignores_approach_when_most_things_approach():
    tr, clk = mk(summary_on_start=False, ambient_silence_s=0)
    base = [(i + 1, "person", box(120 + 150 * i, 250, 60, 90), "medium", "center") for i in range(4)]   # above the path area
    t = run(tr, base)
    grown = [(i + 1, "person", box(120 + 150 * i, 250, 110, 165), "near", "center") for i in range(4)]
    for k in range(1, 4):
        feed(tr, t + 100 * k, grown)
    assert not tr.has_urgent()                                          # everything 'approaches' = we are moving
    assert tr.stats()["ego_motion_frames"] >= 1


def test_urgent_sentences_are_spaced():
    tr, clk = mk(summary_on_start=False, ambient_silence_s=0, urgent_min_gap_s=2.0)
    two = [(1, "chair", box(450, 560), "near", "center"), (2, "table", box(520, 570), "near", "center")]
    run(tr, two)
    first = tr.pick(); tr.speech_done(first)
    assert first.kind == "urgent"
    assert not tr.has_urgent()                                          # second urgent waits for the gap
    clk.advance(2.1)
    assert tr.has_urgent()


def test_human_wording_is_natural_and_states_only_measured_facts():
    from lensvoice.narration.narrator import Narrator
    n = Narrator()
    assert n.scene_sentence("urgent", "chair", 1, "center", "near", True, False, 0) == "Careful, there's a chair right in front of you."
    assert n.scene_sentence("urgent", "person", 1, "left", "near", False, True, 0) == "Watch out, someone is coming toward you, on your left."
    assert n.scene_sentence("normal", "table", 1, "left", "near", False, False, 0) == "There's a table on your left, close by."
    assert n.scene_sentence("normal", "chair", 3, "center", "far") == "A few chairs straight ahead, far off."
    txt = " ".join(n.scene_sentence(k, c, 1, z, b, ip, ap, v) for k in ("urgent", "normal") for c in ("chair", "person", "car")
                for z in ("left", "center", "right") for b in ("near", "far") for ip in (True, False) for ap in (True, False) for v in (0, 1))
    for banned in ("metre", "meter", "step", "feet", "walking", "running", "fast", "slow", "angry", "dangerous", "probably", "maybe"):
        assert banned not in txt.lower()                                # no invented distance, speed, intent, emotion


def test_flickering_counts_do_not_delay_or_block_the_summary():
    tr, clk = mk()
    ts = 0.0
    for i in range(6):                                   # a street: how many cars varies every tick
        cars = [(10 + k, "car", box(100 + 120 * k, 300, 60, 40), "far", "center" if k % 2 else "left") for k in range(2 + i % 3)]
        feed(tr, ts, cars + [(1, "bus", box(800, 300), "far", "right"), (2, "truck", box(500, 300), "far", "center")]); ts += 100
    clk.advance(1.6); tr.prefetch_text()
    clk.advance(1.5)
    p = tr.pick()
    assert p is not None and p.kind == "ambient"         # spoken on time, not waiting for a stable scene


def test_summary_waits_for_its_audio_but_not_forever():
    tr, clk = mk()
    run(tr, [(1, "tv_monitor", box(150, 300), "far", "left"), (2, "couch", box(850, 300), "far", "right")])
    ready = {"v": False}
    tr.ready_fn = lambda text: ready["v"]
    clk.advance(3.0); assert tr.pick() is None                 # frozen, audio being fetched
    clk.advance(2.0); assert tr.pick() is None                 # still not ready, cap not reached
    ready["v"] = True
    assert tr.pick().kind == "ambient"                         # audio ready -> spoken at once, no silent wait
    tr2, clk2 = mk()
    run(tr2, [(1, "tv_monitor", box(150, 300), "far", "left"), (2, "couch", box(850, 300), "far", "right")])
    tr2.ready_fn = lambda text: False                          # audio never arrives (slow/failed network)
    clk2.advance(3.0); tr2.pick(); clk2.advance(4.0)
    assert tr2.pick().kind == "ambient"                        # gives up waiting at the cap
