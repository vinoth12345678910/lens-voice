"""LensVoice pipeline runtime.

Glues camera -> scheduler -> perception -> tracking -> reasoning ->
change detection -> safety -> priority -> narration -> audio -> telemetry.

Everything the user hears is traceable to structured evidence (Context/Event).
Mock instruments (MockCamera + mock models + SilentTTS) make the pipeline
fully runnable with no GPU, no weights and no webcam.
"""
from __future__ import annotations

from typing import Callable, Optional

from lensvoice.audio.queue import PriorityAudioQueue
from lensvoice.audio.tts import SilentTTS, TTS
from lensvoice.camera.camera import Camera, EndOfStream
from lensvoice.config.settings import Settings
from lensvoice.models.schemas import Context, Event, Frame, PriorityScore
from lensvoice.models import create_models
from lensvoice.narration.narrator import Narrator
from lensvoice.reasoning.change_detection import ChangeDetector
from lensvoice.reasoning.context import ContextEngine
from lensvoice.reasoning.priority import PriorityScorer
from lensvoice.reasoning.safety import SafetyAnalyzer
from lensvoice.reasoning.spatial import SpatialAnalyzer
from lensvoice.reasoning.temporal import TemporalAnalyzer
from lensvoice.scheduler.frame_scheduler import FrameScheduler
from lensvoice.telemetry.logger import Logger
from lensvoice.telemetry.metrics import MetricsCollector
from lensvoice.tracking.tracker import MockTracker
from lensvoice.utils.time import Clock, monotonic_ms


class Pipeline:
    def __init__(self, settings: Settings, camera: Camera,
                 clock: Optional[Clock] = None, logger: Optional[Logger] = None,
                 tts: Optional[TTS] = None,
                 on_utterance: Optional[Callable[[str, str], None]] = None):
        self.settings = settings
        self.camera = camera
        self.clock = clock or Clock()
        self.logger = logger or Logger(settings.logging_)
        self.tts = tts or SilentTTS()

        self.scheduler = FrameScheduler(
            settings.scheduler.camera_fps,
            {"yolo": settings.scheduler.detection_fps,
             "ocr": settings.scheduler.ocr_fps,
             "traffic_sign": settings.scheduler.traffic_sign_fps,
             "depth": settings.scheduler.depth_fps,
             "scene": settings.scheduler.scene_fps})

        models = create_models(settings)
        self.vision = models["yolo"]
        self.ocr = models["ocr"]
        self.signs = models["traffic_sign"]
        self.depth_model = models["depth"]
        self.scene = models["scene"]

        self.tracker = MockTracker(settings.tracking)
        self.spatial = SpatialAnalyzer(settings.spatial)
        self.temporal = TemporalAnalyzer(settings.temporal)
        self.change = ChangeDetector(settings.change_detection, settings.temporal)
        self.context_engine = ContextEngine(self.spatial, self.temporal)
        self.safety = SafetyAnalyzer(settings.safety)
        self.priority = PriorityScorer(settings.priority,
                                       cooldown_ms=settings.audio.cooldown_ms)
        self.narrator = Narrator()
        self.audio = PriorityAudioQueue(settings.audio, self.tts)

        self.metrics = MetricsCollector(settings.metrics.window_seconds,
                                        settings.metrics.enabled)
        self.on_utterance = on_utterance

    # -- public -----------------------------------------------------------
    def start(self) -> None:
        self.camera.start()

    def stop(self) -> None:
        self.camera.stop()

    def run(self, run_forever: bool = True) -> None:
        """Main loop. Mock cameras signal EndOfStream to exit cleanly."""
        self.start()
        self.logger.info("pipeline.start", mode=type(self.camera).__name__)
        try:
            while True:
                try:
                    frame = self.camera.read()
                except EndOfStream:
                    self.logger.info("pipeline.end_of_stream")
                    break
                self.process_frame(frame)
        except KeyboardInterrupt:
            self.logger.info("pipeline.interrupted")
        finally:
            self.stop()
            self.logger.info("pipeline.drain")
            self.audio.drain()

    # -- per-frame --------------------------------------------------------
    def process_frame(self, frame: Frame) -> Context:
        t0 = monotonic_ms()
        due = self.scheduler.due(frame.frame_id)

        detections = []
        if "yolo" in due:
            detections = self.vision.predict(frame).detections
            self._track("yolo_fps")

        scene = None
        if "scene" in due:
            scene = self.scene.classify(frame)

        if detections:
            detections = self.tracker.update(detections)
        else:
            # keep trajectory bookkeeping moving even on non-YOLO frames
            self.tracker.update([])

        ctx = self.context_engine.build(
            frame, detections, self.tracker.tracks, scene,
            self.ocr.read(frame) if "ocr" in due else [],
            self.signs.detect(frame) if "traffic_sign" in due else [],
            self.depth_model.estimate(frame) if "depth" in due else None,
        )

        events = self.change.update(
            frame,
            {d.track_id: d for d in detections if d.track_id is not None},
            ctx.object_states, self.tracker.tracks)

        safety_events = self.safety.analyze(ctx, frame.width)
        events += [e for e in safety_events
                   if e.type != "sign_attention"
                   or self.change.sign_ok(e.detail.get("sign", ""), ctx.timestamp)]

        if ctx.text:
            events += self.change.text_changed(
                ctx.timestamp, [(t.text, "text") for t in ctx.text if t.text])

        scored = self.priority.score(events, ctx)
        for s in scored:
            self.priority.remember(s, ctx.timestamp)

        utterances = self.narrator.generate(ctx, scored)
        for level, text in utterances:
            self.audio.enqueue(level, text)
            self.logger.debug("narration", level=level)
            if self.on_utterance:
                self.on_utterance(level, text)
            self._track("narration_count")

        self.metrics.record("frame_ms", monotonic_ms() - t0)
        self.metrics.record("detections", len(detections))
        return ctx

    def _track(self, name: str) -> None:
        self.metrics.record(name, 1.0)


__all__ = ["Pipeline"]