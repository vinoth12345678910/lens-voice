"""Typed runtime settings built from configs/default.yaml (or a custom file).

Structure mirrors configs/default.yaml. Values that point at model weights /
camera indices are never hardcoded in code — they come from here.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "configs", "default.yaml"
)


@dataclass
class CameraConfig:
    source: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class SchedulerConfig:
    camera_fps: int = 30
    detection_fps: int = 10
    ocr_fps: int = 2
    traffic_sign_fps: int = 3
    depth_fps: int = 5
    scene_fps: int = 1


@dataclass
class ModelConfig:
    """Per-model settings. `weight` is configurable and intentionally unset by
    default (never hardcoded in code)."""

    enabled: bool = False
    # Only for ocr / traffic_sign slots that are not enabled: True = deterministic
    # mock (tests, demos); False = report nothing instead of canned text/signs.
    mock: bool = True
    weight: str = ""
    confidence: float = 0.25
    iou: float = 0.45
    device: str = "auto"
    imgsz: int = 640
    language: str = "en"
    guess_language: bool = False
    classifier_weight: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class TrackingConfig:
    method: str = "mock"
    iou_threshold: float = 0.35
    max_age_frames: int = 30
    min_confidence: float = 0.05


@dataclass
class SpatialConfig:
    left_boundary: float = 0.333
    right_boundary: float = 0.666
    upper_boundary: float = 0.333
    lower_boundary: float = 0.666
    distance_bins: dict = field(default_factory=lambda: {
        "very_near": 0.15, "near": 0.35, "medium": 0.60, "far": 0.85,
    })


@dataclass
class TemporalConfig:
    min_observations: int = 3
    disappear_frames: int = 3
    movement_ratio: float = 0.03
    approach_ratio: float = 0.85
    recede_ratio: float = 1.18


@dataclass
class ChangeDetectionConfig:
    cooldown_ms: float = 5000.0
    moved_center_ratio: float = 0.10
    distance_drop_levels: int = 1
    text_cooldown_ms: float = 8000.0


@dataclass
class SafetyConfig:
    center_band: float = 0.5
    close_depth_categories: list = field(default_factory=lambda: ["very_near", "near"])
    require_approaching_for_vehicle: bool = True


@dataclass
class PriorityConfig:
    weights: dict = field(default_factory=lambda: {
        "safety": 100.0, "proximity": 30.0, "approaching": 25.0,
        "new_object": 10.0, "navigation": 15.0, "information": 5.0,
    })
    levels: dict = field(default_factory=lambda: {
        "critical": 80.0, "high": 55.0, "medium": 30.0,
    })
    repetition_penalty: float = 8.0


@dataclass
class AudioConfig:
    max_queue_size: int = 4
    cooldown_ms: float = 5000.0
    interrupt_on_critical: bool = True
    # Live mode: a background worker speaks the queue while the camera loop keeps
    # running (False = legacy behaviour: queue is drained once at shutdown).
    async_playback: bool = False
    # Drop queued speech older than this (0 = never). Stale narration is wrong narration.
    max_age_ms: float = 0.0
    # Do not say the same sentence again within this window (0 = off). The priority
    # cooldown only lowers a score, so without this a hazard repeats every refresh.
    dedupe_ms: float = 0.0
    # Minimum gap between routine (LOW priority) announcements, so a busy scene cannot bury the
    # voice in a backlog (0 = off). MEDIUM / HIGH / CRITICAL are never limited.
    low_gap_ms: float = 0.0


@dataclass
class TTSConfig:
    provider: str = "silent"     # silent | sarvam
    voice: str = ""              # sarvam speaker (empty -> "priya")
    language: str = "en-IN"      # sarvam language_code
    model: str = "bulbul:v3"
    pace: float = 1.0
    sample_rate: int = 22050
    timeout_s: float = 8.0
    api_key_env: str = "SARVAM_API_KEY"   # NAME of the env var holding the key; the key itself is never stored
    warm_on_start: bool = False  # pre-fetch the urgent sentences at start (few dozen, cached on disk afterwards)
    cache_dir: str = ""          # persistent audio cache (e.g. ~/.cache/lensvoice/tts); "" = memory only
    prefetch: bool = True        # synthesise a sentence in the background as soon as it is queued
    prefetch_workers: int = 2
    max_rps: float = 2.0         # starting request rate to the API; halves on HTTP 429, recovers slowly
    # If Sarvam audio is not ready within this many ms, speak the sentence NOW with the built-in
    # system voice (macOS `say`); the Sarvam fetch continues and is cached for next time. 0 = off.
    fallback_local_ms: int = 0
    fallback_voice: str = ""     # e.g. "Samantha"; "" = system default
    fallback_rate: int = 210     # words per minute for the fallback voice


@dataclass
class NarrationConfig:
    """Pull-mode narration: a stable scene picture + 'what is the single most important thing
    to say right now?' asked only when the voice is free. mode 'push' = legacy event queue."""
    mode: str = "push"                 # push | pull
    # --- scene stability
    confirm_hits: int = 3              # object exists once seen in this many of the last `confirm_window` detection ticks
    confirm_window: int = 4
    drop_after_s: float = 1.0          # ... and is dropped after this long unseen
    reid_iou: float = 0.30             # a new track id with this IoU to a recent same-class object is the SAME object
    band_hold_ticks: int = 2           # a new distance band / zone must be seen this many ticks in a row to count
    # --- relevance
    path_center_frac: float = 0.40     # "in path": box centre inside the middle 40 % of the frame width ...
    path_bottom_min: float = 0.55      # ... and its bottom edge at least this far down the frame (0 top .. 1 bottom)
    approach_s: float = 0.5            # approaching = box area grew by approach_area_ratio over about this long
    approach_area_ratio: float = 1.25
    hazard_classes: list = field(default_factory=lambda: [
        "person", "car", "motorcycle", "bus", "truck", "autorickshaw", "bicycle", "dog",
        "chair", "table", "bench", "couch", "bed"])
    handheld_classes: list = field(default_factory=lambda: [
        "cup", "bottle", "bowl", "book", "cell_phone", "mouse", "keyboard", "laptop"])
    # --- pacing
    min_gap_normal_s: float = 3.0      # silence between normal sentences
    urgent_cooldown_s: float = 4.0     # per object
    urgent_min_gap_s: float = 2.0      # between ANY two urgent sentences (a flood of urgents is no longer urgent)
    ego_motion_filter: bool = True     # if most objects 'approach' at once it is the CAMERA moving: ignore approach
    repeat_after_s: float = 20.0       # an unchanged object may be mentioned again after this long (at half score)
    interrupt_urgent: bool = True      # an urgent sentence cuts off non-urgent speech in progress
    summary_on_start: bool = True      # one scene summary once the first objects are confirmed
    ambient_silence_s: float = 25.0    # ... and again after this much silence (0 = off)
    summary_max_wait_s: float = 3.5    # a summary's text is frozen, its audio pre-fetched; spoken when ready or after this long


@dataclass
class LoggingConfig:
    level: str = "INFO"
    structured: bool = True
    json: bool = False


@dataclass
class MetricsConfig:
    enabled: bool = True
    window_seconds: int = 60


@dataclass
class Settings:
    n_classes: int = 26
    class_names: dict = field(default_factory=dict)
    camera: CameraConfig = field(default_factory=CameraConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    models: dict = field(default_factory=lambda: {k: ModelConfig() for k in
                                                  ("yolo", "ocr", "traffic_sign",
                                                   "depth", "scene")})
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    change_detection: ChangeDetectionConfig = field(default_factory=ChangeDetectionConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    priority: PriorityConfig = field(default_factory=PriorityConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    narration: NarrationConfig = field(default_factory=NarrationConfig)
    logging_: LoggingConfig = field(default_factory=LoggingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    def resolution(self) -> tuple[int, int]:
        return (self.camera.width, self.camera.height)

    def model(self, name: str) -> ModelConfig:
        return self.models.get(name, ModelConfig())

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        return _from_dict(cls, data)


def _from_dict(cls: type, data: dict) -> Any:
    kw: dict = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            if f.default is not dataclasses.MISSING:
                kw[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                kw[f.name] = f.default_factory()  # type: ignore[misc]
            continue
        raw = data[f.name]
        # resolve the concrete class from the declared default/factory so we
        # still work under `from __future__ import annotations` (f.type is str)
        target = (f.default_factory
                  if f.default_factory is not dataclasses.MISSING
                  else (type(f.default) if f.default is not dataclasses.MISSING else None))
        if isinstance(raw, dict) and dataclasses.is_dataclass(target):
            kw[f.name] = _from_dict(target, raw)
        elif f.name == "models" and isinstance(raw, dict):
            kw[f.name] = {k: _from_dict(ModelConfig, v)
                          if isinstance(v, dict) else ModelConfig()
                          for k, v in raw.items()}
        else:
            kw[f.name] = raw
    return cls(**kw)


def load_settings(path: Optional[str] = None) -> Settings:
    """Load YAML config; falls back to configs/default.yaml."""
    path = path or DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"config not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    return Settings.from_dict(data)


__all__ = [
    "Settings", "ModelConfig", "CameraConfig", "SchedulerConfig", "TrackingConfig",
    "SpatialConfig", "TemporalConfig", "ChangeDetectionConfig", "SafetyConfig",
    "PriorityConfig", "AudioConfig", "TTSConfig", "NarrationConfig", "LoggingConfig", "MetricsConfig",
    "load_settings",
]