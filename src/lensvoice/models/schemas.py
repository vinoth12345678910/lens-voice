"""Framework-free data contracts for the LensVoice perception pipeline.

Everything here is plain dataclasses (serialisable via `dataclasses.asdict`)
and depends only on NumPy for image / depth buffers. No Ultralytics, OpenCV,
PaddleOCR, or depth-framework types leak into these contracts.

Bounding box convention (documented contract):
    BBox(x1, y1, x2, y2)
    - absolute pixel coordinates in the source frame
    - (x1, y1) top-left, (x2, y2) bottom-right, x2 >= x1, y2 >= y1
    - helper properties: width, height, center_x, center_y, area
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional

import numpy as np


def to_dict(obj: Any) -> dict:
    """Best-effort serialisation (NumPy arrays become shapes/None for logs)."""
    return asdict(obj)


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"invalid bbox (x2>=x1, y2>=y1 required): {self}")

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> "BBox":
        return cls(x, y, x + w, y + h)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def intersection(self, other: "BBox") -> float:
        w = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        h = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        return w * h

    def iou(self, other: "BBox") -> float:
        inter = self.intersection(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def center_delta_to(self, other: "BBox") -> float:
        return abs(self.center_x - other.center_x) + abs(self.center_y - other.center_y)

    def size_ratio_to(self, other: "BBox") -> float:
        if other.area <= 0:
            return 1.0
        return self.area / other.area


@dataclass
class Frame:
    image: np.ndarray  # (H, W, 3) uint8 in BGR or RGB; pipeline does not care
    timestamp: float
    frame_id: int
    width: int = 0
    height: int = 0

    def __post_init__(self) -> None:
        if not self.width or not self.height:
            h, w = self.image.shape[:2]
            self.width = w
            self.height = h


@dataclass
class DetectedObject:
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox
    timestamp: float
    track_id: Optional[int] = None


@dataclass
class ModelOutput:
    timestamp: float
    frame_id: int
    detections: List[DetectedObject] = field(default_factory=list)


@dataclass
class OCRResult:
    text: str
    confidence: float
    timestamp: float
    bbox: Optional[BBox] = None


@dataclass
class TrafficSignResult:
    sign_class: str
    confidence: float
    timestamp: float
    bbox: Optional[BBox] = None


@dataclass
class DepthResult:
    timestamp: float
    depth: Optional[np.ndarray] = None  # (H, W) float, RELATIVE, not metric
    metadata: dict = field(default_factory=dict)  # e.g. {"calibrated": False}


@dataclass
class SceneResult:
    scene_label: str
    confidence: float
    timestamp: float


@dataclass
class ObjectState:
    """Attributed evidence for a tracked object in a given frame."""
    track_id: Optional[int] = None
    status: str = "unknown"          # new | persistent | disappeared
    movement: str = "unknown"        # stationary | moving | approaching | receding | unknown
    h_zone: str = "unknown"          # left | center | right
    v_zone: str = "unknown"          # upper | middle | lower
    distance: str = "unknown"        # very_near | near | medium | far | unknown
    observations: int = 0


@dataclass
class Event:
    """A structured, evidence-based event for narration/priority/audio.

    `severity` is filled in later stages; narration only speaks from these
    fields + the associated Context — never from unsupported evidence.
    """
    type: str                          # e.g. new_object, approaching, hazard_stop...
    source: str                        # module that produced it
    timestamp: float
    confidence: float = 1.0
    track_id: Optional[int] = None
    bbox: Optional[BBox] = None
    detail: dict = field(default_factory=dict)
    severity: str = ""                 # CRITICAL | HIGH | MEDIUM | LOW (set by priority)


@dataclass
class Context:
    """Structured snapshot of world evidence — no natural language here."""
    timestamp: float
    frame_id: int
    scene: Optional[SceneResult] = None
    objects: List[DetectedObject] = field(default_factory=list)
    object_states: dict = field(default_factory=dict)   # track_id -> ObjectState
    text: List[OCRResult] = field(default_factory=list)
    signs: List[TrafficSignResult] = field(default_factory=list)
    hazards: List[Event] = field(default_factory=list)
    changes: List[Event] = field(default_factory=list)


@dataclass(frozen=True)
class PriorityScore:
    """Priority decision for one event."""
    event: Event
    score: float
    level: str  # CRITICAL | HIGH | MEDIUM | LOW


__all__ = [
    "BBox", "Frame", "DetectedObject", "ModelOutput", "OCRResult",
    "TrafficSignResult", "DepthResult", "SceneResult", "ObjectState",
    "Event", "Context", "PriorityScore", "to_dict",
]