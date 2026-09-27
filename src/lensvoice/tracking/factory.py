"""Factory for selecting the LensVoice tracker implementation."""

from lensvoice.config.settings import TrackingConfig
from lensvoice.tracking.bytetrack import ByteTrackTracker
from lensvoice.tracking.tracker import MockTracker


def create_tracker(cfg: TrackingConfig):
    """Create the configured tracker implementation."""

    if cfg.method == "mock":
        return MockTracker(cfg)

    if cfg.method == "bytetrack":
        return ByteTrackTracker(cfg)

    raise ValueError(
        f"Unsupported tracking method: {cfg.method}"
    )