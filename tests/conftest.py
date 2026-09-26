"""Shared test fixtures."""
from __future__ import annotations

import numpy as np
import pytest

from lensvoice.config.settings import Settings, load_settings
from lensvoice.models.schemas import Frame


@pytest.fixture(scope="session")
def settings() -> Settings:
    return load_settings()


@pytest.fixture()
def small_frame() -> Frame:
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    return Frame(image=img, timestamp=1000.0, frame_id=1)


@pytest.fixture()
def mock_frame() -> Frame:
    """Fully deterministic frame for close-to-mid camera wedge tests."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    return Frame(image=img, timestamp=1.0, frame_id=1)


def make_frame(width=640, height=360, frame_id=1, ts=1000.0) -> Frame:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    return Frame(image=img, timestamp=ts, frame_id=frame_id)