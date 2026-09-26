"""Unit: config loading + taxonomy integrity."""
from __future__ import annotations

import pytest

from lensvoice.config.settings import Settings, load_settings

EXPECTED_TAXONOMY = {
    0: "person", 1: "car", 2: "motorcycle", 3: "bus", 4: "truck",
    5: "autorickshaw", 6: "bicycle", 7: "chair", 8: "table", 9: "couch",
    10: "bed", 11: "backpack", 12: "bag", 13: "suitcase", 14: "bottle",
    15: "cup", 16: "bowl", 17: "laptop", 18: "cell_phone", 19: "book",
    20: "keyboard", 21: "mouse", 22: "tv_monitor", 23: "bench",
    24: "umbrella", 25: "dog",
}


def test_defaults_load(settings: Settings):
    assert settings.n_classes == 26
    assert settings.camera.width == 1280
    assert settings.scheduler.detection_fps == 10
    assert settings.tracking.method == "mock"


def test_models_default_disabled(settings: Settings):
    for name in ("yolo", "ocr", "traffic_sign", "depth", "scene"):
        assert settings.model(name).enabled is False
        assert settings.model(name).weight == ""


def test_taxonomy_lock():
    settings = load_settings()
    assert settings.class_names == EXPECTED_TAXONOMY


def test_settings_from_dict_with_custom_override():
    s = Settings.from_dict({"camera": {"width": 640, "height": 480}})
    assert s.camera.width == 640 and s.camera.height == 480
    assert s.scheduler.detection_fps == 10  # untouched defaults


def test_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_settings(str(tmp_path / "nope.yaml"))