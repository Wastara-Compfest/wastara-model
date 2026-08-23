from pathlib import Path

import numpy as np
import pytest

from src.config import load_config
from src.detection.detector import GarmentDetector
from src.detection.preprocessing import prepare_frame
from src.device import resolve_device


def test_prepare_frame_ok():
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    out = prepare_frame(frame)
    assert out.shape == (32, 32, 3)


def test_prepare_frame_invalid():
    with pytest.raises(ValueError):
        prepare_frame(np.zeros((32, 32), dtype=np.uint8))


def test_resolve_device_cpu():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto_returns_known():
    device = resolve_device("auto")
    assert device in {"cuda", "mps", "cpu"}


def test_load_config_defaults():
    cfg = load_config()
    assert cfg.model.dataset_name == "conveyor_clothing"
    assert cfg.inference.confidence_threshold == 0.5
    assert 0 in cfg.model.class_mapping


def test_detector_missing_weights(tmp_path: Path):
    det = GarmentDetector(tmp_path / "missing.pt", device="cpu")
    with pytest.raises(FileNotFoundError):
        det.load()


def test_detector_empty_prediction_path(monkeypatch, tmp_path: Path):
    """When model returns empty results, detector yields []."""
    weights = tmp_path / "dummy.pt"
    weights.write_bytes(b"not-a-real-model")

    class FakeBoxes:
        def __len__(self):
            return 0

    class FakeResult:
        boxes = FakeBoxes()

    class FakeModel:
        names = {0: "short_sleeved_shirt"}

        def predict(self, **kwargs):
            return [FakeResult()]

    det = GarmentDetector(
        weights,
        device="cpu",
        class_mapping={0: "short_sleeved_shirt"},
        garment_class_ids=[0],
    )
    det._model = FakeModel()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    assert det.predict(frame) == []
