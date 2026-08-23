from src.config import load_config


def test_confidence_override():
    cfg = load_config(confidence=0.7)
    assert cfg.inference.confidence_threshold == 0.7


def test_device_override():
    cfg = load_config(device="cpu")
    assert cfg.model.device == "cpu"


def test_class_mapping_keys_are_int():
    cfg = load_config()
    assert all(isinstance(k, int) for k in cfg.model.class_mapping)
