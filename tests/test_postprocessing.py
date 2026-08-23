from src.detection.postprocessing import postprocess_detections, validate_bbox


def test_validate_bbox_ok():
    box = validate_bbox(10, 20, 100, 200, 640, 480)
    assert box is not None
    assert box.x1 == 10
    assert box.y2 == 200


def test_validate_bbox_invalid_order():
    assert validate_bbox(100, 20, 10, 200, 640, 480) is None


def test_validate_bbox_clamp():
    box = validate_bbox(-10, -5, 700, 500, 640, 480)
    assert box is not None
    assert box.x1 == 0
    assert box.y1 == 0
    assert box.x2 == 639
    assert box.y2 == 479


def test_confidence_filter():
    raw = [
        {"class_id": 0, "confidence": 0.9, "x1": 1, "y1": 1, "x2": 50, "y2": 50},
        {"class_id": 1, "confidence": 0.2, "x1": 1, "y1": 1, "x2": 50, "y2": 50},
    ]
    out = postprocess_detections(
        raw,
        class_mapping={0: "short_sleeved_shirt", 1: "long_sleeved_shirt"},
        garment_class_ids=[0, 1],
        confidence_threshold=0.5,
        image_width=640,
        image_height=480,
    )
    assert len(out) == 1
    assert out[0].class_name == "short_sleeved_shirt"


def test_empty_detections():
    out = postprocess_detections(
        [],
        class_mapping={},
        garment_class_ids=[],
        confidence_threshold=0.5,
        image_width=640,
        image_height=480,
    )
    assert out == []


def test_keep_garment_only():
    raw = [
        {"class_id": 2, "confidence": 0.9, "x1": 1, "y1": 1, "x2": 50, "y2": 50},
        {"class_id": 3, "confidence": 0.9, "x1": 1, "y1": 1, "x2": 50, "y2": 50},
    ]
    out = postprocess_detections(
        raw,
        class_mapping={0: "accessories", 1: "bags", 2: "clothing", 3: "shoes"},
        garment_class_ids=[2],
        confidence_threshold=0.5,
        image_width=640,
        image_height=480,
        keep_garment_only=True,
    )
    assert len(out) == 1
    assert out[0].class_name == "clothing"


def test_class_mapping_fallback():
    raw = [
        {
            "class_id": 99,
            "confidence": 0.8,
            "x1": 1,
            "y1": 1,
            "x2": 10,
            "y2": 10,
            "class_name": "custom",
        }
    ]
    out = postprocess_detections(
        raw,
        class_mapping={},
        garment_class_ids=[],
        confidence_threshold=0.5,
        image_width=640,
        image_height=480,
    )
    assert out[0].class_name == "custom"

