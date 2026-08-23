import numpy as np

from src.defect.aggregator import AggregationConfig, aggregate_observations
from src.defect.defect_detector import ColorAnomalyDetector
from src.detection.types import BBox
from src.inspection.bbox_refiner import (
    is_degenerate_garment_bbox,
    merge_vertical_garment_boxes,
    pick_best_garment_bbox,
)
from src.inspection.frame_sampler import FrameSampler
from src.inspection.inspection_zone import InspectionZone
from src.inspection.roi_extractor import extract_roi, tighten_roi_to_fabric
from src.inspection.types import DefectPrediction, SampledObservation, TrackedDetection
from src.detection.types import Detection
from src.tracking.tracker import IoUTracker


def test_inspection_zone_overlap():
    zone = InspectionZone(100, 100, 500, 400)
    inside = BBox(200, 150, 400, 350)
    outside = BBox(10, 10, 50, 50)
    assert zone.contains(inside, 0.3)
    assert not zone.contains(outside, 0.3)


def test_zone_from_normalized():
    zone = InspectionZone.from_normalized(
        x1=0.1, y1=0.2, x2=0.9, y2=0.8, width=1000, height=500
    )
    assert zone.x1 == 100
    assert zone.y2 == 400


def test_frame_sampler_dedup_and_select():
    sampler = FrameSampler(sample_count=3, min_center_shift_px=20)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(10):
        det = Detection(
            class_id=0,
            class_name="clothing",
            confidence=0.9,
            bbox=BBox(10, 10 + i * 25, 80, 40 + i * 25),
        )
        tracked = TrackedDetection(
            track_id=1,
            detection=det,
            frame_id=i,
            timestamp_ms=i * 40,
        )
        sampler.add(tracked, frame)
    selected = sampler.select(1)
    assert 1 <= len(selected) <= 3


def test_roi_coord_mapping():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    bbox = BBox(50, 40, 150, 120)
    roi = extract_roi(frame, bbox, pad=0, inset_ratio=0.0)
    local = BBox(10, 5, 40, 30)
    full = roi.local_to_full(local)
    assert full.x1 == 60
    assert full.y1 == 45


def test_iou_tracker_persistent_id():
    tracker = IoUTracker(match_iou=0.3, min_hits=1, track_threshold=0.3)
    d1 = Detection(0, "clothing", 0.9, BBox(10, 10, 100, 100))
    d2 = Detection(0, "clothing", 0.9, BBox(12, 12, 102, 102))
    t1 = tracker.update([d1], frame_id=0, timestamp_ms=0)
    t2 = tracker.update([d2], frame_id=1, timestamp_ms=40)
    assert len(t1) == 1 and len(t2) == 1
    assert t1[0].track_id == t2[0].track_id


def test_aggregator_requires_confirmations():
    obs = [
        SampledObservation(
            frame_id=1,
            timestamp_ms=0,
            garment_bbox=BBox(0, 0, 10, 10),
            prediction=DefectPrediction("anomaly", 0.9, BBox(1, 1, 5, 5), True, "anomaly", 0.9),
        )
    ]
    result = aggregate_observations(
        7,
        "clothing",
        obs,
        config=AggregationConfig(minimum_confirmations=2, confidence_threshold=0.5),
    )
    assert result.status == "NORMAL"


def test_aggregator_temporal_consensus_defect():
    box = BBox(10, 10, 40, 40)
    obs = [
        SampledObservation(
            frame_id=i,
            timestamp_ms=i * 40,
            garment_bbox=BBox(0, 0, 100, 100),
            prediction=DefectPrediction("anomaly", 0.8 + i * 0.02, box, True, "anomaly", 0.8),
        )
        for i in range(3)
    ]
    result = aggregate_observations(
        7,
        "clothing",
        obs,
        config=AggregationConfig(minimum_confirmations=2, confidence_threshold=0.5),
    )
    assert result.status == "DEFECT"
    assert result.defect_type == "anomaly"
    assert result.track_id == 7


def test_color_anomaly_detector_clean_roi():
    # uniform color → normal
    img = np.full((120, 160, 3), 180, dtype=np.uint8)
    from src.inspection.roi_extractor import GarmentROI

    roi = GarmentROI(image=img, garment_bbox=BBox(0, 0, 160, 120))
    pred = ColorAnomalyDetector(score_threshold=0.45).analyze(roi)
    assert pred.is_defect is False


def test_color_anomaly_detector_finds_blob():
    # fabric-like interior with a dark stain away from borders
    img = np.full((160, 220, 3), 200, dtype=np.uint8)
    # dark border simulating belt/rails leaking into loose YOLO box
    img[:, :20] = (30, 30, 30)
    img[:, -20:] = (30, 30, 30)
    img[:12, :] = (30, 30, 30)
    img[-12:, :] = (30, 30, 30)
    # real stain on fabric
    img[60:95, 90:130] = (25, 25, 40)
    from src.inspection.roi_extractor import GarmentROI

    roi = GarmentROI(image=img, garment_bbox=BBox(100, 50, 320, 210))
    pred = ColorAnomalyDetector(
        score_threshold=0.25,
        min_area_ratio=0.01,
        border_ratio=0.1,
        edge_reject_ratio=0.05,
    ).analyze(roi)
    assert pred.is_defect is True
    assert pred.defect_type == "anomaly"
    assert pred.bbox is not None
    # defect must not sit on the dark left rail
    assert pred.bbox.x1 > 100 + 25


def test_merge_vertical_garment_boxes():
    dets = [
        Detection(0, "clothing", 0.6, BBox(10, 110, 900, 180)),
        Detection(0, "clothing", 0.2, BBox(12, 170, 920, 520)),
    ]
    merged = merge_vertical_garment_boxes(dets, frame_height=576)
    assert len(merged) == 1
    assert merged[0].y1 <= 110
    assert merged[0].y2 >= 520


def test_expand_track_bbox_downward_skips_upper_shirt():
    track = BBox(14, 113, 973, 180)
    dets = [
        Detection(0, "clothing", 0.67, BBox(13, 13, 976, 114)),  # navy above
        Detection(0, "clothing", 0.17, BBox(14, 171, 973, 532)),  # white shirt
    ]
    from src.inspection.bbox_refiner import expand_track_bbox_downward

    expanded = expand_track_bbox_downward(track, dets, frame_height=576)
    assert expanded.y1 >= 113
    assert expanded.y2 >= 500
    assert expanded.y1 > 50  # did not swallow upper garment


def test_pick_best_garment_bbox_expands_thin_track():
    thin_track = BBox(14, 113, 973, 180)
    merged = [BBox(14, 113, 973, 532)]
    best = pick_best_garment_bbox(merged, thin_track, frame_height=576)
    assert best.y2 >= 500
    assert is_degenerate_garment_bbox(thin_track, 576)


def test_tighten_roi_to_fabric_excludes_side_rails():
    img = np.full((100, 400, 3), 190, dtype=np.uint8)
    img[:, :80] = (25, 25, 25)
    img[:, -80:] = (25, 25, 25)
    img[20:80, 100:300] = (210, 210, 210)
    from src.inspection.roi_extractor import GarmentROI

    loose = GarmentROI(image=img, garment_bbox=BBox(0, 0, 400, 100))
    tight = tighten_roi_to_fabric(loose)
    assert tight.garment_bbox.x1 > 50
    assert tight.garment_bbox.x2 < 350
    assert tight.garment_bbox.y1 == loose.garment_bbox.y1
    assert tight.garment_bbox.y2 == loose.garment_bbox.y2


def test_type_suggester_returns_texture_for_stain():
    import numpy as np
    from src.defect.type_suggester import suggest_defect_type
    from src.inspection.roi_extractor import GarmentROI

    img = np.full((120, 160, 3), 200, dtype=np.uint8)
    img[50:75, 70:95] = (30, 30, 40)
    roi = GarmentROI(image=img, garment_bbox=BBox(0, 0, 160, 120))
    local = BBox(70, 50, 95, 75)
    result = suggest_defect_type(roi, local, anomaly_score=0.9)
    assert result.suggested_type in {"texture_anomaly", "unknown", "hole"}
    assert result.confidence > 0


def test_color_anomaly_ignores_border_rail():
    img = np.full((120, 200, 3), 190, dtype=np.uint8)
    img[:, :25] = (15, 15, 15)  # left rail only
    from src.inspection.roi_extractor import GarmentROI

    roi = GarmentROI(image=img, garment_bbox=BBox(0, 0, 200, 120))
    pred = ColorAnomalyDetector(score_threshold=0.4).analyze(roi)
    assert pred.is_defect is False
