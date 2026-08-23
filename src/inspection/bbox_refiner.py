"""Refine garment bboxes before defect analysis.

YOLO sometimes splits one shirt into thin horizontal strips on a conveyor.
Tracking can latch onto a strip, leaving stains outside the ROI. We re-detect
at lower confidence and expand the track bbox downward — not upward into
other garments stacked above on the belt.
"""

from __future__ import annotations

from src.detection.detector import GarmentDetector
from src.detection.types import BBox, Detection


def bbox_area(bbox: BBox) -> float:
    return max(0.0, bbox.x2 - bbox.x1) * max(0.0, bbox.y2 - bbox.y1)


def bbox_union(a: BBox, b: BBox) -> BBox:
    return BBox(
        x1=min(a.x1, b.x1),
        y1=min(a.y1, b.y1),
        x2=max(a.x2, b.x2),
        y2=max(a.y2, b.y2),
    )


def horizontal_overlap_ratio(a: BBox, b: BBox) -> float:
    """Overlap along x-axis divided by the smaller box width."""
    left = max(a.x1, b.x1)
    right = min(a.x2, b.x2)
    overlap = max(0.0, right - left)
    min_w = max(1.0, min(a.x2 - a.x1, b.x2 - b.x1))
    return overlap / min_w


def vertical_gap(a: BBox, b: BBox) -> float:
    """Pixels between boxes; 0 when they overlap vertically."""
    if a.y2 < b.y1:
        return b.y1 - a.y2
    if b.y2 < a.y1:
        return a.y1 - b.y2
    return 0.0


def is_degenerate_garment_bbox(bbox: BBox, frame_height: int) -> bool:
    """Thin horizontal strip — usually a partial YOLO split, not full garment."""
    width = max(1.0, bbox.x2 - bbox.x1)
    height = max(1.0, bbox.y2 - bbox.y1)
    if height < 0.12 * frame_height and width / height > 6.0:
        return True
    return False


def _is_separate_upper_garment(candidate: BBox, track_bbox: BBox) -> bool:
    """Another shirt stacked above the tracked piece on the conveyor."""
    track_h = max(1.0, track_bbox.y2 - track_bbox.y1)
    upper_cutoff = track_bbox.y1 + 0.45 * track_h
    return candidate.y2 <= upper_cutoff and candidate.y1 < track_bbox.y1


def expand_track_bbox_downward(
    track_bbox: BBox,
    detections: list[Detection],
    *,
    frame_height: int,
    horizontal_overlap_threshold: float = 0.45,
    max_vertical_gap_ratio: float = 0.08,
) -> BBox:
    """Grow track bbox using low-conf detections below / beside it on the belt."""
    max_gap = max_vertical_gap_ratio * frame_height
    merged = track_bbox
    candidates = [d.bbox for d in detections]

    changed = True
    while changed:
        changed = False
        for candidate in candidates:
            if horizontal_overlap_ratio(merged, candidate) < horizontal_overlap_threshold:
                continue
            if vertical_gap(merged, candidate) > max_gap:
                continue
            if _is_separate_upper_garment(candidate, track_bbox):
                continue
            # only pull in boxes that extend the garment downward or overlap it
            if candidate.y2 <= merged.y1 + 0.15 * max(1.0, merged.y2 - merged.y1):
                continue
            expanded = bbox_union(merged, candidate)
            if expanded.y2 > merged.y2 + 1 or expanded.y1 < merged.y1:
                merged = expanded
                changed = True

    return merged


def merge_vertical_garment_boxes(
    detections: list[Detection],
    *,
    frame_height: int,
    horizontal_overlap_threshold: float = 0.45,
    max_vertical_gap_ratio: float = 0.08,
) -> list[BBox]:
    """Merge vertically stacked garment boxes (used in tests / optional grouping)."""
    boxes = [d.bbox for d in detections]
    if len(boxes) <= 1:
        return boxes

    max_gap = max_vertical_gap_ratio * frame_height
    changed = True
    while changed:
        changed = False
        merged: list[BBox] = []
        used = [False] * len(boxes)
        for i, base in enumerate(boxes):
            if used[i]:
                continue
            current = base
            used[i] = True
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                other = boxes[j]
                if horizontal_overlap_ratio(current, other) < horizontal_overlap_threshold:
                    continue
                if vertical_gap(current, other) > max_gap:
                    continue
                current = bbox_union(current, other)
                used[j] = True
                changed = True
            merged.append(current)
        boxes = merged
    return boxes


def pick_best_garment_bbox(
    candidates: list[BBox],
    track_bbox: BBox,
    *,
    frame_height: int,
) -> BBox:
    """Choose merged detection that best represents the tracked garment."""
    if not candidates:
        return track_bbox

    degenerate = is_degenerate_garment_bbox(track_bbox, frame_height)
    overlapping = [
        c
        for c in candidates
        if horizontal_overlap_ratio(c, track_bbox) >= 0.45
        and not _is_separate_upper_garment(c, track_bbox)
    ]
    if not overlapping:
        return track_bbox

    if degenerate:
        # prefer the largest downward extension for thin strip tracks
        below = [c for c in overlapping if c.y2 >= track_bbox.y2 - 2]
        if below:
            return max(below, key=bbox_area)

    return max(overlapping, key=bbox_area)


def refine_garment_bbox(
    frame,
    track_bbox: BBox,
    detector: GarmentDetector,
    *,
    low_confidence: float = 0.15,
) -> BBox:
    """Re-detect at lower confidence and expand split garment boxes for defect ROI."""
    frame_height = frame.shape[0]
    original_conf = detector.confidence_threshold
    detector.confidence_threshold = low_confidence
    try:
        detections = detector.predict(frame)
    finally:
        detector.confidence_threshold = original_conf

    expanded = expand_track_bbox_downward(
        track_bbox,
        detections,
        frame_height=frame_height,
    )
    if expanded.y2 > track_bbox.y2 + 5 or expanded.y1 < track_bbox.y1:
        return expanded

    # fallback: pick single best low-conf box
    return pick_best_garment_bbox(
        [d.bbox for d in detections],
        track_bbox,
        frame_height=frame_height,
    )
