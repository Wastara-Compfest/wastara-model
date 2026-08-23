from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.detection.types import BBox
from src.inspection.fabric_mask import fabric_mask
from src.inspection.roi_extractor import GarmentROI


@dataclass
class TypeSuggestion:
    suggested_type: str | None
    confidence: float
    method: str = "rule_v1"


def suggest_defect_type(
    roi: GarmentROI,
    local_bbox: BBox,
    *,
    anomaly_score: float,
    min_confidence: float = 0.55,
) -> TypeSuggestion:
    """Suggest a PRD defect type from blob shape/color — not a final label."""
    image = roi.image
    h, w = image.shape[:2]
    x1 = max(0, int(local_bbox.x1))
    y1 = max(0, int(local_bbox.y1))
    x2 = min(w, int(local_bbox.x2))
    y2 = min(h, int(local_bbox.y2))
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return TypeSuggestion(None, 0.0)

    blob = image[y1:y2, x1:x2]
    bh, bw = blob.shape[:2]
    area = float(bw * bh)
    fabric = fabric_mask(image)
    fabric_area = max(1.0, float((fabric > 0).sum()))
    area_ratio = area / fabric_area

    gray = cv2.cvtColor(blob, cv2.COLOR_BGR2GRAY)
    fabric_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[fabric > 0]
    fabric_med = float(np.median(fabric_gray)) if fabric_gray.size else 128.0
    blob_med = float(np.median(gray))
    darkness = max(0.0, (fabric_med - blob_med) / 80.0)

    aspect = bw / max(bh, 1)
    perimeter = cv2.arcLength(
        cv2.convexHull(
            np.array([[0, 0], [bw, 0], [bw, bh], [0, bh]], dtype=np.float32)
        ),
        True,
    )
    circularity = 4.0 * np.pi * area / max(perimeter * perimeter, 1.0)

    edges = cv2.Canny(gray, 40, 120)
    edge_density = float((edges > 0).mean())

    scores: dict[str, float] = {
        "hole": min(
            1.0,
            0.35 * min(1.0, circularity / 0.75)
            + 0.35 * min(1.0, darkness)
            + 0.3 * min(1.0, area_ratio / 0.04),
        ),
        "broken_yarn": min(
            1.0,
            0.4
            * min(
                1.0,
                aspect / 3.0 if aspect > 1.2 else (1.0 / max(aspect, 0.2)) / 3.0,
            )
            + 0.35 * min(1.0, edge_density * 4.0)
            + 0.25 * min(1.0, area_ratio / 0.02),
        ),
        "texture_anomaly": min(
            1.0,
            0.45 * min(1.0, anomaly_score)
            + 0.35 * min(1.0, area_ratio / 0.03)
            + 0.2 * (1.0 - min(1.0, edge_density * 3.0)),
        ),
        "pattern_anomaly": min(1.0, 0.25 * min(1.0, edge_density * 5.0)),
    }

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    if best_type == "pattern_anomaly" and best_score < 0.6:
        best_type = "texture_anomaly"
        best_score = scores["texture_anomaly"]

    combined = min(1.0, best_score * 0.7 + anomaly_score * 0.3)
    if combined < min_confidence:
        if anomaly_score >= min_confidence:
            return TypeSuggestion("unknown", round(anomaly_score * 0.85, 4))
        return TypeSuggestion(None, round(combined, 4))

    return TypeSuggestion(best_type, round(combined, 4))
