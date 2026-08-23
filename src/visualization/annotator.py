from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.detection.types import BBox
from src.inspection.inspection_zone import InspectionZone
from src.inspection.types import GarmentInspectionResult, TrackedDetection


def draw_zone(frame: np.ndarray, zone: InspectionZone) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(
        out,
        (int(zone.x1), int(zone.y1)),
        (int(zone.x2), int(zone.y2)),
        (80, 180, 80),
        2,
    )
    cv2.putText(
        out,
        "INSPECTION ZONE",
        (int(zone.x1) + 8, int(zone.y1) + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 180, 80),
        1,
        cv2.LINE_AA,
    )
    return out


def draw_tracks(frame: np.ndarray, tracks: list[TrackedDetection]) -> np.ndarray:
    out = frame.copy()
    for t in tracks:
        b = t.detection.bbox
        cv2.rectangle(
            out,
            (int(b.x1), int(b.y1)),
            (int(b.x2), int(b.y2)),
            (180, 120, 40),
            2,
        )
        label = f"ID {t.track_id} {t.detection.class_name} {t.detection.confidence:.2f}"
        cv2.putText(
            out,
            label,
            (int(b.x1), max(16, int(b.y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 120, 40),
            1,
            cv2.LINE_AA,
        )
    return out


def draw_evidence(
    frame: np.ndarray,
    *,
    track_id: int,
    garment_bbox: BBox,
    result: GarmentInspectionResult,
) -> np.ndarray:
    out = frame.copy()
    gb = garment_bbox
    cv2.rectangle(
        out,
        (int(gb.x1), int(gb.y1)),
        (int(gb.x2), int(gb.y2)),
        (180, 120, 40),
        2,
    )
    if result.defect_bbox is not None:
        db = result.defect_bbox
        cv2.rectangle(
            out,
            (int(db.x1), int(db.y1)),
            (int(db.x2), int(db.y2)),
            (40, 40, 220),
            2,
        )
    status = result.status
    dtype = result.defect_type or "-"
    label = f"ID {track_id} {status} {dtype} {result.confidence:.2f}"
    color = (40, 40, 220) if status == "DEFECT" else (60, 180, 80)
    cv2.putText(
        out,
        label,
        (int(gb.x1), max(18, int(gb.y1) - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    return out


def save_evidence_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
