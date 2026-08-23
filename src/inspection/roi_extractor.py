"""Crop garment ROI while preserving coordinate mapping to full frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.detection.types import BBox
from src.inspection.fabric_mask import fabric_mask, fabric_tight_bbox


@dataclass
class GarmentROI:
    image: np.ndarray
    garment_bbox: BBox  # full-frame coords used for crop
    pad: int = 0

    def local_to_full(self, local: BBox) -> BBox:
        """Map a bbox in ROI coords back to full-frame coords."""
        return BBox(
            x1=local.x1 + self.garment_bbox.x1 - self.pad,
            y1=local.y1 + self.garment_bbox.y1 - self.pad,
            x2=local.x2 + self.garment_bbox.x1 - self.pad,
            y2=local.y2 + self.garment_bbox.y1 - self.pad,
        )


def extract_roi(
    frame: np.ndarray,
    bbox: BBox,
    *,
    pad: int = 0,
    inset_ratio: float = 0.06,
) -> GarmentROI:
    """Crop garment ROI.

    ``inset_ratio`` shrinks the detector box slightly so belt/rails that leak
    into a loose YOLO box are less likely to enter defect analysis.
    """
    h, w = frame.shape[:2]
    bw = max(1.0, bbox.x2 - bbox.x1)
    bh = max(1.0, bbox.y2 - bbox.y1)
    ix = bw * inset_ratio
    iy = bh * inset_ratio

    x1 = max(0, int(bbox.x1 + ix) - pad)
    y1 = max(0, int(bbox.y1 + iy) - pad)
    x2 = min(w, int(bbox.x2 - ix) + pad)
    y2 = min(h, int(bbox.y2 - iy) + pad)
    if x2 <= x1 or y2 <= y1:
        # fallback without inset if box too small
        x1 = max(0, int(bbox.x1) - pad)
        y1 = max(0, int(bbox.y1) - pad)
        x2 = min(w, int(bbox.x2) + pad)
        y2 = min(h, int(bbox.y2) + pad)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("invalid garment bbox for ROI crop")
    crop = frame[y1:y2, x1:x2].copy()
    return GarmentROI(
        image=crop,
        garment_bbox=BBox(float(x1), float(y1), float(x2), float(y2)),
        pad=0,
    )


def tighten_roi_to_fabric(
    roi: GarmentROI,
    *,
    max_width_ratio: float = 0.82,
    min_width_height_ratio: float = 4.0,
    pad_ratio: float = 0.04,
    border_ratio: float = 0.12,
) -> GarmentROI:
    """Shrink a loose detector box horizontally to the visible fabric blob.

    Only applied to wide, short ROIs (typical conveyor strip detections). Tall
    expanded garments keep full width so stains lower on the shirt stay reachable.
    """
    image = roi.image
    h, w = image.shape[:2]
    if h < 8 or w < 8:
        return roi
    if (w / max(h, 1)) < min_width_height_ratio:
        return roi

    mask = fabric_mask(image, border_ratio=border_ratio)
    tight = fabric_tight_bbox(mask)
    if tight is None:
        return roi

    fabric_w = tight.x2 - tight.x1
    if fabric_w >= max_width_ratio * w:
        return roi

    pad_x = max(2, int(fabric_w * pad_ratio))
    lx1 = max(0, int(tight.x1) - pad_x)
    lx2 = min(w, int(tight.x2) + pad_x)
    if lx2 <= lx1:
        return roi

    crop = image[:, lx1:lx2].copy()
    full = roi.local_to_full(BBox(float(lx1), 0.0, float(lx2), float(h)))
    return GarmentROI(image=crop, garment_bbox=full, pad=0)
