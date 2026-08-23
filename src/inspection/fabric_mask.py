"""Fabric segmentation helpers to drop conveyor belt / rails from garment ROI."""

from __future__ import annotations

import cv2
import numpy as np

from src.detection.types import BBox


def fabric_mask(image: np.ndarray, border_ratio: float = 0.12) -> np.ndarray:
    """Keep fabric pixels; drop belt/rails that leak into a loose garment bbox."""
    h, w = image.shape[:2]
    by = max(2, int(h * border_ratio))
    bx = max(2, int(w * border_ratio))

    strips = [
        image[:by, :],
        image[-by:, :],
        image[:, :bx],
        image[:, -bx:],
    ]
    border = np.concatenate([s.reshape(-1, 3) for s in strips], axis=0)
    bg = np.median(border.astype(np.float32), axis=0)

    dist_bg = np.linalg.norm(image.astype(np.float32) - bg, axis=2)
    interior = np.zeros((h, w), dtype=np.uint8)
    interior[by : h - by, bx : w - bx] = 1

    thr = max(
        18.0,
        float(np.percentile(dist_bg[interior == 1], 35)) if interior.any() else 25.0,
    )
    mask = ((dist_bg > thr) & (interior == 1)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        fallback = np.zeros((h, w), dtype=np.uint8)
        fallback[by : h - by, bx : w - bx] = 255
        return fallback

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    return (labels == largest).astype(np.uint8) * 255


def fabric_tight_bbox(
    mask: np.ndarray,
    *,
    min_pixels: int = 64,
) -> BBox | None:
    """Axis-aligned bbox of fabric pixels in ROI-local coordinates."""
    ys, xs = np.where(mask > 0)
    if len(xs) < min_pixels:
        return None
    return BBox(
        float(xs.min()),
        float(ys.min()),
        float(xs.max() + 1),
        float(ys.max() + 1),
    )


def point_on_fabric(mask: np.ndarray, x: float, y: float) -> bool:
    h, w = mask.shape[:2]
    ix = int(round(x))
    iy = int(round(y))
    if ix < 0 or iy < 0 or ix >= w or iy >= h:
        return False
    return bool(mask[iy, ix] > 0)
