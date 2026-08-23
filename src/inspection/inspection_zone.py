"""Configurable inspection zone on the conveyor."""

from __future__ import annotations

from dataclasses import dataclass

from src.detection.types import BBox


@dataclass
class InspectionZone:
    """Axis-aligned zone in absolute pixel coords."""

    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_normalized(
        cls,
        *,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width: int,
        height: int,
    ) -> InspectionZone:
        return cls(
            x1=x1 * width,
            y1=y1 * height,
            x2=x2 * width,
            y2=y2 * height,
        )

    def as_bbox(self) -> BBox:
        return BBox(self.x1, self.y1, self.x2, self.y2)

    def overlap_ratio(self, box: BBox) -> float:
        """Intersection area / garment box area."""
        ix1 = max(self.x1, box.x1)
        iy1 = max(self.y1, box.y1)
        ix2 = min(self.x2, box.x2)
        iy2 = min(self.y2, box.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area = max(1.0, (box.x2 - box.x1) * (box.y2 - box.y1))
        return inter / area

    def contains(self, box: BBox, threshold: float = 0.35) -> bool:
        return self.overlap_ratio(box) >= threshold
