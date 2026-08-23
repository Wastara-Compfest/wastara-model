"""Select representative frames per garment track."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.detection.types import BBox
from src.inspection.types import TrackedDetection


@dataclass
class FrameSample:
    frame_id: int
    timestamp_ms: float
    bbox: BBox
    class_name: str
    confidence: float
    frame: np.ndarray


class FrameSampler:
    """Collect in-zone observations and pick diverse representative frames."""

    def __init__(
        self,
        *,
        sample_count: int = 5,
        min_center_shift_px: float = 12.0,
    ) -> None:
        self.sample_count = max(1, int(sample_count))
        self.min_center_shift_px = float(min_center_shift_px)
        self._by_track: dict[int, list[FrameSample]] = {}

    def add(
        self,
        tracked: TrackedDetection,
        frame: np.ndarray,
    ) -> None:
        det = tracked.detection
        sample = FrameSample(
            frame_id=tracked.frame_id,
            timestamp_ms=tracked.timestamp_ms,
            bbox=det.bbox,
            class_name=det.class_name,
            confidence=det.confidence,
            frame=frame.copy(),
        )
        bucket = self._by_track.setdefault(tracked.track_id, [])
        if bucket:
            prev = bucket[-1]
            if _center_shift(prev.bbox, sample.bbox) < self.min_center_shift_px:
                # keep higher-confidence near-duplicate
                if sample.confidence >= prev.confidence:
                    bucket[-1] = sample
                return
        bucket.append(sample)

    def track_ids(self) -> list[int]:
        return sorted(self._by_track.keys())

    def select(self, track_id: int) -> list[FrameSample]:
        samples = self._by_track.get(track_id, [])
        if not samples:
            return []
        if len(samples) <= self.sample_count:
            return list(samples)
        # uniform temporal subsample across collected observations
        idxs = np.linspace(0, len(samples) - 1, self.sample_count)
        chosen = [samples[int(round(i))] for i in idxs]
        # unique by frame_id preserving order
        seen: set[int] = set()
        out: list[FrameSample] = []
        for s in chosen:
            if s.frame_id in seen:
                continue
            seen.add(s.frame_id)
            out.append(s)
        return out

    def clear_track(self, track_id: int) -> None:
        self._by_track.pop(track_id, None)


def _center_shift(a: BBox, b: BBox) -> float:
    ax = (a.x1 + a.x2) / 2
    ay = (a.y1 + a.y2) / 2
    bx = (b.x1 + b.x2) / 2
    by = (b.y1 + b.y2) / 2
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)
