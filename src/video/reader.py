from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoMeta:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int


class VideoReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"video not found: {self.path}")
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise ValueError(f"cannot open video: {self.path}")

    @property
    def meta(self) -> VideoMeta:
        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return VideoMeta(
            path=self.path,
            fps=fps if fps > 0 else 0.0,
            frame_count=frame_count,
            width=width,
            height=height,
        )

    def frames(self, frame_skip: int = 1):
        """Yield (frame_id, timestamp_ms, frame_bgr)."""
        skip = max(1, int(frame_skip))
        meta = self.meta
        fps = meta.fps if meta.fps > 0 else 30.0
        frame_id = 0

        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            if frame_id % skip == 0:
                timestamp_ms = (frame_id / fps) * 1000.0
                yield frame_id, timestamp_ms, frame
            frame_id += 1

    def read_one(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *args) -> None:
        self.close()
