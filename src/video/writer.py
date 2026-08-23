from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.detection.types import Detection


class VideoWriter:
    def __init__(
        self,
        path: str | Path,
        *,
        fps: float,
        width: int,
        height: int,
        fourcc: str = "mp4v",
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fps = fps if fps and fps > 0 else 30.0
        writer = cv2.VideoWriter(
            str(self.path),
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot open video writer: {self.path}")
        self._writer = writer

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *args) -> None:
        self.close()


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    *,
    box_thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (
            int(det.bbox.x1),
            int(det.bbox.y1),
            int(det.bbox.x2),
            int(det.bbox.y2),
        )
        color = (60, 110, 180)  # BGR warm brown-ish
        cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thickness)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        ty = max(0, y1 - 6)
        cv2.rectangle(
            out,
            (x1, ty - th - baseline - 4),
            (x1 + tw + 4, ty + 2),
            color,
            -1,
        )
        cv2.putText(
            out,
            label,
            (x1 + 2, ty - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def save_frame(path: str | Path, frame: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)
