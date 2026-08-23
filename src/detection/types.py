from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def as_int(self) -> dict[str, int]:
        return {
            "x1": int(round(self.x1)),
            "y1": int(round(self.y1)),
            "x2": int(round(self.x2)),
            "y2": int(round(self.y2)),
        }


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox
    is_garment: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": self.bbox.as_int(),
            "is_garment": self.is_garment,
        }


@dataclass
class FrameResult:
    video_id: str
    frame_id: int
    timestamp_ms: float
    image_width: int
    image_height: int
    detections: list[Detection] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "frame_id": self.frame_id,
            "timestamp_ms": round(float(self.timestamp_ms), 2),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "detections": [d.to_dict() for d in self.detections],
        }


@dataclass
class VideoResult:
    video_id: str
    source_path: str
    fps: float
    frame_count: int
    width: int
    height: int
    frames: list[FrameResult] = field(default_factory=list)
    latency_ms_avg: float = 0.0
    processed_frames: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "source_path": self.source_path,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "processed_frames": self.processed_frames,
            "latency_ms_avg": round(self.latency_ms_avg, 2),
            "frames": [f.to_dict() for f in self.frames],
        }


def video_result_to_jsonable(result: VideoResult) -> dict[str, Any]:
    return result.to_dict()


def asdict_safe(obj: Any) -> dict[str, Any]:
    return asdict(obj)
