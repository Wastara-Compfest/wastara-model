"""Inspection / defect pipeline data types (garment-level unit of analysis)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from src.detection.types import BBox, Detection

GarmentStatus = Literal["NORMAL", "DEFECT", "UNKNOWN"]


@dataclass
class TrackedDetection:
    track_id: int
    detection: Detection
    frame_id: int
    timestamp_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "frame_id": self.frame_id,
            "timestamp_ms": round(self.timestamp_ms, 2),
            "detection": self.detection.to_dict(),
        }


@dataclass
class DefectPrediction:
    """Single-frame defect/anomaly prediction on a garment ROI."""

    defect_type: str | None  # generic model label e.g. "anomaly"
    confidence: float
    bbox: BBox | None  # in full-frame coords when available
    is_defect: bool
    mode: str  # supervised | anomaly
    score: float = 0.0  # raw anomaly score when mode=anomaly
    suggested_defect_type: str | None = None
    suggestion_confidence: float = 0.0
    suggestion_method: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "defect_type": self.defect_type,
            "confidence": round(float(self.confidence), 4),
            "bbox": self.bbox.as_int() if self.bbox else None,
            "is_defect": self.is_defect,
            "mode": self.mode,
            "score": round(float(self.score), 4),
            "suggested_defect_type": self.suggested_defect_type,
            "suggestion_confidence": round(float(self.suggestion_confidence), 4),
            "suggestion_method": self.suggestion_method,
        }


@dataclass
class SampledObservation:
    frame_id: int
    timestamp_ms: float
    garment_bbox: BBox
    prediction: DefectPrediction
    evidence_path: str | None = None


@dataclass
class GarmentInspectionResult:
    track_id: int
    garment_type: str
    status: GarmentStatus
    defect_type: str | None
    confidence: float
    inspection_frames: list[int] = field(default_factory=list)
    defect_bbox: BBox | None = None
    suggested_defect_type: str | None = None
    suggestion_confidence: float = 0.0
    suggestion_method: str | None = None
    evidence_paths: list[str] = field(default_factory=list)
    observations: list[SampledObservation] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "garment_type": self.garment_type,
            "status": self.status,
            "defect_type": self.defect_type,
            "confidence": round(float(self.confidence), 4),
            "inspection_frames": self.inspection_frames,
            "defect_bbox": self.defect_bbox.as_int() if self.defect_bbox else None,
            "suggested_defect_type": self.suggested_defect_type,
            "suggestion_confidence": round(float(self.suggestion_confidence), 4),
            "suggestion_method": self.suggestion_method,
            "evidence_paths": self.evidence_paths,
            "observations": [
                {
                    "frame_id": o.frame_id,
                    "timestamp_ms": round(o.timestamp_ms, 2),
                    "garment_bbox": o.garment_bbox.as_int(),
                    "prediction": o.prediction.to_dict(),
                    "evidence_path": o.evidence_path,
                }
                for o in self.observations
            ],
            "latency_ms": {k: round(v, 2) for k, v in self.latency_ms.items()},
        }
