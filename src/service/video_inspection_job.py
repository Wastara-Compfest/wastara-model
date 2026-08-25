from __future__ import annotations

import logging
import threading
from pathlib import Path

from src.config import ROOT
from src.pipeline.garment_inspection_pipeline import GarmentInspectionPipeline
from src.service.backend_client import AnomalyEvent, BackendClient

logger = logging.getLogger(__name__)

# Video jobs share one pipeline instance (avoids reloading YOLO weights per
# upload) and run one at a time — fine at hackathon scale, concurrent uploads
# simply queue behind the lock.
_pipeline_lock = threading.Lock()
_pipeline: GarmentInspectionPipeline | None = None


def _get_pipeline() -> GarmentInspectionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = GarmentInspectionPipeline()
    return _pipeline


def run_video_inspection_job(
    *,
    video_path: Path,
    machine_id: str,
    video_inspection_id: str,
    backend: BackendClient,
) -> None:
    defect_count = 0
    try:
        with _pipeline_lock:
            pipeline = _get_pipeline()
            result = pipeline.run(video_path)

        for garment in result["garments"]:
            if garment["status"] != "DEFECT" or not garment["defect_bbox"]:
                continue

            evidence_paths = garment["evidence_paths"]
            if not evidence_paths:
                continue
            evidence_bytes = (ROOT / evidence_paths[0]).read_bytes()

            db = garment["defect_bbox"]
            frames = garment["inspection_frames"] or [0]
            event = AnomalyEvent(
                machine_id=machine_id,
                anomaly_score=garment["confidence"],
                bbox_x=int(db["x1"]),
                bbox_y=int(db["y1"]),
                bbox_w=max(1, int(db["x2"] - db["x1"])),
                bbox_h=max(1, int(db["y2"] - db["y1"])),
                frame_start=min(frames),
                frame_end=max(frames),
                video_inspection_id=video_inspection_id,
            )
            posted_id = backend.post_anomaly(event, evidence_bytes)
            if posted_id:
                defect_count += 1

        backend.post_inspection_complete(
            video_inspection_id, status="done", defect_count=defect_count
        )
    except Exception as exc:
        logger.exception("video inspection job failed: %s", video_inspection_id)
        backend.post_inspection_complete(
            video_inspection_id, status="failed", error_message=str(exc)
        )
    finally:
        backend.close()
