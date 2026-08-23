from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml

from src.config import ROOT, load_config
from src.defect.aggregator import AggregationConfig, aggregate_observations
from src.defect.defect_detector import create_defect_analyzer
from src.detection.detector import GarmentDetector
from src.device import resolve_device
from src.inspection.bbox_refiner import refine_garment_bbox
from src.inspection.frame_sampler import FrameSampler
from src.inspection.inspection_zone import InspectionZone
from src.inspection.roi_extractor import extract_roi, tighten_roi_to_fabric
from src.inspection.types import GarmentInspectionResult, SampledObservation
from src.tracking.tracker import ByteTrackTracker, IoUTracker, create_tracker
from src.video.reader import VideoReader
from src.video.writer import VideoWriter
from src.visualization.annotator import (
    draw_evidence,
    draw_tracks,
    draw_zone,
    save_evidence_image,
)


def load_inspection_config(path: str | Path = "configs/inspection.yaml") -> dict[str, Any]:
    resolved = Path(path) if Path(path).is_absolute() else ROOT / path
    with resolved.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"invalid inspection config: {resolved}")
    return data


class GarmentInspectionPipeline:
    """Unit of analysis = one garment track, not one frame."""

    def __init__(self, inspection_config: dict[str, Any] | None = None) -> None:
        self.cfg = inspection_config or load_inspection_config()
        model_cfg_path = self.cfg.get("model", {}).get(
            "garment_config", "configs/model.yaml"
        )
        self.app = load_config(model_cfg_path, "configs/inference.yaml")
        device_req = self.cfg.get("video", {}).get("device") or self.app.model.device
        self.device = resolve_device(device_req)

        self.detector = GarmentDetector(
            self.app.model.model_path,
            device=self.device,
            confidence_threshold=self.app.inference.confidence_threshold,
            iou_threshold=self.app.inference.iou_threshold,
            class_mapping=self.app.model.class_mapping,
            garment_class_ids=self.app.model.garment_class_ids,
            input_size=self.app.model.input_size,
            keep_garment_only=self.app.inference.keep_garment_only,
        )
        self.detector.load()

        track_cfg = self.cfg.get("tracking", {})
        tracker_name = track_cfg.get("tracker", "bytetrack")
        self.tracker = create_tracker(
            tracker_name,
            yolo_model=self.detector._model,
            match_iou=float(track_cfg.get("match_iou", 0.3)),
            track_buffer=int(track_cfg.get("track_buffer", 30)),
            min_hits=int(track_cfg.get("min_hits", 2)),
            track_threshold=float(track_cfg.get("track_threshold", 0.4)),
            conf=self.app.inference.confidence_threshold,
            iou=self.app.inference.iou_threshold,
            device=self.device,
            imgsz=self.app.model.input_size,
            class_mapping=self.app.model.class_mapping,
        )
        self.use_bytetrack = isinstance(self.tracker, ByteTrackTracker)

        insp = self.cfg.get("inspection", {})
        self.zone_norm = insp.get("zone", {})
        self.zone_overlap = float(insp.get("zone_overlap_threshold", 0.35))
        self.sampler = FrameSampler(
            sample_count=int(insp.get("sample_count", 5)),
            min_center_shift_px=float(insp.get("min_center_shift_px", 12)),
        )

        defect_cfg = self.cfg.get("defect", {})
        mode = self.cfg.get("model", {}).get("defect_mode", "anomaly")
        defect_path = self.cfg.get("model", {}).get("defect_detector") or None
        if defect_path == "":
            defect_path = None
        self.analyzer = create_defect_analyzer(
            mode,
            defect_model_path=defect_path,
            device=self.device,
            confidence_threshold=float(defect_cfg.get("confidence_threshold", 0.55)),
            anomaly_score_threshold=float(
                defect_cfg.get("anomaly_score_threshold", 0.45)
            ),
            anomaly_min_area_ratio=float(
                defect_cfg.get("anomaly_min_area_ratio", 0.004)
            ),
        )
        self.agg_cfg = AggregationConfig(
            method=str(defect_cfg.get("aggregation_method", "temporal_consensus")),
            minimum_confirmations=int(defect_cfg.get("minimum_confirmations", 2)),
            confidence_threshold=float(defect_cfg.get("confidence_threshold", 0.55)),
        )

        out_cfg = self.cfg.get("output", {})
        self.save_evidence = bool(out_cfg.get("save_evidence", True))
        self.save_annotated_video = bool(out_cfg.get("save_annotated_video", True))
        self.save_json = bool(out_cfg.get("save_json", True))
        self.output_dir = ROOT / out_cfg.get("output_dir", "data/output/inspection")
        self.frame_skip = max(1, int(self.cfg.get("video", {}).get("frame_skip", 1)))

        self.latency = {
            "garment_detection_ms": [],
            "tracking_ms": [],
            "defect_detection_ms": [],
        }

    def run(self, video_path: str | Path) -> dict[str, Any]:
        video_path = Path(video_path)
        if not video_path.is_absolute() and not video_path.exists():
            video_path = ROOT / video_path
        video_id = video_path.stem

        evidence_dir = self.output_dir / "evidence"
        json_dir = self.output_dir / "results"
        annotated_dir = self.output_dir / "annotated_videos"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)
        annotated_dir.mkdir(parents=True, exist_ok=True)

        writer: VideoWriter | None = None
        garment_types: dict[int, str] = {}

        with VideoReader(video_path) as reader:
            meta = reader.meta
            zone = InspectionZone.from_normalized(
                x1=float(self.zone_norm.get("x1", 0.05)),
                y1=float(self.zone_norm.get("y1", 0.15)),
                x2=float(self.zone_norm.get("x2", 0.95)),
                y2=float(self.zone_norm.get("y2", 0.85)),
                width=meta.width,
                height=meta.height,
            )

            if self.save_annotated_video:
                writer = VideoWriter(
                    annotated_dir / f"{video_id}_inspection.mp4",
                    fps=meta.fps,
                    width=meta.width,
                    height=meta.height,
                )

            try:
                for frame_id, timestamp_ms, frame in reader.frames(self.frame_skip):
                    # --- detect + track ---
                    if self.use_bytetrack:
                        t0 = time.perf_counter()
                        tracks = self.tracker.update(
                            frame, frame_id=frame_id, timestamp_ms=timestamp_ms
                        )
                        # ByteTrack includes detection; attribute combined cost mostly to tracking+detect
                        dt = (time.perf_counter() - t0) * 1000.0
                        self.latency["garment_detection_ms"].append(dt * 0.7)
                        self.latency["tracking_ms"].append(dt * 0.3)
                    else:
                        t0 = time.perf_counter()
                        detections = self.detector.predict(frame)
                        self.latency["garment_detection_ms"].append(
                            (time.perf_counter() - t0) * 1000.0
                        )
                        t1 = time.perf_counter()
                        tracks = self.tracker.update(
                            detections,
                            frame_id=frame_id,
                            timestamp_ms=timestamp_ms,
                        )
                        self.latency["tracking_ms"].append(
                            (time.perf_counter() - t1) * 1000.0
                        )

                    for tr in tracks:
                        garment_types[tr.track_id] = tr.detection.class_name
                        if zone.contains(tr.detection.bbox, self.zone_overlap):
                            self.sampler.add(tr, frame)

                    if writer is not None:
                        vis = draw_zone(frame, zone)
                        vis = draw_tracks(vis, tracks)
                        writer.write(vis)
            finally:
                if writer is not None:
                    writer.close()

        # --- per-track defect inspection on sampled frames ---
        results: list[GarmentInspectionResult] = []
        for track_id in self.sampler.track_ids():
            samples = self.sampler.select(track_id)
            if not samples:
                continue
            observations: list[SampledObservation] = []
            defect_latencies: list[float] = []

            for sample in samples:
                garment_bbox = refine_garment_bbox(
                    sample.frame,
                    sample.bbox,
                    self.detector,
                )
                roi = extract_roi(sample.frame, garment_bbox)
                roi = tighten_roi_to_fabric(roi)
                garment_bbox = roi.garment_bbox
                t0 = time.perf_counter()
                pred = self.analyzer.analyze(roi)
                defect_latencies.append((time.perf_counter() - t0) * 1000.0)

                evidence_path = None
                if self.save_evidence and pred.is_defect:
                    # provisional evidence; final status decided after aggregation
                    tmp = evidence_dir / f"track_{track_id:04d}_frame_{sample.frame_id:06d}.jpg"
                    # draw later after aggregation with final status; save raw ROI annotated lightly
                    annotated = draw_evidence(
                        sample.frame,
                        track_id=track_id,
                        garment_bbox=garment_bbox,
                        result=GarmentInspectionResult(
                            track_id=track_id,
                            garment_type=garment_types.get(track_id, "clothing"),
                            status="DEFECT" if pred.is_defect else "NORMAL",
                            defect_type=pred.defect_type,
                            confidence=pred.confidence,
                            defect_bbox=pred.bbox,
                        ),
                    )
                    save_evidence_image(tmp, annotated)
                    evidence_path = str(tmp.relative_to(ROOT))

                observations.append(
                    SampledObservation(
                        frame_id=sample.frame_id,
                        timestamp_ms=sample.timestamp_ms,
                        garment_bbox=garment_bbox,
                        prediction=pred,
                        evidence_path=evidence_path,
                    )
                )

            if defect_latencies:
                self.latency["defect_detection_ms"].extend(defect_latencies)

            avg_lat = {
                "garment_detection": _avg(self.latency["garment_detection_ms"]),
                "tracking": _avg(self.latency["tracking_ms"]),
                "defect_detection": _avg(defect_latencies),
            }
            result = aggregate_observations(
                track_id,
                garment_types.get(track_id, "clothing"),
                observations,
                config=self.agg_cfg,
                latency_ms=avg_lat,
            )

            # rewrite evidence for final DEFECT decisions with consistent naming
            if self.save_evidence and result.status == "DEFECT":
                best_obs = None
                for o in observations:
                    if o.prediction.is_defect:
                        if best_obs is None or o.prediction.confidence > best_obs.prediction.confidence:
                            best_obs = o
                if best_obs is not None:
                    dtype = (result.defect_type or "anomaly").replace(" ", "_")
                    final_path = (
                        evidence_dir
                        / f"track_{track_id:04d}_{dtype}_frame_{best_obs.frame_id:06d}.jpg"
                    )
                    sample_frame = next(
                        s.frame for s in samples if s.frame_id == best_obs.frame_id
                    )
                    annotated = draw_evidence(
                        sample_frame,
                        track_id=track_id,
                        garment_bbox=best_obs.garment_bbox,
                        result=result,
                    )
                    save_evidence_image(final_path, annotated)
                    rel = str(final_path.relative_to(ROOT))
                    if rel not in result.evidence_paths:
                        result.evidence_paths.insert(0, rel)

            results.append(result)

        payload = {
            "video_id": video_id,
            "source_path": str(video_path),
            "defect_mode": getattr(self.analyzer, "mode", "unknown"),
            "tracker": self.cfg.get("tracking", {}).get("tracker", "iou"),
            "garments": [r.to_dict() for r in results],
            "latency_ms_avg": {
                "garment_detection": _avg(self.latency["garment_detection_ms"]),
                "tracking": _avg(self.latency["tracking_ms"]),
                "defect_detection": _avg(self.latency["defect_detection_ms"]),
            },
            "summary": {
                "garments_inspected": len(results),
                "defects": sum(1 for r in results if r.status == "DEFECT"),
                "normal": sum(1 for r in results if r.status == "NORMAL"),
            },
        }

        if self.save_json:
            out_json = json_dir / f"{video_id}_garments.json"
            out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            payload["result_path"] = str(out_json.relative_to(ROOT))

        return payload


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
