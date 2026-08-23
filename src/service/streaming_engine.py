from __future__ import annotations

import cv2
import numpy as np

from src.defect.aggregator import AggregationConfig, aggregate_observations
from src.inspection.bbox_refiner import refine_garment_bbox
from src.inspection.frame_sampler import FrameSample
from src.inspection.roi_extractor import extract_roi, tighten_roi_to_fabric
from src.inspection.types import GarmentInspectionResult, SampledObservation
from src.pipeline.garment_inspection_pipeline import GarmentInspectionPipeline
from src.visualization.annotator import draw_evidence, draw_tracks, draw_zone


class StreamAlert:
    __slots__ = (
        "track_id",
        "frame_id",
        "timestamp_ms",
        "anomaly_score",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "evidence_jpg",
    )

    def __init__(
        self,
        *,
        track_id: int,
        frame_id: int,
        timestamp_ms: float,
        anomaly_score: float,
        bbox_x: int,
        bbox_y: int,
        bbox_w: int,
        bbox_h: int,
        evidence_jpg: bytes,
    ) -> None:
        self.track_id = track_id
        self.frame_id = frame_id
        self.timestamp_ms = timestamp_ms
        self.anomaly_score = anomaly_score
        self.bbox_x = bbox_x
        self.bbox_y = bbox_y
        self.bbox_w = bbox_w
        self.bbox_h = bbox_h
        self.evidence_jpg = evidence_jpg


class StreamingInspectionEngine(GarmentInspectionPipeline):
    """Process frames from a queue; emit anomaly alerts without blocking capture."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._zone = None
        self._emit_last_frame: dict[int, int] = {}
        stream_cfg = self.cfg.get("streaming", {})
        self._cooldown_frames = max(6, int(stream_cfg.get("cooldown_frames", 8)))
        self._live_min_confirmations = int(stream_cfg.get("minimum_confirmations", 1))
        stream_conf = float(stream_cfg.get("confidence_threshold", 0.42))
        self._stream_confidence_threshold = stream_conf
        self._track_last_seen: dict[int, int] = {}
        self._track_buffer = int(self.cfg.get("tracking", {}).get("track_buffer", 30))

    def setup_zone(self, width: int, height: int) -> None:
        from src.inspection.inspection_zone import InspectionZone

        self._zone = InspectionZone.from_normalized(
            x1=float(self.zone_norm.get("x1", 0.05)),
            y1=float(self.zone_norm.get("y1", 0.15)),
            x2=float(self.zone_norm.get("x2", 0.95)),
            y2=float(self.zone_norm.get("y2", 0.85)),
            width=width,
            height=height,
        )

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        frame_id: int,
        timestamp_ms: float,
    ) -> tuple[np.ndarray, list[StreamAlert]]:
        if self._zone is None:
            h, w = frame.shape[:2]
            self.setup_zone(w, h)

        zone = self._zone
        assert zone is not None

        if self.use_bytetrack:
            tracks = self.tracker.update(
                frame, frame_id=frame_id, timestamp_ms=timestamp_ms
            )
        else:
            detections = self.detector.predict(frame)
            tracks = self.tracker.update(
                detections,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
            )

        alerts: list[StreamAlert] = []
        seen_ids: set[int] = set()

        for tr in tracks:
            seen_ids.add(tr.track_id)
            self._track_last_seen[tr.track_id] = frame_id
            if not zone.contains(tr.detection.bbox, self.zone_overlap):
                continue
            self.sampler.add(tr, frame.copy())
            inline = self._inline_anomaly(tr, frame, frame_id, timestamp_ms)
            if inline is not None:
                alerts.append(inline)

        alerts.extend(self._finalize_stale_tracks(seen_ids, frame_id))

        vis = draw_zone(frame, zone)
        vis = draw_tracks(vis, tracks)
        return vis, alerts

    def _inline_anomaly(self, tr, frame, frame_id, timestamp_ms) -> StreamAlert | None:
        if not self._can_emit(tr.track_id, frame_id):
            return None

        garment_bbox = refine_garment_bbox(frame, tr.detection.bbox, self.detector)
        roi = tighten_roi_to_fabric(extract_roi(frame, garment_bbox))
        pred = self.analyzer.analyze(roi)
        threshold = getattr(self, "_stream_confidence_threshold", self.agg_cfg.confidence_threshold)
        if not pred.is_defect or pred.confidence < threshold:
            return None

        db = pred.bbox or roi.garment_bbox
        evidence = self._encode_evidence(
            frame, tr.track_id, roi.garment_bbox, pred.confidence, db
        )
        self._emit_last_frame[tr.track_id] = frame_id
        return StreamAlert(
            track_id=tr.track_id,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            anomaly_score=float(pred.confidence),
            bbox_x=int(db.x1),
            bbox_y=int(db.y1),
            bbox_w=max(1, int(db.x2 - db.x1)),
            bbox_h=max(1, int(db.y2 - db.y1)),
            evidence_jpg=evidence,
        )

    def _finalize_stale_tracks(self, seen_ids: set[int], frame_id: int) -> list[StreamAlert]:
        stale = [
            tid
            for tid, last in self._track_last_seen.items()
            if tid not in seen_ids and frame_id - last >= self._track_buffer
        ]
        out: list[StreamAlert] = []
        for track_id in stale:
            out.extend(self._aggregate_track(track_id, frame_id))
            self._track_last_seen.pop(track_id, None)
            self.sampler.clear_track(track_id)
        return out

    def _aggregate_track(
        self, track_id: int, frame_id: int, *, force: bool = False
    ) -> list[StreamAlert]:
        samples = self.sampler.select(track_id)
        if not samples:
            return []

        observations: list[SampledObservation] = []
        analyzed: list[tuple[FrameSample, SampledObservation]] = []
        for sample in samples:
            garment_bbox = refine_garment_bbox(
                sample.frame, sample.bbox, self.detector
            )
            roi = tighten_roi_to_fabric(extract_roi(sample.frame, garment_bbox))
            garment_bbox = roi.garment_bbox
            pred = self.analyzer.analyze(roi)
            obs = SampledObservation(
                frame_id=sample.frame_id,
                timestamp_ms=sample.timestamp_ms,
                garment_bbox=garment_bbox,
                prediction=pred,
            )
            observations.append(obs)
            analyzed.append((sample, obs))

        agg_cfg = AggregationConfig(
            method=self.agg_cfg.method,
            minimum_confirmations=self._live_min_confirmations,
            confidence_threshold=getattr(
                self, "_stream_confidence_threshold", self.agg_cfg.confidence_threshold
            ),
        )
        result = aggregate_observations(
            track_id, "clothing", observations, config=agg_cfg
        )
        if result.status != "DEFECT" or result.defect_bbox is None:
            return []
        if not force and not self._can_emit(track_id, frame_id):
            return []

        defect_samples = [
            (sample, obs)
            for sample, obs in analyzed
            if obs.prediction.is_defect
        ]
        if not defect_samples:
            return []

        best_sample, best = max(
            defect_samples,
            key=lambda pair: pair[1].prediction.confidence,
        )

        db = result.defect_bbox
        evidence = self._encode_evidence(
            best_sample.frame,
            track_id,
            best.garment_bbox,
            result.confidence,
            db,
            result=result,
        )
        self._emit_last_frame[track_id] = frame_id
        return [
            StreamAlert(
                track_id=track_id,
                frame_id=best.frame_id,
                timestamp_ms=best.timestamp_ms,
                anomaly_score=float(result.confidence),
                bbox_x=int(db.x1),
                bbox_y=int(db.y1),
                bbox_w=max(1, int(db.x2 - db.x1)),
                bbox_h=max(1, int(db.y2 - db.y1)),
                evidence_jpg=evidence,
            )
        ]

    def _can_emit(self, track_id: int, frame_id: int) -> bool:
        last = self._emit_last_frame.get(track_id, -10_000)
        return frame_id - last >= self._cooldown_frames

    def _encode_evidence(self, frame, track_id, garment_bbox, confidence, defect_bbox, result=None):
        payload = result or GarmentInspectionResult(
            track_id=track_id,
            garment_type="clothing",
            status="DEFECT",
            defect_type="anomaly",
            confidence=confidence,
            defect_bbox=defect_bbox,
        )
        annotated = draw_evidence(
            frame,
            track_id=track_id,
            garment_bbox=garment_bbox,
            result=payload,
        )
        ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            ok, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes()

    def flush_pending(self, frame_id: int) -> list[StreamAlert]:
        """Emit aggregated alerts for active tracks before shutdown."""
        out: list[StreamAlert] = []
        for track_id in list(self._track_last_seen.keys()):
            out.extend(self._aggregate_track(track_id, frame_id, force=True))
            self.sampler.clear_track(track_id)
        self._track_last_seen.clear()
        return out
