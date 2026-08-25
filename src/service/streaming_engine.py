from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.inspection.bbox_refiner import refine_garment_bbox
from src.inspection.roi_extractor import extract_roi, tighten_roi_to_fabric
from src.inspection.types import GarmentInspectionResult
from src.pipeline.garment_inspection_pipeline import GarmentInspectionPipeline
from src.visualization.annotator import draw_evidence, draw_tracks, draw_zone


class StreamAlert:
    __slots__ = (
        "track_id",
        "frame_id",
        "frame_start",
        "frame_end",
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
        frame_start: int,
        frame_end: int,
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
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.timestamp_ms = timestamp_ms
        self.anomaly_score = anomaly_score
        self.bbox_x = bbox_x
        self.bbox_y = bbox_y
        self.bbox_w = bbox_w
        self.bbox_h = bbox_h
        self.evidence_jpg = evidence_jpg


@dataclass
class TrackAnomalyState:
    first_frame: int | None = None
    last_frame: int | None = None
    consecutive: int = 0
    confirmed: bool = False
    max_score: float = 0.0
    best_frame: int = 0
    best_timestamp_ms: float = 0.0
    bbox_x: int = 0
    bbox_y: int = 0
    bbox_w: int = 1
    bbox_h: int = 1
    evidence_jpg: bytes = b""

    def reset_candidate(self) -> None:
        if self.confirmed:
            self.consecutive = 0
            return
        self.first_frame = None
        self.last_frame = None
        self.consecutive = 0
        self.max_score = 0.0
        self.evidence_jpg = b""


class StreamingInspectionEngine(GarmentInspectionPipeline):
    """Process frames from a queue; emit anomaly alerts without blocking capture."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._zone = None
        stream_cfg = self.cfg.get("streaming", {})
        self._live_min_confirmations = max(
            3, int(stream_cfg.get("minimum_confirmations", 3))
        )
        stream_conf = float(stream_cfg.get("confidence_threshold", 0.42))
        self._stream_confidence_threshold = stream_conf
        self._track_last_seen: dict[int, int] = {}
        self._anomaly_states: dict[int, TrackAnomalyState] = {}
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
            inline = self._observe_anomaly(tr, frame, frame_id, timestamp_ms)
            if inline is not None:
                alerts.append(inline)

        alerts.extend(self._finalize_stale_tracks(seen_ids, frame_id))

        vis = draw_zone(frame, zone)
        vis = draw_tracks(vis, tracks)
        return vis, alerts

    def _observe_anomaly(self, tr, frame, frame_id, timestamp_ms) -> StreamAlert | None:
        garment_bbox = refine_garment_bbox(frame, tr.detection.bbox, self.detector)
        roi = tighten_roi_to_fabric(extract_roi(frame, garment_bbox))
        pred = self.analyzer.analyze(roi)
        threshold = getattr(
            self, "_stream_confidence_threshold", self.agg_cfg.confidence_threshold
        )
        state = self._anomaly_states.setdefault(tr.track_id, TrackAnomalyState())
        if not pred.is_defect or pred.confidence < threshold:
            state.reset_candidate()
            return None

        db = pred.bbox or roi.garment_bbox
        if state.first_frame is None:
            state.first_frame = frame_id
        state.last_frame = frame_id
        state.consecutive += 1

        if not state.evidence_jpg or pred.confidence >= state.max_score:
            state.max_score = float(pred.confidence)
            state.best_frame = frame_id
            state.best_timestamp_ms = timestamp_ms
            state.bbox_x = int(db.x1)
            state.bbox_y = int(db.y1)
            state.bbox_w = max(1, int(db.x2 - db.x1))
            state.bbox_h = max(1, int(db.y2 - db.y1))
            state.evidence_jpg = self._encode_evidence(
                frame, tr.track_id, roi.garment_bbox, pred.confidence, db
            )

        if state.confirmed or state.consecutive < self._live_min_confirmations:
            return None

        state.confirmed = True
        return self._alert_from_state(tr.track_id, state)

    def _finalize_stale_tracks(
        self, seen_ids: set[int], frame_id: int
    ) -> list[StreamAlert]:
        stale = [
            tid
            for tid, last in self._track_last_seen.items()
            if tid not in seen_ids and frame_id - last >= self._track_buffer
        ]
        out: list[StreamAlert] = []
        for track_id in stale:
            alert = self._finalize_track(track_id)
            if alert is not None:
                out.append(alert)
            self._track_last_seen.pop(track_id, None)
            self.sampler.clear_track(track_id)
        return out

    def _alert_from_state(self, track_id: int, state: TrackAnomalyState) -> StreamAlert:
        assert state.first_frame is not None
        assert state.last_frame is not None
        return StreamAlert(
            track_id=track_id,
            frame_id=state.best_frame,
            frame_start=state.first_frame,
            frame_end=state.last_frame,
            timestamp_ms=state.best_timestamp_ms,
            anomaly_score=state.max_score,
            bbox_x=state.bbox_x,
            bbox_y=state.bbox_y,
            bbox_w=state.bbox_w,
            bbox_h=state.bbox_h,
            evidence_jpg=state.evidence_jpg,
        )

    def _finalize_track(self, track_id: int) -> StreamAlert | None:
        state = self._anomaly_states.pop(track_id, None)
        if state is None or not state.confirmed:
            return None
        return self._alert_from_state(track_id, state)

    def _encode_evidence(
        self, frame, track_id, garment_bbox, confidence, defect_bbox, result=None
    ):
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
            alert = self._finalize_track(track_id)
            if alert is not None:
                out.append(alert)
            self.sampler.clear_track(track_id)
        self._track_last_seen.clear()
        return out
