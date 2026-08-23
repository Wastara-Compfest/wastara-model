from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from src.detection.types import BBox
from src.inspection.types import DefectPrediction, GarmentInspectionResult, SampledObservation


@dataclass
class AggregationConfig:
    method: str = "temporal_consensus"
    minimum_confirmations: int = 2
    confidence_threshold: float = 0.55


def aggregate_observations(
    track_id: int,
    garment_type: str,
    observations: list[SampledObservation],
    *,
    config: AggregationConfig,
    latency_ms: dict[str, float] | None = None,
) -> GarmentInspectionResult:
    if not observations:
        return GarmentInspectionResult(
            track_id=track_id,
            garment_type=garment_type,
            status="UNKNOWN",
            defect_type=None,
            confidence=0.0,
            inspection_frames=[],
            latency_ms=latency_ms or {},
        )

    method = (config.method or "temporal_consensus").lower()
    defect_obs = [
        o
        for o in observations
        if o.prediction.is_defect
        and o.prediction.confidence >= config.confidence_threshold
    ]

    if method == "majority":
        return _majority(track_id, garment_type, observations, defect_obs, config, latency_ms)
    if method == "max_confidence":
        return _max_confidence(
            track_id, garment_type, observations, defect_obs, config, latency_ms
        )
    return _temporal_consensus(
        track_id, garment_type, observations, defect_obs, config, latency_ms
    )


def _frames(observations: list[SampledObservation]) -> list[int]:
    return [o.frame_id for o in observations]


def _evidence(observations: list[SampledObservation]) -> list[str]:
    return [o.evidence_path for o in observations if o.evidence_path]


def _normal_result(
    track_id: int,
    garment_type: str,
    observations: list[SampledObservation],
    confidence: float,
    latency_ms: dict[str, float] | None,
) -> GarmentInspectionResult:
    return GarmentInspectionResult(
        track_id=track_id,
        garment_type=garment_type,
        status="NORMAL",
        defect_type=None,
        confidence=confidence,
        inspection_frames=_frames(observations),
        evidence_paths=_evidence(observations),
        observations=observations,
        latency_ms=latency_ms or {},
    )


def _defect_result(
    track_id: int,
    garment_type: str,
    observations: list[SampledObservation],
    defect_type: str,
    confidence: float,
    bbox: BBox | None,
    latency_ms: dict[str, float] | None,
    suggested_defect_type: str | None = None,
    suggestion_confidence: float = 0.0,
    suggestion_method: str | None = None,
) -> GarmentInspectionResult:
    return GarmentInspectionResult(
        track_id=track_id,
        garment_type=garment_type,
        status="DEFECT",
        defect_type=defect_type,
        confidence=confidence,
        inspection_frames=_frames(observations),
        defect_bbox=bbox,
        suggested_defect_type=suggested_defect_type,
        suggestion_confidence=suggestion_confidence,
        suggestion_method=suggestion_method,
        evidence_paths=_evidence(observations),
        observations=observations,
        latency_ms=latency_ms or {},
    )


def _majority(
    track_id,
    garment_type,
    observations,
    defect_obs,
    config: AggregationConfig,
    latency_ms,
):
    if len(defect_obs) < config.minimum_confirmations:
        normal_conf = float(
            np_mean([o.prediction.confidence for o in observations if not o.prediction.is_defect])
            if any(not o.prediction.is_defect for o in observations)
            else 0.7
        )
        return _normal_result(track_id, garment_type, observations, normal_conf, latency_ms)

    types = Counter(o.prediction.defect_type or "anomaly" for o in defect_obs)
    defect_type, _ = types.most_common(1)[0]
    chosen = [o for o in defect_obs if (o.prediction.defect_type or "anomaly") == defect_type]
    conf = max(o.prediction.confidence for o in chosen)
    bbox = _best_bbox(chosen)
    suggested, sugg_conf, sugg_method = _best_suggestion(chosen)
    return _defect_result(
        track_id,
        garment_type,
        observations,
        defect_type,
        conf,
        bbox,
        latency_ms,
        suggested,
        sugg_conf,
        sugg_method,
    )


def _max_confidence(
    track_id,
    garment_type,
    observations,
    defect_obs,
    config: AggregationConfig,
    latency_ms,
):
    if not defect_obs:
        return _normal_result(track_id, garment_type, observations, 0.8, latency_ms)
    best = max(defect_obs, key=lambda o: o.prediction.confidence)
    if (
        best.prediction.confidence < config.confidence_threshold
        or len(defect_obs) < config.minimum_confirmations
    ):
        return _normal_result(track_id, garment_type, observations, 0.75, latency_ms)
    return _defect_result(
        track_id,
        garment_type,
        observations,
        best.prediction.defect_type or "anomaly",
        best.prediction.confidence,
        best.prediction.bbox,
        latency_ms,
        best.prediction.suggested_defect_type,
        best.prediction.suggestion_confidence,
        best.prediction.suggestion_method,
    )


def _temporal_consensus(
    track_id,
    garment_type,
    observations,
    defect_obs,
    config: AggregationConfig,
    latency_ms,
):
    """Require multiple confirmations; boost when bboxes stay spatially close."""
    if len(defect_obs) < config.minimum_confirmations:
        return _normal_result(track_id, garment_type, observations, 0.8, latency_ms)

    # group by type
    by_type: dict[str, list[SampledObservation]] = {}
    for o in defect_obs:
        key = o.prediction.defect_type or "anomaly"
        by_type.setdefault(key, []).append(o)

    best_type = None
    best_group: list[SampledObservation] = []
    best_score = -1.0
    for dtype, group in by_type.items():
        if len(group) < config.minimum_confirmations:
            continue
        confs = [o.prediction.confidence for o in group]
        spatial = _spatial_consistency(group)
        score = (sum(confs) / len(confs)) * (0.7 + 0.3 * spatial)
        if score > best_score:
            best_score = score
            best_type = dtype
            best_group = group

    if best_type is None:
        return _normal_result(track_id, garment_type, observations, 0.75, latency_ms)

    conf = max(o.prediction.confidence for o in best_group)
    # slight boost for consensus but cap at 1
    conf = min(1.0, conf * (0.9 + 0.1 * min(1.0, len(best_group) / 3)))
    suggested, sugg_conf, sugg_method = _best_suggestion(best_group)
    return _defect_result(
        track_id,
        garment_type,
        observations,
        best_type,
        conf,
        _best_bbox(best_group),
        latency_ms,
        suggested,
        sugg_conf,
        sugg_method,
    )


def _best_suggestion(
    group: list[SampledObservation],
) -> tuple[str | None, float, str | None]:
    with_suggestion = [
        o
        for o in group
        if o.prediction.suggested_defect_type and o.prediction.suggestion_confidence > 0
    ]
    if not with_suggestion:
        return None, 0.0, None
    best = max(with_suggestion, key=lambda o: o.prediction.suggestion_confidence)
    return (
        best.prediction.suggested_defect_type,
        best.prediction.suggestion_confidence,
        best.prediction.suggestion_method,
    )


def _best_bbox(group: list[SampledObservation]) -> BBox | None:
    with_box = [o for o in group if o.prediction.bbox is not None]
    if not with_box:
        return None
    best = max(with_box, key=lambda o: o.prediction.confidence)
    return best.prediction.bbox


def _spatial_consistency(group: list[SampledObservation]) -> float:
    boxes = [o.prediction.bbox for o in group if o.prediction.bbox is not None]
    if len(boxes) < 2:
        return 0.5
    # average IoU to mean center box
    cx1 = sum(b.x1 for b in boxes) / len(boxes)
    cy1 = sum(b.y1 for b in boxes) / len(boxes)
    cx2 = sum(b.x2 for b in boxes) / len(boxes)
    cy2 = sum(b.y2 for b in boxes) / len(boxes)
    mean = BBox(cx1, cy1, cx2, cy2)

    def iou(a: BBox, b: BBox) -> float:
        ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
        ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        aa = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
        bb = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
        return inter / (aa + bb - inter)

    return sum(iou(b, mean) for b in boxes) / len(boxes)


def np_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
