from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from src.detection.types import BBox
from src.defect.type_suggester import suggest_defect_type
from src.inspection.fabric_mask import fabric_mask, point_on_fabric
from src.inspection.roi_extractor import GarmentROI
from src.inspection.types import DefectPrediction


class DefectAnalyzer(ABC):
    """Interface for garment-ROI defect analysis (replaceable)."""

    mode: str

    @abstractmethod
    def analyze(self, roi: GarmentROI) -> DefectPrediction:
        raise NotImplementedError


class NullDefectAnalyzer(DefectAnalyzer):
    """Always NORMAL — used when no defect/anomaly model is configured."""

    mode = "none"

    def analyze(self, roi: GarmentROI) -> DefectPrediction:
        return DefectPrediction(
            defect_type=None,
            confidence=1.0,
            bbox=None,
            is_defect=False,
            mode=self.mode,
            score=0.0,
        )


class SupervisedDefectDetector(DefectAnalyzer):
    """Mode A — YOLO (or similar) trained on labeled defect boxes.

    Requires real labeled weights. Will not invent classes.
    """

    mode = "supervised"

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        confidence_threshold: float = 0.5,
        class_names: dict[int, str] | None = None,
        imgsz: int = 640,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"supervised defect weights not found: {self.model_path}\n"
                "Mode A needs labeled defect training data + exported weights."
            )
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.class_names = class_names or {}
        self.imgsz = imgsz
        self._model = None

    def load(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO(str(self.model_path))
        names = getattr(self._model, "names", None)
        if isinstance(names, dict):
            for k, v in names.items():
                self.class_names.setdefault(int(k), str(v))

    def analyze(self, roi: GarmentROI) -> DefectPrediction:
        if self._model is None:
            self.load()
        results = self._model.predict(
            source=roi.image,
            conf=self.confidence_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return DefectPrediction(
                defect_type=None,
                confidence=1.0,
                bbox=None,
                is_defect=False,
                mode=self.mode,
                score=0.0,
            )

        boxes = results[0].boxes
        confs = boxes.conf.cpu().numpy()
        best = int(confs.argmax())
        conf = float(confs[best])
        class_id = int(boxes.cls.cpu().numpy()[best])
        x1, y1, x2, y2 = boxes.xyxy.cpu().numpy()[best].tolist()
        local = BBox(x1, y1, x2, y2)
        full = roi.local_to_full(local)
        name = self.class_names.get(class_id, f"class_{class_id}")
        return DefectPrediction(
            defect_type=name,
            confidence=conf,
            bbox=full,
            is_defect=True,
            mode=self.mode,
            score=conf,
        )


def _fabric_mask(image: np.ndarray, border_ratio: float = 0.12) -> np.ndarray:
    return fabric_mask(image, border_ratio=border_ratio)


class ColorAnomalyDetector(DefectAnalyzer):
    """Mode B baseline — statistical color anomaly **inside fabric mask**.

    Loose garment boxes often include belt/rails. Those background pixels used
    to be flagged as "defects". Analysis is now restricted to fabric area and
    border anomalies are rejected.
    """

    mode = "anomaly"

    def __init__(
        self,
        *,
        score_threshold: float = 0.55,
        min_area_ratio: float = 0.008,
        border_ratio: float = 0.12,
        edge_reject_ratio: float = 0.08,
    ) -> None:
        self.score_threshold = score_threshold
        self.min_area_ratio = min_area_ratio
        self.border_ratio = border_ratio
        self.edge_reject_ratio = edge_reject_ratio

    def analyze(self, roi: GarmentROI) -> DefectPrediction:
        image = roi.image
        if image.size == 0:
            return DefectPrediction(
                defect_type=None,
                confidence=1.0,
                bbox=None,
                is_defect=False,
                mode=self.mode,
                score=0.0,
            )

        h, w = image.shape[:2]
        fabric = _fabric_mask(image, border_ratio=self.border_ratio)
        fabric_pixels = fabric > 0
        if fabric_pixels.sum() < max(64, int(0.05 * h * w)):
            return DefectPrediction(
                defect_type=None,
                confidence=0.9,
                bbox=None,
                is_defect=False,
                mode=self.mode,
                score=0.0,
            )

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        fabric_lab = lab[fabric_pixels]
        med = np.median(fabric_lab, axis=0)
        dist = np.linalg.norm(lab - med, axis=2)

        fabric_dist = dist[fabric_pixels]
        med_d = float(np.median(fabric_dist))
        mad = float(np.median(np.abs(fabric_dist - med_d))) + 1e-6
        score_map = (dist - med_d) / (1.4826 * mad)
        score_map = np.clip(score_map, 0, None)
        score_map = np.where(fabric_pixels, score_map, 0.0)

        fabric_scores = score_map[fabric_pixels]
        thr = max(3.0, float(np.percentile(fabric_scores, 96)))
        mask = ((score_map >= thr) & fabric_pixels).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0.0
        min_area = self.min_area_ratio * float(fabric_pixels.sum())
        margin_x = int(w * self.edge_reject_ratio)
        margin_y = int(h * self.edge_reject_ratio)

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            # reject blobs glued to ROI border (usually belt/rail leakage)
            if x <= margin_x or y <= margin_y or (x + bw) >= (w - margin_x) or (y + bh) >= (h - margin_y):
                continue
            # require most of the blob to sit on fabric
            blob = fabric[y : y + bh, x : x + bw]
            if blob.size == 0 or (blob > 0).mean() < 0.7:
                continue
            region = score_map[y : y + bh, x : x + bw]
            region_score = float(region[blob > 0].mean()) if (blob > 0).any() else 0.0
            if region_score > best_score:
                best_score = region_score
                best = (x, y, x + bw, y + bh)

        anomaly_score = (
            float(min(1.0, best_score / 7.0))
            if best
            else float(min(1.0, float(fabric_scores.mean()) / 7.0))
        )

        if best is None or anomaly_score < self.score_threshold:
            return DefectPrediction(
                defect_type=None,
                confidence=max(0.0, 1.0 - anomaly_score),
                bbox=None,
                is_defect=False,
                mode=self.mode,
                score=anomaly_score,
            )

        x1, y1, x2, y2 = best
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        if not point_on_fabric(fabric, cx, cy):
            return DefectPrediction(
                defect_type=None,
                confidence=max(0.0, 1.0 - anomaly_score),
                bbox=None,
                is_defect=False,
                mode=self.mode,
                score=anomaly_score,
            )

        full = roi.local_to_full(BBox(float(x1), float(y1), float(x2), float(y2)))
        full = _clamp_bbox(full, roi.garment_bbox)
        local = BBox(float(x1), float(y1), float(x2), float(y2))
        suggestion = suggest_defect_type(
            roi,
            local,
            anomaly_score=anomaly_score,
            min_confidence=self.score_threshold,
        )
        return DefectPrediction(
            defect_type="anomaly",
            confidence=anomaly_score,
            bbox=full,
            is_defect=True,
            mode=self.mode,
            score=anomaly_score,
            suggested_defect_type=suggestion.suggested_type,
            suggestion_confidence=suggestion.confidence,
            suggestion_method=suggestion.method if suggestion.suggested_type else None,
        )


def _clamp_bbox(inner: BBox, outer: BBox) -> BBox:
    return BBox(
        x1=max(outer.x1, min(inner.x1, outer.x2)),
        y1=max(outer.y1, min(inner.y1, outer.y2)),
        x2=max(outer.x1, min(inner.x2, outer.x2)),
        y2=max(outer.y1, min(inner.y2, outer.y2)),
    )


def create_defect_analyzer(
    mode: str,
    *,
    defect_model_path: str | None = None,
    device: str = "cpu",
    confidence_threshold: float = 0.5,
    anomaly_score_threshold: float = 0.55,
    anomaly_min_area_ratio: float = 0.008,
) -> DefectAnalyzer:
    key = (mode or "anomaly").lower()
    if key == "supervised":
        if not defect_model_path:
            raise ValueError(
                "defect_mode=supervised requires model.defect_detector path"
            )
        return SupervisedDefectDetector(
            defect_model_path,
            device=device,
            confidence_threshold=confidence_threshold,
        )
    if key in {"anomaly", "mode_b"}:
        return ColorAnomalyDetector(
            score_threshold=anomaly_score_threshold,
            min_area_ratio=anomaly_min_area_ratio,
        )
    if key in {"none", "null"}:
        return NullDefectAnalyzer()
    raise ValueError(f"unsupported defect mode: {mode}")
