from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.detection.preprocessing import prepare_frame
from src.detection.types import Detection
from src.device import resolve_device


class GarmentDetector:
    """Load pretrained YOLO detect weights and run per-frame detection."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        class_mapping: dict[int, str] | None = None,
        garment_class_ids: list[int] | None = None,
        input_size: int = 640,
        keep_garment_only: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = resolve_device(device)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.class_mapping = class_mapping or {}
        self.garment_class_ids = garment_class_ids or list(self.class_mapping.keys())
        self.input_size = input_size
        self.keep_garment_only = keep_garment_only
        self._model: Any = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"model weights not found: {self.model_path}\n"
                "run: uv run python scripts/download_pretrained.py"
            )
        from ultralytics import YOLO

        self._model = YOLO(str(self.model_path))
        names = getattr(self._model, "names", None)
        if isinstance(names, dict) and names:
            for k, v in names.items():
                self.class_mapping.setdefault(int(k), str(v))

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, frame: np.ndarray) -> list[Detection]:
        if self._model is None:
            self.load()

        image = prepare_frame(frame)
        h, w = image.shape[:2]

        results = self._model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            device=self.device,
            verbose=False,
        )

        raw: list[dict] = []
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            class_id = int(clss[i])
            class_name = self.class_mapping.get(class_id, f"class_{class_id}")
            raw.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": float(confs[i]),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

        from src.detection.postprocessing import postprocess_detections

        return postprocess_detections(
            raw,
            class_mapping=self.class_mapping,
            garment_class_ids=self.garment_class_ids,
            confidence_threshold=self.confidence_threshold,
            image_width=w,
            image_height=h,
            keep_garment_only=self.keep_garment_only,
        )
