from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = _resolve(path)
    with resolved.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {resolved}")
    return data


@dataclass
class ModelConfig:
    model_path: Path
    model_type: str
    dataset_name: str
    input_size: int
    device: str
    class_mapping: dict[int, str]
    garment_class_ids: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        mapping_raw = data.get("class_mapping") or {}
        class_mapping = {int(k): str(v) for k, v in mapping_raw.items()}
        return cls(
            model_path=_resolve(data["model_path"]),
            model_type=str(data.get("model_type", "yolov8")),
            dataset_name=str(data.get("dataset_name", "deepfashion2")),
            input_size=int(data.get("input_size", 640)),
            device=str(data.get("device", "auto")),
            class_mapping=class_mapping,
            garment_class_ids=[int(x) for x in data.get("garment_class_ids", [])],
        )


@dataclass
class InferenceConfig:
    confidence_threshold: float
    iou_threshold: float
    frame_skip: int
    keep_garment_only: bool
    input_dir: Path
    output_dir: Path
    save_annotated_video: bool
    save_detection_frames: bool
    save_all_frames: bool
    box_thickness: int
    font_scale: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InferenceConfig:
        return cls(
            confidence_threshold=float(data.get("confidence_threshold", 0.5)),
            iou_threshold=float(data.get("iou_threshold", 0.45)),
            frame_skip=max(1, int(data.get("frame_skip", 1))),
            keep_garment_only=bool(data.get("keep_garment_only", True)),
            input_dir=_resolve(data.get("input_dir", "data/input/videos")),
            output_dir=_resolve(data.get("output_dir", "data/output")),
            save_annotated_video=bool(data.get("save_annotated_video", True)),
            save_detection_frames=bool(data.get("save_detection_frames", True)),
            save_all_frames=bool(data.get("save_all_frames", False)),
            box_thickness=int(data.get("box_thickness", 2)),
            font_scale=float(data.get("font_scale", 0.55)),
        )

    @property
    def annotated_dir(self) -> Path:
        return self.output_dir / "annotated_videos"

    @property
    def frames_dir(self) -> Path:
        return self.output_dir / "frames"

    @property
    def detections_dir(self) -> Path:
        return self.output_dir / "detections"


@dataclass
class AppConfig:
    model: ModelConfig
    inference: InferenceConfig


def load_config(
    model_config: str | Path = "configs/model.yaml",
    inference_config: str | Path = "configs/inference.yaml",
    *,
    device: str | None = None,
    confidence: float | None = None,
    iou: float | None = None,
    model_path: str | None = None,
) -> AppConfig:
    model = ModelConfig.from_dict(load_yaml(model_config))
    inference = InferenceConfig.from_dict(load_yaml(inference_config))

    # env overrides
    env_device = os.getenv("DEVICE")
    env_conf = os.getenv("CONFIDENCE_THRESHOLD")
    env_iou = os.getenv("IOU_THRESHOLD")
    env_model = os.getenv("MODEL_PATH")

    if device or env_device:
        model.device = device or env_device  # type: ignore[assignment]
    if model_path or env_model:
        model.model_path = _resolve(model_path or env_model)  # type: ignore[arg-type]
    if confidence is not None or env_conf:
        inference.confidence_threshold = float(
            confidence if confidence is not None else env_conf  # type: ignore[arg-type]
        )
    if iou is not None or env_iou:
        inference.iou_threshold = float(
            iou if iou is not None else env_iou  # type: ignore[arg-type]
        )

    return AppConfig(model=model, inference=inference)
