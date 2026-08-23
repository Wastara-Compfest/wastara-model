from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.config import ROOT, load_config
from src.detection.detector import GarmentDetector
from src.detection.types import FrameResult, VideoResult
from src.evaluation.visualization import format_summary, summarize_video_result
from src.video.reader import VideoReader
from src.video.writer import VideoWriter, draw_detections, save_frame


def run_detection(
    video_path: Path,
    *,
    model_config: str = "configs/model.yaml",
    inference_config: str = "configs/inference.yaml",
    device: str | None = None,
    confidence: float | None = None,
) -> VideoResult:
    cfg = load_config(
        model_config,
        inference_config,
        device=device,
        confidence=confidence,
    )
    inf = cfg.inference
    model_cfg = cfg.model

    video_id = video_path.stem
    detector = GarmentDetector(
        model_cfg.model_path,
        device=model_cfg.device,
        confidence_threshold=inf.confidence_threshold,
        iou_threshold=inf.iou_threshold,
        class_mapping=model_cfg.class_mapping,
        garment_class_ids=model_cfg.garment_class_ids,
        input_size=model_cfg.input_size,
        keep_garment_only=inf.keep_garment_only,
    )
    detector.load()

    with VideoReader(video_path) as reader:
        meta = reader.meta
        writer: VideoWriter | None = None
        if inf.save_annotated_video:
            out_video = inf.annotated_dir / f"{video_id}_detected.mp4"
            writer = VideoWriter(
                out_video,
                fps=meta.fps,
                width=meta.width,
                height=meta.height,
            )

        frames_out = inf.frames_dir / video_id
        result = VideoResult(
            video_id=video_id,
            source_path=str(video_path),
            fps=meta.fps,
            frame_count=meta.frame_count,
            width=meta.width,
            height=meta.height,
        )

        latencies: list[float] = []
        processed = 0

        try:
            for frame_id, timestamp_ms, frame in reader.frames(inf.frame_skip):
                t0 = time.perf_counter()
                detections = detector.predict(frame)
                latencies.append((time.perf_counter() - t0) * 1000.0)
                processed += 1

                frame_result = FrameResult(
                    video_id=video_id,
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    image_width=meta.width,
                    image_height=meta.height,
                    detections=detections,
                )
                result.frames.append(frame_result)

                annotated = draw_detections(
                    frame,
                    detections,
                    box_thickness=inf.box_thickness,
                    font_scale=inf.font_scale,
                )

                if writer is not None:
                    writer.write(annotated)

                should_save = inf.save_all_frames or (
                    inf.save_detection_frames and len(detections) > 0
                )
                if should_save:
                    save_frame(frames_out / f"frame_{frame_id:06d}.jpg", annotated)
        finally:
            if writer is not None:
                writer.close()

        result.processed_frames = processed
        result.latency_ms_avg = sum(latencies) / len(latencies) if latencies else 0.0

    inf.detections_dir.mkdir(parents=True, exist_ok=True)
    json_path = inf.detections_dir / f"{video_id}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)

    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 1: pretrained YOLOv8n clothing detection"
    )
    p.add_argument(
        "--video",
        type=str,
        default=None,
        help="path to input video (default: first file in data/input/videos)",
    )
    p.add_argument("--model-config", default="configs/model.yaml")
    p.add_argument("--inference-config", default="configs/inference.yaml")
    p.add_argument("--device", default=None, help="auto | cuda | mps | cpu")
    p.add_argument("--confidence", type=float, default=None)
    p.add_argument("--list", action="store_true", help="list available input videos")
    return p


def resolve_video(arg: str | None, input_dir: Path) -> Path:
    if arg:
        path = Path(arg)
        if not path.is_absolute():
            path = (ROOT / path) if not path.exists() else path
        if not path.exists():
            raise FileNotFoundError(f"video not found: {path}")
        return path

    videos = sorted(
        [
            *input_dir.glob("*.mp4"),
            *input_dir.glob("*.avi"),
            *input_dir.glob("*.mov"),
            *input_dir.glob("*.mkv"),
        ]
    )
    if not videos:
        raise FileNotFoundError(
            f"no videos in {input_dir}\n"
            "place a test video under data/input/videos/ then re-run"
        )
    return videos[0]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.model_config, args.inference_config)

    if args.list:
        videos = sorted(cfg.inference.input_dir.glob("*"))
        for v in videos:
            if v.is_file():
                print(v.name)
        return 0

    video = resolve_video(args.video, cfg.inference.input_dir)
    print(f"device config: {args.device or cfg.model.device}")
    print(f"model: {cfg.model.model_path}")
    print(f"video: {video}")

    result = run_detection(
        video,
        model_config=args.model_config,
        inference_config=args.inference_config,
        device=args.device,
        confidence=args.confidence,
    )
    summary = summarize_video_result(result)
    print(format_summary(summary))
    print(f"\nJSON → {cfg.inference.detections_dir / (result.video_id + '.json')}")
    if cfg.inference.save_annotated_video:
        print(
            f"video → {cfg.inference.annotated_dir / (result.video_id + '_detected.mp4')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
