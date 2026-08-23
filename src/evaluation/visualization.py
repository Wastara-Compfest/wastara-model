"""Qualitative / summary visualization helpers for evaluation reports."""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.detection.types import VideoResult


def summarize_video_result(result: VideoResult) -> dict[str, Any]:
    class_counts: Counter[str] = Counter()
    frames_with_det = 0
    total_dets = 0

    for frame in result.frames:
        if frame.detections:
            frames_with_det += 1
        for det in frame.detections:
            class_counts[det.class_name] += 1
            total_dets += 1

    processed = max(1, result.processed_frames)
    fps_eff = 1000.0 / result.latency_ms_avg if result.latency_ms_avg > 0 else 0.0

    return {
        "video_id": result.video_id,
        "processed_frames": result.processed_frames,
        "frames_with_detection": frames_with_det,
        "detection_rate": frames_with_det / processed,
        "total_detections": total_dets,
        "avg_detections_per_frame": total_dets / processed,
        "class_counts": dict(class_counts),
        "latency_ms_avg": result.latency_ms_avg,
        "approx_fps": round(fps_eff, 2),
        "checklist": {
            "garment_detected": total_dets > 0,
            "multiple_classes": len(class_counts) > 1,
            "empty_frames_present": frames_with_det < result.processed_frames,
        },
    }


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"Video: {summary['video_id']}",
        f"  processed_frames: {summary['processed_frames']}",
        f"  frames_with_detection: {summary['frames_with_detection']} "
        f"({summary['detection_rate'] * 100:.1f}%)",
        f"  total_detections: {summary['total_detections']}",
        f"  latency_ms_avg: {summary['latency_ms_avg']:.1f}",
        f"  approx_fps: {summary['approx_fps']}",
        "  class_counts:",
    ]
    for name, count in sorted(summary["class_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"    - {name}: {count}")
    checklist = summary.get("checklist", {})
    lines.append("  qualitative:")
    for key, ok in checklist.items():
        mark = "✓" if ok else "✗"
        lines.append(f"    {mark} {key}")
    return "\n".join(lines)
