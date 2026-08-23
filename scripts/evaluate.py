#!/usr/bin/env python3
"""Evaluate detection JSON against optional ground-truth annotations.

Ground-truth format (per video):
{
  "frames": [
    {
      "frame_id": 120,
      "boxes": [{"x1": 10, "y1": 20, "x2": 100, "y2": 200}]
    }
  ]
}

If no GT is provided, prints qualitative summary from detection JSON only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection.types import VideoResult  # noqa: E402
from src.evaluation.metrics import Box, match_detections  # noqa: E402
from src.evaluation.visualization import format_summary, summarize_video_result  # noqa: E402


def load_video_result(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_with_gt(pred: dict, gt: dict, iou_threshold: float) -> dict:
    gt_by_frame = {int(f["frame_id"]): f.get("boxes", []) for f in gt.get("frames", [])}
    total = match_detections([], [], iou_threshold=iou_threshold)

    # accumulate manually
    from src.evaluation.metrics import MatchStats

    acc = MatchStats()
    for frame in pred.get("frames", []):
        fid = int(frame["frame_id"])
        preds = [
            Box(d["bbox"]["x1"], d["bbox"]["y1"], d["bbox"]["x2"], d["bbox"]["y2"])
            for d in frame.get("detections", [])
        ]
        gts = [
            Box(b["x1"], b["y1"], b["x2"], b["y2"])
            for b in gt_by_frame.get(fid, [])
        ]
        stats = match_detections(preds, gts, iou_threshold=iou_threshold)
        acc.true_positives += stats.true_positives
        acc.false_positives += stats.false_positives
        acc.false_negatives += stats.false_negatives
        assert acc.ious is not None and stats.ious is not None
        acc.ious.extend(stats.ious)

    return {
        "precision": round(acc.precision, 4),
        "recall": round(acc.recall, 4),
        "f1": round(acc.f1, 4),
        "mean_iou": round(acc.mean_iou, 4),
        "true_positives": acc.true_positives,
        "false_positives": acc.false_positives,
        "false_negatives": acc.false_negatives,
        "missed_garments": acc.false_negatives,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate garment detection outputs")
    parser.add_argument(
        "--detections",
        required=True,
        help="path to detections JSON (data/output/detections/*.json)",
    )
    parser.add_argument(
        "--gt",
        default=None,
        help="optional ground-truth JSON for precision/recall/IoU",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    pred_path = Path(args.detections)
    if not pred_path.exists():
        # try relative to project
        alt = ROOT / args.detections
        pred_path = alt if alt.exists() else pred_path
    if not pred_path.exists():
        print(f"detections not found: {args.detections}")
        return 1

    pred = load_video_result(pred_path)

    # qualitative summary
    # rebuild minimal VideoResult-like summary without full dataclass roundtrip
    class_counts: dict[str, int] = {}
    frames_with = 0
    total_dets = 0
    for frame in pred.get("frames", []):
        dets = frame.get("detections", [])
        if dets:
            frames_with += 1
        for d in dets:
            class_counts[d["class_name"]] = class_counts.get(d["class_name"], 0) + 1
            total_dets += 1
    processed = max(1, int(pred.get("processed_frames") or len(pred.get("frames", []))))
    latency = float(pred.get("latency_ms_avg") or 0.0)
    summary = {
        "video_id": pred.get("video_id"),
        "processed_frames": processed,
        "frames_with_detection": frames_with,
        "detection_rate": frames_with / processed,
        "total_detections": total_dets,
        "avg_detections_per_frame": total_dets / processed,
        "class_counts": class_counts,
        "latency_ms_avg": latency,
        "approx_fps": round(1000.0 / latency, 2) if latency > 0 else 0.0,
        "checklist": {
            "garment_detected": total_dets > 0,
            "multiple_classes": len(class_counts) > 1,
            "empty_frames_present": frames_with < processed,
        },
    }
    print(format_summary(summary))

    if args.gt:
        gt_path = Path(args.gt)
        if not gt_path.exists():
            gt_path = ROOT / args.gt
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        metrics = evaluate_with_gt(pred, gt, args.iou_threshold)
        print("\nQuantitative (vs ground truth):")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    else:
        print("\nNo --gt provided → qualitative only.")
        print("Add manual frame boxes later for Precision / Recall / IoU.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
