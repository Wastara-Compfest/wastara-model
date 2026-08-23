from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import ROOT
from src.pipeline.garment_inspection_pipeline import (
    GarmentInspectionPipeline,
    load_inspection_config,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Garment-level defect inspection (not frame-independent)"
    )
    parser.add_argument("--video", required=True, help="path to inspection video")
    parser.add_argument("--config", default="configs/inspection.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--tracker", default=None, help="bytetrack | iou")
    parser.add_argument("--defect-mode", default=None, help="anomaly | supervised | none")
    args = parser.parse_args(argv)

    cfg = load_inspection_config(args.config)
    if args.device:
        cfg.setdefault("video", {})["device"] = args.device
    if args.tracker:
        cfg.setdefault("tracking", {})["tracker"] = args.tracker
    if args.defect_mode:
        cfg.setdefault("model", {})["defect_mode"] = args.defect_mode

    video = Path(args.video)
    if not video.exists():
        alt = ROOT / args.video
        video = alt if alt.exists() else video

    pipeline = GarmentInspectionPipeline(cfg)
    result = pipeline.run(video)

    print(json.dumps(result["summary"], indent=2))
    print("latency_ms_avg:", json.dumps(result["latency_ms_avg"], indent=2))
    print(f"defect_mode: {result['defect_mode']}  tracker: {result['tracker']}")
    for g in result["garments"]:
        line = (
            f"GARMENT #{g['track_id']:03d}  Type: {g['garment_type']}  "
            f"Status: {g['status']}"
        )
        if g["status"] == "DEFECT":
            line += f"  Defect: {g['defect_type']}  Confidence: {g['confidence']:.0%}"
        print(line)
    if result.get("result_path"):
        print(f"JSON → {result['result_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
