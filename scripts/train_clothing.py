#!/usr/bin/env python3
"""Fine-tune YOLOv8n on conveyor clothing instances and export shareable weights."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve_device(requested: str) -> str:
    import torch

    key = (requested or "auto").strip().lower()
    if key == "auto":
        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if key in {"cuda", "gpu"}:
        return "0"
    return key


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune clothing detector")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg_path = ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    from ultralytics import YOLO

    base = cfg.get("base_model", "yolov8n.pt")
    data = str((ROOT / cfg["data"]).resolve()) if not Path(cfg["data"]).is_absolute() else cfg["data"]
    device = resolve_device(args.device or cfg.get("device", "auto"))
    epochs = args.epochs or int(cfg.get("epochs", 80))

    print(f"base_model: {base}")
    print(f"data: {data}")
    print(f"device: {device}")
    print(f"epochs: {epochs}")

    model = YOLO(base)
    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=int(cfg.get("imgsz", 640)),
        batch=int(cfg.get("batch", 4)),
        device=device,
        workers=int(cfg.get("workers", 2)),
        patience=int(cfg.get("patience", 25)),
        seed=int(cfg.get("seed", 42)),
        project=str(ROOT / cfg.get("project", "runs/detect")),
        name=str(cfg.get("name", "conveyor_clothing")),
        exist_ok=bool(cfg.get("exist_ok", True)),
        pretrained=True,
        plots=True,
    )

    # best checkpoint
    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    src = best if best.exists() else last
    if not src.exists():
        print(f"[FAIL] no weights found under {save_dir / 'weights'}")
        return 1

    export_dir = ROOT / cfg.get("export_dir", "models/clothing/finetuned")
    export_dir.mkdir(parents=True, exist_ok=True)
    export_name = cfg.get("export_name", "yolov8n-conveyor-clothing.pt")
    dest = export_dir / export_name
    shutil.copy2(src, dest)
    # also keep a stable alias
    shutil.copy2(src, export_dir / "best.pt")

    meta = export_dir / "MODEL_CARD.md"
    meta.write_text(
        f"""# Conveyor clothing detector (fine-tuned)

- Base: `{base}`
- Dataset: conveyor flat garments (single class `clothing`)
- Goal: **one bounding box per shirt** on top-down conveyor video
- Checkpoint: `{dest.relative_to(ROOT)}`
- Ultralytics run: `{save_dir.relative_to(ROOT)}`

## For teammates

1. Copy this folder (or at least the `.pt` file):
   `models/clothing/finetuned/yolov8n-conveyor-clothing.pt`
2. Point `configs/model.yaml` → `model_path` to that file
3. Run:
   ```bash
   uv sync
   uv run python -m src.main --video data/input/videos/your.mp4 --device mps
   ```

Class mapping (single class):
```yaml
0: clothing
```
""",
        encoding="utf-8",
    )

    print(f"[OK] exported → {dest}")
    print(f"[OK] model card → {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
