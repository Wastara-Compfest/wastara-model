# Conveyor clothing detector (fine-tuned)

- Base: `yolov8n.pt`
- Dataset: conveyor flat garments (single class `clothing`)
- Goal: **one bounding box per shirt** on top-down conveyor video
- Checkpoint: `models/clothing/finetuned/yolov8n-conveyor-clothing.pt`
- Ultralytics run: `runs/detect/conveyor_clothing`

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
