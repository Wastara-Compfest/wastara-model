# Conveyor clothing dataset (fine-tune)

Single-class instance detection: **one box per garment** on conveyor video.

Labels were bootstrapped with a conveyor-specific band splitter, then filtered.
For production quality, review/correct labels (Label Studio / CVAT / Roboflow).

```text
datasets/conveyor_clothing/
├── data.yaml
├── images/{train,val}/
└── labels/{train,val}/
```

Train:

```bash
uv run python scripts/train_clothing.py --device mps
```
