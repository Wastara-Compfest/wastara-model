"""DeepFashion2 reference dataset (stored only — no training in stage 1)."""

# DeepFashion2

DeepFashion2 is the **reference dataset** for garment categories and future
fine-tuning. In **stage 1** we only download, extract, verify, and store it.

We do **not** train or fine-tune here.

## Official source

- Repo: https://github.com/switchablenorms/DeepFashion2
- Access usually requires accepting academic terms and downloading archives
  from the links published by the authors (often Google Drive).

## Local layout

```text
datasets/deepfashion2/
├── raw/                 # archives + extracted original folders
├── annotations/         # reserved for converted labels (later stages)
├── download_meta.json   # written by download script after extract
└── README.md
```

## Commands

```bash
# show official download instructions
uv run python scripts/download_deepfashion2.py --info

# after you place zip archives into raw/
uv run python scripts/download_deepfashion2.py --extract

# verify presence
uv run python scripts/verify_dataset.py
uv run python scripts/verify_dataset.py --strict
```

## Important

Factory evaluation videos live under `data/input/videos/` — **not** here.
That separation is intentional (domain gap measurement).
