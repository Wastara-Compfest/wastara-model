# Wastara Model — Garment Detection + Inspection Pipeline

**Stage 1:** garment object detection (YOLO)  
**Stage 2:** multi-object tracking + inspection zone + defect/anomaly analysis

Unit of analysis = **one garment track**, not one independent frame.

```text
Video → Garment Detector → Tracker (ByteTrack/IoU)
      → Inspection Zone → Representative Frames → Garment ROI
      → Defect / Anomaly Model → Temporal Aggregation
      → Garment-level JSON + evidence
```

**Important:** defect analysis is a **separate module** from garment detection + tracking.
Default defect mode is **anomaly (Mode B)** — no fabricated supervised defect classes without labeled data.

---

## Related repos

| Repo | Role |
|------|------|
| [wastara-fe](https://github.com/Wastara-Compfest/wastara-fe) | Operator dashboard (Next.js) — never calls this service directly |
| [wastara-be](https://github.com/Wastara-Compfest/wastara-be) | Backend API — proxies camera control here and receives detection results at `/internal/defect-events` |

---

## Stack

| Tool | Role |
|------|------|
| `uv` | environment + deps |
| Ultralytics YOLO | inference |
| PyTorch | CUDA / MPS / CPU |
| OpenCV | video I/O |
| PyYAML | configs |

Device priority: **CUDA → MPS → CPU** (`device: auto`, overridable).

---

## Layout

```text
wastara-model/
├── configs/
│   ├── model.yaml
│   ├── inference.yaml
│   ├── train.yaml
│   └── inspection.yaml
├── models/clothing/
│   ├── pretrained/
│   └── finetuned/
├── src/
│   ├── detection/          # garment detector only
│   ├── tracking/           # ByteTrack / IoU
│   ├── inspection/         # zone, sampler, ROI
│   ├── defect/             # Mode A supervised | Mode B anomaly
│   ├── visualization/
│   ├── pipeline/           # garment_inspection_pipeline
│   ├── video/
│   └── evaluation/
└── data/output/inspection/
    ├── results/
    ├── evidence/
    └── annotated_videos/
```

---

## Setup without Docker (uv)

```bash
cd wastara-model
uv sync
cp .env.example .env   # set BACKEND_URL / INTERNAL_API_KEY to match wastara-be
uv run wastara-serve   # FastAPI service on :8100
```

## Setup with Docker

```bash
docker compose up --build
```

Serves the same FastAPI app on `http://localhost:8100`. `./models` and `./data` are mounted as volumes, so weights and uploaded/output files persist and can be swapped without rebuilding. Set `BACKEND_URL`/`INTERNAL_API_KEY` to match `wastara-be` (defaults assume `wastara-be` is running on the host at `:8000` — see `docker-compose.yml`).

**Live webcam does not work inside Docker** — Docker Desktop on macOS/Windows has no camera device passthrough. Docker mode is for the **mock-data path**: feed a recorded video file as the inspection `source` (see `NEXT_PUBLIC_INSPECTION_SOURCE` in `wastara-fe`, or upload a video through the dashboard's Live View) — the real detection pipeline runs identically either way. For an actual live webcam demo, run `uv run wastara-serve` directly on the host instead.

### Model weights

`configs/model.yaml` defaults to `models/clothing/finetuned/yolov8n-conveyor-clothing.pt`, which is **gitignored** (too large for git) and must be shared out-of-band (Drive/zip/git-lfs) between teammates. Without it, the service fails to start with `FileNotFoundError`. If you don't have that file, use the generic pretrained fallback instead:

```bash
uv run python scripts/download_pretrained.py
MODEL_PATH=models/clothing/pretrained/yolov8n-clothing-detection.pt uv run wastara-serve
# or with Docker: set MODEL_PATH in .env before `docker compose up`
```

## Environment variables

| Variable | Default (`.env.example`) | Keterangan |
|---|---|---|
| `PORT` | `8100` | HTTP port for the FastAPI service |
| `BACKEND_URL` | `http://localhost:8000` | Base URL of `wastara-be`, used to push detections/frames and camera-control acks |
| `INTERNAL_API_KEY` | `change-me` | Shared secret with `wastara-be`'s internal endpoints — must match exactly |
| `DEVICE` | `auto` | Inference device: `auto` \| `cuda` \| `mps` \| `cpu` |
| `MODEL_PATH` | (from `configs/model.yaml`) | Override the garment detector weights path |
| `CONFIDENCE_THRESHOLD` / `IOU_THRESHOLD` | `0.5` / `0.45` | Detection thresholds |

---

## Workflow

### 1) Weights

**Fine-tuned (recommended for conveyor):**

```text
models/clothing/finetuned/yolov8n-conveyor-clothing.pt
```

Share this file with teammates (Drive / zip / git-lfs). It is gitignored because of size.

Or download the older generic pretrained:

```bash
uv run python scripts/download_pretrained.py
```

Fine-tune again:

```bash
uv run python scripts/train_clothing.py --device mps
```

### 2) (Optional) Prepare DeepFashion2 dataset — storage only

DeepFashion2 needs official / academic access. The script will **not** bypass that.

```bash
uv run python scripts/download_deepfashion2.py --info
# place official zip archives into datasets/deepfashion2/raw/
uv run python scripts/download_deepfashion2.py --extract
uv run python scripts/verify_dataset.py
```

### 3) Put a test video

```bash
cp /path/to/garment_test_01.mp4 data/input/videos/
```

### 4) Run detection

```bash
uv run python -m src.main --video data/input/videos/garment_test_01.mp4
# or
uv run wastara-detect --video data/input/videos/garment_test_01.mp4 --device mps
```

Outputs:

```text
data/output/annotated_videos/garment_test_01_detected.mp4
data/output/detections/garment_test_01.json
data/output/frames/garment_test_01/frame_XXXXXX.jpg   # only frames with detections
```

### 6) Run garment inspection (detect → track → defect)

```bash
uv run wastara-inspect \
  --video data/input/videos/demovid.mp4 \
  --device mps \
  --tracker bytetrack \
  --defect-mode anomaly
```

Config: `configs/inspection.yaml`

| Mode | When | Output type |
|------|------|-------------|
| `anomaly` (default) | No labeled defect dataset yet | `anomaly` regions via Mode B baseline |
| `supervised` | Labeled defect YOLO weights ready | `stain` / `hole_or_tear` / … from weights |
| `none` | Wiring only | always NORMAL |

Outputs:

```text
data/output/inspection/results/*_garments.json
data/output/inspection/evidence/track_XXXX_*.jpg
data/output/inspection/annotated_videos/*_inspection.mp4
```

Latency is logged separately for garment detection, tracking, and defect analysis.

---

## HTTP API (`wastara-serve`, called by `wastara-be`)

| Method | Path | Keterangan |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/camera/start` | `{source, machine_id}` — starts the live capture+inference loop (webcam or a video file path) |
| POST | `/camera/stop` | Stops it |
| GET | `/camera/status` | Running state, queue depth, dropped frames |
| POST | `/inspections/video` | `multipart/form-data`: `file`, `machine_id`, `video_inspection_id` — runs the full offline pipeline on an uploaded video in the background, pushing results to `wastara-be` as they're found |

---

## Config knobs

`configs/model.yaml` — garment detector path / classes  
`configs/inference.yaml` — detect thresholds  
`configs/inspection.yaml` — tracker, zone ROI, sample_count, aggregation, defect mode

---

## Tests

```bash
uv run pytest -q
```

---

## Milestones

| ID | Goal |
|----|------|
| M1–M4 | Garment detection baseline |
| M5 | Tracking + inspection zone + sampling |
| M6 | Mode B anomaly wiring (+ PatchCore later) |
| M7 | Mode A supervised defect when labeled data exists |
| M8 | Garment-level evaluation vs GT |
