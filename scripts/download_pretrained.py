#!/usr/bin/env python3
"""Download pretrained YOLOv8n clothing detection weights.

Default source:
  https://huggingface.co/kesimeg/yolov8n-clothing-detection

Detect-only (not segmentation). Re-running skips a valid existing file.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "models" / "clothing" / "pretrained"
DEFAULT_NAME = "yolov8n-clothing-detection.pt"
DEFAULT_URL = (
    "https://huggingface.co/kesimeg/yolov8n-clothing-detection/resolve/main/best.pt"
)
# Hugging Face LFS oid for best.pt
DEFAULT_SHA256 = "fc5d77377268dd14cdfbbb935b7f8f1eb521a7fd385148bc03f6641c8f143eee"


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            print(f"\r  {pct:5.1f}% ({downloaded / 1e6:.1f}/{total_size / 1e6:.1f} MB)", end="")
        else:
            print(f"\r  {downloaded / 1e6:.1f} MB", end="")

    print(f"downloading:\n  {url}\n→ {dest}")
    urllib.request.urlretrieve(url, tmp, reporthook)
    print()
    tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download pretrained YOLOv8n clothing detection weights"
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--sha256", default=DEFAULT_SHA256)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dest = OUT_DIR / args.name
    if dest.exists() and not args.force:
        digest = sha256_file(dest)
        if args.sha256 and digest.lower() == args.sha256.lower():
            print(f"[skip] valid weights already present: {dest}")
            return 0
        print(f"[warn] existing file checksum mismatch, re-downloading: {dest}")

    try:
        download(args.url, dest)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] download error: {exc}")
        print("Place the .pt file manually under models/clothing/pretrained/")
        return 1

    if args.sha256:
        digest = sha256_file(dest)
        if digest.lower() != args.sha256.lower():
            print(f"[FAIL] sha256 mismatch\n  got: {digest}\n  expected: {args.sha256}")
            return 1
        print(f"[OK] sha256 verified: {digest}")

    print(f"[OK] weights ready: {dest}")
    print("config path: configs/model.yaml → model_path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
