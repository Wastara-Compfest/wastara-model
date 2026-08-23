#!/usr/bin/env python3
"""Prepare DeepFashion2 dataset storage (no training).

DeepFashion2 is distributed under academic terms and usually requires
manual registration / download from the official source.

Official project:
  https://github.com/switchablenorms/DeepFashion2

This script does NOT scrape unofficial mirrors. It:
  1. explains access requirements
  2. verifies local archives you place under datasets/deepfashion2/raw/
  3. extracts archives while preserving original structure
  4. is idempotent (skips already-extracted content)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "datasets" / "deepfashion2" / "raw"
META_PATH = ROOT / "datasets" / "deepfashion2" / "download_meta.json"

OFFICIAL_INFO = """
DeepFashion2 — official access
==============================

Project page / repo:
  https://github.com/switchablenorms/DeepFashion2

Typical steps:
  1. Review DeepFashion2 license / academic terms.
  2. Download the official train / validation / test archives
     from the source listed by the authors (Google Drive / Dropbox
     links in the official README — may require request / login).
  3. Place the downloaded .zip archives into:
       datasets/deepfashion2/raw/
  4. Re-run this script with --extract

Expected local layout after extraction (original names preserved):
  datasets/deepfashion2/raw/
    train/
    validation/
    test/
    (or archive files still present + extracted folders)

This stage only STORES the dataset. No training / fine-tuning is run.
""".strip()


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def list_archives(raw_dir: Path) -> list[Path]:
    return sorted(
        [
            *raw_dir.glob("*.zip"),
            *raw_dir.glob("*.tar"),
            *raw_dir.glob("*.tar.gz"),
            *raw_dir.glob("*.tgz"),
        ]
    )


def verify_checksums(raw_dir: Path, checksums: dict[str, str]) -> bool:
    ok = True
    for name, expected in checksums.items():
        path = raw_dir / name
        if not path.exists():
            print(f"[MISS] {name}")
            ok = False
            continue
        digest = sha256_file(path)
        if digest.lower() != expected.lower():
            print(f"[FAIL] {name}: got {digest}, expected {expected}")
            ok = False
        else:
            print(f"[OK]   {name}")
    return ok


def extract_zip(archive: Path, dest: Path) -> None:
    marker = dest / f".extracted_{archive.stem}"
    if marker.exists():
        print(f"[skip] already extracted: {archive.name}")
        return

    print(f"[extract] {archive.name} → {dest}")
    with zipfile.ZipFile(archive) as zf:
        members = zf.namelist()
        total = len(members)
        for i, member in enumerate(members, start=1):
            zf.extract(member, dest)
            if i % 500 == 0 or i == total:
                print(f"  {i}/{total} files", end="\r")
        print()
    marker.write_text(archive.name, encoding="utf-8")


def extract_all(raw_dir: Path) -> None:
    archives = list_archives(raw_dir)
    if not archives:
        print("No archives found in datasets/deepfashion2/raw/")
        print(OFFICIAL_INFO)
        sys.exit(1)

    for archive in archives:
        if archive.suffix == ".zip":
            extract_zip(archive, raw_dir)
        else:
            print(f"[warn] please extract manually (unsupported auto): {archive.name}")


def write_status(raw_dir: Path) -> None:
    archives = [p.name for p in list_archives(raw_dir)]
    folders = [p.name for p in raw_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    status = {
        "archives": archives,
        "folders": folders,
        "note": "Dataset stored for future training only. Stage 1 does not train.",
    }
    META_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"status → {META_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepFashion2 dataset helper")
    parser.add_argument("--info", action="store_true", help="show official download instructions")
    parser.add_argument("--extract", action="store_true", help="extract archives already in raw/")
    parser.add_argument(
        "--checksums",
        type=str,
        default=None,
        help="optional JSON file mapping filename → sha256",
    )
    parser.add_argument("--verify-only", action="store_true", help="verify checksums only")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.info or (not args.extract and not args.verify_only and not args.checksums):
        print(OFFICIAL_INFO)
        archives = list_archives(RAW_DIR)
        print(f"\nArchives currently in raw/: {len(archives)}")
        for a in archives:
            print(f"  - {a.name} ({a.stat().st_size / 1e6:.1f} MB)")
        if not args.extract and not args.verify_only:
            return 0

    if args.checksums:
        checksums = json.loads(Path(args.checksums).read_text(encoding="utf-8"))
        ok = verify_checksums(RAW_DIR, checksums)
        if not ok:
            return 1
        if args.verify_only:
            return 0

    if args.extract:
        extract_all(RAW_DIR)
        write_status(RAW_DIR)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
