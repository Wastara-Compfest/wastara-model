#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "datasets" / "deepfashion2" / "raw"

EXPECTED_HINTS = ("train", "validation", "val", "test")


def scan(raw_dir: Path) -> dict:
    archives = [
        p.name
        for p in raw_dir.glob("*")
        if p.suffix in {".zip", ".tar", ".gz", ".tgz"} or p.name.endswith(".tar.gz")
    ]
    folders = [p.name for p in raw_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    files = [p.name for p in raw_dir.iterdir() if p.is_file()]
    return {"archives": archives, "folders": folders, "files": files}


def looks_ready(info: dict) -> bool:
    names = {n.lower() for n in info["folders"]}
    return any(h in names for h in EXPECTED_HINTS) or bool(info["archives"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify DeepFashion2 dataset storage")
    parser.add_argument("--strict", action="store_true", help="require extracted train/val folders")
    args = parser.parse_args()

    if not RAW_DIR.exists():
        print(f"[FAIL] missing {RAW_DIR}")
        return 1

    info = scan(RAW_DIR)
    print(f"raw dir: {RAW_DIR}")
    print(f"  archives: {len(info['archives'])}")
    for a in info["archives"]:
        print(f"    - {a}")
    print(f"  folders: {len(info['folders'])}")
    for f in info["folders"]:
        print(f"    - {f}")

    if args.strict:
        names = {n.lower() for n in info["folders"]}
        has_split = any(x in names for x in ("train", "validation", "val", "test"))
        if not has_split:
            print("[FAIL] strict mode: expected extracted train/validation/test folders")
            print("run: uv run python scripts/download_deepfashion2.py --info")
            return 1
        print("[OK] extracted split folders present")
        return 0

    if looks_ready(info):
        print("[OK] dataset artifacts present (stored only — no training in stage 1)")
        return 0

    print("[WARN] no DeepFashion2 archives/folders found yet")
    print("run: uv run python scripts/download_deepfashion2.py --info")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
