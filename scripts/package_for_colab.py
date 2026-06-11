#!/usr/bin/env python3
"""
Package dataset for transfer to Google Colab.

Usage:
    # Package the full dataset:
    python scripts/package_for_colab.py

    # Package a lightweight version:
    python scripts/package_for_colab.py --source data_light

    # Custom output name:
    python scripts/package_for_colab.py --out my_dataset.zip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Package dataset for Colab")
    parser.add_argument("--source", default=Path("data"), type=Path,
                        help="Source data directory (default: data)")
    parser.add_argument("--out", default=None, type=Path,
                        help="Output zip path (default: <source>_dataset.zip)")
    args = parser.parse_args()

    data_dir = args.source
    zip_path = args.out or (data_dir.parent / f"{data_dir.name}_dataset.zip")

    print(f"Packaging dataset from {data_dir.resolve()} ...")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as zf:
        # Add dataset.yaml
        yaml_path = data_dir / "dataset.yaml"
        if yaml_path.exists():
            zf.write(yaml_path, "data/dataset.yaml")
            print(f"  Added: {yaml_path.name}")

        # Add train/val/test images and labels
        for split in ["train", "val", "test"]:
            for subdir in ["images", "labels"]:
                path = data_dir / split / subdir
                if not path.exists():
                    continue
                for fpath in sorted(path.iterdir()):
                    arcname = f"data/{split}/{subdir}/{fpath.name}"
                    zf.write(fpath, arcname)

                n = len(list(path.iterdir()))
                print(f"  Added: {split}/{subdir}/ ({n} files)")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\nPackaged to: {zip_path}")
    print(f"Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
