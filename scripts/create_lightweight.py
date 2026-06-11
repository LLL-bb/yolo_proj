#!/usr/bin/env python3
"""
Create a lightweight subset of the dataset for faster Colab training.

Samples images per class from each split to produce a much smaller dataset
that still covers all 76 classes.

Usage:
    # Quick test run (~500 MB, trains in ~15 min on T4):
    python scripts/create_lightweight.py --samples-per-class 20

    # Balanced moderate size (~2 GB):
    python scripts/create_lightweight.py --samples-per-class 100

    # Default (50 per class, ~1 GB):
    python scripts/create_lightweight.py
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


def subsample_split(
    split: str,
    samples_per_class: int,
    source_dir: Path,
    dest_dir: Path,
    seed: int,
) -> None:
    """Copy a subset of *split* keeping at most *samples_per_class* per class."""
    src_labels = source_dir / split / "labels"
    src_images = source_dir / split / "images"
    dst_labels = dest_dir / split / "labels"
    dst_images = dest_dir / split / "images"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    if not src_labels.exists():
        print(f"  !  {src_labels} not found, skipping")
        return

    # Build per-class image lists
    class_to_images: dict[int, list[Path]] = defaultdict(list)
    for lbl_path in src_labels.glob("*.txt"):
        with open(lbl_path) as f:
            first_line = f.readline().strip()
        if not first_line:
            continue
        class_id = int(first_line.split()[0])
        class_to_images[class_id].append(lbl_path)

    rng = random.Random(seed)
    total_copied = 0
    for class_id in sorted(class_to_images.keys()):
        lbl_paths = class_to_images[class_id]
        rng.shuffle(lbl_paths)
        selected = lbl_paths[:samples_per_class]

        for lbl_path in selected:
            img_name = lbl_path.with_suffix(".jpg").name
            if not (src_images / img_name).exists():
                img_name = lbl_path.with_suffix(".jpeg").name
                if not (src_images / img_name).exists():
                    img_name = lbl_path.with_suffix(".png").name
                    if not (src_images / img_name).exists():
                        continue

            shutil.copy2(src_images / img_name, dst_images / img_name)
            shutil.copy2(lbl_path, dst_labels / lbl_path.name)
            total_copied += 1

    # Also copy images that have zero annotations (if any) — not common here
    print(f"  {split}: {total_copied} images ({len(class_to_images)} classes)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create lightweight dataset for fast Colab training"
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=50,
        help="Max images per class per split (default: 50)",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data"),
        help="Source data directory (default: data)",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=Path("data_light"),
        help="Output directory (default: data_light)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    dest = args.dest_dir
    source = args.source_dir

    print(f"Creating lightweight dataset ({args.samples_per_class} samples/class):")
    print(f"  Source: {source}")
    print(f"  Dest:   {dest}")
    print()

    for split in ("train", "val", "test"):
        subsample_split(split, args.samples_per_class, source, dest, args.seed)

    # Copy dataset.yaml with updated path
    yaml_src = source / "dataset.yaml"
    if yaml_src.exists():
        content = yaml_src.read_text()
        content = content.replace(str(source.resolve()), str(dest.resolve()))
        (dest / "dataset.yaml").write_text(content)
        print(f"  dataset.yaml copied")

    # Quick summary
    print()
    total = 0
    for split in ("train", "val", "test"):
        count = len(list((dest / split / "images").glob("*"))) if (dest / split / "images").exists() else 0
        print(f"  {split}: {count} images")
        total += count

    print(f"\nTotal: {total} images")
    print(f"\nNext: python scripts/package_for_colab.py --source {dest} --out data_light.zip")
    print("  (or modify package_for_colab.py to point at data_light/)")


if __name__ == "__main__":
    main()
