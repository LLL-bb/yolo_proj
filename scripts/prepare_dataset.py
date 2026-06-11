#!/usr/bin/env python3
"""
Grab-and-Go dataset preparation pipeline.

Converts raw retail datasets (RPC COCO JSON, SKU-110K CSV) into YOLO format
and organizes them into train/val/test splits.

Usage:
    # Prepare SKU-110K (CSV annotations):
    python scripts/prepare_dataset.py sku110k --raw-dir data/raw/sku110k --out-dir data

    # Prepare RPC dataset (COCO JSON):
    python scripts/prepare_dataset.py rpc --raw-dir data/raw/rpc --out-dir data

    # Prepare your own dataset (COCO JSON):
    python scripts/prepare_dataset.py coco --raw-dir data/raw/custom --out-dir data

    # Split raw YOLO-format data into train/val/test:
    python scripts/prepare_dataset.py split --raw-dir data/raw/yolo --out-dir data
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Optional


# -- Helpers ----------------------------------------------------------------

def _yolo_bbox(img_w: int, img_h: int,
               coco_bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Convert COCO-style (x, y, w, h) to YOLO normalized (cx, cy, w, h)."""
    x, y, w, h = coco_bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    return (cx, cy, w / img_w, h / img_h)


def _write_yolo_label(txt_path: Path, class_id: int,
                      cx: float, cy: float, w: float, h: float) -> None:
    with open(txt_path, "a") as f:
        f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


# -- Converters -------------------------------------------------------------

def convert_coco_json(
    annotations_path: Path,
    images_dir: Path,
    output_dir: Path,
    subset: str,
    category_filter: Optional[set[int]] = None,
) -> int:
    """Convert a COCO-format JSON into YOLO .txt labels and copy images.

    Returns the number of images processed.
    """
    with open(annotations_path) as f:
        coco = json.load(f)

    # Map category_id -> index sequentially.
    cat_id_map: dict[int, int] = {}
    for cat in coco.get("categories", []):
        cid = cat["id"]
        if category_filter is None or cid in category_filter:
            cat_id_map[cid] = len(cat_id_map)

    if not cat_id_map:
        print(f"  !  No matching categories in {annotations_path.name}")
        return 0

    # Build image-id -> filename lookup.
    img_map: dict[int, str] = {}
    img_size: dict[int, tuple[int, int]] = {}
    for img in coco.get("images", []):
        img_map[img["id"]] = img["file_name"]
        img_size[img["id"]] = (img["width"], img["height"])

    # Group annotations by image_id.
    img_anns: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        img_anns.setdefault(ann["image_id"], []).append(ann)

    out_images = output_dir / subset / "images"
    out_labels = output_dir / subset / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_id, file_name in img_map.items():
        anns = img_anns.get(img_id, [])
        src = images_dir / file_name
        if not src.exists():
            continue

        # Copy image.
        dst_img = out_images / file_name
        shutil.copy2(src, dst_img)

        # Write YOLO label.
        w, h = img_size.get(img_id, (0, 0))
        if w == 0 or h == 0:
            continue
        txt_path = out_labels / src.with_suffix(".txt").name
        for ann in anns:
            cat_id = ann.get("category_id")
            mapped = cat_id_map.get(cat_id)
            if mapped is None:
                continue
            cx, cy, nw, nh = _yolo_bbox(w, h, ann["bbox"])
            _write_yolo_label(txt_path, mapped, cx, cy, nw, nh)

        count += 1

    return count


def convert_sku110k_csv(
    csv_path: Path,
    images_dir: Path,
    output_dir: Path,
    subset: str,
) -> int:
    """Convert SKU-110K CSV format into YOLO .txt labels.

    CSV columns: filename,x1,y1,x2,y2,class,image_width,image_height
    (class is always "object" for SKU-110K -> mapped to class_id 0).

    Returns the number of images processed.
    """
    out_images = output_dir / subset / "images"
    out_labels = output_dir / subset / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[dict]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            groups.setdefault(row["filename"], []).append(row)

    count = 0
    for fname, rows in groups.items():
        src = images_dir / fname
        if not src.exists():
            continue

        # Copy image.
        dst_img = out_images / fname
        shutil.copy2(src, dst_img)

        # Write YOLO label.
        img_w = int(rows[0]["image_width"])
        img_h = int(rows[0]["image_height"])
        txt_path = out_labels / Path(fname).with_suffix(".txt").name
        for row in rows:
            x1, y1, x2, y2 = map(float, [row["x1"], row["y1"], row["x2"], row["y2"]])
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            nw = (x2 - x1) / img_w
            nh = (y2 - y1) / img_h
            _write_yolo_label(txt_path, 0, cx, cy, nw, nh)

        count += 1

    return count


# -- Dataset splitting (when you only have a big pile of images + labels) ---

def split_dataset(
    source_dir: Path,
    output_dir: Path,
    val_split: float = 0.15,
    test_split: float = 0.05,
    seed: int = 42,
) -> None:
    """Randomly split images/labels pairs into train/val/test."""
    import random
    random.seed(seed)

    images = sorted(source_dir.glob("*.[jJ][pP][gG]")) + \
             sorted(source_dir.glob("*.[jJ][pP][eE][gG]")) + \
             sorted(source_dir.glob("*.[pP][nN][gG]"))

    if not images:
        print("  !  No images found in source directory.")
        return

    random.shuffle(images)
    n = len(images)
    n_test = int(n * test_split)
    n_val = int(n * val_split)

    splits = {
        "train": images[: n - n_test - n_val],
        "val": images[n - n_test - n_val : n - n_test],
        "test": images[n - n_test :],
    }

    for split_name, img_list in splits.items():
        if not img_list:
            continue
        out_imgs = output_dir / split_name / "images"
        out_lbls = output_dir / split_name / "labels"
        out_imgs.mkdir(parents=True, exist_ok=True)
        out_lbls.mkdir(parents=True, exist_ok=True)

        for src_img in img_list:
            shutil.copy2(src_img, out_imgs / src_img.name)
            # Copy matching label.
            for ext in [".txt", ".json"]:
                src_lbl = src_img.with_suffix(ext)
                if src_lbl.exists():
                    shutil.copy2(src_lbl, out_lbls / src_lbl.name)
                    break

        print(f"  {split_name}: {len(img_list)} images -> {out_imgs}")


# -- Generate data.yaml -----------------------------------------------------

def write_data_yaml(
    output_dir: Path,
    class_names: list[str],
    filename: str = "dataset.yaml",
) -> Path:
    """Write a YOLO-compatible data.yaml."""
    out_dir = output_dir.resolve()
    yaml_path = output_dir / filename
    lines = [
        f"# YOLO dataset config -- auto-generated by prepare_dataset.py",
        f"# {len(class_names)} classes",
        "",
        f"path: {out_dir.as_posix()}",
        f"train: train/images",
        f"val: val/images",
        "test: test/images  # optional",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for i, name in enumerate(class_names):
        lines.append(f"  {i}: {name!r}")
    lines.append("")

    yaml_path.write_text("\n".join(lines))
    print(f"   data.yaml -> {yaml_path}")
    return yaml_path


# --- CLI -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare retail dataset for YOLO training")
    sub = parser.add_subparsers(dest="command", required=True)

    # -- SKU-110K -------------------------------------------------------
    p_sku = sub.add_parser("sku110k", help="Convert SKU-110K (CSV) to YOLO format")
    p_sku.add_argument("--raw-dir", required=True, type=Path)
    p_sku.add_argument("--out-dir", default=Path("data"), type=Path)

    # -- COCO JSON ------------------------------------------------------
    p_coco = sub.add_parser("coco", help="Convert COCO JSON to YOLO format")
    p_coco.add_argument("--raw-dir", required=True, type=Path)
    p_coco.add_argument("--out-dir", default=Path("data"), type=Path)
    p_coco.add_argument("--filter-categories", nargs="*", type=int,
                        help="Only include these category IDs")

    # -- RPC ------------------------------------------------------------
    p_rpc = sub.add_parser("rpc", help="Convert RPC (COCO JSON) to YOLO format")
    p_rpc.add_argument("--raw-dir", required=True, type=Path)
    p_rpc.add_argument("--out-dir", default=Path("data"), type=Path)
    p_rpc.add_argument("--filter-categories", nargs="*", type=int,
                        help="Only include these category IDs")

    # -- Split ----------------------------------------------------------
    p_split = sub.add_parser("split", help="Split a single directory into train/val/test")
    p_split.add_argument("--raw-dir", required=True, type=Path)
    p_split.add_argument("--out-dir", default=Path("data"), type=Path)
    p_split.add_argument("--val-split", default=0.15, type=float)
    p_split.add_argument("--test-split", default=0.05, type=float)
    p_split.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    # -- Dispatch -------------------------------------------------------
    if args.command == "sku110k":
        print("Converting SKU-110K CSV -> YOLO format ...")
        raw = args.raw_dir
        for subset in ("train", "val", "test"):
            csv_file = raw / f"{subset}.csv"
            img_dir = raw / subset
            if not csv_file.exists():
                print(f"  !  Skipping {subset}: {csv_file} not found")
                continue
            n = convert_sku110k_csv(csv_file, img_dir, args.out_dir, subset)
            print(f"  {subset}: {n} images")
        write_data_yaml(args.out_dir, ["object"])

    elif args.command == "coco":
        print("Converting COCO JSON -> YOLO format ...")
        raw = args.raw_dir
        cat_filter = set(args.filter_categories) if args.filter_categories else None
        total = 0
        for subset in ("train", "val", "test"):
            json_file = raw / "annotations" / f"{subset}.json"
            img_dir = raw / subset
            if not json_file.exists():
                json_file = raw / "annotations" / f"instances_{subset}.json"
            if not json_file.exists():
                print(f"  !  Skipping {subset}: no annotations found")
                continue
            n = convert_coco_json(json_file, img_dir, args.out_dir, subset, cat_filter)
            print(f"  {subset}: {n} images")
            total += n

        if total > 0:
            # Read category names from any JSON that had matches.
            for subset in ("train", "val", "test"):
                jp = raw / "annotations" / f"{subset}.json"
                if not jp.exists():
                    continue
                with open(jp) as f:
                    coco = json.load(f)
                cats = [c["name"] for c in coco.get("categories", [])
                        if cat_filter is None or c["id"] in cat_filter]
                if cats:
                    write_data_yaml(args.out_dir, cats)
                    break

    elif args.command == "rpc":
        print("Converting RPC dataset (COCO JSON) -> YOLO format ...")
        raw = args.raw_dir
        cat_filter = set(args.filter_categories) if args.filter_categories else None
        total = 0
        # RPC uses train2019/val2019/test2019 dirs + instances_*2019.json.
        subsets_2019 = [("train", "train2019"), ("val", "val2019"), ("test", "test2019")]
        for subset, dirname in subsets_2019:
            json_file = raw / f"instances_{dirname}.json"
            img_dir = raw / dirname
            if not json_file.exists():
                print(f"  !  Skipping {subset}: {json_file} not found")
                continue
            n = convert_coco_json(json_file, img_dir, args.out_dir, subset, cat_filter)
            print(f"  {subset}: {n} images")
            total += n

        if total > 0:
            # Read RPC category names from the first available JSON.
            for subset, dirname in subsets_2019:
                jp = raw / f"instances_{dirname}.json"
                if jp.exists():
                    with open(jp) as f:
                        coco = json.load(f)
                    cats = [c["name"] for c in coco.get("categories", [])
                            if cat_filter is None or c["id"] in cat_filter]
                    if cats:
                        write_data_yaml(args.out_dir, cats)
                        break

    elif args.command == "split":
        print(f"Splitting {args.raw_dir} -> train/val/test ...")
        split_dataset(args.raw_dir, args.out_dir,
                      args.val_split, args.test_split, args.seed)

    print("\n[OK]  Done.")


if __name__ == "__main__":
    main()
