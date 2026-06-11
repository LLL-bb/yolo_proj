#!/usr/bin/env python3
"""
Data augmentation pipeline for YOLO retail datasets.

Applies on-the-fly augmentations using Albumentations, designed to
simulate real-world checkout-camera conditions (lighting variation,
motion blur, sensor noise).

Usage:
    # View augmentation samples (saves to data/aug_samples/):
    python scripts/augment.py preview --image data/train/images/000001.jpg

    # Generate augmented dataset (2× multiplier):
    python scripts/augment.py generate --input-dir data/train --output-dir data/train_aug --multiplier 2
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def build_augmentation_pipeline() -> dict:
    """Return an Albumentations Compose for retail checkout imagery.

    The transforms simulate:
    - Indoor retail lighting shifts (brightness/contrast)
    - Camera sensor noise (low-light checkout environments)
    - Motion blur from customer hand movement
    - Perspective shifts from cart/basket angle
    """
    try:
        import albumentations as A
    except ImportError:
        raise ImportError("Install albumentations: pip install albumentations")

    return A.Compose(
        [
            # Geometric (applied first so pixel-level transforms don't waste work)
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent=(-0.1, 0.1),
                rotate=(-15, 15),
                p=0.6,
            ),
            # Lighting
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2),
                contrast_limit=(-0.15, 0.15),
                p=0.8,
            ),
            A.HueSaturationValue(
                hue_shift_limit=(-10, 10),
                sat_shift_limit=(-20, 20),
                val_shift_limit=(-20, 20),
                p=0.5,
            ),
            # Noise & blur
            A.GaussNoise(var_limit=(10.0, 40.0), p=0.4),
            A.MotionBlur(blur_limit=(3, 7), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            # Weather / atmosphere
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1,
                           num_shadows_upper=2, p=0.3),
            # Cutout (simulate occlusion by hand/arm)
            A.CoarseDropout(
                max_holes=4, max_height=40, max_width=40,
                fill_value=0, p=0.2,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            min_visibility=0.3,
            label_fields=["class_labels"],
        ),
    )


def preview_augmentation(image_path: Path, num_samples: int = 4) -> None:
    """Generate *num_samples* augmented versions of a single image for inspection."""
    import cv2

    pipeline = build_augmentation_pipeline()
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[X]  Cannot read image: {image_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load YOLO label if it exists.
    label_path = image_path.with_suffix(".txt")
    bboxes, classes = [], []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    bbox = list(map(float, parts[1:]))
                    bboxes.append(bbox)
                    classes.append(cls_id)

    out_dir = Path("data/aug_samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        if bboxes:
            augmented = pipeline(image=img, bboxes=bboxes, class_labels=classes)
            aug_img = augmented["image"]
        else:
            aug_img = pipeline(image=img)["image"]

        aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
        out_path = out_dir / f"aug_{i:02d}_{image_path.name}"
        cv2.imwrite(str(out_path), aug_img_bgr)

    print(f"  [OK]  {num_samples} augmented samples → {out_dir}/")


def generate_augmented_dataset(
    input_dir: Path,
    output_dir: Path,
    multiplier: int = 2,
) -> None:
    """Create an augmented copy of a YOLO-format dataset.

    For each image in *input_dir*/images, generates *multiplier* augmented
    variants in *output_dir*/images and corresponding labels in
    *output_dir*/labels.  Original images are also copied (no dropout).
    """
    import cv2

    pipeline = build_augmentation_pipeline()
    src_imgs = input_dir / "images"
    src_lbls = input_dir / "labels"
    dst_imgs = output_dir / "images"
    dst_lbls = output_dir / "labels"
    dst_imgs.mkdir(parents=True, exist_ok=True)
    dst_lbls.mkdir(parents=True, exist_ok=True)

    image_paths = (
        sorted(src_imgs.glob("*.jpg"))
        + sorted(src_imgs.glob("*.jpeg"))
        + sorted(src_imgs.glob("*.png"))
    )

    if not image_paths:
        print(f"  !  No images found in {src_imgs}")
        return

    total = 0
    for img_path in image_paths:
        # Copy original.
        shutil.copy2(img_path, dst_imgs / img_path.name)
        lbl_src = src_lbls / img_path.with_suffix(".txt").name
        if lbl_src.exists():
            shutil.copy2(lbl_src, dst_lbls / lbl_src.name)
        total += 1

        # Read image + labels.
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bboxes, classes = [], []
        if lbl_src.exists():
            with open(lbl_src) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        classes.append(int(parts[0]))
                        bboxes.append(list(map(float, parts[1:])))

        if not bboxes:
            continue  # no objects to augment, skip

        # Generate augmented variants.
        for i in range(multiplier):
            try:
                augmented = pipeline(
                    image=img_rgb,
                    bboxes=bboxes,
                    class_labels=classes,
                )
            except Exception:
                continue  # skip if bbox falls below min_visibility

            aug_bgr = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)
            stem = img_path.stem
            aug_name = f"{stem}_aug{i:02d}{img_path.suffix}"
            cv2.imwrite(str(dst_imgs / aug_name), aug_bgr)

            # Write augmented labels.
            lbl_path = dst_lbls / f"{stem}_aug{i:02d}.txt"
            with open(lbl_path, "w") as f:
                for bbox, cls_id in zip(augmented["bboxes"], augmented["class_labels"]):
                    f.write(f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} "
                            f"{bbox[2]:.6f} {bbox[3]:.6f}\n")
            total += 1

    print(f"  [OK]  {total} total images (original + {multiplier}× aug) → {output_dir}/")


# --- CLI -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Retail data augmentation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_preview = sub.add_parser("preview", help="Preview augmentations on one image")
    p_preview.add_argument("--image", required=True, type=Path)
    p_preview.add_argument("--samples", default=4, type=int)

    p_gen = sub.add_parser("generate", help="Generate augmented dataset")
    p_gen.add_argument("--input-dir", required=True, type=Path)
    p_gen.add_argument("--output-dir", required=True, type=Path)
    p_gen.add_argument("--multiplier", default=2, type=int)

    args = parser.parse_args()

    if args.command == "preview":
        preview_augmentation(args.image, args.samples)
    elif args.command == "generate":
        generate_augmented_dataset(args.input_dir, args.output_dir, args.multiplier)


if __name__ == "__main__":
    main()
