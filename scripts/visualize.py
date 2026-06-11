#!/usr/bin/env python3
"""
Visualize YOLO-format annotations on images.

Useful for verifying that dataset conversion and augmentation produced
correct bounding boxes.

Usage:
    # Show a single image with annotations:
    python scripts/visualize.py view --image data/train/images/000001.jpg

    # Check a random sample of the dataset:
    python scripts/visualize.py sample --dir data/train --count 10 --save-dir data/check_samples

    # Export all annotations as overlaid images for review:
    python scripts/visualize.py export --dir data/val --out-dir data/val_overlay
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


# Simple colour palette (BGR).
_COLORS = [
    (0, 255, 0),      # green
    (255, 0, 0),      # blue
    (0, 0, 255),      # red
    (0, 255, 255),    # yellow
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
    (128, 255, 0),    # lime
    (255, 128, 0),    # orange
    (128, 0, 255),    # purple
    (0, 128, 255),    # sky
]


def _load_labels(txt_path: Path) -> list[dict]:
    """Parse a YOLO .txt label file."""
    labels = []
    if not txt_path.exists():
        return labels
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                labels.append({
                    "class_id": int(parts[0]),
                    "bbox": list(map(float, parts[1:])),  # cx, cy, w, h (normalized)
                })
    return labels


def _denormalize_bbox(bbox: list[float], img_w: int, img_h: int) -> tuple[int, ...]:
    """Convert YOLO (cx, cy, w, h) normalized → pixel (x1, y1, x2, y2)."""
    cx, cy, w, h = bbox
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return (x1, y1, x2, y2)


def draw_annotations(image_path: Path, class_names: list[str] | None = None) -> np.ndarray:
    """Read *image_path*, overlay YOLO annotations, return BGR image."""
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    h, w = img.shape[:2]

    label_path = image_path.with_suffix(".txt")
    labels = _load_labels(label_path)

    for label in labels:
        cid = label["class_id"]
        x1, y1, x2, y2 = _denormalize_bbox(label["bbox"], w, h)
        color = _COLORS[cid % len(_COLORS)]

        # Draw filled bounding box with transparency overlay.
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)

        # Draw bounding box outline.
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label text.
        label_text = class_names[cid] if class_names else f"cls_{cid}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label_text, (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return img


# --- CLI -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise YOLO annotations")
    sub = parser.add_subparsers(dest="command", required=True)

    p_view = sub.add_parser("view", help="Show one annotated image")
    p_view.add_argument("--image", required=True, type=Path)
    p_view.add_argument("--classes", type=Path, help="Path to data.yaml with class names")

    p_sample = sub.add_parser("sample", help="View random image samples")
    p_sample.add_argument("--dir", required=True, type=Path)
    p_sample.add_argument("--count", default=10, type=int)
    p_sample.add_argument("--save-dir", type=Path, help="Directory to save samples")
    p_sample.add_argument("--classes", type=Path)

    p_export = sub.add_parser("export", help="Export all annotations as overlaid images")
    p_export.add_argument("--dir", required=True, type=Path)
    p_export.add_argument("--out-dir", required=True, type=Path)
    p_export.add_argument("--classes", type=Path)

    args = parser.parse_args()

    # Optional class-name lookup.
    class_names: list[str] | None = None
    if args.classes and args.classes.exists():
        import yaml
        with open(args.classes) as f:
            cfg = yaml.safe_load(f)
        class_names = cfg.get("names", [])

    if args.command == "view":
        import cv2
        img = draw_annotations(args.image, class_names)
        cv2.imshow("Annotations", img)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.command == "sample":
        import cv2
        images = list((args.dir / "images").glob("*")) if (args.dir / "images").exists() else []
        if not images:
            print(f"  !  No images found in {args.dir / 'images'}")
            return
        random.shuffle(images)
        out_dir = Path(args.save_dir) if args.save_dir else args.dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_path in images[: args.count]:
            try:
                annotated = draw_annotations(img_path, class_names)
                dst = out_dir / f"viz_{img_path.name}"
                cv2.imwrite(str(dst), annotated)
                print(f"  [OK]  {dst}")
            except Exception as e:
                print(f"  !  {img_path.name}: {e}")

    elif args.command == "export":
        import cv2
        src_imgs = args.dir / "images"
        dst = args.out_dir
        dst.mkdir(parents=True, exist_ok=True)
        for img_path in sorted(src_imgs.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            try:
                annotated = draw_annotations(img_path, class_names)
                cv2.imwrite(str(dst / img_path.name), annotated)
            except Exception as e:
                print(f"  !  {img_path.name}: {e}")
        print(f"  [OK]  Exported → {dst}/")


if __name__ == "__main__":
    main()
