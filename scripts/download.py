#!/usr/bin/env python3
"""
Download helper for retail checkout datasets.

Usage:
    python scripts/download.py rpc --out-dir data/raw/rpc
    python scripts/download.py rpc --out-dir data/raw/rpc --kaggle
    python scripts/download.py sku110k --out-dir data/raw/sku110k
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path


SKU110K_URL = (
    "http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz"
)


# -- SKU-110K ----------------------------------------------------------------

def download_sku110k(out_dir: Path) -> None:
    """Download and extract SKU-110K dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tarball = out_dir / "SKU110K_fixed.tar.gz"
    extracted = out_dir / "SKU110K_fixed"

    if extracted.exists():
        print(f"  Already extracted: {extracted}")
        return

    if not tarball.exists():
        print(f"  Downloading SKU-110K (~13.6 GB) ...")
        print(f"  URL: {SKU110K_URL}")

        def _progress(block: int, chunk: int, total: int) -> None:
            done = block * chunk / (1024 ** 3)
            total_gb = total / (1024 ** 3) if total > 0 else 0
            pct = 100.0 * block * chunk / total if total > 0 else 0
            if block % 1000 == 0:
                print(f"    {done:.1f} GB / {total_gb:.1f} GB  ({pct:.0f}%)")

        try:
            urllib.request.urlretrieve(SKU110K_URL, tarball, _progress)
            print(f"  Downloaded: {tarball}")
        except Exception as e:
            print(f"  Download failed: {e}")
            print(f"  Try manually downloading to: {tarball}")
            return
    else:
        print(f"  Found existing tarball: {tarball}")

    print(f"  Extracting ... (may take a few minutes)")
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(path=out_dir)
    print(f"  Extracted to: {extracted}")


# -- RPC ---------------------------------------------------------------------

KAGGLE_DATASET = "diyer22/retail-product-checkout-dataset"


def download_rpc_via_kaggle(out_dir: Path) -> None:
    """Download RPC via the Kaggle API."""
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub
    except ImportError:
        print("  Installing kagglehub ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "kagglehub"]
        )
        import kagglehub  # noqa: F811

    print(f"  Downloading {KAGGLE_DATASET} via kagglehub ...")
    print(f"  (This is ~15 GB and will take a while.)")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"  Downloaded to cache: {path}")

    # Copy to output directory.
    import shutil
    src = Path(path)
    for item in src.iterdir():
        dst = out_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)
    print(f"  Copied to: {out_dir}")


def print_rpc_instructions(out_dir: Path) -> None:
    """Print manual download instructions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"""
-------------------------------------------------
  RPC (Retail Product Checkout) Dataset
-------------------------------------------------

  Option A — Kaggle (recommended, fastest):
    python scripts/download.py rpc --out-dir data/raw/rpc --kaggle

    This uses kagglehub to download from:
    https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset

  Option B — Manual download:
    1. Download from Kaggle:
       https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset

    2. Extract into:
       {out_dir}/

    3. Expected structure:
       {out_dir}/
       +-- train2019/              (53,739 single-product images)
       +-- val2019/                (6,000 checkout images)
       +-- test2019/               (24,000 checkout images)
       +-- instances_train2019.json
       +-- instances_val2019.json
       +-- instances_test2019.json

  After downloading, run:
    python scripts/prepare_dataset.py rpc --raw-dir {out_dir} --out-dir data
""")


# --- CLI -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download retail datasets")
    sub = parser.add_subparsers(dest="dataset", required=True)

    p_rpc = sub.add_parser("rpc", help="Download RPC dataset")
    p_rpc.add_argument("--out-dir", default=Path("data/raw/rpc"), type=Path)
    p_rpc.add_argument("--kaggle", action="store_true",
                       help="Auto-download via kagglehub")

    p_sku = sub.add_parser("sku110k", help="Download and extract SKU-110K")
    p_sku.add_argument("--out-dir", default=Path("data/raw/sku110k"), type=Path)

    args = parser.parse_args()

    if args.dataset == "rpc":
        if args.kaggle:
            download_rpc_via_kaggle(args.out_dir)
        else:
            print_rpc_instructions(args.out_dir)
    elif args.dataset == "sku110k":
        download_sku110k(args.out_dir)


if __name__ == "__main__":
    main()
