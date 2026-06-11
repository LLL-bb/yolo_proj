"""
Train YOLO11n on AutoDL — run directly: python scripts/train_autodl.py

Assumes:
  - Dataset is at /root/autodl-tmp/data/  (data_light or full)
  - Ultralytics is installed (pip install ultralytics)
"""

from pathlib import Path
import yaml

# --- Config ---
DATA_DIR = Path("/root/autodl-tmp/data")
EPOCHS = 100
BATCH = 32
WORKERS = 4
PROJECT = "/root/autodl-tmp/runs"
NAME = "yolo11n_rpc"
# -------------

# Fix dataset.yaml path (in case it has hardcoded Windows paths)
yaml_path = DATA_DIR / "dataset.yaml"
with open(yaml_path) as f:
    cfg = yaml.safe_load(f)
cfg["path"] = str(DATA_DIR.resolve())
with open(yaml_path, "w") as f:
    yaml.dump(cfg, f, sort_keys=False)
print(f"Dataset path -> {cfg['path']}")
print(f"  train: {len(list((DATA_DIR/'train'/'images').glob('*')))} images")
print(f"  val:   {len(list((DATA_DIR/'val'/'images').glob('*')))} images")
print(f"  test:  {len(list((DATA_DIR/'test'/'images').glob('*')))} images")
print(f"  nc:    {cfg['nc']}")

from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data=str(yaml_path),
    epochs=EPOCHS,
    imgsz=640,
    batch=BATCH,
    lr0=0.01,
    weight_decay=0.0005,
    warmup_epochs=3,
    cos_lr=True,
    augment=True,
    device=0,
    workers=WORKERS,
    project=PROJECT,
    name=NAME,
    exist_ok=True,
    verbose=True,
)
