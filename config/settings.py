from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Settings:
    # ── Video Source ──────────────────────────────────────────────────────
    camera_id: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    process_every_n_frames: int = 1  # skip N-1 frames between detections

    # ── Detection ─────────────────────────────────────────────────────────
    model_path: str = "data/models/yolo11n-retail.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "cpu"  # "cpu" | "cuda:0" | "mps"

    # ── Tracking (ByteTrack) ──────────────────────────────────────────────
    track_confidence_threshold: float = 0.5
    track_buffer_frames: int = 30  # frames to keep lost track alive
    track_match_threshold: float = 0.8  # IoU threshold for track matching

    # ── Zone Evaluation ───────────────────────────────────────────────────
    point_mode: str = "bottom_center"  # "bottom_center" | "centroid"

    # ── Event Debouncing ──────────────────────────────────────────────────
    debounce_frames: int = 5  # consecutive frames needed to confirm transition
    lost_track_ttl_frames: int = 30  # max age of a lost track for identity-switch matching
    identity_switch_iou_threshold: float = 0.3  # min IoU to match a new track to a lost one

    # ── Cart / Session ────────────────────────────────────────────────────
    max_cart_items: int = 50  # safety cap

    # ── Web Server ────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


settings = Settings()
