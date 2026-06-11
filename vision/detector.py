from __future__ import annotations

from typing import Any, Optional

import numpy as np

from config.settings import settings
from core_logic.models import Detection


class YOLODetector:
    """Wrapper around an Ultralytics YOLO model for object detection.

    Usage::

        detector = YOLODetector("data/models/yolo11n-retail.pt")
        detections = detector.detect(frame_bgr)  # np.ndarray, H×W×3
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        device: Optional[str] = None,
    ) -> None:
        self._model_path = model_path or settings.model_path
        self._conf = conf_threshold or settings.confidence_threshold
        self._iou = iou_threshold or settings.iou_threshold
        self._device = device or settings.device
        self._model: Any = None  # lazily loaded

    # ── Public API ─────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on *frame* and return a list of :class:`Detection`.

        The frame is expected in OpenCV BGR format (H×W×3, uint8).
        """
        if self._model is None:
            self._load_model()

        results = self._model(frame, conf=self._conf, iou=self._iou, verbose=False)
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []

        detections: list[Detection] = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])
            detections.append(
                Detection(bbox=(x1, y1, x2, y2), confidence=conf, class_id=cls_id)
            )

        return detections

    # ── Private ────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is not installed.  Run: pip install ultralytics"
            )
        self._model = YOLO(self._model_path)
