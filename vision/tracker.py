from __future__ import annotations

from typing import Any, Optional

import numpy as np

from config.settings import settings
from core_logic.models import Detection, TrackedObject


class ByteTrackAdapter:
    """Adapts BoxMot (ByteTrack) tracking to the project's :class:`TrackedObject` model.

    Usage::

        tracker = ByteTrackAdapter()
        tracked = tracker.update(detections, frame)
    """

    def __init__(
        self,
        track_threshold: Optional[float] = None,
        match_threshold: Optional[float] = None,
        buffer_frames: Optional[int] = None,
    ) -> None:
        self._track_threshold = track_threshold or settings.track_confidence_threshold
        self._match_threshold = match_threshold or settings.track_match_threshold
        self._buffer_frames = buffer_frames or settings.track_buffer_frames
        self._tracker: Any = None  # lazily created

        # Stable class mapping cache.
        self._class_names: dict[int, str] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[TrackedObject]:
        """Ingest *detections* and return tracked objects with persistent IDs.

        Detections below ``track_threshold`` are still passed to ByteTrack as
        "low-confidence" detections — the tracker may use them for matching.
        """
        if not detections:
            return []

        tracker = self._get_tracker()

        # Convert to the format ByteTrack expects: [[x1, y1, x2, y2, conf, cls_id], ...]
        # ByteTrack's underlying BoxMot uses (tlwh, score, class_id) internally.
        dets_np = np.array([
            [d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence, d.class_id]
            for d in detections
        ], dtype=np.float32)

        tracks = tracker.update(dets_np, frame)

        # Convert ByteTrack output to TrackedObject list.
        # ByteTrack returns: [x1, y1, x2, y2, score, class_id, track_id]
        result: list[TrackedObject] = []
        for t in tracks:
            t = t.tolist() if hasattr(t, "tolist") else t
            x1, y1, x2, y2, score, cls_id, track_id = t[:7]
            track_id = int(track_id)
            cls_id = int(cls_id)
            cname = self._resolve_class_name(cls_id)
            result.append(TrackedObject(
                tracking_id=track_id,
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                class_id=cls_id,
                class_name=cname,
                confidence=float(score),
            ))

        return result

    # ── Private ────────────────────────────────────────────────────────────

    def _get_tracker(self) -> Any:
        if self._tracker is not None:
            return self._tracker

        try:
            from boxmot import BotSort
        except ImportError:
            raise ImportError(
                "boxmot is not installed.  Run: pip install boxmot"
            )

        self._tracker = BotSort(
            track_high_thresh=self._track_threshold,
            match_thresh=self._match_threshold,
            track_buffer=self._buffer_frames,
        )
        return self._tracker

    def _resolve_class_name(self, class_id: int) -> str:
        if class_id not in self._class_names:
            try:
                from config.products import ProductCatalog
                product = ProductCatalog.get_by_class_id(class_id)
                self._class_names[class_id] = product.name
            except KeyError:
                self._class_names[class_id] = f"class_{class_id}"
        return self._class_names[class_id]
