from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from config.settings import settings
from core_logic.event_processor import DebouncedEventProcessor
from core_logic.models import CartEvent, CartState, TrackedObject, ZoneStatus
from core_logic.state_machine import CartStateMachine
from core_logic.zone_checker import PointInPolygonChecker
from vision.detector import YOLODetector
from vision.tracker import ByteTrackAdapter


class FrameProcessor:
    """Orchestrates the full vision pipeline: detect -> track -> zone -> events -> cart.

    This is the main coordinator.  Feed it frames from any source (webcam, video
    file, IP stream) and read the cart state out.

    Usage::

        processor = FrameProcessor()
        for frame in video_source:
            state = processor.process_frame(frame)
            print(state)
    """

    def __init__(
        self,
        detector: Optional[YOLODetector] = None,
        tracker: Optional[ByteTrackAdapter] = None,
        zone_checker: Optional[PointInPolygonChecker] = None,
        event_processor: Optional[DebouncedEventProcessor] = None,
        state_machine: Optional[CartStateMachine] = None,
    ) -> None:
        self._detector = detector or YOLODetector()
        self._tracker = tracker or ByteTrackAdapter()
        self._zone_checker = zone_checker or PointInPolygonChecker()
        self._event_processor = event_processor or DebouncedEventProcessor()
        self._state_machine = state_machine or CartStateMachine()

        self._frame_count: int = 0
        self._skip_every_n = settings.process_every_n_frames
        self._last_process_time: float = 0.0

    # ── Public API ─────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> Optional[CartState]:
        """Run one frame through the full pipeline.

        Returns the updated CartState if detection ran on this frame, or None
        if the frame was skipped (to maintain target FPS).
        """
        self._frame_count += 1

        # Frame skipping — not every frame needs detection.
        if self._frame_count % self._skip_every_n != 0:
            return None

        self._last_process_time = time.time()

        # 1. Detect
        detections = self._detector.detect(frame)
        if not detections:
            return self._state_machine.get_state()

        # 2. Track
        tracked_objects = self._tracker.update(detections, frame)

        # 3. Classify zone
        inside_zone: list[TrackedObject] = []
        outside_zone: list[TrackedObject] = []
        for obj in tracked_objects:
            zone = self._zone_checker.classify(obj)
            if zone is ZoneStatus.INSIDE
                inside_zone.append(obj)
            else:
                outside_zone.append(obj)

        # 4. Process events
        events = self._event_processor.process(
            inside_zone=inside_zone,
            outside_zone=outside_zone,
            timestamp=self._last_process_time,
        )

        # 5. Apply events to state machine
        for event in events:
            self._state_machine.apply_event(event)

        return self._state_machine.get_state()

    def reset_session(self) -> None:
        """Reset all state for a new customer session."""
        self._state_machine.reset()
        self._event_processor.reset()
        self._frame_count = 0

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def state_machine(self) -> CartStateMachine:
        return self._state_machine

    @property
    def last_process_time(self) -> float:
        return self._last_process_time
