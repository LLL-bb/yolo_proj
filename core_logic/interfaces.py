from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from core_logic.models import (
    CartEvent,
    CartState,
    TrackedObject,
    ZoneStatus,
)


class InventoryStateMachine(ABC):
    """Tracks the customer's shopping cart as items are added and removed.

    This is the source of truth for cart contents.  The vision pipeline feeds
    CartEvents into it; the UI queries it for the current state.
    """

    @abstractmethod
    def apply_event(self, event: CartEvent) -> CartState:
        """Process one verified cart event and return the updated state."""
        ...

    @abstractmethod
    def get_state(self) -> CartState:
        """Return the current cart state snapshot."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear the cart for a new customer session."""
        ...


class ZoneChecker(ABC):
    """Determines whether a tracked object is inside the checkout zone."""

    @abstractmethod
    def classify(self, obj: TrackedObject) -> ZoneStatus:
        """Return INSIDE, OUTSIDE, or LOST based on the object's position."""
        ...


class EventProcessor(ABC):
    """Filters raw inside/outside observations into stable, verified CartEvents.

    Responsibilities:
    - Debounce boundary jitter (require N consecutive same-status frames)
    - Detect identity switches (new tracking ID that matches a recently-lost one)
    - Deduplicate events
    """

    @abstractmethod
    def process(
        self,
        inside_zone: list[TrackedObject],
        outside_zone: list[TrackedObject],
        timestamp: float,
    ) -> list[CartEvent]:
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal buffers (debounce histories, lost-track cache)."""
        ...


# ── Vision Pipeline Protocols ───────────────────────────────────────────────

class Detector(Protocol):
    """Protocol for object-detection backends (YOLO, etc.)."""

    def detect(self, frame: object) -> list:
        """Run inference on a frame.  Returns raw detections."""
        ...


class Tracker(Protocol):
    """Protocol for multi-object-tracker backends (ByteTrack, DeepSORT)."""

    def update(self, detections: list, frame: object) -> list[TrackedObject]:
        """Update tracks with new detections and return tracked objects."""
        ...
