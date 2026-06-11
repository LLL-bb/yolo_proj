from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ── Raw Detection ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Detection:
    """A single object detection before tracking."""
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2  (pixel coords)
    confidence: float
    class_id: int


# ── Tracked Object ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrackedObject:
    """An object that has been assigned a persistent tracking ID."""
    tracking_id: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float


# ── Zone Status ─────────────────────────────────────────────────────────────

class ZoneStatus(Enum):
    OUTSIDE = auto()
    INSIDE = auto()
    LOST = auto()


# ── Cart Events ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CartEvent:
    """A verified change to the shopping cart.  Immutable after creation."""
    event_type: str  # "added" | "removed"
    product_id: str
    product_name: str
    tracking_id: int
    timestamp: float
    confidence: float = 1.0
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


# ── Cart Item (per-product aggregation) ──────────────────────────────────────

@dataclass()
class CartItem:
    """A line item in the cart — one product with its count and active track IDs."""
    product_id: str
    product_name: str
    unit_price_cents: int
    quantity: int = 0
    active_tracking_ids: set[int] = field(default_factory=set)

    @property
    def subtotal_cents(self) -> int:
        return self.unit_price_cents * self.quantity


# ── Full Cart Snapshot ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class CartState:
    """Immutable snapshot of the cart at a point in time."""
    items: tuple[CartItem, ...] = ()
    total_cents: int = 0
    item_count: int = 0
    session_id: str = ""

    @classmethod
    def empty(cls) -> CartState:
        return cls(items=(), total_cents=0, item_count=0, session_id="")
