from __future__ import annotations

import uuid
from collections import defaultdict

from config.products import ProductCatalog
from core_logic.interfaces import InventoryStateMachine
from core_logic.models import CartEvent, CartItem, CartState
from core_logic.pricing import calculate_total


class CartStateMachine(InventoryStateMachine):
    """Concrete cart state machine that tracks items per product + tracking ID.

    Internal structure::

        self._items: dict[str, CartItem]  — product_id → live item
        self._track_to_product: dict[int, str] — tracking_id → product_id

    This dual-map design ensures O(1) lookup when removing a tracked object
    and deduplicates correctly when the same tracking ID reappears.
    """

    def __init__(self) -> None:
        self._session_id: str = ""
        self._items: dict[str, CartItem] = {}
        self._track_to_product: dict[int, str] = {}
        self._catalog = ProductCatalog()

    # ── Public API ─────────────────────────────────────────────────────────

    def apply_event(self, event: CartEvent) -> CartState:
        if not self._session_id:
            self._session_id = event.event_id

        product = self._catalog.get_by_id(event.product_id)

        if event.event_type == "added":
            self._handle_add(event, product)
        elif event.event_type == "removed":
            self._handle_remove(event)
        # "transferred" events (identity-switch matches) are no-ops here —
        # they only remap tracking_ids inside the EventProcessor.

        return self._snapshot()

    def get_state(self) -> CartState:
        return self._snapshot()

    def reset(self) -> None:
        self._session_id = ""
        self._items.clear()
        self._track_to_product.clear()

    # ── Internal ───────────────────────────────────────────────────────────

    def _handle_add(self, event: CartEvent, product) -> None:
        pid = event.product_id
        tid = event.tracking_id

        # Maintain the tracking-ID → product-ID reverse map for O(1) removal.
        self._track_to_product[tid] = pid

        if pid not in self._items:
            self._items[pid] = CartItem(
                product_id=pid,
                product_name=product.name,
                unit_price_cents=product.price_cents,
            )

        self._items[pid].active_tracking_ids.add(tid)
        self._items[pid].quantity = len(self._items[pid].active_tracking_ids)

    def _handle_remove(self, event: CartEvent) -> None:
        pid = event.product_id
        tid = event.tracking_id

        item = self._items.get(pid)
        if item is None:
            return  # already removed — defensive

        item.active_tracking_ids.discard(tid)
        item.quantity = len(item.active_tracking_ids)
        self._track_to_product.pop(tid, None)

        if item.quantity <= 0:
            del self._items[pid]

    def _snapshot(self) -> CartState:
        items = tuple(self._items.values())
        total = calculate_total(items)
        count = sum(i.quantity for i in items)
        return CartState(
            items=items,
            total_cents=total,
            item_count=count,
            session_id=self._session_id,
        )
