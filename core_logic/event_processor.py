from __future__ import annotations

import time
from collections import deque
from typing import Optional

from config.products import ProductCatalog
from config.settings import settings
from core_logic.interfaces import EventProcessor
from core_logic.models import CartEvent, TrackedObject, ZoneStatus


class _LostTrack:
    """A track that disappeared from the frame, stored for identity-switch matching.

    If the track was INSIDE the zone when lost and doesn't reappear within
    *ttl_frames*, the processor emits a "removed" event.
    """

    def __init__(
        self,
        tracking_id: int,
        class_id: int,
        class_name: str,
        last_bbox: tuple[float, float, float, float],
        lost_at_frame: int,
        ttl_frames: int,
    ) -> None:
        self.tracking_id = tracking_id
        self.class_id = class_id
        self.class_name = class_name
        self.last_bbox = last_bbox
        self.lost_at_frame = lost_at_frame
        self.expires_at_frame = lost_at_frame + ttl_frames


def _iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Intersection-over-Union of two bounding boxes (x1, y1, x2, y2)."""
    xi1 = max(a[0], b[0])
    yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2])
    yi2 = min(a[3], b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


class DebouncedEventProcessor(EventProcessor):
    """Transforms raw zone observations into stable, deduplicated CartEvents.

    **Boundary jitter** — each tracking ID must hold the same zone status for
    ``debounce_frames`` consecutive frames before a transition fires.

    **Occlusion / identity switches** — when a tracking ID disappears (tracker
    drops it) the processor stores its last-known state.  If a **different**
    tracking ID of the same class later appears near the same position (IoU >=
    ``iou_threshold``), it is treated as the same physical item and no
    duplicate "added" event is emitted.

    **Optimistic lost-track policy** — items are NOT removed from the cart
    when tracking is momentarily lost.  Only when the lost-track TTL expires
    without a reappearance does the processor emit a "removed" event.
    """

    def __init__(
        self,
        debounce_frames: Optional[int] = None,
        lost_track_ttl: Optional[int] = None,
        iou_threshold: Optional[float] = None,
    ) -> None:
        self._debounce_frames = debounce_frames or settings.debounce_frames
        self._lost_track_ttl = lost_track_ttl or settings.lost_track_ttl_frames
        self._iou_threshold = iou_threshold or settings.identity_switch_iou_threshold
        self._catalog = ProductCatalog()

        # tracking_id -> deque of zone status samples (maxlen = debounce_frames)
        self._buffers: dict[int, deque[ZoneStatus]] = {}

        # tracking_id -> current debounced zone status
        self._stable: dict[int, ZoneStatus] = {}

        # tracking_id -> last-known metadata (updated every visible frame)
        self._last_bbox: dict[int, tuple[float, float, float, float]] = {}
        self._last_class_id: dict[int, int] = {}
        self._last_class_name: dict[int, str] = {}

        # Recently-lost tracks, oldest first (maxlen = 60 ~ 2 seconds @ 30 fps).
        self._lost_tracks: deque[_LostTrack] = deque(maxlen=60)

        self._frame_counter: int = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def process(
        self,
        inside_zone: list[TrackedObject],
        outside_zone: list[TrackedObject],
        timestamp: Optional[float] = None,
    ) -> list[CartEvent]:
        self._frame_counter += 1
        ts = timestamp or time.time()
        events: list[CartEvent] = []

        visible_ids: set[int] = set()

        # ── Inside-zone objects ──────────────────────────────────────────
        for obj in inside_zone:
            visible_ids.add(obj.tracking_id)
            self._save_metadata(obj)
            self._feed_buffer(obj.tracking_id, ZoneStatus.INSIDE)
            new_stable = self._read_buffer(obj.tracking_id)
            prev = self._stable.get(obj.tracking_id)

            if new_stable is not None and new_stable != prev:
                if new_stable == ZoneStatus.INSIDE:
                    if not self._try_match_identity_switch(obj):
                        events.append(self._make_event("added", obj, ts))
                elif new_stable == ZoneStatus.OUTSIDE:
                    events.append(self._make_event("removed", obj, ts))
                self._stable[obj.tracking_id] = new_stable

        # ── Outside-zone objects ─────────────────────────────────────────
        for obj in outside_zone:
            visible_ids.add(obj.tracking_id)
            self._save_metadata(obj)
            self._feed_buffer(obj.tracking_id, ZoneStatus.OUTSIDE)
            new_stable = self._read_buffer(obj.tracking_id)
            prev = self._stable.get(obj.tracking_id)

            if new_stable is not None and new_stable != prev:
                if new_stable == ZoneStatus.OUTSIDE and prev == ZoneStatus.INSIDE:
                    events.append(self._make_event("removed", obj, ts))
                elif new_stable == ZoneStatus.INSIDE and prev == ZoneStatus.OUTSIDE:
                    if not self._try_match_identity_switch(obj):
                        events.append(self._make_event("added", obj, ts))
                self._stable[obj.tracking_id] = new_stable

        # ── Newly-lost tracks ────────────────────────────────────────────
        for tid in list(self._stable.keys()):
            if tid not in visible_ids and self._stable[tid] is not ZoneStatus.LOST:
                was_inside = self._stable[tid] == ZoneStatus.INSIDE
                self._stable[tid] = ZoneStatus.LOST
                self._buffers.pop(tid, None)

                if was_inside:
                    bbox = self._last_bbox.get(tid)
                    cid = self._last_class_id.get(tid)
                    cname = self._last_class_name.get(tid)
                    if bbox is not None and cid is not None:
                        self._lost_tracks.append(_LostTrack(
                            tracking_id=tid,
                            class_id=cid,
                            class_name=cname or "",
                            last_bbox=bbox,
                            lost_at_frame=self._frame_counter,
                            ttl_frames=self._lost_track_ttl,
                        ))

        # ── Expired lost tracks → "removed" events ───────────────────────
        events.extend(self._drain_expired_lost_tracks(ts))

        return events

    def reset(self) -> None:
        self._buffers.clear()
        self._stable.clear()
        self._last_bbox.clear()
        self._last_class_id.clear()
        self._last_class_name.clear()
        self._lost_tracks.clear()
        self._frame_counter = 0

    # ── Internal helpers ─────────────────────────────────────────────────

    def _save_metadata(self, obj: TrackedObject) -> None:
        self._last_bbox[obj.tracking_id] = obj.bbox
        self._last_class_id[obj.tracking_id] = obj.class_id
        self._last_class_name[obj.tracking_id] = obj.class_name

    def _feed_buffer(self, tid: int, status: ZoneStatus) -> None:
        if tid not in self._buffers:
            self._buffers[tid] = deque(maxlen=self._debounce_frames)
        self._buffers[tid].append(status)

    def _read_buffer(self, tid: int) -> Optional[ZoneStatus]:
        buf = self._buffers.get(tid)
        if buf is None or len(buf) < self._debounce_frames:
            return None
        if all(s is ZoneStatus.INSIDE for s in buf):
            return ZoneStatus.INSIDE
        if all(s is ZoneStatus.OUTSIDE for s in buf):
            return ZoneStatus.OUTSIDE
        return None  # mixed -> not yet stable

    def _try_match_identity_switch(self, obj: TrackedObject) -> bool:
        """Check if *obj* is a re-appearance of a recently-lost track.

        Returns True when matched (no cart event emitted).  The matched
        lost-track entry is removed so it cannot match a second time.
        """
        for lost in list(self._lost_tracks):
            if lost.class_id != obj.class_id:
                continue
            iou = _iou(lost.last_bbox, obj.bbox)
            if iou >= self._iou_threshold:
                self._lost_tracks.remove(lost)
                # Transfer the old stable status to the new tracking ID.
                old_status = self._stable.pop(lost.tracking_id, None)
                if old_status is not None:
                    self._stable[obj.tracking_id] = old_status
                self._last_bbox.pop(lost.tracking_id, None)
                self._last_class_id.pop(lost.tracking_id, None)
                self._last_class_name.pop(lost.tracking_id, None)
                return True
        return False

    def _drain_expired_lost_tracks(self, ts: float) -> list[CartEvent]:
        """Emit "removed" for lost tracks whose TTL has expired."""
        events: list[CartEvent] = []
        now = self._frame_counter

        while self._lost_tracks and self._lost_tracks[0].expires_at_frame <= now:
            lost = self._lost_tracks.popleft()
            product = self._catalog.get_by_class_id(lost.class_id)
            events.append(CartEvent(
                event_type="removed",
                product_id=product.id,
                product_name=product.name,
                tracking_id=lost.tracking_id,
                timestamp=ts,
                confidence=0.5,  # lower confidence — inferred removal
            ))

        return events

    def _make_event(self, event_type: str, obj: TrackedObject, ts: float) -> CartEvent:
        product = self._catalog.get_by_class_id(obj.class_id)
        return CartEvent(
            event_type=event_type,
            product_id=product.id,
            product_name=product.name,
            tracking_id=obj.tracking_id,
            timestamp=ts,
            confidence=obj.confidence,
        )
