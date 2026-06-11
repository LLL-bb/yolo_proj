from __future__ import annotations

import pytest

from core_logic.event_processor import DebouncedEventProcessor
from core_logic.models import TrackedObject, ZoneStatus


class TestDebouncedEventProcessor:
    """Verify debouncing and identity-switch mitigation."""

    @pytest.fixture
    def ep(self) -> DebouncedEventProcessor:
        return DebouncedEventProcessor(
            debounce_frames=3,
            lost_track_ttl=10,
            iou_threshold=0.3,
        )

    def make_obj(
        self,
        tid: int,
        x1: float = 0,
        y1: float = 0,
        x2: float = 50,
        y2: float = 50,
        class_id: int = 29,  # water_bottle (valid CATALOG entry)
    ) -> TrackedObject:
        return TrackedObject(
            tracking_id=tid,
            bbox=(x1, y1, x2, y2),
            class_id=class_id,
            class_name="test",
            confidence=1.0,
        )

    # ── Debounce: must hold status for N frames ─────────────────────────

    def test_debounce_requires_consecutive_frames(self, ep: DebouncedEventProcessor) -> None:
        obj = self.make_obj(tid=1)
        obj2 = self.make_obj(tid=1, x2=60, y2=60)  # same ID, slight movement

        # Frame 1: first observation -> buffer fills but not yet stable
        events = ep.process(inside_zone=[obj], outside_zone=[])
        assert len(events) == 0

        # Frame 2: still inside -> buffer needs one more frame
        events = ep.process(inside_zone=[obj2], outside_zone=[])
        assert len(events) == 0

        # Frame 3: still inside -> buffer full, status switches
        events = ep.process(inside_zone=[obj], outside_zone=[])
        assert len(events) == 1
        assert events[0].event_type == "added"

    def test_debounce_rejects_jitter(self, ep: DebouncedEventProcessor) -> None:
        obj_i = self.make_obj(tid=1)
        obj_o = self.make_obj(
            tid=1, x1=200, y1=200, x2=250, y2=250  # outside zone
        )

        # Inside: 2 frames
        ep.process(inside_zone=[obj_i], outside_zone=[])
        ep.process(inside_zone=[obj_i], outside_zone=[])

        # Jitter: one outside frame
        ep.process(inside_zone=[], outside_zone=[obj_o])

        # Should need 3 more inside frames to fire again
        ep.process(inside_zone=[obj_i], outside_zone=[])
        ep.process(inside_zone=[obj_i], outside_zone=[])
        events = ep.process(inside_zone=[obj_i], outside_zone=[])
        assert len(events) == 1  # finally debounced
        assert events[0].event_type == "added"

    def test_remove_after_debounce(self, ep: DebouncedEventProcessor) -> None:
        obj = self.make_obj(tid=1)
        obj_o = self.make_obj(
            tid=1, x1=200, y1=200, x2=250, y2=250
        )

        # Add it first (3 frames)
        for _ in range(3):
            ep.process(inside_zone=[obj], outside_zone=[])
        # Now all outside for 3 frames
        for _ in range(2):
            ep.process(inside_zone=[], outside_zone=[obj_o])
        events = ep.process(inside_zone=[], outside_zone=[obj_o])
        assert len(events) == 1
        assert events[0].event_type == "removed"

    # ── Identity switch matching ────────────────────────────────────────

    def test_identity_switch_no_duplicate(self, ep: DebouncedEventProcessor) -> None:
        """A new ID at the same location as a lost ID should not double-add."""
        obj1 = self.make_obj(tid=1)
        obj2 = self.make_obj(tid=2)  # different ID, same bbox

        # Add track 1 — need 3 frames for debounce, event fires on 3rd
        events: list = []
        for _ in range(2):
            ep.process(inside_zone=[obj1], outside_zone=[])
        events = ep.process(inside_zone=[obj1], outside_zone=[])
        assert len(events) == 1  # "added" fires on the debounce-complete frame

        # Track 1 disappears, track 2 appears at same location
        # First, "lose" track 1 by not including it in visible IDs
        ep.process(inside_zone=[], outside_zone=[])
        ep.process(inside_zone=[], outside_zone=[])
        # Actually the lost-track detection happens in process() when the
        # tracking ID is not in visible_ids but IS in _stable.
        # The above should mark track 1 as LOST since it's not visible.

        # Now track 2 appears inside
        for _ in range(2):
            ep.process(inside_zone=[obj2], outside_zone=[])
        events = ep.process(inside_zone=[obj2], outside_zone=[])

        # Should be matched via identity switch -> no "added" event
        assert all(e.event_type != "added" for e in events)

    def test_identity_switch_different_class(self, ep: DebouncedEventProcessor) -> None:
        """Different classes should not match as identity switches."""
        obj1 = self.make_obj(tid=1, class_id=29)  # water
        obj2 = self.make_obj(tid=2, class_id=32)  # coke, same bbox

        # Add track 1
        for _ in range(3):
            ep.process(inside_zone=[obj1], outside_zone=[])

        # Lose track 1
        ep.process(inside_zone=[], outside_zone=[])

        # Track 2 appears
        for _ in range(2):
            ep.process(inside_zone=[obj2], outside_zone=[])
        events = ep.process(inside_zone=[obj2], outside_zone=[])

        # Different class -> no identity match -> "added" fires
        added = [e for e in events if e.event_type == "added"]
        assert len(added) == 1
