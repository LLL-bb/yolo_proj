import pytest

from core_logic.models import CartEvent
from core_logic.state_machine import CartStateMachine


class TestCartStateMachine:
    """Verify that the cart correctly tracks add/remove events."""

    @pytest.fixture
    def sm(self) -> CartStateMachine:
        return CartStateMachine()

    def make_event(self, event_type: str, product_id: str,
                   tracking_id: int = 1, timestamp: float = 0.0) -> CartEvent:
        return CartEvent(
            event_type=event_type,
            product_id=product_id,
            product_name="Test Product",
            tracking_id=tracking_id,
            timestamp=timestamp,
        )

    # ── Happy path ──────────────────────────────────────────────────────

    def test_add_item(self, sm: CartStateMachine) -> None:
        sm.apply_event(self.make_event("added", "coke_can_355"))
        state = sm.get_state()
        assert state.item_count == 1
        assert state.total_cents == 199

    def test_add_and_remove(self, sm: CartStateMachine) -> None:
        sm.apply_event(self.make_event("added", "coke_can_355"))
        sm.apply_event(self.make_event("removed", "coke_can_355"))
        state = sm.get_state()
        assert state.item_count == 0
        assert state.total_cents == 0

    def test_multiple_identical_items(self, sm: CartStateMachine) -> None:
        sm.apply_event(self.make_event("added", "coke_can_355", tracking_id=1))
        sm.apply_event(self.make_event("added", "coke_can_355", tracking_id=2))
        state = sm.get_state()
        assert state.item_count == 2
        assert state.total_cents == 398  # 199 * 2

    # ── Edge cases ──────────────────────────────────────────────────────

    def test_remove_nonexistent(self, sm: CartStateMachine) -> None:
        """Removing an item that was never added should be a no-op."""
        sm.apply_event(self.make_event("removed", "coke_can_355"))
        state = sm.get_state()
        assert state.item_count == 0

    def test_remove_one_of_two_identical(self, sm: CartStateMachine) -> None:
        sm.apply_event(self.make_event("added", "coke_can_355", tracking_id=1))
        sm.apply_event(self.make_event("added", "coke_can_355", tracking_id=2))
        sm.apply_event(self.make_event("removed", "coke_can_355", tracking_id=1))
        state = sm.get_state()
        assert state.item_count == 1
        assert state.total_cents == 199

    def test_reset(self, sm: CartStateMachine) -> None:
        sm.apply_event(self.make_event("added", "coke_can_355"))
        sm.reset()
        state = sm.get_state()
        assert state.item_count == 0
        assert state.total_cents == 0

    def test_empty_state(self, sm: CartStateMachine) -> None:
        state = sm.get_state()
        assert state.item_count == 0
        assert state.total_cents == 0
        assert state.session_id == ""
