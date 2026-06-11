from core_logic.models import CartItem


def calculate_total(items: tuple[CartItem, ...]) -> int:
    """Return the sum of all item subtotals in cents."""
    return sum(item.subtotal_cents for item in items)
